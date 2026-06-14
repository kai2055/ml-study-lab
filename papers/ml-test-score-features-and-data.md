# ML Reliability Pipeline
## Notes: Features and Data Testing
*Connected to: The ML Test Score Paper | Project: Datatroniq Credit Risk Model*

---

## Why ML Testing Is Different

Traditional software systems are tested for correctness. The question is: does this code do what it is supposed to do? Given a fixed input, a correct program always produces the same output. Testing is code-centric — you write tests that verify behaviour, they pass or fail.

ML systems are different. The value of an ML system is not just in its code — it is in its reliability over time. A model does not have one fixed behaviour. It behaves differently depending on the data it receives, and the data in production is never exactly the same as the data it was trained on. The code can be perfectly correct and the system can still be silently failing.

ML testing is therefore not about correctness. It is about reliability — whether the system keeps doing what it was built to do as the world around it changes.

The CACE principle (Changing Anything Changes Everything) makes this concrete. In ML, data, features, and model behaviour are deeply entangled. You cannot isolate and test one feature in the way you can isolate a function. Changing one input affects others. Noise in the data is unavoidable in production. The approach to testing has to match the nature of the system.

---

# Part 1: Tests for Features and Data

Data is not just an input to an ML system — it is woven into the system itself. The model's behaviour depends on the distribution of the data it was trained on. If that distribution changes, the model's outputs change too, silently and without error.

The Datatroniq project is built directly on this problem. The credit risk model was trained on SBA 7(a) loan data from FY2010–2019. The data it receives in production comes from FY2020 onwards — a period that begins with COVID-19, which caused a real, documented shift in who was applying for small business loans and under what conditions. The model does not know this happened. It keeps running. That is exactly the failure mode the pipeline is designed to catch.

---

## Test 1 — Feature Expectations Are Captured in a Schema

Before you can detect that something has gone wrong, you need to have written down what right looks like. For the Datatroniq pipeline, that is the job of `schema.py`. It is the single source of truth for every column in the dataset — its name, its type, and its valid coded values. Nothing downstream touches data without going through what the schema declares.

**What the schema captured:**

- Column names — all 42 expected columns declared explicitly
- Data types — `STRING_COLUMNS`, `INTEGER_COLUMNS`, `FLOAT_COLUMNS` as flat dictionaries
- Coded values — valid values for categorical columns (e.g. `loanstatus` could only be `'pif'`, `'chgoff'`, or `'exempt'`)
- The `program` column was always `'7a'` — no variation, no signal, locked to the scope of the dataset

The `validator.py` layer used these schema declarations to run structural checks on any incoming data before transformation. Column names were checked first. Dtypes were enforced. Coded values were validated. If incoming data failed these checks, it was rejected before it could reach the model.

The schema is the formalisation of feature expectations. It makes implicit assumptions explicit and testable. Without it, there is no way to know whether data arriving in production is the same kind of data the model was trained on.

---

## Test 2 — All Features Are Beneficial and Features Adhere to Meta-Level Requirements

Not every column in a dataset is useful to a model. Some add noise without signal. Some are available during training but not at prediction time — which is a form of cheating. Some have so many unique values that encoding them creates thousands of features that generalise to nothing. Every feature has a cost and needs to be justified.

The feature selection decision was one of the most substantive pieces of work in the project. The raw schema had 42 columns. After a column-by-column review against explicit criteria, 12 features were selected.

### The Inclusion Test

The core question for every column was: **would this column exist when a real prediction has to be made?** If the answer was no — if the column was only known after the loan resolved — it was excluded regardless of how predictive it might look during training.

### Exclusion Reasoning — Grouped by Cause

| Exclusion Reason | Columns | Why |
|---|---|---|
| Target leakage — outcome columns | `paidinfulldate`, `chargeoffdate`, `grosschargeoffamount` | Known only after loan resolves. Including them would let the model see the future during training. |
| High-cardinality identifiers | `borrname`, `borrstreet`, `borrcity`, `borrzip`, `bankname`, `bankfdicnumber`, `bankncuanumber`, `bankstreet`, `bankcity`, `bankzip`, `locationid`, `franchisecode`, `franchisename`, `naicscode`, `naicsdescription`, `projectcounty`, `congressionaldistrict` | Each value is near-unique. One-hot encoding produces thousands of columns the model learns once and never sees again. |
| Geographic — fairness concerns | `borrstate`, `projectstate`, `sbadistrictoffice` | State-level features could act as proxies for protected characteristics. Deferred pending fairness analysis. |
| Date columns — no pre-approval availability | `asofdate`, `approvaldate`, `firstdisbursementdate` | Dates need engineering before use. Raw date values are not usable features. |
| Temporal proxy | `approvalfy` | Fiscal year as an integer is misleading. The model should not treat 2020 as 5 units more than 2015. |
| Informationally null | `program` | Always `'7a'` in this dataset. Zero variation means zero signal. |
| Availability ambiguous | `soldsecmrktind` | Unclear whether this reflects origination intent or post-origination outcome. Excluded under the v1 principle: exclude when uncertain. |
| Target (special case) | `loanstatus` | Not excluded — becomes the binary target: `pif → 0`, `chgoff → 1`. |

**Total: 29 columns excluded + 1 target + 12 features = 42 columns. Audit clean.**

### The 12 Selected Features

| Column | Type | Treatment | Reasoning |
|---|---|---|---|
| `grossapproval` | Numerical | StandardScaler | Total loan amount. Continuous quantity. |
| `sbaguaranteedapproval` | Numerical | StandardScaler | Guaranteed portion of the loan. Continuous quantity. |
| `initialinterestrate` | Numerical | StandardScaler | Interest rate at origination. Continuous quantity. |
| `terminmonths` | Numerical | StandardScaler | Loan term in months. 60 months is genuinely twice 30 months. |
| `jobssupported` | Numerical | StandardScaler | Jobs the loan is intended to support. Self-reported but carries real signal. |
| `subprogram` | Categorical | OneHotEncoder | Sub-classification within 7(a). Manageable cardinality. |
| `processingmethod` | Categorical | OneHotEncoder | How the loan was processed. Documented values per schema. |
| `fixedorvariableinterestind` | Categorical | OneHotEncoder | Binary indicator: fixed or variable rate. |
| `revolverstatus` | Categorical | OneHotEncoder | Integer-typed but treated as categorical — it is a 0/1 flag, not a quantity. |
| `businesstype` | Categorical | OneHotEncoder | Type of business entity. Manageable cardinality. |
| `businessage` | Categorical | OneHotEncoder | Age band of the business at application time. |
| `collateralind` | Categorical | OneHotEncoder | Whether the loan is collateralised. |

High-cardinality columns like `naicscode` and `borrname` were excluded because one-hot encoding them would have created an unmanageable feature space with near-zero generalisation value. `businesstype` was fine to encode because its cardinality is manageable. `revolverstatus` is a clear case where dtype and modelling treatment deliberately disagree — it is integer-typed in the schema but categorical in the model, because it is a flag, not a continuous quantity.

`paidinfulldate`, `chargeoffdate`, and `grosschargeoffamount` were not excluded because they are the target. They were excluded because they are downstream of the thing the model is trying to predict. They only exist after a loan has resolved. At prediction time — when Datatroniq is deciding whether to approve a new loan — these columns do not exist yet. Including them in training would mean the model learns nothing real about default risk.

---

## Test 3 — The Data Pipeline Has Appropriate Privacy Controls

Sensitive personal and financial data should not be exposed to the system without appropriate controls. The model should only use what it needs, and data that could identify individuals or carry consent concerns needs to be handled carefully.

For the Datatroniq pipeline, this was largely handled by the nature of the dataset itself. The SBA 7(a) FOIA dataset is a public record — released under freedom of information, so the privacy baseline was set externally. The columns that could have functioned as identifiers (borrower name, street address, bank identifiers) were excluded from features anyway for the cardinality reason above. The pipeline never stored or processed raw PII beyond what the raw data file contained.

In a real production system this would extend further — data access controls, anonymisation at ingestion, audit trails. The v1 pipeline does not implement these, but the exclusion of identifier columns is a natural first step.

---

## Test 4 — New Features Can Be Added Quickly

A system that is rigid about its features becomes a liability as the world changes. The question is: if the incoming data changes — new columns appear, existing columns are renamed, distributions shift — how much work does it take to accommodate that?

In the Datatroniq pipeline, the schema creates a defined place for this change. If a new column needs to be added to the feature set, the path is: add it to `schema.py` (name and type), add it to the relevant list in `dataset_builder.py` (numerical or categorical), and the rest of the system picks it up. The validator, transformer, and monitoring layer all derive their behaviour from what the schema declares.

**Where it is manageable:**

- Adding a new column with a new name: `schema.py` → `dataset_builder.py` → done
- Adding a new coded value to an existing categorical: `schema.py` coded values tuple → done
- Changing a feature's treatment (numerical to categorical): `dataset_builder.py` list reassignment → done

**Where it gets complicated:**

- Column rename: every reference to the old name across `schema.py`, `transformer.py`, `validator.py`, and `dataset_builder.py` has to be updated. Not catastrophic, but not frictionless.
- Cardinality changes: if a categorical column that was manageable suddenly has many more unique values in production data, the one-hot encoder produces unexpected output shapes.
- The drift monitoring layer holds its own baseline snapshot. If the feature set changes, the baseline has to be regenerated — the old one is no longer a valid comparison point.

The schema makes the pipeline more adaptable than it would be without one, but how quick the change is depends on what kind of change it is. Additive changes (new column, new coded value) are genuinely low-friction. Structural changes (rename, type change) require coordinated edits across multiple files.

---

## Test 5 — All Input Feature Code Is Tested

The code that handles features — encoding, scaling, validation — needs to be verified to work correctly. Having tests that would catch it being wrong is what separates a reliable pipeline from one that just appears to work.

In the Datatroniq pipeline, feature handling code was tested at multiple levels:

- **Schema declarations:** Tests confirmed that `COLUMN_TYPES` contained the expected columns and that `NUMERICAL_FEATURES` and `CATEGORICAL_FEATURES` were valid non-overlapping subsets of the schema.
- **Transformer:** `transformer.py` was tested to confirm that type coercions produced the expected dtypes and that invalid coded values were caught before reaching the model.
- **Dataset builder:** `dataset_builder.py` was tested to confirm that the feature/target split was correct, that leakage columns were absent from X, and that the binary target mapping (`pif → 0`, `chgoff → 1`) was accurate.
- **Sklearn Pipeline:** The full preprocessing pipeline (StandardScaler on numericals, OneHotEncoder on categoricals) was tested to confirm it transformed a valid input dataframe into the expected output shape.

The 110 tests across the project covered all layers. The monitoring layer tests in particular checked that the baseline feature set and the incoming feature set were compared correctly, and that a `FeatureMismatchError` was raised — not silently swallowed — when features were missing from incoming data. Silence on a mismatch is a reliability failure, not a graceful fallback.

---

# The Thread Running Through All Five Tests

These five tests are all connected by the same idea: an ML system's reliability depends on what its data looks like, not just whether its code runs.

The schema captures what the data is supposed to look like. Feature selection decides what the model is actually learning from. Privacy controls decide what it is allowed to learn from. Adaptability tests decide whether the system can survive the world changing. Feature code tests decide whether the implementation of all of the above is correct.

In the Datatroniq pipeline, these were not independent decisions. The schema informed feature selection. Feature selection informed the encoding decisions. The encoding decisions were what the tests verified. The whole layer hangs together — which is also why changing one part of it requires thinking about all the others.

---

*ml-reliability-pipeline | github.com/kai2055/ml-reliability-pipeline*