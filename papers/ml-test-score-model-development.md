# ML Reliability Pipeline
## Notes: Best Practices for ML Model Development
*Connected to: The ML Test Score Paper | Project: Datatroniq Credit Risk Model*

---

# Part 2: Tests for ML Model Development

This section is about the model itself — how it gets built, evaluated, and maintained over time. The tests here are not about whether the code runs. They are about whether the model stays reliable as it is developed, deployed, and left running in the real world.

---

## Test 1 — Every Model Specification Undergoes Code Review and Is Checked Into a Repository

Every decision made about the model needs to be recorded somewhere that can be traced, audited, and if needed, reproduced. In ML this means catching bad decisions — wrong hyperparameters, undocumented assumptions, experiments that ran once and were never recorded anywhere.

In the Datatroniq pipeline, three systems handle this in parallel.

Git and GitHub provided version control for all code decisions. Every commit was one coherent logical change — not a dump of unsaved work, not a checkpoint. Each commit had an imperative subject line and a body explaining what changed and why. The entire history of how the pipeline was built is navigable and traceable. Any commit can be checked out and the system state at that point is fully recoverable.

MLflow provided experiment tracking for every training run. Every time the training script ran, MLflow created a new run and recorded: the parameters used (hyperparameters, model family, search strategy), the metrics produced (AUC, recall, precision, log loss, Brier score), the artifacts generated (the saved model file, the metadata JSON), and a timestamp. The MLflow UI lets you browse and compare runs without any of that living only in memory.

The ADR (Architecture Decision Record) system covered the gap that neither Git nor MLflow fills. Git records what changed. MLflow records what happened. ADRs record why a decision was made — the context, the alternatives considered, and the reasoning for what was rejected. The project has 26 ADRs covering every significant architectural decision, each written after the code was proven to work, each append-only and never retroactively modified.

Together these three systems form the traceability layer. Nothing significant was undocumented.

---

## Test 2 — Offline Proxy Metrics Correlate with Actual Online Impact Metrics

When a model is trained and evaluated offline — before deployment — performance is measured using statistical metrics. These are offline proxy metrics: measured on historical data in a controlled environment, before anyone is actually affected by the predictions.

Once the model is deployed and real decisions are being made, what actually matters is different. Those are online impact metrics — the real-world outcomes the model is supposed to influence.

**Examples for Datatroniq:**

| Offline Proxy Metrics | Online Impact Metrics |
|---|---|
| ROC-AUC | Loan default rate |
| Precision | Revenue from approved loans |
| Recall | Proportion of creditworthy applicants approved |
| Log loss | Customer complaint rate |
| Brier score | Regulatory compliance outcomes |

The concern is that a model can look excellent by every offline metric and still fail to produce the real-world outcomes it was deployed to achieve — when optimising for AUC, for example, does not translate into fewer defaults in the real world.

The Datatroniq pipeline evaluated the model on five offline metrics: ROC-AUC (0.9721 on the test set), recall (0.8643), precision (0.6094), log loss (0.1074), and Brier score (0.0292). These were chosen deliberately. Accuracy was explicitly rejected because it treats approving a defaulter and rejecting a creditworthy applicant as equally costly — which they are not at Datatroniq. The asymmetric cost structure of the business was the reason.

What the pipeline does not do — and this is an honest gap — is build a feedback loop from online impact back into the system. There is no mechanism in v1 to measure what happened to loan outcomes after the model made its predictions and compare those outcomes against what the offline metrics predicted. Closing that loop is a real production concern deferred to v2.

What the pipeline does address is the precondition for that correlation: the threshold decision. ROC-AUC was used as the search metric during hyperparameter tuning because it is threshold-independent — it measures how well the model separates defaulters from repayers across all possible cutoffs. The threshold itself was then treated as a business lever, set separately in `selector.py`, reflecting the cost ratio between false negatives (approving a defaulter) and false positives (rejecting a creditworthy applicant). The offline evaluation preserves the flexibility for the business to align the threshold with real-world cost constraints — that is the closest v1 gets to connecting offline metrics to online impact.

---

## Test 3 — All Hyperparameters Are Tuned

Sklearn default hyperparameter values are designed to be reasonable across a wide range of problems. They are not designed for Datatroniq's credit risk data specifically. Using defaults without tuning means accepting unknown performance — the model might be fine, or it might be significantly underperforming, and there is no way to know either way.

The tuning strategy in the Datatroniq pipeline was chosen per model family based on the number of hyperparameters and the cost of training each model.

Logistic regression received **grid search** (GridSearchCV). It has few hyperparameters — primarily `C` (regularisation strength) and `penalty` — and training is cheap. The entire space is exhaustively coverable without a budget constraint.

Random forest received **random search** (RandomizedSearchCV, `n_iter=20`). It has more hyperparameters (tree depth, number of estimators, minimum samples per split, etc.) and training is more expensive. Random search explores more distinct values of the hyperparameters that actually matter rather than wasting trials on unimportant combinations. Counterintuitively it usually beats grid search in higher-dimensional spaces.

XGBoost was also handled with random search in v1. Bayesian optimisation was considered for XGBoost — it has seven or more meaningful hyperparameters and training is the most expensive of the three — but Bayesian search was deferred to v2 to avoid adding complexity before a v1 baseline existed.

All three searches used **stratified k-fold cross-validation** (k=5). Stratified because the Datatroniq data is imbalanced (roughly 80% repay, 20% default) — unstratified splits can produce folds with skewed class proportions, distorting the search. k=5 balances training data availability against evaluation stability.

The search metric in all three cases was **ROC-AUC**, not precision or recall, because AUC is threshold-independent. Using precision or recall as the search metric would bake a specific threshold into the search — removing the ability to adjust it for business reasons later. AUC keeps that lever free.

---

## Test 4 — The Impact of Model Staleness Is Known

All ML models are trained on a snapshot of the world from a specific point in time. The world keeps changing. At some point, the gap between what the model was trained on and what it sees in production becomes large enough that predictions are no longer reliable. Knowing when this happens is the core purpose of having a monitoring layer.

The drift monitoring layer is the direct answer to this requirement. The model was trained on SBA 7(a) data from FY2010–2019. Production data comes from FY2020 onwards. COVID-19 created a real, documented shift in the applicant population — who was applying, in what amounts, under what conditions. The training distribution no longer matches the production distribution.

The monitoring layer detects this by comparing the statistical distribution of each feature in incoming production data against the saved baseline — the snapshot of the training data distribution. Two tests are used:

**PSI (Population Stability Index)** for all 12 features — measures how much the distribution of a feature has shifted relative to the baseline. Thresholds: below 0.1 is stable, 0.1–0.25 is moderate drift, above 0.25 is significant drift.

**Wasserstein distance** (standard deviation normalised) for the 5 numerical features — measures the minimum work to transform one distribution into the other. Thresholds: below 0.3σ is stable, 0.3σ–0.8σ is moderate, above 0.8σ is significant.

On the Datatroniq COVID drift data: 7 features showed significant drift (PSI > 0.25), 1 showed moderate drift, and 4 remained stable. The monitoring layer flags this automatically.

What v1 does not yet do is automate the response to drift — retraining the model, updating the baseline, or triggering a review workflow. That is planned for v2. The detection is in place and staleness does not go unnoticed, but the mechanism to act on that signal automatically is not yet built.

---

## Test 5 — A Simpler Model Is Regularly Tested Against as a Baseline

The point of this test is not that simpler models are better. The point is that without a fixed simple reference point, there is no way to answer a basic question: how much is the added complexity actually buying?

The idea is to always keep a very simple model in the comparison — something like logistic regression with only the most obvious features, or even just predicting the majority class for every row. It is not meant to be good. It is meant to define the floor. If the complex model barely beats it, that is important information — it might mean the features do not carry the signal assumed, or that the complex model is overfitting and will degrade in production while the simple one holds steady.

In the Datatroniq pipeline, three model families were trained — logistic regression, random forest, and XGBoost. Logistic regression functioned as the simpler reference point relative to the ensemble models, even though it was not framed this way explicitly. XGBoost winning the selection process is more meaningful because logistic regression was in the comparison.

What the pipeline does not have is a truly degenerate baseline — something like always predicting the majority class, or predicting the historical default rate for every applicant. Adding that as a sanity check step in the evaluation would make the comparison more defensible.

---

## Test 6 — Model Quality Is Sufficient on All Important Data Slices

A global summary metric — one number across the entire test set — can hide serious problems at the level of specific subgroups. The approach is to cut the evaluation data along meaningful dimensions and check performance separately for each slice.

The concern is not that global accuracy is low. The concern is that global accuracy can improve while accuracy for a specific subgroup collapses — and the improvement masks the collapse. Global AUC can go up while the model becomes systematically wrong for one category of applicant, and the single headline metric would not show that.

For Datatroniq, meaningful slices would include: loan size bands (small vs. large loan amounts), business age groups (new businesses vs. established ones), business type (LLC vs. sole proprietorship vs. corporation), or processing method. Each of these might have qualitatively different default dynamics. A model that performs well in aggregate might be systematically wrong for small new businesses, and the global AUC of 0.9721 would not reveal that.

Slice evaluation was not implemented in v1. The evaluator computed metrics across the full test set only. This is a genuine gap. The five metrics (precision, recall, AUC, log loss, Brier score) are all global. Adding per-slice evaluation — at minimum for `businesstype` and `businessage`, which are two of the 12 selected features — would make the model's performance claims more defensible and more honest.

---

## Test 7 — The Model Has Been Tested for Fairness and Inclusion

ML models can inadvertently encode biases that are present in the training data and then perpetuate or amplify those biases in production. A model trained on historical loan data learns from a history that may have been discriminatory — if certain groups of applicants were systematically denied loans in the past, the model learns that pattern and applies it going forward, not because it is explicitly coded to, but because the historical data reflects it.

Two types of concern apply here. First, inputs: do any features correlate strongly with protected characteristics (race, gender, national origin, age) in a way that might cause the model to discriminate without using those attributes directly — proxy discrimination. Second, outputs: do predictions differ materially when conditioned on different groups?

Geographic features — `borrstate`, `projectstate`, `sbadistrictoffice` — were explicitly excluded from the feature set. The reason is documented in ADR 017: these columns could act as proxies for protected characteristics, and a proper fairness analysis had not been done. Knowingly including features that are proxies for protected attributes without analysis would be a fairness failure.

What was not done is a full fairness audit of the 12 features that were included. `businesstype` and `businessage` might also carry correlations with protected characteristics in ways that are not immediately obvious. This is tracked in the assumptions file as an open question, deferred to v2 along with the geographic features.

The pipeline took the right precautionary step on the most obvious risk but did not complete a systematic fairness evaluation. That remains an open gap.

---

# The Thread Running Through All Seven Tests

These seven tests are all asking the same question from different angles: does the model deserve to be trusted?

In the Datatroniq pipeline, some of these are fully addressed. The traceability layer (Git, MLflow, ADRs) is solid. Hyperparameter tuning was systematic and documented. Staleness detection is the core capability of the pipeline. Others are partially addressed or honestly acknowledged as gaps — offline-to-online correlation, slice evaluation, and a complete fairness audit all require more than v1 delivered.

Knowing which tests have been passed and which have not is itself a form of reliability. A system that knows its own gaps is more trustworthy than one that does not know what it has missed.

---

*ml-reliability-pipeline | github.com/kai2055/ml-reliability-pipeline*