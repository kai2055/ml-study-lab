# ML Reliability Pipeline
## Notes: Monitoring Tests
*Connected to: The ML Test Score Paper | Project: Datatroniq Credit Risk Model*

---

# Part 4: Monitoring Tests for ML

Knowing the system worked at launch is not enough. The system needs to keep working correctly over time. ML systems are by definition making predictions on data they have never seen before, and the world keeps changing underneath them. Monitoring is how you catch problems before they cause damage — a constantly updated view of the system's health, with alerts when things deviate from what is expected.

For ML systems specifically, three things need to be monitored: the serving system, the training pipeline, and the input data.

---

## Monitor 1 — Dependency Changes Result in Notification

An ML system usually pulls data from many other systems to generate features. If one of those upstream systems changes — a partial outage, a version upgrade, a schema change — the feature values can silently change meaning. The model gets confused but nothing obviously breaks. No error is thrown. The predictions just quietly get worse.

**Mitigation:** The team needs to be subscribed to announcements for every upstream dependency. The upstream team also needs to know their data is being used downstream — so they think before making breaking changes.

**In the Datatroniq pipeline:** The training data comes from the SBA 7(a) FOIA dataset. This is a public release updated periodically, not a live feed. So there is no upstream system sending real-time data and no dependency notification mechanism needed in the same way. The risk here is different — a new annual release of the dataset could change column names, add columns, or change coded values. The `schema.py` and `validator.py` layers would catch structural changes at load time, but there is no subscription or alert for when a new data release drops. For a live production system with real upstream feeds, this monitor would be critical.

---

## Monitor 2 — Data Invariants Hold in Training and Serving Inputs

The internal behaviour of a trained model is hard to inspect directly. The input data is more transparent. Checking that incoming data still matches the expectations set at training time is the first line of defence for detecting when the world has changed in a way that will confuse the model.

**Mitigation:** Use the schema from Test 1 (Data layer) to measure whether incoming data matches expected structure and distributions. Alert when it diverges significantly. Alerting thresholds need careful tuning — too sensitive and alerts become noise, too loose and real problems get missed.

**In the Datatroniq pipeline:** This is the core of what the monitoring layer does. The baseline saved after training captures the training distribution — percentiles for numerical features, value frequencies for categoricals. The `drift_detector.py` compares incoming production data against this baseline feature by feature using PSI for all 12 features and Wasserstein distance for the 5 numerical ones. The thresholds in `drift_policy.py` define what counts as moderate vs significant drift (PSI: 0.1 / 0.25, Wasserstein: 0.3σ / 0.8σ). The `/monitor` FastAPI endpoint exposes this as a callable check. On the Datatroniq COVID data, 7 features showed significant drift, 1 moderate, 4 stable — the system detected the distributional shift that COVID caused between the training period (FY2010–2019) and production (FY2020+).

---

## Monitor 3 — Training and Serving Features Compute the Same Values

The code that generates features at training time and the code that generates features at serving time are often not the same code. In theory they should produce identical values for the same input. In practice this is a common source of silent failure — called training/serving skew. The model was trained on one version of a feature and is being served a slightly different version.

A concrete example: a new feature is added to the serving system based on live data, but at training time it has to be backfilled from stored historical data using a completely different code path. Another example: training uses flexible but slow code for experimentation, serving uses heavily optimised code for latency — and the optimisation accidentally changes the result.

**Mitigation:** Log a sample of real serving traffic and compare feature values at serving time against what training would have produced for the same example. Monitor the number of features showing skew and the number of examples affected. Alternatively, compare distribution statistics (min, max, mean, missing rate) between training features and sampled serving features.

**In the Datatroniq pipeline:** Training/serving skew is a real risk here. At training time, features are built by `dataset_builder.py` using `build_features()`. At serving time, the `/predict` endpoint receives a raw JSON payload which is converted to a DataFrame and passed directly to the fitted sklearn Pipeline. The Pipeline itself handles preprocessing — StandardScaler and OneHotEncoder — so the transformation logic is the same object used at training time, which reduces skew risk significantly. The sklearn Pipeline is the single source of transformation truth. However, there is no logging of serving feature values and no comparison of serving vs training distributions at the feature level. That monitoring does not exist in v1.

---

## Monitor 4 — Models Are Not Too Stale

A model trained on old data becomes less reliable as the world changes. Knowing how old the model in production is — and knowing at what age the model's performance starts to meaningfully degrade — is important for deciding when to retrain.

There is also a less obvious cost to infrequently updated models: if the retraining process is manual and poorly documented, and the person who does it leaves the team, the process can be lost. Even written instructions go stale over time.

**Mitigation:** For frequently retrained models, monitor the age of the model in production and alert when it exceeds a threshold. Also monitor the age of data at each stage of the training pipeline to catch where a stall has occurred. For infrequently retrained models, monitor the age of any upstream data tables the features depend on.

**In the Datatroniq pipeline:** The model in production was trained on FY2010–2019 data. There is no automated monitoring of model age and no alert that fires when the model exceeds a staleness threshold. The drift detection layer indirectly addresses this — when drift is detected, it is a signal that the model may need retraining — but there is no clock-based staleness check independent of drift. This is a gap. Tracking the training date in `metadata.json` (which is already saved in `artifacts/`) and comparing it against the current date as part of the `/health` or `/monitor` response would be a simple addition.

---

## Monitor 5 — The Model Is Numerically Stable

During training, invalid numeric values — NaNs, infinities — can appear without throwing explicit errors. The model keeps running but the results are wrong. Catching these early speeds up diagnosis significantly.

**Mitigation:** Explicitly monitor for NaNs and infinities during training. Set plausible bounds for model weights and alert if values exceed them.

**In the Datatroniq pipeline:** No explicit numerical stability monitoring was implemented. The sklearn Pipeline would raise an error if NaNs propagated into the model — sklearn validators catch this at fit time. But there is no proactive monitoring that checks for NaNs appearing in intermediate steps or tracks whether feature values are within expected numeric ranges during training. For the model families used here (logistic regression, random forest, XGBoost), numerical instability is less of a concern than it would be for deep learning. But it is still worth noting as not addressed.

---

## Monitor 6 — No Regression in Training Speed, Serving Latency, Throughput, or RAM Usage

Predictive quality is not the only thing that can degrade. Computational performance can too — training gets slower, serving latency increases, memory usage creeps up. At scale this becomes a real operational problem. For ML systems specifically, these regressions can come from changes in data volume, feature changes, model changes, or underlying library updates.

**Mitigation:** Monitor computational performance metrics sliced by code version, data version, and model version. Watch for both sudden drops (compare against previous version) and slow leaks (alert when a pre-set threshold is crossed).

**In the Datatroniq pipeline:** GCP Cloud Run provides basic serving metrics — request latency and instance count are visible in the Cloud Run console. No custom latency or throughput monitoring was set up. Training time was not tracked across runs. For a v1 pipeline at this scale this is acceptable, but for a real production system with growing data volumes and regular retraining, tracking training time and serving latency across versions would be important.

---

## Monitor 7 — No Regression in Prediction Quality on Served Data

Validation metrics measured before deployment are measured on data that is already older than real serving input. They are an estimate, not a guarantee. Once the model is serving real traffic, prediction quality needs to be measured on that real traffic — which is harder because labels are often not immediately available.

**Options for measuring served prediction quality:**

- *Statistical bias in predictions* — the average prediction across a slice of data should be roughly zero in aggregate. A systematic bias is a signal something is wrong, even if the exact label is unknown.
- *When labels are available quickly* — some tasks produce labels almost immediately after prediction (e.g. did the user click). For those, quality can be measured in near real-time.
- *Human annotation* — periodically sample logged serving inputs, have humans label them, and compare against model predictions.

Whichever approach is used, thresholds need to be set and alerts need to fire when quality drifts outside them. Both sudden drops and slow degradations need to be monitored.

**In the Datatroniq pipeline:** Loan default outcomes are not known immediately — they are only known when the loan resolves, which can be months or years later. So near-real-time label-based quality monitoring is not possible. Statistical bias monitoring on the prediction distribution was not implemented. There is no logging of served predictions and no comparison of the prediction distribution over time. This is the most significant monitoring gap in the pipeline. The drift detection layer catches input distribution shift, but it does not catch output quality degradation directly. For a real production system, logging predictions and tracking the output distribution over time would be a necessary addition.

---

# Summary

| Monitor | Status in Datatroniq Pipeline |
|---|---|
| Dependency change notifications | Not applicable in the same way — static FOIA dataset, no live upstream feed; schema validation catches structural changes at load time |
| Data invariants in training and serving | Addressed — `drift_detector.py` with PSI and Wasserstein against saved baseline; `/monitor` endpoint exposes this |
| Training/serving feature parity | Partially addressed — sklearn Pipeline is single transformation object reducing skew risk; no serving feature logging or comparison |
| Model staleness | Partially addressed — drift detection signals indirectly; no clock-based age monitoring or alert |
| Numerical stability | Not addressed — sklearn raises on NaNs but no proactive monitoring of intermediate values |
| Computational performance | Not addressed — basic Cloud Run metrics available but no custom latency or training time tracking |
| Prediction quality on served data | Not addressed — no prediction logging, no output distribution monitoring, labels not available in near real-time |

The drift detection layer addresses Monitor 2 well, which is the core of what this pipeline was built to do. The other monitors are mostly gaps, and several of them — staleness tracking, prediction logging, output distribution monitoring — are natural v2 additions that do not require architectural changes.

---

*ml-reliability-pipeline | github.com/kai2055/ml-reliability-pipeline*