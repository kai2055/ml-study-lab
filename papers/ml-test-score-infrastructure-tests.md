# ML Reliability Pipeline
## Notes: Infrastructure Tests
*Connected to: The ML Test Score Paper | Project: Datatroniq Credit Risk Model*

---

# Part 3: Tests for ML Infrastructure

An ML system is not one program running — it is a chain of stages. Each stage can break and affect everything after it. Infrastructure tests are about making sure the machinery that builds, validates, and serves the model is solid, not just the model itself.

---

## Test 1 — Training Is Reproducible

Training the same model twice on the same data should give two identical models. If training is deterministic, debugging becomes much simpler — if something changes between two runs, you know it was not random noise. It also makes diff-testing possible, where you verify a code change did not accidentally change model behaviour.

**The problem:** Training is often not reproducible. The sources of non-determinism are:

- Random number generation — fixable with seeding
- Initialisation order — even with seeding, different parts of the model can initialise in different orders across runs
- Multi-threading — the order threads process training data can vary unpredictably, even on one machine

**Mitigation:** Fix random seeds throughout. Where full determinism is not achievable, ensembling multiple models can smooth out the variance so individual run differences stop mattering.

**In the Datatroniq pipeline:** Fixed seeds were used throughout the project. In `data_splitter.py`, `split_train_val_test()` takes `random_state=42` which goes directly into the sklearn `train_test_split` calls — same rows, same splits, every time. In `tuner.py`, all three `TuningConfig` dataclasses carry `random_state=42` which flows into `GridSearchCV` and `RandomizedSearchCV`. In `trainer.py`, `build_pipeline()` passes `random_state=42` into `RandomForestClassifier` and `XGBClassifier`. The shared `conftest.py` defines the `tiny_xy` fixture with `rng = np.random.default_rng(0)` — every model-layer test uses this, so test inputs are identical across every run.

---

## Test 2 — Model Specification Code Is Unit Tested

Model configuration code looks like simple config but it can have bugs. The problem is that training is slow — by the time a misconfiguration shows up in model performance, a lot of time has already been wasted.

**Two types of tests that help:**

- *API usage tests* — generate random input data and run one step of training. Catches common mistakes fast without a full run. Also useful: test that the model can restore from a checkpoint after a crash.
- *Algorithmic correctness tests* — check that the model is learning for the right reasons, not just producing good numbers. Options: verify that loss decreases over a few training steps, or deliberately overfit on a tiny dataset to confirm learning actually happens. Avoid golden tests — tests that compare a partially trained model against a previously saved output — they break easily and tell you nothing useful when they do.

**In the Datatroniq pipeline:** `tests/test_models/test_trainer.py` checks that `build_pipeline()` produces a valid sklearn Pipeline with the correct steps, and that `train()` returns a fitted pipeline that can call `.predict_proba()` on `tiny_xy` without error. `tests/test_models/test_evaluator.py` checks that `predict()` returns a DataFrame with `proba` and `label` columns, that all probabilities sit in the [0, 1] range, and that `label` contains only 0s and 1s. `tests/test_models/test_data_splitter.py` checks that `split_train_val_test()` with `test_size=0.20, val_size=0.20` on 50 rows gives exactly `len(X_train)==30`, `len(X_val)==10`, `len(X_test)==10`, summing to 50.

Algorithmic correctness tests — loss decrease checks, deliberate overfitting — were not implemented. That is a real gap.

---

## Test 3 — The Full ML Pipeline Is Integration Tested

One stage can break and the error only shows up two or three stages later. Unit tests on individual stages cannot catch this. An integration test runs the whole pipeline end-to-end and checks that data and code move through every stage correctly.

**The problem:** A bug in feature generation might not crash anything — it might just produce a wrong model that passes all local checks. Unit tests would not catch it.

**Mitigation:** The integration test should run continuously and on every new model or server release. Faster versions on a subset of data give quick feedback; slower versions mirroring production catch deeper problems.

**In the Datatroniq pipeline:** GitHub Actions CI runs all 110 tests on every push. The `ci.yml` workflow runs `pytest` across `tests/test_data/`, `tests/test_models/`, and `tests/test_monitoring/` — a failure in any layer is caught before it reaches the main branch.

What is not there is a true end-to-end smoke test — one automated test that runs the full `data/raw/ → transform → validate → train → evaluate → save_baseline` flow on a small synthetic dataset. The layers were tested independently and the full pipeline was manually verified via `scripts/run_training.py` on the real SBA data. An automated end-to-end test would strengthen this.

---

## Test 4 — Model Quality Is Validated Before Serving

After training, before the model touches real traffic, something needs to check whether the model is actually good enough to serve. A model that trains without errors can still be worse than the one currently running in production.

**Two things to watch for:**

- *Slow degradation* — quality declining gradually across many versions. Loose thresholds comparing against a validation set can catch this.
- *Sudden drops* — a new version that is significantly worse than the previous one. Tighter thresholds comparing the new model's predictions directly against the previous model catch this.

**In the Datatroniq pipeline:** No automated gate was implemented. After training, the five metrics (ROC-AUC, recall, precision, log loss, Brier score) are logged to MLflow and `selector.py` picks the best model using a cost-weighted score (`COST_FN=5.0`, `COST_FP=1.0`). But nothing automatically blocks deployment if the model falls below a quality threshold. The model was manually checked in MLflow before deployment to GCP Cloud Run. An automated blessing or veto step is not there and would be a meaningful addition.

---

## Test 5 — The Model Allows Step-by-Step Debugging on a Single Example

When the model returns a strange prediction, there needs to be a way to feed that one example through and see what happens at every stage. Without that, debugging means guessing.

**The problem:** Numerical instability and silent transformation errors are very hard to find without intermediate visibility. The problem could be in the preprocessing, in the encoding, or in the model itself — and a wrong number at the end gives no clue where.

**Mitigation:** Something that lets you pass one example through the full pipeline and inspect each stage's output. Even a simple debug script that prints intermediate values is useful.

**In the Datatroniq pipeline:** The FastAPI `/predict` endpoint accepts a single input and returns a prediction and probability. That shows the output but nothing in between — no post-transformation values, no post-encoding values, no raw model score before the threshold is applied. A utility that exposes those intermediate outputs is not in the codebase and would make debugging production issues much faster.

---

## Test 6 — Models Are Tested via a Canary Process Before Entering Production

**What is a canary?** It is when a new model is deployed to production but only gets a small slice of real traffic — say 1–5% — while the old model handles the rest. The new model's behaviour is watched on real traffic before giving it full responsibility. If something goes wrong, the damage is small. If it behaves well, traffic is gradually increased until the new model fully takes over.

**The problem:** Offline testing cannot catch everything. A common failure is a mismatch between the model and the serving infrastructure. Model code changes more often than serving code. A model trained with a newer library version might expect something the currently deployed server does not have — so the server refuses to load the model. This is invisible offline and only surfaces when the model meets the real serving environment.

**Mitigation:** Before full deployment, test that the model loads into the production serving environment and that inference on real input works. Then increase traffic gradually, monitoring at each step.

**In the Datatroniq pipeline:** No canary process was implemented. The model was deployed directly to GCP Cloud Run as a Docker container. The containerisation reduces the mismatch risk — the model artifact is copied into the Docker image at build time (ADR 026), so the model and serving code are in the same image with no separate binary that could be out of sync. But a gradual traffic rollout was not done and is not currently there.

---

## Test 7 — Models Can Be Quickly and Safely Rolled Back

If a deployed model starts causing problems, the fastest fix is going back to the previous known-good model. Rollback needs to be fast, practiced, and reliable.

**The problem:** Rollback is an emergency action. If it has never been practiced, the first time it is needed is the worst possible moment to find out it does not work.

**Mitigation:** Keep previous model versions stored and accessible. Make rollback a single automated action. Practice it before an incident forces it.

**In the Datatroniq pipeline:** Model versioning was not formally implemented. The `artifacts/` directory holds the current model as two files — `model.joblib` and `metadata.json`. Previous versions are not kept. GCP Cloud Run does support revision management so rolling back to a previous container deployment is possible through the console — but this was never set up as a deliberate, practiced procedure. No formal versioning and no tested rollback path is a real gap.

---

# Summary

| Test | Status in Datatroniq Pipeline |
|---|---|
| Reproducible training | Addressed — `random_state=42` fixed in splitter, tuner, trainer, and test fixtures via `conftest.py` |
| Model spec unit tested | Partially addressed — `test_trainer.py`, `test_evaluator.py`, `test_data_splitter.py` cover API usage; algorithmic correctness tests not done |
| Full pipeline integration tested | Partially addressed — CI runs all 110 tests on every push; no automated end-to-end smoke test |
| Model quality validated before serving | Not addressed — MLflow logging and manual inspection only, no automated gate |
| Step-by-step debugging | Not addressed — `/predict` shows final output only, no intermediate stage visibility |
| Canary deployment | Not addressed — direct deployment; containerisation reduces mismatch risk but no gradual rollout |
| Rollback capability | Not addressed — no formal versioning or practiced rollback procedure |

The infrastructure layer is where the v1 pipeline has the most gaps. Most are fixable in v2 without major changes to the architecture.

---

*ml-reliability-pipeline | github.com/kai2055/ml-reliability-pipeline*