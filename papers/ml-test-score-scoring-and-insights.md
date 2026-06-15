# ML Reliability Pipeline
## Notes: Scoring, Culture, and Real-World Insights
*Connected to: The ML Test Score Paper | Project: Datatroniq Credit Risk Model*

---

# Part 5: The ML Test Score and What Real Teams Looked Like

---

## How the Score Is Computed

The paper turns all these tests into a single number — the ML Test Score. The scoring works like this:

- **0.5 points** if a test is done manually and the results are documented and shared
- **1 point** if there is an automated system that runs the test repeatedly

Scores are summed separately for each of the four sections (features and data, model development, infrastructure, monitoring). The final ML Test Score is the **minimum** of the four section scores — not the average, not the total, the minimum.

The reason for taking the minimum is that all four sections matter. A system that is excellent at monitoring but has no data tests is not a reliable system — it just has one strong area. The minimum score forces every section to be addressed before the overall score can rise.

**Score interpretation:**

| Score | What it means |
|---|---|
| 0 | More of a research project than a production system |
| (0, 1] | Not totally untested but possible serious holes in reliability |
| (1, 2] | First pass at productionisation, more investment needed |
| (2, 3] | Reasonably tested, but many tests could still be automated |
| (3, 5] | Strong automated testing and monitoring, appropriate for mission-critical systems |
| > 5 | Exceptional levels of automated testing and monitoring |

All tests are worth the same points deliberately — the paper does not rank one test as more important than another because different teams have different priorities. Any test implemented raises the score, and that is intentional.

---

## Scoring the Datatroniq Pipeline

Working through the four sections honestly:

**Features and data:** The schema is in place, feature selection is documented, tests cover the feature code. Most of these are automated via CI. Roughly 3–4 points here.

**Model development:** Hyperparameter tuning is systematic, training is reproducible, model code is unit tested. Slice evaluation and a full fairness audit are missing. Roughly 2–3 points.

**Infrastructure:** Seeds are fixed, model spec is unit tested, CI runs the test suite. No automated quality gate before serving, no canary, no rollback procedure. Roughly 1–2 points.

**Monitoring:** Drift detection is the core capability and it is automated via the `/monitor` endpoint. Model staleness tracking, prediction quality monitoring, and numerical stability monitoring are not there. Roughly 1–2 points.

The minimum is roughly **1–2 points** — which puts the pipeline in the "first pass at productionisation" range. That is an honest assessment for a v1 portfolio project. The data and model layers are strong. The infrastructure and monitoring layers are where the gaps are, and those gaps are what pull the minimum score down.

---

## What the Paper Found When It Tested Real Teams

The paper describes interviews with 36 teams at Google across different product areas. The findings are worth knowing.

### Checklists catch things that experts miss

Even experienced teams found gaps when they went through the rubric. One team discovered a thousand-line untested file that generated all their input features. Another realised they had no way to detect if their global service was producing bad predictions for one specific country — their global metric looked fine but the country-level failures were invisible. A third team said confidently that their system could not be biased because it only dealt with speech audio — "we just get vectors of numbers." When asked whether they had tested performance on African American Vernacular English or ensured diversity in their human raters, they paused and acknowledged they had not considered it.

The value of a checklist is not that it tells experts things they already know. It is that it surfaces the specific gaps that are easy to miss when you are inside a system.

### Teams outsourced responsibility for data they did not own

Multiple teams assumed that because their features came from a larger upstream service, any data problems would be caught by that upstream team. This is not a safe assumption. The upstream team validates data for their own requirements — not for the downstream model's requirements. The smaller team's model might have very different sensitivity to data quality issues than the larger team's system.

The same logic appeared on the serving side — some teams assumed that because their model was embedded in a larger system, the larger system's reliability engineers would notice problems. But a small model's errors can be masked in the noise of a larger system.

### Integration testing was rarely done for training

Integration testing for the serving system was more common. But for the training pipeline — actually testing that data flows correctly through every stage and produces a valid model end-to-end — it was rarely implemented. The reason: training pipelines are often built as ad hoc scripts and manual processes. There is no structured framework to hang a generic integration test on. This matches the gap in the Datatroniq pipeline — CI runs all unit tests but there is no automated end-to-end training pipeline test.

### Canarying was common but often came too late

Many teams did canary deployments. But the paper notes two issues. First, canarying catches problems that have already made it through development — it is a late-stage check, not an early one. Catching issues in unit or integration tests is much better than catching them at serving time. Second, teams that did canary deployments usually did so because their existing release tooling made it easy. One team that tried it without that tooling found it so painful they never did it again. The tooling determines whether a practice happens at all.

### Training/serving skew was the most common and least addressed problem

The paper names Monitor 3 — training/serving skew — as the most important and least implemented test. It is responsible for production issues across a wide range of teams. It is also genuinely difficult to implement. The teams that had it working usually had it built into a shared framework so the investment was made once and many teams benefited.

For the Datatroniq pipeline, the sklearn Pipeline as a single transformation object is a meaningful structural protection against skew. But it is not the same as actually logging and comparing serving features against training features, which would be the full implementation.

---

## The Broader Point

The rubric is useful not because it gives a perfect score but because it forces a structured look at every part of the system. The gaps it surfaces are real gaps. Some of them are easy fixes. Some require significant investment. But knowing which is which — and having a number that reflects the current state honestly — is how reliability gets improved over time.

For the Datatroniq pipeline at v1, the score is honest: solid data and model layers, weaker infrastructure and monitoring. That is the right starting point. The monitoring layer being the core purpose of the pipeline means Monitor 2 (data invariants / drift detection) is the strongest part of the system. Everything else in the monitoring section — staleness, numerical stability, prediction quality on served data — is what v2 should address.

---

*ml-reliability-pipeline | github.com/kai2055/ml-reliability-pipeline*