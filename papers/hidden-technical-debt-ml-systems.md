# Hidden Technical Debt in Machine Learning Systems — Personal Reference Note

## 0. The Core Problem
- In normal software, when something breaks, you see it. You get an error, a crash, or wrong output on the screen.
- In ML systems, the code runs fine. No errors. No crashes. But the predictions slowly become wrong.
- The system looks healthy from the outside. Inside, it is rotting silently.
- This is because ML debt is not just in the code. It is in the data, in the model, in the configuration, and in how all these pieces interact.
- Taking shortcuts is not always bad. When you want to test an idea or show a prototype, moving fast makes sense.
- But when the prototype becomes the production system, the shortcuts stay. And they compound like interest on a loan.
- The paper calls this "hidden technical debt" because you cannot see it by just watching the system run. You need specific tools and checks to find it.
- My drift monitoring pipeline for loan applications is exactly one such tool. It watches for data drift, one of the silent killers the paper warns about.

### Data Drift and Why It Matters
- Data drift means the incoming data distribution changes from the training data distribution.
- The model was trained on one kind of applicant profile. Now the market sends a different kind.
- The model still produces predictions. They even look normal. But they are no longer relevant or useful.
- Applicants who should be accepted get rejected. Risky applicants slip through.
- The firm sees the system running and thinks everything is fine. Losses pile up silently.
- Only when the damage is large enough does someone notice.
- My pipeline monitors this drift. It alerts when the distribution shifts. It makes the invisible visible.
- This lets the firm act early: fix a data pipeline bug, retrain the model, or update the business policy.
- Same-distribution input is the model's core assumption. Drift monitoring detects when that assumption breaks — but a drift alert means "investigate," not "the model is broken," and the absence of drift does not prove the model is healthy. A feature can shift hard while accuracy holds (the shifted feature barely mattered), and inputs can sit perfectly in-distribution while the underlying relationship silently rots. Drift detection is a leading indicator, not a verdict.

---

## 1. Complex Models Erode Boundaries
**In ML, you cannot draw clean lines between components the way you can in traditional software. Everything is connected. A change in one place ripples everywhere.**

### 1.1 Entanglement (CACE — Changing Anything Changes Everything)
- **What:** Changing one input or feature changes how the model treats all other features. You cannot isolate the effect.
- **Simple example:** You change how "income" is calculated. Suddenly the model's reliance on "age" flips, and the approval rate shifts in ways nobody predicted.
- **Analogy:** Pulling one thread on a knitted sweater. The whole shape warps, not just the spot you pulled.
- **Why it matters:** In normal software, you can change module A without touching module B if the interface stays the same. In ML, there is no such separation. The model is a single entangled web of weights.
- **Mitigation:** Break the problem into smaller sub-problems. Use separate models for separate tasks. Monitor how prediction behavior changes over time, not just input changes.

### 1.2 Correction Cascades
- **What:** You build model B on top of model A's output. Later you improve model A. Model B breaks because it learned to compensate for A's old mistakes.
- **Simple example:** Model A predicts default risk. Model B takes that risk score and sets the interest rate. You retrain model A with better data. Now model B's interest rates are all wrong because the score distributions no longer match what B expects.
- **Analogy:** Fixing the foundation of a house after you already built three extra floors on it. The floors above crack.
- **Why it matters:** Cascades create hidden dependencies. You think you are improving one thing, but you silently break something downstream.
- **Mitigation:** When the upstream model changes, build a fresh downstream model directly on the target variable. Do not just patch the old cascade. Or better, avoid cascades when possible by training end-to-end.

### 1.3 Undeclared Consumers
- **What:** Other systems or teams use your model's output without telling you. They become silent dependents.
- **Simple example:** The marketing team builds a campaign that targets users based on your loan acceptance flag. You change the threshold one day. Their campaign performance crashes. You had no idea they were using your output.
- **Analogy:** A neighbor secretly connects their house to your electricity line. You turn off your main switch for maintenance, and their lights go dark too.
- **Why it matters:** Without knowing who depends on you, you cannot assess the blast radius of any change. Even small tweaks can cause silent damage across the organization.
- **Mitigation:** Make consumers declare themselves. Set up contracts or service level agreements. Document who uses what. Log access to model outputs.

---

## 2. Data Dependencies Cost More Than Code Dependencies
**In software, you can track which code depends on which library. In ML, data dependencies are heavier, harder to trace, and more prone to silent breakage.**

### 2.1 Unstable Data Dependencies
- **What:** An input feature that the model relies on changes its behavior or format over time, but the model keeps using it as if nothing happened.
- **Simple example:** The model uses "zip code" as a feature. The postal service changes the format from 5 digits to 9 digits. The model silently starts misreading the values. Predictions degrade with no visible error.
- **Analogy:** A recipe that says "add fresh tomatoes." One day the supplier switches to canned tomatoes without telling you. You follow the same recipe, but the dish tastes off. You do not know why until customers complain.
- **Why it matters:** Some features are inherently unstable. Changes happen outside your control. The model does not know the feature changed; it just sees numbers and continues.
- **Mitigation:** Version-lock the input data if possible. Monitor the statistical profile of each feature (mean, variance, missing rate). Your drift pipeline is exactly this defense.

### 2.2 Underutilized Data Dependencies
- **What:** Features that bring almost no predictive value but stay in the system. They add complexity, maintenance burden, and hidden risk.
- **Simple examples:**
  - **Legacy features:** You add a new, better feature but keep the old one because removing it feels risky. Now both are in the pipeline.
  - **Bundled features:** A group of features arrives together in one data package. Most are useless, but one is helpful. You include them all because separating is extra work.
  - **Correlated features:** Two features are nearly identical. Both stay, adding noise and redundancy.
- **Analogy:** Carrying a heavy backpack full of rocks. You put the rocks in on day one. You never take them out. You just add new, useful items on top and carry it all.
- **Why it matters:** These features increase the surface area for bugs, drift, and pipeline breakage. They also make the model harder to understand and debug.
- **Mitigation:** Prune features regularly. Ask: does this feature improve predictions enough to justify its maintenance cost? Use static analysis to map all data dependencies and identify dead or redundant signals.

### Static Analysis of Data Dependencies
- **What:** An automated tool or process that traces every feature back to its raw source.
- **Purpose:** Find unused features, duplicated signals, and hidden chains of data transformation.
- **Analogy:** Getting the full wiring diagram of your house. Once you see it, you can identify the dead wires and safely remove them.
- **Why it matters:** In large systems, nobody knows the full data lineage. Static analysis gives you the map.

---

## 3. Feedback Loops
**ML systems learn. Their predictions change the world. That changed world then becomes the new training data. Without care, this loop amplifies errors.**

### 3.1 Direct Data Feedback Loops
- **What:** The model's own predictions influence which examples it sees in the next training cycle. It starts to train on its own decisions, not on real-world truth.
- **Simple example (loan pipeline):** The model learns that borrowers who pay in full are good. It then only approves people who look like past successful borrowers. The next training data contains only those approved profiles. The model never sees other types of good borrowers. It gets overconfident and narrow-minded.
- **Analogy:** A teacher only calls on students who raise their hands. They answer correctly. The teacher concludes the whole class understands the material perfectly. The quiet students who are struggling remain invisible.
- **Why it matters:** The model's accuracy looks great on training data. In the real world, it fails on anyone who does not fit the narrow profile it keeps reinforcing.
- **Mitigation:** Inject exploration data. Deliberately include some random or diverse applicants in the training set, even if the model would not normally approve them. Track how model decisions are reshaping the input data over time.

### 3.2 Hidden Feedback Loops
- **What:** Two independent systems influence each other indirectly through the world. There is no direct data connection, so the loop is invisible.
- **Simple example:** Your bank's loan model rejects a geographic region heavily. A competitor bank loosens policies in that same region. The region's economy improves as more people get credit. Your model still uses old data showing that region is risky. It keeps rejecting, missing out on good business.
- **Analogy:** Two thermostats in different rooms of the same building. They don't communicate. But each one's heating decision affects the other room's temperature through shared walls. You cannot see the coupling.
- **Why it matters:** You cannot find these loops by looking at code or data pipelines alone. You need to understand the broader system and the real world it operates in.
- **Mitigation:** Build system-level monitoring that includes domain awareness. Talk to domain experts. Look for delayed, external effects of your model's decisions.

---

## 4. ML System Anti-Patterns
**These are common bad patterns that create debt in ML codebases. They are not about the model itself but about the software around it.**

### 4.1 Glue Code
- **What:** A huge amount of supporting code to stitch together generic ML packages and libraries. The actual model training code is tiny. The glue is massive and fragile.
- **Simple example:** 5,000 lines of Python script that preprocess data, transform formats, call the model training function, and reformat the output. The model training itself is 50 lines.
- **Analogy:** A beautiful, small engine completely wrapped in layers of duct tape and improvised plumbing. If you need to change anything, you have to cut through all the tape.
- **Why it matters:** Glue code makes the system rigid. Upgrading a library or trying a new model architecture requires rewriting the glue.
- **Mitigation:** Encapsulate the whole solution cleanly. Do not just wrap the model; design clean interfaces around the entire pipeline. Keep the packaging as clean as the core.

### 4.2 Pipeline Jungles
- **What:** Data preparation turns into a tangled mess of scripts, scrapers, joins, and transformations. There is no clear order or ownership.
- **Simple example:** The feature "income_normalized" is computed in three different scripts by three different teams. Nobody knows which one is the final, correct version used in production.
- **Analogy:** A garden hose assembled from twenty different pieces, each with a small leak. Water gets through, but you lose pressure, and finding the leaks takes forever.
- **Why it matters:** Debugging a pipeline jungle is slow and painful. Adding a new feature means navigating the jungle without a map. Small upstream changes break things downstream silently.
- **Mitigation:** Treat data pipelines like production code. Refactor them. Test them in isolation. Use directed acyclic graphs where each node has a clear input and output. Document the flow.

### 4.3 Dead Experimental Codepaths
- **What:** Old experimental branches left in the codebase with conditional flags. Nobody uses them anymore, but nobody removes them. They slowly rot.
- **Simple example:** A block like `if use_alternative_feature_v2:` sits in the production code. The experiment ended six months ago. v2 was abandoned. The flag is always false. The codepath is dead, but it stays.
- **Analogy:** Unused railway tracks left inside a busy train station. They take up space, confuse new drivers, and occasionally someone accidentally routes a train onto them.
- **Why it matters:** Dead code adds confusion. New team members waste time trying to understand it. Over time, the conditional branches multiply, making the code fragile and hard to test.
- **Mitigation:** Regularly delete unused codepaths. Use feature flags with built-in expiration dates. If an experiment is over, remove the code completely.

### 4.4 Abstraction Debt
- **What:** There are no clear, stable interfaces between the components of the ML system. Changing one part forces you to touch many other parts.
- **Simple example:** To add a new data source, you need to edit the data loader, the feature engineering, the model trainer, the serving function, and the monitoring dashboard. All in one big, coordinated change.
- **Analogy:** A car where changing a headlight bulb requires removing the front bumper, the battery, and three unrelated panels. A five-minute job takes two hours.
- **Why it matters:** Without clean abstractions, improvement velocity slows to a crawl. People become afraid to make changes because the ripple effects are unpredictable.
- **Mitigation:** Define minimal, stable APIs between data, training, model serving, and monitoring. Each piece should be changeable independently as long as the interface contract is respected.

### 4.5 Common Smells
Small patterns that indicate bigger design problems underneath.

- **Plain-Old-Data Type Smell:** Passing raw numbers and arrays between functions without context. A list of floats could be probabilities, logits, raw scores, or something else. The meaning gets lost along the way. **Fix:** Use named types or simple data classes to carry both the values and their interpretation.

- **Multiple-Language Smell:** Using different programming languages for different parts of the pipeline (Python for data prep, R for modeling, Java for serving). This makes end-to-end testing and debugging extremely hard because no single person understands the full stack. **Fix:** Limit the tech stack where possible. If multiple languages are necessary, define clear API contracts between them.

- **Prototype Smell:** The quick prototype model built in a notebook gets pushed to production because "it already works." It lacks tests, monitoring, scaling, and documentation. The prototype becomes the permanent system. **Fix:** Treat the prototype as a disposable first draft. Build a proper production version with tests, monitoring, and documentation before serving real users.

---

## 5. Configuration Debt
- **What:** The model's behavior depends on a huge set of external configuration files, flags, thresholds, and hyperparameters. These configs drift, conflict, and are rarely tested or audited.
- **Simple example:** A 500-line YAML configuration file controls the training pipeline. A single misspelled key silently disables a data validation check. Nobody notices for weeks. Bad data flows into the model.
- **Analogy:** A cockpit with two hundred unlabeled switches. Some switches do nothing. Some switches are critical. Some switches secretly override others. Only one person knows the layout, and that person left the company.
- **Why it matters:** Configurations are often outside the testing and versioning discipline applied to code. A wrong config can silently ruin model performance with no code change visible in git history.
- **Mitigation:** Treat configuration as code. Version it, test it, and keep it small. Validate all configs at system startup. Fail loudly if a required key is missing or misspelled. Do not bury critical thresholds deep in config files where nobody looks.

---

## 6. Dealing with Changes in the External World
**The world outside your system changes. ML systems often assume it stays fixed. This assumption creates silent debt.**

### 6.1 Fixed Thresholds in Dynamic Systems
- **What:** A decision boundary (like "score above 0.7 = accept") is chosen once and never revisited. The world changes around it, but the threshold stays rigid.
- **Simple example:** The credit policy threshold was set two years ago. Since then, the applicant population has shifted economically. The model now rejects far more people than intended, but nobody realizes because the threshold was never reviewed.
- **Why it matters:** A fixed threshold in a moving world is a drift problem by another name. The model can be perfectly fine, but the decision rule built around it becomes stale.
- **Mitigation:** Treat thresholds as parameters that need periodic tuning. Simulate the impact of different thresholds on the current data distribution. Monitor acceptance rates and flag sudden shifts.

### 6.2 Monitoring and Testing
The paper recommends several specific monitoring checks beyond basic system uptime.

- **Prediction Bias:** Compare the average of the model's predicted probabilities against the actual observed label distribution over a window of time. If the model predicts a 5% default rate but the real world shows 8%, a silent shift has occurred. This is easy to set up and catches many problems before they compound.

- **Action Limits:** Define clear numeric boundaries for system behavior. Examples: maximum predictions per hour, maximum fraction of null or missing inputs, maximum latency per request. When a boundary is crossed, trigger an alert. This is a simple circuit breaker that prevents runaway damage.

- **Up-Stream Producers:** Monitor the systems that provide your input data. If an upstream database changes its schema, a data feed changes its format, or a service increases its latency, your model can break without any change in your own code. Your monitoring must watch one step backward, into the data sources themselves.

---

## 7. Other Areas of ML-Related Debt
**These are broader categories beyond code, model, and configuration.**

### 7.1 Data Testing Debt
- **What:** Data arrives and gets used without any tests on its quality or consistency. Assumptions like "age is always positive," "email contains an @ symbol," or "zip code is exactly 5 digits" are never verified in code.
- **Why it matters:** Bad data creates bad models, but the failure is silent. The model trains happily on garbage. Only production results degrade.
- **Mitigation:** Write explicit data unit tests that run before training. Check ranges, types, missing rates, and invariants. Fail the pipeline loudly if a test fails, rather than silently consuming bad data.

### 7.2 Reproducibility Debt
- **What:** You cannot re-run an old experiment and get the same results. The exact dataset version, dependency versions, random seed, or train-test split procedure are lost or never recorded.
- **Why it matters:** Without reproducibility, you cannot debug regressions. You cannot prove that a new model is genuinely better than the old one. Trust in the whole process erodes.
- **Mitigation:** Version everything together: code, data, configuration, and environment. A single commit should capture the full state needed to re-run an experiment.

### 7.3 Process Management Debt
- **What:** There is no clear, automated path to move a model from a research notebook to a production serving system. The handoff is manual, full of ad-hoc steps, and performed by one overworked person.
- **Why it matters:** Manual handoffs introduce errors and delays. They also create a bottleneck where only one person understands the full journey. When that person leaves, the path disappears.
- **Mitigation:** Build an automated MLOps pipeline. Research outputs should feed into a tested, reproducible deployment process. Reduce the manual steps to zero where possible.

### 7.4 Cultural Debt
- **What:** The organization splits research and engineering into separate tribes with different goals, tools, and incentives. Researchers never feel the pain of production failures. Engineers treat the model as a mysterious black box.
- **Why it matters:** Neither side fully owns the system. Researchers optimize for paper metrics. Engineers optimize for uptime. Nobody optimizes for the real business outcome the model is supposed to improve.
- **Mitigation:** Create joint responsibilities. Researchers should be on-call for production model issues. Engineers should understand the model's purpose and basic behavior. Both groups should share the same business metric as their North Star.

---

## 8. Conclusions (Turned into a Checklist)
The paper offers these questions as tools to continuously probe for hidden debt:

- Can we test a completely new algorithmic approach at full scale easily?
- What is the complete, end-to-end chain of all data dependencies?
- Can we measure precisely how a small change will ripple through the system?
- Does improving one model or signal silently degrade another?
- How quickly can a new team member become fully productive?

These questions are not one-time checks. They are probes to carry with you, like a flashlight, into every new ML project or maintenance task. The drift monitoring pipeline is one concrete answer to these questions. It watches the data chain and measures one specific ripple. The rest of the list is the map for finding the other forms of debt before they compound.

---