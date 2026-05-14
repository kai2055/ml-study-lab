# ch02b — GridSearchCV Exercise

## Goal

Build a hyperparameter search from understanding, not memory. The
shape of the problem mirrors `reference.py`, but the details are
different enough that copying won't work — you have to know *why*
each piece is there.

## What you're building

A GridSearchCV that tunes a logistic regression pipeline on an
imbalanced synthetic dataset, using stratified k-fold and AUC.

## Requirements

### 1. Data

Generate a binary classification dataset with `make_classification`:

- 800 samples
- 8 features
- 4 informative features
- Class balance: ~85% / 15% (minority class is positive)
- `random_state=7`

Split off a 25% test set, stratified, with `random_state=7`.

### 2. Pipeline

Build a Pipeline with two steps. **Use different step names than
reference.py used.** Pick names that make sense to you — but commit
to them, because they drive your grid keys.

Step one: `StandardScaler`.
Step two: `LogisticRegression` with `max_iter=1000`.

### 3. Parameter grid

Tune two things:

- The `class_weight` parameter of the classifier, over `[None, "balanced"]`.
  This is meaningful for imbalanced data: `"balanced"` tells the model
  to weight minority-class errors more heavily, which directly affects
  how it handles defaulter-like minorities.
- The `solver` parameter of the classifier, over `["lbfgs", "liblinear"]`.
  Different solvers can produce slightly different coefficients on the
  same data — worth seeing.

You are **not** tuning `C` this time. If you find yourself reaching
for `C` out of habit, that's the signal to slow down and re-read the
spec.

### 4. Cross-validation

`StratifiedKFold` with 5 splits, shuffled, `random_state=7`.

### 5. Search

`GridSearchCV` with AUC scoring. Refit on all training data after
the search.

### 6. Output

Print three things:

- The best parameters
- The best CV AUC, to 4 decimal places
- The held-out test AUC, to 4 decimal places

## The thing to actually look at

After the search finishes, also print `search.cv_results_` — but
just two columns of it:

```python
import pandas as pd
results = pd.DataFrame(search.cv_results_)
print(results[["params", "mean_test_score"]])
```

Look at the full table. Answer these questions for yourself (no
need to write them down — but be ready to discuss):

1. How much does the mean test score vary across the four
   combinations? Is the gap between best and worst meaningful, or
   is the search picking between near-identical options?
2. Did `class_weight="balanced"` help, hurt, or do nothing?
3. Did the solver choice matter at all on this dataset?

The point of this step is to build the habit: **don't just trust
`best_params_`. Look at the spread.** A search that returns a
"winner" with a 0.0003 gap over the runner-up isn't really telling
you anything.

## Constraints

- Type it manually. No copy-paste from `reference.py`.
- If you get stuck on syntax, look back at `reference.py`, close
  it, then type from memory. Don't keep it open while you write.
- Run it. Confirm it produces output. Don't write tests; just
  confirm it executes and the numbers look plausible.

## Done when

- `main.py` runs end-to-end without errors
- All three numbers print
- The `cv_results_` table prints
- You can articulate, in one sentence each, what the splitter,
  estimator, grid, and scoring callable are doing in your code