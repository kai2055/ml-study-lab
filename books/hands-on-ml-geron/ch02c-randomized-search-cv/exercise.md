
# ch02c — RandomizedSearchCV Exercise

## Goal

Build a randomized hyperparameter search from understanding. The
shape mirrors `reference.py` but the parameters and the dataset
are different. The new twist: **run the search twice at different
budgets** and observe what changes.

## What you're building

A RandomizedSearchCV that tunes a random forest pipeline on an
imbalanced synthetic dataset, run at two different `n_iter` values
so you can see the budget tradeoff with your own eyes.

## Requirements

### 1. Data

Generate a binary classification dataset with `make_classification`:

- 800 samples
- 12 features
- 6 informative features
- Class balance: ~85% / 15%
- `random_state=11`

Split off a 25% test set, stratified, with `random_state=11`.

### 2. Pipeline

Build a Pipeline with two steps. Use different step names than
`reference.py` used.

Step one: `StandardScaler`. (Yes, still — same reason as ch02b:
keeps the step__param syntax consistent. Random forest doesn't
need scaling, but it doesn't hurt it either.)

Step two: `RandomForestClassifier` with `random_state=11`.

### 3. Parameter distributions

Tune three parameters of the classifier:

- `n_estimators` over `[50, 100, 200, 400]`
- `min_samples_split` over `[2, 5, 10, 20]`
- `max_depth` over `[None, 5, 10, 20]`

You are **not** tuning `max_features` this time. If you reach for
it out of habit, slow down and re-read the spec.

Compute the exhaustive grid size yourself: how many combinations
would grid search try? Write the number in a comment near the
distributions. (You'll need this for the budget comparison
print at the end.)

### 4. Cross-validation

`StratifiedKFold` with 5 splits, shuffled, `random_state=11`.

### 5. The twist — two searches

Run the search **twice**, with the same Pipeline and the same
splitter, but at two different budgets:

- First: `n_iter=5`
- Second: `n_iter=15`

Both with `random_state=11` so the sampling is reproducible.

For each search, print:

- The best parameters
- The best CV AUC (4 decimal places)
- The held-out test AUC (4 decimal places)
- The full `cv_results_` table, sorted by `mean_test_score`
  descending

Label the output clearly so you can see which numbers belong to
which budget. Something like:

```
========== n_iter=5 ==========
Best parameters: ...
...

========== n_iter=15 ==========
Best parameters: ...
...
```

### 6. The thing to actually look at

After both searches run, look at the two tables side by side and
answer for yourself:

1. Did the higher budget find a better best score? Or did the
   small budget happen to land on a good combination by luck?
2. How much overlap is there between the two tables? (Same
   parameter combinations sampled in both runs?)
3. If you were running this on a real model that took 30 minutes
   per fit, would `n_iter=5` have been enough, or would you have
   regretted not running `n_iter=15`?

The point: **n_iter is a decision, not a default.** Too small and
you might miss the good combinations. Too large and you've spent
compute on diminishing returns. Seeing the same search at two
budgets makes that tradeoff concrete.

## Constraints

- Type it manually. No copy-paste from `reference.py`.
- If you get stuck on syntax, look back at `reference.py`, close
  it, then type from memory.
- Run it. Confirm both searches produce output.

## Done when

- `main.py` runs end-to-end without errors
- Both search results print, clearly labelled
- Both `cv_results_` tables print
- You can articulate, in one sentence, what tradeoff `n_iter`
  controls and how you'd choose it for a real problem