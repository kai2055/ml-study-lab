"""
ch02c - RandomizedSearchCV with a Pipeline

What this file demonstrates
----------------------------
RandomizedSearchCV is the budget-conscious alternative to GridSearchCV.
Instead of tryinng every combination in the grid, it samples a fixed number
of combinations at random. That fixed number is the budget - n_iter.

The core tradeoff:
    GridSeacrchCV       - exhaustive, finds the true best, expensive
    RandomizedSearchCV  - approximate, finds a good-enough best, cheap


When to use which:
    GridSearchCV        - small parameter spaces (logistic regression: C,
                            class_weight, solver - maybe 12-24 combinations)
    RandomizedSearchCV  - large parameter spaces (random forest: n_estimators,
                            max_depth, max_features, min_samples_split... - thes
                            combinations multiply fast)

The mechanics are identical to GridSearchCV:
    - same Pipeline as the estimator
    - same StratifiedKFold as the splitter
    - same scoring callable (AUC)

What changes:
    - param_distributions instead of param_grid (can use scipy distributions)
    - n_iter caps the number of combinations tried
    - random_state on the search itself (sampling is random - fix the seed
        or results change every run)


Note on StandardScaler with random forest.
------------------------------------------
Random forests split on feature thresholds, not distances ot dot products.
Scaling doesn't affect where a threshold falls - it only shifts the scale 
of the values, not their rank order. So StandardScaler is not doing useful
work here. It's included only so the step__param syntax looks familiar from 
ch02b. In the real project, the preprocessing step for tree-based models
will differ from the preprocessing step for logistic regression - the
sklearn Pipeline hamdles this cleanly via different ColumnTransformers.


"""

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# scipy.stats distributions are optional in RandomSearchCV - you can
# use plain lists instead. loguniform is useful for parameters like C in 
# logistic regression where the interesting range spans orders of magnitude
# (0.001 to 1000). For the parameters we are tuning herem lists are fine.




# Synthetic data

# Same imbalanced setup as ch03b - minority class is ~10% of samples.
# Stratification matters for the same reason.


X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    weights=[0.9, 0.1],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)


# The Pipeline
# ---------------------
# Two named steps: "scaler" and "classifier"
# The classifier is now RandomForestClassifier, not LogisticRegression.
# The step names drive the grid keys - same rule as ch02b

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])


# Parameter distributions
#-------------------------------
# Three parameters, all from the classifier step.

# n_estimators - number of trees in the forest.
#   More trees = more stable predictions, more compute
#   - 100-300 is a pracrical range for a first search


# max_depth - maximum depth of each tree.
#   - None means trees grow until all leaves are pure (can overfit)
#   - Shallow trees (3-5) underfit. 10 is a middle ground
#   - Inlcuding None is deliberate - it's meaningful option


# max_features - number of features considered at each split
#   - "sqrt" = square root of total features (default for classification)
#   - "log2" = log base 2 total features
#   - Both are standard. The difference matters more on high-dimensonal data


# Exhaustive grid search: 3 x 4 x 2 = 24 combinations
# We will sample only 10 - watch cv_results_ to see which 10 were drawn.

param_distributions = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__max_features": ["sqrt", "log2"],
}


# The splitter

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# The search
#------------------------------
# n_iter - sample 10 combinations from the distribution above
# random_state fixes the sampling - same seed = same 10 combinatioms
# every run. Without this, results change on every execution.
#
# Total fits: 10 iterations x 5 folds = 50 fits
# Exhaustive grid search run: 24 combinations x 5 folds = 120 fits
# RandomizedSearchCV runs 58% fewer fits for a good-enough result.

n_iter = 10

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
    random_state=42,
)

search.fit(X_train, y_train)


# Results 
 
print("Best parameters:", search.best_params_)
print(f"Best CV AUC:````{search.best_score_:.4f}")


y_test_proba = search.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Held-out test AUC:  {test_auc:.4f}")

# Budget visibility - makes the n_iter tradeoff concrete
print(f"\nFits run: {search.n_splits_} folds x {n_iter} iterations = {search.n_splits_ * n_iter} total")
print(f"Exhaustive grid would have run: {search.n_splits_} folds x 24 combinations= {search.n_splits_ * 24} total")


# The cv_results_ habit - same as ch02b.
# Only 10 rows will appear, not 24
# Look at the spread: howmuch does the score vary across the 10 sampples?
import pandas as pd
results = pd.DataFrame(search.cv_results_)
print("\n", results[["params", "mean_test_score"]].sort_values("mean_test_score", ascending=False))