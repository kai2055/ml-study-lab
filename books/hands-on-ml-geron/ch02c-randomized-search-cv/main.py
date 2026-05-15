

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Synthetic data

X, y = make_classification(
    n_samples=800,
    n_features=12,
    n_informative=6,
    weights=[0.85, 0.15],
    random_state=11,
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=11,

)


# The pipeline

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("classi", RandomForestClassifier(random_state=11))
])


# Parameter distribution
param_distributions = {
    "classi__n_estimators": [50, 100, 200, 400],
    "classi__min_samples_split": [2, 5, 10, 20],
    "classi__max_depth": [None, 5, 10, 20],
}



# The splitter

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)


# The search

# First search: n_iter = 5
print("============== n_iter=5 ============")
search1 = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=5,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
    random_state=11,
)

search1.fit(X_train, y_train)

print("Best parameters:", search1.best_params_)
print(f"Best CV AUC:    {search1.best_score_:.4f}")

y_test_proba1 = search1.predict_proba(X_test)[:, 1]
test_auc1 = roc_auc_score(y_test, y_test_proba1)
print(f"Held-out test AUC: {test_auc1:.4f}")


import pandas as pd
results1 = pd.DataFrame(search1.cv_results_)
print("\nCV Results (sorted by mean_test_score):")
print(results1[["params", "mean_test_score"]].sort_values("mean_test_score", ascending=False))
print("\n")



# Second search: n_iter = 15
print("========== n_iter=15 ==========")
search2 = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
    random_state=11,
)
search2.fit(X_train, y_train)

print("Best parameters:", search2.best_params_)
print(f"Best CV AUC: {search2.best_score_:.4f}")

y_test_proba2 = search2.predict_proba(X_test)[:, 1]
test_auc2 = roc_auc_score(y_test, y_test_proba2)
print(f"Held-out test AUC: {test_auc2:.4f}")

results2 = pd.DataFrame(search2.cv_results_)
print("\nCV Results (sorted by mean_test_score):")
print(results2[["params", "mean_test_score"]].sort_values("mean_test_score", ascending=False))