
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Synthetic data

X, y = make_classification(
    n_samples=800,
    n_features=8,
    n_informative=4,
    weights=[0.85, 0.15],
    random_state=7,

)


# Dataset split off
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42,
)

# Pipeline

pipeline = Pipeline([
    ("stanScale", StandardScaler()),
    ("regressClassifier", LogisticRegression(max_iter=1000))
])


# Parameter grid

param_grid = {
    "regressClassifier__class_weight": [None, "balanced"],
    "regressClassifier__solver": ["lbfgs", "liblinear"],

}


# The splitter

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)


# The search 

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print(f"Best CV AUC:        {search.best_score_:.3f}")


from sklearn.metrics import roc_auc_score
y_test_proba = search.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Held-out test AUC: {test_auc:.3f}")


import pandas as pd
results = pd.DataFrame(search.cv_results_)
print(results[["params", "mean_test_score"]])