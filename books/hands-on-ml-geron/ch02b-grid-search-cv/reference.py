"""
ch02b - GridSearchCV with a Pipeline

What this file demonstrates
---------------------------------
GridSearchCV combines four things into a single search:
    1. A slpitter       - how the data is divided for cross-validation
    2. An estimator     - what we train (here, a full Pipeline)
    3. A parameter grid - which hyperparameter to try, and at what values
    4. A scoring callable - the single metric the search optimizes


The mental model: GridSearchCV is a nested loop.
    outer loop: every combination of hyperparameters in the grid
    inner loop: k-fold cross-validation for that combination
The combination with best average CV score wins


The one piece of syntax to internalise
-----------------------------------------
When the estimator is a Pipeline, parameters are adressed by:

        "<step_name>__<parameter_name>"

The double underscore is sklearn's namespacing convention. If you write
"C" instead of "classifier__C", the grid key is silently ignored - no
crash, no warning, just a search that never tunes what you thought it
would. This is the failure mode to remember.

"""

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Synthetic data

# make_classification builds a fake binary classification problem
# wwights=[0.9, 0.1] makes class 1 the minority - ~10% positives.
# This mirrors Datatroniq's reality: defaulters are the minority class.
# Stratification matters precisely because of this imbalance


X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    weights=[0.9, 0.1],
    random_state=42,
)

# Hold out a test set the search never sees
# The search will cross-validate inside X_train only
# X_test is reserved for evaluating the final winner - exactly like the 
# project: GridSearchCV optimises one metric (AUC), the othr four
# metrics get computed on the held-out test set afterwards.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)




# The Pipeline
# Two named steps: "scaler" and "classifier"
# The names matter - they become the prefixes in the parametr grid
# Choose names you will be happy typing dozens of times

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])



# The parameter grid
# -------------------------------------------------
# We tune two things:
#   - whether the scaler centres the data (with_mean True/False)
#   - the regularisation strength C of logistic regression

# Notice the syntax: "<step_name>__<parameter_name>"
# Two underscores. Not one. Not a dot.

# Total combinations = 2 (with_mean) * 3 (C) = 6
# With cv=5, that's 6 * 5 = 30 fits


param_grid ={
    "scaler__with_mean": [True, False],
    "classifier__C": [0.1, 1.0, 10.0],
}


# The splitter

# StraitifiedKFold preserves the class ratio in every fold.
# With ~10% positives, plain KFold could produce a fold with zero
# defaulters - AUC on that fold would be undefined or nonsensical
# Stratification removes that failure mode.


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



