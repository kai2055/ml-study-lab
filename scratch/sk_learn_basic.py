
"""
Demonstrate's skleatn's Pipeline + ColumnTransfer concept.
We will build an assembly line that:
    - seperates columns by type (numerical vs categorical),
    = scales numbers and one-hot encodes categories.
    - then feeds everything to a Logistic Regression model.


Think of it like a factory:
    Pipeline    = the whole assembly line
    ColumnTransfer = a sorting station that splits items into different lanes,
                     processes each lane differently, then reassemnles them.
    StandardScaler  = a machine that makes sure all numerical items are on the same scale
                    (like converting inches, feet, and miles into meters)

    OneHotEncoder   = a machine that creates a separate on/off switch for each category.

    LogisticRegression = the final inspector that makes a yes/no decision. 

"""


# 1. Imports - tools from sklearn's toolbox
from sklearn.compose import ColumnTransformer           # the "split & merge" station
from sklearn.pipeline import Pipeline                   # the whole assembly line
from sklearn.preprocessing import StandardScaler, OneHotEncoder # transformation on machines
from sklearn.linear_model import LogisticRegression     # the decision maker
import pandas as pd                                     # to hold our tiny demo data


# Define which columns are which (placeholders - in a real project these come from your data)

numerical_cols = ["loan_amount", "income"]      # columns with numbers
categorical_cols = ["loan_program", "region"]    # columns with categories



# build the pre-processing "branching" station (ColumnTransformer)
    # Inside it we have two (name, transformer, column_list) tuples.
    # Analogy: "Put the red items on belt A, blue items on belt B, then merge later"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),    # scale numerical columns
        ("cat", OneHotEncoder(), categorical_cols)  # one-hot encode categorical columns
    ],
    remainder="drop"    # if there were extra columns we didn't list, ignore them
    # (In real project you might use 'passthrough' to keep them unchanged)
)



# Build the full pipeline - a list of steps.
#   Each step is a tuple: ("step_name", transformer_or_estimator)
#   The preprocessor is just one of those steps, followed by the model.



pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),    # first: clean & prepare the data
        ("calssifier", LogisticRegression())    # then: learn to make predictions
    ]
)


# (Optional) Quick sanity check with a tiny made-up dataset
# We don't need real data - just 3 rows to see the pipeline works.

if __name__ == "__main__":
    # Simulate a few rows of incoming raw data (like what Datatroniq would see)
    demo_data = pd.DataFrame({
        "loan_amount": [10000, 20000, 15000],
        "income": [50000, 70000, 60000],
        "loan_program": ["A", "B", "A"],
        "region": ["North", "South", "East"]
    })

    # We also need a dummy target variable (what we want to predict)
    # e.g., whether the loan was repaid (1) or not (0)
    y = [1, 0, 1]

    print("Raw data before pipeline:")
    print(demo_data)
    print("\nTarget (y):", y)

    # Fit the whole pipeline: the preprocessor learns the scaling parameters
    # and the category mappings, then the LogisticRegression learns from the
    # transformed data.
    pipeline.fit(demo_data, y)


    # Make a prediction on the same tiny dataset (just for demonstration)
    predictions = pipeline.predict(demo_data)
    print("\nPredictions:", predictions)

    # Show the internal steps
    print("\nPipeline steps:")
    print(pipeline.steps)



