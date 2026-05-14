
"""
Reference: StratifiedKFold demonstrated on a tiny imbalanced dataset.

This file shows what StratifiedKFold produces when you call .split() on it.
Read through, run it, see the output, then attempt the exercise yoourself in main.py

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# Step 1: Build a tiny imbalanced dataset.
# 20 rows total, 16 zeroes and 4 ones - roughly the kind of imbalance
# You'd see in a credit risk problem (most loans repay, some default)

np.random.seed(42)
n = 20
data = {
    "id": range(1, n + 1),
    "score": np.round(np.random.uniform(0, 100, n), 1),
    "label": [0] * 16 + [1] * 4,
}
df = pd.DataFrame(data)

# Shuffle so the labels aren't clustered. Stratified k-fold doesnn't actually
# care about roe order - it groups by class internally - but shuffling makes
# the printed output easier to read.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


print(df)
print(f"\nClass balance - 0s: {sum(df['label'] == 0)}, 1s: {sum(df['label'] ==1)}")
print()

# Step 2: Create the splitter object
# StratifiedKFold is not a function - it's a class. You instantiate it with
# the number of folds and configuration, then call .split() on it later.
# shuffle=True is recommended when calss labels might be ordered int the data
# random_state makes the splits reproducible across runs.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 3: Call .split() on the data
# Note: .split() does NOT return the folds directly. It returns a generator -
# an object that produces (train_indices, val_indices) tuples one at a time
# when iterated. The indices are positional into the rows, not values.
# You pass X (features) and y (labels) seperately because stratification needs
# the labels to balance the classes across folds.
folds = skf.split(df["score"], df["label"])

# Step 4: Iterate over the generator
# Each iteration produces one fold. enumerate gives a 1-based fold number.
for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
    print(f"----- Fold {fold_num} ---")
    print(f"Train indices: {train_idx}")
    print(f"Val indices:  {val_idx}")

    # Pull the actual labels at those indices to verify stratification
    train_labels = df["label"].iloc[train_idx]
    val_labels = df["label"].iloc[val_idx]

    print(f"Train labels: {list(train_labels)}")
    print(f"Val labels: {list(val_labels)}")
    print(f"Train - 1s: {sum(train_labels == 1)}/{len(train_labels)}")
    print(f"Val -1s: {sum(val_labels == 1)}/{len(val_labels)}")
    print()

    