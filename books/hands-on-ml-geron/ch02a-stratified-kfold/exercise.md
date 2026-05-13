# Exercise — StratifiedKFold on an imbalanced dataset

## Goal

Without copying from `reference.py`, build a small program that demonstrates
how `StratifiedKFold` distributes the minority class across folds — and
contrast it with what plain `KFold` does on the same data.

Read `reference.py` first to understand the mechanics. Close it. Then
implement the exercise in `main.py` from scratch.

## What `main.py` should do

1. Create a dataset of 30 rows with a binary label. Make it imbalanced —
   24 rows of class 0 and 6 rows of class 1. The features can be anything
   simple. Do not shuffle this time — keep the labels in order so the
   imbalance is visually obvious when you print the dataframe.

2. Apply `StratifiedKFold` with `n_splits=3, shuffle=True, random_state=42`.
   For each fold, print:
   - The fold number
   - The validation indices
   - The number of class-1 rows in the validation set

3. Now apply plain `KFold` with the same parameters
   (`n_splits=3, shuffle=True, random_state=42`). For each fold, print the
   same three things.

4. At the end of the script, write a one-sentence observation as a
   `print()` statement comparing what you saw between the two. What does
   stratification visibly do that plain k-fold does not?

## What you should be able to answer after running it

- What does `.split()` actually return, and why is it a generator?
- Why does StratifiedKFold need `y` passed in, but plain KFold could work
  without it?
- In your own words: when would using plain KFold on imbalanced data
  silently hurt model evaluation?

## Constraints

- Type everything manually. No copy-paste from `reference.py`.
- If you get stuck, look at `reference.py` to find the specific piece you
  need, then close it and type from understanding — not from memory of
  the exact text.
- One commit when the exercise runs cleanly and the observation makes
  sense.