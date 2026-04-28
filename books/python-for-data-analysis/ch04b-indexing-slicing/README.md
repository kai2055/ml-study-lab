

# Lesson 4b — Indexing, Slicing, Boolean Indexing, Fancy Indexing, Transposing

**Source:** Python for Data Analysis (Wes McKinney), Chapter 4 — "Basic Indexing and Slicing" through "Transposing Arrays and Swapping Axes"

**Lesson goal:** Understand how NumPy selects subsets of data, why the mechanism differs between slicing and other selection methods, and why that difference is not cosmetic — it has direct consequences for every feature pipeline you will ever write.

---

## The concept

When you slice a list in plain Python, you get a new list. A copy. Independent. Changing the copy changes nothing in the original.

NumPy does something different and it is the most important thing to understand about this chapter.

When you slice a NumPy array, you get a **view** — a window into the same block of memory, not a new allocation. The view shares the underlying data buffer with the original. Mutating the view mutates the original.

The analogy that maps to mechanics: think of a NumPy array as a roll of film negatives. The `ndarray` object in Python is just the header — it describes where the film starts, how many frames there are, and how wide each frame is. When you slice the array, NumPy creates a new header pointing partway into the same roll. You haven't duplicated the film. Both headers look at the same physical frames.

This matters because: when you change a pixel in the view, you are changing the original negative. There is only one physical film.

Boolean indexing and fancy indexing work differently. They produce **copies** — new allocations, no shared memory. The distinction is not arbitrary. Slicing is a contiguous region that can be expressed as a pointer offset, so NumPy can share memory. Boolean and fancy indexing select arbitrary non-contiguous elements — they cannot be expressed as a pointer offset, so NumPy must allocate and copy.

The rule you will use forever: **slicing gives you a view. Boolean and fancy indexing give you a copy.**

If you are not sure which you have, check: `arr.base is None` is True for a copy (owns its data), False for a view (borrows from another array).

---

## Why it matters for ML reliability

**Failure mode 1: Silent feature mutation in preprocessing**

A preprocessing function receives a feature matrix. It slices out a column to normalise it. Because it got a view, modifying the column also modifies the original matrix — which another part of the pipeline is still holding a reference to. The normalisation is applied twice, silently. The model receives corrupted features. No exception is raised. The predictions are wrong and the logs show nothing.

This is not a contrived scenario. It is one of the most common silent bugs in sklearn preprocessing code.

**Failure mode 2: Unintended data leakage through shared memory**

You split a dataset into train and validation. If the split is done via slicing, they share memory. An augmentation step applied to the training set silently modifies the validation set. Your validation metrics are computed on contaminated data. The model looks better than it is. You deploy it. It underperforms. The monitoring layer's baseline snapshot is now also incorrect because it was built from the contaminated validation data.

**Failure mode 3: Memory blowup from unnecessary copies**

A drift detection function applies boolean indexing repeatedly inside a loop — once per feature, once per batch. Each boolean index creates a full copy. On a 50-feature dataset with 100k rows, each copy is 40MB. A loop with 50 iterations allocates 2GB. The Cloud Run container OOMs and dies. It would not have died if the function had been written to avoid boolean indexing inside the loop and done the selection differently.

---

## Gap-fills — what the book skipped or glossed over

**1. The view-vs-copy rule has a corollary for higher-dimensional slicing**

When you slice a 2D array along both axes simultaneously — `arr[1:3, 0:2]` — you still get a view. NumPy can represent any rectangular contiguous subregion as a view by adjusting the header's strides. What breaks the view guarantee is non-contiguous selection (fancy indexing) or boolean masks. Keep this in mind: `arr[[0, 2], :]` is fancy indexing and gives a copy even though it looks like regular slicing.

**2. `arr[5]` and `arr[5:6]` are not the same thing**

`arr[5]` on a 2D array gives you a 1D view of row 5. Its shape is `(n_cols,)`, not `(1, n_cols)`. This matters when you pass it to a function that expects a 2D input — sklearn estimators do. `arr[5:6]` gives you a 2D view of shape `(1, n_cols)`. The difference is one colon and it silently changes the shape. When your model refuses a prediction with a shape error, check this first.

**3. Boolean indexing always copies — the filter does not know the shape in advance**

The reason boolean indexing must copy is mechanical. NumPy cannot know the shape of the result until it scans the mask. A view requires knowing strides at construction time. A boolean mask with an unknown number of True values cannot produce a view — the header cannot be written before the data is examined. So NumPy allocates and fills. This is correct behaviour, not inefficiency.

**4. Fancy indexing copies even for contiguous selections**

`arr[[0, 1, 2]]` copies even though rows 0, 1, 2 are contiguous. NumPy does not check whether your integer list happens to describe a contiguous range. It treats all integer-array indexing as fancy and always copies. If you want a view of the first three rows, use `arr[0:3]`, not `arr[[0, 1, 2]]`.

**5. Multidimensional boolean indexing with `np.ix_`**

When you want to select rows matching one mask and columns matching another, you cannot combine two boolean arrays directly in most cases. `np.ix_` converts two 1D boolean arrays into index grids that broadcast correctly. This is the right tool when your drift detection needs to select specific features (columns) and specific sample batches (rows) simultaneously.

**6. Transposing does not copy — it is a view with reversed strides**

`arr.T` returns a view. The memory layout is unchanged. NumPy simply swaps the stride values in the header — reading across rows now steps by column stride and vice versa. This is important for two reasons: it is free (no allocation), and mutating `arr.T` mutates `arr`. If you compute something on the transposed matrix and accidentally write back to it, you are writing to the original.

---

## Warm-up — retype and observe before the spec

This block is short. Retype it exactly, run it, and read every output line before moving on to the spec. The goal is to put the view-vs-copy distinction into muscle memory before you build anything larger.

```python
import numpy as np

arr = np.arange(12).reshape(3, 4)
print("original:\n", arr)

# Slice — view
view = arr[0:2, 1:3]
print("\nview before mutation:\n", view)
view[0, 0] = 999
print("original after mutating view:\n", arr)

# Boolean index — copy
mask = arr > 5
copy = arr[mask]
print("\ncopy (1D) from boolean index:\n", copy)
copy[0] = -1
print("original unchanged after mutating copy:\n", arr)

# Check ownership
arr2 = np.arange(6)
s = arr2[1:4]
print("\nslice base is arr2:", s.base is arr2)

b = arr2[arr2 > 2]
print("boolean index base is None (owns data):", b.base is None)
```

Read the output. Before moving to the spec, make sure you can answer: why did mutating `view` change `arr`, but mutating `copy` did not?

---

## Retype spec — feature inspection toolkit

Build a module called `main.py` with the following functions. The spec gives you structure and requirements, not code. You must decide how to implement each function. Use type hints throughout, the module-level logger pattern, lazy log formatting, and the `__main__` guard.

---

### Function 1: `slice_features`

**Signature:** `slice_features(matrix: np.ndarray, row_start: int, row_end: int, col_indices: list[int]) -> np.ndarray`

Select a rectangular block of rows using slicing, then select specific columns from the result using fancy indexing. Return the final selection.

**Requirements:**
- The row selection must use slicing (view).
- The column selection must use fancy indexing (copy).
- Log: the shape of the input, shape of the row slice, shape of the final result.
- Think carefully about whether the return value is a view or copy, and why. You do not need to log this — just know it.

---

### Function 2: `filter_by_threshold`

**Signature:** `filter_by_threshold(matrix: np.ndarray, col: int, threshold: float) -> np.ndarray`

Select all rows where the value in column `col` exceeds `threshold`, using boolean indexing. Return the filtered matrix.

**Requirements:**
- Build the boolean mask from the specified column only.
- Apply the mask to the full matrix (all columns) to return rows.
- Log: how many rows passed the filter, out of how many total.
- Think about whether the return value is a view or copy, and why.

---

### Function 3: `select_samples`

**Signature:** `select_samples(matrix: np.ndarray, indices: list[int]) -> np.ndarray`

Select specific rows by position using fancy indexing. Return the selected rows.

**Requirements:**
- Use a Python list of integers as the index (this is fancy indexing, not slicing).
- Log: which indices were requested, shape of result.
- Raises `IndexError` with a descriptive message if any index is out of bounds for the matrix. Check this before indexing, not after.

---

### Function 4: `demonstrate_view_vs_copy`

**Signature:** `demonstrate_view_vs_copy(matrix: np.ndarray) -> None`

A teaching function. It must perform two demonstrations and log the result of each:

**Demo A — view mutation:**
- Slice a subregion of the matrix.
- Mutate one element of the slice.
- Log: the element's value in the original matrix before and after. Log both with one message each so the before/after is clear.

**Demo B — copy isolation:**
- Build a boolean mask. Select a subset using the mask (boolean index → copy).
- Mutate one element of the copy.
- Log: the corresponding element in the original matrix to confirm it was not changed.

The function's purpose is to make the distinction visible and logged. Keep it clean and readable — this function might be read by someone debugging a silent mutation.

---

### Function 5: `transpose_and_inspect`

**Signature:** `transpose_and_inspect(matrix: np.ndarray) -> np.ndarray`

Transpose the input matrix and log its shape before and after. Return the transpose.

**Requirements:**
- Use `.T`.
- Log: original shape, transposed shape.
- In a comment directly above the return, note whether the return value is a view or copy, and why.

---

### `__main__` block

Build a 6×5 float matrix using `np.random.default_rng(42)` and call each function with sensible arguments. The output when you run `python main.py` should tell a readable story about each operation. Log at `INFO` level throughout.

---

## Pre-submit checklist

Before committing, verify every item:

- [ ] `python main.py` runs end-to-end with no exceptions
- [ ] All five functions are called from `__main__` and produce visible log output
- [ ] `slice_features` uses slicing for rows and fancy indexing for columns — not the reverse
- [ ] `filter_by_threshold` uses boolean indexing — not integer indexing
- [ ] `select_samples` raises `IndexError` on an out-of-bounds index — test this manually before committing
- [ ] `demonstrate_view_vs_copy` shows both a mutation that propagates (view) and one that does not (copy)
- [ ] Type hints on every function signature
- [ ] Module-level logger (`logger = logging.getLogger(__name__)`)
- [ ] No f-strings in log calls — `%s` formatting only
- [ ] `if __name__ == "__main__":` guard present

---

## Refactor pass

After `main.py` runs cleanly: read through the code and find one place where you repeated a pattern — logging a shape, bounds-checking, building a mask — that could be extracted into a helper or expressed more cleanly. Refactor it. Commit separately as:

```
ch4b: refactor — [one-line description of what you tightened]
```

---

## Exercises

### Concept questions

**Q1.** You write `row = matrix[3]` and `row[0] = -999`. A colleague says the matrix is unchanged because you assigned to `row`, not to `matrix`. Who is right, and why? What would you change in the code to make your colleague's assumption correct?

**Q2.** A filtering function applies a boolean mask inside a loop, once per feature (column), 50 times total. The matrix has 100,000 rows and 50 float64 columns. Estimate the memory allocated per loop iteration from the boolean indexing alone. What is the total extra allocation across all 50 iterations? What alternative avoids this?

**Q3.** `arr.T` is called a free operation. In one or two sentences, explain what "free" means mechanically — what NumPy does (and does not do) when you call `.T`.

---

### Code problems

**Problem 1**

Write a function `safe_row_select(matrix: np.ndarray, indices: list[int]) -> np.ndarray` that:
- Returns the selected rows as a copy, regardless of how it was selected internally.
- Raises a `ValueError` (not `IndexError`) with the message `"index {i} out of bounds for matrix with {n} rows"` for the first offending index found.
- Works correctly when `indices` is empty — return an empty array with the correct number of columns, not a crash.

Test with: a valid list, an out-of-bounds index, and an empty list.

**Problem 2**

You have a 2D feature matrix. Write a function `drift_candidate_rows(matrix: np.ndarray, col: int, low: float, high: float) -> np.ndarray` that returns all rows where the value in `col` is **outside** the range `[low, high]` (i.e., either below `low` or above `high`). Use boolean indexing. The function must work in a single mask expression — no loops.

Test with: a matrix where some rows are inside the range, some outside, and at least one row exactly on the boundary (confirm boundary behaviour).

---

### Solutions

---

*(Solutions below — do not read until you have attempted both problems)*

---
---
---

**Q1 answer:** You are right. `matrix[3]` returns a view of row 3. `row[0] = -999` writes through the view into the underlying memory. The matrix is changed. To make your colleague's assumption correct: `row = matrix[3].copy()` — an explicit copy breaks the shared memory link.

**Q2 answer:** Each column has 100,000 float64 elements. float64 is 8 bytes. One copy per boolean index = 100,000 × 8 = 800,000 bytes = ~0.76 MB per iteration. 50 iterations = ~38 MB in extra allocations (ignoring GC). The alternative: build one mask and apply it once outside the loop, or restructure so boolean indexing is not needed inside the loop at all.

**Q3 answer:** `.T` swaps the stride values in the array header — it changes how NumPy steps through memory when reading elements, without moving any data. No allocation, no data copy. The same bytes in the same locations, read in a different order.

---

**Problem 1 solution:**

```python
def safe_row_select(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    n_rows = matrix.shape[0]
    for i in indices:
        if i < 0 or i >= n_rows:
            raise ValueError(f"index {i} out of bounds for matrix with {n_rows} rows")
    if len(indices) == 0:
        return np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
    return matrix[indices].copy()
```

Note: `matrix[indices]` is fancy indexing which already produces a copy, so `.copy()` is technically redundant here — but calling it explicitly makes the contract clear to anyone reading this function. Explicit is better than relying on the caller knowing fancy indexing semantics.

---

**Problem 2 solution:**

```python
def drift_candidate_rows(matrix: np.ndarray, col: int, low: float, high: float) -> np.ndarray:
    mask = (matrix[:, col] < low) | (matrix[:, col] > high)
    return matrix[mask]
```

Boundary check: `< low` and `> high` are strict inequalities. A value exactly equal to `low` or `high` is inside the range and will not be returned. If the spec required inclusive boundaries to also be flagged, the operators would be `<=` and `>=`. The boundary behaviour should be explicit in any production function — document it or encode it in the function name.

---

## Reflection prompts

Fill in after completing the exercises:

1. When you ran the warm-up and mutated the view, did it change the original in the way you expected? What would have happened if you hadn't run the warm-up first?
2. Which gap-fill was most surprising to you — the one you would have gotten wrong without reading it?
3. Was there a moment in the retype where you had to stop and think about whether you were getting a view or a copy? Describe what you were doing.
4. What would you add to LEARNINGS.md after this lesson?