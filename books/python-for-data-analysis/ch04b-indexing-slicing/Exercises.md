

# Lesson 4b — Exercises

## Concept questions

**Q1.** You write `row = matrix[3]` and `row[0] = -999`. A colleague says the matrix is unchanged because you assigned to `row`, not to `matrix`. Who is right, and why? What would you change in the code to make your colleague's assumption correct?

**Q2.** A filtering function applies a boolean mask inside a loop, once per feature (column), 50 times total. The matrix has 100,000 rows and 50 float64 columns. Estimate the memory allocated per loop iteration from the boolean indexing alone. What is the total extra allocation across all 50 iterations? What alternative avoids this?

**Q3.** `arr.T` is called a free operation. In one or two sentences, explain what "free" means mechanically — what NumPy does (and does not do) when you call `.T`.

---

## Code problems

**Problem 1**

Write a function `safe_row_select(matrix: np.ndarray, indices: list[int]) -> np.ndarray` that:
- Returns the selected rows as a copy, regardless of how it was selected internally.
- Raises a `ValueError` (not `IndexError`) with the message `"index {i} out of bounds for matrix with {n} rows"` for the first offending index found.
- Works correctly when `indices` is empty — return an empty array with the correct number of columns, not a crash.

Test with: a valid list, an out-of-bounds index, and an empty list.

**Problem 2**

Write a function `drift_candidate_rows(matrix: np.ndarray, col: int, low: float, high: float) -> np.ndarray` that returns all rows where the value in `col` is **outside** the range `[low, high]` (values below `low` or above `high`). Use boolean indexing. The function must work in a single mask expression — no loops.

Test with: a matrix where some rows are inside the range, some outside, and at least one row exactly on the boundary (confirm boundary behaviour — is it inside or outside? is that what you intended?).

---
---
---

# Solutions — do not read until you have attempted both problems

---
---
---

**Q1 answer:** You are right. `matrix[3]` returns a view of row 3. `row[0] = -999` writes through the view into the underlying memory. The matrix is changed. To make your colleague's assumption correct: `row = matrix[3].copy()` — an explicit copy breaks the shared memory link, and subsequent writes to `row` no longer affect `matrix`.

**Q2 answer:** Each column has 100,000 float64 elements. float64 is 8 bytes. One boolean-index copy per iteration = 100,000 × 8 = 800,000 bytes ≈ 0.76 MB. 50 iterations = ~38 MB in extra allocations (not counting the original matrix). The alternative: build one mask and apply it once before the loop, or restructure so boolean indexing is not needed inside the loop at all.

**Q3 answer:** `.T` swaps the stride values in the array header — it changes how NumPy steps through memory when reading elements, without moving any data. No allocation, no data copy. The same bytes in the same locations, read in a different traversal order.

---

**Problem 1 solution:**

```python
def safe_row_select(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    n_rows = matrix.shape[0]
    for i in indices:
        if i < 0 or i >= n_rows:
            raise ValueError(
                f"index {i} out of bounds for matrix with {n_rows} rows"
            )
    if len(indices) == 0:
        return np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
    return matrix[indices].copy()
```

Note: `matrix[indices]` is fancy indexing which already produces a copy. The explicit `.copy()` call is redundant here but makes the contract visible — a future reader should not have to remember fancy indexing semantics to know whether this function returns owned data.

---

**Problem 2 solution:**

```python
def drift_candidate_rows(
    matrix: np.ndarray, col: int, low: float, high: float
) -> np.ndarray:
    mask = (matrix[:, col] < low) | (matrix[:, col] > high)
    return matrix[mask]
```

Boundary note: `< low` and `> high` are strict inequalities. A value exactly equal to `low` or `high` is considered inside the range and will not be returned. If your use case requires boundary values to be flagged, use `<=` and `>=`, and document the choice explicitly — boundary behaviour in drift thresholds has direct consequences for alert frequency.