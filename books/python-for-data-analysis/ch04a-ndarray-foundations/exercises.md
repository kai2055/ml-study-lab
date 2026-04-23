
# Exercises — Lesson 4a

Attempt all of these before scrolling past the divider. Write concept answers in your own words; don't paste book text.

## Concept questions

1. Explain in one paragraph why `arr * 2` is roughly 70× faster than `[x * 2 for x in lst]` for a million-element sequence. Reference both *contiguous memory* and *interpreted vs compiled execution* in your answer.

2. You see this in a teammate's code:
```python
   arr = np.array([1, 2, 3])
   arr = arr.sort()
   print(arr)
```
   What gets printed, and why? What's the fix?

3. A junior engineer says: "We should always use `float32` instead of `float64` because it uses half the memory." Give two reasons this is not always good advice in an ML pipeline.

## Code problems

**Problem 1 — Safe array constructor**

Write a function `safe_array(data, dtype=None)` that:
- Accepts any sequence-like input (list, tuple, ndarray).
- Converts it to a NumPy array.
- If the input is already an ndarray with the correct dtype, returns it without copying (hint: which NumPy function gives you that behavior?).
- If `dtype` is `None`, infers the dtype from the data.
- Raises a clear `ValueError` with a helpful message if the input cannot be converted to a numeric array (e.g., contains strings that aren't numeric).

**Problem 2 — Memory budget checker**

Write a function `fits_in_memory(n_rows, n_features, dtype, budget_mb)` that returns `True` if a `(n_rows, n_features)` array of the given dtype would fit within `budget_mb` megabytes, `False` otherwise. Don't actually allocate the array — compute the size from the dtype's `itemsize` attribute. Then write a small demo that shows how many `float64` rows of 100 features fit in a 512 MB budget.

---
---
---

## SOLUTIONS — do not read until you have attempted both problems

### Concept answers

1. Two compounding effects. **Contiguous memory:** a NumPy array stores all values in one block with a known fixed stride, so the CPU can prefetch the next values, and the loop has no pointer indirection. A Python list stores pointers to scattered `PyObject` instances, so each access is an unpredictable memory jump that defeats the cache. **Interpreted vs compiled execution:** `arr * 2` dispatches a single call to a precompiled C routine that walks the buffer with no per-element Python overhead. `[x * 2 for x in lst]` runs a Python interpreter step for every single element — type check, bytecode dispatch, allocate a new `PyLong` or `PyFloat`, append. The constant-factor cost per element is something like 50–100× higher in pure Python.

2. Prints `None`. `arr.sort()` mutates the array in place and returns `None`, and the `arr = arr.sort()` line then rebinds the name `arr` to that `None`. The original sorted array is gone (no other references). Fix: either `arr.sort()` on its own line (in-place, keep the original name), or `arr = np.sort(arr)` (returns a new sorted array). Same family as the `list.append` and `list.sort` traps, just with a NumPy version of the API to memorize.

3. (a) **Precision loss matters in some models.** `float32` has about 7 significant decimal digits; `float64` has about 15–16. Gradient computations in deep networks, scaled features with very small variances, or financial calculations can accumulate rounding errors that meaningfully shift model behavior. (b) **Compatibility downstream.** Many libraries (scikit-learn, pandas, scipy) default to `float64`, and mixing dtypes triggers silent upcasts that allocate copies — you can end up using *more* memory than if you'd stayed in `float64`, plus the conversions cost CPU. The right answer is "measure both" — `float32` is great for inference at scale, often wrong for training-time numerical stability.

### Problem 1 solution

```python
import numpy as np


def safe_array(data, dtype=None):
    """Convert input to ndarray, avoiding a copy when possible."""
    try:
        # asarray avoids copy if input is already an ndarray with matching dtype.
        # If dtype is None, asarray infers it.
        arr = np.asarray(data, dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not convert input to a NumPy array. "
            f"Input type: {type(data).__name__}. Original error: {exc}"
        ) from exc

    # Reject object arrays — they usually mean we got mixed/non-numeric data.
    if arr.dtype == object:
        raise ValueError(
            f"Input was converted to an object array, which usually means "
            f"the data is non-homogeneous or non-numeric. Got dtype={arr.dtype}."
        )

    return arr
```

Key points: `np.asarray` is the no-copy version of `np.array`. Wrapping `from exc` in the raise preserves the original error chain — a small habit that helps massively when debugging deep stack traces in production.

### Problem 2 solution

```python
import numpy as np


def fits_in_memory(n_rows: int, n_features: int, dtype, budget_mb: float) -> bool:
    """Check if an array of given shape and dtype would fit in a memory budget."""
    itemsize = np.dtype(dtype).itemsize  # bytes per element
    total_bytes = n_rows * n_features * itemsize
    total_mb = total_bytes / (1024 ** 2)
    return total_mb <= budget_mb


if __name__ == "__main__":
    budget = 512  # MB, e.g. a small Cloud Run container
    n_features = 100
    itemsize = np.dtype(np.float64).itemsize
    max_rows = (budget * 1024 ** 2) // (n_features * itemsize)
    print(f"Max float64 rows of {n_features} features in {budget}MB: {max_rows:,}")
    # ~671,088 rows
```

The lesson here is that you can reason about array memory **without ever allocating the array**. This is exactly the kind of pre-flight check that prevents OOM crashes in production — and it's a one-liner with `np.dtype().itemsize`.