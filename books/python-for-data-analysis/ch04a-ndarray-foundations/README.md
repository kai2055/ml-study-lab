
# Lesson 4a — NumPy ndarray Foundations

**Source:** *Python for Data Analysis* (Wes McKinney), Chapter 4, sections introduction through "Data Types for ndarrays" plus "Arithmetic with NumPy Arrays."

**Goal:** Build a correct mental model of what an ndarray *is*, why it's fast, how dtypes work, and what vectorized arithmetic actually means — before touching indexing or any of the harder material.

---

## 1. The concept

An ndarray is **one contiguous block of memory** holding values of a single, fixed type, plus a small header describing how to interpret that block (shape, dtype, strides).

That sentence is the whole foundation. Everything else NumPy does — speed, broadcasting, vectorization — falls out of those two facts: **contiguous memory** and **homogeneous type**.

### Analogy that maps to the mechanics

A Python list is a **box of envelopes**. Each envelope sits somewhere different in memory, and inside each envelope is a pointer to the actual value (which itself could be any type — int, string, another list, anything). To add `1` to every element, Python has to: open envelope, check what type is inside, decide how to add, write result, move to the next envelope. Repeat a million times. Each step is an interpreted Python instruction.

A NumPy array is a **railway carriage of identical seats**. All values sit shoulder-to-shoulder in one continuous stretch of memory. The dtype is the seat specification — every seat is exactly 8 bytes (for `float64`), no gaps, no envelopes, no per-element type checks. To add `1` to every element, NumPy hands the whole carriage to a C loop that walks the memory once with no Python overhead.

This is why the book's benchmark shows ~70× speedup. It's not a clever algorithm. It's the absence of overhead.

The analogy also explains why NumPy is **strict about dtype**: you can't put a string in a seat sized for a `float64`. If you try to mix types, NumPy upcasts everything to a common type (or refuses) — because the carriage can only have one seat specification.

### What "vectorization" actually means

When you write `arr * 2`, you are not asking Python to loop. You are asking NumPy to dispatch a single C-level operation that walks the contiguous block and produces another contiguous block. The Python expression is a *handle* to the operation; the work happens below Python.

This is the mental flip the chapter is trying to install: **stop thinking element-by-element, start thinking whole-array-at-once.**

---

## 2. Why it matters for ML reliability

Three reasons this foundation matters for the kind of work you're building toward:

**Performance is correctness at scale.** Drift detection on a streaming pipeline that processes 10M rows a day cannot afford Python-level loops. The PSI/KS calculations in Project 2 are vectorized NumPy under the hood — if you understand *why* `arr * 2` is fast, you understand why scipy's KS test is fast, which is the same reason your drift detector can run inline with predictions instead of as a nightly batch job.

**Dtype awareness prevents silent bugs.** A real failure mode in production ML: a feature column gets cast from `float64` to `float32` somewhere in the pipeline, the model loses three decimal places of precision, drift metrics start shifting, no one notices for a week. Knowing that `int64`, `float64`, and `float32` are *different objects in memory*, and that `astype` always copies, is what lets you reason about precision loss and memory blow-ups in a feature pipeline.

**Memory is a real constraint.** A `float64` array of one million rows × 100 features is 800 MB. A `float32` version is 400 MB. In a Cloud Run container with 512 MB, that difference is OOM vs. running. The dtype isn't a detail — it's a deployment constraint.

---

## 3. Gap-fills (things the chapter assumed or skipped)

**Why "homogeneous type" is non-negotiable.** The book says it but doesn't dwell. The reason is memory layout: if seat N is 8 bytes and seat N+1 is 8 bytes, NumPy can compute the address of any element with `base_address + N * 8`. With a Python list, there is no such formula — every access is a pointer chase. This `O(1)` random access with no pointer chase is the foundation of every fast NumPy operation.

**`np.empty` is not `np.zeros`.** The book warns about this in passing. The mechanism: `np.empty` allocates memory and *does not touch it*. Whatever bits were left there from the previous occupant are what you get. This is faster than zeroing, which is why it exists, but it means `np.empty(5)` can return `array([3.7e-310, 0., 0., 6.9e-310, 0.])` or anything. Use `np.empty` only when you will *immediately* fill every element. Otherwise use `np.zeros`.

**The `int64` vs platform-dependent default.** On modern 64-bit Linux/Mac, `np.array([1, 2, 3]).dtype` is `int64`. On older Windows builds it can be `int32`. If you write code assuming `int64` and someone runs it on a machine that defaults to `int32`, integer overflow can silently appear at large values. Best practice: when integer range matters, **specify the dtype explicitly**: `np.array([1, 2, 3], dtype=np.int64)`. This is exactly the kind of cross-environment surprise that makes ML pipelines flaky.

**`astype` always copies — even when the dtype is the same.** This is in the book but worth flagging because it's a memory trap. `arr.astype(arr.dtype)` allocates a fresh array. If you're doing this inside a loop over millions of rows, you've just built an O(N²) memory pattern by accident.

**`np.string_` is a trap and the book softly warns you.** It's fixed-width ASCII and silently truncates. In any real text-handling code, use object arrays or pandas. Never reach for `np.string_` unless you're interfacing with a C library that demands it.

**Pitfall: in-place vs returning a new array.** This is the cousin of your known `.sort()` / `.append()` returning `None` trap, but with a NumPy twist:
- `arr.astype(np.float32)` — returns a new array, original unchanged.
- `arr.sort()` — sorts **in place**, returns `None`.
- `np.sort(arr)` — returns a new array, original unchanged.
- `arr * 2` — returns a new array, original unchanged.
- `arr *= 2` — modifies in place, returns the array.

The pattern: **methods that mutate in place return `None`. Methods and functions that produce a new array return the new array.** Same trap, same fix: never write `arr = arr.sort()` or you'll lose your data and feel briefly insane.

**`np.array` vs `np.asarray`.** Both convert input to ndarray. `np.array` *always copies* by default; `np.asarray` only copies if the input isn't already an ndarray. If you're writing a function that accepts "anything array-like," use `np.asarray` — it's a no-op when the caller already passed you an array. This pattern is everywhere in scikit-learn's source.

---

## 4. Retype program — spec only

You're going to build a small **dataset profiler** — a script that takes a 2D numeric array (think: a feature matrix) and reports its shape, dtype, memory footprint, and basic per-column health stats, then demonstrates that dtype choice meaningfully changes memory usage.

This is the same structural shape as the book's "create array, inspect properties, do arithmetic" arc — but framed as something you'd actually use when sanity-checking a CSV before training a model.

### Requirements

Create `main.py`. Do **not** look at `reference.py` until you've finished or are genuinely stuck.

1. **Imports and setup**
   - Import `numpy as np`.
   - Import `logging` from the stdlib. Configure a basic logger at INFO level with a format that includes the level name and message. (Yes, use `logging`, not `print`. Get used to it now — every production script you write will use logging, and you might as well build the muscle on day one.)

2. **Function: `make_feature_matrix(n_rows: int, n_features: int, dtype: np.dtype, seed: int = 42) -> np.ndarray`**
   - Use `np.random.default_rng(seed)` to create a generator (the modern API, not the legacy `np.random.rand`).
   - Generate a `(n_rows, n_features)` array of standard normal values.
   - Cast it to the requested dtype using `.astype()`.
   - Return the result.

3. **Function: `report_array(name: str, arr: np.ndarray) -> None`**
   - Log the array's name, shape, ndim, dtype, total number of elements, and memory footprint in MB.
   - Memory footprint = `arr.nbytes / (1024 ** 2)`, rounded to 3 decimal places.
   - Log the per-column mean and per-column standard deviation as separate lines. Hint: this is where `axis=0` matters; we'll go deeper on axis semantics in lesson 4c, but for now: `axis=0` collapses rows, leaving one value per column.

4. **Function: `compare_dtypes(n_rows: int, n_features: int) -> None`**
   - Build the same logical matrix in three dtypes: `np.float64`, `np.float32`, `np.float16`.
   - Call `report_array` for each.
   - Log a final summary line stating the memory ratio of float64 vs float16 (it should be ~4×).

5. **Function: `demo_vectorization() -> None`**
   - Create an array of one million `float64` values from 0 to 999_999 using `np.arange`.
   - Multiply by `2.5` — store result.
   - Square it — store result.
   - Compare against the equivalent Python `list` operation using `time.perf_counter()` (not `time.time()` — `perf_counter` is monotonic and has higher resolution, which matters for short benchmarks).
   - Log both elapsed times and the speedup ratio (e.g., "NumPy was 73.2× faster").

6. **Main block**
   - Use the `if __name__ == "__main__":` guard. (The reason: this lets the file be imported without auto-executing — tests will need this later.)
   - Call `compare_dtypes(100_000, 50)`.
   - Call `demo_vectorization()`.

### Things to think about while writing it

- Where does `astype` introduce a copy, and where could that be a problem?
- What happens to your standard normal values when you cast to `float16`? (Run it and see — note any precision warnings or oddities.)
- Why is `axis=0` the right axis for "per-column" stats on a `(rows, features)` matrix?

---

## 5. Exercises

See `exercises.md`. Attempt the concept questions in writing (a comment block at the bottom of `main.py` is fine), then the code problems. Solutions are below the divider in that file — no peeking.