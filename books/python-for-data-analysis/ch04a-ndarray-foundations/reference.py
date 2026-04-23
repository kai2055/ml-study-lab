
"""
Reference implementation for Lesson 4a — NumPy ndarray Foundations.

Do not read this until you have attempted main.py yourself.
"""

import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def make_feature_matrix(
    n_rows: int,
    n_features: int,
    dtype: np.dtype,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic feature matrix with the requested dtype."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=(n_rows, n_features))
    return arr.astype(dtype)


def report_array(name: str, arr: np.ndarray) -> None:
    """Log shape, dtype, memory, and per-column stats for an array."""
    memory_mb = round(arr.nbytes / (1024 ** 2), 3)
    logger.info(
        "Array '%s' | shape=%s | ndim=%d | dtype=%s | n_elements=%d | memory=%.3f MB",
        name, arr.shape, arr.ndim, arr.dtype, arr.size, memory_mb,
    )
    col_means = arr.mean(axis=0)
    col_stds = arr.std(axis=0)
    logger.info("  per-column mean (first 5): %s", col_means[:5])
    logger.info("  per-column std  (first 5): %s", col_stds[:5])


def compare_dtypes(n_rows: int, n_features: int) -> None:
    """Build the same matrix in three dtypes and report the memory difference."""
    dtypes = [np.float64, np.float32, np.float16]
    sizes = {}
    for dtype in dtypes:
        arr = make_feature_matrix(n_rows, n_features, dtype)
        report_array(f"matrix_{np.dtype(dtype).name}", arr)
        sizes[np.dtype(dtype).name] = arr.nbytes

    ratio = sizes["float64"] / sizes["float16"]
    logger.info("Memory ratio float64 / float16 = %.2fx", ratio)


def demo_vectorization() -> None:
    """Compare NumPy vectorized arithmetic against equivalent Python list ops."""
    n = 1_000_000

    # NumPy path
    np_start = time.perf_counter()
    arr = np.arange(n, dtype=np.float64)
    arr_scaled = arr * 2.5
    arr_squared = arr_scaled ** 2
    np_elapsed = time.perf_counter() - np_start

    # Pure Python path
    py_start = time.perf_counter()
    py_list = list(range(n))
    py_scaled = [x * 2.5 for x in py_list]
    py_squared = [x ** 2 for x in py_scaled]
    py_elapsed = time.perf_counter() - py_start

    speedup = py_elapsed / np_elapsed
    logger.info("NumPy elapsed:  %.4f s", np_elapsed)
    logger.info("Python elapsed: %.4f s", py_elapsed)
    logger.info("NumPy was %.1fx faster", speedup)

    # Sanity check: outputs match (within float tolerance)
    assert np.allclose(arr_squared, np.array(py_squared))


if __name__ == "__main__":
    compare_dtypes(n_rows=100_000, n_features=50)
    demo_vectorization()