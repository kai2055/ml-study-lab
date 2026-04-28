

"""
Lesson 4b — reference implementation: feature inspection toolkit.

Demonstrates: basic slicing (view), boolean indexing (copy), fancy indexing (copy),
view-vs-copy inspection, and transposing. All selection operations are logged so
the caller can trace exactly what data was selected and why.
"""

import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def slice_features(
    matrix: np.ndarray,
    row_start: int,
    row_end: int,
    col_indices: list[int],
) -> np.ndarray:
    """Return a block of rows (via slice) then specific columns (via fancy index).

    The row slice is a view of the original. Fancy indexing on columns produces
    a copy — the return value owns its data.
    """
    logger.info("slice_features: input shape=%s", matrix.shape)

    row_block = matrix[row_start:row_end]  # view
    logger.info(
        "slice_features: row slice [%d:%d] shape=%s",
        row_start,
        row_end,
        row_block.shape,
    )

    result = row_block[:, col_indices]  # fancy index on columns → copy
    logger.info("slice_features: final result shape=%s", result.shape)
    return result


def filter_by_threshold(
    matrix: np.ndarray,
    col: int,
    threshold: float,
) -> np.ndarray:
    """Return all rows where matrix[:, col] > threshold.

    Boolean indexing always produces a copy. The return value owns its data.
    """
    mask = matrix[:, col] > threshold
    n_pass = int(mask.sum())
    n_total = matrix.shape[0]
    logger.info(
        "filter_by_threshold: col=%d threshold=%.4f — %d/%d rows passed",
        col,
        threshold,
        n_pass,
        n_total,
    )
    return matrix[mask]


def select_samples(
    matrix: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """Return specific rows by position using fancy indexing.

    Raises IndexError with a descriptive message if any index is out of bounds.
    Fancy indexing always produces a copy.
    """
    n_rows = matrix.shape[0]
    for i in indices:
        if i < 0 or i >= n_rows:
            raise IndexError(
                "index %d is out of bounds for matrix with %d rows" % (i, n_rows)
            )
    logger.info(
        "select_samples: requested indices=%s from matrix with %d rows",
        indices,
        n_rows,
    )
    result = matrix[indices]  # fancy indexing → copy
    logger.info("select_samples: result shape=%s", result.shape)
    return result


def demonstrate_view_vs_copy(matrix: np.ndarray) -> None:
    """Log the view-vs-copy distinction through live mutation.

    Demo A: slice → view → mutating the slice mutates the original.
    Demo B: boolean index → copy → mutating the copy leaves the original unchanged.
    """
    # --- Demo A: view mutation ---
    view = matrix[0:2, 0:2]
    before = matrix[0, 0]
    view[0, 0] = -9999.0
    after = matrix[0, 0]
    logger.info("demo A (view): matrix[0,0] before mutation = %.4f", before)
    logger.info("demo A (view): matrix[0,0] after mutating slice = %.4f", after)

    # Restore the original value so later functions see clean data
    matrix[0, 0] = before

    # --- Demo B: copy isolation ---
    mask = matrix[:, 0] > 0
    copy = matrix[mask]
    original_val = matrix[mask][0, 0]  # sample the same position before mutation
    copy[0, 0] = -9999.0
    logger.info(
        "demo B (copy): matrix value at sampled position after mutating copy = %.4f",
        matrix[mask][0, 0],
    )
    logger.info(
        "demo B (copy): original unchanged — value is still %.4f", original_val
    )


def transpose_and_inspect(matrix: np.ndarray) -> np.ndarray:
    """Return the transpose of the input matrix, logging shape before and after.

    .T returns a view with swapped strides — no data is copied.
    """
    logger.info("transpose_and_inspect: input shape=%s", matrix.shape)
    transposed = matrix.T
    logger.info("transpose_and_inspect: transposed shape=%s", transposed.shape)
    # .T is a view — it swaps the stride values in the array header without
    # allocating new memory. Mutating the return value mutates the original.
    return transposed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((6, 5))

    logger.info("=" * 60)
    logger.info("feature inspection toolkit — lesson 4b")
    logger.info("matrix shape=%s  dtype=%s", matrix.shape, matrix.dtype)
    logger.info("=" * 60)

    logger.info("--- slice_features ---")
    result = slice_features(matrix, row_start=1, row_end=4, col_indices=[0, 2, 4])

    logger.info("--- filter_by_threshold ---")
    filtered = filter_by_threshold(matrix, col=2, threshold=0.0)

    logger.info("--- select_samples ---")
    selected = select_samples(matrix, indices=[0, 3, 5])

    logger.info("--- demonstrate_view_vs_copy ---")
    demonstrate_view_vs_copy(matrix)

    logger.info("--- transpose_and_inspect ---")
    transposed = transpose_and_inspect(matrix)

    logger.info("=" * 60)
    logger.info("all functions completed")