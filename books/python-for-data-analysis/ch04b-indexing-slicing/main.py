


"""
Select a rectangular block of rows using slicing,
then select specific columns from the result using fancy indexing.
Return the final selection

"""
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)



def slice_features(
        matrix: np.ndarray,
        row_start: int,
        row_end: int,
        col_indices: list[int]
        )-> np.ndarray:
    view = matrix[row_start:row_end, ]
    result = view[:, col_indices]
    
    logger.info(
        "input_shape=%s | row_shape=%s | result_shape=%s",
        matrix.shape, view.shape, result.shape 
    )
    
    return result




if __name__ == "__main__":

    matrix = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])

    row_start = 1
    row_end = 3  
    col_indices = [0, 2]

    result = slice_features(matrix, row_start, row_end, col_indices)
    print(result)



