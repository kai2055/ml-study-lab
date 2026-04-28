


"""
Write a function `safe_array(data, dtype=None)` that:
- Accepts any sequence-like input (list, tuple, ndarray).
- Converts it to a NumPy array.
- If the input is already an ndarray with the correct dtype, returns it without copying (hint: which NumPy function gives you that behavior?).
- If `dtype` is `None`, infers the dtype from the data.
- Raises a clear `ValueError` with a helpful message if the input cannot be converted to a numeric array (e.g., contains strings that aren't numeric).

"""

import numpy as np
def safe_array(data, dtype=None):
    try:
        arr = np.asarray(data, dtype)
        
    except ValueError:
        print("Could not convert input intp a NumPy array")

    if arr.dtype == object:
        raise ValueError(
            f"Input was converted to an object array (data is non-numeric)"


        ) 
    
    return arr




def fits_in_memory(n_rows: int, n_features: int, dtype,
    budget_mb: float) -> bool:
    """Check if an array of given shape and dtype would fit 
    in a memory budget"""
    itemsize = np.dtype(dtype).itemsize 
    total_bytes = n_rows * n_features * itemsize
    total_mb = total_bytes / (1024 ** 2)
    return total_mb <= budget_mb



if __name__ == "__main__":
    # safe_array
    print(safe_array([1, 2, 3]))
    print(safe_array([1.5, 2.5], dtype=np.float32))
    print(safe_array(["hello", "world"]))   # should raise ValueError
    
    # fits_in_memory
    print(fits_in_memory(1000, 100, np.float64, 512))      # True
    print(fits_in_memory(10_000_000, 100, np.float64, 512))  # False
    print(fits_in_memory(671_000, 100, np.float64, 512))   # True, just under