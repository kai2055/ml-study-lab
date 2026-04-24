
"""
Lesson 4a — NumPy ndarray Foundations

Retype implementation from the spec in README.md.
Do not look at reference.py until the implementation works.
"""

import numpy as np
import logging
import time


# Configuring a basic logger

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def make_feature_matrix(n_rows: int, n_features: int, dtype: np.dtype, seed: int= 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f_matrix = rng.standard_normal(size=(n_rows,n_features)).astype(dtype)
    return f_matrix


def report_array(name: str, arr: np.ndarray) -> None:
    s = arr.shape
    el_type = arr.dtype
    el_im = arr.ndim
    memory_footprint = arr.nbytes / (1024 ** 2)
    col_mean = arr.mean(axis=0)
    logging.info(f"Basic info: shape: {s}, dtype: {el_type}, memory footprint: {memory_footprint}, {el_im}, Mean: {col_mean}")

    return None


def compare_dtypes(n_rows: int, n_feature: int) -> None:
    mat_a = np.zeros((3,3), dtype=np.float64)
    mat_b = np.zeros((3,3), dtype=np.float32)
    mat_c = np.zeros((3,3), dtype=np.float16)
    rep1 = report_array("mat_a", mat_a)
    rep2 = report_array("mat_b", mat_b)
    rep3 = report_array("mat_c", mat_c)
    foot_print1 = mat_a.nbytes / (1024 ** 2)
    logging.info(f"Memory footprint for {mat_a.dtype}: {foot_print1}")
    foot_print2 = mat_b.nbytes / (1024 ** 2)
    logging.info(f"Memory footprint for {mat_b.dtype}: {foot_print2}")


    return None



def demo_vectorization() -> None:

    start1 = time.perf_counter()
    data = np.arange(0, 999_999, dtype=np.float64)
    result = data * 2.5
    square = data ** 2
    start2 = time.perf_counter()

    elapsed1 = start2 - start1 

    start = time.perf_counter() 

    lst1 = []
    for i in range(0, 999_999):
        lst1.append(i ** 2)

    end = time.perf_counter()

    elapsed = end - start

    print(elapsed, elapsed1)




compare_dtypes(100_000, 50)
demo_vectorization()









    
