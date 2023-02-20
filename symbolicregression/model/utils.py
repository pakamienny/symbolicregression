import numpy as np


def extend_with_nans(arr, max_cols):
    assert arr.ndim == 2 and arr.shape[1] <= max_cols
    pad_width = max_cols - arr.shape[1]
    pad_tuple = ((0, 0), (0, pad_width))
    return np.pad(arr, pad_tuple, mode='constant', constant_values=np.nan)