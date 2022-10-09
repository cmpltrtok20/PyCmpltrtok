"""
Common routines for numpy
"""
import numpy as np


def shuffle(*args, seed=None):
    if seed is not None:
        np.random.seed(seed)
    m = len(args[0])
    idx = np.random.permutation(m)
    res_list = []
    for x in args:
        res_list.append(x[idx])
    return tuple(res_list)


def uint8_to_flt_by_lut(x, dtype=np.float32):
    lut = np.arange(256, dtype=dtype) / 255.  # look up table
    x_flt = lut[x]
    return x_flt


def onehot_by_lut(y, n):
    lut = np.eye(n, dtype=np.int)
    y_oh = lut[y.ravel()]
    return y_oh
