"""
Common routines for numpy
"""
import numpy as np
from typing import List


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
    lut = np.eye(n, dtype=np.int64)
    y_oh = lut[y.ravel()]
    return y_oh


def normalize(embeddings: List[List[float]], is_broadcast: bool = True) -> np.ndarray:
    '''
    sklearn.preprocessing.normalize 的替代（使用 L2），避免安装 scipy, scikit-learn
    '''
    norm = np.linalg.norm(embeddings, axis=1)
    norm = np.reshape(norm, (norm.shape[0], 1))
    if not is_broadcast:
        norm = np.tile(norm, (1, len(embeddings[0])))
    norm = np.divide(embeddings, norm)
    return norm


def normalize_one(vector: List[float]) -> np.ndarray:
    embeddings = [vector]
    embeddings = normalize(embeddings)
    vector = embeddings[0]
    return vector
