__author__ = 'dzhang'

import numpy as np

from scipy.stats import norm, rankdata
from sklearn.utils import check_array


def check_nan_friendly_finite_array(X: np.ndarray, copy=False):
    # allow nan in X but not +/- inf
    assert X.ndim <= 2, "X should have no more than 2 dimensions"
    if X.ndim == 2 and np.min(X.shape) == 1:
        return X.ravel() if not copy else X.ravel().copy()
    assert np.isfinite(np.nansum(X)), "+/- Inf or extremely large values exist in X"
    return np.array(X, copy=copy)


def winsorize(X: np.ndarray, const=3):
    """
    winsorize out-liers using median-absolute deviation method
    Note: by default, we clip everything within +/- 3 Median-Absolute-Deviations
    """
    assert const > 0, "`const` must be a positive number"
    X = check_nan_friendly_finite_array(X, copy=False)
    medians = np.nanmedian(X, axis=0)
    mads = np.nanmedian(np.abs(X - medians), axis=0)
    return np.clip(X, medians - const * mads, medians + const * mads)


def standardize(X: np.ndarray, fuzz=1e-6):
    X = check_nan_friendly_finite_array(X, copy=False)
    return (X - np.nanmean(X, axis=0)) / np.sqrt(np.nanvar(X, axis=0) + fuzz)


def ztransform(X: np.ndarray, c=3./8):
    """
    rank-based inverse normal transformation of X
    data are converted to ranks and passed through the inverse CDF of a normal
    Note : we restrict X to be a 2d numpy array
            Other possible choices of c include 1/3, 1/2. 3/8 seems to be most commonly used
    e.g.
    -------------------------------------------------------------------------------------
    X = np.random.randint(1000, size=(3000, 100))
    In [75]: %timeit ztransform(X, 0.5)
    10 loops, best of 3: 65 ms per loop
    """
    X = check_array(X, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False)
    n_rows, n_cols = X.shape
    ranks_X = np.vstack([rankdata(X[:, col], method='average') for col in range(n_cols)]).T
    return norm.ppf((ranks_X - c) / (n_rows - 2. * c + 1.))















