__author__ = 'Zilun Shen'


# Factor Covariance Matrix Estimation

from .factor_cov import FactorCovEstimator
import numpy as np

def get_cov(X):
    """Compute the factor covariance matrix from the set of daily factor returns.

    :X: N-by-F numpy 2d array, where N is the number of `time units` in the window, and F is the number of factors
            by convention, from top to bottom, factor return vectors are aligned in descending order in time. (i.e., top
            row means the latest factor return)
    :returns: F-by-F numpy 2d array
    """
    est = FactorCovEstimator('vanilla_EWMA', 'vanilla_EWMA')
    # Should call fit
    n_obs, n_feature = X.shape
    wgts = est._get_ewma_weights(hl=est._hl, tau=n_obs-1) # `tau` means how many time units from now
    wgts /= np.sum(wgts)    # normalize weights
    mu = wgts.dot(X)
    var = wgts.dot(X**2) - mu ** 2
    D = np.diag(np.sqrt(var))
    corr = est._ewma(X)
    cov = D.dot(corr).dot(D)
    # cov = var.T.dot(corr).dot(var)
    return var, corr, cov
