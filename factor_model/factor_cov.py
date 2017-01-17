# Note: The factor covariance matrix estimation process is usually broken down into multi-steps
# 1. Estimation of factor correlation matrix
# 2. Estimation of factor volatitlies
# 1. and 2. combined will output the estimated covariance matrix


__author__ = 'dzhang'

import numpy as np
import numbers

from sklearn.utils import check_array

# a list of possible factor correlation estimation methods
__all_corr_methods__ = [
    'vanilla_EWMA',
    'newey_west_adj_EWMA',
    'random_matrix',
    # ...
]

# a list of possible factor volatility estimation methods
__all_vol_methods__ = [
    'vanilla_EWMA',
    'newey_west_adj_EWMA',
    'rough_vol',
    # ...
]


class FactorCovEstimator(object):

    def __init__(self, corr_est_method : str, vol_est_method : str, hl=252):
        assert corr_est_method in __all_corr_methods__, \
            "Correlation estimation method {} is not currently supported.\n" \
            "Valid correlation estimation methods include {}\n".format(corr_est_method, __all_corr_methods__)
        assert vol_est_method in __all_vol_methods__, \
            "Volatility estimation method {} is not currently supported.\n" \
            "Valid volatility estimation methods include {}".format(vol_est_method, __all_vol_methods__)

        self._corr_est_method, self._vol_est_method = corr_est_method, vol_est_method
        self._hl = hl

    def _get_ewma_weights(self, hl, tau):
        """
        Parameters
        ----------
        hl : int, half-life, how many time units from now (T) to the past (t) will receive a weight of 0.5
        tau : int, time units from date t in the past to now (T)

        Note the "time units" used in `hl` and `tau` must be the same (e.g., we can't have `hl` measured in days,
        whereas `tau` measured in months.

        Returns
        -------
        numpy 1d array of weights

        e.g.
        ------------------
        In [31]: _get_ewma_weights(5, 10)
        Out[31]:
        array([ 1.        ,  0.87055056,  0.75785828,  0.65975396,  0.57434918,
                0.5       ,  0.43527528,  0.37892914,  0.32987698,  0.28717459,
                0.25      ])
        """
        if not isinstance(hl, (numbers.Integral, np.integer)) & isinstance(tau, (numbers.Integral, np.integer)):
            raise ValueError("`hl` and `tau` must be integer type")
        assert hl > 0 and tau >= 0, "`hl` must be positive, and `tau` must be non-negative by definition"
        delta = 0.5 ** (1. / hl)
        return delta ** np.arange(start=0, stop=tau+1, step=1)

    # This is a fast implementation - for matrix of size 100000-by-1000, this function takes:
    # 1 loop, best of 3: 1.53 s per loop (as a comparison, numpy corrcoef takes
    # 1 loop, best of 3: 1.13 s per loop)
    def _ewma(self, X):
        """
        Compute the factor correlation matrix from the set of daily factor returns.
        We employ exponentially weighted averages, characterized by the factor correlation half-life parameter.
        This approach gives more weight to recent observations and is an effective method for dealing with data non-stationarity.

        To ensure a well-conditioned correlation matrix, the half-life must be sufficiently long
        so that the effective number of observations T is significantly greater than the number of factors K .
        On the other hand, if the correlation half-life is too long, then undue weight is placed on distant observations
        that have little relation to current market conditions.

        Parameters
        ----------
        X : N-by-F numpy 2d array, where N is the number of `time units` in the window, and F is the number of factors
            by convention, from top to bottom, factor return vectors are aligned in descending order in time. (i.e., top
            row means the latest factor return)
        """
        X = check_array(X, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False)
        n_obs, n_feature = X.shape
        wgts = self._get_ewma_weights(hl=self._hl, tau=n_obs-1) # `tau` means how many time units from now
        wgts /= np.sum(wgts)    # normalize weights
        mu = wgts.dot(X)
        var = wgts.dot(X**2) - mu ** 2
        X_normalized_wgted = ((X - mu) / np.sqrt(var)) * np.sqrt(wgts[:, None])
        return X_normalized_wgted.T.dot(X_normalized_wgted)

    def fit(self, X):
        """
        This should be a wrapper on all methods
        """
        pass










