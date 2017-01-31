# This script reads factor return and factor exposure from .csv files
# and gives the first an second moments of the asset returns
# Author: Zilun Shen
# Date: 1/26/2017

import pandas as pd
import numpy as np
from return_cov.factor_return_cov import get_cov

def GetExpectedReturn(uf, F, D, X):
    """Calculate expected return and covariance matrix

    :uf: expected factor return
    :F: expected factor return covariance matrix
    :D: A diagonal matrix of variance of error term in APT model
    :X: factor exposure
    :returns: (expected return, expected return variance)

    """
    expected_return = np.dot(X, uf)
    expected_return_cov = np.dot(np.dot(X, F), X.T) + D
    return expected_return, expected_return_cov

if __name__ == "__main__":
    # Read factor returns
    factor_returns = pd.read_csv('./factor_returns_20140103_20160101.csv')
    expected_factor_return = factor_returns.mean()
    expected_factor_cov_naive = factor_returns.cov()

    # Read factor exposures
    factor_exposure = pd.read_csv('./factor_exposure_matrix_20160101.csv', header=None).head()

    # Exclude the first column (date)
    var, corr, expected_factor_cov = get_cov(factor_returns.iloc[:,1:])
    print("EWMA covariance matrix")
    print(expected_factor_cov)
    print("Naive covariance matrix")
    print(expected_factor_cov_naive)

    # TODO: replace 0 with D
    print(GetExpectedReturn(expected_factor_return, expected_factor_cov, 0, factor_exposure))

    # estimate D, a diagonal matrix of variance of error term in APT model
    price = pd.read_csv('./r3000_price.csv')
