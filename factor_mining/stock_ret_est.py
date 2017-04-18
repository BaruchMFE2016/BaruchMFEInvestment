__author__ = 'Zilun Shen'
# This script reads factor return and factor exposure from .csv files
# and gives the first an second moments of the asset returns
# Author: Zilun Shen
# Date: 1/26/2017

import pandas as pd
import numpy as np
from .factor_return_cov import get_cov

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

def GenReturn(factor_returns, factor_exposure):
    """This function gives expcted return and its covariance, as well as
    the stock list

    :factor_returns: dataframe of all factor returns in a period of time
    :factor_exposure: factor exposure of a single period of all stocks, it should be filtered
    :returns: (expected return, expected return variance, stock_list)

    """

    stock_list = factor_exposure['ticker'].tolist()
    factor_returns = factor_returns.ix[:,1:] # exclude date column
    factor_exposure = factor_exposure.ix[:,2:] # exclude date and ticker
    
    expected_factor_return = factor_returns.mean()
    expected_factor_cov_naive = factor_returns.cov()

    var, corr, expected_factor_cov = get_cov(factor_returns)
    # print("EWMA covariance matrix")
    # print(expected_factor_cov)
    # print("Naive covariance matrix")
    # print(expected_factor_cov_naive)

    # TODO: the varaince of error is hard coded here
    expected_return, expected_return_cov = GetExpectedReturn(
                                            expected_factor_return, 
                                            expected_factor_cov,
                                            np.identity(factor_exposure.shape[0]) * 0.00700037700382,  # 0.0166834997298, 
                                            factor_exposure)

    return expected_return, expected_return_cov, stock_list

if __name__ == "__main__":

    # Now the filter is set up as vol > 1M
    # There are 1822 stocks in the stock pool
    datadir = '../datainput/'
    expected_return, expected_return_cov, stock_list = GenReturn('../datainput/factor_return_new.csv',
                                                                 '../datainput/factor_exposure_matrix.csv',
                                                                 '../datainput/stock_list.csv')
    print(expected_return.shape)
    print(expected_return_cov.shape)
    print(len(stock_list))
