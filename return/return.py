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

def GenReturn(factor_return_file, factor_exposure_file, stock_list_file):
    """This function gives expcted return and its covariance, as well as
    the stock list

    :factor_return_file: factor return file
    :factor_exposure_file: factor exposure file
    :stock_list_file: a list consist of all stocks considered in our model
    :returns: (expected return, expected return variance, stock_list)

    """
    # Read factor returns
    factor_returns = pd.read_csv(factor_return_file, header=None)
    expected_factor_return = factor_returns.mean()
    expected_factor_cov_naive = factor_returns.cov()

    # Read factor exposures
    factor_exposure = pd.read_csv(factor_exposure_file, header=None)

    # Exclude the first column (date)
    var, corr, expected_factor_cov = get_cov(factor_returns.iloc[:,1:])
    # print("EWMA covariance matrix")
    # print(expected_factor_cov)
    # print("Naive covariance matrix")
    # print(expected_factor_cov_naive)

    # TODO: the varaince of error is hard coded here
    expected_return, expected_return_cov = GetExpectedReturn(
                                            expected_factor_return, 
                                            expected_factor_cov,
                                            np.identity(factor_exposure.shape[0]) * 0.00700037700382  # 0.0166834997298, 
                                            factor_exposure)

    # load stock list
    stock_list = [line.strip() for line in open(stock_list_file, 'r')]
    return expected_return, expected_return_cov, stock_list

if __name__ == "__main__":

    # Now the filter is set up as vol > 1M
    # There are 1822 stocks in the stock pool
    expected_return, expected_return_cov, stock_list = GenReturn('./factor_return_new.csv', 
                                                                 './factor_exposure_matrix.csv',
                                                                 './stock_list.csv')
    print(expected_return.shape)
    print(expected_return_cov.shape)
    print(len(stock_list))
