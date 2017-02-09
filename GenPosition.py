__author__ = 'Derek Qi'

# Generates the optimized position for a single period

import numpy as np
import pandas as pd
from factor_model import alpha
from optimization import optimizer as opt
from time import time


def GenPosition(factor_return_file, factor_exposure_file, stock_list_file, hasshort=False):
    """
    :return: stock list and the corresponding weight in optimized portfolio
    """
    ret, sigma, stock_list = alpha.GenReturn(factor_return_file, factor_exposure_file, stock_list_file)
    N_INS = ret.shape[0]
    ret = np.reshape(ret, (N_INS, 1))
    w_old = np.ones([N_INS, 1]) / N_INS # Start from an evenly-split portfolio and assign no position-changing limits

    if hasshort:
        w_opt = opt.optimizerlongshort(w_old, alpha=ret, sigma=sigma)
    else:
        w_opt = opt.optimizer(w_old, alpha=ret, sigma=sigma)
    return stock_list, w_opt


def PositionFilter(w, tol=1e-4):
    """
    Filter out very small positions (in absolute values) and re-normalize
    :return: position vector filtered
    """
    w[abs(w) < tol] = 0
    w /= sum(w)
    return w


if __name__ == "__main__":
    start = time()
    datadir = './datainput/'
    outputdir = './output/'
    factor_return_file, factor_exposure_file, stock_list_file = datadir + 'factor_return_new.csv', datadir + 'factor_exposure_matrix.csv', datadir + 'stock_list.csv'
    # Long-only position
    stock_list, w_opt = GenPosition(factor_return_file, factor_exposure_file, stock_list_file, hasshort=False)
    w_opt = PositionFilter(w_opt)
    result = pd.DataFrame({"Ticker": stock_list, "Weight": list(w_opt.T[0])})
    result.to_csv(outputdir + 'portfolio_long_only.csv', index=False)
    pause = time()
    print(pause - start)
    # Long-Short position
    stock_list, w_opt = GenPosition(factor_return_file, factor_exposure_file, stock_list_file, hasshort=True)
    w_opt = PositionFilter(w_opt)
    result = pd.DataFrame({"Ticker": stock_list, "Weight": list(w_opt.T[0])})
    result.to_csv(outputdir + 'portfolio_long_short.csv', index=False)
    end = time()
    print(end - pause)
