# Author: Derek Qi
# Date: 2/6/2017
# Check consistency of 2 pools, re-formulate the old pool to the new pool.
import numpy as np
import pandas as pd

def poolcheck(oldpool, w_old, newpool):
    """
    :param oldpool: list of strings of tickers contains the old pool
    :param w_old: numpy array with the old weight in old pool
    :param newpool: list of strings of tickers contains the new pool
    :return: adjusted numpy array with only new pool products and old pool weights
    """
    newdf = pd.DataFrame({'ticker': newpool})
    olddf = pd.DataFrame({'ticker': oldpool, 'weight': w_old})
    newdf = newdf.join(olddf.set_index('ticker'), on='ticker')
    newdf = newdf.fillna(0)
    upd_w_old = np.array(newdf['weight'])
    n = len(newpool)
    upd_w_old = np.reshape(upd_w_old, (n, 1))
    return upd_w_old


if __name__ == "__main__":
    newpool = ['AAPL', 'AMZN', 'SBUX', 'TGT']
    oldpool = ['AAPL', 'GOOG', 'SBUX']
    w_old = np.array([0.4, 0.5, 0.1])
    upd_w_old = poolcheck(oldpool, w_old, newpool)
    print(upd_w_old)
