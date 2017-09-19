__author__ = 'Derek Qi'

import numpy as np
import pandas as pd

def _rmmt_sn(log_return, head, tail, c=0.5):
    assert head < tail, "head %d is greater than or equal to tail" % (head, tail)
    y = np.array([0] * head + [1] * (tail - head))
    y = y / np.sum(y)
    # rmmt = np.convolve(log_return, y) - c * np.convolve(log_return ** 2, y)
    rmmt = np.convolve(log_return, y) / np.sqrt(np.convolve(log_return ** 2, y)) # momentum for unit variance within the same period
    return rmmt[:-tail+1]


def revised_momentum(univ_table, head, tail, c=0.5, naming='simple'):
    '''
    Calculates the revised momentum factor defined as follows:
    mmt_r[t] = sum(lr[head:tail]) - c * sum(lr[head:tail] ** 2)
    head and tail are numbers of time periods

    Comparing to the vanilla type of momentum, this new momentum added
    penalties on the curvature of the log return series
    '''
    name = 'revised_momentum'
    if naming == 'full':
        name += '_%s_%s' % (head, tail)
    univ_table[name] = np.nan

    rmmt_dict = {}
    datelst = np.unique(univ_table['date'])
    allTickers = np.unique(univ_table['ticker'])
    for ticker in allTickers:
        table = univ_table.xs(ticker, level='ticker')
        lr = np.diff(np.log(table['price'])) # log return series
        lr = np.insert(lr, 0, 0)
        rmmt = _rmmt_sn(lr, head, tail, c)
        univ_table.xs(ticker, level='ticker')[name] = rmmt

    for t in datelst:
        table = univ_table.xs(t, level='date')[[name]].copy()
        table = table.reset_index()
        table.dropna(inplace = True)
        rmmt_dict[t] = table
    return rmmt_dict
