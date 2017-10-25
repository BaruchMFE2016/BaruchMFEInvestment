__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime

def _smmt_sn(log_return, head, tail, c=0.5):
    assert head < tail, "head %d is greater than or equal to tail" % (head, tail)
    y = np.array([0] * head + [1] * (tail - head))
    y = y / np.sum(y)
    rmmt = np.convolve(log_return, y) / np.sqrt(np.convolve(log_return ** 2, y) + 1e-6) # momentum for unit variance within the same period
    return rmmt[:-tail+1]


def standard_momentum(univ_table, head, tail, c=0.5, naming='simple'):
    '''
    Calculates the standard momentum factor defined as follows:
    smmt_r[t] = sum(lr[head:tail]) / sqrt(sum(lr[head:tail] ** 2))
    head and tail are numbers of time periods
    '''
    name = 'standard_momentum'
    if naming == 'full':
        name += '_%s_%s' % (head, tail)
    univ_table[name] = np.nan
    rmmt_dict = {}
    datelst = np.unique(univ_table['date'])
    
    def _smmt_single_name(table):
        lr = np.diff(np.log(table['price'])) # log return series
        lr = np.insert(lr, 0, 0)
        table.loc[:, name] = _smmt_sn(lr, head, tail, c)
        return table
    
    univ_table = univ_table.groupby('ticker').apply(_smmt_single_name)
    
    for t in datelst:
        table = univ_table.loc[univ_table.date == t, ['date', 'ticker', name]].copy()
        table.dropna(inplace = True)
        if type(t) == str:
            t = datetime.strptime(t, '%Y-%m-%d') #XXX this is a temporary fix
        rmmt_dict[t] = table
    return rmmt_dict
