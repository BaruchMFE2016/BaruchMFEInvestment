__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


def _mmt(log_return, head, tail):
    assert head < tail, "head %d is greater than or equal to tail" % (head, tail)
    y = np.array([0] * head + [1] * (tail - head))
    y = y / np.sum(y)
    rmmt = np.convolve(log_return, y)
    return rmmt[:-tail+1]


def momentum(univ_table, head, tail, naming='simple', **kwargs):
    '''
    mmt_t = \sum_{i=head}^{tail} log return_{t-i}
    '''
    name = 'momentum'
    if naming == 'full':
        name += '_%s_%s' % (head, tail)
    univ_table[name] = np.nan
    mmt_dict = {}
    datelst = np.unique(univ_table['date'])

    def _mmt_single_name(table):
        lr = np.diff(np.log(table['price'])) # log return series
        lr = np.insert(lr, 0, 0)
        mmt = _mmt(lr, head, tail)
        table.loc[:, name] = mmt
        if kwargs.get('weight') == 'ewma':
            halflife = kwargs.get('halflife') or 13
            table[name] = pd.ewma(table[name], halflife=halflife, ignore_na=True)       
        return table
    
    univ_table = univ_table.groupby('ticker').apply(_mmt_single_name)

    for t in datelst:
        table = univ_table.loc[univ_table.date == t, ['date', 'ticker', name]].copy()
        table.dropna(inplace = True)
        if type(t) == str:
            t = datetime.strptime(t, '%Y-%m-%d') #XXX this is a temporary fix
        mmt_dict[t] = table
    return mmt_dict


def momentum_ewma(univ_table, head, tail, halflife=13, naming='simple', **kwargs):
    '''
    mmt_ewma_t = \sum_{i=head}^{tail} \lambda^{i-head} log return_{t-i}
    '''
    name = 'momentum_ewma'
    if naming == 'full':
        name += '_%s_%s' % (head, tail)
    univ_table[name] = np.nan
    mmt_dict = {}
    datelst = np.unique(univ_table['date'])

    def _mmt_single_name(table):
        lr = np.diff(np.log(table['price'])) # log return series
        lr = np.insert(lr, 0, 0)
        table['log_return'] = lr
        halflife = kwargs.get('halflife') or 13
        table[name] = pd.ewma(table['log_return'], halflife=halflife, ignore_na=True)
        return table
    
    univ_table = univ_table.groupby('ticker').apply(_mmt_single_name)

    for t in datelst:
        table = univ_table.loc[univ_table.date == t, ['date', 'ticker', name]].copy()
        table.dropna(inplace = True)
        if type(t) == str:
            t = datetime.strptime(t, '%Y-%m-%d') #XXX this is a temporary fix
        mmt_dict[t] = table
    return mmt_dict