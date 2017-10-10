__author__ = 'Derek Qi'

'''
Momentum Gap idea from
Huang, S. (2015). The momentum gap and return predictability.
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def momentum_gap(univ_table, head, tail, q1=75, q2=25, naming='simple'):
    '''
    Momentum gap is defined as:
    q1 quantile - q2 quantile of the return series.
    '''
    assert q1 > q2, 'higher quantile %d should be larger than lower quantile %d' % (q1, q2)
    name = 'momentum_gap'
    retname = 'f_log_ret_1'
    if naming == 'full':
        name += 'time_%d_%d_range_%d_%d' % (head, tail, q1, q2)
    univ_table[name] = np.nan

    def _mmt_gap_single_name(table):
        window = tail - head
        table['log_ret'] = np.ediff1d(np.log(table['price']), to_begin=0)
        table['high'] = table['log_ret'].rolling(window).quantile(q1)
        table['low'] = table['log_ret'].rolling(window).quantile(q2)
        table[name] = table['high'].values - table['low'].values
        table[name] = table[name].shift(head)
        # table.drop('high', inplace=True)
        # table.drop('low', inplace=True)
        return table

    univ_table = univ_table.groupby('ticker').apply(_mmt_gap_single_name)

    mmt_gap_dict = {}
    datelst = np.unique(univ_table['date'])
    for t in datelst:
        table = univ_table.loc[univ_table.date == t, ['date', 'ticker', name]].copy()
        table.dropna(inplace = True)
        if type(t) == str:
            t = datetime.strptime(t, '%Y-%m-%d') #XXX this is a temporary fix
        mmt_gap_dict[t] = table
    return mmt_gap_dict