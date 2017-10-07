__author__ = 'Derek Qi'

'''
Momentum Gap idea from
Huang, S. (2015). The momentum gap and return predictability.
'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _mmt_gap(ret_series, q1, q2):
    if ret_series.empty:
        return np.nan

    assert q1 > q2
    return np.percentile(ret_series, q1) - np.percentile(ret_series, q2)


def momentum_gap(univ_table, head, tail, q1=75, q2=25, naming='simple'):
    '''
    Momentum gap is defined as:
    q1 quantile - q2 quantile of the return series.
    '''
    name = 'momentum_gap'
    retname = 'f_log_ret_1'
    if naming == 'full':
        name += 'time_%d_%d_range_%d_%d' % (head, tail, q1, q2)
    univ_table[name] = np.nan

    for idx, row in univ_table.iterrows():
        t, ticker = row.date, row.ticker
        ret_series = univ_table.loc[(univ_table.date >= t-timedelta(weeks=tail+1)) 
                        & (data.date <= t-timedelta(weeks=head+1)), retname]
        row[name] = _mmt_gap(ret_series, q1, q2)

    mmt_gap_dict = {}
    datelst = np.unique(univ_table['date'])
    for t in datelst:
        table = univ_table.xs(t, level='date')[[name]].copy()
        table = table.reset_index()
        table.dropna(inplace = True)
        mmt_gap_dict[t] = table
    return mmt_gap_dict