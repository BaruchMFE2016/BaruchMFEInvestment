__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def diff(univ_table, name_ori, lag=1):
    univ_table = univ_table.copy()
    assert name_ori in univ_table.columns, 'Column does not exist %s' % name_ori
    name = 'd%d_' % lag + name_ori
    factor_dict = {}
    datelst = np.unique(univ_table['date'])
    
    def _diff(table):
        table[name] = table[name_ori].diff(lag)
        table.fillna(0, inplace=True)
        return table
    
    univ_table = univ_table.groupby('ticker').apply(_diff)
    
    for t in datelst:
        table = univ_table.loc[univ_table.date == t, ['date', 'ticker', name]].copy()
        table.dropna(inplace = True)
        if type(t) == str:
            t = datetime.strptime(t, '%Y-%m-%d')
        factor_dict[t] = table
    return factor_dict