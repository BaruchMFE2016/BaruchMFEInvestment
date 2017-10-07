__author__ = 'Derek Qi'

# Some utility functions to work with the data structure of universe and portfolio

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def get_key(univ, n):
    ks = sorted(univ.keys())
    return ks[n]


def get_val(univ, n):
    return univ[get_key(univ, n)].copy()


def get_col(univ, n=0):
    v = get_val(univ, n)
    return v.columns


def get_count_sp(univ, n):
    return get_val(univ, n).shape[0]


def get_count(univ, rtype='df'):
    ks = sorted(univ.keys())
    all_count = [get_count_sp(univ, i) for i in range(len(ks))]
    if rtype == 'dict':
        return dict(zip(ks, all_count))
    elif rtype == 'df':
        return pd.DataFrame({'date': ks, 'count': all_count})


def merge(left, right, on='ticker', how='inner', **kwargs):
    k_left = sorted(left.keys())
    k_right = sorted(right.keys())
    assert (k_left > k_right) - (k_left < k_right) == 0, 'Dict keys do not have 1-1 match'

    left_cols, right_cols = kwargs.get('left_cols'), kwargs.get('right_cols')

    res_lst = []
    for k in k_left:
        v_left, v_right = left[k], right[k]
        if left_cols:
            v_left = v_left[left_cols + [on]]
        if right_cols:
            v_right = v_right[[on] + right_cols]
        res = pd.merge(v_left, v_right, on=on, how=how)
        res_lst.append(res)

    return dict(zip(k_left, res_lst))


def stack(univ, **kwargs):
    ''' stack the dataframes saved in a dict '''
    columns = get_val(univ, 0).columns
    result = pd.concat(univ.values()).sort_values(['date', 'ticker'])
    return result[columns].copy()
