__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


### inplace filtering for the entire universe

def filt_na(univ):
    '''
    filter out na and infinity values in a universe
    '''
    datelst = sorted(univ.keys())
    N_T = len(datelst)

    for ti in range(N_T):
        t = datelst[ti]
        univ_ti = univ[t]
        univ_ti = univ_ti.replace([np.inf, -np.inf], np.nan)
        univ_ti.dropna(inplace=True, how='any')
        if univ_ti.empty:
            del univ[t]
        else:
            univ[t] = univ_ti


def filt_byval(univ, varname, thrd, keep='above'):
    '''
    filter out any element under varname that has value thrd or below
    '''
    datelst = sorted(univ.keys())
    N_T = len(datelst)

    for ti in range(N_T):
        t = datelst[ti]
        univ_ti = univ[t]
        idx_in = univ_ti[varname] > thrd
        if keep == 'below':
            idx_in = ~idx_in
        univ[t] = univ_ti.ix[idx_in,:]


def filt_byval_single_period(univ_sp, varname, thrd, keep='above'):
    '''
    filter out any element under varname that has value thrd or below for a single period
    snapshot of the universe
    '''
    univ_sp = univ_sp.copy()
    idx_in = univ_sp[varname] > thrd
    if keep == 'below':
        idx_in = 1 - ind_in
    univ_sp = univ_sp.ix[idx_in,:]
    return univ_sp


def filt_by_name(univ):
    '''
    Filt out tickers with special characters and numbers
    '''
    datelst = sorted(univ.keys())
    N_T = len(datelst)
    for ti in range(N_T):
        t = datelst[ti]
        univ_ti = univ[t]
        idx_in = [not '.' in row[1].ticker for row in univ_ti.iterrows()] # iterrows gives a tuple (index, row)
        univ[t] = univ_ti.ix[idx_in,:]
        
