__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime
from time import time
import pickle
import os

from .filter.filters import *

def excel_date_trans(s):
    sl = s.split('/')
    m = int(sl[0])
    d = int(sl[1])
    y = int(sl[2])

    dt = datetime(y, m, d)
    return dt


def univ_setup_from_table(big_table_dir, fund_table_dir=None):
    '''
    Read the big_table format cleaned data and set universe base on that
    return type: dictionary, keys are dates in datetime.datetime format
    values are corresponding security information
    '''
    big_table = pd.read_csv(big_table_dir)
    big_table['date'] = pd.to_datetime(big_table['date'])
    if fund_table_dir:
        fund_table = pd.read_csv(fund_table_dir)
        fund_table['date'] = pd.to_datetime(fund_table['date'])
        fund_table['annc_date'] = pd.to_datetime(fund_table['annc_date'])
    
    try:
        big_table.drop('Unnamed: 0', axis=1, inplace=True)
        big_table.drop('gross_profit', axis=1, inplace=True)
        big_table.eps.fillna(0, inplace=True)
        big_table.div_ratio.fillna(0, inplace=True)
        
    except:
        pass

    if fund_table_dir:
        ix = np.isnat(fund_table.annc_date.values)
        fund_table.loc[ix, 'annc_date'] = fund_table.loc[ix, 'date'] # fillna of annc_date
        fund_table.drop('date', axis=1, inplace=True)
        fund_table.fillna(0, inplace=True)
        fund_table = fund_table.sort_values('annc_date')
        big_table = pd.merge_asof(big_table, fund_table, left_on='date', right_on='annc_date', by='ticker', tolerance=pd.Timedelta('182d'))

    datecol = big_table.date.unique()

    N_T = datecol.shape[0]
    subtable = [0] * N_T

    for ti in range(N_T):
        t = datecol[ti]
        subtable[ti] = big_table.loc[big_table.date == t,:]

    return dict(zip(datecol, subtable))


def univ_setup(datadir, version=4, wFund=True):
    print('Setup R3000 universe')
    
    datadir = '/home/derek-qi/Documents/R3000_Data/data/r3000/'
    if os.path.exists(datadir + 'univ_v%d.pkl' % version):
        print('use existing binary file')
        with open(datadir + 'univ_v%d.pkl' % version, 'rb') as univ_fh:
            univ = pickle.load(univ_fh)
    
    else:
        print('construct from csv')
        big_table_dir = datadir + 'big_table_full_v%d.csv' % version
        if wFund:
            fund_table_dir = datadir + 'fund_data_v%d.csv' % version
        else:
            fund_table_dir = None
        univ = univ_setup_from_table(big_table_dir, fund_table_dir)
        with open(datadir + 'univ_v%d.pkl' % version,'wb') as fh:
            pickle.dump(univ, fh)
    
    return univ


if __name__ == '__main__':
    big_table_dir = '/home/derek-qi/Documents/R3000_Data/data/r3000/big_table_fullv4.csv'
    r3000_univ = univ_setup(big_table_dir)

    filt_na(r3000_univ)

    filt_byval(r3000_univ, 'price', 10)
    filt_byval(r3000_univ, 'volume', 300000 * 5)