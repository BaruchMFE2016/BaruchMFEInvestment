#!/home/quan/anaconda3/bin/python

import sys
from optparse import OptionParser

import csv
import math
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from datetime import datetime
from datetime import timedelta

from momentum import get_momentum
from zscore import gaussianize, mean_rms_zscore, median_mad_zscore

np.set_printoptions(precision=6)

DATA_DIRECTORY = "../data"
WEEKDAYS = 7
TIMESTEP = 1
VOLUME = "volume"
STOCKS = "stocks"
ZSCORE = { 'gauss': gaussianize, 
           'mean_rms': mean_rms_zscore, 
           'median_mad': median_mad_zscore }
   
def usage():
    print("bcf.py -u <universe>")
    print("       -p <price>")
    print("       -m <start-end>")
    print("       -f <factor>")
    print("       -d <dummy>")
    print("       -v <volume>")
    print("       -z <z-score>")
    print("       -r YYYYMMDD-YYYYMMDD")
    print("where: -u, --universe ")
    print("       -p, --price ")
    print("       -m, --momentum, head-tail ")
    print("       -f, --factor ")
    print("       -d, --dummy, ")
    print("       -v, --volume, ")
    print("       -z, --zscore, ")
    print("       -r, --range, YYYYMMDD-YYYYMMDD")
    sys.exit(1)

def get_commandline():

    if len(sys.argv) < 3: usage()

    parser = OptionParser()
    parser.add_option("-u", "--universe")
    parser.add_option("-p", "--price")
    parser.add_option("-m", "--momentum")
    parser.add_option("-f", "--factor", action="append", default=[])
    parser.add_option("-d", "--dummy", action="append", default=[])
    parser.add_option("-v", "--volume")
    parser.add_option("-z", "--zscore")
    parser.add_option("-r", "--range")

    options, args = parser.parse_args()
    assert(options.universe is not None)
    assert(options.price is not None)
    assert(options.factor is not None)

    config = vars(options)
    return config

if __name__ == "__main__":

    ### Command line input ###
    commandline = get_commandline()
    universe = commandline["universe"]
    prices = commandline["price"]
    momentum_head_tail = commandline["momentum"]
    factors = commandline["factor"]
    min_volume = commandline["volume"]
    dummys = commandline["dummy"]
    min_volume = float(min_volume) if min_volume is not None else 0

    ### z-score calculation routine ###
    zscore_method = commandline["zscore"]
    zscore = None
    if zscore_method is None:
        z_score = gaussianize
    else:
        z_score = ZSCORE[zscore_method]
   
    ### Date range ###
    date_range = commandline["range"]
    if date_range is None:
        pass
    else:
        start_end_date_pair = date_range.split("-")
        start_date = datetime.strptime(start_end_date_pair[0], '%Y%m%d')
        end_date = datetime.strptime(start_end_date_pair[1], '%Y%m%d')

    ### Stock list (removing duplicates) ###
    stock_list = None
    stocks_file = DATA_DIRECTORY + "/" + universe + "/" + universe + "_" + STOCKS + ".csv"
    with open(stocks_file, 'r') as f:
        reader = csv.reader(f)
        stock_list = list(reader)
        stock_list = list(itertools.chain(*stock_list))
        stock_set = set(stock_list)
        stock_list = []
        for stock in sorted(stock_set): stock_list.append(stock)

    ### Price dataframe ###
    price_file = DATA_DIRECTORY + "/" + universe + "/" + universe + "_" + prices + ".csv"
    price_df = pd.read_csv(price_file, index_col='Date', parse_dates=['Date'])
    price_df = price_df[stock_list]
    
    ### Return dataframe - calculated based on price dataframe ###
    ### Align the next week's return with the current week's factors. ###
    ### This is why the head is -1 (next week according to our convention) ###
    return_df = get_momentum(price_df, -1, 0)
    
    ### factor dataframe list ###
    factor_list = []
    factor_df_list = []

    ### Momentum dataframe - calculated based on price dataframe ###
    momentum_df = None
    if momentum_head_tail is not None:
        momentum_head_tail_pair = momentum_head_tail.split("-")
        momentum_head = int(momentum_head_tail_pair[0])
        momentum_tail = int(momentum_head_tail_pair[1])
        momentum_df = get_momentum(price_df, momentum_head, momentum_tail)
        ### restricting to selected date range ###
        momentum_df = momentum_df[start_date:end_date]
        factor_list.append("momentum")
        factor_df_list.append(momentum_df)
    
    ### restricting to selected date range ###
    price_df = price_df[start_date:end_date]
    return_df = return_df[start_date:end_date]

    ### Volume dataframe ###
    volume_file = DATA_DIRECTORY + "/" + universe + "/" + universe + "_" + VOLUME + ".csv"
    volume_df = pd.read_csv(volume_file, index_col='Date', parse_dates=['Date'])
    volume_df = volume_df[start_date:end_date]
    volume_df = volume_df[stock_list]

    ### Factor exposure dataframes ###
    for factor in factors:
        factor_file = DATA_DIRECTORY + "/" + universe + "/" + universe + "_" + factor + ".csv"
        factor_df = pd.read_csv(factor_file, index_col='Date', parse_dates=['Date'])
        factor_df = factor_df[stock_list]
        factor_df.fillna(method='pad', inplace=True, axis='columns')
        factor_df = factor_df[start_date:end_date]
        factor_list.append(factor)
        factor_df_list.append(factor_df)
        
    ### Dummy variable dataframes ###
    ### TODO ###
    for dummy in dummys:
        dummy_file = DATA_DIRECTORY + "/" + universe + "/" + universe + "_" + dummy + ".csv"
        print(dummy_file)

    ### Loop over date range ###
    date_list = []
    return_list = []
    d = start_date
    delta = timedelta(days=WEEKDAYS)
    while d <= end_date:
        date_list.append(d)

        ### return and volume ###
        return_cs = return_df.ix[d]
        volume_cs = volume_df.ix[d]

        ### use return and volume to filter stocks ###
        reduced_list = []
        for stock in stock_list:
            volume = volume_cs[stock]
            ret = return_cs[stock]
            if not np.isnan(ret) and not np.isnan(volume) and volume > min_volume:
                reduced_list.append(stock)

        r = return_cs[reduced_list].values
        r = r.T

        ### factor exposure matrix, at the current step ###
        X = np.empty((0,len(reduced_list)))
        for factor, factor_df in zip(factor_list, factor_df_list):
            factor_cs = factor_df.ix[d]
            factor_cs = factor_cs[reduced_list].values
            if 'beta' not in factor: 
                factor_cs = z_score(factor_cs)
            X = np.append(X, [factor_cs], 0)
        X = X.T
        
        #X = sm.add_constant(X)
        ### regression ###
        model = sm.OLS(r, X)
        results = model.fit()
        coeffs = results.params
        pvalues = results.pvalues
        stderrs = results.bse

        #print(coeffs)
        return_list.append(coeffs)

        np.savetxt('factor_exposure_matrix.csv', X, delimiter=",")

        d += delta

    ### weekly returns ###
    pct_returns = []
    for coeffs in return_list:
        pct_return = []
        for coeff in coeffs:
            pct_return.append(coeff)
        pct_returns.append(pct_return)
    
    #for date, factor_returns in zip(date_list, pct_returns):
    #    print(date.date(), end=" ")
    #    for factor_return in factor_returns:
    #        print(", {:1.6f}".format(factor_return), end=" ")
    #    print("")

    num_of_factors = len(factor_list)
    cum_pct_returns = []
    cum_pct_return = [ 1 for i in range(num_of_factors)]

    for factor_returns in pct_returns:
        for i in range(num_of_factors):
            cum_pct_return[i] *= (1+factor_returns[i])
        cum_pct_returns.append(list(cum_pct_return))

    #for date, factor_returns in zip(date_list, cum_pct_returns):
    #    print(date, end=" ")
    #    for factor_return in factor_returns:
    #        print(", {:1.6f}".format(factor_return), end=" ")
    #    print("")




