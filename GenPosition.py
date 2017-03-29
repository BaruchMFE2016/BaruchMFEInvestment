__author__ = 'Derek Qi'

# Generates the optimized position for a single period

import numpy as np
import pandas as pd
from datetime import datetime
from time import time
from yahoo_finance import Share

from factor_model import alpha
from optimization_old import optimizer as opt


def GenPosition(factor_return_file, factor_exposure_file, stock_list_file, hasshort=False):
    """
    :return: stock list and the corresponding weight in optimized portfolio
    """
    ret, sigma, stock_list = alpha.GenReturn(factor_return_file, factor_exposure_file, stock_list_file)
    N_INS = ret.shape[0]
    ret = np.reshape(ret, (N_INS, 1))
    w_old = np.ones([N_INS, 1]) / N_INS # Start from an evenly-split portfolio and assign no position-changing limits

    if hasshort:
        w_opt = opt.optimizerlongshort(w_old, alpha=ret, sigma=sigma, L=-1, U=1)
    else:
        w_opt = opt.optimizer(w_old, alpha=ret, sigma=sigma, L=-1, U=1)
    return stock_list, w_opt


def PositionFilter(w, tol=1e-4):
    """
    Filter out very small positions (in absolute values) and re-normalize
    :return: position vector filtered
    """
    w[abs(w) < tol] = 0
    w /= sum(w)
    return w


def GenPortfolioReport(ptfl_full, report_file, pt=False):
    # Generates the portfolio report, fetch data from Yahoo finance
    ptfl_sel = ptfl_full[ptfl_full['Weight'] != 0]

    ticker_list = list(ptfl_sel['Ticker'])

    row_dict_list = []
    # get Name, prev_close, market_cap, PE, avg_daily_volume, from Yahoo finance
    for ticker in ticker_list:
        stock = Share(ticker)
        stock.refresh()

        name = stock.get_name()
        prev_close = stock.get_prev_close()
        avg_daily_volume = stock.get_avg_daily_volume()
        market_cap = stock.get_market_cap()
        pe_ratio = stock.get_price_earnings_ratio()
        # last_tradetime = stock.get_trade_datetime()
        last_tradetime = 0
        row_dict_list.append({'Ticker':ticker, 'Name':name, 'Prev Close':prev_close, 'Avg Daily Volume':avg_daily_volume, 'Market Cap':market_cap, 'PE ratio':pe_ratio, 'Last Tradetime':last_tradetime})

    report = pd.DataFrame(row_dict_list)
    report = pd.merge(ptfl_sel, report, on='Ticker')
    report = report[['Ticker', 'Weight', 'Name', 'Prev Close', 'Avg Daily Volume', 'Market Cap', 'PE ratio']]
    if pt:
        print(report)
    report.to_csv(report_file)


if __name__ == "__main__":
    start = time()
    datadir = './data/factor_ret_exp/'
    outputdir = './output/'
    factor_return_file, factor_exposure_file, stock_list_file = datadir + 'factor_return_20101231_20131227.csv', datadir + 'factor_exposure_matrix_20101231_20131227.csv', datadir + 'stock_list_20140103.csv'
    # Long-only position
    stock_list, w_opt = GenPosition(factor_return_file, factor_exposure_file, stock_list_file, hasshort=False)
    w_opt = PositionFilter(w_opt)
    ptfl_full = pd.DataFrame({"Ticker": stock_list, "Weight": list(w_opt.T[0])})
    now = datetime.now()
    nowstr = now.strftime('%Y%m%d_%H:%M:%S')
    GenPortfolioReport(ptfl_full, report_file=outputdir + 'portfolio_report_long_only'+nowstr+'.csv', pt=True)
    # result.to_csv(outputdir + 'portfolio_test_long_only.csv', index=False)
    pause = time()
    print(pause - start)
    # Long-Short position
    stock_list, w_opt = GenPosition(factor_return_file, factor_exposure_file, stock_list_file, hasshort=True)
    w_opt = PositionFilter(w_opt)
    ptfl_full_ls = pd.DataFrame({"Ticker": stock_list, "Weight": list(w_opt.T[0])})
    GenPortfolioReport(ptfl_full_ls, report_file=outputdir + 'portfolio_report_long_short'+nowstr+'.csv', pt=True)
    # result.to_csv(outputdir + 'portfolio_test_long_short.csv', index=False)
    end = time()
    print(end - pause)
