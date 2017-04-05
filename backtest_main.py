__author__ = 'Derek Qi'
# Doing portfolio backtest and generates output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

from setup.univ_setup import *
from factor_mining.combine_factors import *
from factor_mining.factor_model_regression import *

from factor_model.stock_ret_est import GenReturn 
from GenPosition import *
from performance_analysis.pa_core import *

from factor_mining.factors.momentum import *

from factor_mining.Mark0 import *


def backtest_single_period(univ, factor_exp_mat, ret_series, t, silent=True):
	'''
	Do a single period backtest on univ[t]
	t: datetime object that is one of the element in keys of univ
	factor_exp_mat, ret_series: factor exposure and return time series
	'''
	# Set backtest params
	lookback = timedelta(weeks=104)
	dend = t
	dstart = dend - lookback

	# Calc stock return
	ret_series = momentum(univ, 0, 1)

	# Fit single period factor return
	fr, fr_mse = factor_model_fit(factor_exp_mat, ret_series, dstart, dend)

	fx = factor_exp_mat[dend]
	fx = fx.dropna()
	# Filt the available pool
	univ_fin = univ[dend]
	univ_fin = univ_fin.dropna()
	# Throw away penny stocks
	univ_fin = filt_byval_single_period(univ_fin, 'price', 10)
	# Throw away illiquid stocks
	univ_fin = filt_byval_single_period(univ_fin, 'volume', 1500000)
	# Throw away things in MA
	univ_fin = filt_byval_single_period(univ_fin, 'inMA', 0)
	fx = pd.merge(fx, univ_fin[['ticker']], how='inner', on='ticker')

	# Calculate position
	stock_list, w_opt = GenPosition(fr, fx, U=0.2)
	w_opt = PositionFilter(w_opt) # filt away very small number in portfolio
	ptfl_full = pd.DataFrame({"ticker": stock_list, "weight": list(w_opt.T[0])})
	ptfl_full = pd.merge(ptfl_full, univ_fin[['ticker', 'log_ret']], how='inner', on='ticker')
	pnl_sp = np.dot(ptfl_full.weight, ptfl_full.log_ret)

	if not silent:
		print('Pool size: %d' % univ_fin.shape[0])
		print(ptfl_full[ptfl_full['weight'] != 0])
		print('Period log pnl: %f' % pnl_sp)
	return ptfl_full, pnl_sp


def backtest_batch(univ, factor_exp_mat, ret_series, dstart, dend, silent=True):
	'''
	Run backtest in batch to portfolio from dstart to dend
	'''
	datelst = sorted(univ.keys())
	tin_lst, ptfl_lst, pnl_lst = [], [], []
	for ti in range(len(datelst)):
		t = datelst[ti]
		
		if t < dstart or t > dend:
			continue

		if not silent:
			print(t)

		tin_lst.append(t)
		ptfl, pnl_sp = backtest_single_period(univ, factor_exp_mat, ret_series, t, silent)
		ptfl_lst.append(ptfl)
		pnl_lst.append(pnl_sp)


	pnl = pd.DataFrame({'date': tin_lst, 'pnl': pnl_lst})
	return ptfl_lst[-1], pnl


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', type=str)
	parser.add_argument('-e', type=str)
	args = parser.parse_args()

	start_str = args.s
	ends_str = args.e


	### universe setup ###
	print('universe setup')
	big_table_dir = '/home/derek-qi/Documents/R3000_Data/data/r3000/big_table_full_v2.csv'
	univ = univ_setup(big_table_dir)
	filt_by_name(univ)

	### model configuration ###
	# define and calculate all factors
	print('calculating factors')
	factors = alpha_four_factors(univ)
	# concat into factor exposure matrices
	factor_exp_mat = combine_factors(factors)

	# const setup
	factor_names = [k for k in factors.keys()]
	N_f = len(factor_names)
	datelst = sorted(factor_exp_mat.keys())
	N_T = len(datelst)
	
	# Calc stock returns
	ret_series = momentum(univ, 0, 1)

	print('running backtest')
	# dstart = datetime(2014, 1, 1)
	# dend = datetime(2014, 1, 31)
	dstart = datetime.strptime(start_str, '%Y-%m-%d')
	dend = datetime.strptime(ends_str, '%Y-%m-%d')
	ptfl_fin, pnl = backtest_batch(univ, factor_exp_mat, ret_series, dstart, dend, silent=False)

	# plt.plot(pnl['pnl'])
	# plt.show()

	#output the final portfolio
	now = datetime.now()
	nowstr = now.strftime('%Y%m%d_%H:%M:%S')
	GenPortfolioReport(ptfl_fin, report_file=outputdir + 'portfolio_report_long_only'+nowstr+'.csv', pt=True)
	pnl.to_csv('./output/pnl_series' + nowstr + '.csv')


	# Do performance analysis
	pnl.columns = ['Date', 'Pnl']
	pmfc = (cagr(pnl), vol(pnl), sharpe_ratio(pnl), max_drawdown(pnl), drawdown_length(pnl))
	print('CAGR:%f \nVolatility:%f\nSharpe_ratio:%f\nmax drawdown: %f\ndrawdown length: %f\n' % pmfc)



	# ### model fitting ###
	# # Single period fit and build portfolio
	# lookback = timedelta(weeks=104)
	# dend = datetime(2017, 2, 10)
	# dstart = dend - lookback
	# # fit the factor return and mse
	# fr, fr_mse = factor_model_fit(factor_exp_mat, ret_series, dstart, dend)

	# ### current period pool selection and stock return estimation ###
	
	# ### portfolio optimization ###
	# fx = factor_exp_mat[dend]
	# fx = fx.dropna()
	# stock_list, w_opt = GenPosition(fr, fx)
	# ptfl_full = pd.DataFrame({"Ticker": stock_list, "Weight": list(w_opt.T[0])})
	# now = datetime.now()
	# nowstr = now.strftime('%Y%m%d_%H:%M:%S')
	# GenPortfolioReport(ptfl_full, report_file=outputdir + 'portfolio_report_long_only'+nowstr+'.csv', pt=True)
	# ptfl_full.to_csv(outputdir + 'portfolio_long_only.csv', index=False)
    

	# ### generate pnl (single period) ###