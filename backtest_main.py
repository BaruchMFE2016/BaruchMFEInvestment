__author__ = 'Derek Qi'
# Doing portfolio backtest and generates output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os
import pickle

from setup.univ_setup import *
from factor_mining.combine_factors import *
from factor_mining.factor_model_regression import *
from factor_mining.factor_preprocessing import *

from factor_mining.stock_ret_est import GenReturn 
from GenPosition import *
from performance_analysis.pa_core import *

from factor_mining.factors.stock_return import *

from factor_mining.Mark0 import *

def backtest_single_period(univ, factor_exp_mat, ret_series, t, silent=True):
	'''
	Do a single period backtest on univ[t]
	t: datetime object that is one of the element in keys of univ
	factor_exp_mat, ret_series: factor exposure and return time series
	'''
	# Set backtest params
	lookback = timedelta(weeks=30)
	dend = t
	dstart = dend - lookback

	# Fit single period factor return
	fr, fr_mse = factor_model_fit(univ, factor_exp_mat, ret_series, dstart, dend)

	fx = factor_exp_mat[dend]
	fx = fx.dropna()
	# Filt the available pool
	# univ_fin = univ[dend]
	# univ_fin = univ_fin.dropna()
	# # Throw away penny stocks
	# univ_fin = filt_byval_single_period(univ_fin, 'price', 10)
	# # Throw away illiquid stocks
	# univ_fin = filt_byval_single_period(univ_fin, 'volume', 1500000)
	# # Throw away things in MA
	# univ_fin = filt_byval_single_period(univ_fin, 'inMA', 0)
	fx = pd.merge(fx, univ_fin[['ticker']], how='inner', on='ticker')

	if 0:
		fx.to_csv('./temp/factor_exposure_' + t.strftime('%Y%m%d') + '.csv', index=False)
		fr.to_csv('./temp/factor_return_' + t.strftime('%Y%m%d') + '.csv', index=False)

	# Calculate position
	stock_list, w_opt = GenPosition(fr, fx, U=0.2) # Todo: modify this to be flexible. 4/30 Derek
	w_opt = PositionFilter(w_opt) # filt away very small number in portfolio
	
	ptfl_full = pd.DataFrame({"ticker": stock_list, "weight": list(w_opt.T[0])})
	ptfl_full = pd.merge(ptfl_full, univ_fin[['ticker', 'log_ret']], how='inner', on='ticker')
	ptfl_full.loc[ptfl_full.log_ret < -2.5, 'log_ret'] = 0 # Emergency process for stocks in MA for over 6 months
	
	pnl_sp = np.dot(ptfl_full.weight, ptfl_full.log_ret)

	if not silent:
		print('Pool size: %d' % univ_fin.shape[0])
		print(ptfl_full[ptfl_full['weight'] != 0])
		print('Period log pnl: %f' % pnl_sp)
	return ptfl_full, pnl_sp, np.mean(fr)

# This is deprecated and will be removed in later versions, use backtest_multi_period_rebalance and set rebal = 1
def backtest_batch(univ, factor_exp_mat, ret_series, dstart, dend, silent=True):
	'''
	Run backtest in batch to portfolio from dstart to dend
	'''
	datelst = sorted(univ.keys())
	tin_lst, ptfl_lst, pnl_lst = [], [], []
	factor_names = factor_exp_mat[datelst[0]].columns[2:].tolist() # exclude date and ticker column
	fr_df = pd.DataFrame(columns = factor_names)
	for ti in range(len(datelst)):
		t = datelst[ti]
		
		if t < dstart or t > dend:
			continue

		if not silent:
			print(t)

		tin_lst.append(t)
		ptfl, pnl_sp, fr_sp = backtest_single_period(univ, factor_exp_mat, ret_series, t, silent)
		ptfl_lst.append(ptfl)
		pnl_lst.append(pnl_sp)
		fr_df = fr_df.append(fr_sp, ignore_index=True)

	fr_df['date'] = tin_lst
	fr_df = fr_df[['date'] + factor_names]
	pnl = pd.DataFrame({'date': tin_lst, 'pnl': pnl_lst})
	return ptfl_lst[-1], pnl, fr_df


def backtest_multi_period_rebalance(univ, factor_exp_mat, ret_series, dstart, dend, rebalance=1, silent=True):
	'''
	Backtest with multi-period rebalancing
	'''
	datelst = sorted(univ.keys())
	tin_lst, ptfl_lst, pnl_lst = [], [], []
	factor_names = factor_exp_mat[datelst[0]].columns[2:].tolist() # exclude date and ticker column
	fr_df = pd.DataFrame(columns = factor_names)
	count = 0
	for ti in range(len(datelst)):
		t = datelst[ti]
		
		if t < dstart or t > dend:
			continue
		
		if not silent:
			print(t)
		
		tin_lst.append(t)
		
		if count == 0:
			# Do rebalance
			ptfl, pnl_sp, fr_sp = backtest_single_period(univ, factor_exp_mat, ret_series, t, silent)
			ptfl_lst.append(ptfl)
			pnl_lst.append(pnl_sp)
			fr_df = fr_df.append(fr_sp, ignore_index=True)
		else:
			# Use prev portfolio
			ptfl = ptfl_lst[-1].copy()
			ptfl = ptfl[['ticker','weight']]
			# Filt the available pool
			# univ_fin = univ[t]
			# univ_fin = univ_fin.dropna()
			# # Throw away penny stocks
			# univ_fin = filt_byval_single_period(univ_fin, 'price', 10)
			# # Throw away illiquid stocks
			# univ_fin = filt_byval_single_period(univ_fin, 'volume', 1500000)
			# # Throw away things in MA
			# univ_fin = filt_byval_single_period(univ_fin, 'inMA', 0)
			
			# Force clear what is not in the pool now and re-normalize the weight
			ptfl = pd.merge(ptfl, univ_fin[['ticker', 'log_ret']], how='inner', on='ticker')
			ptfl.loc[ptfl.log_ret < -2.5, 'log_ret'] = 0 # Emergency process for stocks in MA for over 6 months
			pnl_sp = np.dot(ptfl.weight, ptfl.log_ret)
			
			ptfl_lst.append(ptfl)
			pnl_lst.append(pnl_sp)
			fr_df = fr_df.append(fr_sp, ignore_index=True)
			
			if not silent:
				print('Pool size: %d' % univ_fin.shape[0])
				print(ptfl[ptfl['weight'] != 0])
				print('Period log pnl: %f' % pnl_sp)	
		count -= 1
		count %= rebalance
	pnl = pd.DataFrame({'date': tin_lst, 'pnl': pnl_lst})
	fr_df['date'] = tin_lst
	fr_df = fr_df[['date'] + factor_names]
	return ptfl_lst, pnl, fr_df



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', type=str)
	parser.add_argument('-e', type=str)
	parser.add_argument('-d', type=str)
	args = parser.parse_args()

	start_str = args.s
	ends_str = args.e
	data_dir = args.d


	### universe setup ###
	print('setup R3000 universe')
	if os.path.exists(data_dir + 'univ.pkl'):
		print('use existing binary')
		with open(data_dir + 'univ.pkl', 'rb') as univ_fh:
			univ = pickle.load(univ_fh)
	else:
		print('construct from csv')
		big_table_dir = data_dir + 'big_table_full_v3.csv'
		univ = univ_setup(big_table_dir)
		filt_by_name(univ)
		with open(data_dir + 'univ.pkl', 'wb') as fh:
			pickle.dump(univ, fh)

	### Filt ###
	filt_na(univ)
	filt_byval(univ, 'price', 10)
	filt_byval(univ, 'inMA', 0)
	filt_byval(univ, 'volume', 1500000)

	### model configuration ###
	# define and calculate all factors
	print('==========================')
	print('calculating factors')
	factors = alpha_four_factors(univ)
	# concat into factor exposure matrices
	factor_exp_mat = combine_factors(factors)
	# preprocessing
	print('standardize each factor')
	factor_exp_mat = process_batch(factor_exp_mat, pp.scale)
	# factor_exp_mat = process_batch(factor_exp_mat, standardize)
	print('winsorize each factor with +/- 3 standard deviation')
	factor_exp_mat = process_batch(factor_exp_mat, winsorize_std)

	# const setup
	factor_names = [k for k in factors.keys()]
	N_f = len(factor_names)
	datelst = sorted(factor_exp_mat.keys())
	N_T = len(datelst)

	print('==========================')
	print('running backtest')

	dstart = datetime.strptime(start_str, '%Y-%m-%d')
	dend = datetime.strptime(ends_str, '%Y-%m-%d')

	# Calc stock returns
	# Todo: This might be packed into backtest function. 4/30 Derek
	rebal = 1
	ret_series = log_return(univ, -rebal)

	ptfl_lst, pnl, fr_hist = backtest_multi_period_rebalance(univ, factor_exp_mat, ret_series, dstart, dend, rebalance=rebal, silent=False)

	#output the final portfolio
	now = datetime.now()
	nowstr = now.strftime('%Y%m%d_%H:%M:%S')
	
	outputdir = './output/'
	# GenPortfolioReport(ptfl_fin, report_file=outputdir + 'portfolio_report_long_only'+nowstr+'.csv', pt=True)
	
	
	# Do performance analysis
	print('===========================')
	pmfc = (cagr(pnl), vol(pnl), sharpe_ratio(pnl), max_drawdown(pnl), drawdown_length(pnl))
	print('CAGR:%f \nVolatility:%f\nSharpe_ratio:%f\nmax drawdown: %f\ndrawdown length: %d\n' % pmfc)

	pnl['cumpnl'] = np.cumsum(pnl['pnl'])
	pnl.to_csv('./output/pnl_series_' + nowstr + '.csv', index=False)

	fr_hist.to_csv('./output/fitted_factor_return_' + nowstr + '.csv', index=False)

	plot_save_dir = outputdir
	plot_nav(pnl, savedir=plot_save_dir)
	