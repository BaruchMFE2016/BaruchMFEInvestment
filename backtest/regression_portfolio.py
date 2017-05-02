__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from 


def factor_model_regression_single_period(univ, factor_exp_mat, ret_series, t, silent=True):
	'''
	Do a single period backtest on univ[t]
	t: datetime object that is one of the element in keys of univ
	factor_exp_mat, ret_series: factor exposure and return time series
	'''
	# Set backtest params
	lookback = timedelta(weeks=30)
	dend = t
	dstart = dend - lookback
	ret_name = 'f_log_ret_1'

	# Fit single period factor return
	fr, fr_mse = factor_model_fit(factor_exp_mat, ret_series, dstart, dend)

	fx = factor_exp_mat[dend]
	fx = fx.dropna()
	# Filt the available pool
	univ_fin = univ[dend]
	fx = pd.merge(fx, univ_fin[['ticker']], how='inner', on='ticker')

	# Calculate position
	stock_list, w_opt = GenPosition(fr, fx, U=0.2)
	w_opt = PositionFilter(w_opt) # filt away very small number in portfolio
	ptfl_full = pd.DataFrame({"ticker": stock_list, "weight": list(w_opt.T[0])})
	ptfl_full = pd.merge(ptfl_full, univ_fin[['ticker', ret_name]], how='inner', on='ticker')
	ptfl_full.loc[ptfl_full.f_log_ret_1 < -2.5, ret_name] = 0 # Emergency process for stocks in MA for over 6 months
	pnl_sp = np.dot(ptfl_full.weight, ptfl_full[ret_name])

	if not silent:
		print('Pool size: %d' % univ_fin.shape[0])
		print(ptfl_full[ptfl_full['weight'] != 0])
		print('Period log pnl: %f' % pnl_sp)
	return ptfl_full, pnl_sp, np.mean(fr)
