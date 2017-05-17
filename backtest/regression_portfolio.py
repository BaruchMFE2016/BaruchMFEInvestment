__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from factor_mining.robust import check_nan_friendly_finite_array

from BackTestSinglePeriod import BackTestSinglePeriod
from factor_mining.factor_model_regression import *
from GenPosition import *


LR = LinearRegression()

class RegressionPtflSpcalc(BackTestSinglePeriod):
	def __init__(self, all_factor_names, fitter=LR, weighting=None):
		self.all_factor_names = all_factor_names
		self.all_factor_returns = {}
		self.weighting = weighting

	def get_config(self):
		config = {}
		config['Signal variable'] = self.signal
		config['Selection range'] = self.sel_range
		config['Weighting'] = self.weighting
		return config

	def get_func_name(self):
		return 'Percentile Portfolio'

	def calc_pnl(self, univ, factor_exp_mat, ret_series, t, **kwargs):
		'''
		Do a single period backtest on univ[t]
		t: datetime object that is one of the element in keys of univ
		factor_exp_mat, ret_series: factor exposure and return time series
		'''
		n_lookback = 30 if not 'lookback' in kwargs.keys() else lookback = kwargs['lookback']
		datelst = [t - timedelta(weeks=i) for i in range(n_lookback)]
		datelst = datelst[::-1]
		ret_name = 'f_log_ret_1'

		for dt in datelst:
			if not dt in self.all_factor_returns.keys():
				u_sp = univ[dt].copy()
				factor_exp_sp = factor_exp_mat[dt].copy()
				ret_sp = ret_series[dt].copy()

				ret_name = ret_sp.columns[-1]
				merged = pd.merge(factor_exp_sp, ret_sp, how='inner', on='ticker')
				merged = merged.replace([np.inf, -np.inf], np.nan)
				merged = merged.dropna()
				X = np.array(merged[self.factor_names])
				r = np.array(merged[return_name])

				if self.weighting is None:
					self.fitter.fit(X, r)
				else:
					u_sel = pd.merge(u_sp, merged[['ticker']], how='right', on='ticker')
					w = np.asarray(u_sel[self.weighting])
					self.fitter.fit(X, r, sample_weight=w)

				# r_pred = self.fitter.predict(X)
				mse = mean_squared_error(r, r_pred)
				fr = self.fitter.coef_
				self.all_factor_returns[dt] = {'factor_returns': fr, 'mse': mse}

			else:
				pass
				# fr, mse = self.all_factor_returns[dt]['factor_returns'], self.all_factor_returns[dt]['mse']

		fr_all_period_avg = np.mean([self.all_factor_returns[dt]['factor_return'] for dt in datelst], axis=0)

		######
		fr. fr_mse = factor_model_fit(univ_sp, factor_exp_mat, ret_series, dstart, dend)
		fx = factor_exp_mat[dend].copy()
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

		return ptfl_sp, pnl_sp	


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
