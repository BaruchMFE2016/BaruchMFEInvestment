__author__ = 'Derek Qi'

import numpy as np
import pandas as pd

from .BackTestSinglePeriod import BackTestSinglePeriod
from factor_mining.robust import check_nan_friendly_finite_array


class PercentilePtflSpcalc(BackTestSinglePeriod):
	def __init__(self, signal, sel_range, weighting='equal'):
		self.signal = signal
		self.sel_range = sel_range
		self.weighting = weighting

	def get_config(self):
		config = {}
		config['Signal variable'] = self.signal
		config['Selection range'] = self.sel_range
		config['Weighting'] = self.weighting
		return config

	def get_func_name(self):
		return 'Percentile Portfolio'

	def calc_pnl(self, univ, factor_exp_mat, t, **kwargs):
		ret_name = 'f_log_ret_1'
		univ_sp, factor_exp_mat_sp = univ[t].copy(), factor_exp_mat[t].copy()
		if 'demean' in kwargs:
			if kwargs['demean'] == 'industry':
				demean = kwargs['demean']
				univ_sp['f_log_ret_1_demean'] = univ_sp[ret_name] - univ_sp.groupby(demean)[ret_name].transform('mean')           
				ret_name = 'f_log_ret_1_demean'

		pct_low, pct_high = np.min(self.sel_range), np.max(self.sel_range)
		signal_var = np.asarray(factor_exp_mat_sp[self.signal])
		cutoff_low, cutoff_high = np.percentile(signal_var[~np.isnan(signal_var)], [pct_low, pct_high])
		ix_in = (signal_var >= cutoff_low) * (signal_var <= cutoff_high)
		ticker_in = factor_exp_mat_sp['ticker'][ix_in]

		ptfl = univ_sp.loc[univ_sp['ticker'].isin(ticker_in), :]
		if self.weighting == 'market_cap':
			ptfl['weight'] = ptfl['market_cap']
		elif self.weighting == 'equal':
			ptfl['weight'] = [1] * len(ptfl.index)
		else:
			raise('unknown weighting method', self.weighting)

		ptfl['weight'] = ptfl['weight'] / np.nansum(ptfl['weight']) # normalize to 1
		pnl_sp = np.dot(ptfl['weight'], ptfl[ret_name])
		ptfl_sp = ptfl[['date', 'ticker', 'weight']]
		return ptfl_sp, pnl_sp