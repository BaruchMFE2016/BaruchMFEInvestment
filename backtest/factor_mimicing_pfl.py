__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .BackTestSinglePeriod import BackTestSinglePeriod


class FactorMimicPtflSpcalc(BackTestSinglePeriod):
	def __init__(self, factor_name, weighting=None):
		self.factor_name = factor_name
		self.weighting = weighting

	def get_config(self):
		config = {}
		config['Factor Name'] = self.factor_name
		config['Weighting'] = self.weighting
		return config

	def get_func_name(self):
		return 'Factor Mimicing Portfolio'

	def calc_pnl(self, univ, factor_exp_mat, t, **kwargs):
		'''
		Create a factor mimicing portfolio
		'''
		ret_name = 'f_log_ret_1'
		univ_sp, factor_exp_sp = univ[t].copy(), factor_exp_mat[t].copy()

		if self.weighting:
			w = np.diag(1 / factor_exp_sp[self.weighting])
		else:
			w = np.eye(factor_exp_sp.shape[0])
		beta = np.asarray(factor_exp_sp[self.factor_name])
		beta = beta.reshape(len(beta), 1)
		h = (np.linalg.inv((beta.T).dot(w.dot(beta))).dot((beta.T).dot(w))).T
		fmpfl = factor_exp_sp[['date', 'ticker', ret_name]]
		fmpfl['weight'] = h
		pnl_sp = np.dot(fmpfl['weight'], fmpfl[ret_name])
		ptfl_sp = fmpfl[['date', 'ticker', 'weight']]
		return ptfl_sp, pnl_sp