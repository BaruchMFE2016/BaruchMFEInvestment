__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from BackTestSinglePeriod import *

class BackTest(object):
	def __init__(self, univ:dict, daterange:list, sp_calc:BackTestSinglePeriod, rebal=1):
		self.univ = univ
		self.dstart, self.dend = np.min(daterange), np.max(daterange)
		self.rebal = rebal
		self.sp_calc = sp_calc
		self.has_pnl, self.has_pa = False, False

	def get_config(self):
		config = {}
		config['Strategy Name'] = self.sp_calc.get_func_name()
		config['Strategy config'] = self.sp_calc.get_config()
		config['Date range'] = [self.dstart, self.dend]
		config['Rebalance period'] = self.rebal

		if has_pnl:
			pass
		
		if has_pa:
			pass

		return config

	def calc_pnl(self, **kwargs):
		datelst = sorted(self.univ.keys())
		ptfl_lst, pnl_lst = [], []
		count = 0
		for t in datelst:
			if t < self.dstart or t > self.tend:
				continue

			if count == 0:
				ptfl_sp, pnl_sp = self.sp_calc.calc_pnl(univ, factor_exp_mat, t, kwargs)
			else:
				ret_name = 'f_log_ret_1'
				op_na = pd.merge(ptfl_sp, univ[t], on='ticker', how='inner') # This stands for old portfolio, new analytics
				pnl_sp = np.dot(op_na['weight'], op_na[ret_name])

			ptfl_lst.append(ptfl_sp.copy())
			pnl_lst = append(pnl_sp)

			count -= 1
			count %= self.rebal

		self.pnl_lst = pnl_lst
		self.ptfl_lst = ptfl_lst
		self.has_pnl = True
		return ptfl_lst, pnl_lst

	def simple_pa(self, **kwargs):
		pass
		self.has_pa = True