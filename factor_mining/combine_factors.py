__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


def combine_factors_2(factors_dict):
	'''
	Take a dictionary with 
	keys: factor names
	values: factor content (another dictionary with a same list of time as keys)

	Returns a dictionary with
	keys: time periods
	values: factor exposure matrix within that time
	'''

	factor_names = [k for k in factors_dict.keys()]
	N_f = len(factor_names)
	datelst = sorted(factors_dict[factor_names[0]].keys())
	N_T = len(datelst)

	f_ex_lst = [0] * N_T
	for ti in range(N_T):
		# if ti % 50 == 0:
			# print(ti)
		f_ti = [factors_dict[k][datelst[ti]] for k in factor_names]
		f_ex_ti = [f_ti[0].date, f_ti[0].ticker] + [f.iloc[:,2] for f in f_ti]
		f_ex_ti = [f.reset_index(drop=True) for f in f_ex_ti]
		f_ex = pd.concat(f_ex_ti, axis=1, ignore_index=True, keys=['date', 'ticker'] + factor_names)
		f_ex.columns = ['date', 'ticker'] + factor_names
		f_ex.dropna(how='all', inplace=True)
		f_ex_lst[ti] = f_ex

	return dict(zip(datelst, f_ex_lst))


def combine_factors(factors_dict):
	'''
	Take a dictionary with 
	keys: factor names
	values: factor content (another dictionary with a same list of time as keys)

	Returns a dictionary with
	keys: time periods
	values: factor exposure matrix within that time
	'''

	factor_names = [k for k in factors_dict.keys()]
	N_f = len(factor_names)
	datelst = sorted(factors_dict[factor_names[0]].keys())
	N_T = len(datelst)

	f_ex_lst = [0] * N_T
	for ti in range(N_T):
		t = datelst[ti]
		for fi in range(N_f):
			if fi == 0:
				f_ex = factors_dict[factor_names[fi]][t].copy()
			else:
				f_ex = pd.merge(f_ex, factors_dict[factor_names[fi]][t].ix[:, ['ticker', factor_names[fi]]], how='inner', on='ticker')

		f_ex_lst[ti] = f_ex

	return dict(zip(datelst, f_ex_lst))
