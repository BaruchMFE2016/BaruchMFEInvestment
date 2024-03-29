__author__ = 'Derek Qi'
# Takes the calculated factor exposure matrix and historical return sequence
# to generate estimated factor return and covariance

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from .robust import check_nan_friendly_finite_array
from pdb import set_trace as bp


LR = LinearRegression()


def factor_model_fit_single_period(univ_sp, fac_exp, ret, fitter=LR, **kwargs):
	'''
	Takes a factor exposure matrix and a return snapshot and generate estimated
	factor returns and covariance matrix by some fitting method
	fac_exp: N-by-f np array
	ret: N-by-1 np array
	coef: f-by-1 coefficient array
	mse: mean squared error of the fitting
	return: coef, mse
	'''

	N_1 = ret.shape[0]
	N_2, N_f = fac_exp.shape
	# if N_1 != N_2:
	# 	raise ValueError('Dimensions mismatch for return and factor exposure.')

	factor_names = fac_exp.columns[2:] # exclude date and ticker
	return_name = ret.columns[-1]

	merged = pd.merge(fac_exp, ret, how='inner', on='ticker')
	merged = merged.replace([np.inf, -np.inf], np.nan)
	merged = merged.dropna()
	# FACTOR NAMES ARE HARDCODED FOR NOW!
	# X = np.array(merged[['vol10', 'momentum', 'market_cap', 'beta']])
	X = np.array(merged[factor_names])
	r = np.array(merged[return_name])

	if not 'weight' in kwargs.keys():
		fitter.fit(X, r)
	else:
		bp()
		w = 1 / np.array(univ_sp.loc[merged.ticker, kwargs['weight']])
		w = check_nan_friendly_finite_array(w)
		fitter.fit(X, r, sample_weight=w)
	
	r_pred = fitter.predict(X)
	mse = mean_squared_error(r, r_pred)
	coef = fitter.coef_

	return coef, mse



def factor_model_fit(univ, factor_exp, ret, dstart, dend, fitter=LR, **kwargs):
	'''
	input: factor_exposure time series and stock return time series
	fit starting time and end time
	return: pandas dataframe
	'''
	datelst = sorted(factor_exp.keys())
	N_T = len(datelst)
	factor_names = factor_exp[datelst[0]].columns.tolist()[2:]
	N_f = len(factor_names)
	

	t_in, factor_return_lst, mse_lst = [], [], []

	for ti in range(N_T):
		t = datelst[ti]
		if t < dstart or t >= dend: # changed t > dend to t >= dend, because regression uses forward return, which is not available for the last period
			continue

		univ_sp = univ[t].copy()
		fac_exp_t = factor_exp[t].copy()
		ret_t = ret[t].copy()
		fr, mse = factor_model_fit_single_period(univ_sp, fac_exp_t, ret_t, fitter, **kwargs)
		factor_return_lst.append(fr)
		mse_lst.append(mse)
		t_in.append(t.strftime('%Y/%m/%d'))

	factor_return_lst = list(np.array(factor_return_lst).reshape([len(t_in), N_f]))
	factor_return_df = pd.concat([pd.DataFrame(t_in), pd.DataFrame(factor_return_lst)], axis=1)
	factor_return_df.columns = ['date'] + factor_names
	factor_return_mse = pd.concat([pd.DataFrame(t_in), pd.DataFrame(mse_lst)], axis=1)
	factor_return_mse.columns = ['date', 'mse']

	return factor_return_df, factor_return_mse