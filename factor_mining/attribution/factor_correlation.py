__author__ = 'Derek Qi'
# This module aims at examine the effect of factors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..robust import check_nan_friendly_finite_array
from datetime import datetime, timedelta

def nancov_2s(x:np.array, y:np.array):
	'''
	simple covariance function for 2 numpy 1-d array
	'''
	assert x.shape == y.shape, 'x and y has different shape'
	xy = np.vstack((x, y))
	xy = check_nan_friendly_finite_array(xy)
	xy = xy[:,(~np.isnan(xy[0]) * ~np.isnan(xy[1]))]
	x_new, y_new = xy[0], xy[1]
	cov = np.dot((x_new - np.mean(x_new)) ** 2, (y_new - np.mean(y_new)) ** 2) / (x_new.shape[0] - 1)
	return cov

def nancorr_2s(x:np.array, y:np.array):
	cov = nancov_2s(x, y)
	corr = cov / np.sqrt(np.nanvar(x) * np.nanvar(y))
	return corr


def factor_correlation_single_period(univ_sp, factor_exp_mat_sp, demean=None):

	ret_name = 'f_log_ret_1'
	factor_names = factor_exp_mat_sp.columns[2:] # exclude date and ticker

	if demean == 'industry':
		merged = pd.merge(univ_sp[['date', 'ticker', demean, ret_name]], factor_exp_mat_sp, on='ticker', how='inner')
		merged = merged.dropna()
		demean_return = merged[ret_name] - merged.groupby(demean)[ret_name].transform('mean')
		demean_return.name = 'demean_ret'
		# demean_return = pd.concat([merged['ticker'], demean_return], axis=1) # might be useless
		corr = [nancorr_2s(merged[ret_name], merged[name]) for name in factor_names]
		
	else:
		merged = pd.merge(univ_sp[['date', 'ticker', ret_name]], factor_exp_mat_sp, on='ticker', how='inner')
		merged = merged.dropna()
		corr = [nancorr_2s(merged[ret_name], merged[name]) for name in factor_names]

	t = univ_sp.date.tolist()[0]
	return dict(zip(['date'] + factor_names.tolist(), [t] + corr))


def factor_correlation(univ, factor_exp_mat, lag=0, demean=None):
	datelst = sorted(univ.keys())
	lag_w = timedelta(weeks=lag)
	factor_names = factor_exp_mat[datelst[0]].columns[2:].tolist()
	corr_df = pd.DataFrame(columns = ['date'] + factor_names)
	for t in datelst:
		t_lagged = max(datelst[0], t - lag_w)
		corr = factor_correlation_single_period(univ[t], factor_exp_mat[t_lagged], demean)
		corr_df = corr_df.append(corr, ignore_index=True)
	return corr_df


def factor_correlation_plot(corr_df, ma=12):
	factor_names = corr_df.columns[1:].tolist()
	corr_df = pd.rolling_mean(corr_df, window=ma)
	for name in factor_names:
		plt.plot(corr_df[name], label=name)
	plt.legend(loc=0)
	plt.show()
