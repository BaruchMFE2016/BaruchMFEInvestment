__author__ = 'Derek Qi'

import numpy as np
import sklearn.preprocessing as pp


def winsorize(X, thresh=3, copy=True):
	'''
	Doing winsorization to a dataset along an axis
	Truncate data with 

	Parameters:
	------------
	X: array-like
		dataset
	thresh: int
		threshold of cutting down features
	copy: bool
		default is True, set to false to do in-place

	Return:
	------------
	X_wins: array-like
		winsorized dataset
	'''

	X = np.asarray(X)
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	up_thresh = mean + thresh * std
	down_thresh = mean - thresh * std

	N, F = X.shape
	if copy:
		X_wins = X.copy()
	else:
		X_wins = X

	for i in range(F):
		f = X_wins[:,i]
		ut, dt = up_thresh[i], down_thresh[i]
		f[f > ut] = ut
		f[f < dt] = dt

	return X_wins


def process_batch(fac_ex, processor):
	'''
	Doing a process procedure on a series of factor exposure matrix

	Parameters
	-----------
	fac_ex: dictionary
		keys: time
		values: factor exposure matrix with respect to that time
	processor: function handle
		data processing function that could work with the following call:
			X_new = processor(X)
		where X is a single factor exposure matrix
	'''
	datelst = sorted(fac_ex.keys())
	N_T = len(datelst)
	fx_lst = []

	for ti in range(N_T):
		t = datelst[ti]
		fx = fac_ex[t]
		fx = fx.dropna()
		X = fx.iloc[:,2:]
		if X.shape[0] > 0:
			X_p = processor(X)
		else:
			X_p = np.zeros(X.shape)
		fx.iloc[:,2:] = X_p
		fx_lst.append(fx)

	return dict(zip(datelst, fx_lst))



if __name__ == '__main__':
	X = np.random.normal(size=(10000, 2))
	print(np.max(X, axis=0))

	X_wins = winsorize(X, thresh=2)
	print(np.max(X_wins, axis=0))
