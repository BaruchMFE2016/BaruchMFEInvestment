__author__ = 'Derek Qi'

import numpy as np
import sklearn.preprocessing as pp
from .robust import *


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
		fx = fac_ex[t].copy()
		# fx = fx.dropna()
		factor_names = fx.columns[2:] # exclude date and ticker
		X = fx.loc[:,factor_names]
		if X.shape[0] > 0:
			X = np.asarray(X)
			X_p = processor(X)
		else:
			X_p = np.zeros(X.shape)
		fx.loc[:,factor_names] = X_p
		fx_lst.append(fx)

	return dict(zip(datelst, fx_lst))



if __name__ == '__main__':
	X = np.random.normal(size=(10000, 2))
	print(np.max(X, axis=0))
	print(np.median(X), np.median(np.abs(X - np.median(X))))
	X_wins = winsorize(X, const=3)
	print(np.max(X_wins, axis=0))
