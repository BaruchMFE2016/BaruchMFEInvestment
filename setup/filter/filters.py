__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


def filt_na(univ):
	'''
	filter out na in a universe
	'''
	datelst = sorted(univ.keys())
	N_T = len(datelst)

	for ti in range(N_T):
		t = datelst[ti]
		univ_ti = univ[t]
		univ_ti.dropna(inplace=True, how='any')


def filt_byval(univ, varname, thrd, keep='above'):
	'''
	filter out any element under varname that has value thrd or below
	'''
	datelst = sorted(univ.keys())
	N_T = len(datelst)

	for ti in range(N_T):
		t = datelst[ti]
		univ_ti = univ[t]
		idx_in = univ_ti[varname] > thrd
		if keep == 'below':
			idx_in = 1 - idx_in
		univ[t] = univ_ti.ix[idx_in,:]