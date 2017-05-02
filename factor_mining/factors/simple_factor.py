__author__ = 'Derek Qi'

import pandas as pd
import numpy as np
from datetime import datetime


def simple_factor(univ, name):
	'''
	use existing columns in the table as factor
	return type: dictionary
	'''

	datelst = sorted(univ.keys())
	N_T = len(datelst)

	simp_fac = [0] * N_T

	for ti in range(N_T):
		simp_fac[ti] = univ[datelst[ti]].ix[:,['date','ticker',name]]

	return dict(zip(datelst, simp_fac))


def simple_factor_1step_math(univ, name, math_func):
	'''
	Doing one step math on an existing column
	math_func is a univariate math function operates on an array-like
	return type: dictionary
	'''
	datelst = sorted(univ.keys())
	N_T = len(datelst)
	new_name = math_func.__name__ + name

	simp_fac = [0] * N_T
	for ti in range(N_T):
		simp_fac[ti] = univ[datelst[ti]].ix[:, ['date', 'ticker', name]]
		simp_fac[ti][new_name] = math_func(simp_fac[ti][name])
		simp_fac[ti] = simp_fac[ti][['date', 'ticker', new_name]]

	return dict(zip(datelst, simp_fac))

