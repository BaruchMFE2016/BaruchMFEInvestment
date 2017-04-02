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

