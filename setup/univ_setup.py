__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime

from .filter.filters import *

def excel_date_trans(s):
	sl = s.split('/')
	m = int(sl[0])
	d = int(sl[1])
	y = int(sl[2])

	dt = datetime(y, m, d)
	return dt


def univ_setup(big_table_dir):
	'''
	Read the big_table format cleaned data and set universe base on that
	return type: dictionary, keys are dates in datetime.datetime format
	values are corresponding security information
	'''
	big_table = pd.read_csv(big_table_dir)
	
	try:
		big_table.drop('Unnamed: 0', axis=1)
	except:
		pass

	datecol = big_table.date.unique()
	if datecol[0].find('/') != -1:
		datelst = [excel_date_trans(s) for s in datecol]
	else:
		datelst = [datetime.strptime(dstr,'%Y-%m-%d') for dstr in datecol]

	N_T = datecol.shape[0]
	subtable = [0] * N_T

	for ti in range(N_T):
		t = datecol[ti]
		subtable[ti] = big_table.ix[big_table.date == t,:]

	return dict(zip(datelst, subtable))



if __name__ == '__main__':
	big_table_dir = '/home/derek-qi/Documents/R3000_Data/data/r3000/big_table_fullv4.csv'
	r3000_univ = univ_setup(big_table_dir)

	filt_na(r3000_univ)

	filt_byval(r3000_univ, 'price', 10)
	filt_byval(r3000_univ, 'volume', 300000 * 5)