__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


def log_return(univ, period=1):
	'''
	Calculates multiperiod log return of names
	'''

	datelst = sorted(univ.keys())
	N_T = len(datelst)

	log_ret = [0] * N_T
	for ti in range(N_T):
		t = datelst[ti]
		if ti < period:
			log_ret[ti] = univ[datelst[ti]].ix[:,['date', 'ticker']]
			log_ret[ti]['log_return'] = np.nan
			continue

		p_head = univ[datelst[ti]].ix[:,['date', 'ticker', 'price']]
		p_tail = univ[datelst[ti - period]].ix[:,['date', 'ticker', 'price']]
		p = pd.merge(p_head, p_tail, how='inner', on='ticker', suffixes=('_h', '_t'), )

		p['log_return'] = np.log(p.price_h / p.price_t)
		p['date'] = univ[datelst[ti]].date.tolist()

		log_ret[ti] = p.ix[:,['date', 'ticker', 'log_return']]

	return dict(zip(datelst, log_ret))


def pct_return(univ, period=1):
	'''
	Calculates multiperiod percentage return of names
	'''

	datelst = sorted(univ.keys())
	N_T = len(datelst)

	pct_ret = [0] * N_T
	for ti in range(N_T):
		t = datelst[ti]
		if ti < period:
			pct_ret[ti] = univ[datelst[ti]].ix[:,['date', 'ticker']]
			pct_ret[ti]['pct_return'] = np.nan
			continue

		p_head = univ[datelst[ti]].ix[:,['date', 'ticker', 'price']]
		p_tail = univ[datelst[ti - period]].ix[:,['date', 'ticker', 'price']]
		p = pd.merge(p_head, p_tail, how='inner', on='ticker', suffixes=('_h', '_t'), )

		p['pct_return'] = (p.price_h - p.price_t) / p.price_t
		p['date'] = univ[datelst[ti]].date.tolist()

		pct_ret[ti] = p.ix[:,['date', 'ticker', 'pct_return']]

	return dict(zip(datelst, pct_ret))