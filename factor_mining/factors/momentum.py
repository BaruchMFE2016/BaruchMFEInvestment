__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


def momentum(univ, head, tail):
	'''
	Calculates the momentum factor defined as follows
	momentum[t] = (p[t - head] - p[t - tail]) / p[t - tail]
	head and tail are numbers of time periods
	'''

	datelst = sorted(univ.keys())
	N_T = len(datelst)

	momentum = [0] * N_T
	for ti in range(N_T):
		t = datelst[ti]
		if ti < tail:
			momentum[ti] = univ[datelst[ti]].ix[:,['date', 'ticker']]
			momentum[ti]['momentum'] = np.nan
			continue

		p_head = univ[datelst[ti - head]].ix[:,['date', 'ticker', 'price']]
		p_tail = univ[datelst[ti - tail]].ix[:,['date', 'ticker', 'price']]
		p = pd.merge(p_head, p_tail, how='inner', on='ticker', suffixes=('_h', '_t'), )

		p['momentum'] = (p.price_h - p.price_t) / p.price_t
		p['date'] = p['date_h'].tolist()

		momentum[ti] = p.ix[:,['date', 'ticker', 'momentum']]

	return dict(zip(datelst, momentum))