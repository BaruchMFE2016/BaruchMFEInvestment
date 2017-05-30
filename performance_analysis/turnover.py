__author__ = 'Derek Qi'

# This module calculates turnover of strategies
import numpy as np
import pandas as pd


def sp_turnover(ptfl_old, ptfl_new):
	merged = pd.merge(ptfl_old, ptfl_new, on='ticker', how='outer', suffixes=('_old', '_new'))
	merged.fillna(0, inplace=True)
	to = np.sum(np.abs(merged['weight_old'] - merged['weight_new']))
	return to


def mp_turnover(ptfl_lst):
	num_period = len(ptfl_lst)
	period_to = [sp_turnover(ptfl_lst[i], ptfl_lst[i+1]) for i in range(num_period-1)]
	period_to = [0] + period_to
	return period_to