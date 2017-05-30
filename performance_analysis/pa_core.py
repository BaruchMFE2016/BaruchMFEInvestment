__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# This module is trying to calculate core metrics in performance analysis
# Take portfolio and benchmark pnls of a period


def annualize_factor(ptfl):
	N_PERIOD = ptfl.shape[0]
	if N_PERIOD <= 1:
		raise('pnl series has less than 2 periods')

	start_date = ptfl['date'][0]
	end_date = ptfl['date'][N_PERIOD - 1]
	diff_date = end_date - start_date
	
	period_per_year = N_PERIOD / (diff_date.days / 365.25) * (N_PERIOD) / (N_PERIOD-1)
	return period_per_year


def tot_ret(ptfl):
	'''
	Gives percentage total return
	'''
	return np.exp(sum(ptfl['pnl'])) - 1


def cagr(ptfl, annualize=True):
	'''
	Cumulative annual growth rate
	This number is in percentage
	'''
	c = 1
	if annualize:
		c = annualize_factor(ptfl)

	r = np.mean(ptfl['pnl']) * c
	cagr = np.exp(r) - 1
	return cagr


def vol(ptfl, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)

	return np.std(ptfl['pnl']) * np.sqrt(c)


def sharpe_ratio(ptfl, bmk=None, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)


	if bmk is None:
		bmk = pd.DataFrame({'pnl' : np.zeros(ptfl.shape[0])})

	excess_return = ptfl['pnl'] - bmk['pnl']
	sharpe = np.mean(excess_return) / np.std(ptfl['pnl']) * np.sqrt(c)
	return sharpe


def information_ratio(ptfl, bmk=None, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)
	if bmk is None:
		bmk = pd.DataFrame({'pnl': np.zeros(ptfl.shape[0])})

	excess_return = ptfl['pnl'] - bmk['pnl']
	ir = np.mean(excess_return) / np.std(excess_return) * np.sqrt(c)
	return ir


def max_drawdown(ptfl):
	 pnl = ptfl['pnl']
	 ptfl_v = np.cumsum(pnl)
	 max_dd = 0
	 for i in range(1, len(ptfl_v)):
		 cur_max = max(ptfl_v[0:i])
		 cur_dd = cur_max - ptfl_v[i]
		 if cur_dd > max_dd:
			 max_dd = cur_dd
	 return max_dd


def min_wealth(ptfl):
	pnl = ptfl['pnl']
	cpnl = np.cumsum(pnl)
	return np.exp(min(cpnl)) - 1

def drawdown_length(ptfl):
	pnl = ptfl['pnl']
	return sum(np.cumsum(pnl) < 0)


def simple_pa(ptfl, annualize=True, silent=False):
	pfmc = {'CAGR': cagr(ptfl, annualize=annualize),\
			'Volatility': vol(ptfl, annualize=annualize),\
			'Sharpe': sharpe_ratio(ptfl, annualize=annualize),\
			'Max_Drawdown': max_drawdown(ptfl),\
			'Drawdown_Length': drawdown_length(ptfl)}
	if not silent:
		for k in ['CAGR', 'Volatility', 'Sharpe', 'Max_Drawdown', 'Drawdown_Length']:
			print(k, ':\t', pfmc[k])
	return pfmc


def plot_nav(ptfl, show=True, savedir=None):
	pnl = ptfl['pnl']
	cumlogret = np.cumsum(pnl)
	nav = np.exp(cumlogret)
	plt.plot(nav, label='Net Asset Value')
	plt.legend(loc=0)

	if savedir:
		plt.savefig(savedir + 'NAV_curve_' + datetime.now().strftime('%Y%m%d_%H:%M:%S') + '.png', format='png', dpi=300)
		plt.close()
	if show:
		plt.show()


if __name__ == '__main__':
	# Generate some random data and pack them
	n = 52
	alpha = 1e-4
	pnl_p = np.random.normal(0, 1, n) + alpha
	pnl_b = np.random.normal(0, 1, n)

	dstart = datetime.strptime('2015/01/02', '%Y/%m/%d')
	dt = timedelta(weeks=1)
	dseries = [dstart + i * dt for i in range(n)]

	ptfl_pnl = pd.DataFrame({'date':dseries, 'pnl':pnl_p})
	bmk_pnl  = pd.DataFrame({'date':dseries, 'pnl':pnl_b})

	# test functions
	simple_pa(ptfl_pnl, annualize=True)