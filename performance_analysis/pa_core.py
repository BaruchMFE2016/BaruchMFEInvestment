__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# This module is trying to calculate core metrics in performance analysis
# Take portfolio and benchmark pnls of a period


def annualize_factor(ptfl):
	return 52


def tot_ret(ptfl):
	return np.exp(sum(ptfl['Pnl'])) - 1


def cagr(ptfl, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)

	r = np.mean(ptfl['Pnl']) * c
	cagr = np.exp(r) - 1
	return cagr


def vol(ptfl, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)

	return np.std(ptfl['Pnl']) * np.sqrt(c)


def sharpe_ratio(ptfl, bmk=None, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)


	if bmk is None:
		bmk = pd.DataFrame({'Pnl' : np.zeros(ptfl.shape[0])})

	excess_return = ptfl['Pnl'] - bmk['Pnl']
	sharpe = np.mean(excess_return) / np.std(ptfl['Pnl']) * np.sqrt(c)
	return sharpe


def information_ratio(ptfl, bmk=None, annualize=True):
	c = 1
	if annualize:
		c = annualize_factor(ptfl)
	if bmk is None:
		bmk = pd.DataFrame({'Pnl': np.zeros(ptfl.shape[0])})

	excess_return = ptfl['Pnl'] - bmk['Pnl']
	ir = np.mean(excess_return) / np.std(excess_return) * np.sqrt(c)
	return ir


# def max_drawdown(ptfl):
#	 pnl = ptfl['Pnl']
#	 ptfl_v = np.cumsum(pnl)
#	 max_dd = 0
#	 for i in range(1, len(ptfl_v)):
#		 cur_max = max(ptfl_v[0:i])
#		 cur_dd = cur_max - ptfl_v[i]
#		 if cur_dd > max_dd:
#			 max_dd = cur_dd
#	 return max_dd


def max_drawdown(ptfl):
	pnl = ptfl['Pnl']
	cpnl = np.cumsum(pnl)
	return np.exp(min(cpnl)) - 1

def drawdown_length(ptfl):
	pnl = ptfl['Pnl']
	return sum(np.cumsum(pnl) < 0)


def simple_pa(ptfl, annualize=True):
	pmfc = (cagr(ptfl, annualize=annualize), vol(ptfl, annualize=annualize), sharpe_ratio(ptfl, annualize=annualize), max_drawdown(ptfl), drawdown_length(ptfl))
	print('CAGR: %f\nVolatility: %f\nSharpe_ratio: %f\nmax drawdown: %f\ndrawdown length: %d\n' % pmfc)


def plot_nav(ptfl, show=True, savedir=None):
	pnl = ptfl['Pnl']
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

	ptfl_pnl = pd.DataFrame({'Date':dseries, 'Pnl':pnl_p})
	bmk_pnl  = pd.DataFrame({'Date':dseries, 'Pnl':pnl_b})

	# test functions
	simple_pa(ptfl_pnl, annualize=True)