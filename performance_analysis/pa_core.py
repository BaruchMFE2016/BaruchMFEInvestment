__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# This module is trying to calculate core metrics in performance analysis
# Take portfolio and benchmark pnls of a period


def tot_ret(ptfl):
    return sum(ptfl['Pnl'])


def cagr(ptfl):
    r = np.mean(ptfl['Pnl'])
    cagr = np.exp(r) - 1
    return cagr


def vol(ptfl):
    return np.std(ptfl['Pnl'])


def sharpe_ratio(ptfl, bmk=None):
    if bmk is None:
        bmk = pd.DataFrame({'Pnl' : np.zeros(ptfl.shape[0])})

    excess_return = ptfl['Pnl'] - bmk['Pnl']
    sharpe = np.mean(excess_return) / np.std(ptfl['Pnl'])
    return sharpe


def information_ratio(ptfl, bmk=None):
    if bmk is None:
        bmk = pd.DataFrame({'Pnl': np.zeros(ptfl.shape[0])})

    excess_return = ptfl['Pnl'] - bmk['Pnl']
    ir = np.mean(excess_return) / np.std(excess_return)
    return ir


# def max_drawdown(ptfl):
#     pnl = ptfl['Pnl']
#     ptfl_v = np.cumsum(pnl)
#     max_dd = 0
#     for i in range(1, len(ptfl_v)):
#         cur_max = max(ptfl_v[0:i])
#         cur_dd = cur_max - ptfl_v[i]
#         if cur_dd > max_dd:
#             max_dd = cur_dd
#     return max_dd


def max_drawdown(ptfl):
    pnl = ptfl['Pnl']
    cpnl = np.cumsum(pnl)
    return min(cpnl)

def drawdown_length(ptfl):
    pnl = ptfl['Pnl']
    return sum(pnl < 0)


def simple_pa(ptfl):
	pmfc = (cagr(ptfl), vol(ptfl), sharpe_ratio(ptfl), max_drawdown(ptfl), drawdown_length(ptfl))
	print('CAGR:%f \nVolatility:%f\nSharpe_ratio:%f\nmax drawdown: %f\ndrawdown length: %f\n' % pmfc)


def plot_nav(ptfl, show=True, savedir=None):
	pnl = ptfl['Pnl']
	cumlogret = np.cumsum(pnl)
	nav = np.exp(cumlogret)
	plt.plot(nav, label='Net Asset Value')
	plt.legend(loc=0)
	if show:
		plt.show()
	if savedir:
		plt.savefig(savedir + 'NAV_curve.png')


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
    pfmc = [cagr(ptfl_pnl), vol(ptfl_pnl), sharpe_ratio(ptfl_pnl, bmk_pnl), information_ratio(ptfl_pnl), max_drawdown(ptfl_pnl)]
    print(pfmc)

    plt.plot(np.cumsum(ptfl_pnl['Pnl']))
    plt.show()