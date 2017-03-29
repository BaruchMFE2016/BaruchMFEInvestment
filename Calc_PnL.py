__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from performance_analysis.pa_core import *

def pnl_series_simple(ptfl_s, ret_s):
	'''
	calculate the series portfolio pnl in a period of time
	constant portfolio, simple version
	return: pandas dataframe with 2 columns, date and pnl
	'''
	ret_s = ret_s.fillna(0)
	dates = ret_s['Date'].tolist()

	ret_sel = ret_s[ptfl_s['Ticker']]
	ret_sel = np.array(ret_sel)

	w = np.array(ptfl_s['Weight'])

	pnl = np.dot(ret_sel, w)

	pnl_s = pd.DataFrame({'Date': dates, 'Pnl': pnl})
	return pnl_s




if __name__ == '__main__':
	lret = pd.read_csv('./data/raw_input/r3000_log_return.csv')
	ptfl = pd.read_csv('./output/portfolio_long_only.csv')
	
	pnl_s = pnl_series_simple(ptfl, lret)

	print('Period:', pnl_s['Date'].tolist()[1], 'to', pnl_s['Date'].tolist()[-1])
	print('Total Return:', tot_ret(pnl_s))
	print('Sharpe Ratio:', sharpe_ratio(pnl_s) * np.sqrt(52))
	print('Max DD:', max_drawdown(pnl_s))
	print('DD Length:', drawdown_length(pnl_s))
	
	plt.plot(np.cumsum(pnl_s['Pnl']))
	plt.show()