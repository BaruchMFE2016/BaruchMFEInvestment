__author__ = 'Derek Qi'
# Doing portfolio backtest and generates output

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse

from setup.univ_setup import *
from factor_mining.combine_factors import *
from factor_mining.factor_model_regression import *

from factor_model.stock_ret_est import GenReturn 
from GenPosition import *

from factor_mining.factors.momentum import *

from factor_mining.Mark0 import *



if __name__ == '__main__':
	### universe setup ###
	big_table_dir = '/home/derek-qi/Documents/R3000_Data/data/r3000/big_table_full.csv'
	univ = univ_setup(big_table_dir)

	### model configuration ###
	# define and calculate all factors
	factors = alpha_four_factors(univ)
	# concat into factor exposure matrices
	factor_exp_mat = combine_factors(factors)

	# const setup
	factor_names = [k for k in factors.keys()]
	N_f = len(factor_names)
	datelst = sorted(factor_exp_mat.keys())
	N_T = len(datelst)
	
	# Calc stock returns
	ret_series = momentum(univ, 0, 1)

	### model fitting ###
	# Single period fit and build portfolio
	lookback = timedelta(weeks=104)
	dend = datetime(2017, 2, 10)
	dstart = dend - lookback
	# fit the factor return and mse
	fr, fr_mse = factor_model_fit(factor_exp_mat, ret_series, dstart, dend)

	### current period pool selection and stock return estimation ###
	
	### portfolio optimization ###
	fx = factor_exp_mat[dend]
	fx = fx.dropna()
	stock_list, w_opt = GenPosition(fr, fx)
	ptfl_full = pd.DataFrame({"Ticker": stock_list, "Weight": list(w_opt.T[0])})
	now = datetime.now()
	nowstr = now.strftime('%Y%m%d_%H:%M:%S')
	GenPortfolioReport(ptfl_full, report_file=outputdir + 'portfolio_report_long_only'+nowstr+'.csv', pt=True)
	ptfl_full.to_csv(outputdir + 'portfolio_long_only.csv', index=False)
    

	### generate pnl (single period) ###