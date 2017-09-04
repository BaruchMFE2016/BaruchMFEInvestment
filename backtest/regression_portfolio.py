__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from factor_mining.robust import check_nan_friendly_finite_array
from .BackTestSinglePeriod import BackTestSinglePeriod
from .ptfl_optim import PtflOptimizer

LR = LinearRegression()
optimzr = PtflOptimizer(U=0.2, L=-0.2)


def get_factor_cov(fr, method, **kwargs):
    N_t, N_f = fr.shape
    if method == 'EWMA':
        assert 'hl' in kwargs.keys(), 'No halflife param given'
        hl = kwargs['hl']
        # First get EWMA weights
        delta = 0.5 ** (1. / hl)
        w = delta ** np.arange(start=0, stop=N_t, step=1)
        w /= np.sum(w)

        # Then calculate the variance and correlation
        mu = w.dot(fr)
        var = w.dot(fr**2) - mu ** 2
        fr_nor_w = ((fr - mu) / np.sqrt(var)) * np.sqrt(w[:, None])
        corr = (fr_nor_w.T).dot(fr_nor_w)  # Estimated factor correlation
        D = np.diag(np.sqrt(var))  # Estimated single std
        cov = D.dot(corr).dot(D)  # Estimated covariance matrix
        return cov


class RegressionPtflSpcalc(BackTestSinglePeriod):
    def __init__(self, all_factor_names, fitter=LR, optimzr=optimzr, weighting=None):
        self.all_factor_names = all_factor_names
        self.all_factor_returns = {}
        self.fitter = fitter
        self.optimzr = optimzr
        self.weighting = weighting
        self.model_rsq = {}

    def get_config(self):
        config = {}
        config['All_Factor_Names'] = self.all_factor_names
        config['Weighting'] = self.weighting
        return config

    def get_func_name(self):
        return 'Regression Portfolio'

    def est_factor_ret_mse(self, univ, factor_exp_mat, t, **kwargs):
        '''
        Cross sectional fit on a single period of the data
        '''
        # print('calculate regression coef for time ', datetime.strftime(t, '%Y%m%d'))
        ret_name = 'f_log_ret_1'
        # u_sp = univ[t].copy()
        factor_exp_sp = factor_exp_mat[t].copy()

        X = np.array(factor_exp_sp[self.all_factor_names])
        r = np.array(factor_exp_sp[ret_name])

        if self.weighting is None:
            self.fitter.fit(X, r)
        else:
            w = np.asarray(factor_exp_mat[self.weighting])
            self.fitter.fit(X, r, sample_weight=w)

        r_pred = self.fitter.predict(X)
        mse = mean_squared_error(r, r_pred)
        fr = self.fitter.coef_
        return fr, mse

    def calc_pnl(self, univ, factor_exp_mat, t, n_lookback=30, **kwargs):
        '''
        Do a single period backtest on univ[t]
        t: datetime object that is one of the element in keys of univ
        factor_exp_mat, ret_series: factor exposure and return time series
        '''
        ### Step 1: Get the estimated factor return and covariance
        datelst = [t - timedelta(weeks=i) for i in range(1, n_lookback + 1)]
        datelst = datelst[::-1]
        ret_name = 'f_log_ret_1'

        for dt in datelst:
            if not dt in self.all_factor_returns.keys():
                fr, mse = self.est_factor_ret_mse(univ, factor_exp_mat, dt, **kwargs)
                self.all_factor_returns[dt] = {'factor_returns': fr, 'mse': mse}

            else:
                pass
                # fr, mse = self.all_factor_returns[dt]['factor_returns'], self.all_factor_returns[dt]['mse']

        all_fr = np.asarray([self.all_factor_returns[dt]['factor_returns'] for dt in datelst])
        fr_sp = np.mean(all_fr, axis=0)

        ### Step 2: Generate estimates of stock returns and covariance
        fr_cov = get_factor_cov(all_fr, method='EWMA', hl=120)
        fx_sp = factor_exp_mat[t].copy()
        fx = np.asarray(fx_sp[self.all_factor_names])
        D = np.eye(fx_sp.shape[0]) * self.all_factor_returns[dt]['mse'] # diagonal term of all same numbers
        alpha_est = fx.dot(fr_sp)
        sigma_est = (fx.dot(fr_cov)).dot(fx.T) + D
        if 1:
            rsq = r2_score(fx_sp[ret_name], alpha_est)
            self.model_rsq[t] = rsq

        ### Step 3: use min-var optimization to calculate portfolio and pnl
        has_short = False
        if 'has_short' in kwargs.keys():
            has_short = kwargs['has_short']

        all_tickers = fx_sp['ticker'].tolist()
        N_INS = len(all_tickers)
        w_opt = optimzr.opt(alpha=alpha_est, sigma=sigma_est, w_old=np.ones([N_INS, 1]) / N_INS, has_short=has_short)
        w_opt[abs(w_opt) < 1e-5] = 0
        w_opt /= np.sum(w_opt)
        ptfl_sp = pd.DataFrame({'date': fx_sp['date'].tolist(),
                                'ticker': all_tickers, 'weight': w_opt[:, 0]})
        pnl_sp = np.dot(ptfl_sp.weight, fx_sp[ret_name])
        return ptfl_sp, pnl_sp
