'''
Simple version of backtest, does not do optimization
'''

__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from .BackTestSinglePeriod import BackTestSinglePeriod

class ModelResearch(object):
    def __init__(self, univ:dict, factor_exp_mat:dict, daterange:list, sp_calc:BackTestSinglePeriod):
        self.univ = univ
        self.factor_exp_mat = factor_exp_mat
        self.dstart, self.dend = np.min(daterange), np.max(daterange)
        self.sp_calc = sp_calc
        self.metric_func_name = ['r2_score', 'mean_squared_error']
        self.all_metrics = {}

    def get_config(self):
        config = {}
        config['Strategy Name'] = self.sp_calc.get_func_name()
        config['Strategy config'] = self.sp_calc.get_config()
        config['Date range'] = [self.dstart.strftime('%Y-%m-%d'), self.dend.strftime('%Y-%m-%d')]
        return config

    def test_pfmc(self, **kwargs):
        ret_name = 'f_log_ret_1'
        datelst = sorted(self.univ.keys())
        tin_lst, ptfl_lst, pnl_lst = [], [], []
        count = 0
        for t in datelst:
            if t < self.dstart or t > self.dend:
                continue
            if kwargs.get('silence'):
                print(t)
            ret_pred, _ = self.sp_calc.gen_alpha_sigma(self.univ, self.factor_exp_mat, t, **kwargs)
            ret_true = self.factor_exp_mat[t][ret_name]
            metric_funcs = [eval(funcname) for funcname in self.metric_func_name]
            sp_metrics = [func(ret_true, ret_pred) for func in metric_funcs]
            self.all_metrics.update({t: dict(zip(self.metric_func_name, sp_metrics))})

    def plot_result(self, metric_name='r2_score'):
        all_times = sorted(list(self.all_metrics.keys()))
        result = [self.all_metrics[t][metric_name] for t in all_times]
        plt.plot(result)
        plt.show()
                

