__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .BackTestSinglePeriod import *
from performance_analysis.pa_core import simple_pa
from performance_analysis.turnover import *

class BackTest(object):
    def __init__(self, univ:dict, factor_exp_mat:dict, daterange:list, sp_calc:BackTestSinglePeriod, rebal=1):
        self.univ = univ
        self.factor_exp_mat = factor_exp_mat
        self.dstart, self.dend = np.min(daterange), np.max(daterange)
        self.rebal = rebal
        self.sp_calc = sp_calc
        self.commission = 0.0003
        self.has_pnl, self.has_pa = False, False

    def get_config(self):
        config = {}
        config['Strategy Name'] = self.sp_calc.get_func_name()
        config['Strategy config'] = self.sp_calc.get_config()
        config['Date range'] = [self.dstart.strftime('%Y-%m-%d'), self.dend.strftime('%Y-%m-%d')]
        config['commission'] = self.commission
        config['Rebalance period'] = self.rebal

        if self.has_pnl:
            pass
        
        if self.has_pa:
            config['Performance'] = self.pa

        return config

    def calc_pnl(self, **kwargs):
        datelst = sorted(self.univ.keys())
        tin_lst, ptfl_lst, pnl_lst = [], [], []
        count = 0
        for t in datelst:
            if t < self.dstart or t > self.dend:
                continue

            if kwargs.get('silence'):
                print(t)

            if count == 0:
                ptfl_sp, pnl_sp = self.sp_calc.calc_pnl(self.univ, self.factor_exp_mat, t, **kwargs)
            else:
                ret_name = 'f_log_ret_1'
                op_na = pd.merge(ptfl_sp, self.univ[t], on='ticker', how='inner') # This stands for old portfolio, new analytics
                pnl_sp = np.dot(op_na['weight'], op_na[ret_name])

            tin_lst.append(t)
            ptfl_lst.append(ptfl_sp.copy())
            pnl_lst.append(pnl_sp)

            count -= 1
            count %= self.rebal

        # self.pnl_lst = pd.DataFrame({'date':tin_lst, 'pnl':pnl_lst})
        pnl_lst = pd.DataFrame({'date':tin_lst, 'pnl':pnl_lst})
        self.ptfl_lst = ptfl_lst
        self.has_pnl = True

        if 'net_cost' in kwargs.keys():
            if kwargs['net_cost']:
                print(1)
                period_to = self.calc_turnover()
                period_commission = self.commission * np.asarray(period_to)
                pnl_lst['pnl'] -= period_commission

        self.pnl_lst = pnl_lst
        return ptfl_lst, pnl_lst

    def calc_turnover(self, **kwargs):
        assert self.has_pnl, 'Calculate pnl and portfolio first.'
        period_to = mp_turnover(self.ptfl_lst)
        return period_to


    def calc_pa(self, **kwargs):
        self.pa = simple_pa(self.pnl_lst, **kwargs)
        self.pa['TurnOver'] = np.mean(self.calc_turnover())
        self.has_pa = True
