__author__ = 'Derek Qi'
'''
Markowitz mean-variance optimization with constraints
Another implementation, with scipy.optimize
'''

import numpy as np
from scipy import optimize
from .Optimizer import Optimizer

class MeanVarOptim(Optimizer):
    def __init__(self, **kwargs):
        self.param_config = kwargs
        if not 'gamma' in kwargs.keys():
            self.param_config['gamma'] = 1
        if not 'lambd' in kwargs.keys():
            self.param_config['lambd'] = 0
        if not 'L' in kwargs.keys():
            self.param_config['L'] = -1
        if not 'U' in kwargs.keys():
            self.param_config['U'] = 1
        if not 'dlt' in kwargs.keys():
            self.param_config['dlt'] = 1
        if not 'Lev' in kwargs.keys():
            self.param_config['Lev'] = 2

    def _opt_long(self, w_old, alpha, sigma, **kwargs):
        gamma, lambd, L, U, dlt = kwargs['gamma'], kwargs['lambd'], kwargs['L'], kwargs['U'], kwargs['dlt']
        
        def objective(w):
            return 0.5 * gamma * w.T.dot(sigma).dot(w) - w.dot(alpha) + lambd * np.sum(abs(w - w_old))
        
        cons = ({'type':'eq',   'fun':lambda x: np.sum(x)-1}, # sum(w_i) = 1
                {'type':'ineq', 'fun':lambda x: x}, # w_i >= 0
                {'type':'ineq', 'fun':lambda x: U-x}, # x_i <= U
                {'type':'ineq', 'fun':lambda x: x-L},) # x_i >= L

        res = optimize.minimize(objective, w_old, constraints=cons)
        if res.status:
            raise Exception(res.message)
        w_opt = res.x.reshape(-1, 1)
        return w_opt

    def _opt_long_short(self, w_old, alpha, sigma, **kwargs):
        import pickle
        with open('./output/temp/alpha.pkl', 'wb+') as f:
           pickle.dump(alpha, f)
        with open('./output/temp/sigma.pkl', 'wb+') as f:
            pickle.dump(sigma, f)
        gamma, lambd, L, U, dlt, lev = kwargs['gamma'], kwargs['lambd'], kwargs['L'], kwargs['U'], kwargs['dlt'], kwargs['Lev']
        
        def objective(w):
            return 0.5 * gamma * w.T.dot(sigma).dot(w) - w.dot(alpha) + lambd * np.sum((w - w_old)**2)
        
        cons = ({'type':'eq',   'fun':lambda x: np.sum(x)-1}, # sum(w_i) = 1
                {'type':'ineq', 'fun':lambda x: lev - np.sum(abs(x))}, # sum(|w_i|) <= Lev
                {'type':'ineq', 'fun':lambda x: U-x}, # x_i <= U
                {'type':'ineq', 'fun':lambda x: x-L},) # x_i >= L

        res = optimize.minimize(objective, w_old, constraints=cons)
        if res.status:
            raise Exception(res.message)
        w_opt = res.x.reshape(-1, 1)
        return w_opt

    def opt(self, alpha, sigma, w_old, has_short=False, **kwargs):
        kwargs = self.get_config()
        if has_short:
            w_opt = self._opt_long_short(w_old, alpha, sigma, **kwargs)
        else:
            w_opt = self._opt_long(w_old, alpha, sigma, **kwargs)

        return w_opt
