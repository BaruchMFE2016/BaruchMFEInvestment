__author__ = 'Derek Qi'
'''
simple weighted optimiaztion, pick large alphas with absolute values and assign weights accordingly.abs
DO NOT TREAT THIS AS A RIGOROUS OPTIMIAZTION MODEL!
This is only used to examine the validity of alpha model
'''

import numpy as np
from .Optimizer import Optimizer

class SimpleOptimizer(Optimizer):
    def __init__(self, **kwargs):
        self.param_config = kwargs
        self.param_config['n'] = kwargs.get('n') or 10

    def _opt_long(self, alpha, n):
        long_id = alpha.argsort()[-n:]
        w = np.zeros(alpha.shape[0])
        w[long_id] += 1/n
        return w

    def _opt_long_short(self, alpha, n):
        aas = alpha.argsort()
        long_id = aas[-n:]
        short_id = aas[:n]
        w = np.zeros(alpha.shape[0])
        w[long_id] += 1/n
        w[short_id] -= 1/n
        return w

    def opt(self, alpha, has_short=False, *args, **kwargs):
        kwargs = self.get_config()
        n = kwargs.get('n') or 10
        if has_short:
            w_opt = self._opt_long_short(alpha, n)
        else:
            w_opt = self._opt_long(alpha, n)

        return w_opt.reshape(-1, 1)