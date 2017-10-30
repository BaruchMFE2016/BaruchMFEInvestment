__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime

from .factors.momentum import momentum, momentum_ewma
from .factors.standard_momentum import standard_momentum
from .factors.momentum_gap import momentum_gap
from .factors.simple_factor import simple_factor, simple_factor_1step_math
from .factors.diff import diff


def get_key(univ, n):
    ks = sorted(univ.keys())
    return ks[n]


def get_val(univ, n):
    return univ[get_key(univ, n)].copy()


def stack(univ, **kwargs):
    ''' stack the dataframes saved in a dict '''
    columns = get_val(univ, 0).columns
    result = pd.concat(univ.values()).sort_values(['date', 'ticker'])
    return result[columns].copy()


def alpha_four_factors(univ):
    '''
    This is a sample on how one should write his/her alpha function
    when doing equity research with this simple platform
    simple_factor means directly taking a column as factor
    momentum is calculating momentum factor with the given params
    code ones own factor calculation methods in the factors module
    if needed.
    '''
    factors = {}
    factors['beta'] = simple_factor(univ, 'beta')
    factors['vol60'] = simple_factor(univ, 'vol60')
    factors['log_market_cap'] = simple_factor_1step_math(univ, 'market_cap', np.log)
    factors['momentum'] = momentum(univ, 4, 52)

    return factors

def alpha_00(univ):
    factors = {}
    factors['beta'] = simple_factor(univ, 'beta')
    factors['vol60'] = simple_factor(univ, 'vol60')
    factors['momentum'] = momentum(stack(univ), 4, 52)

    return factors

def alpha_01(univ):
    factors = {}
    factors['beta'] = simple_factor(univ, 'beta')
    factors['vol60'] = simple_factor(univ, 'vol60')
    factors['standard_momentum'] = standard_momentum(stack(univ), 4, 52)

    return factors


def alpha_02(univ):
    factors = {}
    factors['beta'] = simple_factor(univ, 'beta')
    factors['vol60'] = simple_factor(univ, 'vol60')
    factors['momentum_ewma'] = momentum_ewma(stack(univ), 4, 52)

    return factors


def alpha_wFund_00(univ):
    factors = {}
    univ_table = stack(univ)
    factors['beta']                 = simple_factor(univ, 'beta')
    factors['vol60']                = simple_factor(univ, 'vol60')
    factors['net_debt_to_ebitda']   = simple_factor(univ, 'net_debt_to_ebitda')
    factors['d52_eps_ttm']          = diff(univ_table, 'eps_ttm', 52)
    factors['d52_gross_profit_ttm'] = diff(univ_table, 'gross_profit_ttm', 52)
    factors['momentum_ewma']        = momentum_ewma(univ_table, 4, 52)
    
    return factors
