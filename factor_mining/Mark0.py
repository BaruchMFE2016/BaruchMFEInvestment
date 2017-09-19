__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime

from .factors.momentum import momentum
from .factors.simple_factor import simple_factor, simple_factor_1step_math


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
    factors['momentum'] = momentum(univ, 6, 26)

    return factors