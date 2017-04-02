__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime

from .factors.momentum import *
from .factors.simple_factor import *


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
	factors['vol10'] = simple_factor(univ, 'vol10')
	factors['market_cap'] = simple_factor(univ, 'market_cap')
	factors['momentum'] = momentum(univ, 6, 25)

	return factors

