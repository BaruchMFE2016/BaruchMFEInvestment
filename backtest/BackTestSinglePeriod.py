'''
Abstract Base class for single period backtesting
'''
__author__ = 'Derek Qi'

class BackTestSinglePeriod(object):
    '''
    This serves as an abstract base class
    Each strategy should have its own version of single period backtest caclculator derived from this class
    '''

    def get_config(self):
        assert False

    def get_func_name(self):
        assert False

    def calc_pnl(self, univ, factor_exp_mat, t, **kwargs):
        assert False
	