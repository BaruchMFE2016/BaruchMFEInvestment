__author__ = 'Derek Qi'
'''
Base class of optimizer
'''


class Optimizer(object):
    def __init__(self, **kwargs):
        self.param_config == kwargs

    def get_config(self):
        return self.param_config

    def _opt_long(self, **kwargs):
        assert False

    def _opt_long_short(self, **kwargs):
        assert False

    def opt(self, **kwargs):
        assert False