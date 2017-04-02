#!/home/quan/anaconda3/bin/python

import math
import numpy as np
import pandas as pd

def get_momentum(df, head, tail):
    
    """ For example, head=1M, tail=12M """

    momentum_df = pd.DataFrame()
    for stock in df:
        momentum_df[stock] = (df[stock].shift(head)-df[stock].shift(tail))/df[stock].shift(tail)

    return momentum_df


