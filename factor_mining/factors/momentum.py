__author__ = 'Derek Qi'

import numpy as np
import pandas as pd
from datetime import datetime


''' def momentum(univ, head, tail, naming='simple'):
    
    # Calculates the momentum factor defined as follows
    # momentum[t] = (p[t - head] - p[t - tail]) / p[t - tail]
    # head and tail are numbers of time periods

    name = 'momentum'
    if naming == 'full':
        name += '_%s_%s' % (head, tail)
        

    datelst = sorted(univ.keys())
    N_T = len(datelst)

    momentum = [0] * N_T
    for ti in range(N_T):
        t = datelst[ti]
        u = univ[t].copy()
        if ti < tail:
            momentum[ti] = univ[datelst[ti]].ix[:, ['date', 'ticker']]
            momentum[ti][name] = np.nan
            continue

        p_head = univ[datelst[ti - head]].ix[:, ['date', 'ticker', 'price']]
        p_tail = univ[datelst[ti - tail]].ix[:, ['date', 'ticker', 'price']]
        p = pd.merge(p_head, p_tail,
                     how='inner', on='ticker', suffixes=('_h', '_t'))

        p[name] = (p.price_h - p.price_t) / p.price_t
        p['date'] = p['date_h'].tolist()
        p = pd.merge(u.ix[:, ['date', 'ticker', 'vol60']],
                     p.ix[:, ['ticker', name]], on='ticker', how='inner')
        momentum[ti] = p.ix[:, ['date', 'ticker', name]]

    return dict(zip(datelst, momentum))
 '''
def _mmt(log_return, head, tail):
    assert head < tail, "head %d is greater than or equal to tail" % (head, tail)
    y = np.array([0] * head + [1] * (tail - head))
    y = y / np.sum(y)
    rmmt = np.convolve(log_return, y)
    return rmmt[:-tail+1]


def momentum(univ_table, head, tail, naming='simple', **kwargs):
    '''
    mmt_t = \sum_{i=head}^{tail} log return_{t-i}
    '''
    name = 'momentum'
    if naming == 'full':
        name += '_%s_%s' % (head, tail)
    univ_table[name] = np
    mmt_dict = {}
    datelst = np.unique(univ_table['date'])

    def _mmt_single_name(table):
        lr = np.diff(np.log(table['price'])) # log return series
        lr = np.insert(lr, 0, 0)
        mmt = _mmt(lr, head, tail)
        table.loc[:, name] = mmt
        if kwargs.get('weight') == 'ewma':
            halflife = kwargs.get('halflife') or 13
            table[name] = pd.ewma(table[name], halflife=halflife, ignore_na=True)       
        return table
    
    univ_table = univ_table.groupby('ticker').apply(_mmt_single_name)

    for t in datelst:
        table = univ_table.loc[univ_table.date == t, ['date', 'ticker', name]].copy()
        table.dropna(inplace = True)
        if type(t) == str:
            t = datetime.strptime(t, '%Y-%m-%d') #XXX this is a temporary fix
        mmt_dict[t] = table
    return mmt_dict
