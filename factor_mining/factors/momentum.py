__author__ = 'Derek Qi'

import numpy as np
import pandas as pd


def momentum(univ, head, tail, naming='simple'):
    '''
    Calculates the momentum factor defined as follows
    momentum[t] = (p[t - head] - p[t - tail]) / p[t - tail]
    head and tail are numbers of time periods
    '''

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
        # p[name] = p[name] / p['vol60']
        momentum[ti] = p.ix[:, ['date', 'ticker', name]]

    return dict(zip(datelst, momentum))
