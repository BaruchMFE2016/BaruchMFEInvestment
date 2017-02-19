#!/home/quan/anaconda3/bin/python

import math
import numpy as np
from scipy import stats
from statsmodels import robust

def gaussianize(v):

    v_gauss = v.copy()
    v_gauss[~np.isnan(v)] = stats.norm.ppf((stats.rankdata(v[~np.isnan(v)])-0.5)/len(v[~np.isnan(v)]))
    v_gauss[np.isnan(v_gauss)] = 0

    return v_gauss

def mean_rms_zscore(v):

    v_zscore = v.copy()
    v_zscore[~np.isnan(v)] = stats.zscore(v[~np.isnan(v)])
    v_zscore[np.isnan(v_zscore)] = 0
    return v_zscore

def median_mad_zscore(v):
    
    v_zscore = v.copy()
    v_zscore[~np.isnan(v)] = (v[~np.isnan(v)] - np.median(v[~np.isnan(v)]))/robust.mad(v[~np.isnan(v)])
    v_zscore[np.isnan(v_zscore)] = 0
    return v_zscore

if __name__ == "__main__":

    v = np.array([40,15,np.nan,20,25,np.nan,35])
    print(v)
    print(gaussianize(v))
    print(mean_rms_zscore(v))
    print(median_mad_zscore(v))

