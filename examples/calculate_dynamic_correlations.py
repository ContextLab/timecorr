# -*- coding: utf-8 -*-
"""
=============================
Calculate dynamic correlations
=============================

In this example, we calculate dynamic correlations

"""
# Code source: Lucy Owen
# License: MIT

# load timecorr and other packages
import timecorr as tc
import numpy as np


S = 2
T = 1000
K = 10
B = 5

# define your weights parameters
width = 100
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

# calculate the dynamic correlation of the two datasets

subs_data = tc.simulate_data(datagen='block', return_corrs=False, set_random_seed=True, S=S, T=T, K=K, B=B)


wcorred_data = tc.wcorr(np.array(subs_data[0]),  np.array(subs_data[1]), weights=laplace['weights'](T))