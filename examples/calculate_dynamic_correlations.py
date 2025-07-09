# -*- coding: utf-8 -*-
"""
=============================
Calculate dynamic correlations
=============================

In this example, we calculate dynamic correlations

"""
# Code source: Lucy Owen
# License: MIT

import numpy as np

# load timecorr and other packages
import timecorr as tc

S = 1
T = 1000
K = 10
B = 5

# define your weights parameters
width = 100
laplace = {"name": "Laplace", "weights": tc.laplace_weights, "params": {"scale": width}}

# calculate the dynamic correlation of the two datasets

subs_data_2 = tc.simulate_data(
    datagen="ramping", return_corrs=False, set_random_seed=1, S=S, T=T, K=K, B=B
)

subs_data_1 = tc.simulate_data(
    datagen="ramping", return_corrs=False, set_random_seed=2, S=S, T=T, K=K, B=B
)


wcorred_data = tc.wcorr(
    np.array(subs_data_1), np.array(subs_data_2), weights=laplace["weights"](T)
)
