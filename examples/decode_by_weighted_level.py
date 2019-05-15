# -*- coding: utf-8 -*-
"""
=======================================
Optimized weights by level for decoding
=======================================

In this example, we load in some example data, and find optimal level weights for decoding.

"""
# Code source: Lucy Owen
# License: MIT

# load timecorr and other packages
import timecorr as tc
import hypertools as hyp
import numpy as np

# load example data
data = hyp.load('weights').get_data()

# define your weights parameters
width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

# set your number of levels
# if integer, returns decoding accuracy, error, and rank for specified level
level = 2


# run timecorr with specified functions for calculating correlations, as well as combining and reducing
results = tc.weighted_timepoint_decoder(np.array(data), level=level, combine=tc.corrmean_combine,
                               cfun=tc.isfc, rfun='eigenvector_centrality', weights_fun=laplace['weights'],
                               weights_params=laplace['params'])


# returns optimal weighting for mu for all levels up to 2 as well as decoding results for each fold
print(results)