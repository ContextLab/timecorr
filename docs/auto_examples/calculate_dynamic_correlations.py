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
import hypertools as hyp
import numpy as np

# load helper functions
from timecorr.helpers import wcorr, generate_template_data, generate_subject_data, generate_random_covariance_matrix


# define your weights parameters
width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

# create synthetic data with specified paramters
S = 1  # number of subjects
T = 100  # number of timepoints per event
E = 10  # number of events
K = 100  # number of features

# make a timeseries of covariance matrices
covs = np.zeros((E, int((K**2 - K)/2 + K)))
for event in np.arange(E):
    covs[event, :] = generate_random_covariance_matrix(K)

# generate 2 synthetic datasets from same covariance matrices
template_data = generate_template_data(covs, T)
data_1 = generate_subject_data(T, E, template_data)
data_2 = generate_subject_data(T, E, template_data)

# calculate the dynamic correlation of the two datasets
# total time = T*E
try_wcorr = wcorr(np.array(data_1),  np.array(data_2), weights=laplace['weights'](T*E))