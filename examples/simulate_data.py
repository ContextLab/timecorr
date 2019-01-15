# -*- coding: utf-8 -*-
"""
=============================
Simulate subject data
=============================

In this example, we simulate data

"""
# Code source: Lucy Owen
# License: MIT

# load timecorr and other packages
import timecorr as tc
import hypertools as hyp
import numpy as np

# load helper functions
from timecorr.helpers import generate_template_data, generate_subject_data, generate_random_covariance_matrix

# create synthetic data with specified paramters
S = 5  #number of subjects
T = 100  #number of timepoints per event
E = 10  #number of events
K = 110  #number of features

#make a timeseries of covariance matrices
covs = np.zeros((E, int((K**2 - K)/2 + K)))
for event in np.arange(E):
    covs[event, :] = generate_random_covariance_matrix(K)

# generate synthetic data from covariance matrices
template_data = generate_template_data(covs, T)
data = []
for s in np.arange(S):
    data.append(generate_subject_data(T, E, template_data))

