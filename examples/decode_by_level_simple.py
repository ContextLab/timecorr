# -*- coding: utf-8 -*-
"""
================================
Decode by level (simulated data)
================================

This example decodes timepoints from features of orders 0 and 1 in simulated
data. ``simulate_data`` gives every subject the same block-structured template
of correlations (5 events), and draws each subject's samples independently.
The activations are therefore not shared across subjects, so order 0 decodes
at chance. The correlations are shared, so order 1 decodes the event each
timepoint belongs to. Because the samples are independent across subjects,
the order-1 features here are each subject's own dynamic correlations
(``cfun=tc.autofc``), averaged within each group; inter-subject correlations
(``tc.isfc``) would be zero in expectation for these data.

"""
# Code source: Lucy Owen
# License: MIT

import numpy as np

import timecorr as tc

# 8 subjects who share one template: 60 timepoints, 10 features, 5 events
data = tc.simulate_data(datagen="block", S=8, T=60, K=10, B=5, set_random_seed=0)
print(f"{len(data)} subjects, each {data[0].shape} (timepoints, features)")

# the group split is random: seed the global NumPy random state to repeat it
np.random.seed(0)

results = tc.timepoint_decoder(
    data,
    level=[0, 1],
    cfun=tc.autofc,
    weights_fun=tc.laplace_weights,
    weights_params={"scale": 5},
)

# accuracy: fraction of timepoints decoded exactly (chance 1/60);
# error: mean absolute decoding error as a fraction of the 60 timepoints
print(results.groupby("level")[["accuracy", "error", "rank"]].mean())
