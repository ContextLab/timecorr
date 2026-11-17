# -*- coding: utf-8 -*-
"""
=======================================
Optimized weights by level for decoding
=======================================

In this example, we load the ``weights`` example dataset from hypertools (36
subjects, 300 timepoints, 100 features) and decode timepoints from a weighted
mix of the decoding matrices of orders 0, 1 and 2 with
``weighted_timepoint_decoder``.

In each fold, one group of subjects is held out. The other subjects form the
training group, which is split in two halves; the weights of the orders
(non-negative, summing to one) are fit to the half-versus-half decoding
matrices and then used to decode the held-out group. Orders are added one at a
time, so the results show the decoding after mixing order 0, orders 0-1, and
orders 0-2.

"""
# Code source: Lucy Owen
# License: MIT

import hypertools as hyp
import numpy as np

import timecorr as tc

# load example data: a list with one (timepoints, features) array per subject
data = hyp.load("weights")
print(f"{len(data)} subjects, each {data[0].shape} (timepoints, features)")

# define the kernel
width = 10
laplace = {"name": "Laplace", "weights": tc.laplace_weights, "params": {"scale": width}}

# the group splits are random: seed the global NumPy random state to repeat them
np.random.seed(0)

# corrmean_combine averages the groups' correlations (orders 1 and 2) in Fisher
# z space; order-0 activations are always averaged with mean_combine
results = tc.weighted_timepoint_decoder(
    data,
    level=2,
    combine=tc.corrmean_combine,
    cfun=tc.isfc,
    rfun="eigenvector_centrality",
    weights_fun=laplace["weights"],
    weights_params=laplace["params"],
)

# one row per fold and step; level_<k> is the weight given to order k
print(results)
print(results.groupby("level")[["accuracy", "error", "level_0", "level_1", "level_2"]].mean())
