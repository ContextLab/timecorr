# -*- coding: utf-8 -*-
"""
=============================
Decode by level
=============================

In this example, we load the ``weights`` example dataset from hypertools (36
subjects, 300 timepoints, 100 features) and decode the timepoint from features
of orders 0, 1 and 2 with ``timepoint_decoder``.

The subjects are split into two groups. For each order, each group's features
are averaged across the group's subjects, and each timepoint of one group is
assigned the timepoint of the other group whose features are most correlated
with it. Order 0 is the data; order ``k >= 1`` is the dynamic inter-subject
correlations (ISFC) of the order ``k - 1`` co-fluctuations, reduced here to one
value per feature with eigenvector centrality.

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

# the group split is random: seed the global NumPy random state to repeat it
np.random.seed(0)

# decode with features of orders 0, 1 and 2 (every order up to the highest is
# computed; the results for each listed order are returned)
# combine takes one function per order: order 0 holds activations, which are
# averaged as they are (mean_combine); orders 1 and 2 hold correlations, which
# are averaged in Fisher z space (corrmean_combine)
results = tc.timepoint_decoder(
    data,
    level=[0, 1, 2],
    combine=[tc.mean_combine, tc.corrmean_combine, tc.corrmean_combine],
    cfun=tc.isfc,
    rfun="eigenvector_centrality",
    weights_fun=laplace["weights"],
    weights_params=laplace["params"],
)

# one row per fold and order; chance accuracy is 1 / 300
print(results)
print(results.groupby("level")[["accuracy", "error", "rank"]].mean())
