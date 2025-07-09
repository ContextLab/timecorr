# -*- coding: utf-8 -*-
"""
=============================
Simulate subject data
=============================

In this example, we simulate data

"""
# Code source: Lucy Owen
# License: MIT

import os

import matplotlib.pyplot as plt
import seaborn as sns

# load timecorr
import timecorr as tc

# Configure matplotlib for CI environments
if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend in CI

# simulate some data
data, corrs = tc.simulate_data(
    datagen="block", return_corrs=True, set_random_seed=True, S=1, T=100, K=10, B=5
)

# calculate correlations  - returned squareformed
tc_vec_data = tc.timecorr(
    tc.simulate_data(),
    weights_function=tc.gaussian_weights,
    weights_params={"var": 5},
    combine=tc.helpers.corrmean_combine,
)

# convert from vector to matrix format
tc_mat_data = tc.vec2mat(tc_vec_data)


# Helper function to show plots conditionally
def show_plot():
    """Show plot only in interactive environments, not in CI."""
    if not (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")):
        plt.show()


# plot the 3 correlation matrices different timepoints
sns.heatmap(tc_mat_data[:, :, 48])
show_plot()
plt.clf()
sns.heatmap(tc_mat_data[:, :, 50])
show_plot()
plt.clf()
sns.heatmap(tc_mat_data[:, :, 52])
show_plot()
plt.clf()
