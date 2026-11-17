# -*- coding: utf-8 -*-
"""
=============================
Simulate subject data
=============================

In this example, we simulate data whose correlation matrix changes between
events (``datagen="block"``), estimate the dynamic correlations with
``timecorr``, and plot the true and estimated correlation matrices at one
timepoint in the middle of each of three events.

"""
# Code source: Lucy Owen
# License: MIT

import numpy as np
from matplotlib import pyplot as plt

import timecorr as tc

# simulate 300 timepoints of 10 features with 3 events; corrs holds the true
# (vectorized) correlation matrix of every timepoint
T, K, B = 300, 10, 3
data, corrs = tc.simulate_data(
    datagen="block", return_corrs=True, set_random_seed=1, S=1, T=T, K=K, B=B
)

# estimate the dynamic correlations
estimated = tc.timecorr(
    data, weights_function=tc.gaussian_weights, weights_params={"var": 100}
)

# convert from vectorized to matrix format: (K, K, T)
true_mats = tc.vec2mat(corrs)
est_mats = tc.vec2mat(estimated)

r = np.corrcoef(estimated[:, K:].ravel(), corrs[:, K:].ravel())[0, 1]
print(f"correlation between estimated and true off-diagonal entries: {r:.2f}")

timepoints = [50, 150, 250]  # one per event
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
for col, t in enumerate(timepoints):
    for row, (mats, name) in enumerate([(true_mats, "True"), (est_mats, "Estimated")]):
        ax = axes[row, col]
        im = ax.imshow(mats[:, :, t], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(0, K, 3))
        ax.set_yticks(range(0, K, 3))
        if row == 0:
            ax.set_title(f"Timepoint {t}")
        ax.set_xlabel("Feature")
        if col == 0:
            ax.set_ylabel(f"{name}\nfeature")
fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
plt.show()
