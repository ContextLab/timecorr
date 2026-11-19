# -*- coding: utf-8 -*-
"""
=============================
One-sided kernels
=============================

A kernel can be restricted to the past or to the future of each timepoint.
``direction="backward"`` keeps only the timepoints at or before ``t``;
``direction="forward"`` keeps only those at or after ``t``; ``include_zero=False``
also leaves out ``t`` itself. Every kernel function accepts these arguments,
``tc.one_sided`` applies them to any kernel matrix, and ``tc.timecorr`` applies
them through ``include_timepoints`` ("backward" or "forward") and
``include_zero``.

The first figure shows row ``t = 50`` of a Gaussian kernel in each variant.
The second applies the kernels to simulated data whose correlations change
abruptly between events: the backward estimate changes after each event
boundary, the forward estimate before it, and the two-sided estimate around it.

"""
# Code source: Contextual Dynamics Laboratory
# License: MIT

import warnings

import numpy as np
from matplotlib import pyplot as plt

import timecorr as tc

# --- the kernels -------------------------------------------------------------
T = 100
t = 50
params = {"var": 25}  # standard deviation of 5 timepoints

variants = [
    ("two-sided", {}),
    ("backward", {"direction": "backward"}),
    ("backward, without t", {"direction": "backward", "include_zero": False}),
    ("forward", {"direction": "forward"}),
    ("forward, without t", {"direction": "forward", "include_zero": False}),
]

fig, axes = plt.subplots(1, len(variants), figsize=(14, 3), sharey=True)
x = np.arange(T)
for ax, (name, kwargs) in zip(axes, variants):
    w = tc.gaussian_weights(T, params, **kwargs)
    ax.plot(x, w[t], "k.-", ms=4, lw=1)
    ax.axvline(t, color="0.6", lw=0.8, ls="--")
    ax.set_xlim(t - 15, t + 15)
    ax.set_title(name)
    ax.set_xlabel("Timepoint")
axes[0].set_ylabel(f"Weight (row {t})")
fig.tight_layout()
plt.show()

# --- applied to data with abrupt changes ----------------------------------------
T = 300
data, corrs = tc.simulate_data(
    datagen="block", return_corrs=True, set_random_seed=3, T=T, K=6, B=3
)
truth = tc.vec2mat(corrs)  # (K, K, T)

estimates = {}
with warnings.catch_warnings():
    # a one-sided kernel has fewer than 3 non-zero weights in its first (backward)
    # or last (forward) rows; timecorr warns and sets those timepoints to 0
    warnings.filterwarnings("ignore", message=".*fewer than 3 non-zero kernel weights")
    for name in ["all", "backward", "forward"]:
        estimates[name] = tc.vec2mat(
            tc.timecorr(data, weights_params={"var": 100}, include_timepoints=name)
        )

# the pair of features whose true correlation changes the most across events
iu = np.triu_indices(6, 1)
pair = np.argmax(np.ptp(truth[iu[0], iu[1], :], axis=1))
i, j = iu[0][pair], iu[1][pair]
boundaries = np.flatnonzero(np.diff(truth[i, j, :]) != 0) + 1

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(truth[i, j], color="k", lw=2, label="true correlation")
styles = {"all": ("#2a78d6", "two-sided"), "backward": ("#eb6834", "backward"),
          "forward": ("#1baf7a", "forward")}
for name, (color, label) in styles.items():
    ax.plot(estimates[name][i, j], color=color, lw=1.5, label=label)
for b in boundaries:
    ax.axvline(b, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("Timepoint")
ax.set_ylabel(f"Correlation of features {i} and {j}")
ax.set_ylim(-1, 1)
ax.set_title("Estimates with two-sided and one-sided Gaussian kernels (var=100)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
