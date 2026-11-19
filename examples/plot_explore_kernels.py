# -*- coding: utf-8 -*-
"""
=============================
Explore kernels
=============================

In this example, we plot the kernels provided by timecorr. Each kernel
function returns a ``T x T`` matrix whose row ``t`` holds the weight given to
every timepoint when the correlations at timepoint ``t`` are estimated; we plot
row ``t = 50`` of each.

The delta kernel (``eye_weights``) puts all of its weight on ``t`` itself and
gives co-fluctuations (products of the z-scored features). The Mexican hat
kernel has negative weights away from ``t``.

"""
# Code source: Lucy Owen
# License: MIT

from matplotlib import pyplot as plt

import timecorr as tc
from timecorr.helpers import plot_weights

# number of timepoints and the timepoint whose kernel is plotted
T = 100
t = 50

# kernel width
width = 10

kernels = [
    ("Delta", tc.eye_weights, tc.eye_params),
    ("Gaussian (var=10)", tc.gaussian_weights, {"var": width}),
    ("Laplace (scale=10)", tc.laplace_weights, {"scale": width}),
    ("Mexican hat (sigma=10)", tc.mexican_hat_weights, {"sigma": width}),
    ("t (df=10)", tc.t_weights, {"df": width}),
    ("Boxcar (width=10)", tc.boxcar_weights, {"width": width}),
]

fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for ax, (name, weights_function, params) in zip(axes.ravel(), kernels):
    plot_weights(weights_function(T, params), t=t, ax=ax, title=name,
                 xlab="Timepoint", ylab="Weight")
fig.suptitle(f"Row {t} of each kernel matrix (plot_weights numbers timepoints from 1)")
fig.tight_layout()
plt.show()
