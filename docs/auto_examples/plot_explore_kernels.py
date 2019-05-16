# -*- coding: utf-8 -*-
"""
=============================
Explore kernels
=============================

In this example, we plot the kernel options provided.

"""
# Code source: Lucy Owen
# License: MIT

# load
import timecorr as tc
import numpy as np
from matplotlib import pyplot as plt

# load helper functions
from timecorr.helpers import plot_weights

# define number of timepoints
T = 100

# define width
width = 10

# define functions
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
delta = {'name': '$\delta$', 'weights': tc.eye_weights, 'params': tc.eye_params}
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
mexican_hat = {'name': 'Mexican hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}

# plot delta
plot_weights(delta['weights'](T), title='Delta')
plt.show()
plt.clf()

# plot gaussian
plot_weights(gaussian['weights'](T), title='Gaussian')
plt.show()
plt.clf()

# plot laplace
plot_weights(laplace['weights'](T), title='Laplace')
plt.show()
plt.clf()

# plot mexican hat
plot_weights(mexican_hat['weights'](T), title='Mexican hat')
plt.show()
plt.clf()