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
import os

# load helper functions
from timecorr.helpers import plot_weights

# Configure matplotlib for CI environments
if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend in CI

# define number of timepoints
T = 100

# define width
width = 10

# define functions
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
delta = {'name': r'$\delta$', 'weights': tc.eye_weights, 'params': tc.eye_params}
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
mexican_hat = {'name': 'Mexican hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}

# Helper function to show plots conditionally
def show_plot():
    """Show plot only in interactive environments, not in CI."""
    if not (os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')):
        plt.show()

# plot delta
plot_weights(delta['weights'](T), title='Delta')
show_plot()
plt.clf()

# plot gaussian
plot_weights(gaussian['weights'](T), title='Gaussian')
show_plot()
plt.clf()

# plot laplace
plot_weights(laplace['weights'](T), title='Laplace')
show_plot()
plt.clf()

# plot mexican hat
plot_weights(mexican_hat['weights'](T), title='Mexican hat')
show_plot()
plt.clf()