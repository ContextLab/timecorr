**timecorr**: A Python toolbox for calculating dynamic correlations and exploring higher order correlations
===========================================================================================================

.. image:: _static/100_factors.gif
   :width: 400pt
   :align: center

Welcome to timecorr
===================

The timecorr toolbox computes and explores the correlational structure of
timeseries data. It estimates how correlations between features change from
moment to moment (dynamic correlations), and it computes higher-order dynamic
correlations: dynamic correlations of features derived from lower-order
dynamic correlations. The toolbox accompanies Owen, Chang & Manning (2021),
*Nature Communications* 12, 5728 (`doi:10.1038/s41467-021-25876-x
<https://doi.org/10.1038/s41467-021-25876-x>`_).

Version 0.3.0 changed the correlation estimator (a kernel-weighted Pearson
correlation), so its numbers differ from version 0.2.0; the changes are listed
in the `changelog <https://github.com/ContextLab/timecorr/blob/master/CHANGELOG.md>`_.

Features
--------

* **Dynamic correlations**: kernel-weighted correlations at every timepoint,
  within one dataset or across subjects (inter-subject functional connectivity).
* **Kernels**: Gaussian, Laplace, Mexican hat, t, boxcar, uniform and delta,
  each two-sided or one-sided (past or future timepoints only).
* **Higher-order features**: ``tc.features`` computes features of every order
  through the two paths of Fig. 10 of the paper.
* **Dimensionality reduction**: hypertools / scikit-learn models, graph
  measures, or a function of your own.
* **Simulation**: data with known dynamic correlations, including structure
  embedded at exactly one chosen correlation order.
* **Decoding**: timepoint decoders that compare features of each order across
  groups of subjects.

Quick start
-----------

New to timecorr? The :doc:`quickstart tutorial <tutorial/quickstart>` computes
dynamic correlations, combines subjects and computes higher-order features, and
the :doc:`tutorial gallery <tutorials>` shows every part of the toolbox.
Four tutorials use real data: :doc:`brain networks during story listening
<tutorial/htfa_network_animations>`, :doc:`the stock market <tutorial/stocks>`,
:doc:`causal features of stock returns (no look-ahead) <tutorial/causal>`
and :doc:`trajectories of features at orders 1 to 4 <tutorial/trajectories>`.

.. code-block:: python

   import numpy as np
   import timecorr as tc

   rng = np.random.default_rng(0)
   data = rng.standard_normal((100, 5))  # 100 timepoints, 5 features

   corrs = tc.timecorr(data, weights_function=tc.gaussian_weights,
                       weights_params={"var": 10})
   print(corrs.shape)  # (100, 15): one vectorized 5 x 5 matrix per timepoint

Several subjects and higher orders
----------------------------------

.. code-block:: python

   subjects = [rng.standard_normal((100, 5)) for _ in range(10)]

   # each subject versus the mean of the others, averaged in Fisher z space
   group = tc.timecorr(subjects, cfun=tc.isfc, combine=tc.corrmean_combine,
                       weights_params={"var": 10})
   print(group.shape)  # (100, 15)

   # features of orders 0 to 3 with the original two-path pipeline of the paper
   # (co-fluctuations feed the next order), each reduced to 5 features per subject
   X = tc.features(subjects, order=3, weights_params={"var": 10},
                   rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
   print(len(X), X[3][0].shape)  # 4 (100, 5)

   # the defaults: a Gaussian kernel and squared-correlation summaries on the
   # path that feeds the next order (108 nodes in 18 modules)
   nested, truth = tc.simulate_nested(S=2, T=1000, set_random_seed=0)
   X = tc.features(nested, order=2, cfun=tc.autofc, weights_params={"var": 33 ** 2})
   print(X[1][0].shape, X[2][0].shape)  # (1000, 108) (1000, 36)

Installation
------------

Install the latest stable version using pip:

.. code-block:: bash

   pip install timecorr

Or for the latest development version:

.. code-block:: bash

   pip install --upgrade git+https://github.com/ContextLab/timecorr.git

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`