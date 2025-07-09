**timecorr**: A Python toolbox for calculating dynamic correlations and exploring higher order correlations
===========================================================================================================

.. image:: _static/100_factors.gif
   :width: 400pt
   :align: center

Welcome to TimeCorr
====================

The TimeCorr toolbox provides tools for computing and exploring the correlational structure of timeseries data. This powerful package allows you to analyze how correlations between variables change over time, making it ideal for studying dynamic brain networks, market correlations, climate patterns, and other time-varying systems.

Key Features
------------

* **Dynamic Correlations**: Compute moment-by-moment correlations between features
* **Higher-Order Analysis**: Explore correlations between correlations (and beyond)
* **Multi-Subject Analysis**: Compare patterns across different datasets/participants
* **Flexible Kernels**: Gaussian, Laplace, Mexican Hat, and custom weighting functions
* **Dimensionality Reduction**: PCA, ICA, factor analysis, and graph-theoretic measures
* **Statistical Testing**: Built-in permutation testing and multiple comparisons correction

Quick Start Example
-------------------

.. code-block:: python

   import numpy as np
   import timecorr as tc

   # Generate sample data: 100 timepoints, 5 features
   data = np.random.randn(100, 5)

   # Compute dynamic correlations with Gaussian weighting
   correlations = tc.timecorr(data, 
                             weights_function=tc.gaussian_weights, 
                             weights_params={'var': 10})

   print(f"Input shape: {data.shape}")
   print(f"Output shape: {correlations.shape}")  # (100, 15) - vectorized correlation matrices

Multi-Subject Analysis
---------------------

.. code-block:: python

   # Analyze correlations across multiple subjects
   subjects_data = [np.random.randn(100, 5) for _ in range(10)]  # 10 subjects

   # Inter-Subject Functional Connectivity (ISFC)
   isfc_results = tc.timecorr(subjects_data, 
                             cfun=tc.isfc, 
                             weights_function=tc.gaussian_weights)

   # Weighted ISFC for similarity-based averaging
   wisfc_results = tc.timecorr(subjects_data, 
                              cfun=tc.wisfc, 
                              weights_function=tc.gaussian_weights)

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
   auto_examples/index

.. toctree::
   :maxdepth: 2  
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   tutorial/timecorr_notebook

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`