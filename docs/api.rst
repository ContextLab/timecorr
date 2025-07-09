.. _api_ref:

.. currentmodule:: timecorr

API Reference
=============

This page contains the complete API reference for all timecorr functions. Each function includes detailed parameter descriptions, return values, and usage examples.

Core Functions
--------------

The main functions for computing dynamic correlations and exploring temporal structure.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timecorr

Primary Analysis Function
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: timecorr

**Quick Example:**

.. code-block:: python

   import numpy as np
   import timecorr as tc
   
   # Basic usage
   data = np.random.randn(100, 5)
   correlations = tc.timecorr(data, weights_function=tc.gaussian_weights)

Correlation Functions (cfun)
----------------------------

Functions for computing correlations within or across datasets.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   isfc
   wisfc
   autofc

Inter-Subject Functional Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isfc

.. autofunction:: wisfc

.. autofunction:: autofc

**Examples:**

.. code-block:: python

   # Multi-subject analysis
   subjects = [np.random.randn(100, 5) for _ in range(10)]
   
   # ISFC: Each subject vs. average of others
   isfc_result = tc.timecorr(subjects, cfun=tc.isfc)
   
   # WISFC: Weighted similarity-based averaging
   wisfc_result = tc.timecorr(subjects, cfun=tc.wisfc)

Weighting Functions
-------------------

Kernel functions that control temporal smoothing and local correlation computation.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gaussian_weights
   laplace_weights
   mexican_hat_weights
   eye_weights

**Examples:**

.. code-block:: python

   # Different kernel types
   gaussian_corr = tc.timecorr(data, weights_function=tc.gaussian_weights, 
                              weights_params={'var': 10})
   
   laplace_corr = tc.timecorr(data, weights_function=tc.laplace_weights,
                             weights_params={'scale': 5})

Dimensionality Reduction
------------------------

Functions for reducing correlation dimensionality and exploring higher-order structure.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reduce
   smooth

**String-based reduction options:**
   * ``'PCA'`` - Principal Component Analysis
   * ``'ICA'`` - Independent Component Analysis  
   * ``'FactorAnalysis'`` - Factor Analysis
   * ``'pagerank_centrality'`` - PageRank centrality
   * ``'eigenvector_centrality'`` - Eigenvector centrality

**Example:**

.. code-block:: python

   # Higher-order correlations with PCA reduction
   higher_order = tc.timecorr(data, rfun='PCA')

Data Simulation
---------------

Functions for generating synthetic datasets for testing and validation.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   simulate_data

**Example:**

.. code-block:: python

   # Generate synthetic data
   data = tc.simulate_data(datagen='block', T=100, K=5, B=10)

Utility Functions
-----------------

Helper functions for data manipulation and analysis.

Matrix Operations
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mat2vec
   vec2mat
   wcorr

**Examples:**

.. code-block:: python

   # Convert between matrix and vector formats
   correlation_matrix = np.random.randn(5, 5)
   correlation_vector = tc.mat2vec(correlation_matrix)
   reconstructed_matrix = tc.vec2mat(correlation_vector)

Statistical Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   r2z
   z2r

**Example:**

.. code-block:: python

   # Fisher z-transform for correlation statistics
   correlations = np.array([0.3, 0.5, 0.8])
   z_values = tc.r2z(correlations)
   back_to_r = tc.z2r(z_values)

Decoder Functions
-----------------

Functions for decoding and cross-validation analysis.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timepoint_decoder
   weighted_timepoint_decoder
   pca_decoder

**Example:**

.. code-block:: python

   # Decode temporal patterns
   subjects_data = [np.random.randn(100, 10) for _ in range(5)]
   accuracy = tc.timepoint_decoder(subjects_data, nfolds=3)

Visualization
-------------

Functions for plotting and visualizing results.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_weights

**Example:**

.. code-block:: python

   # Visualize kernel weights
   weights = tc.gaussian_weights(50, var=10)
   tc.plot_weights(weights)

Function Parameters
-------------------

Common Parameters
^^^^^^^^^^^^^^^^^

Most timecorr functions accept these common parameters:

* **data** : array-like or list of arrays
    Input timeseries data with shape (timepoints, features)

* **weights_function** : callable
    Kernel function for temporal weighting (e.g., ``gaussian_weights``)

* **weights_params** : dict
    Parameters for the weighting function

* **cfun** : callable or None
    Correlation function (``isfc``, ``wisfc``, ``autofc``, or None for within-subject)

* **rfun** : callable, str, or None
    Dimensionality reduction function or method name

* **combine** : callable
    Function for combining results across subjects/datasets

See Also
--------

* :doc:`tutorials` : Step-by-step tutorials and examples
* :doc:`auto_examples/index` : Gallery of example applications
* `GitHub Repository <https://github.com/ContextLab/timecorr>`_ : Source code and issues
* `Research Paper <https://doi.org/10.1038/s41467-021-25876-x>`_ : Theoretical background