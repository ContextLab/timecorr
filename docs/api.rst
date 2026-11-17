.. _api_ref:

.. currentmodule:: timecorr

API Reference
=============

Every function listed here has a page with its parameters, return values and a
runnable example. Unless a section names another module, the functions are
available as ``timecorr.<name>`` (for example ``tc.timecorr`` after
``import timecorr as tc``).

Dynamic correlations
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timecorr
   wcorr
   isfc
   wisfc
   autofc

``timecorr`` builds the kernel, restricts it (``include_timepoints``,
``include_zero``, ``exclude_timepoints``), applies the correlation function
(``cfun``) to every subject, combines the subjects (``combine``) and reduces the
result (``rfun``). ``wcorr`` is the estimator: a kernel-weighted Pearson
correlation in which row ``t`` of the kernel weights the means, covariances and
variances used at timepoint ``t``. Version 0.3.0 changed this estimator, so its
results differ from version 0.2.0 and from Eq. 4 of Owen, Chang & Manning
(2021).

.. code-block:: python

   import numpy as np
   import timecorr as tc

   rng = np.random.default_rng(0)
   data = rng.standard_normal((100, 5))
   corrs = tc.timecorr(data, weights_params={"var": 10})   # (100, 15)

   subjects = [rng.standard_normal((100, 5)) for _ in range(4)]
   isfc = tc.timecorr(subjects, cfun=tc.isfc)               # list of 4 (100, 15) arrays
   group = tc.timecorr(subjects, cfun=tc.isfc, combine=tc.corrmean_combine)  # (100, 15)

Higher-order features
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   features

``features`` computes features of every order through the two paths of Fig. 10
of Owen, Chang & Manning (2021): at each order the analysis kernel is applied to
the previous order's delta features (co-fluctuations), and a delta kernel gives
the co-fluctuations passed to the next order.

.. currentmodule:: timecorr.higher_order

The building blocks are in ``timecorr.higher_order``:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   two_path_step
   estimate_memory_gb

.. code-block:: python

   X = tc.features(subjects, order=2, weights_params={"var": 10})
   # X[0]: the data; X[1], X[2]: one (100, 5) array per subject (PCA-reduced)

.. currentmodule:: timecorr

Kernels
-------

Each kernel function returns a ``T x T`` matrix whose row ``t`` is the kernel
used to estimate at timepoint ``t``. Every kernel function accepts
``direction`` (``None``, ``"backward"`` or ``"forward"``) and ``include_zero``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gaussian_weights
   laplace_weights
   mexican_hat_weights
   t_weights
   boxcar_weights
   uniform_weights
   eye_weights
   one_sided

.. code-block:: python

   w = tc.gaussian_weights(100, {"var": 10}, direction="backward", include_zero=False)
   backward = tc.timecorr(data, weights_params={"var": 10}, include_timepoints="backward")
   # UserWarning: the first 2 timepoints have fewer than 3 past samples; set to 0

Combining subjects
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   null_combine
   corrmean_combine
   mean_combine
   tstat_combine

Dimensionality reduction and smoothing
--------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reduce
   smooth

``reduce`` accepts a hypertools / scikit-learn model name (for example
``"PCA"``, ``"FastICA"``, ``"FactorAnalysis"``, ``"IncrementalPCA"``), a dict
such as ``{"model": "PCA", "params": {"whiten": True}}``, a graph measure
(``"eigenvector_centrality"``, ``"pagerank_centrality"``, ``"strength"``), or a
callable that maps a list of arrays to a list of arrays.

Simulation
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   simulate_data
   simulate_multi_order
   min_features
   carrier_reducer
   carrier_columns
   nested_template

.. code-block:: python

   sim, truth = tc.simulate_data(datagen="block", K=8, T=200, order=2,
                                 return_corrs=True, set_random_seed=0)
   # truth["corrs"]: (200, 10) correlations among the 4 block products

Decoders
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timepoint_decoder
   weighted_timepoint_decoder

.. currentmodule:: timecorr.decoding

Further decoder functions are in ``timecorr.decoding`` (and importable from
``timecorr.helpers``):

.. autosummary::
   :toctree: generated/
   :nosignatures:

   weighted_timepoint_decoder_ec
   optimize_weights
   folding_levels
   folding_levels_ec
   pca_decoder
   decoder
   weight_corrs
   get_xval_assignments

.. currentmodule:: timecorr

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mat2vec
   vec2mat
   format_data

.. currentmodule:: timecorr.helpers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   r2z
   z2r
   plot_weights

.. currentmodule:: timecorr

See Also
--------

* :doc:`tutorials`: tutorials
* :doc:`auto_examples/index`: example gallery
* `GitHub repository <https://github.com/ContextLab/timecorr>`_: source code and issues
* `Owen, Chang & Manning (2021) <https://doi.org/10.1038/s41467-021-25876-x>`_: the paper describing the method
