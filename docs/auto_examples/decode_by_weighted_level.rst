
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/decode_by_weighted_level.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_decode_by_weighted_level.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_decode_by_weighted_level.py:


=======================================
Optimized weights by level for decoding
=======================================

In this example, we load in some example data, and find optimal level weights for decoding.

NOTE: This example currently has compatibility issues with the weighted_timepoint_decoder function.
For a working example, please see the enhanced version in docs/auto_examples/decode_by_weighted_level.py

.. GENERATED FROM PYTHON SOURCE LINES 13-72

.. code-block:: Python

    # Code source: Lucy Owen
    # License: MIT

    import hypertools as hyp
    import numpy as np

    # load timecorr and other packages
    import timecorr as tc

    print("Weighted Timepoint Decoding Example")
    print("=" * 40)
    print("NOTE: This example currently has compatibility issues.")
    print(
        "Please see docs/auto_examples/decode_by_weighted_level.py for a working version."
    )
    print("=" * 40)

    # load example data
    data = hyp.load("weights").get_data()

    # Convert to numpy array format required by weighted_timepoint_decoder
    data_array = np.array(data)
    print(f"Data shape: {data_array.shape} (subjects, timepoints, features)")

    # define your weights parameters
    width = 10
    laplace = {"name": "Laplace", "weights": tc.laplace_weights, "params": {"scale": width}}

    # set your number of levels
    # if integer, returns decoding accuracy, error, and rank for specified level
    level = 2

    print(f"\nAttempting weighted timepoint decoding at level {level}...")

    try:
        # run timecorr with specified functions for calculating correlations, as well as combining and reducing
        results = tc.weighted_timepoint_decoder(
            data_array,
            level=level,
            combine=tc.corrmean_combine,
            cfun=tc.isfc,
            rfun="eigenvector_centrality",
            weights_fun=laplace["weights"],
            weights_params=laplace["params"],
        )

        # returns optimal weighting for mu for all levels up to 2 as well as decoding results for each fold
        print("✓ SUCCESS: Weighted decoding results:")
        print(results)

    except Exception as e:
        print(f"✗ ERROR: {e}")
        print("This function has compatibility issues with the current version.")

    print("\n" + "=" * 60)
    print(
        "RECOMMENDATION: Use the enhanced version in docs/auto_examples/decode_by_weighted_level.py"
    )
    print("which uses synthetic data and includes comprehensive error handling.")


.. _sphx_glr_download_auto_examples_decode_by_weighted_level.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: decode_by_weighted_level.ipynb <decode_by_weighted_level.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: decode_by_weighted_level.py <decode_by_weighted_level.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: decode_by_weighted_level.zip <decode_by_weighted_level.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
