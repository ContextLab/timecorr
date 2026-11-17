:orphan:

.. _examples-index:

Gallery of Examples
===================

Short scripts that use timecorr. The scripts whose names start with ``plot_``
are run when the documentation is built, and their figures are shown here;
the decoding scripts take up to about a minute each and are shown without output.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example decodes timepoints from features of orders 0 and 1 in simulated data. simulate_data gives every subject the same block-structured template of correlations (5 events), and draws each subject&#x27;s samples independently. The activations are therefore not shared across subjects, so order 0 decodes at chance. The correlations are shared, so order 1 decodes the event each timepoint belongs to. Because the samples are independent across subjects, the order-1 features here are each subject&#x27;s own dynamic correlations (``cfun=tc.autofc``), averaged within each group; inter-subject correlations (``tc.isfc``) would be zero in expectation for these data.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_decode_by_level_simple_thumb.png
    :alt:

  :doc:`/auto_examples/decode_by_level_simple`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Decode by level (simulated data)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we plot the kernels provided by timecorr. Each kernel function returns a T x T matrix whose row t holds the weight given to every timepoint when the correlations at timepoint t are estimated; we plot row t = 50 of each.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_explore_kernels_thumb.png
    :alt:

  :doc:`/auto_examples/plot_explore_kernels`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Explore kernels</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we load the weights example dataset from hypertools (36 subjects, 300 timepoints, 100 features) and decode timepoints from a weighted mix of the decoding matrices of orders 0, 1 and 2 with weighted_timepoint_decoder.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_decode_by_weighted_level_thumb.png
    :alt:

  :doc:`/auto_examples/decode_by_weighted_level`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optimized weights by level for decoding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we simulate data whose correlations change over time, compute their dynamic correlations with timecorr, and compare the estimates with the true correlations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_calculate_dynamic_correlations_thumb.png
    :alt:

  :doc:`/auto_examples/calculate_dynamic_correlations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculate dynamic correlations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we load the weights example dataset from hypertools (36 subjects, 300 timepoints, 100 features) and decode the timepoint from features of orders 0, 1 and 2 with timepoint_decoder.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_decode_by_level_thumb.png
    :alt:

  :doc:`/auto_examples/decode_by_level`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Decode by level</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we simulate data whose correlation matrix changes between events (``datagen=&quot;block&quot;``), estimate the dynamic correlations with timecorr, and plot the true and estimated correlation matrices at one timepoint in the middle of each of three events.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_simulate_data_thumb.png
    :alt:

  :doc:`/auto_examples/plot_simulate_data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Simulate subject data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A kernel can be restricted to the past or to the future of each timepoint. direction=&quot;backward&quot; keeps only the timepoints at or before t; direction=&quot;forward&quot; keeps only those at or after t; include_zero=False also leaves out t itself. Every kernel function accepts these arguments, tc.one_sided applies them to any kernel matrix, and tc.timecorr applies them through include_timepoints (&quot;backward&quot; or &quot;forward&quot;) and include_zero.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_one_sided_kernels_thumb.png
    :alt:

  :doc:`/auto_examples/plot_one_sided_kernels`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">One-sided kernels</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="simulate_data(order=n) embeds a known 4 x 4 dynamic correlation matrix (the template) at exactly order n: the features form 4 blocks of 2(n-1) columns, and only the product of all of a block&#x27;s columns carries the template. Products of fewer columns have mean zero, so orders below n hold no structure.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_higher_order_recovery_thumb.png
    :alt:

  :doc:`/auto_examples/plot_higher_order_recovery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recover structure at a chosen order</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/decode_by_level_simple
   /auto_examples/plot_explore_kernels
   /auto_examples/decode_by_weighted_level
   /auto_examples/calculate_dynamic_correlations
   /auto_examples/decode_by_level
   /auto_examples/plot_simulate_data
   /auto_examples/plot_one_sided_kernels
   /auto_examples/plot_higher_order_recovery


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
