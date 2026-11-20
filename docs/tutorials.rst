.. _tutorial:

Tutorials
=========

The tutorials are Jupyter notebooks in the ``docs/tutorial/`` folder of the
repository. The pages below show them with the output of a full run. Each
tutorial has an "Open in Colab" link at the top that opens the notebook in
Google Colab, where its first code cell installs timecorr.

.. toctree::
   :maxdepth: 1

   tutorial/timecorr_notebook
   tutorial/synthetic_data_tutorial
   tutorial/higher_order_tutorial
   tutorial/event_kernels_tutorial
   tutorial/applications_tutorial
   tutorial/htfa_network_animations

Using timecorr
--------------

``timecorr_notebook.ipynb``: simulating data, computing dynamic correlations
with different kernels, the vectorized format (``mat2vec`` / ``vec2mat``),
several subjects (ISFC), and reducing correlations with ``rfun``.

Synthetic data
--------------

``synthetic_data_tutorial.ipynb``: the four ways a simulated correlation
template can change over time (constant, block, ramping, random); the kernels,
and how their shape and width affect recovery of the template; the delta kernel
and co-fluctuations; several subjects who share a template; and a test, across
subjects, of each timepoint's correlation.

Higher-order correlations
-------------------------

``higher_order_tutorial.ipynb``: the two paths of Fig. 10 of Owen, Chang &
Manning (2021) and ``tc.features`` with the original pipeline's settings
(co-fluctuations on the path that feeds the next order); one-sided kernels; simulating data whose
structure exists only at a chosen order and recovering it; and the limits of the
approach (the number of features an order needs, the memory of unreduced runs,
and how much of the structure variance-based reduction keeps at orders 2 and 3,
as measured there); and the defaults of ``tc.features`` (a Gaussian kernel and
squared-correlation summaries on the path that feeds the next order) on
``simulate_nested`` data.

Event-restricted kernels
------------------------

``event_kernels_tutorial.ipynb``: what restricting a kernel to events (blocks
of consecutive timepoints) does to it; passing boundaries to ``timecorr``;
segmenting synthetic stimulus embeddings with ``tc.segment`` (ordered events,
with the number of events given or chosen automatically, and recurring states);
recovery of block-wise correlations in synthetic brain data with the
restricted kernel, compared with the unrestricted kernel and with wrong
boundaries; per-subject events and cross-subject correlations; different
events at each order in ``tc.features``; and events in the timepoint decoders.

Applications
------------

``applications_tutorial.ipynb``: dynamic correlations in simulated data from
four domains (brain networks, financial returns, climate variables and social
groups), at more than one kernel width.

Brain-network animations with HTFA
----------------------------------

``htfa_network_animations.ipynb``: features of orders 1 to 4 (the defaults
of ``tc.features``) of the hierarchical topographic factor analysis (HTFA)
node weights of the story-listening fMRI data analyzed by Owen, Chang &
Manning (2021), downloaded from the data archive of the paper's repository;
the record of each step (``return_info=True``); which nodes each order's
features belong to; and an animation of each order on the brain, drawn with
the `htfa <https://htfa.readthedocs.io/en/latest/>`_ toolbox.

Running the tutorials
---------------------

From the root of the repository:

.. code-block:: bash

   pip install -e ".[docs]"
   jupyter notebook docs/tutorial/

Each notebook runs top to bottom in under about two minutes, except
``htfa_network_animations.ipynb``: it downloads a 460 MB archive on its first
run, and its last full run took about 7 minutes after the download.
