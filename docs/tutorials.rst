.. _tutorial:

Tutorials
=========

The tutorials are Jupyter notebooks in the ``docs/tutorial/`` folder of the
repository. The pages below show them with the output of a full run.

.. toctree::
   :maxdepth: 1

   tutorial/timecorr_notebook
   tutorial/synthetic_data_tutorial
   tutorial/higher_order_tutorial
   tutorial/applications_tutorial

Using timecorr
--------------

``timecorr_notebook.ipynb``: simulating data, computing dynamic correlations
with different kernels, the vectorized format (``mat2vec`` / ``vec2mat``),
several subjects (ISFC), and reducing correlations with ``rfun``.

Synthetic data
--------------

``synthetic_data_tutorial.ipynb``: the four simulated dataset types (random,
constant, block, ramping), the kernels, dynamic correlations estimated with each
kernel and compared with the true correlations, and ISFC across subjects who
share a template.

Higher-order correlations
-------------------------

``higher_order_tutorial.ipynb``: the two paths of Fig. 10 of Owen, Chang &
Manning (2021) and ``tc.features``; one-sided kernels; simulating data whose
structure exists only at a chosen order and recovering it; and the limits of the
approach (the number of features an order needs, the memory of unreduced runs,
and why variance-based reduction misses structure that exists only at order 3 or
higher).

Applications
------------

``applications_tutorial.ipynb``: dynamic correlations in simulated data from
four domains (brain networks, financial returns, climate variables and social
groups), at more than one kernel width.

Running the tutorials
---------------------

From the root of the repository:

.. code-block:: bash

   pip install -e ".[docs]"
   jupyter notebook docs/tutorial/

Each notebook runs top to bottom in under about two minutes.
