.. _tutorial:

Tutorials
=========

The tutorials are Jupyter notebooks in the ``docs/tutorial/`` folder of the
repository, shown here with the output of a full run. Click a card to open a
tutorial. Each tutorial has an "Open in Colab" link at the top; on Google
Colab its first code cell installs timecorr. Start with the quickstart.

.. nbgallery::

   tutorial/quickstart
   tutorial/kernels
   tutorial/subjects
   tutorial/reduce
   tutorial/simulate
   tutorial/higher_order
   tutorial/events
   tutorial/decoding
   tutorial/htfa_network_animations

Running the tutorials
---------------------

From the root of the repository:

.. code-block:: bash

   pip install -e ".[tutorials]"
   jupyter notebook docs/tutorial/

To run a notebook from top to bottom and save its output:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute --inplace docs/tutorial/quickstart.ipynb

Notebooks that download data keep it in the folder named by the
``TIMECORR_DATA`` environment variable (default: ``~/timecorr_data``). Each
notebook states its run time if it takes more than a couple of minutes.
