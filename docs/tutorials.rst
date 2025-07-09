.. _tutorial:

Tutorials and Examples
======================

The timecorr package provides comprehensive tutorials to help you get started with dynamic correlation analysis. These tutorials cover everything from basic concepts to advanced applications across multiple domains.

Getting Started
---------------

If you're new to timecorr, we recommend starting with the synthetic data tutorial to understand the core concepts and functionality.

Core Tutorials
--------------

.. toctree::
   :maxdepth: 2
   :caption: Fundamental Concepts

   tutorial/synthetic_data_tutorial
   tutorial/applications_tutorial
   tutorial/timecorr_notebook

Synthetic Data Tutorial
^^^^^^^^^^^^^^^^^^^^^^^

Learn the fundamentals of timecorr using synthetic datasets:

* Generate different types of synthetic data (random, block, ramping, constant)
* Explore various kernel functions and their effects on correlation estimates
* Understand dynamic correlations and higher-order correlation analysis
* Perform multi-subject analysis with ISFC and WISFC methods
* Conduct statistical testing and significance assessment

**Location**: ``docs/tutorial/synthetic_data_tutorial.ipynb``

Applications Tutorial
^^^^^^^^^^^^^^^^^^^^

Discover real-world applications across multiple research domains:

* **Neuroscience**: Brain network dynamics and functional connectivity patterns
* **Economics**: Market correlations and financial network analysis  
* **Climate Science**: Environmental variable relationships over time
* **Social Sciences**: Social network dynamics and behavioral pattern analysis

**Location**: ``docs/tutorial/applications_tutorial.ipynb``

Key Concepts Covered
-------------------

Weighting Functions
^^^^^^^^^^^^^^^^^^

* **Gaussian weights**: Smooth temporal averaging with adjustable variance
* **Laplace weights**: Sparser temporal kernels for precise timing
* **Mexican Hat weights**: Derivative-based kernels for temporal dynamics
* **Custom kernels**: How to define your own weighting functions

Correlation Methods
^^^^^^^^^^^^^^^^^^

* **Within-subject correlations**: Standard dynamic functional connectivity
* **ISFC (Inter-Subject Functional Connectivity)**: Shared patterns across subjects
* **WISFC (Weighted ISFC)**: Similarity-weighted multi-subject analysis
* **Auto-correlation functions**: Temporal structure analysis

Dimensionality Reduction
^^^^^^^^^^^^^^^^^^^^^^^^

* **PCA**: Principal component analysis for correlation patterns
* **ICA**: Independent component analysis
* **Factor Analysis**: Latent factor modeling
* **Graph measures**: Network centrality and connectivity metrics

Advanced Topics
^^^^^^^^^^^^^^

* **Higher-order correlations**: Correlations between correlation patterns
* **Multi-level analysis**: Hierarchical temporal structure
* **Statistical testing**: Permutation tests and multiple comparisons
* **Optimization**: Memory and performance considerations

Common Use Cases
---------------

**Neuroscience Applications**

.. code-block:: python

   # Analyze brain network dynamics
   brain_data = load_fmri_data()  # Shape: (timepoints, brain_regions)
   
   # Compute dynamic functional connectivity
   dfc = tc.timecorr(brain_data, 
                     weights_function=tc.gaussian_weights,
                     weights_params={'var': 8})
   
   # Visualize connectivity at specific timepoint
   connectivity_matrix = tc.vec2mat(dfc[100, :])

**Financial Market Analysis**

.. code-block:: python

   # Analyze market correlations over time
   stock_returns = load_market_data()  # Shape: (days, stocks)
   
   # Compute dynamic correlations
   market_correlations = tc.timecorr(stock_returns,
                                    weights_function=tc.laplace_weights,
                                    weights_params={'scale': 5})

**Climate Science**

.. code-block:: python

   # Study climate variable relationships
   climate_data = load_climate_measurements()  # Shape: (months, variables)
   
   # Analyze long-term correlation patterns
   climate_correlations = tc.timecorr(climate_data,
                                     weights_function=tc.gaussian_weights,
                                     weights_params={'var': 12})  # Annual scale

Running the Tutorials
--------------------

To run the interactive tutorials locally:

.. code-block:: bash

   # Navigate to the timecorr directory
   cd /path/to/timecorr
   
   # Start Jupyter notebook
   jupyter notebook
   
   # Open tutorials in docs/tutorial/ folder

Next Steps
----------

After completing the tutorials, explore:

* **API Reference**: Detailed function documentation
* **Example Gallery**: Additional code examples and use cases
* **Research Paper**: Theoretical background and validation studies
* **GitHub Repository**: Latest updates and community contributions


