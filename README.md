# Overview

The timecorr toolbox provides tools for computing and exploring the correlational structure of timeseries data. This powerful package allows you to analyze how correlations between variables change over time, making it ideal for studying dynamic brain networks, market correlations, climate patterns, and other time-varying systems.

Everything in the toolbox is built around one main function, `timecorr`, which computes **dynamic correlations** from timeseries data and can explore **higher-order correlational structure**. For comprehensive examples, tutorials, and API documentation, visit our [readthedocs page](https://timecorr.readthedocs.io/en/latest/).

## Key Features

- **Dynamic Correlations**: Compute moment-by-moment correlations between features
- **Higher-Order Analysis**: Explore correlations between correlations (and beyond)
- **Multi-Subject Analysis**: Compare patterns across different datasets/participants
- **Flexible Kernels**: Gaussian, Laplace, Mexican Hat, and custom weighting functions
- **Dimensionality Reduction**: PCA, ICA, factor analysis, and graph-theoretic measures
- **Statistical Testing**: Built-in permutation testing and multiple comparisons correction

# Quick Start

## Simple Example: Basic Dynamic Correlations

```python
import numpy as np
import timecorr as tc

# Generate sample data: 100 timepoints, 5 features
data = np.random.randn(100, 5)

# Compute dynamic correlations with Gaussian weighting
correlations = tc.timecorr(data, weights_function=tc.gaussian_weights, weights_params={'var': 10})

print(f"Input shape: {data.shape}")
print(f"Output shape: {correlations.shape}")  # (100, 15) - vectorized correlation matrices
```

## Multi-Subject Analysis Example

```python
# Analyze correlations across multiple subjects
subjects_data = [np.random.randn(100, 5) for _ in range(10)]  # 10 subjects

# Inter-Subject Functional Connectivity (ISFC)
# Returns a list of correlations (one per subject vs. others)
isfc_results = tc.timecorr(subjects_data, cfun=tc.isfc, weights_function=tc.gaussian_weights)
print(f"ISFC results: {len(isfc_results)} subjects, each shape {isfc_results[0].shape}")

# Weighted ISFC for similarity-based averaging
# Also returns a list of correlations
wisfc_results = tc.timecorr(subjects_data, cfun=tc.wisfc, weights_function=tc.gaussian_weights)
print(f"WISFC results: {len(wisfc_results)} subjects, each shape {wisfc_results[0].shape}")

# To get a single averaged result, use combine parameter
combined_isfc = tc.timecorr(subjects_data, cfun=tc.isfc, combine=tc.mean_combine, 
                           weights_function=tc.gaussian_weights)
print(f"Combined ISFC shape: {combined_isfc.shape}")
```

## Higher-Order Correlations with Dimensionality Reduction

```python
# Compute correlations between correlations using PCA
data = np.random.randn(200, 10)

higher_order = tc.timecorr(data, 
                          weights_function=tc.gaussian_weights,
                          weights_params={'var': 20},
                          rfun='PCA')  # Reduces dimensionality to prevent explosion

print(f"Higher-order correlations shape: {higher_order.shape}")  # Same as input: (200, 10)
```

# Comprehensive Usage Guide

## 1. Data Formatting

Your data should be formatted as:
- **Single dataset**: NumPy array or Pandas DataFrame with shape `(timepoints, features)`
- **Multiple datasets**: List of arrays/DataFrames with identical shapes
- **Features**: Columns represent variables (e.g., brain regions, sensors, financial assets)
- **Timepoints**: Rows represent observations over time

```python
# Single subject data
data = np.random.randn(100, 8)  # 100 timepoints, 8 features

# Multiple subjects
multi_data = [np.random.randn(100, 8) for _ in range(5)]  # 5 subjects
```

## 2. Weighting Functions (`weights_function`)

Control how much each timepoint contributes to correlations:

```python
# Gaussian kernel (most common)
correlations = tc.timecorr(data, weights_function=tc.gaussian_weights, 
                          weights_params={'var': 10})

# Laplace kernel (sparser, more temporal precision)
correlations = tc.timecorr(data, weights_function=tc.laplace_weights,
                          weights_params={'scale': 5})

# Mexican Hat kernel (captures temporal dynamics)
correlations = tc.timecorr(data, weights_function=tc.mexican_hat_weights,
                          weights_params={'sigma': 15})

# Eye/Delta kernel (uniform weighting)
correlations = tc.timecorr(data, weights_function=tc.eye_weights,
                          weights_params={})
```

## 3. Correlation Functions (`cfun`)

Choose how correlations are computed:

```python
# Within-subject correlations (default)
within_corr = tc.timecorr(data)

# Inter-Subject Functional Connectivity (ISFC)
# Each subject vs. average of others (returns list)
isfc_corr = tc.timecorr(multi_data, cfun=tc.isfc)

# Weighted ISFC (similarity-weighted averaging, also returns list)
wisfc_corr = tc.timecorr(multi_data, cfun=tc.wisfc)

# Combined ISFC (single averaged result)
combined_isfc = tc.timecorr(multi_data, cfun=tc.isfc, combine=tc.mean_combine)

# Auto-correlation function
auto_corr = tc.timecorr(data, cfun=tc.autofc)
```

## 4. Dimensionality Reduction (`rfun`)

Prevent "correlation explosion" and explore higher-order structure:

```python
# Principal Component Analysis
pca_corr = tc.timecorr(data, rfun='PCA')

# Independent Component Analysis (note: may not be available in all versions)
# ica_corr = tc.timecorr(data, rfun='ICA')

# Factor Analysis
fa_corr = tc.timecorr(data, rfun='FactorAnalysis')

# Graph-theoretic measures
pagerank_corr = tc.timecorr(data, rfun='pagerank_centrality')
eigenvector_corr = tc.timecorr(data, rfun='eigenvector_centrality')
```

## 5. Common Analysis Patterns

### Pattern 1: Basic Dynamic Connectivity
```python
# Analyze how brain region correlations change over time
brain_data = np.random.randn(300, 50)  # 300 TRs, 50 brain regions

dynamic_fc = tc.timecorr(brain_data, 
                        weights_function=tc.gaussian_weights,
                        weights_params={'var': 8})

# Visualize correlation matrix at specific timepoint
import matplotlib.pyplot as plt
timepoint_50_corr = tc.vec2mat(dynamic_fc[50, :])
plt.imshow(timepoint_50_corr, cmap='RdBu_r')
plt.title('Brain Connectivity at Timepoint 50')
```

### Pattern 2: Multi-Level Analysis  
```python
# Compute correlations at multiple temporal scales
correlations_level1 = tc.timecorr(data, weights_params={'var': 5})   # Fine scale
correlations_level2 = tc.timecorr(data, weights_params={'var': 20})  # Coarse scale

# Compare temporal dynamics across scales
```

### Pattern 3: Group Analysis with Statistics
```python
# Compare two groups
group1_data = [np.random.randn(100, 10) for _ in range(15)]
group2_data = [np.random.randn(100, 10) for _ in range(15)]

# Compute group-level ISFCs
group1_isfc = tc.timecorr(group1_data, cfun=tc.isfc)
group2_isfc = tc.timecorr(group2_data, cfun=tc.isfc)

# Statistical comparison (implement your own t-tests, permutation tests, etc.)
```

# Tutorials and Examples

We provide comprehensive Jupyter notebook tutorials to help you get started:

## 📊 Synthetic Data Tutorial
**Location**: `docs/tutorial/synthetic_data_tutorial.ipynb`

Learn the fundamentals of timecorr using synthetic data:
- Generate different types of synthetic datasets (random, block, ramping, constant)
- Explore various kernel functions and their effects
- Understand dynamic correlations and higher-order analysis
- Perform multi-subject analysis with ISFC and WISFC
- Statistical testing and significance assessment

```bash
# Run the tutorial
jupyter notebook docs/tutorial/synthetic_data_tutorial.ipynb
```

## 🔬 Applications Tutorial  
**Location**: `docs/tutorial/applications_tutorial.ipynb`

Discover real-world applications across multiple domains:
- **Neuroscience**: Brain network dynamics and connectivity patterns
- **Economics**: Market correlations and financial network analysis  
- **Climate Science**: Environmental variable relationships over time
- **Social Sciences**: Social network dynamics and behavioral patterns

```bash
# Run the applications tutorial
jupyter notebook docs/tutorial/applications_tutorial.ipynb
```

## 📚 Additional Resources

- **Full Documentation**: [timecorr.readthedocs.io](http://timecorr.readthedocs.io/)
- **API Reference**: Complete function documentation and parameters
- **Example Gallery**: Collection of use cases and code snippets
- **Research Paper**: [Nature Communications](https://doi.org/10.1038/s41467-021-25876-x)

# Installation

## Recommended way of installing the toolbox
You may install the latest stable version of our toolbox using [pip](https://pypi.org):

```
pip install timecorr
```

or if you have a previous version already installed:

```
pip install --upgrade timecorr
```


## Dangerous (hacker) developer way of installing the toolbox (use caution!)
To install the latest (bleeding edge) version directly from this repository use:

```
pip install --upgrade git+https://github.com/ContextLab/timecorr.git
```


# Requirements

The toolbox is currently supported on Mac and Linux.  It has not been tested on Windows (and we expect key functionality not to work properly on Windows systems).
Dependencies:
  - hypertools>=0.7.0
  - scipy>=1.2.1
  - scikit-learn>=0.19.2

# Citing this toolbox

If you use (or build on) this toolbox in your work, we'd appreciate a citation!  Please cite the following paper:

> Owen LLW, Chang TH, Manning JR (2021) High-level cognition during story listening is reflected in high-order dynamic correlations in neural activity patterns.  Nature Communications 12(5728): [doi.org/10.1038/s41467-021-25876-x](https://doi.org/10.1038/s41467-021-25876-x).

# Contributing

Thanks for considering adding to our toolbox!  Some text below has been borrowed from the [Matplotlib contributing guide](http://matplotlib.org/devdocs/devel/contributing.html).

## Submitting a bug report

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2 sentences.
2. A short, self-contained code snippet to reproduce the bug, ideally allowing a simple copy and paste to reproduce. Please do your best to reduce the code snippet to the minimum required.
3. The actual outcome of the code snippet
4. The expected outcome of the code snippet

## Contributing code

The preferred way to contribute to timecorr is to fork the main repository on GitHub, then submit a pull request.

- If your pull request addresses an issue, please use the title to describe the issue and mention the issue number in the pull request description to ensure a link is created to the original issue.

- All public methods should be documented in the README.

- Each high-level plotting function should have a simple example in the examples folder. This should be as simple as possible to demonstrate the method.

- Changes (both new features and bugfixes) should be tested using `pytest`.  Add tests for your new feature to the `tests/` repo folder.

- Please note that the code is currently in beta thus the API may change at any time. BE WARNED.

# Testing

<!-- [![Build Status](https://travis-ci.com/ContextLab/quail.svg?token=hxjzzuVkr2GZrDkPGN5n&branch=master) -->

The timecorr package includes comprehensive tests to ensure reliability and correctness.

## Running Tests

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=timecorr --cov-report=html

# Run specific test files
pytest tests/test_timecorr.py
pytest tests/test_helpers.py
```

## Test Coverage

The package maintains high test coverage (>85%) across all modules:
- **Core functions**: `timecorr`, `isfc`, `wisfc`, `autofc`
- **Helper functions**: Kernels, decoders, dimensionality reduction
- **Simulation functions**: Synthetic data generation
- **Utility functions**: Matrix operations, statistical functions

All tests use actual data (never mocks) to ensure numerical correctness and real-world applicability.

# Troubleshooting

## Common Issues

### Installation Problems
```bash
# If you encounter dependency conflicts
pip install --upgrade --force-reinstall timecorr

# For development installation
pip install -e .
```

### Memory Issues with Large Datasets
```python
# Use chunking for very large datasets
import timecorr as tc
import numpy as np

# Instead of processing all at once
large_data = np.random.randn(10000, 100)  # Very large dataset

# Process in chunks
chunk_size = 1000
results = []
for i in range(0, len(large_data), chunk_size):
    chunk = large_data[i:i+chunk_size]
    result = tc.timecorr(chunk, weights_function=tc.gaussian_weights)
    results.append(result)

# Combine results
final_result = np.vstack(results)
```

### Performance Optimization
```python
# For faster computation with large numbers of features
# Use dimensionality reduction
fast_result = tc.timecorr(data, 
                         weights_function=tc.gaussian_weights,
                         rfun='PCA')  # Reduces computational complexity

# Adjust kernel width for faster computation
# Smaller variance = faster but less smoothing
fast_correlations = tc.timecorr(data, 
                               weights_function=tc.gaussian_weights,
                               weights_params={'var': 5})  # Smaller kernel
```

## Getting Help

- **Documentation**: [timecorr.readthedocs.io](http://timecorr.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/ContextLab/timecorr/issues)
- **Tutorials**: Check `docs/tutorial/` folder for Jupyter notebooks
- **Examples**: See applications tutorial for domain-specific examples
