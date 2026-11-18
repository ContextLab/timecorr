# Overview

The timecorr toolbox computes and explores the correlational structure of timeseries data. It estimates how correlations between features change from moment to moment (**dynamic correlations**), and it computes **higher-order** dynamic correlations: correlations between the moment-by-moment co-fluctuations of lower-order features.

The toolbox accompanies Owen, Chang & Manning (2021), *Nature Communications* 12, 5728 ([doi:10.1038/s41467-021-25876-x](https://doi.org/10.1038/s41467-021-25876-x)). Version 0.3.0 changed the correlation estimator, so its numbers differ from version 0.2.0 and from Eq. 4 of the paper; see [CHANGELOG.md](CHANGELOG.md). Tutorials, examples and the API reference are on [Read the Docs](https://timecorr.readthedocs.io/en/latest/).

## Features

- **Dynamic correlations** (`tc.timecorr`): kernel-weighted correlations at every timepoint, within one dataset or across subjects (inter-subject functional connectivity, ISFC).
- **Kernels**: Gaussian, Laplace, Mexican hat, t, boxcar, uniform and delta (`eye_weights`), each two-sided or one-sided (only past or only future timepoints, with or without the timepoint itself).
- **Higher-order features** (`tc.features`): features of every order up to a chosen order, computed through the two paths of Fig. 10 of the paper, with optional dimensionality reduction at each order.
- **Dimensionality reduction**: hypertools / scikit-learn models (PCA, FastICA, factor analysis and others), graph measures (eigenvector centrality, PageRank centrality, strength), or your own function.
- **Simulation** (`tc.simulate_data`): data with known dynamic correlations, including structure embedded at exactly one chosen correlation order.
- **Decoding**: timepoint decoders that measure how well features of each order identify the moment a pattern came from, across groups of subjects.
- **Inputs**: numpy arrays, pandas DataFrames and polars DataFrames, or lists of them (one per subject).

# Quick Start

## Dynamic correlations

```python
import numpy as np
import timecorr as tc

rng = np.random.default_rng(0)
data = rng.standard_normal((100, 5))  # 100 timepoints, 5 features

corrs = tc.timecorr(data, weights_function=tc.gaussian_weights, weights_params={"var": 10})
print(corrs.shape)  # (100, 15): one vectorized 5 x 5 correlation matrix per timepoint
print(tc.vec2mat(corrs).shape)  # (5, 5, 100)
```

Each row of the output is a correlation matrix in vectorized form (`tc.mat2vec`): the 5 diagonal entries first, then the 10 entries above the diagonal. `tc.vec2mat` converts back to matrices.

## Several subjects

```python
subjects = [rng.standard_normal((100, 5)) for _ in range(10)]  # 10 subjects

# each subject's features versus the mean of the other subjects' features
isfc = tc.timecorr(subjects, cfun=tc.isfc, weights_params={"var": 10})
print(len(isfc), isfc[0].shape)  # 10 (100, 15)

# one result: the subjects' correlations averaged in Fisher z space
group = tc.timecorr(subjects, cfun=tc.isfc, combine=tc.corrmean_combine,
                    weights_params={"var": 10})
print(group.shape)  # (100, 15)
```

## Higher-order features

```python
X = tc.features(subjects, order=3, weights_params={"var": 10})
print(len(X), X[3][0].shape)  # 4 (100, 5): orders 0-3, each reduced (PCA) to 5 features
```

`X[0]` is the data and `X[k]` holds the order-`k` features of each subject. At each order the analysis kernel is applied to the previous order's co-fluctuations (computed with a delta kernel), so every order is blurred by the kernel once. By default the features of each order are reduced with PCA, fit jointly across subjects, to as many features as the data have; `rfun=None` keeps every correlation (the number of features then grows from `K` to `K(K+1)/2` at each order, and `tc.features` raises `MemoryError` before a step whose estimated peak memory, for all subjects together, exceeds `max_memory_gb`, 4 GB by default).

## One-sided kernels

```python
# use only the current and earlier timepoints to estimate each timepoint's correlations
backward = tc.timecorr(data, weights_params={"var": 10}, include_timepoints="backward")
# UserWarning: 2 of 100 timepoints have fewer than 3 non-zero kernel weights;
# their correlations are undefined and set to 0. 1 of 100 timepoints have an
# effective sample size ((sum |w|)**2 / sum w**2 over the kernel row) below 3;
# the kernel is too narrow to estimate correlations reliably there.
```

`include_timepoints="forward"` uses only the current and later timepoints, and `include_zero=False` leaves out the timepoint being estimated. The same options are available on every kernel function (`direction`, `include_zero`) and through `tc.one_sided`.

## Simulated structure at a chosen order

```python
# 4 blocks of 2 features: the 4 x 4 template is carried only by order-2 correlations
sim, truth = tc.simulate_data(datagen="block", K=8, T=1000, order=2,
                              return_corrs=True, set_random_seed=0)
X = tc.features(sim, order=2, cfun=tc.autofc, rfun=None,
                delta_rfun=tc.carrier_reducer(truth["blocks"]),
                weights_params={"var": 625})
off_diagonal = slice(4, None)  # mat2vec layout: 4 diagonal entries, then 6 pairs
r = np.corrcoef(X[2][:, off_diagonal].ravel(), truth["corrs"][:, off_diagonal].ravel())[0, 1]
print(round(r, 2))  # 0.9
```

`tc.carrier_reducer` keeps, at each order, only the features that carry the simulated structure, so the estimate at order 2 is the 4 x 4 correlation matrix among the block products. See the [higher-order tutorial](docs/tutorial/higher_order_tutorial.ipynb) for recovery at each order and for the limits of variance-based reduction.

# Usage

## Data

- One dataset: a numpy array, pandas DataFrame or polars DataFrame of shape `(timepoints, features)`.
- Several datasets (for example one per subject): a list of such matrices, all with the same shape.
- Inputs are converted to float arrays and never modified. NaN samples carry no weight in correlations.

## Kernels (`weights_function`, `weights_params`)

Row `t` of a kernel matrix, `weights[t, :]`, holds the weights given to every timepoint when estimating the correlations at timepoint `t`.

```python
gaussian = tc.timecorr(data, weights_function=tc.gaussian_weights, weights_params={"var": 10})
laplace = tc.timecorr(data, weights_function=tc.laplace_weights, weights_params={"scale": 5})
mexican_hat = tc.timecorr(data, weights_function=tc.mexican_hat_weights, weights_params={"sigma": 15})
boxcar = tc.timecorr(data, weights_function=tc.boxcar_weights, weights_params={"width": 10})

# delta kernel: co-fluctuations (products of the z-scored features at each timepoint),
# whose mean over timepoints is the static correlation
delta = tc.timecorr(data, weights_function=tc.eye_weights)
```

Kernel functions take `(T, params)` and return a `T x T` matrix, so a function of your own with that signature can be passed as `weights_function`.

## Correlation functions (`cfun`)

```python
within = tc.timecorr(data)                     # one dataset: its own correlations
isfc = tc.timecorr(subjects, cfun=tc.isfc)     # each subject versus the mean of the others
wisfc = tc.timecorr(subjects, cfun=tc.wisfc)   # versus a similarity-weighted mean of the others
auto = tc.timecorr(subjects, cfun=tc.autofc)   # each subject's own correlations
smoothed = tc.timecorr(data, cfun=None)        # kernel-weighted mean of the data, not correlations
```

## Combining subjects (`combine`)

`tc.null_combine` (default) returns one result per subject; `tc.corrmean_combine` averages correlations in Fisher z space; `tc.mean_combine` takes the plain mean; `tc.tstat_combine` returns a t statistic per entry.

## Dimensionality reduction (`rfun`)

```python
pca = tc.timecorr(data, rfun="PCA")                         # 15 correlations -> 5 features
ica = tc.timecorr(data, rfun={"model": "FastICA", "params": {"max_iter": 1000}})
fa = tc.timecorr(data, rfun="FactorAnalysis")
pagerank = tc.timecorr(data, rfun="pagerank_centrality")    # one value per feature (node)
eigenvector = tc.timecorr(data, rfun="eigenvector_centrality")
```

A model name, a dict such as `{"model": "PCA", "params": {"whiten": True}}`, a graph measure, or a callable `f(list_of_arrays) -> list_of_arrays` is accepted. When a list of subjects is reduced, one model is fit jointly to all of them.

# Tutorials and Examples

The tutorials are Jupyter notebooks in `docs/tutorial/`, also rendered on Read the Docs:

- **Using timecorr** (`timecorr_notebook.ipynb`): simulation, kernels, the vectorized format, several subjects.
- **Synthetic data** (`synthetic_data_tutorial.ipynb`): the simulated dataset types, kernel shape and width compared against the true correlations, co-fluctuations from the delta kernel, several subjects, and a per-timepoint test across subjects.
- **Higher-order correlations** (`higher_order_tutorial.ipynb`): the two paths of Fig. 10, `tc.features`, one-sided kernels, simulation and recovery at a chosen order, and the limits (feature counts, memory, variance-based reduction).
- **Applications** (`applications_tutorial.ipynb`): dynamic correlations in simulated neuroscience, finance, climate and social-network data.

Each tutorial has an "Open in Colab" link at the top that opens the notebook in Google Colab, where its first code cell installs timecorr.

The `examples/` folder holds short scripts, shown in the example gallery on Read the Docs.

# Installation

Install the latest release with [pip](https://pypi.org):

```
pip install timecorr
```

or upgrade an existing installation:

```
pip install --upgrade timecorr
```

To install the development version from this repository:

```
pip install --upgrade git+https://github.com/ContextLab/timecorr.git
```

# Requirements

Python 3.10 or later. The toolbox is currently supported on Mac and Linux.  It has not been tested on Windows (and we expect key functionality not to work properly on Windows systems).

Dependencies (installed by pip): pydata-wrangler, hypertools>=0.8, numpy, scipy, scikit-learn, pandas, matplotlib. Optional: polars (for polars DataFrame input), and the `test` and `docs` extras (`pip install "timecorr[test]"`, `pip install "timecorr[docs]"`).

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

The tests use real computations on simulated data (no mocks).

```bash
pip install -e ".[test]"

# run the whole suite
pytest

# skip the slow integration tests (minutes, several GB of memory)
pytest -m "not slow"

# run the examples in the docstrings
pytest --doctest-modules timecorr/
```

# Memory and run time

- `timecorr` builds a `T x T` kernel and a `K x K x T` array of correlations for each subject, so memory grows with `T**2` and with `K**2 * T`.
- Without reduction, the number of features grows from `K` to `K(K+1)/2` at every order. `tc.features(..., rfun=None)` estimates the memory of each step and raises `MemoryError` before allocating more than `max_memory_gb` (default 4 GB); the message gives the largest number of features that fits. `timecorr.higher_order.estimate_memory_gb(K, order, T, n_subjects)` returns the estimates (in GB, for all subjects together).
- Dimensionality reduction (`rfun`, `delta_rfun`) keeps the number of features fixed at each order.

# Getting Help

- **Documentation**: [timecorr.readthedocs.io](https://timecorr.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/ContextLab/timecorr/issues)
- **Tutorials**: Jupyter notebooks in `docs/tutorial/`
- **Examples**: scripts in `examples/`
