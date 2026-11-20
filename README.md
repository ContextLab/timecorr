# Overview

The timecorr toolbox computes and explores the correlational structure of timeseries data. It estimates how correlations between features change from moment to moment (**dynamic correlations**), and it computes **higher-order** dynamic correlations: dynamic correlations of features derived from lower-order dynamic correlations.

The toolbox accompanies Owen, Chang & Manning (2021), *Nature Communications* 12, 5728 ([doi:10.1038/s41467-021-25876-x](https://doi.org/10.1038/s41467-021-25876-x)). Version 0.3.0 changed the correlation estimator, so its numbers differ from version 0.2.0 and from Eq. 4 of the paper; see [CHANGELOG.md](https://github.com/ContextLab/timecorr/blob/master/CHANGELOG.md). Tutorials and the API reference are on [Read the Docs](https://timecorr.readthedocs.io/en/latest/).

## Features

- **Dynamic correlations** (`tc.timecorr`): kernel-weighted correlations at every timepoint, within one dataset or across subjects (inter-subject functional connectivity, ISFC).
- **Kernels**: Gaussian, Laplace, Mexican hat, t, boxcar, uniform and delta (`eye_weights`), each two-sided or one-sided (only past or only future timepoints, with or without the timepoint itself).
- **Higher-order features** (`tc.features`): features of every order up to a chosen order, computed through the two paths of Fig. 10 of the paper (an analysis path and a path that feeds the next order), with a reducer on each path at each order. By default the path that feeds the next order uses a Gaussian kernel and a summary of squared correlations by community; the original pipeline of the paper (co-fluctuations and PCA) is selected with three arguments.
- **Dimensionality reduction**: hypertools / scikit-learn models (PCA, FastICA, factor analysis and others), graph measures (eigenvector centrality, PageRank centrality, strength), or your own function.
- **Event-restricted kernels** (`events=` in `tc.timecorr` and `tc.features`, and `tc.segment`): confine the kernel to blocks of consecutive timepoints (events), given as boundary indices or found by segmenting a separate timeseries (for example a stimulus embedding) with a hidden Markov model; events may differ per subject and, in `tc.features` and the correlation decoders, per order.
- **Simulation** (`tc.simulate_data`): data with known dynamic correlations, including structure embedded at exactly one chosen correlation order.
- **Decoding**: timepoint decoders that measure how well features of each order identify the moment a pattern came from, across groups of subjects. They accept `events`: the correlation decoders restrict the analysis kernel at levels 1 and above (level 0 is unchanged), and `pca_decoder` restricts its smoothing kernel; the events are supplied by the caller and used as given in every fold.
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
# 108 nodes in 18 modules in 6 supergroups, 2 subjects
nested, truth = tc.simulate_nested(S=2, T=1000, set_random_seed=0)
X = tc.features(nested, order=2, cfun=tc.autofc, weights_params={"var": 33 ** 2})
print(len(X), X[1][0].shape, X[2][0].shape)  # 3 (1000, 108) (1000, 36)
```

`X[0]` is the data and `X[k]` holds the order-`k` features of each subject. Each order has two paths: the analysis path gives `X[k]`, and the path that feeds the next order (the feeding path, named `delta_*` in the arguments for historical reasons) gives the input of order `k + 1`. With the defaults, `X[1]` is the dynamic correlations of the data reduced with PCA, fit jointly across subjects, to as many features as the data have; the feeding path computes each subject's correlations with a Gaussian kernel (standard deviation 5 samples) and summarizes them with `tc.Communities()` (here 18 communities, one column each); `X[2]` is the dynamic correlations of those columns, summarized with `tc.WeightedStrength(signed=True)` (two columns per input column). The default feeding path needs at least 4 features and 40 timepoints; when a step has more features than distinct shift-null lags (21 at T = 40 with the default feeding kernel), the shift null uses each lag for several features (each lag once or twice at T = 40 with 22 to 42 features), and features that share a lag keep their dependence in that surrogate: with K = q n + r features and n lags, a fraction [n q (q - 1) + 2 r q] / [K (K - 1)] of the pairs (1/231 at K = 22 and 1/41 at K = 42 with the 21 lags at T = 40), and a `UserWarning` says so (give the number of communities `n` to avoid the shift null). When the communities found at a feeding step would give fewer than 4 columns (at most half the step's input features; one when the shift null finds no structure), that step gives `tc.WeightedStrength(split=False, signed=False, center=False)` of the same correlations instead (one column per input feature; uncentered pair weights, without median clipping), so the feeding width never drops below 4 (a warning is issued only when the shift null found no structure; `return_info=True` records every fallback); with 4 to 7 input features, every default feeding step does so. This also changes calls that completed without the fallback: on 4 independent noise features (T 100), `D[1]` of `tc.features(x, order=1, return_delta=True)` had 1 community column and now has 4 weighted-strength columns, and `X[2]` of `tc.features(x, order=2)` has 8 columns instead of 2. Summary columns whose range over time is nonzero but at most `1e-9 * max(1, max|column|)` are round-off; they are set to their means exactly and reported by the constant-output warning. An explicit `tc.Communities()` keeps its communities. `delta_rfun="weighted_strength"` (one column per feature) takes any number of features.

```python
# the original two-path pipeline of the paper: co-fluctuations feed the next order,
# PCA on both paths
X = tc.features(subjects, order=3, weights_params={"var": 10},
                rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
print(len(X), X[3][0].shape)  # 4 (100, 5): orders 0-3, each reduced (PCA) to 5 features
```

In the original pipeline the analysis kernel is applied to the previous order's co-fluctuations (computed with a delta kernel, `eye_weights`), so every order is blurred by the kernel once, and the features of each order are reduced with PCA. With `rfun=None` and `delta_rfun=None`, nothing is reduced and every correlation is kept (the number of features then grows from `K` to `K(K+1)/2` at each order, and `tc.features` raises `MemoryError` before a step whose estimated peak memory, for all subjects together, exceeds `max_memory_gb`, 4 GB by default).

## One-sided kernels

```python
# use only the current and earlier timepoints to estimate each timepoint's correlations
backward = tc.timecorr(data, weights_params={"var": 10}, include_timepoints="backward")
# UserWarning: 2 of 100 timepoints have fewer than 3 non-zero kernel weights;
# their correlations are undefined and set to 0.
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

`tc.carrier_reducer` keeps, at each order, only the features that carry the simulated structure, so the estimate at order 2 is the 4 x 4 correlation matrix among the block products. See the [higher-order tutorial](https://github.com/ContextLab/timecorr/blob/master/docs/tutorial/higher_order.ipynb) for recovery at each order and for the limits of variance-based reduction.

## Squared-correlation summaries and the feeding kernel

```python
# the subjects of simulate_nested share only the order-2 signal h
comm = tc.Communities()  # number of communities set by a shift null
X = tc.features(nested, order=2, cfun=tc.autofc, weights_params={"var": 33 ** 2},
                rfun="auto",           # the default: PCA at order 1, WeightedStrength(signed=True) above
                delta_rfun=comm,       # the default "auto" gives a new Communities() per order
                delta_weights_function=tc.gaussian_weights,  # the default kernel of the feeding path
                delta_weights_params={"var": 25})
print(comm.fits_[0]["n"], X[2][0].shape)  # 18 (1000, 36)
```

This call gives the same features as the default call above; passing a `tc.Communities` object makes its fit readable in `fits_`. `tc.WeightedStrength` and `tc.Communities` summarize each subject's correlations by weighted sums of squared correlations per node or by the mean squared correlation within each community of nodes; they are accepted wherever `rfun` is. `delta_weights_function` and `delta_weights_params` set the kernel of the feeding path (default: `gaussian_weights` with variance 25; `eye_weights` gives co-fluctuations). In the higher-order tutorial (`higher_order.ipynb`; this call with a third, control subject, seed 0), a ridge map from subject A's order-2 features to `h`, applied to subject B, gives a mean correlation of 0.74 with `h` over the 6 supergroups (a raw score for one seed; over 24 seeds in `tests/test_nested_recovery.py`, with this configuration (`cfun=tc.autofc`, an analysis kernel with standard deviation 33, wider than the default 10, `Communities()` on the feeding path and the default analysis reducers), the score is .753 +- .019, mean +- standard error, with a control of -.025 +- .018 from subjects with an independent `h`). The default `cfun=tc.isfc` correlates each subject's features with the other subjects' and so reflects structure shared across subjects; `cfun=tc.autofc` computes within-subject correlations. Measured over 12 seeds (108 nodes, T 1000, 6 subjects per call, every argument other than those named left at its default; mean +- standard error):

| data (`shared=`) | call | `X[2]` score (control) | `D[2]` score (control) |
|-|-|-|-|
| `"order2"` (default; order-1 couplings drawn per subject) | `tc.features(data, order=2, return_delta=True)` (`cfun=tc.isfc`, standard deviation 10) | -.035 +- .029 (.055 +- .019) | .199 +- .021 (.013 +- .015) |
| `"order2"` | `cfun=tc.autofc` | .559 +- .022 (-.008 +- .025) | .199 +- .021 (.013 +- .015) |
| `"order1"` (order-1 couplings shared) | `tc.features(data, order=2, return_delta=True)` | .638 +- .028 (.074 +- .009) | .293 +- .027 (.012 +- .014) |
| `"order1"` | `cfun=tc.autofc` | .702 +- .012 (-.008 +- .025) | .293 +- .027 (.012 +- .014) |

So with the literal defaults on the generator's default data, `h` was not recovered from `X[2]` but was from the order-2 feeding features `D[2]`, which do not depend on `cfun`. Recovery of structure above order 2 was not measured with the default reducers. A step reduced by a summary (with `cfun` `autofc`, `isfc` or `wisfc` and `combine=tc.null_combine`) is computed in blocks of nodes; memory is bounded only when every step uses a summary, since the default order-1 reducer (PCA) holds `K(K+1)/2` columns per subject; with `rfun=tc.WeightedStrength(signed=True)` the order-1 step is also computed in blocks. At 702 nodes (T 1000, 3 subjects) the default order-1 step is estimated at 10.31 GB (`estimate_memory_gb(702, 1, 1000, 3)`), above the default `max_memory_gb=4`; the 702-node run in `tests/test_nested_recovery.py` uses weighted strength on both paths at every order, order 1 included.

# Usage

## Data

- One dataset: a numpy array, pandas DataFrame or polars DataFrame of shape `(timepoints, features)`.
- Several datasets (for example one per subject): a list of such matrices, all with the same shape.
- Inputs are converted to float arrays and never modified. NaN samples carry no weight in correlations. In `tc.features` with the default feeding kernel, NaN samples are left out pairwise and the features passed to the next order are finite. With `eye_weights` as the feeding kernel, this holds at every order when the delta path is unreduced (`delta_rfun=None` with `rfun=None`) or reduced by a callable; before a model-based reducer (such as PCA) or a graph measure, the co-fluctuations of NaN samples are set to 0 and a warning says that NaN handling is approximate from order 2 on.

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

# Tutorials

The tutorials are Jupyter notebooks in `docs/tutorial/`; [Read the Docs](https://timecorr.readthedocs.io/en/latest/tutorials.html) shows them as a gallery with the output of a full run. Each link below opens the notebook in Google Colab.

- [**Quickstart**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/quickstart.ipynb) (`quickstart.ipynb`): up and running in four calls: one dataset, several subjects, higher orders.
- [**Kernels**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/kernels.ipynb) (`kernels.ipynb`): the kernels and their parameters, kernel shape and width, one-sided kernels, the delta kernel (co-fluctuations), the output format, and smoothing.
- [**Several subjects**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/subjects.ipynb) (`subjects.ipynb`): data formats (NumPy, pandas, polars), `autofc`, `isfc` and `wisfc`, the combine functions, and a test at each timepoint.
- [**Reducing correlations**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/reduce.ipynb) (`reduce.ipynb`): models by name, graph measures, `WeightedStrength` and `Communities`.
- [**Simulating data**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/simulate.ipynb) (`simulate.ipynb`): `simulate_data` and its templates, structure at a chosen order, `simulate_multi_order`, `simulate_nested` and `simulate_hierarchy`.
- [**Higher-order correlations**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/higher_order.ipynb) (`higher_order.ipynb`): the two paths, `tc.features` and `two_path_step`, `return_info`, reducers and kernels by order, memory, the original pipeline of the paper, and recovery of an injected order-2 signal.
- [**Event-restricted kernels**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/events.ipynb) (`events.ipynb`): passing event boundaries, finding them with `tc.segment`, several subjects, and events by order.
- [**Decoding timepoints**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/decoding.ipynb) (`decoding.ipynb`): one decoding step by hand and the four decoders.
- [**Brain-network animations with HTFA**](https://colab.research.google.com/github/ContextLab/timecorr/blob/master/docs/tutorial/htfa_network_animations.ipynb) (`htfa_network_animations.ipynb`): features of orders 1 to 4 of the story-listening fMRI data analyzed by Owen, Chang & Manning (2021), drawn on the brain with the [htfa](https://htfa.readthedocs.io/en/latest/) toolbox.

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

Dependencies (installed by pip): pydata-wrangler, hypertools>=0.8, numpy, scipy, scikit-learn, pandas, matplotlib. Optional: polars (for polars DataFrame input); hmmlearn, used only by `tc.segment`, which installs it on demand the first time it is needed (the `segment` extra, `pip install "timecorr[segment]"`; setting `TIMECORR_AUTO_INSTALL=0` turns automatic installation off); and the `test`, `docs` (building the documentation) and `tutorials` (running the tutorial notebooks) extras (`pip install "timecorr[test]"`, `pip install "timecorr[docs]"`, `pip install "timecorr[tutorials]"`).

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

- Each high-level plotting function should have a simple example in a tutorial in `docs/tutorial/`. This should be as simple as possible to demonstrate the method.

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

- `timecorr` builds a `T x T` kernel and a `K x K x T` array of correlations for each subject, so memory grows with `T**2` and with `K**2 * T`. The estimator multiplies each chunk of 128 kernel rows only with the samples where those rows have non-zero weight, when those spans cover at most half of the kernel's entries (and the data and their squares are finite). Narrow kernels (a Gaussian underflows to exactly 0 beyond 192 samples at variance 25, also when one-sided) and event-restricted kernels with short blocks then take time roughly proportional to `T` times the kernel's width instead of `T**2` for the products of each pair of features; wide kernels (a one-sided uniform kernel, for example) use the full products. Building the `T x T` kernel and finding its spans still take time proportional to `T**2`. The peak memory measured with `tracemalloc` was the same with and without slicing to within 0.01% (`autofc` on one subject of T 2000, K 4 with a width-12 boxcar, and of T 6000, K 4 with a Gaussian of variance 25).
- Without reduction, the number of features grows from `K` to `K(K+1)/2` at every order. `tc.features(..., rfun=None, delta_rfun=None)` estimates the memory of each step and raises `MemoryError` before allocating more than `max_memory_gb` (default 4 GB); the message gives the largest number of features that fits. `timecorr.higher_order.estimate_memory_gb(K, order, T, n_subjects)` returns the estimates (in GB, for all subjects together; with `delta_weights_function=tc.eye_weights` for the original pipeline, whose delta path has no block temporaries).
- With reduction (`rfun`, `delta_rfun`) the widths depend on the reducer: PCA gives `ndims` columns (the input width by default), `tc.Communities()` one column per community found (at most half its input features), and `tc.WeightedStrength(signed=True)` twice its input width. With the defaults the order-`k` analysis output (`k >= 2`) is twice the number of community columns found at order `k - 1`.

# Getting Help

- **Documentation**: [timecorr.readthedocs.io](https://timecorr.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/ContextLab/timecorr/issues)
- **Tutorials**: Jupyter notebooks in `docs/tutorial/`, shown as a gallery on [Read the Docs](https://timecorr.readthedocs.io/en/latest/tutorials.html)
