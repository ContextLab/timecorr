# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-17

### Changed

- **Numerical results differ from 0.2.0: the correlation estimator changed.**
  `wcorr` (and so `timecorr`, `isfc`, `wisfc`, `autofc`, `tc.features` and the
  decoders) now computes a kernel-weighted Pearson correlation: for timepoint
  `t` the kernel row `weights[t, :]` weights the covariance and variance sums
  as well as the means,
  `r = sum w d_a d_b / sqrt(sum |w| d_a**2 * sum |w| d_b**2)`, with deviations
  `d` taken from the kernel-weighted mean. In 0.2.0 (and in Eq. 4 of Owen,
  Chang & Manning, 2021) the kernel set only the mean, and the deviations were
  summed over all timepoints with equal weight. There is no option to select
  the 0.2.0 estimator. On block-structured simulated data (T=300, K=10,
  Laplace kernel with scale 20, 10 seeds) the mean correlation between the
  estimated and true correlation matrices is 0.86 with the new estimator and
  0.42 with the 0.2.0 estimator.
- A true delta kernel (`eye_weights`: one non-zero weight per row) gives
  co-fluctuations: products of the globally z-scored series at each timepoint.
  Their average over timepoints is the static correlation.
- A kernel row with fewer than 3 non-zero weights gives an undefined
  correlation; those timepoints are set to 0 and one `UserWarning` per call
  (not per subject), attributed to the caller's line, reports how many there
  are. A kernel whose rows are all zero raises `ValueError`.
- Kernel convention: row `t` of a weights matrix (`weights[t, :]`) is the
  kernel used to estimate at timepoint `t`, everywhere in the package.
- ISFC (`isfc`, `wisfc`): each subject-versus-others cross-correlation matrix
  `C` is symmetrized by averaging the Fisher z-transforms of `C` and `C.T`,
  which with `combine=corrmean_combine` gives Eq. 5 of the paper. 0.2.0 kept
  only the upper triangle of the asymmetric matrix.
- NaN samples carry no weight in correlations (each pair of features uses the
  timepoints where both are observed), and NaN kernel weights count as 0.
- `corrmean_combine`: the diagonal of averaged autocorrelations is 1 (it was
  0); where +1 and -1 both occur in a Fisher z average they are excluded and
  the remaining values are averaged.
- Fisher z averaging applies to correlations only: in `corrmean_combine`,
  `tstat_combine` and the ISFC symmetrization, an entry where any value lies
  outside [-1, 1] (the co-fluctuations of a delta kernel, `eye_weights`) is
  combined on the raw scale (plain mean; t-statistic of the raw values)
  instead of giving NaN.
- Input formatting goes through pydata-wrangler: numpy arrays, pandas
  DataFrames, polars DataFrames, and lists of them are accepted; inputs are
  never modified; subjects with mismatched shapes and non-numeric columns
  raise `ValueError`. A 1-D array or a pandas or polars Series is one feature
  (shape `(T, 1)`); 0-dimensional inputs and arrays with more than two
  dimensions raise `ValueError`. `format_data` always returns C-contiguous
  arrays, so numpy, pandas and polars inputs with equal values give
  bit-identical results.
- `reduce`: a list is reduced with one model fit jointly to all subjects;
  new `ndims` argument; callable reducers; `random_state=0` is passed to the
  model; a one-element list returns a one-element list; the model is never
  asked for more components than the data have; dict `'params'` are passed to
  hypertools as `'kwargs'`, merged with any `'kwargs'` the dict also gives (a
  key given in both with different values raises `ValueError`).
- Decoders moved to the new module `timecorr.decoding` (same public names,
  still importable from `timecorr.helpers`; `timepoint_decoder` and
  `weighted_timepoint_decoder` also from `timecorr`). They compute
  every order's features with the same two-path pipeline as `tc.features`.
- `weighted_timepoint_decoder` and `weighted_timepoint_decoder_ec` fit the
  level weights on two halves of the training group and apply them to the
  held-out group (0.2.0 fit them on the matrices being scored). The weights
  minimize a smooth cross-entropy surrogate of decoding accuracy (a softmax
  over each row of the mixed Fisher z decoding matrix) with SLSQP; the paper
  describes maximizing decoding accuracy directly, which is piecewise
  constant in the weights, and in 0.2.0 the optimizer stayed at its starting
  weights. Dimensionality reduction remains unsupervised and is fit jointly
  on all subjects.
- `timecorr(cfun=None)` returns the kernel-weighted mean of the samples using
  the same kernel rows as the correlation path (including `include_timepoints`,
  `include_zero` and `exclude_timepoints`) and accepts lists.
- `simulate_data`: subjects share one template (0.2.0 drew an independent
  template per subject); seeding uses a `numpy.random.Generator` and no longer
  touches the global NumPy state or silences warnings globally;
  `set_random_seed=0` is a real seed.
- `plot_weights` draws on a given axis, returns it, and does not call
  `plt.show()`.

### Added

- `tc.features(data, order=...)`: features of every order from 0 to `order`
  through two paths per order (Fig. 10 of the paper). The analysis path applies
  the analysis kernel to the previous order's delta features; the delta path
  (`eye_weights`, `autofc`) gives the unblurred co-fluctuations passed to the
  next order, so each order is blurred by the kernel once. Separate reducers
  for the two paths (`rfun`, `delta_rfun`); `rfun=None` runs without
  reduction, guarded by an estimate of each step's peak memory for all
  subjects together (`max_memory_gb`, default 4 GB) that raises `MemoryError`
  before allocating. The estimate counts the `(d, d, T)` correlation array,
  every subject's results, the `T x T` kernels and the block temporaries; it
  does not count reducers or `combine` functions other than `null_combine`.
- `timecorr.higher_order.two_path_step` (one order of the pipeline) and
  `timecorr.higher_order.estimate_memory_gb`.
- One-sided kernels: `tc.one_sided`; `direction` (`'backward'`, `'forward'`)
  and `include_zero` arguments on every `*_weights` function;
  `include_timepoints='backward'` / `'forward'` and `include_zero` on
  `timecorr`.
- Simulation at a chosen correlation order: `simulate_data(order=n)` embeds
  the template at exactly order `n` with a sign-parity cascade (`K` must be
  `p * 2**(n-1)`), with `amplitude_noise` (standard deviation of the
  log-amplitude of each block's product). New functions
  `simulate_multi_order`, `carrier_reducer`, `carrier_columns`,
  `nested_template` and `min_features`. For `order > 1`, `return_corrs=True`
  returns a dict with keys `order`, `corrs` (the correlations the block
  products carry: the latent-sign correlation with off-diagonal entries
  multiplied by `exp(-amplitude_noise**2)`), `requested` and `blocks`. For
  `order >= 3` a `UserWarning` says that variance-based reduction may not
  retain the structure. In the higher-order tutorial's measurement
  (lag-matched event contrast, 20 seeds), PCA keeps part of structure embedded
  at order 2 (0.21, against 0.08 for noise and 0.47 with `carrier_reducer`);
  for structure embedded at order 3, PCA's contrast was not distinguishable
  from its noise control (-0.00 versus 0.02, Welch p = 0.20; 0.48 with
  `carrier_reducer`). Orders above 3 were not measured.
- New tutorial (`docs/tutorial/higher_order_tutorial.ipynb`) and gallery
  examples for one-sided kernels and recovery of simulated structure by order.
  The other three tutorials and the existing examples were updated to run on
  this version.

### Fixed

- `t_weights` used the exponent `-(df - 1) / 2` with the normalizing constant
  of Student's t density, so its rows were not a t density (with `df=1` every
  weight was equal). Each row is now Student's t density of the lag,
  `scipy.stats.t.pdf(t - tau, df)`, so `t_weights` results differ from 0.2.0.
- `smooth` returned integer (truncated) values for integer input; it now
  returns floating-point values equal to smoothing the input as floats.
- `tstat_combine` with a one-element list returned NaN with scipy
  `RuntimeWarning`s; it now raises `ValueError` (a t statistic across
  subjects needs at least two).
- `include_timepoints='pre'` used future timepoints and `'post'` past ones;
  both now mean what their names say (aliases of `'backward'` and
  `'forward'`).
- `wcorr` crashed for inputs with different numbers of features.
- `vec2mat` failed on a 1 x V input; `z2r(inf)` returned 0 (now 1).
- `mat2vec` and `vec2mat` are vectorized; `mat2vec` copies row by row, which
  lowers its peak memory from about 3.1 to about 1.1 times the output size.
- `weighted_timepoint_decoder` and `weighted_timepoint_decoder_ec` crashed
  (pandas `join_axes`, level indexing) and printed debug output.
- Decoders: with `combine=corrmean_combine`, order-0 activations were Fisher
  z averaged, which gave NaN (and chance accuracy) in `timepoint_decoder` and
  `weighted_timepoint_decoder`. Order 0 is now always averaged with
  `mean_combine`; `combine` applies to orders 1 and higher.
- `timepoint_decoder`: weights `mu` were mixed in the wrong transform
  direction and failed for array input; list input crashed; `nfolds=1` now
  warns that the analysis is circular and runs.
- `timecorr(cfun=None)` crashed on list input and ignored the kernel options.
- `smooth` applied the kernel row mirrored, crashed for series shorter than
  the window, and its default parameters broke non-Laplace kernels.
- `get_xval_assignments` made empty folds when the number of folds did not
  divide the number of items (for example 7 subjects in 5 folds), and the
  decoders then crashed; folds now differ in size by at most one, and more
  folds than items raise `ValueError`.
- The weighted decoders raise a clear `ValueError` for an empty
  `optimize_levels`.
- `wisfc` default subject weights: with 2 or fewer features, or a subject
  whose connectome is uncorrelated with every other one, the subject's own
  data entered its reference (or the reference was NaN and the output all
  zeros); the other subjects are now weighted equally, as documented.
- `weighted_mean` failed with its default weights and modified the caller's
  weights.
- `random_corrmat` returned matrices that were not positive semidefinite;
  block lengths came from a float; the `simulate_data` docstring described the
  wrong return values.
- `format_data` failed on read-only pandas 3 arrays.
- `folding_levels` and `folding_levels_ec` failed for scalar settings and
  `rfun=None`.
- Importing timecorr changed matplotlib's global font sizes (to 18-24
  points); it no longer changes any matplotlib setting.

### Documentation

- Every public name now has an API page, including the default kernel
  parameter dictionaries (`gaussian_params`, `laplace_params`, `eye_params`,
  `t_params`, `mexican_hat_params`, `uniform_params`, `boxcar_params`), which
  are listed in a table of their keys, default values and meanings, and
  `iseven` and `isodd`. Each kernel function's docstring names its default
  dictionary and keys. `tests/test_api_docs.py` checks that every
  user-facing function has Parameters, Returns and Examples sections, that
  every user-facing name is listed on the API page, and that the parameter
  table matches the code.

### Dependencies and packaging

- Python 3.10 or later; continuous integration on Python 3.10 to 3.13.
- Added `pydata-wrangler`; `hypertools>=0.8`; `numpy`, `pandas` and
  `matplotlib` are listed explicitly.
- Removed `nose`, `sphinx` and `duecredit` from the install requirements.
  Tests run with `pytest` (`pip install timecorr[test]` adds `pytest` and
  `polars`); documentation dependencies are in the `docs` extra.

## [0.2.0] - 2025-07-09

### 🎉 Major Release - Dependency Removal & Codebase Modernization

This release represents a substantial modernization of the timecorr package, removing external dependencies, improving code quality, and enhancing documentation.

### 🔥 Breaking Changes

- **Removed brainconn dependency**: Implemented core graph theory functions directly (eigenvector centrality, PageRank, node strength) to eliminate external dependency issues
- **Cleaned up public API**: Removed non-existent functions from API documentation that were causing import errors

### ✨ New Features

- **GitHub Actions CI/CD**: Added automated testing on Python 3.8-3.11 for all pushes and pull requests
- **Enhanced documentation**: All user-facing functions now have comprehensive NumPy-style docstrings with examples
- **Direct graph theory implementation**: Native implementations of `eigenvector_centrality_und()`, `pagerank_centrality()`, and `strengths_und()`

### 🛠️ Improvements

- **Code formatting**: Applied black and isort across entire codebase for consistent style
- **Import organization**: Cleaned up import statements throughout the project
- **Documentation cleanup**: Removed build artifacts and redundant files from docs/ directory
- **Tutorial enhancements**: Fixed and verified all tutorial notebooks and examples
- **Test coverage**: Maintained 100% test coverage with 131 passing tests

### 🐛 Bug Fixes

- Fixed kernel parameter errors in Mexican Hat weights (changed 'var' to 'sigma')
- Fixed reshape errors in applications tutorial
- Resolved deprecation warnings by replacing `np.math.pi` with `math.pi`
- Fixed PCA dimension errors by limiting to number of samples
- Fixed padding logic for different array dimensions and list inputs

### 📚 Documentation

- **API Documentation**: Updated to reflect only existing functions, removed references to non-existent ones
- **Enhanced docstrings**: Added comprehensive documentation for all major user-facing functions:
  - `gaussian_weights()`, `laplace_weights()`, `mexican_hat_weights()`, `eye_weights()`
  - `isfc()`, `autofc()`, `timecorr()`
  - All functions now include parameter descriptions, return values, and usage examples
- **Tutorial improvements**: Fixed and verified all Jupyter notebooks and Python examples
- **README updates**: Enhanced with working examples and current API information

### 🧹 Cleanup

- **Removed outdated files**: Cleaned up PNG files, temporary scripts, and build artifacts
- **Docker removal**: Removed outdated Docker setup with broken notebook
- **Makefile cleanup**: Removed outdated Makefile using deprecated nosetests
- **File organization**: Streamlined repository structure removing unnecessary files

### 🔧 Development

- **Modern workflow**: Transitioned from nose to pytest for testing
- **Linting**: Applied comprehensive code formatting and style improvements
- **CI/CD**: Automated testing pipeline for continuous integration
- **Development guide**: Added CLAUDE.md for development guidance

### 📦 Dependencies

- **Removed**: brainconn (replaced with native implementations)
- **Updated**: All dependencies to supported versions
- **Simplified**: Reduced external dependency footprint

### 🧪 Testing

- **131 tests passing**: Complete test suite with comprehensive coverage
- **Cross-platform**: Verified on multiple Python versions (3.8-3.11)
- **Automated**: GitHub Actions ensures tests run on every change

### 📈 Performance

- **Reduced dependencies**: Faster installation and reduced potential conflicts
- **Native implementations**: Direct graph theory calculations without external calls
- **Optimized imports**: Improved import organization and reduced overhead

---

## [0.1.7] - Previous Release

Previous functionality maintained with various bug fixes and improvements.
