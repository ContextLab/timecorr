# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-17

### Changed

- **Changed defaults of the higher-order pipeline (`features`,
  `two_path_step`).** The defaults are `rfun="auto"` (PCA at order 1,
  `WeightedStrength(signed=True)` at orders 2 and above),
  `delta_rfun="auto"` (`Communities()`, with the number of communities
  counted with a shift null) and a Gaussian feeding kernel with variance 25
  (`delta_weights_function=None`; standard deviation 5 samples). `X[1]` is
  unchanged; `D[1]` (`return_delta=True`) and orders 2 and above differ.
  `rfun=None` alone no longer leaves the pipeline unreduced: the feeding
  path is still reduced by `Communities()`; `rfun=None, delta_rfun=None`
  reduces neither path. The original two-path
  pipeline of Owen, Chang & Manning (2021, Fig. 10), with co-fluctuations on
  the feeding path and PCA on both paths (the defaults of `features` earlier
  in the development of this release, commit d4bb5ba), is
  `tc.features(..., rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)`.
  The default feeding reducer needs at least 4 features, 40 timepoints and
  as many distinct shift-null lags in `[L, T - L]` as features (for
  example at most 21 features at T = 40) at every feeding step that is
  computed; otherwise `ValueError` names `delta_rfun="weighted_strength"`
  (any number of features and timepoints) and the original pipeline's
  settings. At each default feeding step, the communities are fitted to
  that step's correlations; when they would emit fewer than 4 columns (at
  most half the step's input features; one when no eigenvalue exceeds the
  shift null), the step emits `WeightedStrength(split=False, signed=False,
  center=True)` of the same correlations instead (one column per input
  feature, from the same pair weights), so the feeding width never drops below 4 and, for example, 4 to 7
  input features complete at `order=3`. A `UserWarning` (one per order) is
  issued only when no eigenvalue exceeds the shift null; `features(...,
  return_info=True)` returns `(result, info)` with one dict per computed
  step and path (reducer requested and used, input and output widths, and
  for default feeding steps the community columns, exceedances, threshold
  and fallback reason). An explicit `Communities()` keeps
  its communities (and its "no community structure" warning). `ndims` with the defaults raises
  `ValueError` in `features` at `order >= 2` (the order-2 analysis reducer
  is a summary) and at `order=1` with `return_delta=True` (the order-1
  feeding reducer is `Communities()`), and in `two_path_step` at every
  order.
  `delta_weights_params` without `delta_weights_function` sets the
  parameters of the Gaussian feeding kernel. The decoders are unchanged
  (co-fluctuations on the path that feeds the next level).
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
  are. The same warning also reports the other timepoints whose kernel row
  has an effective sample size `(sum |w|)**2 / sum w**2` below 2 (a kernel too
  narrow for a reliable estimate, such as a Gaussian with `var=0.05`), and the
  timepoints where every feature has zero
  kernel-weighted variance, so that all correlations there are 0 (constant
  data, or a kernel so narrow that every variance cancels). A delta kernel
  never warns. A kernel whose rows are all zero raises `ValueError`.
- Kernel convention: row `t` of a weights matrix (`weights[t, :]`) is the
  kernel used to estimate at timepoint `t`, everywhere in the package.
- ISFC (`isfc`, `wisfc`): each subject-versus-others cross-correlation matrix
  `C` is symmetrized by averaging the Fisher z-transforms of `C` and `C.T`,
  which with `combine=corrmean_combine` gives Eq. 5 of the paper. 0.2.0 kept
  only the upper triangle of the asymmetric matrix.
- NaN samples carry no weight in correlations (each pair of features uses the
  timepoints where both are observed), and NaN kernel weights count as 0. In
  the decoders, and in `tc.features` with `eye_weights` as the feeding kernel,
  a co-fluctuation involving a NaN sample is
  NaN, so later orders also leave it out pairwise when the delta path is
  unreduced or reduced by a callable; before a model-based or graph-measure
  delta reduction the NaNs are set to 0 with one `UserWarning` per call
  (NaN handling is then approximate from order 2 on).
- A correlation for a pair of features with fewer than 3 observed (non-NaN)
  samples with non-zero kernel weight is undefined: it is set to 0 and the
  timepoint is counted in the call's one `UserWarning`. Before commit 5068179
  it was returned as +1 or -1 without a warning. This changes results for
  data with NaNs, with or without `events`.
- `corrmean_combine`: the diagonal of averaged autocorrelations is 1 (it was
  0); where +1 and -1 both occur in a Fisher z average they are excluded and
  the remaining values are averaged.
- Fisher z averaging applies to correlations only: in `corrmean_combine`,
  `tstat_combine` and the ISFC symmetrization, an entry where any value lies
  outside [-1, 1] (the co-fluctuations of a delta kernel, `eye_weights`) is
  combined on the raw scale (plain mean; t-statistic of the raw values)
  instead of giving NaN. Values within 1e-12 of [-1, 1] are correlations with
  round-off and are clipped to +-1.
- `tstat_combine`: an entry with the same value in every subject (zero
  variance, for example the diagonal of `autofc` results) gives t = +inf or
  -inf with the sign of that value (p = 0), or t = 0 (p = 1) for a value of
  0, with no warning (it was NaN with a scipy `RuntimeWarning`).
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
  key given in both with different values raises `ValueError`). `ndims` must
  be a positive integer, and giving it with a graph measure or a callable
  (which set their own width) raises `ValueError`. When the data supply fewer
  components than requested, the zero columns added to reach `ndims` are
  reported in a `UserWarning`. For UMAP, `n_jobs=1` and (where the installed
  UMAP has it) `compatibility_layout=True` are passed unless given, which
  keeps UMAP's current layout and silences its `FutureWarning`.
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
- Decoders and `smooth` name the kernel arguments as `timecorr` and
  `tc.features` do: `weights_function` in `timepoint_decoder`,
  `weighted_timepoint_decoder`, `weighted_timepoint_decoder_ec`,
  `folding_levels`, `folding_levels_ec` and `pca_decoder`, and
  `weights_function` / `weights_params` in `smooth`. The 0.2.0 names
  (`weights_fun`; `kernel_fun`, `kernel_params`) are still accepted as
  aliases, positional order is unchanged, and passing a name together with
  its alias raises `TypeError`. A decoder `level` is an `order` in
  `tc.features`. The declared kernel default of `timepoint_decoder`,
  `weighted_timepoint_decoder`, `weighted_timepoint_decoder_ec`,
  `pca_decoder` and `smooth` is now `None` (it was `laplace_weights`); `None`
  means the Laplace kernel, with the same `weights_params` defaults as
  before, so results do not change.
- The decoders (`timepoint_decoder`, `weighted_timepoint_decoder`,
  `weighted_timepoint_decoder_ec`, `folding_levels`) and
  `two_path_step` take `max_memory_gb` (default 4.0; None, as in
  `tc.features`, checks nothing) and raise the same
  `MemoryError` as `tc.features` before an unreduced correlation step whose
  estimated peak memory exceeds it; the check now lives in the pipeline's
  shared path functions.
- `weight_corrs` clips correlations to +-(1 - 1e-7) before the Fisher z
  transform, as `optimize_weights` does (an exact 1 with weight 0 gave NaN, then
  0, with a `RuntimeWarning`).
- `pca_decoder` raises `ValueError` (not `AssertionError`) for subjects of
  different shapes; its `cfun`, `combine` and `rfun` arguments are accepted
  and ignored, as documented. `folding_levels_ec` without a kernel function
  raises `ValueError` (it raised `TypeError: 'NoneType' object is not
  callable`). `get_xval_assignments` and the decoders reject a non-integer
  `nfolds` with `ValueError` (2.5 was truncated to 2); an integer-valued
  number such as 2.0 is still accepted as that integer.
- `decoding.reduce_wrapper` (internal, unused) reduces the raw features of
  both groups into one space and returns them in the structure they were
  given; it returned the first two subjects instead of the two groups.
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
  (`autofc` with the feeding kernel) gives the features passed to the next
  order. With `eye_weights` as the feeding kernel (the original pipeline)
  they are unblurred co-fluctuations, so each order is blurred by the kernel
  once. Separate reducers
  for the two paths (`rfun`, `delta_rfun`); `rfun=None, delta_rfun=None`
  runs without reduction, guarded by an estimate of each step's peak memory for all
  subjects together (`max_memory_gb`, default 4 GB; None checks nothing) that
  raises `MemoryError` before allocating. `rfun` and `delta_rfun` are validated before anything is
  computed (also a `delta_rfun` that a run would not call), and `ndims` is
  passed only to model-based reducers. The estimate counts the `(d, d, T)` correlation array,
  every subject's results, the `T x T` kernels and the block temporaries; it
  does not count reducers or `combine` functions other than `null_combine`.
- `timecorr.higher_order.two_path_step` (one order of the pipeline) and
  `timecorr.higher_order.estimate_memory_gb` (keyword `delta_weights_function`:
  with None, the default Gaussian feeding kernel, or any kernel other than
  `eye_weights`, the feeding path's block temporaries are counted).
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
- `simulate_data(subject_noise=...)` (order 1): all subjects share one draw
  from the template plus independent Gaussian noise whose standard deviation
  is `subject_noise` times each feature's standard deviation, so ISFC has a
  shared signal to find (a feature's cross-subject correlation is about
  `1 / (1 + subject_noise**2)`). The default, None, keeps independent draws.
  `subject_noise` applies to `order=1` only; with `order > 1` it raises
  `ValueError` (those simulations already share their latent signs across
  subjects).
- `carrier_columns(blocks, level, K=None)`: `K` defaults to
  `blocks.max() + 1`, as in `carrier_reducer`.
- Event-restricted kernels: `events=` and `events_params=` (keyword-only) in
  `tc.timecorr`, `tc.features` (also a dict keyed by order, one specification
  per order), `timecorr.higher_order.two_path_step` and the decoders
  (`timepoint_decoder`, `weighted_timepoint_decoder`,
  `weighted_timepoint_decoder_ec`, `pca_decoder`). The kernel weight
  `weights[t, u]` is set to 0 whenever `t` and `u` lie in different events.
  `events` accepts boundary indices (block starts), a features array with one
  row per timepoint (segmented with `tc.segment` and `events_params`), or a
  list with one such entry (or None) per subject. Without `events`, results
  are unchanged.
- `tc.segment`: divides a timeseries into events with a Gaussian hidden
  Markov model (`hmmlearn`), either ordered events (`kind="ordered"`, every
  state visited once, in order; with `n=None` the number of events is chosen
  automatically) or recurring states (`kind="recurring"`), and returns the
  block starts. Building blocks in `timecorr.events`: `check_boundaries`,
  `block_labels`, `resolve_events`, `resolve_events_by_order`, `mask_kernel`
  and `distinct_event_sets`.
- The `segment` extra (`pip install "timecorr[segment]"`) installs
  `hmmlearn`. Without it, `hmmlearn` is installed with `python -m pip install`
  on the first use of `tc.segment`; setting the environment variable
  `TIMECORR_AUTO_INSTALL=0` turns that off (a missing `hmmlearn` then raises
  `ImportError` with the manual command).
- New tutorial on event-restricted kernels
  (`docs/tutorial/event_kernels_tutorial.ipynb`).
- New tutorial (`docs/tutorial/higher_order_tutorial.ipynb`) and gallery
  examples for one-sided kernels and recovery of simulated structure by order.
  The other three tutorials and the existing examples were updated to run on
  this version.
- Squared-correlation summaries (`timecorr.summaries`, exported from
  `timecorr`): `tc.WeightedStrength(split=False, signed=False, n=None,
  n_null=5, center=True)` (weighted sums of each node's squared
  correlations, with pair weights `mean_t r_ij**2` centered on their median
  and clipped at 0 for K >= 4 nodes, or used as they are with
  `center=False`; with centering, coupling equal for every pair gives zero
  magnitude columns; `signed=True` adds the correlations weighted by
  `mean_t r_ij`) and `tc.Communities(n=None, signed=False, n_null=5)`
  (mean squared correlation within each community of nodes,
  found by spectral k-means on the pair weights). Both are accepted as `rfun`
  by `reduce`, `timecorr` and `features`, as instances or as the names
  `"weighted_strength"`, `"weighted_strength_split"` and `"communities"`;
  `fit` and `transform` separate fitting from applying, and `fits_` records
  the fits (calling an instance, as `reduce`, `timecorr` and the decoders
  do, keeps only that call's fit; `fit` and each step of `features` and
  `two_path_step` append one). The decoders accept the summaries as `rfun`.
  In `features`, `n=None` sets the number of communities from a
  shift null (eigenvalues of the centered pair weights above the largest
  top eigenvalue of `n_null` surrogates whose columns are circularly
  shifted by lags that are distinct within a subject; a recording with
  fewer such lags than columns raises `ValueError` asking for `n`; fits
  counted this way record `"n_null"`). The null assumes that dependence
  between columns does not survive the shifts; if the data and surrogate
  top eigenvalues are exchangeable, any exceedance (chance at most
  `1 / (n_null + 1)`; ties make it smaller) gives at least two communities. The summaries reject
  NaN or infinite correlations, empty inputs and a non-finite
  `null_threshold` with `ValueError`.
- `features` and `two_path_step`: keyword-only `delta_weights_function` and
  `delta_weights_params` set the kernel of the path that feeds the next
  order (the delta path, also called the feeding path); `delta_weights_params`
  with `eye_weights` raises `ValueError`. A kernel other than `eye_weights`
  gives kernel-weighted correlations (`autofc`) instead of co-fluctuations,
  and is shaped by `include_timepoints`, `include_zero` and the events of
  each order.
  `rfun` and `delta_rfun` accept `"auto"` (analysis path: PCA at order 1 and
  `WeightedStrength(signed=True)` above; delta path: `Communities()`) and
  dicts keyed by order. A step whose reducer is a summary, with
  `combine=null_combine` and `cfun` `autofc`, `isfc` or `wisfc`, is computed
  in blocks of nodes (streamed), sized to `max_memory_gb` (the estimator's
  default block size when it is None), so that the `(K, K, T)` correlation
  array is held only when one block covers all K rows;
  `estimate_memory_gb(..., streamed=True)` gives the smallest peak of a
  streamed step, and `features` checks every streamed step whose input
  width is known in advance against `max_memory_gb` before computing
  anything (steps after `Communities`, whose width is found by the fit,
  when they run; after a callable reducer that is not a summary, this
  check takes the number of features of the data, K, as its output width,
  whatever the callable's own input width); a streamed step also counts the kernels of distinct
  per-subject event sets, the analysis results held while the feeding step
  runs, and the arrays kept in summaries' `fits_`. The default order-1 reducer (PCA)
  still holds `K(K+1)/2` columns per subject, so memory is bounded only when
  every step uses a summary (for example `rfun=tc.WeightedStrength(signed=True)`);
  the `MemoryError` of an analysis step that holds its correlations although
  a summary would stream it names `rfun=tc.WeightedStrength(signed=True)`
  and `rfun={1: tc.WeightedStrength(signed=True)}`.
  These are the defaults (see "Changed defaults of the higher-order
  pipeline" above).
- `tc.simulate_nested`: nodes in modules in supergroups, with an order-2
  signal `h` (per supergroup) that sets the time course of each module's
  coupling; returns `(data, truth)` with the modules, supergroups, `h` and
  the couplings `g`. Options for sign-flipping and transient coupling, a
  global signal, and coupling shared across subjects (`shared="order1"`).
- `tests/test_nested_recovery.py`: decoding of `h` from order-2 features
  across subjects on `simulate_nested` data, with controls from an
  independent `h`, the raw data and the order-1 features; recovery of the
  modules by `Communities`; the automatic community count on autocorrelated
  noise; and a streamed run at K = 702 (marked slow). These use
  `cfun=tc.autofc` (except the isfc cases) and an analysis kernel wider
  than the default (standard deviation 33 at T 1000); the K = 702 run uses
  weighted strength on both paths at every order, order 1 included (the
  default order-1 PCA step is estimated at 10.31 GB there). With the
  literal defaults, `h` was recovered from `D[2]` but not from `X[2]` (see
  the module docstring); structure above order 2 was not measured with the
  default reducers.

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
- With a delta kernel, a feature with a constant offset of about 1e10
  standard deviations or more was treated as constant (all zeros); the
  zero-variance test now uses the centered data, with a round-off allowance
  so that a constant feature stays constant.
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
  warns that a single fold would be circular and runs with `nfolds=2` (two
  groups, each decoded against the other).
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
- `format_data` failed on read-only pandas 3 arrays. An empty list (for
  example `timecorr([])`) raises a clear `ValueError` (it raised
  `IndexError`), and `simulate_data(K=1)` says that order 1 needs `K >= 2`
  (the message quoted the formula for higher orders).
- `folding_levels` and `folding_levels_ec` failed for scalar settings and
  `rfun=None`.
- Importing timecorr changed matplotlib's global font sizes (to 18-24
  points); it no longer changes any matplotlib setting.

### Documentation

- Every user-facing name now has an API page, including the default kernel
  parameter dictionaries (`gaussian_params`, `laplace_params`, `eye_params`,
  `t_params`, `mexican_hat_params`, `uniform_params`, `boxcar_params`), which
  are listed in a table of their keys, default values and meanings, and
  `iseven` and `isodd`. Each kernel function's docstring names its default
  dictionary and keys. `tests/test_api_docs.py` checks that every
  user-facing function has Parameters, Returns and Examples sections, that
  every user-facing name is listed on the API page, and that the parameter
  table matches the code.

### Dependencies and packaging

- Python 3.10 or later; continuous integration on Python 3.10 to 3.13 (the
  tests marked slow run in a separate job on Python 3.12).
- Added `pydata-wrangler`; `hypertools>=0.8`; `numpy`, `pandas` and
  `matplotlib` are listed explicitly.
- Removed `nose`, `sphinx` and `duecredit` from the install requirements.
  Tests run with `pytest` (`pip install timecorr[test]` adds `pytest` and
  `polars`); documentation dependencies are in the `docs` extra (Read the
  Docs installs that extra; the duplicate `docs/doc_requirements.txt` was
  removed).

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
