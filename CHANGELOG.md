# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Centering of `WeightedStrength`.** With `center=True` (the default,
  and the analysis reducer of `features` at orders >= 2), the pair weights
  are centered on the mean of their lower cluster instead of their median:
  the sorted pair weights are split into a lower and an upper group with
  the least total within-group sum of squares, and the lower group's mean
  is subtracted before clipping at 0. `center="median"` gives the previous
  rule; `center=False` is unchanged; other values raise `ValueError`.
  Median centering removed every weight of a smaller, more weakly coupled
  community when more than half the pairs lie within one larger community.
  Measured on 16 held-out seeds (planted communities of 18 and 6 nodes,
  coupling 0.5 and 0.25 varying over time, 2 subjects, T 1000): the small
  community's columns tracked its coupling at .425 +- .021 (median
  centering: .000, all of its weights 0), with the large community
  unchanged (.833 +- .007 versus .834 +- .007). On the six T 1000 order-2
  recovery cases of `tests/test_nested_recovery.py` the decode scores
  changed by -.002 to -.006, each below the standard error of its score
  (.022 to .038). The `WeightedStrength` Notes give the full comparison.
  `Communities` still centers its matrix `C` on the median.

- **Shift null with more columns than lags.** The shift null of the
  automatic community count (`Communities()` and split `WeightedStrength`
  without `n`, and the default feeding reducer) no longer raises
  `ValueError` when a step has more columns K than the n distinct lags in
  `[L, T - L]` (for example 22 or more features at T = 40, or the 150
  features of a T = 120 recording with the default feeding kernel). Each
  lag is then used `K // n` or `ceil(K / n)` times within a subject (a
  random permutation of the lags, repeated, truncated to K and shuffled
  with the same generator); columns that share a lag keep their dependence
  in that surrogate. With K = q n + r, the fraction of column pairs that
  share a lag is exactly [n q (q - 1) + 2 r q] / [K (K - 1)] (1/231 at
  K = 22, n = 21; below 1 / n, and near it only for K much larger than n),
  and one `UserWarning` per fit gives K, n and that fraction and says that
  the number of communities `n` can be given instead. With
  K <= n the lags are drawn as before (distinct, without replacement), so
  those results are unchanged. The other requirements (at least 4
  columns, T >= 40) still raise.

- **Documentation theme.** The Sphinx documentation uses Furo with the
  Contextual Dynamics Lab look of the htfa and hypertools documentation
  (Nunito Sans, lowercase light headings, green brand colors) in place of
  `sphinx-book-theme`; the `docs` extra installs `furo`.

### Added

- Graph measure `rfun="pagerank_centrality_abs"` (in `reduce`, `timecorr`
  and as `rfun` or `delta_rfun` in `features`): PageRank (damping 0.85,
  uniform prior) on the absolute correlations `|r|`. It is defined for
  every finite matrix, so it never falls back and never warns, unlike
  `"pagerank_centrality"` on signed correlations. It equals
  `"pagerank_centrality"` when no weight is negative. Timepoints with a
  non-finite entry give NaNs. `"pagerank_centrality"` is unchanged.

- Keyword `fit_index` of `WeightedStrength.transform` and
  `Communities.transform`: names the fitted unit whose pair weights (and,
  when signed, mean correlations) each unit uses. An int applies one fitted
  unit to every unit (for example a held-out subject), a sequence gives one
  fitted unit per unit (so a reordered list can keep each subject's own
  fit), and `"mean"` applies the mean over the fitted units. A sequence
  with the wrong number of entries, an index with no fitted unit, or any
  other value raises `ValueError`. The default (None) keeps the previous
  rule: by position when the counts are equal, the mean otherwise.

- Tutorial `docs/tutorial/htfa_network_animations.ipynb`: features of
  orders 1 to 4 (defaults of `features`, with `return_info=True`) of the
  HTFA node weights of the story-listening fMRI data analyzed by Owen,
  Chang & Manning (2021), downloaded from the data archive of the paper's
  repository; which nodes each order's features belong to; and one
  animation per order on the brain, drawn with the htfa toolbox (1920 x
  1080 H.264 videos at 30 frames per second, with linearly interpolated
  frames between timepoints, in `docs/_static/tutorial_media/`). The `docs`
  extra now includes `htfa[plot,brain3d]>=0.1.0`.

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
  The default feeding reducer needs at least 4 features and 40 timepoints
  at every feeding step that is computed; otherwise `ValueError` names `delta_rfun="weighted_strength"`
  (any number of features and timepoints) and the original pipeline's
  settings. At each default feeding step, the communities are fitted to
  that step's correlations; when they would emit fewer than 4 columns (at
  most half the step's input features; one when no eigenvalue exceeds the
  shift null), the step emits `WeightedStrength(split=False, signed=False,
  center=False)` of the same correlations instead (one column per input
  feature, from the same pair weights, used uncentered: no median
  clipping), so the feeding width never drops below 4 and, for example, 4 to 7
  input features complete at `order=3`. This also changes calls that
  completed before the fallback was added (earlier in the development of
  this release), not only calls that failed: on 4 independent noise
  features (T 100), `features(x, order=1, return_delta=True)` returned 1
  community column in `D[1]` and now returns 4 weighted-strength columns,
  and `features(x, order=2)` returns 8 columns in `X[2]` instead of 2.
  Summary output columns whose range over time is nonzero but at most
  `1e-9 * max(1, max|column|)` (round-off, for example when nearly
  identical input features give identical summaries) are set to their
  means exactly and counted in the constant-output `UserWarning`, so the
  next order meets exact zero variance (correlations set to 0) instead of
  amplified round-off. A `UserWarning` (one per order) is
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
  The `features` docs state that the default `cfun=isfc` correlates each
  subject's features with the other subjects' (structure shared across
  subjects) and `cfun=autofc` computes within-subject correlations, with
  measured recovery of `h` on `simulate_nested` data (12 seeds, 108 nodes,
  T 1000): from `X[2]`, -.035 +- .029 with `isfc` and .559 +- .022 with
  `autofc` when the order-1 couplings are drawn per subject
  (`shared="order2"`), .638 +- .028 and .702 +- .012 when they are shared
  (`shared="order1"`); from `D[2]`, .199 +- .021 and .293 +- .027.
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
- **Time-sliced estimation.** The estimator (`wcorr`, and through it
  `isfc`, `wisfc`, `autofc`, `timecorr` and the streamed steps of
  `features`) processes the output timepoints in chunks of 128 kernel rows
  and multiplies each chunk only with the samples in the span of columns
  where its rows have non-zero weight (NaN weights count as 0). Kernels with
  zero bands benefit: a Gaussian underflows to exactly 0 beyond a finite
  distance (beyond 192 samples for variance 25), one-sided kernels are zero
  on one side, and event-restricted kernels are block diagonal. When the
  spans cover more than half of the kernel's entries, or when the samples,
  their squares or their cross products are not all finite (where the full
  products give `0 * inf = NaN`), the full products are computed as before.
  Results equal the full computation up to the order of floating-point
  summation. Entries whose kernel-weighted variance nearly cancels (a small
  fraction of the kernel-weighted mean square: data constant within most of
  the kernel, or observed samples that carry only tail weights) are
  sensitive to that order in either path: measured on Linux x86-64 with
  OpenBLAS, such entries differed by up to 2.4e-6, while entries whose
  variance is at least 1e-3 of the mean square differed by at most 1e-12.
  The warnings are the same unless a variance lies within round-off of the
  estimator's zero tolerance (1e-10 of the mean square), where the two
  orders can disagree about setting an entry to 0. The spans are found one chunk
  of rows at a time and the chunk products are written in place, so the
  peak memory measured with `tracemalloc` is the same with and without
  slicing to within 0.01% (`autofc` on one subject of T 2000, K 4 with a
  width-12 boxcar kernel: 98,619,812 and 98,617,860 bytes; T 6000, K 4 with
  a Gaussian of variance 25: 871,844,068 and 871,837,860 bytes). Building
  the kernel and finding the spans still take time proportional to `T**2`.
  Measured:
  `features(order=1, cfun=autofc, rfun=WeightedStrength(signed=True))` on 2
  subjects of T 6000, K 96 (random data, default kernel, Apple Accelerate
  BLAS with default threads) took 38.6 s before and 12.3 s after. The
  memory estimate (`estimate_memory_gb`) is unchanged.

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
- `features`: `weights_params` and `delta_weights_params` also accept an
  order map, a dict whose keys are all integers in `1..order` and whose
  values are that order's kernel parameters (a dict, or None for the
  defaults), for example `{1: {"var": 100}, 2: {"var": 400}}`; orders
  missing from it use the default parameters. A dict without integer keys is
  the parameters of every order, as before; a dict mixing integer and other
  keys, boolean keys (Python or NumPy) or keys outside `1..order` raise
  `ValueError`. A dict whose keys are all integers is read as an order map,
  so a custom kernel whose own parameters are keyed by integers (previously
  passed through unchanged) must be wrapped in an order map, for example
  `{1: params, 2: params}`; only the outer dict is examined, and each value
  is passed to the kernel unchanged. `two_path_step` takes the parameters of
  one order and raises `ValueError` for a dict with integer keys (an order
  map, a dict mixing integer and other keys, or boolean keys; the message
  names which).
- `tc.two_path_step` (also `timecorr.higher_order.two_path_step`; one order
  of the pipeline) and
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
  and `rfun={k: tc.WeightedStrength(signed=True)}` for the failing order
  `k`, only when that order's analysis reducer comes from `rfun="auto"`
  (the caller did not choose it) and the change would let the order run:
  the feeding step is absent, streamed, or materialized but fitting the
  budget alone, and its reducer is not inherited from `rfun`
  (`delta_rfun=None`). With an explicitly given `rfun` (including the
  original pipeline's `"PCA"`) the `MemoryError` has no hint, and its text
  is unchanged.
  These are the defaults (see "Changed defaults of the higher-order
  pipeline" above).
- `tc.simulate_nested`: nodes in modules in supergroups, with an order-2
  signal `h` (per supergroup) that sets the time course of each module's
  coupling; returns `(data, truth)` with the modules, supergroups, `h` and
  the couplings `g`. Options for sign-flipping and transient coupling, a
  global signal, and coupling shared across subjects (`shared="order1"`).
- `tc.simulate_hierarchy`: nodes in nested groups of any depth `n =
  len(taus)` (modules of `nodes_per_module` nodes; `branching[l-1]`
  level-`l` groups per level-`(l+1)` group; `n_top` top groups), with one
  signal per group at every level. Each level's signal sets how strongly its
  children are coupled, on the timescale `taus[l-1]`; with two levels it has
  the form of the sustained coupling of `simulate_nested`. Subjects share
  the top signals (`shared="top"`), every signal (`"all"`) or nothing
  (`"none"`); returns `(data, truth)` with the group of each node at each
  level, the signals (separate arrays per subject, also when shared) and
  the node signs (`truth["signs"]`, one row per subject). Node signs are
  drawn per subject (`signs="private"`, the default) or once for every
  subject (`signs="shared"`; the shared signs come from the same random
  generator, so at the same seed every later private draw differs from the
  `signs="private"` output). `taus` and `branching` are lists, tuples or
  1-D numpy arrays of positive integers; other types (0-d arrays, sets,
  strings, ranges) raise `ValueError`.
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
- `wcorr`: a kernel row whose only weights are tiny (for example the far tail
  of a narrow Gaussian next to NaN samples, weights near 1e-165) made the
  product of the two variances underflow to 0, which raised a NumPy
  "divide by zero" `RuntimeWarning` and gave correlations of +-1 or NaN.
  Products that were subnormal but non-zero (weights near 1e-162, or data
  scaled by 2e-81) gave inaccurate correlations without a warning (0.145 in
  place of 0.160), and products that overflowed (data scaled by 1e80, or
  weights near 1e160) gave 0. The denominator is now `sqrt(var_a * var_b)`
  where that product is finite and at least 1e-290 (so ordinary results are
  unchanged bit for bit) and `sqrt(var_a) * sqrt(var_b)` elsewhere, and
  kernel rows whose largest absolute weight is below 1e-100 or above 1e100
  are divided by that weight before the moment sums (a correlation does not
  depend on the scale of its kernel row; other rows are used as given). The
  effective-sample-size check is computed on each kernel row scaled by its
  largest weight, so such rows are no longer reported as having an
  effective sample size below 2.
- `reduce(..., rfun='pagerank_centrality')`: for some valid signed
  correlation matrices (for example 2 x 2 with off-diagonal -3/37) the
  PageRank system `(I - 0.85 W diag(1 / column sums)) r = b` is singular
  (negative weights can cancel the column sums); the solve returned
  arbitrary values without a warning (0.474 and 0.526 for that symmetric
  matrix). A timepoint is now ill-posed when the solve fails, the solution
  is not finite, its sum is below 1e-12 times `max(1, max|r|)`, its relative
  residual `||B r - b|| / ||b||` exceeds 1e-8, or LAPACK's estimate of the
  reciprocal 1-norm condition number of the system is below 1e-14 (from a
  separate LU factorization; no singular value decomposition). Such a
  timepoint gets the minimum-norm least-squares solution
  (`numpy.linalg.lstsq`, relative cutoff 1e-14), normalized, when its
  residual and sum pass the same tests (0.5 and 0.5 for the example above),
  and otherwise the uniform vector 1/K. One `UserWarning` per `reduce` call
  gives the number of ill-posed timepoints, of least-squares solutions and of
  uniform vectors, and says that PageRank on signed correlations is not
  always defined. Outputs for other matrices are unchanged bit for bit. The
  `reduce` docstring states how each graph measure treats signed weights.

### Documentation

- `WeightedStrength.transform` and `Communities.transform`: the docstrings
  and the API overview state that, when the new list has as many units as
  the fit, units are matched to the fitted units by position (unit u gets
  the weights fitted to unit u, whether or not it is the same subject).
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
