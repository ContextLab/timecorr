# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-24

### Added

- Causal mode: `causal=True` (keyword-only; default False) in `tc.timecorr`,
  `tc.features` and `tc.two_path_step`, and in the estimator and correlation
  functions `tc.wcorr`, `tc.autofc`, `tc.isfc` and `tc.wisfc`. Row `t` of
  every output is computed from rows `0..t` of the data only. What is
  documented and tested:
  - when the rows after `t` are replaced by any values (finite, NaN or inf),
    row `t` is unchanged bit for bit, with every supported reducer;
  - when the rows after `t` are removed, row `t` is unchanged bit for bit
    as well with the built-in kernel functions, with every supported
    reducer and combine function (the graph measures and `tc.tstat_combine`
    included) and for smoothed values (`cfun=None`): the estimator computes
    a row with the same operations whatever the length of the series.
    Kernel rows are taken in groups with the shapes they have in a series
    without an end (rows and columns beyond the data are zeros), each row of
    features is a matrix product of its own, and the columns of a chunk of
    kernel rows are those of its first row; the effective sample sizes that
    `min_samples` is compared with are summed over the same columns; a
    streamed step leaves every correlation as the product of its own row of
    features gave it (no average with the transposed pair, which depended
    on the blocks of nodes, so on T and `max_memory_gb`: the two
    orientations of a pair agree to round-off); a fitted model reduces 128
    rows at a time, the last batch filled with rows of zeros. No output of
    a causal call depends on `max_memory_gb`. This is tested, not proven:
    it holds when the linear-algebra library computes a row of a matrix
    product of a given shape from that row's operands alone, and NumPy's
    elementwise functions a value from that value alone; the tests ran
    with Accelerate (macOS) and, in continuous integration, with the
    OpenBLAS of NumPy's Linux wheels. Tested at cuts at, just before
    and just after multiples of 16 and 128 rows, for every built-in kernel,
    events, NaN samples, `min_samples`, kernels wider than 512 timepoints,
    150 features (several blocks of features), every length from 1, fitted
    reducers, `refit_every`, `FeaturesModel.transform`, `min_samples` at
    the effective sizes that occur, several memory limits
    (`tests/test_causal_estimator.py`, `tests/test_causal_features.py`,
    `tests/test_causal_fit.py`), and in configurations drawn at random
    (`tests/test_causal_removal_random.py`: 300 draws in every run, 3000
    in the slow run);
  - with a kernel array or a kernel function of your own, the columns of a
    chunk are found from its weights and can depend on the length, so
    removal is documented to 1e-9 (absolute) for outputs that depend
    continuously on the correlations (no reduction, `"strength"`, the
    summaries and the linear models listed below once fitted), not for the
    eigenvector-based graph measures at a correlation matrix whose leading
    eigenvalues coincide, and to about 1e-9 of their magnitude for the t
    statistics of `tc.tstat_combine` where the subjects' correlations
    nearly coincide (no changed bit was measured with such kernels either);
  - with `include_zero=False`, row `t` is also unchanged when row `t` itself
    changes;
  - with a fitted reducer (`fit_timepoints=n`, below) this holds for rows
    `t >= n`; a row `t < n` is unchanged when rows from `n` on are replaced or
    removed;
  - smoothed values (`cfun=None`) are on the scale of the data (each chunk of
    128 rows is centered on the kernel-weighted mean at its first row, which
    is added back, never on a mean over later rows): replacement
    changes no bit, nor removal with the built-in kernel functions; with a
    kernel of your own they agree under removal to round-off relative to
    the magnitude of the data (to 1e-9 or 8 units in the last place of the
    largest magnitude of the data, whichever is larger);
  - a series of any length runs (one row included), and its output is the
    first rows of the output of a longer series.

  Largest differences recorded during development (`tests/test_causal_*.py`
  hold the checks): under replacement 0.0 in every case (the estimator;
  `features` at orders 1 to 4; fitted reducers; refits); under removal 1.6e-13
  for the estimator, at most 7.8e-12 for `features` with reducers that need
  no fit (at most 5.6e-13 on series with level steps of up to 1e8 standard
  deviations), 2.2e-14 with fitted reducers, and at most 2.9e-15 with refits,
  cuts at and around a refit row included.
- In causal mode:
  - the kernels are backward (`include_timepoints` may be left out;
    `"forward"` / `"post"` raises `ValueError`), and kernel parameters must be
    given, in `features` for both paths and every order: widths are not chosen
    from the data (`weights_params=None` raises `ValueError`; `{}` for a
    kernel without parameters);
  - `inf`, `-inf` and samples with `|x| > 1e150` are missing data, like NaN;
  - a timepoint with fewer than 3 observed timepoints in its kernel row is a
    row of NaN (the rows that are 0 without `causal`), the diagonal included.
    `min_samples` (keyword-only, default None) also sets the effective sample
    size `(sum |w|)**2 / sum w**2` a row needs (compared with a relative
    tolerance of 1e-12); a number that a built-in kernel of the call cannot
    reach with any number of earlier timepoints raises `ValueError` before
    anything is computed. In `features` the NaN rows of an order's feeding
    features are missing data for the next order, so each order has more of
    them; reducers skip them and return them as NaN;
  - a feature whose kernel-weighted standard deviation at a timepoint is below
    1e-9 of the magnitude of its kernel-weighted mean has correlation 0 with
    every feature there (a column constant up to round-off);
  - these raise `ValueError`, with a message that names an alternative:
    `tc.eye_weights` on either path (co-fluctuations use series z-scored over
    the whole recording: use a kernel with a width); `cfun=tc.wisfc` (its
    default subject weights use the whole recording: use
    `functools.partial(tc.wisfc, subject_weights=...)`); `events` given as
    features to segment (give boundaries; that they are known at their own
    time is the caller's responsibility; boundaries at or beyond the end of
    the data are ignored); a kernel array, or a kernel function of your own,
    with weight above the diagonal; a reducer that is fitted to data without
    `fit_timepoints`;
  - a function of your own as `cfun`, `combine`, `rfun` or `delta_rfun` is called as
    without `causal`, with one `UserWarning` that the guarantee does not cover
    what it does (removal is not covered either for a kernel function of your
    own whose weights depend on the number of timepoints);
  - with `max_memory_gb`, a step of a later order that is streamed and cannot
    fit raises `MemoryError` before order 1 runs, as without `causal`; this
    includes the steps streamed with
    `functools.partial(tc.wisfc, subject_weights=...)`, which are streamed in
    causal mode only;
  - without `max_memory_gb`, a streamed step reads every recording in the
    blocks of one with NaN samples, so that NaN samples after a row change
    no bit of it;
  - `tc.timecorr(..., causal=True, cfun=None)` (smoothing) does not use
    `rfun`, as without `causal`, and asks for no `fit_timepoints`;
  - the decoders have no causal mode.
- `fit_timepoints=n` (keyword-only; `timecorr`, `features`, `two_path_step`;
  needs `causal=True`): every reducer that is fitted to data is fitted on rows
  `0..n-1` and then applied, unchanged, to every row: hypertools models,
  `tc.WeightedStrength`, `tc.Communities` (with `n=None` its shift null runs
  on those rows) and the automatic reducers of `features` (`"auto"`, the
  defaults), whose feeding step chooses between communities and weighted
  strength on those rows. None and the graph measures need no fit. The fits
  of a series and of a longer series with the same first `n` rows are equal
  bit for bit.
  - Every reducer (a model, a summary, a graph measure) reads the rows without
    any NaN or infinite value only: any other row (a smoothed row of
    `cfun=None` in `features` can be partly NaN) is left out of the fit and is
    a row of NaN in the reduced output. (hypertools fills missing values in
    from every row it is given, and the summaries and two graph measures raise
    for them, so NaN in later rows would change earlier rows or stop the
    call.)
  - A model must be a linear map applied to each row by itself. The guarantee
    covers `"PCA"`, `"IncrementalPCA"`, `"TruncatedSVD"`, `"FastICA"` and
    `"FactorAnalysis"` (scikit-learn's classes of those names, by exact
    class: a subclass is not one). Any other model whose fitted estimator has
    `components_`, reproduces its training rows to 1e-9 and reduces rows one
    at a time as it reduces them together is used with one `UserWarning` per
    call that the guarantee does not cover what its transform does. A model
    without `components_`, or that fails a check, raises `ValueError`
    (`"UMAP"`, `"KernelPCA"`, `"Isomap"`, `"MDS"`, `"TSNE"`,
    `"SpectralEmbedding"`, `"LocallyLinearEmbedding"`).
  - A model asked for at least as many components as its input has columns
    is not fitted and returns the columns as they are, whatever the number
    of training rows (the number asked for is `n_components` given with the
    model, in its `kwargs` or of a model object, otherwise `ndims`). The
    widths are those of the call without `causal=True`: `n_components`
    given with the model is the width of a fitted model's output, and
    decides whether a model is fitted; when none is, the output has the
    columns of the input and zero columns up to `ndims` only. A model
    object given in a model dict is not fitted or changed by a causal call
    (a copy is fitted; a call without `causal=True` fits the object, as in
    0.3.1). Asked
    for fewer components than columns but more than there are defined
    training rows, it is fitted with as many components as those rows (an
    `n_components` given with the model is replaced), and zero columns are
    appended for the others, with the warning of `reduce`.
  - `"FastICA"` (the name, or a model dict with the name, scikit-learn's
    class or an object of it), when it is asked for fewer components than
    it has columns (with at least as many, the columns are returned as they
    are, as for every model), is given the columns that vary over the
    training rows first
    and is asked for no more components than the rank of the centered
    training rows (`numpy.linalg.matrix_rank`; at most the columns that
    vary and the rows less one: zero columns are appended for the others,
    with the warning of `reduce`; `ValueError` when no column varies).
    FastICA divides by the singular values of its rows, and a fit with more
    components than their rank did not reproduce its training rows (the
    call raised; found by the random removal test, for example 9 defined
    training rows and 15 components). scikit-learn's FastICA sets the
    sign of each component from its loading on the first column, which for
    vectorized correlations is a diagonal entry (1 in every row): the
    loading is 0 up to the round-off of the SVD, and where it was exactly 0
    the fit held NaN and the call raised ("A has a NaN entry": 136 of 240
    calls of `features(..., order=2, rfun="FastICA", delta_rfun="FastICA")`
    measured on macOS, one test on Linux). Calls without `causal=True` are
    not changed.
  - `"PCA"` is fitted with `svd_solver="full"` in causal mode unless a model
    dict gives a solver (the randomized solver that scikit-learn chooses for
    large inputs failed the check that a model reproduces its training rows,
    by 0.225 in the case measured).
  - An output column of a model that is constant on the training rows up to
    round-off (for example a principal component along which those rows do
    not vary) is frozen: a value within that tolerance of the column's
    training mean is returned as exactly that mean (without this, a null PCA
    component changed order 2 by 2.8 under removal in the case measured).
  - Output columns of a summary that are constant up to round-off are not set
    to their means in causal mode, and no constant-output warning is issued;
    `"constant_columns"` of `return_info` still lists constant columns.
- `refit_every=m` (keyword-only; `features`; needs `fit_timepoints=n`): the
  reducers are fitted again at rows `s_j = n + j * m`, each time on rows
  `0..s_j - 1`; rows `s_j` to `s_{j+1} - 1` of every output are those of the
  same call with `fit_timepoints=s_j` and no refit, and rows before `n` are
  those of the first fit. The output columns are those of the first fit: the
  choice of an automatic feeding step (the shift null runs in the first fit
  only), the number of components of a model, and the number of output
  columns of a communities step, which is its number of communities of 2 or
  more nodes. The number of clusters is not fixed: a later fit clusters its
  rows with the first fit's number of clusters `n0` and then, while it has
  not that many communities of 2 or more nodes, with `n0 + 1`, `n0 - 1`,
  `n0 + 2`, `n0 - 2`, ... up to `n0 + 5` and `n0 - 5` (between 2 and the
  number of features), and takes the first clustering that has them. A later
  fit that cannot have the columns of the first is not used: the fit before
  stays for its rows (at every order), `return_info` lists the row
  (`"refit_kept_previous"`), and one `UserWarning` per call names the rows
  left out and the step (order and path) that could not be fitted. The
  components of a model are matched to those of the fit before (largest sum of
  absolute cosines; a component is negated when its cosine is negative), and
  the communities of `tc.Communities` by the most shared nodes. Each fit is a
  pass over its rows and one over all rows.
- `tc.FeaturesModel` (`timecorr/model.py`) and `return_model=True` in
  `features` (needs `causal=True`): the call also returns the fitted pipeline
  (the order, kernels, `cfun`, `combine`, `include_zero`, `min_samples`,
  `ndims`, `max_memory_gb`, and the reducers of every order and path as
  fitted). `model.transform(data, return_delta=False, return_info=False, *,
  events=None)` computes the features of other data with the same number of
  columns and any number of rows, causally and without fitting anything;
  `transform` of the data of the call returns what the call returned. With
  `refit_every` the model holds the last fit. Attributes include
  `fit_timepoints`, `refits`, `fits` and `info`. A model can be pickled when
  the arguments of the call can (not with a lambda or a local function);
  unpickling needs the same versions of timecorr, hypertools and
  scikit-learn. A pipeline with a fitted summary takes the number of subjects
  it was fitted with.
- `tc.features(..., causal=True, return_info=True)`: every entry also has
  `"weights_params"` (the kernel parameters of the step) and `"warmup_rows"`
  (the number of rows of the step's output that are entirely NaN; one number
  per subject for list outputs); the entry of a step whose reducer was fitted
  has `"fit_timepoints"`, `"refit_rows"` and `"refit_kept_previous"`. With
  `refit_every`, what describes a fit is that of the last fit; when that fit
  took another number of clusters than the `n` of a `tc.Communities` object,
  `"effective"` shows the object with the number taken (for example
  `Communities(n=5, ...)` for `tc.Communities(n=6)`). Entries of calls without
  `causal=True` are unchanged.
- Timings recorded during development, on 5,033 timepoints of 49 features:
  `timecorr` with `causal=True` 0.32 s (0.2 s for the same backward kernel
  without `causal`; 1.77 s with 0.1% of the samples NaN); `features` at order
  3 with `"strength"` 1.85 s (1.15 s with a backward kernel and without
  `causal`); `features` at order 3 with the default reducers and
  `fit_timepoints=2500` 7.97 s and, in a later measurement, 9.8 s,
  `model.transform` 1.80 s, and `refit_every=250` (11 fits) 68.9 s. On 2,700 timepoints of 500 features,
  order 2 with the default reducers took 161 s.
- Tutorial `docs/tutorial/causal.ipynb` ("Causal features: no look-ahead"), on
  the daily returns of the 49 Industry Portfolios
  (`tc.datasets.industry_returns`): the look-ahead of a call without `causal`,
  reducers fitted on a training span and applied to later days with
  `model.transform`, warm-up rows and `min_samples`, refits with the default
  reducers (it prints the rows of the fits, the fits left out and the warning
  about them), and what causal mode does not accept. Its Colab cell installs `timecorr>=0.4.0`.
- `tests/test_packaging.py`: the declared hypertools requirement is at least
  1.1, and the installed `hypertools.reduce` has `random_state`,
  `return_model` and `model`.

### Changed

- Calls without `causal=True` return what they returned in 0.3.1: the
  stored-output tests (`tests/test_legacy_pipeline.py`,
  `tests/test_legacy_timecorr.py`, `tests/test_kernels_reference.py`,
  `tests/test_chunked_kernels.py`) pass unchanged. `min_samples`,
  `fit_timepoints`, `refit_every` or `return_model=True` given without
  `causal=True` raise `ValueError`.
- `tc.tstat_combine`: an entry whose values across subjects differ by
  round-off only now has the statistic of an entry whose values are equal:
  t = +inf or -inf with the sign of the mean of its values (p = 0), or 0
  (p = 1) for a mean of 0. Round-off is a range across the subjects of at
  most 128 eps (2.8e-14), or that times the largest magnitude when it
  exceeds 1 (`helpers._round_off_spread`). The bound is absolute for
  magnitudes up to 1: values closer than 2.8e-14 are treated as equal
  whatever their size, so `tc.tstat_combine` of 1e-16 and 2e-16 is +inf
  (3 in 0.3.1). The round-off of a correlation is absolute, at the scale
  of 1: the same correlation below 0.01 computed in two orders differed by
  at most 0.19 eps, which was up to 14,000 eps of its own size. In 0.3.1 such an entry had the mean divided
  by that round-off: a finite t of up to 1e16 in magnitude, and scipy's
  `RuntimeWarning` about precision loss. This is what differs from 0.3.1,
  with and without `causal=True`: the entries of `tc.isfc` with two subjects
  (its two correlation matrices are the same numbers computed in two
  orders) that were not equal bit for bit, 32 of 700 in one case measured
  (2 subjects, 70 timepoints, 4 features; |t| up to 1.3e16; the other 668
  were already +-inf) and 1190 of 39,000 in another (120 timepoints, 25
  features), and any entry of a list given to `tc.tstat_combine` with a range
  that small. Every other output is unchanged bit for bit (3000 random
  lists of 3 to 6 matrices, 288,000 entries, and the stored-output tests).
  The same correlation computed in two orders differed by at most 61 eps in
  33.8 million entries measured, except in the first defined row of
  `causal=True` (3 samples), where 13 entries differed by more than 128 eps
  (up to 3108 eps) and keep their finite statistic. A range of 1e-13 or
  more is a difference between subjects, as before.
- `install_requires`: `hypertools>=1.1` (was `hypertools>=0.8`; see Fixed).
  The `tutorials` extra keeps `hypertools>=1.2`.

### Fixed

- Versions 0.3.0 and 0.3.1 declared `hypertools>=0.8`, but
  `timecorr.helpers.reduce` calls `hypertools.reduce(..., random_state=0)`,
  an argument that hypertools has from version 1.0 on. With hypertools 0.8.x
  installed, every call that reduces with a model, the default `tc.features`
  call included, raised `TypeError: reduce() got an unexpected keyword
  argument 'random_state'`. The declared floor (`setup.py`,
  `requirements.txt`) is now 1.1, the oldest version the test suite runs
  with.

## [0.3.1] - 2026-11-23

### Added

- `tc.datasets` (`timecorr/datasets.py`), the example data of the real-data
  tutorials: `tc.datasets.fetch(url, path, sha256=None, member=None)` downloads
  a file once (through uniquely named `.part` temporary files and an atomic
  replace, so calls for one target at the same time do not disturb each other; with an optional SHA-256 check
  and optional extraction of one member of a zip archive) into the cache root
  (`$TIMECORR_DATA`, default `~/timecorr_data`; `tc.datasets.cache_dir()`);
  `tc.datasets.story_listening(n_nodes=100, return_nodes=False)` returns the
  HTFA node weights of the story-listening data of Owen, Chang & Manning
  (2021) (36 participants, 300 timepoints, the nodes spread over the brain;
  with `return_nodes=True` also their centers, widths and indices; `n_nodes`
  must be an integer between 1 and the 679 nodes that vary in every
  participant, otherwise `ValueError`);
  `tc.datasets.industry_returns(start="2005-01-01", end="2024-12-31")` returns
  the daily log returns of the 49 Industry Portfolios of the Kenneth R. French
  Data Library as a DataFrame. The loaders return the same arrays as the
  loading code of the tutorial notebooks and use the same cache files.
- `tc.features(..., return_info=True)`: every entry has `"constant_columns"`
  (the columns of the step's output that are constant over time in the
  returned arrays, `np.ptp(x, axis=0) == 0`; one list per subject for list
  outputs), and feeding entries have `"labels"` (the community of each input
  column when the output columns are the communities of a `Communities`
  object, the default included; None after a fallback, for other reducers and
  for the name `"communities"`). A second run with an explicit
  `tc.Communities()` is no longer needed to read the labels.
- `tc.render_video(draw, n_frames, path, fps=30, crf=23, hold=0, n_jobs=None,
  chunk_size=25, retries=0)` draws the frames of a video in parallel worker
  processes (`draw(frames, paths)` is called once per run of consecutive
  frames) and encodes them as H.264 with the ffmpeg of imageio-ffmpeg;
  `tc.show_video(path, width=960)` shows a video in a notebook (linked inside
  a Sphinx documentation tree and for a URL, embedded elsewhere);
  `tc.media_dir(name="tutorial_media")` gives the directory for the two
  cases. joblib, imageio-ffmpeg and IPython are imported on first use and
  stay in the `tutorials` extra. `render_video` gives `draw` consecutive frames also when drawing is started
  again with `retries`, and encodes under a temporary name next to the target,
  so a failed encoding leaves an existing video as it was;
  `show_video` percent-encodes and escapes the linked path.
- The plot style of the tutorials as a matplotlib style in the package:
  `plt.style.use("timecorr.tutorial")` (`timecorr/tutorial.mplstyle`).
  Importing timecorr does not apply it.

### Changed

- `install_requires`: `matplotlib>=3.7` (before: any version), the first
  version that loads a style from a package. The `test` extra also installs
  imageio-ffmpeg and ipython (for the tests of the video helpers).
- The eleven tutorial notebooks use the helpers above in place of their own
  copies of that code (about a fifth fewer code lines). What they compute,
  print and show is unchanged: every figure of the re-executed notebooks is
  pixel-identical to the one before, the printed text is the same but for the
  two lines named below, and the videos of `stocks` and `trajectories` are
  byte-identical (the brain renderings of `htfa_network_animations` differ a
  little from run to run, with the old code as with the new; its committed
  videos are kept).
  - The Colab install cell is one shell line, `!pip install -q
    "timecorr>=0.3.1" || pip install -q "timecorr @
    git+https://github.com/ContextLab/timecorr.git@master"` (with each
    notebook's extra and other packages), since the notebooks need release
    0.3.1. It no longer prints which source was installed, and pip's messages
    are no longer hidden: when PyPI does not have the release, pip's error
    shows before the install from GitHub.
  - The eight simulated-data notebooks set their plot style with
    `plt.style.use('timecorr.tutorial')`. Thumbnail cells show their figure
    with `plt.show()` after `%config InlineBackend.print_figure_kwargs =
    {'bbox_inches': None}` (the same 600 x 450 picture as the PNG they wrote
    to a buffer before).
  - `htfa_network_animations`, `stocks` and `trajectories` load their data
    with `tc.datasets`, write their videos into `tc.media_dir()` and show
    them with `tc.show_video`; `stocks` draws its frames with
    `tc.render_video` (frames in a temporary folder; before: under
    `$TIMECORR_DATA`). `stocks` keeps the two `Siccodes`
    text files of the library in the cache (before: their zip archives).
  - `htfa_network_animations` and `stocks` read the communities from the
    `"labels"` of `return_info` and no longer run `tc.features` a second
    time; the printed lines that compared the two runs are gone
    (`explicit Communities() at orders [2]: outputs equal to the default run;
    same warnings` and `same output as the default run: True`).
    `htfa_network_animations` and `trajectories` read the constant columns
    from `"constant_columns"`.
  - `higher_order`: `ridge_decode` fits scikit-learn's
    `Ridge(fit_intercept=False)` (before: a hand-written solve; the printed
    scores and the figure are the same).
  - `htfa_network_animations`, preview quality: the check for the
    full-quality videos of the online documentation sends a `User-Agent`
    header. Without one the documentation server answered HTTP 403, so the
    notebook reported that no full-quality video was online.

- Tutorials on the toolboxes' own animation calls (the `tutorials` extra now
  needs `htfa[plot,brain3d]>=0.2.0` and `hypertools>=1.2`):
  `htfa_network_animations.ipynb` draws each video with one
  `htfa.animate_network` call (per-node colors, per-frame radii, two views,
  caption, legend rows, parallel frames; x264's preset `slow`) in place of
  its own scene, frame composition and retry code, and its stills with
  `plot_network(..., views=, legend=, caption=)`; the layout of the videos is
  that of the earlier ones (brains and legend rows within 5 pixels).
  `trajectories.ipynb` draws each video with one
  `hyp.plot(..., panels=..., animate=True)` call; the videos show the line up
  to the current sample faint and the last 30 samples solid, without the head
  marker, the thicker tail and the one-second hold of the earlier videos, and
  the notebook no longer filters hypertools' messages (fixed in hypertools
  1.2).
- Tutorial `docs/tutorial/trajectories.ipynb`: the trajectories are drawn in
  hypertools' own cubes with `hyp.plot(..., panels=...)`, after reducing each
  order to three dimensions with `hyp.reduce` and rescaling each component to
  its own range (before: boxes fitted to each trajectory, drawn by the
  notebook). The three videos are 1920 x 600 with the four orders in a row.
  The text no longer says that hypertools' animations interpolate between
  samples (they do so only with more frames than samples).

## [0.3.0] - 2026-11-21

### Changed

- **Changed defaults of the higher-order pipeline (`features`,
  `two_path_step`).** The defaults are `rfun="auto"` (PCA at order 1,
  `WeightedStrength(signed=True)` at orders 2 and above),
  `delta_rfun="auto"` (`Communities()`, with the number of communities
  counted with a shift null) and a Gaussian feeding kernel
  (`delta_weights_function=None`) whose width, like the analysis kernel's,
  is set from each order's input (see "Automatic kernels" below; standard
  deviation 5 samples on white data at order 1). On white data `X[1]` is
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
  feature, from the same pair weights, used uncentered: no centering or
  clipping), so the feeding width never drops below 4 and, for example, 4 to 7
  input features complete at `order=3`. This also changes calls that
  completed before the fallback was added (earlier in the development of
  this release), not only calls that failed: on 4 independent noise
  features (T 100), `features(x, order=1, return_delta=True)` returned 1
  community column in `D[1]` and now returns 4 weighted-strength columns,
  and `features(x, order=2)` returns 8 columns in `X[2]` instead of 2.
  Summary output columns whose range over time is nonzero but within
  round-off (see "Constant summary columns" below; for example when nearly
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
  T 1000): from `X[2]`, -.031 +- .033 with `isfc` and .552 +- .023 with
  `autofc` when the order-1 couplings are drawn per subject
  (`shared="order2"`), .652 +- .026 and .700 +- .012 when they are shared
  (`shared="order1"`); from `D[2]`, .199 +- .021 and .293 +- .027. These
  were measured with the fixed kernels (variances 100 and 25); with the
  automatic kernels `X[2]` is the same there and `D[2]` with `isfc` gives
  .560 +- .020 (see "Automatic kernels").
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
  of rows at a time and the chunk products are written in place, so
  slicing adds no memory. Measured when slicing was added (T x T kernels):
  `features(order=1, cfun=autofc, rfun=WeightedStrength(signed=True))` on 2
  subjects of T 6000, K 96 (random data, default kernel, Apple Accelerate
  BLAS with default threads) took 38.6 s before and 12.3 s after. Built-in
  kernels are no longer built as T x T matrices (see "Kernels without a
  T x T matrix").
- **Automatic kernels in `features` and `two_path_step`.** Kernel
  parameters of None (`weights_params=None`, `delta_weights_params=None`,
  the defaults) with Gaussian kernels on both paths now set each order's
  kernels from that order's input: the feeding kernel's standard deviation
  is 5 autocorrelation widths of the input (the lag at which the mean
  autocorrelation of its columns falls below `exp(-1/2)`, at least 1), at
  most a hundredth of the input's rows but at least 5 rows (which wins for
  inputs of fewer than 500 rows), and the analysis kernel's is that or 10
  rows, whichever is larger. Constant columns (constant up to round-off,
  by the estimator's rule, whatever their value) are left out of the
  width, NaN samples count as the column's mean, and each column is scaled
  by its largest absolute value first, so the width does not depend on the
  units of the data. Orders missing
  from an order map (or mapped to None) are automatic too. What changes:
  white data at order 1 keep the earlier kernels (standard deviations 10
  and 5 samples); at orders 2 and above, whose input is smooth, the
  kernels can be wider: the feeding kernel is still 5 samples below 500
  timepoints and at most a hundredth of the recording above (10 samples
  at 1000 timepoints), and the analysis kernel stays at 10 samples up to
  1000 timepoints, so there `X[2]` on white data is unchanged and `D[2]`
  differs from 500 timepoints on; both are wider for longer recordings;
  autocorrelated data get wider kernels at order 1 too. The earlier fixed kernels are requested with
  `weights_params={"var": 100}, delta_weights_params={"var": 25}`. With
  another kernel function, or with `delta_weights_function=tc.eye_weights`
  (the original pipeline, whose kernels and reducers are not automatic),
  None means the kernel function's own defaults, as before. The numerical
  changes that apply to every kernel apply to the original pipeline too
  (for example the estimator's round-off rule of 48 eps: `wcorr` of
  `np.arange(5.) + 1e14` is 1, where a bound of 64 eps gave 0). When no kernel parameters and no events are given
  and `include_timepoints` is `"all"`,
  each order's feeding features are passed on at every `floor(sd / 2)`-th
  row while at least 1000 rows remain, so later orders of recordings of
  2000 or more timepoints are computed at fewer timepoints; every output is
  returned at all timepoints by linear interpolation (a computed row keeps
  its value; after the last computed row its value is held; a timepoint is
  NaN only when a computed row next to it is NaN). Giving any kernel
  parameter, events or a one-sided kernel makes every order be computed
  at every timepoint; on `simulate_nested` data (2 subjects, T 4000,
  `autofc`, order 3, seeds 0 to 2), giving the kernels the default chose
  changed `X[2]` by at most .034 (columns correlated above .9999 with the
  default's) and `X[3]` by at most .097 (column correlations .927 to
  .9995). `return_info=True`
  gives each automatic kernel's `"kernel_sd"` (rows of its order's input)
  and `"sampling"` (timepoints per row). The up-front memory checks of
  later orders (streamed and unreduced steps) use the fewest rows they can
  have, and each step is checked again with its rows when it runs.
  `max_memory_gb` also covers returning the outputs at all timepoints:
  each output is interpolated into one array, a chunk of rows at a time,
  and a run whose outputs at full length would not fit raises
  `MemoryError` naming their size (up front when both paths are
  unreduced, otherwise after the order that exceeds it).
  Measured on simulated nested hierarchies (ridge decoding of the top-level
  signal from one subject's features, applied to independent subjects and
  scored against their own top-level signal; mean +- standard error; the
  same data with the fixed kernels in parentheses; weighted strength in
  place of PCA at order 1 for memory): 108 nodes in two levels, T 6000,
  16 recordings: order-2 features .747 +- .005 (.496 +- .004); 256 nodes
  in three levels whose timescales grow by factors of 20 and 15 (5, 100,
  1500 samples), T 60,000, 8 recordings: order-3 features .662 +- .009
  (.002 +- .003), with the right number of communities at every order;
  216 nodes in three levels whose timescales grow by factors of 10 and
  7.5 (4, 40, 300 samples), T 12,000, 8 recordings: order-3 features
  .156 +- .028 (.007 +- .004). The null controls of these three were
  -.002 +- .015, -.046 +- .023 and .004 +- .026. On the first three-level
  design the kernels were 5, 46 and 526 samples wide. The two three-level
  designs differ in timescale ratios, children per group (4 against 3),
  nodes per module (4 against 6), number of nodes and length; which
  difference decides was not measured. Order 4
  was not recovered in the one design tried (162 nodes in four levels with
  3 children per group and 3 nodes per module, timescales 5, 100, 1100
  and 12,000 samples, T 400,000): the order-4 features of the default
  call gave .17 +- .07 over 2 recordings (null control -.04 +- .12; fixed
  kernels -.00 +- .01), with the right number of communities at every
  order (54, 18, 6, 2), in 1 h 47 min and 33 GB per call of 5 subjects;
  with the order-1 step replaced by the true modules, .02 +- .08 over 4
  other recordings.
  On `simulate_nested` data (T 1000, 12 seeds) `X[2]` is unchanged and the
  decoding score of `D[2]` rises from .199 +- .021 to .560 +- .020
  (control .002 +- .025; its spread over seeds is .086 against .051 with
  the fixed kernels). A limit of a two-hundredth of the rows instead of a
  hundredth gives the fixed kernels' numbers there and lowers the
  two-level result above from .775 +- .013 to .717 +- .012 (4 development
  recordings), so the hundredth is kept.
- **Constant summary columns.** A summary column counts as constant
  (set to its mean when its range is nonzero, and reported by the
  constant-output warning) when its range over time is at most
  `1e-9 * max|column|`, or at most 1e-12 of the largest absolute value in
  that unit's output (a column that is zero up to round-off). The rule is
  relative, so real variation of columns with values far below 1 is kept
  (a column of values near 3e-6 varying by 1e-6 of them).
- **Shift null evaluated at every s-th timepoint.** In `features` and
  `two_path_step`, the surrogates of the automatic community count
  average their squared correlations over every `s`-th timepoint, where
  `s` is the largest step at which two rows of the kernel still overlap by
  `exp(-1/4)` (the standard deviation, rounded down, for a Gaussian
  kernel; 1 for `eye_weights`), for built-in kernels without events,
  one-sided restriction or `exclude_timepoints`. Every other kernel (an
  array, a kernel function of your own, a restricted kernel), whose rows
  can differ (zero rows, rows with a single weight), is evaluated at every
  timepoint. The step is computed only when the null runs, from one FFT of
  a kernel row. On 96 noise inputs of four sizes the threshold was 1.000 to
  1.002 times the threshold from every timepoint (standard errors .001 to
  .003), and the communities found on 24 structured inputs were the same;
  the default call on 5 subjects of 6000 x 108 took 45 s instead of 81 s
  (the null: 11 s instead of 44 s; one run each on a shared machine). Thresholds, and so community counts
  near the threshold, can differ from the previous version. Phase
  randomization was compared with the circular shift on the same inputs
  and found the same counts (one worse of 24) and the same rate of false
  structure on noise (20 % against 18 % for any eigenvalue above the
  threshold; neither gave the 4 communities at which the default reducer
  emits community columns), so the circular shift is kept.
- **Kernels without a T x T matrix.** For the built-in kernel functions
  (`gaussian_weights`, `laplace_weights`, `t_weights`,
  `mexican_hat_weights`, `boxcar_weights`, `uniform_weights`,
  `eye_weights`), `timecorr`, `features`, `two_path_step`, the streamed
  summaries and the decoders no longer build the T x T kernel. The
  estimator computes it 128 rows at a time, over the columns where those
  rows can have weight, with `include_timepoints`, `include_zero`,
  `exclude_timepoints` and `events` applied, and keeps at most 32 MB of
  such blocks, so the kernel's memory grows linearly in T. (The decoders
  still compare every pair of timepoints, so their memory still grows with
  `T**2`.) A kernel given as a T x T array (a kernel function of your own)
  is read in the same chunks, so a built-in kernel and the same kernel as
  an array give identical correlations. The estimator's results can differ
  from earlier versions in the last bits. On 3 subjects of T = 1,500 and
  K = 12 (Apple M2 Max, Accelerate), compared with the previous version:
  `timecorr` with `isfc` and the default kernel, and order 1 of `features`
  with its defaults, were identical; `timecorr` with `autofc`, a one-sided
  Laplace kernel (scale 20), events and NaN samples differed in 9,973 of
  351,000 entries, by at most 1.7e-16 (relative 5.4e-15 where the value is
  at least 0.01 in magnitude); order 2 of `features` differed in 101,848
  of 108,000 entries, by at most 7.8e-16 (relative 2.6e-14). Smoothing
  (`cfun=None`) is computed in the same chunks, and an infinite sample
  still enters every row's sums, as in the full product. Run time grows
  linearly in T once T is well above the largest lag with a non-zero
  weight in float64 (192 samples for a Gaussian of variance 25, 385 at
  variance 100). The uniform kernel, which has weight at every lag, still
  takes time proportional to `T**2`, and wide kernels take longer than
  before (best of 3 runs, `timecorr` on 3 subjects of T = 4,000 and
  K = 10: uniform kernel with `autofc` 0.29 s before, 0.45 s after;
  Laplace kernel at its default scale with `isfc` 0.44 s before, 0.64 s
  after; t kernel at its default df with `isfc` 0.54 s before, 0.63 s
  after; on one subject of T = 10,000, the uniform kernel took 1.07 s
  before and 0.96 s after, and the Gaussian of variance 100 1.27 s before
  and 0.07 s after). A correlation function of your own gets T x T arrays
  as before. An event restriction of `None` keeps the events a kernel
  already has; labels that return to an earlier value (such as 0, 1, 1, 0)
  form separate blocks, as the boundaries do.
- **Memory estimates.** `estimate_memory_gb` and the `max_memory_gb`
  checks count the kernel memory (`3 T**2` elements up to T = 1,270,
  then at most 32 MB plus 4 blocks of 128 x T per correlation call).
  `estimate_memory_gb` has a keyword `weights_function` that counts a
  kernel function of your own as T x T arrays. The estimate of a step
  that holds its correlations counts 7 block temporaries of the
  estimator (measured with `tracemalloc`: 7.3 to 8 blocks, the kernel's
  blocks included), the temporaries of the Fisher z symmetrization of
  `isfc` and `wisfc` (the default `cfun`) with several subjects when they
  are larger, and the stacked subjects and one subject's reference that
  the estimator holds (K = 60, T = 5,000, 6 subjects, `isfc`: 1.130 GB
  traced, 1.151 GB estimated; K = 100, T = 200, one subject, `autofc`:
  0.1338 GB traced, 0.1378 GB estimated; on 11 sizes from K = 10 to 300
  and T = 200 to 10,000 the traced peak was 0.35 to 0.99 of the
  estimate). For example,
  `estimate_memory_gb(100, 2, 200, n_subjects=3)` is
  `[0.0, 0.18, 104.372]`. With a correlation function of your own
  (`cfun`), which receives T x T kernel arrays, the guards and
  `estimate_memory_gb` (keyword `cfun`) count those arrays; smoothing
  (`cfun=None`) with a built-in kernel is read in chunks and counted so. A
  feeding kernel function of your own counts only at the orders whose
  feeding path runs (also with `streamed=True`), and the T x T kernels of
  per-subject events only on the path whose kernel is a function of your
  own.
- **Memory of the combine functions.** `estimate_memory_gb` has a keyword
  `combine`, and the `max_memory_gb` checks of `features`, `two_path_step`
  and the decoders count the temporaries of `corrmean_combine`,
  `mean_combine` and `tstat_combine` (measured with `tracemalloc`: about
  5.3, S + 1 and 7.7 to 9.8 times the S subjects' results; at K = 300,
  T = 300 and 3 subjects, `corrmean_combine` traced 2.05 GB against an
  estimate of 2.12 GB, and `tstat_combine` 3.21 GB against 3.31 GB). A
  `combine` function of your own cannot be counted.
- **Streamed `wisfc` within its budget.** The default subject weights of
  `wisfc` (correlations between the subjects' static correlation matrices)
  are computed in the streamed step's blocks of rows, so a streamed `wisfc`
  step stays within its `max_memory_gb` (traced: 0.54 times it at K = 150,
  T = 200, 3 subjects; with the estimator's default blocks it was 1.84
  times), and they warn about constant or sparse data as the unstreamed
  computation does.
- **`exclude_timepoints` with a non-integer value.** It is rounded to the
  nearest integer, as documented, for a kernel function of your own too
  (it raised `TypeError` before; 2.5, for example).
- **Events applied twice.** `events.mask_kernel` applied to a kernel that
  already has events keeps the weights only between timepoints that share
  a block of both, for a built-in kernel as for an array.
- Measured on one subject with K = 100 features and a Gaussian of variance
  100 (Apple M2 Max, 2 BLAS threads, peak resident memory of the process),
  before and after:

  | call | T | before | after |
  |-|-|-|-|
  | `timecorr` | 3,000 | 1.8 s, 1.71 GB | 0.9 s, 1.49 GB |
  | `timecorr` | 20,000 | 14.0 s, 15.5 GB | 4.5 s, 3.88 GB |
  | `timecorr` | 100,000 | not run (estimate 253 GB) | 38 s, 14.8 GB |
  | `features`, order 2 | 3,000 | 9.6 s, 1.13 GB | 6.7 s, 0.92 GB |
  | `features`, order 2 | 20,000 | 138 s, 12.7 GB | 55 s, 1.13 GB |
  | `features`, order 2 | 100,000 | not run (estimate 564 GB) | 239 s, 2.13 GB |

  (`features` with `cfun=tc.autofc`, `rfun=tc.WeightedStrength(signed=True)`,
  `delta_rfun="weighted_strength"`, `max_memory_gb=None`; one run each on a
  shared machine, so the times vary by up to about a factor of 2 between
  runs; the estimates
  are those of `estimate_memory_gb` before the change. Most of the
  `timecorr` memory at T = 100,000 is the (K, K, T) correlations and their
  vectorized copy. The resident memory is larger than the memory allocated
  at the peak (traced with `tracemalloc`: 12.78 GB for `timecorr` at
  T = 100,000, when the resident memory reached 14.82 GB), because macOS
  keeps freed pages in the process until the system reclaims them: after
  that call the process still held 6.82 GB while 4.04 GB was allocated.)
- **Shift null with more columns than lags.** When a step has more
  columns K than the n distinct lags in `[L, T - L]` (for example 22 or
  more features at T = 40, or the 150 features of a T = 120 recording with
  the default feeding kernel), the shift null of the automatic community
  count (`Communities()` and split `WeightedStrength` without `n`, and the
  default feeding reducer) uses each lag `K // n` or `ceil(K / n)` times within a subject (a
  random permutation of the lags, repeated, truncated to K and shuffled
  with the same generator); columns that share a lag keep their dependence
  in that surrogate. With K = q n + r, the fraction of column pairs that
  share a lag is exactly [n q (q - 1) + 2 r q] / [K (K - 1)] (1/231 at
  K = 22, n = 21; below 1 / n, and near it only for K much larger than n),
  and one `UserWarning` per fit gives K, n and that fraction and says that
  the number of communities `n` can be given instead. With
  K <= n the lags are distinct (drawn without replacement). The other requirements (at least 4
  columns, T >= 40) still raise.

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
  every subject's results, the kernels (see "Memory estimates"), the block
  temporaries and the temporaries of the built-in `combine` functions; it
  does not count reducers or a `combine` function of your own.
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
  `eye_weights`, the feeding path's block temporaries are counted; keywords
  `weights_function`, `combine` and `cfun`, see "Memory estimates").
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
- Squared-correlation summaries (`timecorr.summaries`, exported from
  `timecorr`): `tc.WeightedStrength(split=False, signed=False, n=None,
  n_null=5, center=True)` (weighted sums of each node's squared
  correlations, with pair weights `mean_t r_ij**2` centered and clipped
  at 0 for K >= 4 nodes, or used as they are with `center=False`;
  with centering, coupling equal for every pair gives zero magnitude
  columns; `signed=True` adds the correlations weighted by
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
  shifted by lags that are distinct within a subject when there are at
  least as many lags as columns (otherwise see "Shift null with more
  columns than lags"; the timepoints the surrogates are averaged over: see
  "Shift null evaluated at every s-th timepoint"); fits counted this way
  record `"n_null"`). The null assumes that dependence
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
- Centering of the `WeightedStrength` pair weights (`center`, K >= 4
  nodes): with `center=True` (the default, and the analysis reducer of
  `features` at orders >= 2), the sorted pair weights are split into a
  lower and an upper group with the least total within-group sum of
  squares, and the lower group's mean is subtracted before clipping at 0;
  `center="median"` subtracts the median instead; `center=False` uses
  the weights as they are; 1 and 0 mean True and False; other values
  raise `ValueError`. The split is searched on the weights mapped to
  [0, 1], so it does not depend on their scale (pair weights of
  co-fluctuations from 1e-150 to 1e150 give the same split), and splits
  within a relative 1e-9 of the best are tied, the smallest lower group
  winning; the level jumps where the best split changes.
  `Communities` centers its matrix `C` on the median. Median centering
  removes every weight of a smaller, more weakly coupled community when
  more than half the pairs lie within one larger community. Measured on
  16 held-out seeds (planted communities of 18 and 6 nodes, coupling 0.5
  and 0.25 varying over time, 2 subjects, T 1000): the small community's
  columns tracked its coupling at .425 +- .021 with the default (median
  centering: .000, all of its weights 0), with the large community
  unchanged (.833 +- .007 versus .834 +- .007). On the six T 1000
  order-2 recovery cases of `tests/test_nested_recovery.py` the decode
  scores of the default were .002 to .006 below those of median
  centering, each difference below the standard error of its score (.022
  to .038). The `WeightedStrength` Notes give the full comparison.
- Graph measure `rfun="pagerank_centrality_abs"` (in `reduce`, `timecorr`
  and as `rfun` or `delta_rfun` in `features`): PageRank (damping 0.85,
  uniform prior) on the absolute correlations `|r|`. It is defined for
  every finite matrix, so it never falls back and never warns, unlike
  `"pagerank_centrality"` on signed correlations. It equals
  `"pagerank_centrality"` up to round-off when no weight is negative.
  Timepoints with a
  non-finite entry give NaNs. Each column is scaled by its own largest
  weight (so a column of 1e-20 next to one of 1e308 gives 0.5 and 0.5), and
  zero nodes give an empty array.
- Keyword `fit_index` of `WeightedStrength.transform` and
  `Communities.transform`: names the fitted unit whose pair weights (and,
  when signed, mean correlations) each unit uses. An int applies one fitted
  unit to every unit (for example a held-out subject), a sequence gives one
  fitted unit per unit (so a reordered list can keep each subject's own
  fit), and `"mean"` applies the mean over the fitted units. A sequence
  with the wrong number of entries, an index with no fitted unit, or any
  other value raises `ValueError`. The default (None) matches units by
  position when the counts are equal and uses the mean otherwise.
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
- Tutorial `docs/tutorial/htfa_network_animations.ipynb`: features of
  orders 1 to 4 (defaults of `features`, with `return_info=True`) of the
  HTFA node weights of the story-listening fMRI data analyzed by Owen,
  Chang & Manning (2021), downloaded from the data archive of the paper's
  repository; which nodes each order's features belong to; and one
  animation per order on the brain, drawn with the htfa toolbox (1920 x
  1080 H.264 videos at 30 frames per second, one frame per timepoint
  (300 frames, 10 s), with edges at every order (order-k correlations
  between order-(k-1) features), in `docs/_static/tutorial_media/`).
- Tutorials `quickstart` (one dataset, several subjects, higher orders),
  `higher_order` (recovery of simulated structure by order; it prints the
  order-2 recovery over 5 seeds), `events` (event-restricted kernels),
  `kernels`, `subjects`, `reduce`, `simulate` and `decoding` in
  `docs/tutorial/`. Every public name of the package is used (as
  `tc.<name>`, imported or called) in a code cell of at least one tutorial;
  `tests/test_docs_tutorials.py` checks this, that the gallery and the
  notebooks match, and that each notebook has one thumbnail cell with an
  image, a Colab badge naming its own file and no error outputs.
- Tutorial `docs/tutorial/stocks.ipynb`: 20 years (2005-2024) of daily
  returns of the 49 US industry portfolios of the Kenneth R. French Data
  Library (downloaded at run time): correlations between industries and
  around market crises, `tc.features` at orders 1 to 3, and a video of the
  market as a network (`docs/_static/tutorial_media/stocks_network.mp4`).
- Tutorial `docs/tutorial/trajectories.ipynb`: features of orders 1 to 4
  of three real datasets (story-listening brain data, the stock market,
  human movement from the CMU Motion Capture Database), reduced to three
  dimensions and animated as trajectories with hypertools (three videos in
  `docs/_static/tutorial_media/`); the edge left out at each end is twice
  the largest kernel standard deviation chosen by `tc.features`.
- `tutorials` extra (`htfa[plot,brain3d]>=0.1.0`, `hypertools>=1.1`,
  `pyvista>=0.43`, imageio-ffmpeg, pillow, joblib, ipython): what the
  real-data notebooks (`htfa_network_animations`, `stocks`,
  `trajectories`) import beyond the runtime dependencies. In Google Colab,
  their first code cell installs `timecorr[tutorials]`; the other notebooks
  install `timecorr` (with `polars` for `subjects`, and
  `timecorr[segment]` for `events`).

### Removed

- The sphinx-gallery example gallery (`examples/`, `docs/auto_examples/`)
  and the `sphinx-gallery` dependency. What the examples showed is in the
  tutorials. The simulated-domains tutorial (`applications_tutorial.ipynb`)
  is removed.

### Fixed

- Graph measures (`rfun` `"strength"`, `"eigenvector_centrality"`,
  `"pagerank_centrality"`, `"pagerank_centrality_abs"`) with one feature
  (K = 1) raised `TypeError` or `ValueError`: the 1 x 1 matrix of each
  timepoint was squeezed to a scalar. They now return one column:
  `"strength"` gives the matrix entry (1 for correlations) and the other
  measures give 1.
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
- A feature equal to a constant up to round-off (samples `c (1 + d)` with
  `|d|` of a few float64 spacings, for example 1 times `1 + 1e-16` noise)
  had a kernel-weighted variance of round-off size rather than exactly 0,
  so the estimator returned ratios of round-off as its correlations (up to
  0.77 in magnitude with a Gaussian kernel of variance 10 on 80
  timepoints, where an exactly constant feature gives 0). A kernel-weighted variance at most `(48 eps)**2` times the
  kernel-weighted mean square of the raw samples (`eps` is the float64
  machine epsilon) now counts as zero, as an exact zero does: correlation 0
  and the same count in the zero-variance warning, for array and built-in
  kernels, with or without NaN samples, and in the streamed summaries. The
  rule is relative, so it does not depend on the scale of the feature
  (tested at 1e150 and 1e-150). The z-scores of the delta kernel and of
  `segment` use the same rule; `segment` previously turned such a feature
  into unit-variance round-off noise. The bound covers the round-off the
  library produces: sd / rms of its outputs that are constant in exact
  arithmetic reached 38 eps. A feature on an offset of more than about
  9.4e13 local standard deviations counts as constant (`wcorr` of
  0, ..., 4 on an offset of 1e14, ratio 1.41e-14, is 1). When the rule was
  introduced (with a bound of 64 eps), on 4,566 outputs of random data
  without such features (12 seeds; 8 kernel functions; `wcorr`, `timecorr`
  with `isfc`, `wisfc` and `autofc`, and `features`) every value and
  warning was identical to the code before it.
- Fisher z averaging (the symmetrization of `isfc` and `wisfc`,
  `corrmean_combine`, the streamed summaries) and `tstat_combine` gave an
  infinite z to a correlation of exactly +-1 but a finite one to a value
  one float spacing away: averaging +1 with -1 gave 0, but +1 with
  -0.9999999999999998 gave 1. Values at most 128 eps (2.8e-14) below +-1,
  or at most 1e-12 above, now count as +-1 (one helper,
  `helpers._correlation_z`): on outputs of `timecorr` and of the combine
  functions that are +-1 in exact arithmetic, 1 - |r| reached 79 eps with
  Accelerate and OpenBLAS. So +1 averaged with -1 + 1 ulp is 0,
  `tstat_combine` gives +inf for 1, 1 - 1.1e-16 and 1, and values further
  below 1 are correlations: `tstat_combine` of 1 - 1e-13, 0.5 and 0.4 gives
  t = 1.098 (p = 0.387) and `corrmean_combine` of 1 - 1e-13 and
  -1 + 9e-13 gives 0.49993. `r2z` itself is unchanged.
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
- `uniform_weights` and `eye_weights` build their T x T output directly
  (their peak was twice the output, from a T x T matrix of lags).
- `exclude_timepoints` of inf or NaN raises `ValueError` for every kernel
  function (built-in kernels accepted it; a function of your own raised
  `OverflowError`).

### Documentation

- `WeightedStrength.transform` and `Communities.transform`: the docstrings
  and the API overview state that, when the new list has as many units as
  the fit, units are matched to the fitted units by position (unit u gets
  the weights fitted to unit u, whether or not it is the same subject);
  `fit_index` chooses the fitted units explicitly.
- Every user-facing name now has an API page, including the default kernel
  parameter dictionaries (`gaussian_params`, `laplace_params`, `eye_params`,
  `t_params`, `mexican_hat_params`, `uniform_params`, `boxcar_params`), which
  are listed in a table of their keys, default values and meanings, and
  `iseven` and `isodd`. Each kernel function's docstring names its default
  dictionary and keys. `tests/test_api_docs.py` checks that every
  user-facing function has Parameters, Returns and Examples sections, that
  every user-facing name is listed on the API page, and that the parameter
  table matches the code.
- One-sided kernels (`include_timepoints="backward"` or `"forward"`): the
  `timecorr`, `features` and `wcorr` docstrings state what is guaranteed.
  The estimator centers each feature on its mean over all timepoints, so an
  estimate near the zero-variance threshold (variance about 1e-10 of the
  mean square of the centered feature) can depend on timepoints outside its
  kernel row: in `features(x, order=2, cfun=tc.autofc, rfun=None,
  delta_rfun=None, include_timepoints="backward")` on 300 x 5 noise, rows 69,
  72, 78 and 79 of `X[2]` differ between `x` and `x[:200]` by more than
  1e-9 (by 1 at row 79).
  Estimates whose variance is at least 1e-3 of that mean square do not
  depend on the excluded timepoints (to 1e-9, for `autofc`, `isfc` and
  `wisfc` with fixed subject weights, at orders 1 and 2;
  `tests/test_estimator_causality.py`). `wisfc`'s default subject weights
  use the static correlations over all timepoints.
- **Documentation theme.** The Sphinx documentation uses Furo with the
  Contextual Dynamics Lab look of the htfa and hypertools documentation
  (Nunito Sans, lowercase light headings, green brand colors) in place of
  `sphinx-book-theme`; the `docs` extra installs `furo`.
- **Tutorials are the gallery.** `docs/tutorials.rst` shows the notebooks
  as a gallery of cards (nbsphinx `nbgallery`); each card shows a dedicated
  4:3 picture drawn by its notebook (the cell tagged `nbsphinx-thumbnail`)
  and opens it; the Colab install cell of each notebook installs from PyPI,
  falls back to GitHub, prints which source it installed, and shows pip's
  output only when both installs fail. The
  tutorials were rewritten with shorter text: `timecorr_notebook` became
  `subjects` and `synthetic_data_tutorial` became `kernels`.
- **Documentation extras.** The `docs` extra holds only what the Sphinx build
  needs (sphinx, furo, numpydoc, nbsphinx, ipython); the notebooks are
  rendered from their saved outputs. `.readthedocs.yaml` installs the
  package with the `docs` extra, installs pandoc, and fails on warnings.

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
