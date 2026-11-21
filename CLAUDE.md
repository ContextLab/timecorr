# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

timecorr is a Python toolbox for computing and exploring correlational structure in timeseries data. The `timecorr` function computes kernel-weighted dynamic correlations from timeseries observations, and `features` computes higher-order dynamic correlations through the two-path pipeline of Owen, Chang & Manning (2021), Fig. 10 (by default with Gaussian kernels whose widths are set from each order's input and squared-correlation summaries at orders >= 2; the original pipeline is selected with `rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights`).

## Development Commands

### Setup
```bash
# Install the package in development mode with test and documentation extras
pip install -e ".[test,docs]"
# or only the runtime dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run the whole suite (includes slow integration tests that need several GB of memory)
pytest

# Skip the slow tests
pytest -m "not slow"

# Run the docstring examples
pytest --doctest-modules timecorr/
```

### Documentation
```bash
# Build the HTML documentation (notebooks are rendered from their saved outputs)
cd docs
make html

# Execute a tutorial notebook in place
jupyter nbconvert --to notebook --execute --inplace docs/tutorial/<notebook>.ipynb
```

### Building
```bash
# Build distribution packages
python setup.py sdist bdist_wheel
```

## Code Architecture

### Core Components

- **`timecorr/timecorr.py`**: the `timecorr()` function, the primary entry point for computing dynamic correlations (or, with `cfun=None`, kernel-weighted smoothing).
- **`timecorr/_kernels.py`**: `LagKernel`, the kernels of `timecorr`, `features`, `two_path_step`, the streamed summaries and the decoders for the built-in kernel functions: the lag profile (length T; the kernel functions in `helpers` build their T x T matrices from the same profile functions), `include_timepoints`/`include_zero`/`exclude_timepoints` and the event labels (`events.mask_kernel` accepts it), with `block(t0, t1, c0, c1)`, `span(t0, t1)` and `key(t0, t1)` (chunks away from edges and event boundaries share one block; `with_events` intersects the events it already has); no T x T array is built. A kernel function of the user's gives T x T arrays (`helpers._lag_kernel` returns None for it), wrapped in `ArrayKernel` (the same `shape`/`chunks`/`block` interface; every chunk spans all columns; blocks are read-only views); `as_kernel` wraps either, and `blocks(kernel, size, transform, full)` is the one chunk iterator (blocks computed once per key), used by the estimator, the smoothing of `timecorr` and the row statistics. A custom `cfun` receives T x T arrays.
- **`timecorr/helpers.py`**:
  - Kernel functions (`gaussian_weights`, `laplace_weights`, `eye_weights`, `mexican_hat_weights`, `t_weights`, `uniform_weights`, `boxcar_weights`) and `one_sided`
  - The estimator `wcorr` and correlation functions (`isfc`, `wisfc`, `autofc`)
  - Combine functions (`null_combine`, `corrmean_combine`, `mean_combine`, `tstat_combine`)
  - `reduce` (dimensionality reduction), `format_data` (input formatting via pydata-wrangler), `mat2vec`/`vec2mat`, `smooth`, `plot_weights`
  - Re-exports of the decoders from `decoding.py`
- **`timecorr/higher_order.py`**: the two-path pipeline: `features`, `two_path_step`, `estimate_memory_gb` (`streamed=True` for the peak of a streamed step; `combine=` counts the temporaries of the built-in combine functions, not of a function of your own). Keyword-only `delta_weights_function` / `delta_weights_params` set the kernel of the delta (feeding) path (default None: `gaussian_weights`, giving `autofc` correlations; `eye_weights` gives co-fluctuations). Kernel parameters of None (the defaults `weights_params=None`, `delta_weights_params=None`) with Gaussian kernels on both paths are automatic (`_auto_sd`, `_auto_params`): at each order the feeding kernel's standard deviation is 5 autocorrelation widths of that order's input, at least 5 rows and at most a hundredth of its rows, and the analysis kernel's is that or 10 rows, whichever is larger (white data: 10 and 5 at order 1, the fixed kernels of earlier versions, which are requested with `weights_params={"var": 100}, delta_weights_params={"var": 25}`; recordings of up to 1000 timepoints: at most 10 samples at every order). With no kernel parameters and no events given, `features` passes each order's feeding features on at every `floor(sd / 2)`-th row while at least 1000 rows remain and returns every output at all timepoints by linear interpolation (`return_info` entries of automatic kernels have `kernel_sd` and `sampling`). With another kernel function, or `eye_weights` on the feeding path, None is that function's own defaults. `weights_params` / `delta_weights_params` also accept order maps (dicts whose keys are all integers in 1..order; missing orders use the default; `two_path_step`, exported as `tc.two_path_step`, raises for one). `rfun` / `delta_rfun` accept `"auto"` (the defaults; analysis: PCA at order 1, `WeightedStrength(signed=True)` above; feeding: `Communities()`, which needs >= 4 features and T >= 40 (with more features than distinct shift-null lags in `[L, T - L]`, each lag is used for several features, balanced, and features sharing a lag keep their dependence in that surrogate; the null's squared correlations are averaged over every s-th timepoint, s the kernel's standard deviation rounded down for a Gaussian: `summaries._null_row_stride`); when its communities would emit fewer than 4 columns, the step emits `WeightedStrength(split=False, signed=False, center=False)` instead, one column per input feature, warning only when the shift null found no structure) and dicts keyed by order. `return_info=True` returns `(result, info)` with one dict per step and path (reducers requested and used, widths, community counts, fallback reason). Summary columns constant up to round-off (range <= 1e-9 * max|column|) are set to their means and reported by the constant-output warning. Steps with a summary reducer, `null_combine` and `autofc`/`isfc`/`wisfc` are streamed in blocks of nodes. The original two-path pipeline of the paper is `rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights` (pinned by `tests/test_legacy_pipeline.py`); the decoders always use co-fluctuations on the feeding path.
- **`timecorr/summaries.py`**: squared-correlation summaries `WeightedStrength` (weighted sums of squared correlations per node; `signed`, `split`; `center`: pair weights centered on the mean of their lower cluster (default), `"median"`, or `False`) and `Communities` (mean squared correlation within communities found by spectral k-means; `n=None` in `features` uses a shift null), with `fit`/`transform`/`fits_` (`transform(..., fit_index=...)` names the fitted unit of each unit), and the streaming engine used by `features`.
- **`timecorr/decoding.py`**: timepoint decoders (`timepoint_decoder`, `weighted_timepoint_decoder`, `weighted_timepoint_decoder_ec`, `pca_decoder`) and their helpers. All four accept `events` / `events_params`, resolved once on the full subject list before the folds (per-subject entries follow their subject through the folds; segmentation does not advance NumPy's global random state): the correlation decoders restrict the analysis kernel at levels >= 1 and leave level 0 unchanged; `pca_decoder` restricts its level-0 smoothing kernel and takes one specification, not a dict by order.
- **`timecorr/events.py`**: event-restricted kernels: parsing of `events` specifications into boundary sets (`resolve_events`, `resolve_events_by_order`, `check_boundaries`, `block_labels`), `mask_kernel` (zero weight between timepoints in different blocks), `distinct_event_sets`, and `segment` (hidden Markov model segmentation with hmmlearn: `kind="ordered"`, with `n` given or chosen automatically, or `kind="recurring"`).
- **`timecorr/_lazy.py`**: `lazy_import`, which installs an optional dependency (hmmlearn) with pip on first use; `TIMECORR_AUTO_INSTALL=0` turns installation off.
- **`timecorr/simulate.py`**: simulated data with known dynamic correlations (`simulate_data`, `simulate_multi_order`) and helpers for structure embedded at a chosen order (`min_features`, `carrier_reducer`, `carrier_columns`, `nested_template`), and `simulate_nested` (nodes in modules in supergroups; returns `(data, truth)` with the order-2 signal `truth["h"]`), and `simulate_hierarchy` (nested groups of any depth `len(taus)`, one signal per group per level; returns `(data, truth)` with `truth["groups"]`, `truth["signals"]`, `truth["top"]` and the node signs `truth["signs"]`; `signs="private"` (default) or `"shared"`).

### Key Concepts

The `timecorr` function accepts:
- `data`: numpy array, pandas DataFrame or polars DataFrame (timepoints x features), or a list of them (one per subject)
- `weights_function` / `weights_params`: the kernel; row `t` of the kernel matrix (`weights[t, :]`) is the kernel used at timepoint `t`
- `include_timepoints` (`'all'`, `'backward'`/`'pre'`, `'forward'`/`'post'`), `include_zero`, `exclude_timepoints`: restrict the kernel
- `events` / `events_params`: restrict the kernel to blocks of consecutive timepoints (boundaries, features segmented with `segment`, or a per-subject list; in `features` and the correlation decoders also a dict keyed by order); also accepted by the decoders
- `cfun`: correlation function: `isfc` (each subject versus the others), `wisfc` (weighted), `autofc` (within subject), or `None` (smoothing)
- `combine`: how per-subject results are combined (`null_combine` keeps them separate)
- `rfun`: dimensionality reduction applied to the vectorized correlations (a model name, a graph measure, a summary such as `tc.WeightedStrength(signed=True)` or `"communities"`, or a callable); in `features` also `"auto"` (the default) and dicts keyed by order, with `delta_rfun` for the feeding path

The estimator (`wcorr`) is a kernel-weighted Pearson correlation; a true delta kernel (`eye_weights`) gives co-fluctuations (products of z-scored series). It reads every kernel (a T x T array or a `LagKernel`) through `helpers._KernelRows`, in chunks of 128 rows (`_kernels.CHUNK`), each a contiguous block over its columns, with one span finder (`_time_slices`, which also reads the prepared rows of `_KernelRows`) and one rescaling of extreme rows, so an array and the equal `LagKernel` give identical results (`tests/test_chunked_kernels.py`; `tests/test_kernels_reference.py` compares with stored outputs of commit 6b3d23a, written by `tests/fixtures/make_chunked_kernels_reference.py`); blocks are kept up to `_KERNEL_CACHE_ELEMENTS` (32 MB) and computed again beyond it, and the kernel memory of a `LagKernel` is at most that plus about 4 blocks of 128 x T (`_kernel_work_elements`, used by `estimate_memory_gb`, whose keyword `weights_function` counts a user's kernel function as T x T arrays). Its kernel products are time sliced: each chunk is multiplied only with the span of columns where it has non-zero weight, unless the spans cover more than half the kernel or the samples, their squares or cross products are not all finite; then each chunk spans every column (`helpers._TIME_SLICING = False` forces that; `tests/test_time_sliced_estimation.py` compares both). The two differ only in the order of floating-point summation, but entries whose kernel-weighted variance nearly cancels are sensitive to that order in either path, so the tests compare those entries by a conditioning measure computed from the data and kernel. A kernel-weighted variance counts as zero (correlations 0, counted in the zero-variance warning) when it is at most 1e-10 of the mean square of the globally centered column or at the round-off level of the raw column (`helpers._zero_variance` and `_round_off_variance`, also used by the delta kernel's and `segment`'s z-scores). Columns are centered on their overall means, so with a one-sided kernel only well-conditioned estimates are independent of the excluded timepoints; estimates near the 1e-10 threshold can differ (`tests/test_estimator_causality.py`). Fisher z averaging and `tstat_combine` treat values within 1e-12 of +-1 as +-1 (`helpers._correlation_z`).

### Data Flow

1. Inputs are formatted as float timepoints x features matrices (`format_data`)
2. The weights function builds the T x T kernel; the include/exclude options restrict it
3. The correlation function computes moment-by-moment correlations, vectorized with `mat2vec`
4. The combine function merges subjects; the reduction function reduces the number of features
5. `features` repeats this per order: the analysis path gives `X_k` from the delta (feeding) features `D_{k-1}`, and the feeding path (`autofc` with a Gaussian kernel whose width is set from the order's input by default, or the kernel in `delta_weights_function`; `eye_weights` gives the original pipeline's co-fluctuations) gives `D_k`, reduced by `delta_rfun` (default: `Communities()`)

## Testing Framework

- Uses `pytest`; tests are in `tests/` (one file per area: estimator, kernels, formatting, reduce, features, decoders, simulate, higher-order recovery, events (`test_events.py`), segmentation and on-demand installation (`test_segment.py`), recovery with event-restricted kernels (`test_events_recovery.py`), summaries and their streaming (`test_summaries.py`, `test_summaries_streaming.py`), the feeding path and reducer grammar in `features` (`test_features_summaries.py`), the defaults of `features` and `two_path_step` (`test_features_defaults.py`), their automatic kernels and the lower rate between orders (`test_features_auto_kernels.py`), the timepoints of the shift null (`test_shift_null_rows.py`), outputs and warnings of the original pipeline frozen at commit d4bb5ba (`test_legacy_pipeline.py`, `test_legacy_timecorr.py`), time-sliced versus full estimator products (`test_time_sliced_estimation.py`), features constant up to round-off (`test_round_off_variance.py`), what one-sided kernels guarantee (`test_estimator_causality.py`), kernels read in chunks of rows (`LagKernel`) versus T x T arrays, allocation and memory estimates at T up to 50,000 (`test_chunked_kernels.py`), outputs compared with those of commit 6b3d23a (`test_kernels_reference.py`; `tests/fixtures/chunked_kernels_reference.npz`), order-2 recovery on `simulate_nested` data (`test_nested_recovery.py`; its T 3000, isfc and K 702 cases are slow), kernels by order in `features` (`test_features_kernels_by_order.py`), `simulate_hierarchy` (`test_simulate_hierarchy.py`; reference values of the pre-`signs` output in `tests/fixtures/private_signs_reference.npz`, written by `tests/fixtures/make_private_signs_reference.py` at commit 012f5e0^), plus coverage files)
- Tests run real computations on simulated data; no mocks
- `pytest.ini` registers the `slow` marker for long-running integration tests (continuous integration runs them in a separate job on Python 3.12)
- Docstring examples are tested with `pytest --doctest-modules timecorr/`
- CI: GitHub Actions (`.github/workflows/tests.yml`): the tests without the slow ones (`-m "not slow"`) and the doctests on Python 3.10-3.13, and a separate `slow` job (`-m slow`) on Python 3.12

## Documentation

- Sphinx with numpydoc, autosummary and nbsphinx; source in `docs/`; theme: Furo with the Contextual Dynamics Lab look shared with the htfa and hypertools documentation (Nunito Sans, lowercase light headings, green brand colors set in `html_theme_options` in `docs/conf.py`, rules in `docs/_static/custom.css`)
- API pages: `docs/api.rst` (autosummary stubs in `docs/generated/`)
- Gallery: the tutorials are the gallery. `docs/tutorials.rst` holds an nbsphinx `nbgallery` of the notebooks in `docs/tutorial/` (quickstart first); each card shows the image output of the notebook's one code cell tagged `nbsphinx-thumbnail` and opens the notebook (card styles in `docs/_static/custom.css`, light and dark). `tests/test_docs_tutorials.py` checks that the gallery and the notebooks match, the thumbnail cell, the Colab badge, no error outputs, and that every public name is used (as `tc.<name>`, imported or called; comments and strings do not count) in a tutorial's code cell
- Tutorial notebooks: `docs/tutorial/` (`quickstart`, `kernels`, `subjects`, `reduce`, `simulate`, `higher_order`, `events`, `decoding`, `htfa_network_animations`, `stocks`, `trajectories`; rendered without execution; execute them with nbconvert (needs jupyter) before committing, except the three real-data notebooks, which take minutes to tens of minutes and download data). Each starts with a Colab badge and a title cell; its first code cell installs, on Google Colab only, what the notebook imports: `timecorr` (`timecorr[segment]` for `events`, plus named packages such as polars where imported; the real-data notebooks install `timecorr[tutorials]`), from PyPI, falling back to git master, and every notebook must run top to bottom in a fresh Colab runtime from that cell alone (the Colab badge is the way to run a tutorial; the docs give no other run instructions); text before a code cell is at most 3 short sentences, and numbers computed by the code are printed, not typed in the text. `higher_order.ipynb` holds the PCA event-contrast measurement that `simulate_data`'s order >= 3 warning cites. `htfa_network_animations.ipynb` downloads the HTFA fit of the story-listening data of Owen, Chang & Manning (2021) (460 MB archive, cached under `$TIMECORR_DATA`, default `~/timecorr_data`) and draws the features of orders 1 to 4 on the brain with the htfa toolbox; it writes its four 1920 x 1080, 30 fps H.264 movies (frames rendered in parallel with joblib, encoded with imageio-ffmpeg's ffmpeg at CRF 23 with `+faststart`; one frame per timepoint, 300 frames, 10 s) to `docs/_static/tutorial_media/htfa_order{1..4}.mp4` (committed; at most about 8 MB each) and links them with `<video>` tags (relative path `../_static/tutorial_media/...`) instead of embedding them; outside `docs/tutorial` it writes them to `./htfa_videos` and embeds them. `stocks.ipynb` downloads the daily returns of the 49 Industry Portfolios of the Kenneth R. French Data Library (cached under `$TIMECORR_DATA/french`; never committed) and writes `docs/_static/tutorial_media/stocks_network.mp4` (the market as a network, 42 s). `trajectories.ipynb` uses the HTFA archive of the HTFA tutorial, the French library returns and trial 86_02 of the CMU Motion Capture Database (fetched over http, checked by SHA-256, cached under `$TIMECORR_DATA/cmu_mocap`), and writes `docs/_static/tutorial_media/trajectories_{story_listening,stock_market,human_movement}.mp4` (hypertools animations). Both link their videos in the docs tree and embed them elsewhere (written to `./tutorial_media`)

## Dependencies

Runtime (`setup.py` `install_requires`, Python >= 3.10):
- pydata-wrangler
- hypertools>=0.8
- numpy, scipy, scikit-learn, pandas, matplotlib

Extras:
- `segment`: hmmlearn (used only by `segment`, and installed on demand if missing)
- `test`: pytest, polars, hmmlearn
- `docs`: what the Sphinx build needs: sphinx, furo, numpydoc, nbsphinx, ipython (`.readthedocs.yaml` installs the package with this extra, plus pandoc, and fails on warnings)
- `tutorials`: what the real-data notebooks (htfa_network_animations, stocks, trajectories) import beyond the runtime dependencies; their Colab install cell installs `timecorr[tutorials]`: htfa[plot,brain3d]>=0.1.0, hypertools>=1.1, pyvista>=0.43, imageio-ffmpeg, pillow, joblib, ipython
