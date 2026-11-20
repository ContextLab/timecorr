# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

timecorr is a Python toolbox for computing and exploring correlational structure in timeseries data. The `timecorr` function computes kernel-weighted dynamic correlations from timeseries observations, and `features` computes higher-order dynamic correlations through the two-path pipeline of Owen, Chang & Manning (2021), Fig. 10 (by default with a Gaussian feeding kernel and squared-correlation summaries at orders >= 2; the original pipeline is selected with `rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights`).

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
# Build the HTML documentation (runs the plot_* gallery examples)
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
- **`timecorr/helpers.py`**:
  - Kernel functions (`gaussian_weights`, `laplace_weights`, `eye_weights`, `mexican_hat_weights`, `t_weights`, `uniform_weights`, `boxcar_weights`) and `one_sided`
  - The estimator `wcorr` and correlation functions (`isfc`, `wisfc`, `autofc`)
  - Combine functions (`null_combine`, `corrmean_combine`, `mean_combine`, `tstat_combine`)
  - `reduce` (dimensionality reduction), `format_data` (input formatting via pydata-wrangler), `mat2vec`/`vec2mat`, `smooth`, `plot_weights`
  - Re-exports of the decoders from `decoding.py`
- **`timecorr/higher_order.py`**: the two-path pipeline: `features`, `two_path_step`, `estimate_memory_gb` (`streamed=True` for the peak of a streamed step). Keyword-only `delta_weights_function` / `delta_weights_params` set the kernel of the delta (feeding) path (default None: `gaussian_weights` with `{"var": 25}`, giving `autofc` correlations; `eye_weights` gives co-fluctuations). `weights_params` / `delta_weights_params` also accept order maps (dicts whose keys are all integers in 1..order; missing orders use the default parameters; `two_path_step`, exported as `tc.two_path_step`, raises for one). `rfun` / `delta_rfun` accept `"auto"` (the defaults; analysis: PCA at order 1, `WeightedStrength(signed=True)` above; feeding: `Communities()`, which needs >= 4 features and T >= 40 (with more features than distinct shift-null lags in `[L, T - L]`, each lag is used for several features, balanced, and features sharing a lag keep their dependence in that surrogate); when its communities would emit fewer than 4 columns, the step emits `WeightedStrength(split=False, signed=False, center=False)` instead, one column per input feature, warning only when the shift null found no structure) and dicts keyed by order. `return_info=True` returns `(result, info)` with one dict per step and path (reducers requested and used, widths, community counts, fallback reason). Summary columns constant up to round-off (range <= 1e-9 * max(1, max|column|)) are set to their means and reported by the constant-output warning. Steps with a summary reducer, `null_combine` and `autofc`/`isfc`/`wisfc` are streamed in blocks of nodes. The original two-path pipeline of the paper is `rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights` (pinned by `tests/test_legacy_pipeline.py`); the decoders always use co-fluctuations on the feeding path.
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

The estimator (`wcorr`) is a kernel-weighted Pearson correlation; a true delta kernel (`eye_weights`) gives co-fluctuations (products of z-scored series). Its kernel products are time sliced: chunks of 128 kernel rows are multiplied only with the span of columns where they have non-zero weight, unless the spans cover more than half the kernel or the samples, their squares or cross products are not all finite (`helpers._TIME_SLICING = False` forces the full products; `tests/test_time_sliced_estimation.py` compares both). The two differ only in the order of floating-point summation, but entries whose kernel-weighted variance nearly cancels are sensitive to that order in either path, so the tests compare those entries by a conditioning measure computed from the data and kernel.

### Data Flow

1. Inputs are formatted as float timepoints x features matrices (`format_data`)
2. The weights function builds the T x T kernel; the include/exclude options restrict it
3. The correlation function computes moment-by-moment correlations, vectorized with `mat2vec`
4. The combine function merges subjects; the reduction function reduces the number of features
5. `features` repeats this per order: the analysis path gives `X_k` from the delta (feeding) features `D_{k-1}`, and the feeding path (`autofc` with the Gaussian kernel of variance 25 by default, or the kernel in `delta_weights_function`; `eye_weights` gives the original pipeline's co-fluctuations) gives `D_k`, reduced by `delta_rfun` (default: `Communities()`)

## Testing Framework

- Uses `pytest`; tests are in `tests/` (one file per area: estimator, kernels, formatting, reduce, features, decoders, simulate, higher-order recovery, events (`test_events.py`), segmentation and on-demand installation (`test_segment.py`), recovery with event-restricted kernels (`test_events_recovery.py`), summaries and their streaming (`test_summaries.py`, `test_summaries_streaming.py`), the feeding path and reducer grammar in `features` (`test_features_summaries.py`), the defaults of `features` and `two_path_step` (`test_features_defaults.py`), outputs and warnings of the original pipeline frozen at commit d4bb5ba (`test_legacy_pipeline.py`, `test_legacy_timecorr.py`), time-sliced versus full estimator products (`test_time_sliced_estimation.py`), order-2 recovery on `simulate_nested` data (`test_nested_recovery.py`; its T 3000, isfc and K 702 cases are slow), kernels by order in `features` (`test_features_kernels_by_order.py`), `simulate_hierarchy` (`test_simulate_hierarchy.py`; reference values of the pre-`signs` output in `tests/fixtures/private_signs_reference.npz`, written by `tests/fixtures/make_private_signs_reference.py` at commit 012f5e0^), plus coverage files)
- Tests run real computations on simulated data; no mocks
- `pytest.ini` registers the `slow` marker for long-running integration tests (continuous integration runs them in a separate job on Python 3.12)
- Docstring examples are tested with `pytest --doctest-modules timecorr/`
- CI: GitHub Actions (`.github/workflows/tests.yml`): the tests without the slow ones (`-m "not slow"`) and the doctests on Python 3.10-3.13, and a separate `slow` job (`-m slow`) on Python 3.12

## Documentation

- Sphinx with numpydoc, autosummary, sphinx-gallery and nbsphinx; source in `docs/`; theme: Furo with the Contextual Dynamics Lab look shared with the htfa and hypertools documentation (Nunito Sans, lowercase light headings, green brand colors set in `html_theme_options` in `docs/conf.py`, rules in `docs/_static/custom.css`)
- API pages: `docs/api.rst` (autosummary stubs in `docs/generated/`)
- Gallery examples: `examples/` (rendered into `docs/auto_examples/`; only `plot_*` scripts are executed)
- Tutorial notebooks: `docs/tutorial/` (rendered without execution; execute them with nbconvert before committing). `htfa_network_animations.ipynb` downloads the HTFA fit of the story-listening data of Owen, Chang & Manning (2021) (460 MB archive, cached under `$TIMECORR_DATA`, default `~/timecorr_data`) and draws the features of orders 1 to 4 on the brain with the htfa toolbox; it writes its four 1920 x 1080, 30 fps H.264 movies (frames rendered in parallel with joblib, encoded with imageio-ffmpeg's ffmpeg at CRF 23 with `+faststart`) to `docs/_static/tutorial_media/htfa_order{1..4}.mp4` (committed; at most about 8 MB each) and links them with `<video>` tags (relative path `../_static/tutorial_media/...`) instead of embedding them

## Dependencies

Runtime (`setup.py` `install_requires`, Python >= 3.10):
- pydata-wrangler
- hypertools>=0.8
- numpy, scipy, scikit-learn, pandas, matplotlib

Extras:
- `segment`: hmmlearn (used only by `segment`, and installed on demand if missing)
- `test`: pytest, polars, hmmlearn
- `docs`: sphinx, furo, sphinx-gallery, numpydoc, nbsphinx, seaborn, ipykernel, nbconvert, hmmlearn, htfa[plot,brain3d] (for running `htfa_network_animations.ipynb`)
