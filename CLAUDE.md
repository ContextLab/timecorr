# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

timecorr is a Python toolbox for computing and exploring correlational structure in timeseries data. The `timecorr` function computes kernel-weighted dynamic correlations from timeseries observations, and `features` computes higher-order dynamic correlations through the two-path pipeline of Owen, Chang & Manning (2021), Fig. 10.

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
- **`timecorr/higher_order.py`**: the two-path pipeline: `features`, `two_path_step`, `estimate_memory_gb`.
- **`timecorr/decoding.py`**: timepoint decoders (`timepoint_decoder`, `weighted_timepoint_decoder`, `weighted_timepoint_decoder_ec`, `pca_decoder`) and their helpers.
- **`timecorr/events.py`**: event-restricted kernels: parsing of `events` specifications into boundary sets (`resolve_events`, `resolve_events_by_order`, `check_boundaries`, `block_labels`), `mask_kernel` (zero weight between timepoints in different blocks), `distinct_event_sets`, and `segment` (hidden Markov model segmentation with hmmlearn: `kind="ordered"`, with `n` given or chosen automatically, or `kind="recurring"`).
- **`timecorr/_lazy.py`**: `lazy_import`, which installs an optional dependency (hmmlearn) with pip on first use; `TIMECORR_AUTO_INSTALL=0` turns installation off.
- **`timecorr/simulate.py`**: simulated data with known dynamic correlations (`simulate_data`, `simulate_multi_order`) and helpers for structure embedded at a chosen order (`min_features`, `carrier_reducer`, `carrier_columns`, `nested_template`).

### Key Concepts

The `timecorr` function accepts:
- `data`: numpy array, pandas DataFrame or polars DataFrame (timepoints x features), or a list of them (one per subject)
- `weights_function` / `weights_params`: the kernel; row `t` of the kernel matrix (`weights[t, :]`) is the kernel used at timepoint `t`
- `include_timepoints` (`'all'`, `'backward'`/`'pre'`, `'forward'`/`'post'`), `include_zero`, `exclude_timepoints`: restrict the kernel
- `events` / `events_params`: restrict the kernel to blocks of consecutive timepoints (boundaries, features segmented with `segment`, or a per-subject list; in `features` also a dict keyed by order)
- `cfun`: correlation function: `isfc` (each subject versus the others), `wisfc` (weighted), `autofc` (within subject), or `None` (smoothing)
- `combine`: how per-subject results are combined (`null_combine` keeps them separate)
- `rfun`: dimensionality reduction applied to the vectorized correlations

The estimator (`wcorr`) is a kernel-weighted Pearson correlation; a true delta kernel (`eye_weights`) gives co-fluctuations (products of z-scored series).

### Data Flow

1. Inputs are formatted as float timepoints x features matrices (`format_data`)
2. The weights function builds the T x T kernel; the include/exclude options restrict it
3. The correlation function computes moment-by-moment correlations, vectorized with `mat2vec`
4. The combine function merges subjects; the reduction function reduces the number of features
5. `features` repeats this per order: the analysis path gives `X_k` from the delta features `D_{k-1}`, and the delta path (`eye_weights`, `autofc`) gives `D_k`

## Testing Framework

- Uses `pytest`; tests are in `tests/` (one file per area: estimator, kernels, formatting, reduce, features, decoders, simulate, higher-order recovery, events (`test_events.py`), segmentation and on-demand installation (`test_segment.py`), recovery with event-restricted kernels (`test_events_recovery.py`), plus coverage files)
- Tests run real computations on simulated data; no mocks
- `pytest.ini` registers the `slow` marker for long-running integration tests (continuous integration runs them)
- Docstring examples are tested with `pytest --doctest-modules timecorr/`
- CI: GitHub Actions on Python 3.10-3.13 (`.github/workflows/tests.yml`)

## Documentation

- Sphinx with numpydoc, autosummary, sphinx-gallery and nbsphinx; source in `docs/`
- API pages: `docs/api.rst` (autosummary stubs in `docs/generated/`)
- Gallery examples: `examples/` (rendered into `docs/auto_examples/`; only `plot_*` scripts are executed)
- Tutorial notebooks: `docs/tutorial/` (rendered without execution; execute them with nbconvert before committing)

## Dependencies

Runtime (`setup.py` `install_requires`, Python >= 3.10):
- pydata-wrangler
- hypertools>=0.8
- numpy, scipy, scikit-learn, pandas, matplotlib

Extras:
- `segment`: hmmlearn (used only by `segment`, and installed on demand if missing)
- `test`: pytest, polars, hmmlearn
- `docs`: sphinx, sphinx-book-theme, sphinx-gallery, numpydoc, nbsphinx, seaborn, ipykernel, nbconvert, hmmlearn
