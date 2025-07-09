# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

timecorr is a Python toolbox for computing and exploring correlational structure in timeseries data. The main functionality is built around the `timecorr` function which computes dynamic correlations from timeseries observations and finds higher-order structure in the data.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
# or
make init
```

### Testing
```bash
# Run tests using nosetests
nosetests tests
# or
make test

# Run tests using pytest (alternative)
pytest
```

### Documentation
```bash
# Build documentation
cd docs
make html

# Build notebooks (for tutorials)
make notebooks
```

### Building/Installation
```bash
# Install package in development mode
pip install -e .

# Build distribution packages
python setup.py sdist bdist_wheel
```

## Code Architecture

### Core Components

- **`timecorr/timecorr.py`**: Main module containing the `timecorr()` function - the primary entry point for computing dynamic correlations
- **`timecorr/helpers.py`**: Utility functions including:
  - Weight functions (gaussian_weights, laplace_weights, etc.)
  - Correlation functions (isfc, wisfc, autofc, wcorr)
  - Data formatting and manipulation utilities
  - Dimensionality reduction functions
- **`timecorr/simulate.py`**: Data simulation utilities for generating test datasets

### Key Concepts

The `timecorr` function accepts:
- `data`: Single numpy array/dataframe or list of arrays (one per subject)
- `weights_function`: How timepoints contribute to correlations (default: gaussian_weights)
- `cfun`: Correlation function - `isfc` (within/across subjects), `wisfc` (weighted), `autofc` (within subject only)
- `rfun`: Dimensionality reduction function to prevent feature explosion when computing higher-order correlations

### Data Flow

1. Input data formatted as timepoints × features matrices
2. Weights function determines temporal smoothing
3. Correlation function computes moment-by-moment correlations
4. Optional reduction function prevents dimensionality explosion
5. Output maintains temporal structure with correlational features

## Testing Framework

- Uses `nosetests` (legacy) and `pytest` (modern)
- Tests in `tests/` directory
- Test files: `test_timecorr.py`, `test_helpers.py`
- Uses simulated data from `simulate_data()` function
- Current tests are basic - more comprehensive tests needed per TODO comments

## Documentation

- Uses Sphinx for documentation generation
- Documentation source in `docs/` directory
- Examples in `examples/` directory
- Tutorial notebooks in `docs/tutorial/`
- API documentation auto-generated from docstrings

## Dependencies

Core dependencies:
- hypertools>=0.7.0
- scipy>=1.2.1
- scikit-learn>=0.19.2
- numpy, pandas (via hypertools)

Development dependencies:
- nose (testing)
- sphinx (documentation)
- duecredit (citations)