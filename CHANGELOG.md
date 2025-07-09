# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-09

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