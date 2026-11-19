"""
Tests to cover the remaining missing lines for 100% coverage.

This module targets the specific missing lines identified in the coverage report.
"""

import warnings

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend

from timecorr.helpers import (
    _is_empty,
    calculate_error,
    decoder,
    folding_levels_ec,
    gaussian_weights,
    get_xval_assignments,
    isfc,
    mean_combine,
    pca_decoder,
    plot_weights,
    reduce,
    smooth,
    sum_to_x,
    timepoint_decoder,
    wcorr,
    weight_corrs,
    weighted_timepoint_decoder,
    weighted_timepoint_decoder_ec,
)
from timecorr.simulate import *
from timecorr.timecorr import timecorr


class TestRemainingCoverage:
    """Test remaining uncovered lines for 100% coverage."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.T = 10
        self.K = 6
        self.data = np.random.randn(self.T, self.K)

    def test_wcorr_exception_branch(self):
        """Constant inputs and tiny weights give zero correlations, not an exception."""
        a = np.ones((5, 3)) * 1e-16
        b = np.ones((5, 3)) * 1e-16
        weights = np.ones((5, 5)) * 1e-16
        with pytest.warns(
            UserWarning, match="5 of 5 timepoints have zero kernel-weighted variance"
        ):
            result = wcorr(a, b, weights)
        assert result.shape == (3, 3, 5)
        assert np.all(np.isfinite(result))
        assert np.allclose(result, 0)

    def test_timepoint_decoder_nfolds_warning(self):
        """Test nfolds=1 warning at lines 592-594."""
        # Create simple test data
        data = np.array([np.random.randn(8, 4) for _ in range(2)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = timepoint_decoder(data, nfolds=1, level=0)
            # Should trigger warning about circular analysis
            assert len(w) > 0
            assert "circular" in str(w[0].message)
        assert result["folds"].tolist() == [0, 1]

    def test_timepoint_decoder_level_types(self):
        """An integer level computes levels 0..level and reports that level."""
        data = np.array([np.random.randn(8, 4) for _ in range(2)])
        np.random.seed(0)
        result = timepoint_decoder(data, level=1, nfolds=2)
        assert result["level"].tolist() == [1, 1]
        assert result["folds"].tolist() == [0, 1]
        with pytest.raises(ValueError, match="level"):
            timepoint_decoder(data, level=-1, nfolds=2)

    def test_weighted_timepoint_decoder_branches(self):
        """optimize_levels adds levels one at a time; each step reports its weights."""
        data = np.array([np.random.randn(8, 4) for _ in range(3)])
        with pytest.raises(ValueError, match="at least 2"):
            weighted_timepoint_decoder(
                data, nfolds=2, level=[0, 1], optimize_levels=[0, 1]
            )

        data = np.array([np.random.randn(8, 4) for _ in range(4)])
        np.random.seed(0)
        result = weighted_timepoint_decoder(
            data, nfolds=2, level=[0, 1], optimize_levels=[0, 1]
        )
        assert result["level"].tolist() == [0, 1, 0, 1]
        assert np.allclose(result.loc[result["level"] == 0, "level_0"], 1)
        with pytest.raises(ValueError, match="optimize_levels"):
            weighted_timepoint_decoder(data, nfolds=2, level=0, optimize_levels=[0, 1])

    def test_weighted_timepoint_decoder_ec_branches(self):
        """nfolds=1 warns that the analysis is circular."""
        data = np.array([np.random.randn(8, 4) for _ in range(4)])
        np.random.seed(0)
        with pytest.warns(UserWarning, match="circular"):
            result = weighted_timepoint_decoder_ec([data], nfolds=1, level=0)
        assert result["folds"].tolist() == [0, 1]

    def test_folding_levels_ec_branches(self):
        """Level 1: each group's ISFC, averaged; raw inputs passed through."""
        in_data = [np.random.randn(8, 4) for _ in range(2)]
        out_data = [np.random.randn(8, 4) for _ in range(2)]

        in_s, out_s, in_r, out_r = folding_levels_ec(
            in_data,
            out_data,
            level=1,
            cfun=isfc,
            weights_fun=gaussian_weights,
            combine=mean_combine,
        )
        expected = timecorr(in_data, cfun=isfc, weights_function=gaussian_weights, combine=mean_combine)
        assert in_s.shape == (8, 10)
        assert np.allclose(in_s, expected)
        assert in_r is in_data and out_r is out_data

    def test_smooth_function_parameters(self):
        """Custom kernel parameters are passed to the kernel function."""
        w = np.random.randn(10, 5)
        result = smooth(w, kernel_fun=gaussian_weights, kernel_params={"var": 50})
        assert result.shape == w.shape
        narrow = smooth(w, kernel_fun=gaussian_weights, kernel_params={"var": 0.01})
        # a very narrow Gaussian barely smooths
        assert np.allclose(narrow, w, atol=1e-6)
        assert not np.allclose(result, w)

    def test_reduce_function_edge_cases(self):
        """Graph measures give one value per node: 15 = 5 * 6 / 2 -> 5 nodes."""
        corr_data = np.random.randn(10, 15)
        result = reduce(corr_data, rfun="pagerank_centrality")
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)
        assert np.all(np.isfinite(result))

    def test_calculate_error_metrics(self):
        """Metrics are decoder columns; an unknown metric raises KeyError."""
        mu = np.array([0.3, 0.4, 0.3])
        corrs = [np.random.randn(10, 15) for _ in range(3)]
        expected = decoder(weight_corrs(corrs, mu))
        assert np.allclose(calculate_error(mu, corrs, metric="rank"), expected["rank"].values)
        with pytest.raises(KeyError):
            calculate_error(mu, corrs, metric="correlation")
        with pytest.raises(KeyError):
            calculate_error(mu, corrs, metric="other")

    def test_timecorr_return_list_edge_case(self):
        """cfun=None on one array returns the smoothed (T, K) array."""
        data = np.random.randn(10, 4)
        result = timecorr(data, cfun=None)  # No correlation function
        # Task 6: an array, not the nested Python list the old .tolist() made
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape

    def test_simulate_data_single_subject_return_corrs(self):
        """Test simulate_data single subject with return_corrs."""
        data, corrs = simulate_data(S=1, T=10, K=4, return_corrs=True)
        assert isinstance(data, np.ndarray) and data.shape == (10, 4)
        assert isinstance(corrs, np.ndarray) and corrs.shape == (10, 10)

    def test_pca_decoder_edge_cases(self):
        """dims above the number of features still gives one row per dims value."""
        data = np.array([np.random.randn(10, 6) for _ in range(4)])
        np.random.seed(0)
        result = pca_decoder(data, nfolds=2, dims=8)  # High dimensions
        assert isinstance(result, pd.DataFrame)
        assert result["dims"].tolist() == list(range(1, 9)) * 2

    def test_plot_weights_outfile_branch(self, tmp_path):
        """outfile saves a PNG of the plotted kernel row."""
        import matplotlib.pyplot as plt

        weights = gaussian_weights(10)
        outfile = tmp_path / "weights.png"
        fig, ax = plt.subplots()
        plot_weights(weights, ax=ax, outfile=str(outfile))
        assert outfile.exists()
        assert outfile.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert np.allclose(ax.lines[0].get_ydata(), weights[5, :])
        plt.close(fig)

    def test_edge_case_functions(self):
        """Test various edge case functions for remaining coverage."""
        assert _is_empty(None) == True
        assert _is_empty(False) == True
        assert _is_empty(0) == True
        assert _is_empty("") == True
        assert _is_empty([]) == True
        assert _is_empty({}) == True

        assignments = get_xval_assignments(8, 4)
        assert len(assignments) == 8
        assert np.bincount(assignments.astype(int)).tolist() == [2, 2, 2, 2]

        result = sum_to_x(4, 2.0)
        assert len(result) == 4
        assert np.isclose(np.sum(result), 2.0)

if __name__ == "__main__":
    pytest.main([__file__])
