"""
Full coverage tests for timecorr package to achieve 100% coverage.

This module tests all remaining uncovered functions and edge cases with
actual data to ensure numerical correctness and robustness.
"""

import warnings

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import toeplitz

matplotlib.use("Agg")  # Use non-interactive backend

import timecorr as tc
from timecorr.helpers import (
    _is_empty,
    apply_by_row,
    autofc,
    calculate_error,
    corrmean_combine,
    decoder,
    eye_weights,
    folding_levels,
    folding_levels_ec,
    gaussian_params,
    gaussian_weights,
    get_xval_assignments,
    iseven,
    isfc,
    isodd,
    laplace_params,
    laplace_weights,
    mat2vec,
    mean_combine,
    null_combine,
    optimize_weights,
    pca_decoder,
    r2z,
    reduce,
    reduce_wrapper,
    rmdiag,
    smooth,
    sum_to_x,
    symmetric,
    timepoint_decoder,
    tstat_combine,
    vec2mat,
    wcorr,
    weight_corrs,
    weighted_mean,
    weighted_timepoint_decoder,
    weighted_timepoint_decoder_ec,
    wisfc,
    z2r,
)
from timecorr.simulate import *
from timecorr.timecorr import timecorr


class TestFullCoverageHelpers:
    """Test all remaining uncovered functions in helpers.py."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.T = 20
        self.K = 8
        self.S = 4

        # Create realistic test data
        self.single_data = np.random.randn(self.T, self.K)
        self.multi_data = [np.random.randn(self.T, self.K) for _ in range(self.S)]

        # Create time series with some structure
        t = np.linspace(0, 4 * np.pi, self.T)
        structured_data = np.column_stack(
            [
                np.sin(t) + 0.1 * np.random.randn(self.T),
                np.cos(t) + 0.1 * np.random.randn(self.T),
                np.sin(2 * t) + 0.1 * np.random.randn(self.T),
                np.cos(2 * t) + 0.1 * np.random.randn(self.T),
                np.random.randn(self.T),
                np.random.randn(self.T),
                np.random.randn(self.T),
                np.random.randn(self.T),
            ]
        )

        # For timepoint_decoder, data needs to be numpy array with shape (n_subjects, T, K)
        structured_list = [
            structured_data + 0.05 * np.random.randn(self.T, self.K)
            for _ in range(self.S)
        ]
        self.structured_data = np.array(structured_list)

        self.weights = gaussian_weights(self.T)
        self.laplace_params = {"scale": 10}

    def test_is_empty_function(self):
        """Test _is_empty helper function."""
        # Test empty dict
        assert _is_empty({}) == True

        # Test non-empty dict
        assert _is_empty({"a": 1}) == False

        # Test None
        assert _is_empty(None) == True

    def test_wcorr_exception_handling(self):
        """Test wcorr exception handling branch."""
        # Create data that might cause numerical issues
        data_a = np.ones((10, 3))  # Constant data
        data_b = np.ones((10, 3))  # Constant data
        weights = gaussian_weights(10)

        # This should trigger the exception handling branch
        result = wcorr(data_a, data_b, weights)
        assert result.shape == (3, 3, 10)

    def test_simulate_data_edge_cases(self):
        """Test simulate_data edge cases for full coverage."""
        # Test with set_random_seed=True (boolean)
        result1 = simulate_data(S=1, T=10, K=3, set_random_seed=True)
        result2 = simulate_data(S=1, T=10, K=3, set_random_seed=True)
        assert np.allclose(result1, result2)  # Should be identical

        # Test with return_corrs=True for single subject
        data, corrs = simulate_data(
            S=1, T=8, K=4, return_corrs=True, set_random_seed=456
        )
        assert isinstance(data, np.ndarray)
        assert isinstance(corrs, np.ndarray)
        assert data.shape == (8, 4)

    def test_timecorr_edge_cases(self):
        """Test timecorr edge cases for full coverage."""
        # Test invalid include_timepoints
        data = np.random.randn(10, 4)
        with pytest.raises(Exception) as excinfo:
            timecorr(data, include_timepoints="invalid")
        assert "Invalid option for include_timepoints" in str(excinfo.value)

        # Test that function runs without error
        result = timecorr(data)
        assert isinstance(result, np.ndarray)

    def test_timepoint_decoder(self):
        """Test timepoint_decoder function with actual data."""
        np.random.seed(0)
        results = timepoint_decoder(self.structured_data, nfolds=2, level=0)
        assert isinstance(results, pd.DataFrame)
        assert results["folds"].tolist() == [0, 1]
        accuracy = results["accuracy"].mean()
        assert 0 <= accuracy <= 1
        # the subjects share a strong signal: decoding is far above chance (1/T)
        assert accuracy > 3 / self.T

        # Test with nfolds=1 (should trigger warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            timepoint_decoder(self.structured_data, nfolds=1, level=0)
            assert len(w) > 0
            assert "circular" in str(w[0].message)

    def test_weighted_timepoint_decoder(self):
        """Test weighted_timepoint_decoder function."""
        np.random.seed(0)
        result = weighted_timepoint_decoder(self.structured_data, nfolds=2, level=0)
        assert isinstance(result, pd.DataFrame)
        assert result["level"].tolist() == [0, 0]
        # a single level gets all the weight
        assert np.allclose(result["level_0"], 1)

    def test_weighted_timepoint_decoder_ec(self):
        """Test weighted_timepoint_decoder_ec function."""
        # data are indexed by level: [subjects] for raw data only
        np.random.seed(0)
        result = weighted_timepoint_decoder_ec(
            [self.structured_data], nfolds=2, level=0
        )
        assert isinstance(result, pd.DataFrame)
        assert result["level"].tolist() == [0, 0]
        assert np.allclose(result["level_0"], 1)
        with pytest.raises(ValueError, match="indexed by level"):
            weighted_timepoint_decoder_ec(self.structured_data, nfolds=2, level=0)

    def test_folding_levels(self):
        """Test folding_levels function."""
        in_data = self.multi_data[:2]
        out_data = self.multi_data[2:4]

        in_smooth, out_smooth, in_raw, out_raw = folding_levels(
            in_data,
            out_data,
            level=0,
            cfun=None,
            weights_fun=gaussian_weights,
            combine=mean_combine,
        )
        # level 0: each group's kernel-smoothed data, averaged; raw is the input
        assert isinstance(in_smooth, np.ndarray) and isinstance(out_smooth, np.ndarray)
        expected = timecorr(in_data, cfun=None, weights_function=gaussian_weights, combine=mean_combine)
        assert np.allclose(in_smooth, expected)
        assert in_raw is in_data and out_raw is out_data

    def test_folding_levels_ec(self):
        """Test folding_levels_ec function."""
        in_data = self.multi_data[:2]
        out_data = self.multi_data[2:4]

        in_smooth, out_smooth, in_raw, out_raw = folding_levels_ec(
            in_data,
            out_data,
            level=0,
            cfun=None,
            weights_fun=gaussian_weights,
            combine=mean_combine,
        )
        assert isinstance(in_smooth, np.ndarray) and isinstance(out_smooth, np.ndarray)
        assert in_smooth.shape == (self.T, self.K)
        expected = timecorr(out_data, cfun=None, weights_function=gaussian_weights, combine=mean_combine)
        assert np.allclose(out_smooth, expected)
        assert in_raw is in_data and out_raw is out_data

    def test_pca_decoder(self):
        """Test pca_decoder function."""
        np.random.seed(0)
        results = pca_decoder(self.structured_data, nfolds=2, dims=3)
        assert isinstance(results, pd.DataFrame)
        assert results["dims"].tolist() == [1, 2, 3, 1, 2, 3]
        accuracy = results["accuracy"].to_numpy()
        assert np.all((accuracy >= 0) & (accuracy <= 1))

    def test_reduce_wrapper(self):
        """reduce_wrapper passes level 0 through and reduces both paths jointly above it."""
        parts = (self.multi_data[0], self.multi_data[1], self.multi_data[:2], self.multi_data[2:4])
        assert reduce_wrapper(parts, level=0) == parts

        V = [mat2vec(np.corrcoef(np.random.randn(self.K, 30))) for _ in range(self.T)]
        smooth_in = np.array(V)
        smooth_out = np.array(V[::-1])
        raw = [np.array(V)] * 2
        result = reduce_wrapper(
            (smooth_in, smooth_out, raw, raw), level=1, rfun=[None, "PCA"]
        )
        joint = reduce([smooth_in, smooth_out], rfun="PCA")
        assert len(result) == 4
        assert np.allclose(result[0], joint[0]) and np.allclose(result[1], joint[1])
        assert result[2].shape == (self.T, self.K)

    def test_optimization_functions(self):
        """Test optimization-related functions."""
        corrs = [np.random.randn(10, 15) for _ in range(3)]
        result = optimize_weights(corrs)
        assert isinstance(result, np.ndarray) and result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1e-12) and np.isclose(result.sum(), 1)

    def test_utility_functions(self):
        """Test utility functions for edge cases."""
        result = sum_to_x(3, 1.0)
        assert len(result) == 3
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result >= 0)

        mu = np.array([0.3, 0.4, 0.3])
        corrs = [np.random.randn(10, 15) for _ in range(3)]
        error = calculate_error(mu, corrs)
        assert isinstance(error, np.ndarray) and error.shape == (1,)
        assert np.allclose(error, decoder(weight_corrs(corrs, mu))["error"].values)

        weighted = weight_corrs(corrs, mu)
        assert isinstance(weighted, np.ndarray) and weighted.shape == (10, 15)
        # Fisher-z weighted mean of the inputs
        assert np.allclose(weighted, z2r(sum(m * r2z(c) for m, c in zip(mu, corrs))))

        corr_matrix = np.eye(10)
        decoder_result = decoder(corr_matrix)
        assert isinstance(decoder_result, pd.DataFrame)
        # the identity decodes every timepoint exactly
        assert decoder_result["accuracy"].iloc[0] == 1
        assert decoder_result["error"].iloc[0] == 0
        assert decoder_result["rank"].iloc[0] == 1

    def test_weighted_mean_edge_cases(self):
        """Test weighted_mean with various edge cases."""
        x = np.random.randn(5, 4)

        # Test with custom weights
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = weighted_mean(x, axis=0, weights=weights)
        assert result.shape == (4,)

        # Test with NaN weights
        weights_nan = np.array([0.2, np.nan, 0.3, 0.3, 0.2])
        result_nan = weighted_mean(x, axis=0, weights=weights_nan)
        assert result_nan.shape == (4,)
        assert not np.any(np.isnan(result_nan))

        # Test with zero weights
        weights_zero = np.zeros(5)
        result_zero = weighted_mean(x, axis=0, weights=weights_zero)
        assert result_zero.shape == (4,)

        # Test with different axis
        result_axis1 = weighted_mean(x, axis=1, weights=np.ones(4))
        assert result_axis1.shape == (5,)

    def test_smooth_function_edge_cases(self):
        """Test smooth function with edge cases."""
        w = np.random.randn(15, 6)
        # any kernel function works with its own default parameters
        result_gauss = smooth(w, kernel_fun=gaussian_weights)
        assert result_gauss.shape == w.shape
        assert np.allclose(
            result_gauss, smooth(w, kernel_fun=gaussian_weights, kernel_params=gaussian_params)
        )

        w_list = [np.random.randn(12, 4) for _ in range(2)]
        result_list = smooth(w_list, windowsize=8)
        assert isinstance(result_list, list) and len(result_list) == 2
        assert np.allclose(result_list[1], smooth(w_list[1], windowsize=8))

    def test_parameter_validation(self):
        """Test parameter validation and type checking."""
        assignments = get_xval_assignments(10, 3)
        assert len(assignments) == 10
        assert max(assignments) < 3
        # balanced groups (sizes differ by at most one): 4, 3 and 3
        assert sorted(np.bincount(assignments.astype(int)).tolist()) == [3, 3, 4]

class TestFullCoverageNumericalCorrectness:
    """Test numerical correctness of basic utility functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(12345)
        self.test_data = np.random.randn(8, 4)

    def test_basic_matrix_operations(self):
        """Test basic matrix operations for correctness."""
        # Test mat2vec and vec2mat roundtrip
        matrix = np.random.randn(5, 5)
        matrix = (matrix + matrix.T) / 2  # Make symmetric

        vec = mat2vec(matrix)
        matrix_recovered = vec2mat(vec)

        assert np.allclose(matrix, matrix_recovered, atol=1e-10)

        # Test rmdiag
        matrix_no_diag = rmdiag(matrix)
        assert np.allclose(np.diag(matrix_no_diag), 0)

    def test_statistical_transformations(self):
        """Test statistical transformation functions."""
        # Test r2z and z2r roundtrip
        r_values = np.array([0, 0.3, -0.7, 0.9, -0.2])
        z_values = r2z(r_values)
        r_recovered = z2r(z_values)

        assert np.allclose(r_values, r_recovered, atol=1e-10)

        # Test edge cases
        z_inf = np.array([np.inf, -np.inf, np.nan])
        r_inf = z2r(z_inf)
        assert not np.any(np.isnan(r_inf[2:]))  # NaN should be handled

    def test_utility_function_correctness(self):
        """Test utility functions give correct results."""
        # Test isodd and iseven
        numbers = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(isodd(numbers), [True, False, True, False, True])
        assert np.array_equal(iseven(numbers), [False, True, False, True, False])

        # Test symmetric function
        sym_matrix = np.array([[1, 2], [2, 1]])
        asym_matrix = np.array([[1, 2], [3, 1]])
        assert symmetric(sym_matrix) == True
        assert symmetric(asym_matrix) == False


if __name__ == "__main__":
    pytest.main([__file__])
