"""
Final push to 100% coverage - targeting very specific missing lines.
"""


# Configure matplotlib to use non-interactive backend for testing
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend

from timecorr.helpers import (
    calculate_error,
    decoder,
    isfc,
    weight_corrs,
    folding_levels,
    gaussian_weights,
    laplace_weights,
    pca_decoder,
    reduce,
    smooth,
    timepoint_decoder,
    wcorr,
    weighted_timepoint_decoder,
    weighted_timepoint_decoder_ec,
)
from timecorr.timecorr import timecorr


class TestFinalCoverage:
    """Final tests for 100% coverage."""

    def test_wcorr_mystery_exception(self):
        """Constant (all-zero) inputs give zero correlations, not an exception."""
        a_zero = np.zeros((5, 3))
        b_zero = np.zeros((5, 3))
        result = wcorr(a_zero, b_zero, np.eye(5))
        assert result.shape == (3, 3, 5)
        assert np.all(np.isfinite(result))
        assert np.allclose(result, 0)

    def test_reduce_edge_case(self):
        """An unknown reducer name raises a ValueError naming it."""
        corrs = np.random.randn(10, 15)
        with pytest.raises(ValueError, match="nonexistent_function"):
            reduce(corrs, rfun="nonexistent_function")

    def test_smooth_edge_cases(self):
        """A one-sample window leaves the data unchanged."""
        w = np.random.randn(10, 5)
        result = smooth(w, windowsize=1, kernel_fun=laplace_weights)
        assert result.shape == w.shape
        assert np.allclose(result, w)

    def test_timepoint_decoder_complex_branches(self):
        """Per-level cfun and rfun lists; a cfun with the wrong output shape raises."""
        data = np.array([np.random.randn(6, 3) for _ in range(4)])

        # a well-formed per-level specification runs and reports both levels
        np.random.seed(0)
        result = timepoint_decoder(
            data,
            level=[0, 1],
            nfolds=2,
            cfun=[None, isfc],
            rfun=[None, "PCA"],
            weights_fun=gaussian_weights,
            weights_params={"var": 4},
        )
        assert isinstance(result, pd.DataFrame)
        assert result["level"].tolist() == [0, 1, 0, 1]
        assert np.all((result["accuracy"] >= 0) & (result["accuracy"] <= 1))

        # a cfun returning a (K, K, T) array instead of a (T, V) matrix is rejected
        with pytest.raises(ValueError):
            timepoint_decoder(
                data,
                level=[0, 1],
                nfolds=2,
                cfun=[None, lambda x, w: np.random.randn(3, 3, 6)],
                rfun=["PCA", None],
            )

    def test_weighted_timepoint_decoder_complex(self):
        """optimize_levels with too few training subjects to split raises."""
        data = np.array([np.random.randn(6, 3) for _ in range(3)])
        # 3 subjects in 2 folds leave a training group of 1 in one fold
        with pytest.raises(ValueError, match="at least 2"):
            weighted_timepoint_decoder(
                data, level=[0, 1], optimize_levels=[0, 1], nfolds=2
            )

        data = np.array([np.random.randn(6, 3) for _ in range(6)])
        np.random.seed(0)
        result = weighted_timepoint_decoder(
            data, level=[0, 1], optimize_levels=[0, 1], nfolds=2
        )
        assert result["level"].tolist() == [0, 1, 0, 1]
        full = result[result["level"] == 1]
        assert np.allclose(full[["level_0", "level_1"]].sum(axis=1), 1)

    def test_folding_levels_complex(self):
        """Level 2 with a custom cfun: one pipeline step on the two groups."""
        in_data = [np.random.randn(6, 3), np.random.randn(6, 3)]
        out_data = [np.random.randn(6, 3), np.random.randn(6, 3)]

        in_s, out_s, in_r, out_r = folding_levels(
            in_data,
            out_data,
            level=2,  # Higher level
            cfun=lambda x, w: [np.ones((6, 4)) * (j + 1) for j in range(len(x))],
            weights_fun=gaussian_weights,
        )
        # the analysis path combines each group's cfun output (mean_combine)
        assert np.allclose(in_s, 1.5) and np.allclose(out_s, 1.5)
        # the delta path gives one (T, K(K+1)/2) co-fluctuation matrix per subject
        assert len(in_r) == 2 and len(out_r) == 2
        assert all(r.shape == (6, 6) for r in in_r + out_r)

    def test_pca_decoder_edge_cases(self):
        """dims above the number of features: one result per dims value and fold."""
        data = np.array([np.random.randn(8, 5) for _ in range(4)])
        np.random.seed(0)
        result = pca_decoder(data, nfolds=2, dims=10)  # dims > features
        assert result["dims"].tolist() == list(range(1, 11)) * 2
        assert result["folds"].tolist() == [0] * 10 + [1] * 10
        assert np.all((result["accuracy"] >= 0) & (result["accuracy"] <= 1))

    def test_calculate_error_branches(self):
        """Metrics are decoder columns; an unknown metric raises KeyError."""
        mu = np.array([0.2, 0.3, 0.5])
        corrs = [np.random.randn(5, 10) for _ in range(3)]

        expected = decoder(weight_corrs(corrs, mu))
        for metric in ("error", "rank", "accuracy"):
            assert np.allclose(
                calculate_error(mu, corrs, metric=metric, sign=-1),
                -expected[metric].values,
            )
        with pytest.raises(KeyError):
            calculate_error(mu, corrs, metric="correlation", sign=-1)
        with pytest.raises(KeyError):
            calculate_error(mu, corrs, metric="unknown", sign=1)

    def test_timecorr_return_list_branch(self):
        """The smoothing path returns what combine returns (one entry unwrapped)."""
        data = np.random.randn(8, 4)
        result = timecorr(data, cfun=None, combine=lambda x: [x])
        # combine wraps the one-subject list in another list; one level is unwrapped
        assert isinstance(result, list) and len(result) == 1
        assert np.allclose(result[0], timecorr(data, cfun=None))

    def test_comprehensive_edge_cases(self):
        """Two subjects: enough for timepoint and PCA decoding, too few to fit weights."""
        minimal_data = np.array([np.random.randn(3, 2), np.random.randn(3, 2)])

        np.random.seed(0)
        result = timepoint_decoder(minimal_data, nfolds=2, level=0)
        assert result["folds"].tolist() == [0, 1]

        with pytest.raises(ValueError, match="at least 2"):
            weighted_timepoint_decoder(minimal_data, nfolds=2, level=0)

        # the _ec decoder takes data indexed by level, and splits each fold again
        with pytest.raises(ValueError, match="indexed by level"):
            weighted_timepoint_decoder_ec(minimal_data, nfolds=2, level=0)
        with pytest.raises(ValueError, match="cannot fill"):
            weighted_timepoint_decoder_ec([minimal_data], nfolds=2, level=0)

        np.random.seed(0)
        result = pca_decoder(minimal_data, nfolds=2, dims=1)
        assert result["dims"].tolist() == [1, 1]

if __name__ == "__main__":
    pytest.main([__file__])
