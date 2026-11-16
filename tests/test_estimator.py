"""Tests for the kernel-weighted estimator (``wcorr``) and the functions built on it."""

import inspect
import warnings

import numpy as np
import pytest
import scipy.spatial.distance as sd

import timecorr as tc
from timecorr.helpers import (
    autofc,
    boxcar_weights,
    eye_weights,
    gaussian_weights,
    isfc,
    laplace_weights,
    mat2vec,
    mexican_hat_weights,
    r2z,
    uniform_weights,
    vec2mat,
    wcorr,
    wisfc,
    z2r,
)


def _weighted_pearson(a, b, w):
    """Direct weighted Pearson correlation matrix between columns of a and b."""
    ka = a.shape[1]
    c = np.cov(np.hstack([a, b]).T, aweights=w)
    s = np.sqrt(np.diag(c))
    return (c / np.outer(s, s))[:ka, ka:]


def _rng(seed=0):
    return np.random.RandomState(seed)


# (a) uniform kernel equals np.corrcoef
def test_uniform_kernel_equals_corrcoef():
    rng = _rng(1)
    T = 50
    a = rng.randn(T, 4) @ rng.randn(4, 4)
    b = rng.randn(T, 3) + a[:, :3]
    w = uniform_weights(T)
    auto = wcorr(a, a, w)
    cross = wcorr(a, b, w)
    ref_auto = np.corrcoef(a.T)
    ref_cross = np.corrcoef(np.hstack([a, b]).T)[:4, 4:]
    for t in range(T):
        assert np.allclose(auto[:, :, t], ref_auto, atol=1e-10, rtol=0)
        assert np.allclose(cross[:, :, t], ref_cross, atol=1e-10, rtol=0)


# (b) Gaussian kernel equals a direct weighted Pearson correlation at every t
def test_gaussian_kernel_equals_direct_weighted_pearson():
    rng = _rng(2)
    T = 60
    a = rng.randn(T, 4) + 3.0
    b = rng.randn(T, 5) - 2.0
    w = gaussian_weights(T, {"var": 30})
    cross = wcorr(a, b, w)
    auto = wcorr(a, a, w)
    for t in range(T):
        assert np.allclose(
            cross[:, :, t], _weighted_pearson(a, b, w[t]), atol=1e-10, rtol=0
        )
        assert np.allclose(
            auto[:, :, t], _weighted_pearson(a, a, w[t]), atol=1e-10, rtol=0
        )


def test_rows_not_columns_define_the_kernel():
    rng = _rng(3)
    T = 40
    a = rng.randn(T, 3)
    w = rng.rand(T, T)
    out = wcorr(a, a, w)
    for t in (0, 17, 39):
        assert np.allclose(
            out[:, :, t], _weighted_pearson(a, a, w[t]), atol=1e-10, rtol=0
        )


def test_autocorrelation_diagonal_is_exactly_one():
    rng = _rng(4)
    x = rng.randn(80, 6) * 5 + 10
    out = wcorr(x, x, laplace_weights(80, {"scale": 10}))
    for t in range(80):
        assert np.all(np.diag(out[:, :, t]) == 1.0)
        assert np.array_equal(out[:, :, t], out[:, :, t].T)


# (c) Mexican hat output lies in [-1, 1]
def test_mexican_hat_output_bounded():
    rng = _rng(5)
    T = 120
    a = rng.randn(T, 5)
    b = rng.randn(T, 4) + a[:, :4]
    w = mexican_hat_weights(T, {"sigma": 10})
    assert np.any(w < 0)
    for out in (wcorr(a, a, w), wcorr(a, b, w)):
        assert np.all(np.isfinite(out))
        assert np.all(out <= 1) and np.all(out >= -1)
        assert np.abs(out).max() > 0.5  # not trivially zero


# (d) delta kernel: co-fluctuation (product of z-scores)
def test_delta_kernel_is_zscore_product():
    rng = _rng(6)
    T = 70
    a = rng.randn(T, 4) @ rng.randn(4, 4) + 2
    b = rng.randn(T, 3)
    za = (a - a.mean(0)) / a.std(0)
    zb = (b - b.mean(0)) / b.std(0)
    w = eye_weights(T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cross = wcorr(a, b, w)
        auto = wcorr(a, a, w)
    for t in range(T):
        assert np.allclose(cross[:, :, t], np.outer(za[t], zb[t]), atol=1e-10, rtol=0)
        assert np.allclose(auto[:, :, t], np.outer(za[t], za[t]), atol=1e-10, rtol=0)
    assert np.allclose(auto.mean(axis=2), np.corrcoef(a.T), atol=1e-10, rtol=0)
    ref = np.corrcoef(np.hstack([a, b]).T)[:4, 4:]
    assert np.allclose(cross.mean(axis=2), ref, atol=1e-10, rtol=0)


# (e) block data recovery
def test_block_recovery():
    T, K = 300, 10
    iu = np.triu_indices(K, 1)
    rs = []
    for seed in range(100, 110):
        y, corrs = tc.simulate_data(
            datagen="block", return_corrs=True, T=T, K=K, set_random_seed=seed
        )
        truth = vec2mat(corrs)
        est = wcorr(y, y, laplace_weights(T, {"scale": 20}))
        rs.append(
            np.mean(
                [
                    np.corrcoef(est[:, :, t][iu], truth[:, :, t][iu])[0, 1]
                    for t in range(T)
                ]
            )
        )
    assert np.mean(rs) >= 0.6


# (f) Review focus 1: NaN samples carry no weight
def test_nan_samples_carry_no_weight_pairwise():
    rng = _rng(7)
    T = 50
    a = rng.randn(T, 3)
    b = rng.randn(T, 2) + a[:, :2]
    a[[4, 20, 21], 0] = np.nan
    b[[20, 33], 1] = np.nan
    w = gaussian_weights(T, {"var": 40})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = wcorr(a, b, w)
    assert out.shape == (3, 2, T)
    assert np.all(np.isfinite(out))
    for t in (0, 20, 49):
        for i in range(3):
            for j in range(2):
                ok = ~np.isnan(a[:, i]) & ~np.isnan(b[:, j])
                ref = _weighted_pearson(a[ok, i : i + 1], b[ok, j : j + 1], w[t, ok])
                assert np.isclose(out[i, j, t], ref[0, 0], atol=1e-10, rtol=0)


@pytest.mark.parametrize(
    "kernel, params",
    [(gaussian_weights, {"var": 20}), (mexican_hat_weights, {"sigma": 5})],
)
def test_nan_sample_in_all_features_equals_zero_weight(kernel, params):
    rng = _rng(8)
    T = 40
    x = rng.randn(T, 4)
    w = kernel(T, params)
    xn = x.copy()
    xn[10, :] = np.nan
    w0 = w.copy()
    w0[:, 10] = 0
    assert np.allclose(wcorr(xn, xn, w), wcorr(x, x, w0), atol=1e-10, rtol=0)


def test_timecorr_nan_input_finite_and_unweighted():
    rng = _rng(9)
    T, K = 60, 3
    x = rng.randn(T, K)
    x[5, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tc.timecorr(x, weights_params={"var": 50})
    assert out.shape == (T, K * (K + 1) // 2)
    assert np.all(np.isfinite(out))
    w = gaussian_weights(T, {"var": 50})
    ok = ~np.isnan(x[:, 0])
    m = vec2mat(out)
    for t in (0, 5, 59):
        ref = _weighted_pearson(x[ok][:, [0]], x[ok][:, [1]], w[t, ok])[0, 0]
        assert np.isclose(m[0, 1, t], ref, atol=1e-10, rtol=0)


@pytest.mark.parametrize("cfun", [isfc, wisfc, autofc])
def test_multisubject_nan_input_finite(cfun):
    rng = _rng(10)
    data = [rng.randn(40, 3) for _ in range(3)]
    data[1][7, 2] = np.nan
    data[2][7, 2] = np.nan
    data[0][30, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tc.timecorr(data, cfun=cfun, weights_params={"var": 20})
    out = np.asarray(out)
    assert out.shape == (3, 40, 6)
    assert np.all(np.isfinite(out))


# (f) Review focus 2: zero variance inside a kernel window gives 0
def test_zero_variance_window_gives_zero():
    rng = _rng(11)
    T = 60
    x = rng.randn(T, 3)
    x[:30, 0] = 4.0  # constant in the first half
    w = boxcar_weights(T, {"width": 10})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = wcorr(x, x, w)
        cross = wcorr(x, x[:, 1:], w)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(cross))
    for t in range(25):  # windows fully inside the constant segment
        assert np.all(out[0, :, t] == 0) and np.all(out[:, 0, t] == 0)
        assert np.all(cross[0, :, t] == 0)
    assert np.all(np.diag(out[:, :, 50]) == 1.0)


def test_constant_feature_everywhere():
    rng = _rng(12)
    x = rng.randn(30, 3)
    x[:, 1] = 7.0
    for w in (gaussian_weights(30, {"var": 10}), eye_weights(30)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = wcorr(x, x, w)
        assert np.all(np.isfinite(out))
        assert np.all(out[1, :, :] == 0) and np.all(out[:, 1, :] == 0)


# (f) Review focus 3: short series
@pytest.mark.parametrize(
    "T, kernel, params",
    [
        (2, gaussian_weights, {"var": 100}),
        (5, gaussian_weights, {"var": 100}),
        (3, boxcar_weights, {"width": 10}),
        (2, laplace_weights, {"scale": 100}),
    ],
)
def test_short_series(T, kernel, params):
    K = 4
    x = _rng(13).randn(T, K)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tc.timecorr(x, weights_function=kernel, weights_params=params)
    assert out.shape == (T, K * (K + 1) // 2)
    assert np.all(np.isfinite(out))


# Undefined kernel rows
def test_undefined_rows_zero_with_one_warning():
    rng = _rng(14)
    T = 30
    x = rng.randn(T, 3)
    w = gaussian_weights(T, {"var": 10})
    w[3, :] = 0
    w[5, :] = 0
    w[7, :] = 0
    w[7, 2] = 1.0  # single support
    with pytest.warns(UserWarning, match=r"\b3\b") as rec:
        out = wcorr(x, x, w)
    assert len([r for r in rec if issubclass(r.category, UserWarning)]) == 1
    for t in (3, 5, 7):
        assert np.all(out[:, :, t] == 0)
    assert np.all(np.isfinite(out))
    assert np.all(np.diag(out[:, :, 4]) == 1.0)


def test_nan_kernel_weights_are_zero():
    rng = _rng(15)
    x = rng.randn(20, 3)
    w = gaussian_weights(20, {"var": 10})
    w[0, :] = np.nan
    with pytest.warns(UserWarning, match=r"\b1\b"):
        out = wcorr(x, x, w)
    assert np.all(out[:, :, 0] == 0) and np.all(np.isfinite(out))


def test_all_rows_zero_raises():
    x = _rng(16).randn(10, 2)
    with pytest.raises(ValueError):
        wcorr(x, x, np.zeros([10, 10]))


# (g) mat2vec / vec2mat
def _mat2vec_loop(m):
    """The 0.2.0 loop implementation of mat2vec, kept here as a reference."""
    K = m.shape[0]
    V = int((((K**2) - K) / 2) + K)
    if m.ndim == 2:
        y = np.zeros(V)
        y[0:K] = np.diag(m)
        m = np.triu(m - np.diag(np.diag(m)))
        m[np.isnan(m)] = 0
        m += m.T
        y[K:] = sd.squareform(m - np.diag(np.diag(m)))
        return y
    T = m.shape[2]
    y = np.zeros([T, V])
    for t in np.arange(T):
        y[t, :] = _mat2vec_loop(np.squeeze(m[:, :, t]))
    return y


def test_mat2vec_matches_loop_version():
    rng = _rng(17)
    m = rng.randn(6, 6, 25)
    m[1, 4, 3] = np.nan
    m[2, 2, 5] = np.nan
    assert np.array_equal(mat2vec(m), _mat2vec_loop(m), equal_nan=True)
    assert np.array_equal(mat2vec(m[:, :, 0]), _mat2vec_loop(m[:, :, 0]))


def test_mat2vec_vec2mat_roundtrip():
    rng = _rng(18)
    v = rng.randn(12, 15)  # K = 5
    assert np.array_equal(mat2vec(vec2mat(v)), v)
    assert np.array_equal(mat2vec(vec2mat(v[0])), v[0])
    m = vec2mat(v)
    assert m.shape == (5, 5, 12)
    assert np.array_equal(m, m.transpose(1, 0, 2))


def test_vec2mat_single_row_2d():
    v = np.arange(6.0)[np.newaxis, :]  # K = 3, one timepoint
    m = vec2mat(v)
    assert m.shape == (3, 3, 1)
    assert np.array_equal(mat2vec(m), v)


# (h) ISFC symmetrization
def test_isfc_is_symmetrized_cross_correlation():
    rng = _rng(19)
    T, K = 40, 4
    data = [rng.randn(T, K) + rng.randn(T, 1) for _ in range(3)]
    w = gaussian_weights(T, {"var": 25})
    out = isfc(data, w)
    for s in range(3):
        others = np.mean([d for k, d in enumerate(data) if k != s], axis=0)
        m = vec2mat(out[s])
        for t in (0, 13, 39):
            C = _weighted_pearson(data[s], others, w[t])
            assert np.allclose(m[:, :, t], (C + C.T) / 2, atol=1e-10, rtol=0)


def test_wisfc_has_no_bare_except():
    src = inspect.getsource(wisfc)
    assert "mystery" not in src and "except:" not in src


# Fisher z helpers
def test_z2r_inf_is_one():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = z2r(np.array([np.inf, -np.inf, 0.0, np.nan]))
    assert np.array_equal(r, [1.0, -1.0, 0.0, 0.0])


def test_r2z_at_plus_minus_one_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z = r2z(np.array([1.0, -1.0, 0.5]))
        assert np.array_equal(z[:2], [np.inf, -np.inf])
        assert np.isclose(z[2], np.arctanh(0.5))
        assert np.allclose(z2r(z), [1.0, -1.0, 0.5], atol=1e-15, rtol=0)


# timecorr keeps the 0.2.0 positional order; include_zero is last
def test_timecorr_positional_order():
    names = list(inspect.signature(tc.timecorr).parameters)
    assert names == [
        "data",
        "weights_function",
        "weights_params",
        "include_timepoints",
        "exclude_timepoints",
        "combine",
        "cfun",
        "rfun",
        "include_zero",
    ]
