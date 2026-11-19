"""Tests for the kernel-weighted estimator (``wcorr``) and the functions built on it."""

import inspect
import warnings

import numpy as np
import pytest
import scipy.spatial.distance as sd

import timecorr as tc
from timecorr.helpers import (
    autofc,
    _fisher_symmetrize,
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
    if T < 3:
        # R12: every row of a 2-timepoint kernel has 2 weights, so is undefined
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = tc.timecorr(x, weights_function=kernel, weights_params=params)
        assert [(r.category, str(r.message)) for r in rec] == [(
            UserWarning,
            f"{T} of {T} timepoints have fewer than 3 non-zero kernel weights; "
            "their correlations are undefined and set to 0.",
        )]
        assert np.all(out == 0)
    else:
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


def test_backward_kernel_rows_with_fewer_than_3_weights_are_zero():
    # R12: two samples always give r = +-1, so a row needs 3 non-zero weights
    rng = _rng(30)
    T = 30
    x = rng.randn(T, 3)
    w = tc.one_sided(gaussian_weights(T, {"var": 10}), "backward", include_zero=True)
    with pytest.warns(
        UserWarning, match="2 of 30 timepoints have fewer than 3 non-zero kernel weights"
    ) as rec:
        out = wcorr(x, x, w)
    assert len([r for r in rec if issubclass(r.category, UserWarning)]) == 1
    assert np.all(out[:, :, 0] == 0) and np.all(out[:, :, 1] == 0)
    defined = out[:, :, 2]
    assert np.all(np.diag(defined) == 1.0)
    off = defined[np.triu_indices(3, 1)]
    assert np.all(np.abs(off) < 1) and np.all(off != 0)
    ref = _weighted_pearson(x, x, w[2])
    assert np.allclose(off, ref[np.triu_indices(3, 1)])


def test_row_with_exactly_two_weights_is_undefined():
    rng = _rng(31)
    T = 20
    x = rng.randn(T, 2)
    w = gaussian_weights(T, {"var": 10})
    w[10, :] = 0
    w[10, [4, 9]] = 1.0
    with pytest.warns(UserWarning, match=r"1 of 20 timepoints have fewer than 3"):
        out = wcorr(x, x, w)
    assert np.all(out[:, :, 10] == 0)
    assert np.all(np.diag(out[:, :, 11]) == 1.0)


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


def test_mat2vec_allocates_little_beyond_its_output():
    # Unreduced higher-order runs vectorize (d, d, T) stacks of several GB; full
    # size temporaries (a fancy-indexed copy of the upper triangle and a
    # NaN-replaced copy of it) used to triple mat2vec's peak memory.
    import tracemalloc

    m = _rng(19).randn(150, 150, 200)
    m[3, 7, 11] = np.nan
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        y = mat2vec(m)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak <= 1.25 * y.nbytes
    assert np.array_equal(y, _mat2vec_loop(m), equal_nan=True)


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
            # Eq. 5: average the Fisher z of C and C^T, then invert
            expected = np.tanh((np.arctanh(C) + np.arctanh(C.T)) / 2)
            assert np.allclose(m[:, :, t], expected, atol=1e-10, rtol=0)


def test_isfc_symmetrization_handles_unit_correlations():
    # subject 0 equals the others' average in feature 0 and its negative in
    # feature 1, so C contains entries of exactly +1 and -1
    rng = _rng(20)
    T = 30
    base = rng.randn(T, 3)
    data = [base.copy(), base.copy(), base.copy()]
    data[0][:, 1] = -base[:, 0]
    data[1][:, 1] = base[:, 1]
    data[2][:, 1] = base[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = isfc(data, gaussian_weights(T, {"var": 20}))
    m = vec2mat(out[0])
    assert np.all(np.isfinite(m))
    # C[0, 0] = 1 and C[1, 0] = -1 (subject feature 1 vs others' feature 0)
    C = wcorr(data[0], data[1], gaussian_weights(T, {"var": 20}))
    assert np.allclose(C[0, 0], 1) and np.allclose(C[1, 0], -1)
    # A cross-correlation of identical columns is a ratio of sums: it is
    # within a few ulp of 1 (not always exactly 1, depending on the BLAS), so
    # it is compared within 1e-12 (~4500 ulp, far below any real error).  The
    # symmetrized diagonal entry is tanh(arctanh(C[0, 0])), which is within
    # an ulp or two of C[0, 0] even when arctanh(C[0, 0]) is large (~9).
    assert np.allclose(m[0, 0], 1.0, atol=1e-12, rtol=0)
    # pair (0, 1): when C[1, 0] is exactly -1 its z is -inf and the pair is
    # -1 (exact, by construction).  When the platform rounds C[1, 0] to within
    # an ulp of -1 instead, its z is finite (~-9), and the pair is the Fisher
    # average of the two entries; either way the result is what
    # _fisher_symmetrize gives for the cross-correlation computed here (same
    # function, same inputs, same machine), which is checked next to it.
    if np.all(C[1, 0] == -1.0):
        assert np.all(m[0, 1] == -1.0) and np.all(m[1, 0] == -1.0)
    assert np.allclose(
        m, _fisher_symmetrize(wcorr(data[0], data[1], gaussian_weights(T, {"var": 20}))),
        atol=1e-12, rtol=0,
    )


def test_fisher_symmetrize_exact_unit_entries_dominate_their_pair():
    # platform independent: the cross-correlation has entries of exactly
    # +-1 by construction (not computed), so the rule is exact on any machine
    C = np.zeros((3, 3, 4))
    C[...] = 0.3
    C[0, 0] = 1.0
    C[1, 0] = -1.0
    C[2, 1] = 1.0
    C[1, 2] = -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _fisher_symmetrize(C.copy())
    assert np.all(out[0, 0] == 1.0)
    assert np.all(out[1, 0] == -1.0) and np.all(out[0, 1] == -1.0)
    assert np.all(out[2, 1] == 0.0) and np.all(out[1, 2] == 0.0)  # +1 and -1 cancel
    assert np.allclose(out[0, 2], 0.3, atol=1e-12, rtol=0)


def test_corrmean_combine_keeps_unit_diagonal():
    rng = _rng(21)
    data = [rng.randn(30, 3) for _ in range(3)]
    auto = autofc(data, gaussian_weights(30, {"var": 20}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = tc.corrmean_combine(auto)
    assert mean.shape == (30, 6)
    assert np.all(mean[:, :3] == 1.0)
    ref = np.tanh(np.mean([np.arctanh(x[:, 3:]) for x in auto], axis=0))
    assert np.allclose(mean[:, 3:], ref, atol=1e-12, rtol=0)


def _no_warning_corrmean(subjects):
    # one row per subject; each row is one timepoint of vectorized values
    corrs = [np.array([row], dtype=float) for row in subjects]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tc.corrmean_combine(corrs)[0]


def test_corrmean_combine_opposite_infinities_use_remaining_finite():
    # R11: +1 and -1 together are excluded; the finite z values are averaged
    out = _no_warning_corrmean([[1.0], [-1.0], [0.5]])
    assert np.allclose(out, [0.5], atol=1e-12, rtol=0)


def test_corrmean_combine_only_opposite_infinities_give_zero():
    assert np.array_equal(_no_warning_corrmean([[1.0], [-1.0]]), [0.0])
    assert np.array_equal(_no_warning_corrmean([[1.0], [-1.0], [-1.0]]), [0.0])


def test_corrmean_combine_same_sign_infinities_win():
    out = _no_warning_corrmean([[1.0, -1.0], [1.0, -1.0], [0.3, 0.3]])
    assert np.array_equal(out, [1.0, -1.0])


def test_corrmean_combine_mixed_columns_in_one_call():
    out = _no_warning_corrmean(
        [[1.0, 1.0, 0.2], [-1.0, 1.0, 0.4], [0.5, 0.3, -0.1]]
    )
    ref = np.tanh(np.mean(np.arctanh([0.2, 0.4, -0.1])))
    assert np.allclose(out, [0.5, 1.0, ref], atol=1e-12, rtol=0)


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


# timecorr keeps the 0.2.0 positional order; include_zero is the last
# positional parameter, and the event options are keyword-only
def test_timecorr_positional_order():
    params = inspect.signature(tc.timecorr).parameters
    positional = [
        n for n, p in params.items() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
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
    assert list(params) == positional + ["events", "events_params"]
    assert all(
        params[n].kind == inspect.Parameter.KEYWORD_ONLY for n in ("events", "events_params")
    )


# Fisher z averaging applies to correlations only: entries with any value
# outside [-1, 1] (delta-kernel co-fluctuations) are combined on the raw scale.
def test_corrmean_combine_raw_mean_outside_unit_interval():
    out = _no_warning_corrmean([[2.0, 0.5, 0.2], [0.5, 1.0, -3.0], [-0.3, 0.3, 0.4]])
    assert np.allclose(out[0], np.mean([2.0, 0.5, -0.3]), atol=1e-15, rtol=0)
    assert out[1] == 1.0  # all inside [-1, 1]: Fisher rule (+1 wins)
    assert np.allclose(out[2], np.mean([0.2, -3.0, 0.4]), atol=1e-15, rtol=0)


def test_tstat_combine_raw_scale_outside_unit_interval():
    from scipy.stats import ttest_1samp

    corrs = [np.array([[2.0, 0.5]]), np.array([[0.5, 0.2]]), np.array([[1.5, -0.1]])]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts = tc.tstat_combine(corrs)
    assert np.all(np.isfinite(ts))
    assert np.isclose(ts[0, 0], ttest_1samp([2.0, 0.5, 1.5], 0).statistic)
    assert np.isclose(
        ts[0, 1], ttest_1samp(np.arctanh([0.5, 0.2, -0.1]), 0).statistic
    )


@pytest.mark.parametrize("combine", ["corrmean_combine", "tstat_combine"])
def test_delta_kernel_group_combines_are_finite(combine):
    rng = _rng(31)
    data = [rng.randn(40, 4) for _ in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tc.timecorr(
            data, weights_function=eye_weights, cfun=tc.isfc,
            combine=getattr(tc, combine),
        )
    assert out.shape == (40, 10)
    assert np.all(np.isfinite(out))
    if combine == "corrmean_combine":
        # co-fluctuations are unbounded: the group value is their plain mean
        per = tc.timecorr(data, weights_function=eye_weights, cfun=tc.isfc)
        raw = np.any(np.abs(np.stack(per)) > 1, axis=0)
        assert raw.any()
        assert np.allclose(out[raw], np.mean(np.stack(per), axis=0)[raw],
                           atol=1e-14, rtol=0)


def test_features_corrmean_combine_finite_with_pca_and_delta_kernel():
    rng = _rng(32)
    data = [rng.randn(60, 5) for _ in range(3)]
    for kernel in (gaussian_weights, eye_weights):
        # no NaN reaches the reducer (hypertools would warn and impute it)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X = tc.features(data, order=2, rfun="PCA", combine=tc.corrmean_combine,
                            weights_function=kernel, weights_params=None)
        for k in (1, 2):
            assert np.all(np.isfinite(X[k]))


# wisfc default subject weights: zero weight on the subject itself, always
def _with_exact_correlation(R, T, seed):
    """(T, K) data whose sample correlation matrix is exactly R (to rounding)."""
    rng = _rng(seed)
    Z = rng.randn(T, R.shape[0])
    Z -= Z.mean(axis=0)
    Q, _ = np.linalg.qr(Z)  # centered, orthonormal columns
    return Q @ np.linalg.cholesky(R).T


def test_wisfc_default_weights_uncorrelated_connectomes_use_the_other_subject():
    # The two connectomes (0.5, 0, -0.5) and (0.3, -0.3, 0.3) are uncorrelated,
    # so the computed subject weight is ~1e-16; the reference must still be
    # the other subject (it was all NaN, giving all-zero output).
    R1 = np.array([[1, 0.5, 0.0], [0.5, 1, -0.5], [0.0, -0.5, 1]])
    R2 = np.array([[1, 0.3, -0.3], [0.3, 1, 0.3], [-0.3, 0.3, 1]])
    data = [_with_exact_correlation(R1, 40, 1), _with_exact_correlation(R2, 40, 2)]
    w = gaussian_weights(40, {"var": 20})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = wisfc(data, w)
    ref = wisfc(data, w, subject_weights=1 - np.eye(2))
    for o, r in zip(out, ref):
        assert np.array_equal(o, r)
        assert np.any(o[:, 3:] != 0)


@pytest.mark.parametrize("K", [1, 2])
def test_wisfc_default_weights_with_too_few_features_exclude_the_subject(K):
    # with K <= 2 there are fewer than 3 connectome entries to correlate, so
    # every other subject gets the same weight and the subject itself none
    rng = _rng(40 + K)
    data = [rng.randn(30, K) for _ in range(3)]
    w = gaussian_weights(30, {"var": 20})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = wisfc(data, w)
    ref = wisfc(data, w, subject_weights=1 - np.eye(3))
    for o, r in zip(out, ref):
        assert np.array_equal(o, r)


def test_tstat_combine_needs_at_least_two_subjects():
    one = [np.full((4, 3), 0.5)]
    with pytest.raises(ValueError, match="at least two"):
        tc.tstat_combine(one)
    with pytest.raises(ValueError, match="at least two"):
        tc.tstat_combine([])


# --- all-issues wave: narrow kernels (R-A), signed-kernel diagonal (R-B),
# --- large offsets (R-N), the |r| <= 1 tolerance (R-O), tstat on constant
# --- entries (R-H)


def _user_warnings(rec):
    return [r for r in rec if issubclass(r.category, UserWarning)]


def test_narrow_kernel_warns_once_about_effective_sample_size():
    # R-A (M1): var=0.05 has full support (no row has fewer than 3 non-zero
    # weights) but each row is one sample plus weights of 4.5e-5 and less,
    # so its effective sample size (sum|w|)**2 / sum(w**2) is about 1
    rng = _rng(50)
    T = 40
    x = rng.randn(T, 3)
    w = gaussian_weights(T, {"var": 0.05})
    assert np.all(np.count_nonzero(w, axis=1) >= 3)
    with pytest.warns(UserWarning, match="40 of 40 timepoints have an effective") as rec:
        wcorr(x, x, w)
    assert len(_user_warnings(rec)) == 1


def test_kernel_whose_variances_all_cancel_is_reported():
    # R-A (M1): var=0.01 gives weights of 1e-22 off the diagonal, so every
    # kernel-weighted variance is below the zero-variance tolerance and the
    # output is all zeros: that must be reported, not returned silently
    rng = _rng(51)
    T = 30
    x = rng.randn(T, 3)
    w = gaussian_weights(T, {"var": 0.01})
    with pytest.warns(UserWarning) as rec:
        out = wcorr(x, x, w)
    msgs = [str(r.message) for r in _user_warnings(rec)]
    assert len(msgs) == 1
    assert "30 of 30 timepoints" in msgs[0] and "variance" in msgs[0]
    assert np.all(out == 0)


def test_timepoints_where_every_feature_is_constant_are_reported():
    # R-A: a well-sampled kernel (boxcar, 5 equal weights, effective sample
    # size 5) over a stretch where every feature is constant: the timepoints
    # whose whole window is constant (12 to 27) have no variance at all
    rng = _rng(52)
    T = 50
    x = rng.randn(T, 3)
    x[10:30] = x[10]
    w = boxcar_weights(T, {"width": 5})
    with pytest.warns(UserWarning, match="16 of 50 timepoints") as rec:
        out = wcorr(x, x, w)
    assert len(_user_warnings(rec)) == 1
    assert np.all(out[:, :, 12:28] == 0)
    assert np.all(np.diag(out[:, :, 5]) == 1.0)


def test_one_constant_feature_is_not_reported():
    # only timepoints where every correlation cancels are reported
    rng = _rng(53)
    x = rng.randn(40, 3)
    x[:, 1] = 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = wcorr(x, x, gaussian_weights(40, {"var": 10}))
    assert np.all(out[1, :, :] == 0)


@pytest.mark.parametrize("cfun", [isfc, autofc, wisfc])
def test_narrow_kernel_warns_once_per_call_for_several_subjects(cfun):
    rng = _rng(54)
    data = [rng.randn(30, 3) for _ in range(3)]
    with pytest.warns(UserWarning, match="effective sample size") as rec:
        cfun(data, gaussian_weights(30, {"var": 0.05}))
    assert len(_user_warnings(rec)) == 1


def test_delta_kernel_never_warns_about_effective_sample_size():
    rng = _rng(55)
    x = rng.randn(30, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wcorr(x, x, eye_weights(30))
        tc.timecorr([x, rng.randn(30, 4)], weights_function=eye_weights, cfun=autofc)


def test_signed_kernel_diagonal_exception_is_documented():
    # R-B (M2): the diagonal of an autocorrelation is pinned to 1, which for a
    # signed kernel differs from the formula; the docstring formula says so
    doc = " ".join(wcorr.__doc__.split())
    assert "for signed kernels the diagonal is an exception" in doc
    rng = _rng(56)
    x = rng.randn(60, 2)
    out = wcorr(x, x, mexican_hat_weights(60, {"sigma": 3}))
    assert np.all(np.diag(out[:, :, 30]) == 1.0)


@pytest.mark.parametrize("offset", [1e9, 1e10, 1.1e10, 1e11])
def test_large_constant_offset_does_not_zero_a_feature(offset):
    # R-N: an offset of 1e9 (1e11) standard deviations must not make a
    # feature look constant.  On these data 0.2.0's tolerance zeroed the
    # offset features on the delta path from an offset of about 1.088e10 on:
    # 1e10 still passed there and 1.1e10 is the first case here that failed.
    # Storing offset + x rounds x to the spacing of floats near the offset
    # (1.2e-7 at 1e9, 1.5e-5 at 1e11), so results agree with the offset-free
    # ones to about that spacing times a small factor.
    rng = _rng(57)
    T = 60
    x = rng.randn(T, 3)
    shifted = x + np.array([offset, 0.0, -offset])
    tol = 100 * np.spacing(offset)
    for w in (gaussian_weights(T, {"var": 10}), eye_weights(T)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ref = wcorr(x, x, w)
            out = wcorr(shifted, shifted, w)
        assert np.max(np.abs(out - ref)) < tol


def test_constant_column_with_large_offset_stays_constant():
    # the round-off of centering a constant column must not pass as signal
    rng = _rng(58)
    x = rng.randn(40, 2)
    x[:, 1] = 1e9 + 0.1
    out = wcorr(x, x, eye_weights(40))
    assert np.all(out[1, :, :] == 0)
    assert np.all(out[:, 1, :] == 0)


def test_values_within_round_off_above_one_are_correlations():
    # R-O: 1 + 1e-13 is a correlation of 1 with round-off, not a
    # co-fluctuation: Fisher z (infinite, so the mean is 1), not a raw mean
    corrs = [np.array([[1 + 1e-13, 0.5]]), np.array([[0.5, 0.5]])]
    out = tc.corrmean_combine(corrs)
    assert out[0, 0] == 1.0
    assert np.isclose(out[0, 1], 0.5)
    # a value clearly outside [-1, 1] is still a co-fluctuation (raw mean)
    raw = tc.corrmean_combine([np.array([[1.5]]), np.array([[0.5]])])
    assert np.isclose(raw[0, 0], 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts = tc.tstat_combine([np.array([[-1 - 1e-13]]), np.array([[-1.0]])])
    assert ts[0, 0] == -np.inf


def test_tstat_combine_constant_entries_are_signed_infinities():
    # R-H (M11): zero variance across subjects gives +-inf with the sign of
    # the mean (0 when the mean is 0), p = 0 (p = 1 for t = 0), no warning
    a = np.array([[0.3, -0.2, 0.0, 1.0, 0.1]])
    b = np.array([[0.3, -0.2, 0.0, 1.0, 0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts, ps = tc.tstat_combine([a, b, a.copy()], return_pvals=True)
    assert ts[0, 0] == np.inf and ts[0, 1] == -np.inf and ts[0, 2] == 0
    assert ts[0, 3] == np.inf
    assert ps[0, 0] == 0 and ps[0, 1] == 0 and ps[0, 2] == 1 and ps[0, 3] == 0
    z = r2z(np.array([0.1, 0.5, 0.1]))
    ref = z.mean() / (z.std(ddof=1) / np.sqrt(3))
    assert np.isclose(ts[0, 4], ref)


def test_tstat_combine_with_autofc_has_an_infinite_diagonal_and_no_warning():
    # R-H (M11): autofc's diagonal is exactly 1 for every subject
    rng = _rng(59)
    data = [rng.randn(30, 3) for _ in range(4)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts = tc.timecorr(
            data, cfun=autofc, combine=tc.tstat_combine, weights_params={"var": 10}
        )
    diag = [0, 1, 2]  # mat2vec puts the diagonal first
    assert np.all(ts[:, diag] == np.inf)
    assert np.all(np.isfinite(np.delete(ts, diag, axis=1)))


# Ruling R-A (revised): the low-effective-sample-size threshold is 2


def test_one_sided_kernel_warns_only_about_undefined_rows():
    # row 2 of a backward Gaussian (var 100) has 3 unequal weights, effective
    # sample size just under 3 but above 2: only the undefined-row clause
    rng = _rng(60)
    x = rng.randn(100, 4)
    for direction in ("backward", "forward"):
        with pytest.warns(UserWarning) as rec:
            tc.timecorr(x, weights_params={"var": 100}, include_timepoints=direction)
        msgs = [str(r.message) for r in _user_warnings(rec)]
        assert msgs == [
            "2 of 100 timepoints have fewer than 3 non-zero kernel weights; their "
            "correlations are undefined and set to 0."
        ]


def test_exclude_timepoints_two_has_no_low_effective_sample_size_warning():
    # |lag| <= 2 keeps 3 weights at the edges (effective sample size about
    # 3) and no row has fewer than 3: nothing to warn about
    rng = _rng(61)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tc.timecorr(rng.randn(12, 4), exclude_timepoints=2)


def test_effective_sample_size_threshold_is_two():
    # rows of 3 non-zero weights: (1, 1, 1) has ESS 3, (1, 0.1, 0.1) has
    # 1.44 / 1.02 = 1.41 < 2; only the second warns
    rng = _rng(62)
    T = 20
    x = rng.randn(T, 3)
    w = gaussian_weights(T, {"var": 10})
    w[10, :] = 0
    w[10, [9, 10, 11]] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wcorr(x, x, w)
    w[10, [9, 11]] = 0.1
    with pytest.warns(UserWarning, match="1 of 20 timepoints have an effective sample size"):
        wcorr(x, x, w)


def test_warns_sections_name_all_three_estimator_clauses():
    # m3: every function whose correlations go through the estimator lists
    # the three clauses its UserWarning can carry, as the message words them
    from timecorr.higher_order import two_path_step
    for f in (tc.timecorr, wcorr, tc.isfc, tc.wisfc, tc.autofc, tc.features,
              two_path_step):
        doc = " ".join(f.__doc__.split())
        warns = doc.split("Warns ----- ", 1)
        assert len(warns) == 2, f.__name__
        section = warns[1]
        assert "fewer than 3 non-zero kernel weights" in section, f.__name__
        assert "effective sample size" in section and "below 2" in section, f.__name__
        assert "zero kernel-weighted variance in every feature" in section, f.__name__
