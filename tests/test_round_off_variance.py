"""
Features that are constant up to round-off.

A feature whose samples agree to within a few units of round-off of their
magnitude carries no variance that can be told apart from round-off.  The
estimator treats its kernel-weighted variance as zero, exactly as for a
feature whose samples are equal: correlation 0 with every feature (itself
included), and the same count in the zero-variance warning.  The rule is
relative to the feature's own kernel-weighted mean square, so it does not
depend on the units of the data.
"""

import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.helpers import _lag_kernel, _zscore_columns, wcorr

T = 80
GAUSS = {"var": 10}


def _noisy_constant(rng, level, size, rel=1e-16):
    """``level * (1 + rel * noise)``: equal to ``level`` up to round-off, not exactly."""
    col = level * (1.0 + rel * rng.standard_normal(size))
    assert np.ptp(col) > 0  # not exactly constant
    return col


def _data(seed=0, column=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 4))
    if column is not None:
        x[:, 1] = column
    return x


def _kernels():
    return {
        "dense": tc.gaussian_weights(T, GAUSS),
        "built-in": _lag_kernel(T, tc.gaussian_weights, GAUSS),
    }


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = f(*args, **kwargs)
    return out, [str(r.message) for r in rec if issubclass(r.category, UserWarning)]


@pytest.mark.parametrize("kernel", ["dense", "built-in"])
@pytest.mark.parametrize("level", [1.0, 1e150, 1e-150, -3.7])
@pytest.mark.parametrize("rel", [0.0, 1e-16, 1e-15])
def test_wcorr_round_off_constant_column_is_constant(kernel, level, rel):
    rng = np.random.default_rng(1)
    col = np.full(T, level) if rel == 0 else _noisy_constant(rng, level, T, rel)
    x = _data(2, col)
    exact = _data(2, np.full(T, level))
    w = _kernels()[kernel]
    for b_cols in (slice(None), slice(1, None)):  # autocorrelation and cross
        out, msgs = _quiet(wcorr, x, x[:, b_cols], w)
        ref, ref_msgs = _quiet(wcorr, exact, exact[:, b_cols], w)
        assert np.all(np.isfinite(out))
        assert np.all(out[1] == 0)
        col_b = 1 if b_cols == slice(None) else 0
        assert np.all(out[:, col_b] == 0)
        assert np.array_equal(out, ref)
        assert msgs == ref_msgs


@pytest.mark.parametrize("kernel", ["dense", "built-in"])
def test_wcorr_round_off_constant_column_with_nan_samples(kernel):
    rng = np.random.default_rng(3)
    x = _data(4, _noisy_constant(rng, 1.0, T))
    exact = _data(4, np.ones(T))
    for d in (x, exact):
        d[7, 1] = np.nan
        d[20, 2] = np.nan
    w = _kernels()[kernel]
    out, msgs = _quiet(wcorr, x, x, w)
    ref, ref_msgs = _quiet(wcorr, exact, exact, w)
    assert np.all(out[1] == 0) and np.all(out[:, 1] == 0)
    assert np.array_equal(out, ref)
    assert msgs == ref_msgs


def test_round_off_constant_stretch_counts_in_the_zero_variance_warning():
    # every feature constant on timepoints 10:30, with a boxcar of width 5:
    # 16 timepoints have zero variance in every feature (as for an exactly
    # constant stretch in test_estimator.py)
    rng = np.random.default_rng(5)
    exact = rng.standard_normal((50, 3))
    exact[10:30] = exact[10]
    noisy = exact.copy()
    noisy[10:30] = exact[10] * (1 + 1e-16 * rng.standard_normal((20, 3)))
    assert np.ptp(noisy[10:30], axis=0).min() > 0
    w = tc.boxcar_weights(50, {"width": 5})
    out, msgs = _quiet(wcorr, noisy, noisy, w)
    ref, ref_msgs = _quiet(wcorr, exact, exact, w)
    assert len(ref_msgs) == 1 and "16 of 50 timepoints" in ref_msgs[0]
    assert msgs == ref_msgs
    assert np.array_equal(out == 0, ref == 0)
    # windows that reach the stretch see the round-off noise in its samples
    assert np.allclose(out, ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("cfun", [tc.isfc, tc.wisfc, tc.autofc])
@pytest.mark.parametrize("form", ["dense", "built-in"])
def test_timecorr_round_off_constant_column(cfun, form):
    rng = np.random.default_rng(6)
    exact = [_data(10 + s, np.full(T, 2.5)) for s in range(3)]
    noisy = [d.copy() for d in exact]
    for d in noisy:
        d[:, 1] = _noisy_constant(rng, 2.5, T)
    if form == "dense":
        kw = dict(weights_function=lambda n, p: tc.gaussian_weights(n, GAUSS))
    else:
        kw = dict(weights_function=tc.gaussian_weights, weights_params=GAUSS)
    out, msgs = _quiet(tc.timecorr, noisy, cfun=cfun, combine=tc.null_combine, **kw)
    ref, ref_msgs = _quiet(tc.timecorr, exact, cfun=cfun, combine=tc.null_combine, **kw)
    for o, r in zip(out, ref):
        assert np.allclose(o, r, atol=1e-12, rtol=0)
        m = tc.vec2mat(o)
        assert np.all(m[1] == 0) and np.all(m[:, 1] == 0)
    assert msgs == ref_msgs


@pytest.mark.parametrize("scale", [1e150, 1e-150, 3.7, -0.001])
def test_zero_variance_rule_is_scale_invariant(scale):
    # multiplying one feature by a constant changes neither which entries are
    # treated as constant nor the correlations
    rng = np.random.default_rng(7)
    x = rng.standard_normal((T, 4))
    x[:30, 0] = 4.0 * (1 + 1e-16 * rng.standard_normal(30))  # constant up to round-off
    x[50:, 2] = -1.5  # exactly constant
    scaled = x.copy()
    scaled[:, 0] *= scale
    scaled[:, 2] *= scale
    for w in _kernels().values():
        ref, ref_msgs = _quiet(wcorr, x, x, w)
        out, msgs = _quiet(wcorr, scaled, scaled, w)
        assert np.array_equal(out == 0, ref == 0)
        sign = np.sign(scale)
        flip = np.ones(4)
        flip[[0, 2]] = sign
        expected = ref * flip[:, None, None] * flip[None, :, None]
        assert np.allclose(out, expected, atol=1e-10, rtol=0)
        assert msgs == ref_msgs


def test_delta_kernel_round_off_constant_column():
    rng = np.random.default_rng(8)
    for level in (1.0, 1e150, 1e-150):
        x = _data(9, _noisy_constant(rng, level, T))
        exact = _data(9, np.full(T, level))
        out = wcorr(x, x, tc.eye_weights(T))
        assert np.all(out[1] == 0) and np.all(out[:, 1] == 0)
        assert np.array_equal(out, wcorr(exact, exact, tc.eye_weights(T)))


def test_zscore_rule_matches_the_estimator_rule():
    # the delta path's z-scores treat a column constant up to round-off of its
    # root mean square as constant, whatever its scale
    rng = np.random.default_rng(11)
    for level in (1.0, 1e150, 1e-150, -7.0):
        x = np.column_stack([rng.standard_normal(T), _noisy_constant(rng, level, T)])
        z = _zscore_columns(x)
        assert np.all(z[:, 1] == 0)
        assert np.isclose(z[:, 0].std(), 1.0)


def test_streamed_summary_round_off_constant_column():
    # the streaming engine of the summaries reads the estimator's blocks
    rng = np.random.default_rng(12)
    exact = rng.standard_normal((120, 6))
    exact[:, 3] = 1.0
    noisy = exact.copy()
    noisy[:, 3] = _noisy_constant(rng, 1.0, 120)
    kw = dict(order=1, cfun=tc.autofc, weights_params={"var": 25},
              rfun=tc.WeightedStrength(signed=True))
    out, _ = _quiet(tc.features, noisy, **kw)
    ref, _ = _quiet(tc.features, exact, **kw)
    assert np.allclose(out[1], ref[1], atol=1e-12, rtol=0)


def test_segment_zscores_round_off_constant_column():
    # segment z-scores the features before fitting: a column constant up to
    # round-off becomes 0, not unit-variance round-off noise
    from timecorr.events import _zscore_columns as segment_zscores

    rng = np.random.default_rng(14)
    for level in (1.0, 1e150, -7.0):
        x = np.column_stack([rng.standard_normal(T), _noisy_constant(rng, level, T)])
        z = segment_zscores(x)
        assert np.all(z[:, 1] == 0)
        assert np.isclose(z[:, 0].std(), 1.0)


def test_segment_round_off_constant_column():
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(13)
    exact = np.repeat(rng.standard_normal((4, 3)), 25, axis=0)
    exact += 0.1 * rng.standard_normal(exact.shape)
    exact = np.column_stack([exact, np.full(100, 3.0)])
    noisy = exact.copy()
    noisy[:, 3] = _noisy_constant(rng, 3.0, 100)
    a = tc.segment(noisy, n=4, random_state=0)
    b = tc.segment(exact, n=4, random_state=0)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# --- Fisher averaging: a correlation within round-off of +-1 is +-1 ---------

ONE_ULP_BELOW = np.nextafter(1.0, 0.0)  # 1 - 1.1e-16
# 100 eps below 1: above the largest round-off measured on the estimator's
# +-1 outputs (79 eps), within the bound (128 eps).  These tests used
# 1 - 1e-13 (450 eps) until the bound was narrowed from 1e-12; 1 - 1e-13 is
# now a correlation (see test_tstat_combine_keeps_a_correlation_1e13_below_one)
NEAR_ONE = 1 - 100 * np.finfo(np.float64).eps


@pytest.mark.parametrize("near", [ONE_ULP_BELOW, NEAR_ONE, 1 + 1e-13])
def test_fisher_symmetrize_treats_near_unit_correlations_as_unit(near):
    # {+1, -1} averages to 0; one ulp (or 100 eps) away from -1 must not turn
    # that into +1 (z of 1 is infinite, z of 1 - 1e-16 is finite)
    from timecorr.helpers import _fisher_symmetrize

    c = np.zeros((2, 2, 1))
    c[0, 1, 0] = 1.0
    c[1, 0, 0] = -near
    out = _fisher_symmetrize(c.copy())
    exact = np.zeros((2, 2, 1))
    exact[0, 1, 0], exact[1, 0, 0] = 1.0, -1.0
    assert np.array_equal(out, _fisher_symmetrize(exact))
    assert out[0, 1, 0] == 0


@pytest.mark.parametrize("near", [ONE_ULP_BELOW, NEAR_ONE])
def test_corrmean_combine_treats_near_unit_correlations_as_unit(near):
    a = [np.array([[1.0, near]]), np.array([[-near, 0.5]])]
    b = [np.array([[1.0, 1.0]]), np.array([[-1.0, 0.5]])]
    assert np.array_equal(tc.corrmean_combine(a), tc.corrmean_combine(b))
    assert tc.corrmean_combine(a)[0, 1] == 1.0


@pytest.mark.parametrize("near", [ONE_ULP_BELOW, NEAR_ONE])
def test_tstat_combine_treats_near_unit_correlations_as_unit(near):
    a = [np.array([[1.0]]), np.array([[near]]), np.array([[1.0]])]
    b = [np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ta = tc.tstat_combine(a)
    assert ta[0, 0] == tc.tstat_combine(b)[0, 0] == np.inf


def test_correlations_clearly_below_one_keep_their_z():
    # 1 - 1e-9 is not round-off: it keeps its finite z (about 10.7)
    out = tc.corrmean_combine([np.array([[1 - 1e-9]]), np.array([[0.0]])])
    z = np.arctanh(1 - 1e-9) / 2
    assert np.isclose(out[0, 0], np.tanh(z), rtol=1e-12, atol=0)


# --- the bounds cover round-off only (outside review, round 1) --------------


def test_tstat_combine_keeps_a_correlation_1e13_below_one():
    # 1 - 1e-13 is about 450 units of round-off below 1: a correlation, not
    # +-1 with round-off, so it keeps its finite z
    r = [np.array([[v]]) for v in (1 - 1e-13, 0.5, 0.4)]
    t, p = tc.tstat_combine(r, return_pvals=True)
    z = np.arctanh([1 - 1e-13, 0.5, 0.4])
    expect = z.mean() / (z.std(ddof=1) / np.sqrt(3))
    assert np.isclose(t[0, 0], expect, rtol=1e-9) and np.isclose(t[0, 0], 1.098403,
                                                                rtol=1e-6)
    assert np.isclose(p[0, 0], 0.386596, rtol=1e-5)


@pytest.mark.parametrize("values, expect", [
    ((1 - 1e-13, -1 + 9e-13), 0.4999306166),
    ((1 - 5e-13, -0.99), 0.999986),
    ((1 - 5e-13, -(1 - 5e-13), 0.5), 0.181),
])
def test_corrmean_combine_averages_correlations_near_one_by_their_z(values, expect):
    out = tc.corrmean_combine([np.array([[v]]) for v in values])
    ref = np.tanh(np.mean(np.arctanh(values)))
    assert np.isclose(out[0, 0], ref, rtol=1e-9, atol=1e-12)
    assert np.isclose(out[0, 0], expect, rtol=1e-3 if expect == 0.181 else 1e-6)


def _measured_round_off(monkeypatch):
    """
    A small version of the measurement behind the two bounds (see
    helpers._ROUND_OFF and helpers._UNIT_TOLERANCE), regenerated here: with
    both rules off, on timecorr outputs that are +-1 in exact arithmetic
    (diagonals, and pairs of proportional columns: column 0 times 4 and
    column 1 times -0.5, in subjects that are 1, 4 and 2 times one base),
    the largest 1 - |r| and the largest zero-variance statistic sd / rms of
    those columns under each kernel as the next order's kernel, in eps.
    """
    from timecorr import helpers as H

    monkeypatch.setattr(H, "_UNIT_TOLERANCE", 0.0)
    monkeypatch.setattr(H, "_ROUND_OFF", 0.0)
    eps = np.finfo(np.float64).eps
    kernels = [(tc.gaussian_weights, {"var": 10}), (tc.laplace_weights, {"scale": 20}),
               (tc.t_weights, None), (tc.boxcar_weights, {"width": 9})]
    unit = rule = 0.0
    for T, K in ((60, 6), (200, 3)):
        base = np.random.default_rng(0).standard_normal((T, K))
        dup = np.column_stack([base, base[:, :1] * 4, -base[:, 1:2] * 0.5])
        iu = np.triu_indices(K + 2)
        cols = [n for n, (i, j) in enumerate(zip(*iu))
                if i == j or (i, j) in {(0, K), (1, K + 1)}]
        for wf, wp in kernels:
            for cfun in (tc.autofc, tc.isfc):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = tc.timecorr([dup * c for c in (1.0, 4.0, 2.0)],
                                      weights_function=wf, weights_params=wp,
                                      cfun=cfun, combine=tc.null_combine)
                for o in out:
                    x = o[:, cols]
                    x = x[:, np.isfinite(x).all(axis=0) & (np.abs(x) > 0.5).all(axis=0)]
                    unit = max(unit, (1 - np.abs(x)).max() / eps)
                    for nf, npar in kernels:
                        w = np.abs(nf(T, npar))
                        m = (w @ x) / w.sum(axis=1, keepdims=True)
                        var = np.stack([(w * (x[:, j][None, :] - m[:, j][:, None]) ** 2)
                                        .sum(axis=1) for j in range(x.shape[1])], axis=1)
                        rule = max(rule, np.sqrt(var / (w @ (x * x))).max() / eps)
    return unit, rule


def test_the_unit_tolerance_covers_the_estimators_round_off(monkeypatch):
    # the bound covers 1 - |r| of correlations that are exactly +-1, measured
    # again here (79 eps in the full measurement), and the one-ulp flip of
    # {+1, -1 + 1 ulp} to +1 is still removed
    from timecorr import helpers as H
    bound = H._UNIT_TOLERANCE
    eps = np.finfo(np.float64).eps
    unit, _ = _measured_round_off(monkeypatch)
    monkeypatch.undo()
    assert unit > 10  # the measurement sees round-off
    assert bound >= unit * eps
    # and the recorded maximum of the full measurement: 79 eps (outputs of
    # timecorr with autofc, isfc and wisfc and of corrmean_combine and
    # mean_combine that are +-1 in exact arithmetic; T 60 to 2000, K 3 and
    # 6, nine built-in kernels; Accelerate and OpenBLAS alike; see the
    # comment above helpers._UNIT_TOLERANCE)
    assert bound >= 79 * eps
    assert bound < 1e-13
    near = 1 - H._UNIT_TOLERANCE
    out = tc.corrmean_combine([np.array([[1.0]]), np.array([[-near]])])
    assert out[0, 0] == 0.0
    out = tc.corrmean_combine([np.array([[1.0]]), np.array([[-ONE_ULP_BELOW]])])
    assert out[0, 0] == 0.0


def test_a_large_offset_keeps_its_spread():
    # samples 0..4 on an offset of 1e14: the relative spread sd / rms is
    # 1.41e-14 (63.7 eps), which is a genuine spread above the round-off bound
    x = np.arange(5.0)[:, None] + 1e14
    w = np.ones((1, 5))
    assert wcorr(x, x, w)[0, 0, 0] == 1.0


def test_round_off_bound_covers_the_measured_round_off(monkeypatch):
    # the zero-variance statistic of output columns that are constant in
    # exact arithmetic, measured again here (37.95 eps in the full
    # measurement): the bound is above it and below the 1e14-offset example
    # (63.7 eps)
    from timecorr import helpers as H
    bound = H._ROUND_OFF
    eps = np.finfo(np.float64).eps
    _, rule = _measured_round_off(monkeypatch)
    monkeypatch.undo()
    assert rule > 5  # the measurement sees round-off
    assert rule * eps < bound < 63 * eps
    # and the recorded maximum of the full measurement: 37.95 eps (the same
    # outputs, T 60 to 2000, K 3 and 6, nine built-in kernels as the next
    # order's kernel; Accelerate and OpenBLAS alike; see the comment above
    # helpers._ROUND_OFF)
    assert bound > 37.95 * eps
