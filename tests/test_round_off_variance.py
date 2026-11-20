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
