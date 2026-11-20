"""
Time-sliced estimation: the estimator processes the output timepoints in
chunks, each against the span of kernel columns where the chunk has non-zero
weight.  These tests compare it with the full products (the private switch
``helpers._TIME_SLICING = False``) on real computations.

The two differ only in the order of floating-point summation.  Where a
kernel-weighted variance nearly cancels (it is a small fraction of the
kernel-weighted mean square: data constant within most of the kernel, or a
kernel row whose observed samples carry only tail weights), that round-off is
amplified, and either path can differ from the other visibly (by up to about
1e-6 measured on Linux x86-64 with OpenBLAS).  ``_check_conditioned``
compares such data entry by entry according to a conditioning measure
computed from the data and the kernel.
"""

import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import helpers
from timecorr.helpers import _corr_row_blocks, _time_slices, _wcorr

RTOL, ATOL = 1e-10, 1e-12


@pytest.fixture(autouse=True)
def _restore_switches(monkeypatch):
    """Every test restores the module switches it changes (monkeypatch)."""
    monkeypatch.setattr(helpers, "_TIME_SLICING", helpers._TIME_SLICING)
    monkeypatch.setattr(helpers, "_time_slice_log", helpers._time_slice_log)


def _run(fn, sliced):
    """``fn()`` with time slicing on or off; returns (result, warning texts)."""
    saved = helpers._TIME_SLICING
    helpers._TIME_SLICING = sliced
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn()
    finally:
        helpers._TIME_SLICING = saved
    return result, [str(w.message) for w in caught]


def _sliced_and_full(fn):
    return _run(fn, True), _run(fn, False)


def _assert_close(x, y):
    if isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for xi, yi in zip(x, y):
            _assert_close(xi, yi)
        return
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape
    assert np.allclose(x, y, rtol=RTOL, atol=ATOL)


def _check(fn):
    (s, ws), (f, wf) = _sliced_and_full(fn)
    _assert_close(s, f)
    assert ws == wf
    return s, ws


def _engaged(fn):
    """The slices recorded while ``fn()`` runs with time slicing on."""
    saved = helpers._time_slice_log
    helpers._time_slice_log = []
    try:
        _run(fn, True)
        return helpers._time_slice_log
    finally:
        helpers._time_slice_log = saved


@pytest.fixture(scope="module")
def x():
    return np.random.RandomState(0).randn(1200, 5)


@pytest.fixture(scope="module")
def y():
    return np.random.RandomState(1).randn(1200, 4)


def test_gaussian_narrow_kernel_auto_and_cross(x, y):
    w = tc.gaussian_weights(len(x), {"var": 25})
    _check(lambda: _wcorr(x, x, w))
    _check(lambda: _wcorr(x, y, w))
    slices = _engaged(lambda: _wcorr(x, x, w))
    assert slices and slices[0] is not None
    spans = [c1 - c0 for _, _, c0, c1 in slices[0]]
    assert max(spans) < len(x) / 2


def test_slicing_skipped_for_wide_kernel(x):
    w = tc.gaussian_weights(len(x), {"var": 1e5})
    assert _time_slices(w) is None
    assert _engaged(lambda: _wcorr(x, x, w)) == [None]


def test_slicing_off_switch(x, monkeypatch):
    w = tc.gaussian_weights(len(x), {"var": 25})
    assert _time_slices(w) is not None
    monkeypatch.setattr(helpers, "_TIME_SLICING", False)
    assert _time_slices(w) is None


def test_boxcar(x, y):
    w = tc.boxcar_weights(len(x), {"width": 30})
    _check(lambda: _wcorr(x, y, w))
    assert _engaged(lambda: _wcorr(x, y, w))[0] is not None


def test_mexican_hat(x):
    # a signed kernel narrow enough to be sliced (it underflows to 0 beyond
    # about 116 samples at sigma 3)
    w = tc.mexican_hat_weights(len(x), {"sigma": 3})
    assert (w < 0).any()
    _check(lambda: _wcorr(x, x, w))
    slices = _engaged(lambda: _wcorr(x, x, w))
    assert slices and slices[0] is not None


def test_one_sided(x):
    for direction in ("backward", "forward"):
        for include_zero in (True, False):
            w = tc.gaussian_weights(
                len(x), {"var": 25}, direction=direction, include_zero=include_zero
            )
            _, msgs = _check(lambda: _wcorr(x, x, w))
            # the first (or last) rows have fewer than 3 weights
            assert msgs and "fewer than 3" in msgs[0]
    _check(
        lambda: tc.timecorr(
            x, weights_params={"var": 25}, cfun=tc.autofc, include_timepoints="pre"
        )
    )


def test_event_masked_kernels(x):
    # a kernel as wide as the data, masked to blocks: block diagonal
    events = [150, 400, 401, 403, 800, 1000]
    fn = lambda: tc.timecorr(  # noqa: E731
        x,
        weights_function=tc.uniform_weights,
        cfun=tc.autofc,
        events=events,
    )
    _, msgs = _check(fn)
    assert msgs and "fewer than 3" in msgs[0]
    slices = _engaged(fn)
    assert any(s is not None for s in slices)
    _check(
        lambda: tc.timecorr(
            x, weights_params={"var": 400}, cfun=tc.autofc, events=events
        )
    )


# The conditioning of entry (i, j) at kernel row t: the smaller, over the two
# features, of the kernel-weighted variance (computed in two passes) divided
# by the kernel-weighted mean square, over the samples where both features
# are observed, of the column-centered data (as the estimator centers them).
# Round-off of the estimator's one-pass variance is about eps / conditioning
# relative to the correlation; at conditioning >= _WELL_CONDITIONED (1e-3)
# the sliced and full products differ by at most about 1e-12 (measured: at
# most 9.8e-13 on Linux x86-64 with OpenBLAS, while entries below 1.1e-4
# differed by up to 2.4e-6).  The estimator sets an entry to 0 when a
# variance is at most _ZERO_VARIANCE_TOL (1e-10) of its mean square: entries
# well below that are 0 in both paths, entries well above it are not.
_WELL_CONDITIONED = 1e-3
_SURELY_ZERO = helpers._ZERO_VARIANCE_TOL / 10
_SURELY_NONZERO = helpers._ZERO_VARIANCE_TOL * 10


def _conditioning(a, b, w):
    """
    ``(cond, undefined)``, both (Ka, Kb, T'): the conditioning of each entry
    (see above) and the entries with fewer than 3 observed samples with
    non-zero weight (undefined; the estimator sets them to 0).  Computed from
    the data and the kernel only.
    """
    a = a - np.nanmean(a, axis=0)
    b = b - np.nanmean(b, axis=0)
    A = np.abs(np.nan_to_num(w, nan=0.0))
    T_out, T = A.shape
    cond = np.empty((a.shape[1], b.shape[1], T_out))
    undefined = np.empty(cond.shape, dtype=bool)
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            ok = ~np.isnan(a[:, i]) & ~np.isnan(b[:, j])
            Wk = A * ok
            s = Wk.sum(axis=1)
            ratios = []
            for v in (a[:, i], b[:, j]):
                v = np.where(ok, v, 0.0)
                mean = np.divide(Wk @ v, s, out=np.zeros(T_out), where=s > 0)
                var = (Wk * (v[np.newaxis, :] - mean[:, np.newaxis]) ** 2).sum(axis=1)
                msq = Wk @ v**2
                ratios.append(np.divide(var, msq, out=np.zeros(T_out), where=msq > 0))
            cond[i, j] = np.minimum(*ratios)
            undefined[i, j] = (Wk > 0).sum(axis=1) < 3
    return cond, undefined


def _well_conditioned_close(s, f, good):
    return bool(np.allclose(s[good], f[good], rtol=RTOL, atol=ATOL))


def _check_conditioned(a, b, w):
    """
    ``_check`` for ``_wcorr(a, b, w)`` on data with ill-conditioned entries:
    well-conditioned entries agree at RTOL / ATOL; every entry is finite and
    within [-1, 1] in both paths; entries the estimator defines as 0
    (undefined, or variance far below its zero tolerance) are 0 in both, and
    entries with variance far above it are non-zero in both; the warnings are
    the same.
    """
    (s, ws), (f, wf) = _sliced_and_full(lambda: _wcorr(a, b, w))
    assert s.shape == f.shape
    cond, undefined = _conditioning(a, b, w)
    good = ~undefined & (cond >= _WELL_CONDITIONED)
    # most entries are well conditioned, and some are not (the case is real)
    assert good.mean() > 0.6 and not good.all()
    assert _well_conditioned_close(s, f, good)
    for r in (s, f):
        assert np.all(np.isfinite(r)) and np.all(np.abs(r) <= 1)
        assert np.all(r[undefined | (cond <= _SURELY_ZERO)] == 0)
        assert np.all(r[~undefined & (cond >= _SURELY_NONZERO)] != 0)
    assert ws == wf
    return s, ws, good


def _shortened_slices(monkeypatch):
    """Make every non-empty span of ``_time_slices`` one column too short."""
    original = helpers._time_slices

    def short(weights, finite=True):
        slices = original(weights, finite)
        if slices is None:
            return None
        return [(t0, t1, c0, c1 - 1 if c1 > c0 else c1) for t0, t1, c0, c1 in slices]

    monkeypatch.setattr(helpers, "_time_slices", short)


def _assert_dropped_column_detected(a, b, w, monkeypatch):
    """Negative control: with spans one column too short, the comparison of
    the well-conditioned entries fails."""
    s, _, good = _check_conditioned(a, b, w)
    assert _engaged(lambda: _wcorr(a, b, w))[0] is not None
    with monkeypatch.context() as m:
        _shortened_slices(m)
        short = _run(lambda: _wcorr(a, b, w), True)[0]
    assert not _well_conditioned_close(short, s, good)


def test_nan_data(x, y, monkeypatch):
    a = x.copy()
    b = y.copy()
    rng = np.random.RandomState(2)
    a[rng.rand(*a.shape) < 0.1] = np.nan
    b[rng.rand(*b.shape) < 0.1] = np.nan
    # some pairs with fewer than 3 observed samples; tens of samples inside
    # the stretch, the observed samples carry only tail weights (about 1e-196
    # of the peak), so the variances nearly cancel
    a[450:750, 1] = np.nan
    w = tc.gaussian_weights(len(x), {"var": 4})
    _, msgs, _ = _check_conditioned(a, a, w)
    assert any("observed (non-NaN)" in m for m in msgs)
    _check_conditioned(a, b, w)
    assert _engaged(lambda: _wcorr(a, b, w))[0] is not None
    # a Gaussian's last column of support has negligible weight, so the
    # negative control uses a boxcar, where every supported column counts
    box = tc.boxcar_weights(len(x), {"width": 30})
    _assert_dropped_column_detected(a, a, box, monkeypatch)
    _assert_dropped_column_detected(a, b, box, monkeypatch)


def test_cancelled_and_low_ess_warnings(x, monkeypatch):
    a = x.copy()
    a[300:700] = 1.0  # constant: zero kernel-weighted variance there
    w = tc.gaussian_weights(len(x), {"var": 25})
    _, msgs, _ = _check_conditioned(a, a, w)
    assert msgs and "zero kernel-weighted variance" in msgs[0]
    _assert_dropped_column_detected(
        a, a, tc.boxcar_weights(len(x), {"width": 30}), monkeypatch
    )
    narrow = tc.gaussian_weights(len(x), {"var": 0.1})
    _, msgs = _check(lambda: _wcorr(x, x, narrow))
    assert msgs and "effective sample size" in msgs[0]


def test_isfc_wisfc_three_subjects():
    rng = np.random.RandomState(3)
    data = [rng.randn(900, 4) for _ in range(3)]
    for cfun in (tc.isfc, tc.wisfc, tc.autofc):
        _check(
            lambda: tc.timecorr(
                data, weights_params={"var": 25}, cfun=cfun, combine=tc.null_combine
            )
        )


def test_row_blocks_streamed(x, y):
    w = tc.gaussian_weights(len(x), {"var": 25})
    for a, b, symmetric in ((x, x, True), (x, y, False)):
        (s, _), (f, _) = _sliced_and_full(
            lambda: list(_corr_row_blocks(a, b, w, symmetric, rows_per_block=2))
        )
        assert len(s) == len(f)
        for bs, bf in zip(s, f):
            assert bs[:2] == bf[:2]
            _assert_close(bs[2], bf[2])
            assert np.array_equal(bs[3], bf[3]) and np.array_equal(bs[4], bf[4])


def test_features_order_2_end_to_end():
    rng = np.random.RandomState(4)
    data = [rng.randn(1500, 8) for _ in range(2)]
    fn = lambda: tc.features(  # noqa: E731
        data, order=2, cfun=tc.autofc, weights_params={"var": 25}
    )
    _check(fn)
    slices = _engaged(fn)
    assert any(s is not None for s in slices)


def test_features_streamed_summary():
    rng = np.random.RandomState(5)
    data = [rng.randn(700, 12) for _ in range(2)]
    _check(
        lambda: tc.features(
            data,
            order=1,
            cfun=tc.autofc,
            weights_params={"var": 25},
            rfun=tc.WeightedStrength(signed=True),
        )
    )


def test_empty_kernels_are_not_sliced():
    assert _time_slices(np.zeros((5, 0))) is None
    assert _time_slices(np.zeros((0, 5))) is None
    assert _time_slices(np.zeros((0, 0))) is None


def test_support_scan_does_not_hold_a_full_boolean_kernel():
    # the support bounds are found one chunk of rows at a time
    import tracemalloc

    T = 2000
    w = tc.boxcar_weights(T, {"width": 12})
    _time_slices(w)  # warm up
    tracemalloc.start()
    try:
        assert _time_slices(w) is not None
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < T * T / 4  # a T x T boolean matrix is T * T bytes


@pytest.mark.parametrize("kind", ["overflow", "inf"])
def test_non_finite_products_are_not_sliced(kind):
    # squares of 1e200 overflow: the full products give 0 * inf = NaN, so
    # the sliced path is not used and the result is the full one exactly,
    # NaN included; data with inf are not sliced either
    a = np.random.RandomState(0).randn(600, 2)
    big = 1e200 if kind == "overflow" else np.inf
    a[500] = big
    a[501] = -big
    w = np.zeros((600, 600))
    w[:, 0:3] = 1.0
    (s, ws), (f, wf) = _sliced_and_full(lambda: _wcorr(a, a, w))
    if kind == "overflow":
        assert np.isnan(f).any()
    np.testing.assert_array_equal(s, f)  # NaN positions equal too
    assert ws == wf
    assert _engaged(lambda: _wcorr(a, a, w)) == [None]
    # the same kernel on finite data with ordinary squares is sliced
    b = np.random.RandomState(0).randn(600, 2)
    assert _engaged(lambda: _wcorr(b, b, w))[0] is not None


def test_nan_weights(x):
    w = tc.gaussian_weights(len(x), {"var": 25})
    rng = np.random.RandomState(6)
    band = np.abs(np.subtract.outer(np.arange(len(x)), np.arange(len(x)))) < 10
    w[band & (rng.rand(*w.shape) < 0.05)] = np.nan  # inside the band
    w[~band & (rng.rand(*w.shape) < 1e-4)] = np.nan  # outside it
    _check(lambda: _wcorr(x, x, w))
    assert _engaged(lambda: _wcorr(x, x, w))[0] is not None


def test_rows_with_disjoint_supports(x, y):
    # alternating rows weight two windows that do not overlap, and one chunk
    # of rows has no weight at all
    T = len(x)
    w = np.zeros((T, T))
    for t in range(T):
        lo = t if t % 2 == 0 else min(T - 7, t + 20)
        w[t, max(0, lo - 3):lo + 4] = 1.0 + 0.1 * (t % 5)
    w[256:384] = 0
    slices = _engaged(lambda: _wcorr(x, y, w))[0]
    assert slices is not None
    assert (256, 384, 0, 0) in slices
    _check(lambda: _wcorr(x, y, w))
    _check(lambda: _wcorr(x, x, w))


def test_events_that_differ_by_subject_under_isfc():
    rng = np.random.RandomState(7)
    data = [rng.randn(900, 4) for _ in range(3)]
    events = [[150, 400], None, [300, 301, 700]]
    for cfun in (tc.isfc, tc.wisfc):
        fn = lambda: tc.timecorr(  # noqa: E731
            data, weights_params={"var": 25}, cfun=cfun, events=events,
            combine=tc.null_combine,
        )
        _check(fn)
        assert any(s is not None for s in _engaged(fn))
