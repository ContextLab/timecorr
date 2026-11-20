"""
Time-sliced estimation: the estimator processes the output timepoints in
chunks, each against the span of kernel columns where the chunk has non-zero
weight.  These tests compare it with the full products (the private switch
``helpers._TIME_SLICING = False``) on real computations.
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


def test_nan_data(x, y):
    a = x.copy()
    b = y.copy()
    rng = np.random.RandomState(2)
    a[rng.rand(*a.shape) < 0.1] = np.nan
    b[rng.rand(*b.shape) < 0.1] = np.nan
    a[450:750, 1] = np.nan  # some pairs with fewer than 3 observed samples
    w = tc.gaussian_weights(len(x), {"var": 4})
    _, msgs = _check(lambda: _wcorr(a, a, w))
    assert any("observed (non-NaN)" in m for m in msgs)
    _check(lambda: _wcorr(a, b, w))
    assert _engaged(lambda: _wcorr(a, b, w))[0] is not None


def test_cancelled_and_low_ess_warnings(x):
    a = x.copy()
    a[300:700] = 1.0  # constant: zero kernel-weighted variance there
    w = tc.gaussian_weights(len(x), {"var": 25})
    _, msgs = _check(lambda: _wcorr(a, a, w))
    assert msgs and "zero kernel-weighted variance" in msgs[0]
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
