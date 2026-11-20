"""
Kernels read a chunk of rows at a time.

For the built-in kernel functions, ``timecorr``, ``features``,
``two_path_step``, the streamed summaries and the decoders never build a
T x T kernel: a ``LagKernel`` computes the rows of each chunk over the span
of columns where they can have weight.  These tests compare it with the T x T
arrays a kernel function of the user's returns (the same estimator reads
both, chunk by chunk, so the results are equal entry for entry), check that
no T x T array is allocated, and compare the memory estimates with traced
peaks.
"""

import tracemalloc
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import higher_order as ho
from timecorr import summaries as sm
from timecorr._kernels import LagKernel
from timecorr.events import mask_kernel
from timecorr.timecorr import _shaped_kernel

KERNELS = [
    (tc.gaussian_weights, {"var": 25}),
    (tc.gaussian_weights, {"var": 100}),
    (tc.laplace_weights, {"scale": 3}),
    (tc.laplace_weights, None),
    (tc.t_weights, {"df": 3}),
    (tc.mexican_hat_weights, {"sigma": 4}),
    (tc.uniform_weights, None),
    (tc.boxcar_weights, {"width": 20}),
    (tc.eye_weights, None),
]
KERNEL_IDS = [f"{f.__name__}-{p}" for f, p in KERNELS]

OPTIONS = [
    dict(),
    dict(include_timepoints="pre"),
    dict(include_timepoints="post", include_zero=False),
    dict(include_zero=False),
    dict(exclude_timepoints=15),
    dict(exclude_timepoints=-3),
    dict(events=[100, 250, 251, 330]),
    dict(events=[[150, 300], None, [200, 201, 350]]),
]


def as_array_function(fn):
    """The same kernel through a function of the user's (T x T arrays)."""
    return lambda T, params: fn(T, params)


def run(f, *args, **kwargs):
    """``f(*args, **kwargs)`` and its warning texts (or the exception)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = f(*args, **kwargs)
        except Exception as err:  # compared as text below
            out = f"{type(err).__name__}: {err}"
    return out, [str(w.message) for w in caught]


def assert_identical(x, y):
    if isinstance(x, (list, tuple)):
        assert isinstance(y, (list, tuple)) and len(x) == len(y)
        for a, b in zip(x, y):
            assert_identical(a, b)
        return
    if isinstance(x, str) or isinstance(y, str):
        assert x == y
        return
    np.testing.assert_array_equal(x, y)


def traced_peak(f):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = f()
        return tracemalloc.get_traced_memory()[1], out
    finally:
        tracemalloc.stop()


@pytest.fixture(scope="module")
def subjects():
    rng = np.random.RandomState(0)
    return [rng.randn(400, 4) for _ in range(3)]


@pytest.fixture(scope="module")
def subjects_nan(subjects):
    rng = np.random.RandomState(1)
    out = [x.copy() for x in subjects]
    for x in out:
        x[rng.rand(*x.shape) < 0.05] = np.nan
    return out


# ---------------------------------------------------------------- the kernel


@pytest.mark.parametrize("fn, params", KERNELS, ids=KERNEL_IDS)
def test_lag_kernel_equals_the_kernel_matrix(fn, params):
    T = 400
    for options in OPTIONS:
        include = options.get("include_timepoints", "all")
        include_zero = options.get("include_zero", True)
        exclude = options.get("exclude_timepoints")
        lazy = _shaped_kernel(T, fn, params, include, include_zero, exclude)
        dense = _shaped_kernel(T, as_array_function(fn), params, include, include_zero,
                               exclude)
        assert isinstance(lazy, LagKernel) and isinstance(dense, np.ndarray)
        np.testing.assert_array_equal(lazy.dense(), dense)
        events = options.get("events")
        if events is not None and not isinstance(events[0], list):
            np.testing.assert_array_equal(
                mask_kernel(lazy, events).dense(), mask_kernel(dense, events)
            )
        # every rectangle, including ones that cross the chunk boundaries
        for t0, t1, c0, c1 in ((0, 1, 0, T), (5, 140, 0, 37), (390, 400, 300, 400),
                               (127, 257, 100, 300)):
            np.testing.assert_array_equal(lazy.block(t0, t1, c0, c1),
                                          dense[t0:t1, c0:c1])
    # the span holds every non-zero weight of its rows
    lazy = _shaped_kernel(T, fn, params)
    dense = lazy.dense()
    for t0 in range(0, T, 128):
        t1 = min(T, t0 + 128)
        c0, c1 = lazy.span(t0, t1)
        assert not np.any(dense[t0:t1, :c0]) and not np.any(dense[t0:t1, c1:])


def test_only_built_in_kernel_functions_are_lazy():
    assert isinstance(_shaped_kernel(50, tc.gaussian_weights, None), LagKernel)
    assert isinstance(_shaped_kernel(50, as_array_function(tc.gaussian_weights), None),
                      np.ndarray)
    with pytest.raises(ValueError, match="include_timepoints"):
        _shaped_kernel(50, tc.gaussian_weights, None, "sideways")


def test_events_that_leave_every_row_short_raise_for_both_kernels():
    lazy = _shaped_kernel(10, tc.uniform_weights, None)
    with pytest.raises(ValueError, match="fewer than 3"):
        mask_kernel(lazy, list(range(1, 10, 2)))
    with pytest.raises(ValueError, match="fewer than 3"):
        mask_kernel(lazy.dense(), list(range(1, 10, 2)))


# ------------------------------------------------- equality with T x T arrays


@pytest.mark.parametrize("fn, params", KERNELS, ids=KERNEL_IDS)
@pytest.mark.parametrize("cfun", [tc.isfc, tc.wisfc, tc.autofc],
                         ids=["isfc", "wisfc", "autofc"])
def test_timecorr_equals_the_array_kernel(fn, params, cfun, subjects, subjects_nan):
    for data in (subjects, subjects_nan):
        for options in OPTIONS:
            kw = dict(weights_params=params, cfun=cfun, **options)
            lazy = run(tc.timecorr, data, weights_function=fn, **kw)
            dense = run(tc.timecorr, data, weights_function=as_array_function(fn), **kw)
            assert_identical(lazy[0], dense[0])
            assert lazy[1] == dense[1]


@pytest.mark.parametrize("fn, params", KERNELS, ids=KERNEL_IDS)
def test_smoothing_equals_the_array_kernel(fn, params, subjects_nan):
    # the smoothing path sums over each chunk's span: equal up to the order
    # of summation of non-negative weights
    for options in OPTIONS:
        kw = dict(weights_params=params, cfun=None, **options)
        lazy = run(tc.timecorr, subjects_nan, weights_function=fn, **kw)
        dense = run(tc.timecorr, subjects_nan, weights_function=as_array_function(fn),
                    **kw)
        assert lazy[1] == dense[1]
        for a, b in zip(lazy[0], dense[0]):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan], ids=["inf", "-inf", "nan"])
@pytest.mark.parametrize("row", [127, 128, 129, 150])
def test_smoothing_with_a_non_finite_sample(value, row):
    # an infinite sample enters every row's sums, as in the full product of
    # the T x T kernel: 0 * inf is NaN where the kernel gives it no weight
    # and +-inf where it does; a NaN sample gets no weight.  The result does
    # not depend on where the chunks of 128 rows start.
    x = np.random.RandomState(0).randn(300, 2)
    x[row, 1] = value
    kw = dict(cfun=None, weights_params={"width": 10})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lazy = tc.timecorr(x, weights_function=tc.boxcar_weights, **kw)
        dense = tc.timecorr(x, weights_function=as_array_function(tc.boxcar_weights),
                            **kw)
    np.testing.assert_array_equal(lazy, dense)
    assert np.all(np.isfinite(lazy[:, 0]))
    weighted = np.abs(np.arange(300) - row) < 5  # boxcar of width 10
    if np.isnan(value):
        assert np.all(np.isfinite(lazy[:, 1]))
    else:
        assert np.all(lazy[weighted, 1] == value)
        assert np.all(np.isnan(lazy[~weighted, 1]))


@pytest.mark.parametrize("exclude", [2.5, -0.4, 1.6, -2.5, 1e9, -1e9])
def test_non_integer_exclude_timepoints_is_rounded_for_every_kernel(exclude, subjects):
    # documented: real values are rounded to the nearest integer
    kw = dict(weights_params={"var": 4}, cfun=tc.autofc, exclude_timepoints=exclude)
    lazy = run(tc.timecorr, subjects, weights_function=tc.gaussian_weights, **kw)
    dense = run(tc.timecorr, subjects,
                weights_function=as_array_function(tc.gaussian_weights), **kw)
    assert_identical(lazy[0], dense[0])
    assert lazy[1] == dense[1]
    rounded = run(tc.timecorr, subjects, weights_function=tc.gaussian_weights,
                  **dict(kw, exclude_timepoints=int(np.round(exclude))))
    assert_identical(lazy[0], rounded[0])


def test_masking_twice_intersects_the_events():
    lazy = _shaped_kernel(300, tc.gaussian_weights, {"var": 400})
    dense = lazy.dense()
    twice = mask_kernel(mask_kernel(lazy, [100]), [200])
    np.testing.assert_array_equal(twice.dense(),
                                  mask_kernel(mask_kernel(dense, [100]), [200]))
    np.testing.assert_array_equal(twice.dense(), mask_kernel(dense, [100, 200]))
    assert twice.dense()[50, 150] == 0 and twice.dense()[150, 250] == 0


def test_a_precomputed_matrix_keeps_working(subjects):
    w = tc.gaussian_weights(400, {"var": 25})
    lazy = tc.timecorr(subjects, weights_params={"var": 25}, cfun=tc.autofc)
    for a, b in zip(lazy, tc.autofc(subjects, w)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        tc.wcorr(subjects[0], subjects[0], w),
        tc.wcorr(subjects[0], subjects[0], _shaped_kernel(400, tc.gaussian_weights,
                                                          {"var": 25}).dense()),
    )


def test_custom_cfun_receives_arrays(subjects):
    seen = []

    def cfun(data, weights):
        seen.append(weights)
        return tc.autofc(data, weights)

    out = tc.timecorr(subjects, cfun=cfun, events=[[100], [100], None])
    assert all(isinstance(w, np.ndarray) and w.shape == (400, 400) for w in seen)
    assert len(seen) == 2
    ref = tc.timecorr(subjects, cfun=tc.autofc, events=[[100], [100], None])
    for a, b in zip(out, ref):
        np.testing.assert_array_equal(a, b)


def _defaults_as_arrays():
    # the default analysis and feeding kernels (Gaussian, variance 100 and 25)
    # as functions of the user's; the parameters carry "var", so the shift
    # null reads the same bandwidth
    return dict(
        weights_function=as_array_function(tc.gaussian_weights),
        weights_params={"var": 100},
        delta_weights_function=as_array_function(tc.gaussian_weights),
        delta_weights_params={"var": 25},
    )


@pytest.mark.parametrize("cfun", [tc.isfc, tc.autofc], ids=["isfc", "autofc"])
def test_features_equal_the_array_kernels(cfun):
    rng = np.random.RandomState(2)
    data = [rng.randn(300, 6) for _ in range(2)]
    lazy = run(tc.features, data, order=2, cfun=cfun, return_delta=True,
               weights_params={"var": 100}, delta_weights_params={"var": 25})
    dense = run(tc.features, data, order=2, cfun=cfun, return_delta=True,
                **_defaults_as_arrays())
    assert_identical(lazy[0], dense[0])
    assert lazy[1] == dense[1]
    # streamed analysis steps, with per-subject events
    kw = dict(order=2, cfun=cfun, rfun=tc.WeightedStrength(signed=True),
              events=[[100], [150]])
    lazy = run(tc.features, data, weights_params={"var": 100},
               delta_weights_params={"var": 25}, **kw)
    dense = run(tc.features, data, **_defaults_as_arrays(), **kw)
    assert_identical(lazy[0], dense[0])
    assert lazy[1] == dense[1]


def test_two_path_step_equals_the_array_kernels():
    rng = np.random.RandomState(3)
    data = [rng.randn(300, 6) for _ in range(2)]
    lazy = run(tc.two_path_step, data, order=1, cfun=tc.autofc,
               weights_params={"var": 100}, delta_weights_params={"var": 25})
    dense = run(tc.two_path_step, data, order=1, cfun=tc.autofc, **_defaults_as_arrays())
    assert_identical(lazy[0], dense[0])
    assert lazy[1] == dense[1]


def test_decoders_equal_the_array_kernel():
    rng = np.random.RandomState(4)
    data = [rng.randn(120, 5) for _ in range(4)]
    for decoder in (tc.timepoint_decoder, tc.weighted_timepoint_decoder):
        np.random.seed(0)
        lazy = run(decoder, np.array(data), level=1, nfolds=2,
                   weights_function=tc.laplace_weights, weights_params={"scale": 5})
        np.random.seed(0)
        dense = run(decoder, np.array(data), level=1, nfolds=2,
                    weights_function=as_array_function(tc.laplace_weights),
                    weights_params={"scale": 5})
        assert lazy[1] == dense[1]
        assert type(lazy[0]) is type(dense[0])
        if isinstance(lazy[0], str):
            assert lazy[0] == dense[0]
        else:
            assert lazy[0].equals(dense[0])


# ------------------------------------------------------- no T x T allocation

T_LONG = 30_000
ONE_KERNEL = T_LONG * T_LONG * 8  # bytes of one T x T float64 array (7.2 GB)


@pytest.fixture(scope="module")
def long_subjects():
    rng = np.random.RandomState(5)
    return [rng.randn(T_LONG, 4) for _ in range(2)]


@pytest.mark.parametrize("kw", [
    dict(cfun=tc.autofc),
    dict(cfun=tc.isfc, events=[[1000, 20_000], None]),
    dict(cfun=tc.wisfc, include_timepoints="pre", exclude_timepoints=200),
    dict(cfun=None, weights_function=tc.boxcar_weights, weights_params={"width": 40}),
    dict(cfun=tc.autofc, weights_function=tc.mexican_hat_weights),
    dict(cfun=tc.autofc, weights_function=tc.eye_weights),
], ids=["autofc", "isfc-events", "wisfc-pre-exclude", "smoothing-boxcar",
        "mexican-hat", "eye"])
def test_timecorr_allocates_no_t_by_t_array(long_subjects, kw):
    peak, out = traced_peak(lambda: run(tc.timecorr, long_subjects, **kw))
    assert not isinstance(out[0], str), out[0]
    # a T x T boolean mask alone would be ONE_KERNEL / 8 bytes
    assert peak < ONE_KERNEL / 40, peak


def test_features_and_two_path_step_allocate_no_t_by_t_array(long_subjects):
    peak, out = traced_peak(lambda: run(
        tc.features, long_subjects, order=2, cfun=tc.autofc,
        rfun=tc.WeightedStrength(signed=True),
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak < ONE_KERNEL / 40, peak
    peak, out = traced_peak(lambda: run(
        tc.two_path_step, long_subjects, order=1, cfun=tc.isfc,
        rfun=tc.WeightedStrength(signed=True), delta_rfun="weighted_strength",
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak < ONE_KERNEL / 40, peak


def test_memory_guard_admits_long_recordings(long_subjects):
    # the estimates no longer hold T x T kernels: a step at T 30,000 fits a
    # budget far below one such array (max_memory_gb is checked before the
    # step runs, and a T x T kernel would not fit it)
    budget = ONE_KERNEL / 1e9 / 20
    est = ho.estimate_memory_gb(4, 1, T_LONG, 2)[1]
    assert est < budget
    X = run(tc.features, long_subjects, order=1, cfun=tc.autofc, rfun=None,
            delta_rfun=None, max_memory_gb=budget)[0]
    assert not isinstance(X, str), X
    # a kernel function of the user's returns T x T arrays, counted as such
    dense = ho.estimate_memory_gb(4, 1, T_LONG, 2,
                                  weights_function=as_array_function(tc.gaussian_weights))
    assert dense[1] > ONE_KERNEL / 1e9


# ------------------------------------------------------------ memory linearity


def _timecorr_peak(T):
    rng = np.random.RandomState(6)
    data = [rng.randn(T, 4) for _ in range(2)]
    return traced_peak(lambda: tc.timecorr(data, cfun=tc.isfc))[0]


def test_peak_memory_grows_linearly_in_T():
    # T 5,000 / 10,000 / 20,000 with the default Gaussian kernel: the peak is
    # a fixed workspace plus a part proportional to T, so the increase from
    # 10,000 to 20,000 is twice the increase from 5,000 to 10,000 (a T x T
    # kernel would make it four times; measured 2.0)
    peaks = [_timecorr_peak(T) for T in (5_000, 10_000, 20_000)]
    ratio = (peaks[2] - peaks[1]) / (peaks[1] - peaks[0])
    assert 1.6 < ratio < 2.4, peaks


# ------------------------------------------- estimates against traced peaks


@pytest.mark.parametrize("T", [3_000, 20_000, 50_000])
def test_step_estimate_is_not_below_the_traced_peak(T):
    K, S = 10, 2
    rng = np.random.RandomState(7)
    data = [rng.randn(T, K) for _ in range(S)]
    est = ho.estimate_memory_gb(K, 1, T, S)[1] * 1e9
    peak, out = traced_peak(lambda: run(
        tc.features, data, order=1, cfun=tc.autofc, rfun=None, delta_rfun=None,
        weights_params={"var": 100},
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak <= est, (peak, est)


CFUNS = [tc.isfc, tc.wisfc, tc.autofc]
CFUN_IDS = ["isfc", "wisfc", "autofc"]


def _materialized_peak_and_estimate(K, T, S, cfun):
    rng = np.random.RandomState(7)
    data = [rng.randn(T, K) for _ in range(S)]
    est = ho.estimate_memory_gb(K, 1, T, S)[1] * 1e9
    peak, out = traced_peak(lambda: run(
        tc.features, data, order=1, cfun=cfun, rfun=None, delta_rfun=None,
        max_memory_gb=None,
    ))
    assert not isinstance(out[0], str), out[0]
    return peak, est


@pytest.mark.parametrize("cfun", CFUNS, ids=CFUN_IDS)
def test_step_estimate_covers_several_subjects_and_cross_subject_cfuns(cfun):
    # K 60, T 2,500, 4 subjects: the estimate of commit 2a64d24 (6 block
    # temporaries, neither the stacked subjects nor the reference of isfc /
    # wisfc) was below the traced peak (by 3% for isfc, 2% for autofc)
    peak, est = _materialized_peak_and_estimate(60, 2_500, 4, cfun)
    assert peak <= est, (peak, est)


@pytest.mark.slow
@pytest.mark.parametrize("cfun", CFUNS, ids=CFUN_IDS)
@pytest.mark.parametrize("K, T, S", [(60, 5_000, 6), (30, 10_000, 12)])
def test_step_estimate_covers_large_cross_subject_steps(K, T, S, cfun):
    peak, est = _materialized_peak_and_estimate(K, T, S, cfun)
    assert peak <= est, (peak, est)


@pytest.mark.parametrize("factor", [1.0, 3.0])
@pytest.mark.parametrize("T", [3_000, 20_000])
@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc], ids=["autofc", "isfc"])
def test_streamed_step_stays_within_the_budget_it_was_given(cfun, T, factor):
    # through the guard of features: the block size is chosen from
    # max_memory_gb, and the traced peak stays within it
    K, S = 10, 3
    rng = np.random.RandomState(8)
    data = [rng.randn(T, K) for _ in range(S)]
    data[0][rng.rand(T, K) < 0.02] = np.nan
    budget = ho.estimate_memory_gb(K, 1, T, S, streamed=True)[1] * factor
    peak, out = traced_peak(lambda: run(
        tc.features, data, order=1, cfun=cfun, rfun=tc.WeightedStrength(signed=True),
        delta_rfun=None, max_memory_gb=budget,
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak <= budget * 1e9, (peak, budget)


@pytest.mark.parametrize("T", [3_000, 20_000, 50_000])
@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc], ids=["autofc", "isfc"])
def test_streamed_estimate_is_not_below_the_traced_peak(T, cfun):
    K, S = 10, 2
    rng = np.random.RandomState(8)
    data = [rng.randn(T, K) for _ in range(S)]
    data[0][rng.rand(T, K) < 0.02] = np.nan  # the estimator's costlier path
    kernels = ho._step_kernels(T, S, tc.gaussian_weights, {"var": 100})
    est = 8 * sm._streamed_peak_elements(T, K, S, 1)
    peak, _ = traced_peak(lambda: sm._streamed_summary(
        data, kernels, tc.WeightedStrength(signed=True), cfun,
        max_block_elements=K * T,
    ))
    assert peak <= est, (peak, est)


@pytest.mark.slow
def test_heavy_tailed_kernel_at_20000_timepoints_stays_within_its_estimate():
    # the uniform kernel has weight everywhere: every chunk spans all T
    # columns, so the time is quadratic, but the memory stays linear
    T, K, S = 20_000, 3, 2
    rng = np.random.RandomState(9)
    data = [rng.randn(T, K) for _ in range(S)]
    est = ho.estimate_memory_gb(K, 1, T, S, weights_function=tc.uniform_weights)[1]
    peak, out = traced_peak(lambda: run(
        tc.features, data, order=1, cfun=tc.autofc, rfun=None, delta_rfun=None,
        weights_function=tc.uniform_weights,
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak <= est * 1e9, (peak, est)
    assert peak < T * T * 8 / 20
