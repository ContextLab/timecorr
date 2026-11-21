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
    """``f(*args, **kwargs)`` and its warning texts; an exception propagates
    (a valid case must succeed: two equal crashes are not an equality)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*args, **kwargs)
    return out, [str(w.message) for w in caught]


def zero_kernel(T, fn, params, options):
    """True when the options leave the kernel without any weight (for
    example the delta kernel without its diagonal): an invalid input, which
    ``timecorr`` refuses."""
    shape = {k: v for k, v in options.items() if k != "events"}
    return not np.any(_shaped_kernel(T, fn, params, **shape).dense())


def run_invalid(f, *args, **kwargs):
    """The exception of ``f(*args, **kwargs)`` as text, for an input that is
    invalid on purpose (both kernel forms must refuse it alike)."""
    with pytest.raises(Exception) as err:
        f(*args, **kwargs)
    return f"{err.type.__name__}: {err.value}"


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
            if zero_kernel(400, fn, params, options):
                assert (run_invalid(tc.timecorr, data, weights_function=fn, **kw)
                        == run_invalid(tc.timecorr, data,
                                       weights_function=as_array_function(fn), **kw))
                continue
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
    if np.isnan(value):
        # a NaN sample gets no weight, so both forms sum finite terms over
        # each chunk's span: equal up to the order of summation (differences
        # of 6e-17 were seen with NumPy 2.5 on Linux), as in
        # test_smoothing_equals_the_array_kernel
        np.testing.assert_array_equal(np.isfinite(lazy), np.isfinite(dense))
        np.testing.assert_allclose(lazy, dense, rtol=1e-12, atol=1e-14)
    else:
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
    if exclude <= -400:
        # every timepoint removed: refused alike by both kernel forms
        assert (run_invalid(tc.timecorr, subjects, weights_function=tc.gaussian_weights,
                            **kw)
                == run_invalid(tc.timecorr, subjects,
                               weights_function=as_array_function(tc.gaussian_weights),
                               **kw))
        return
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


@pytest.mark.parametrize("K, T", [(150, 200), (100, 200), (100, 300)])
def test_streamed_wisfc_stays_within_the_budget_it_was_given(K, T):
    # wisfc's default subject weights correlate the subjects' static
    # connectomes; those correlations must be computed within the budget too
    # (before the fix they used the estimator's default blocks: up to 1.84
    # times the budget at K 150, T 200, 3 subjects)
    S = 3
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((T, K)) for _ in range(S)]
    budget = ho.estimate_memory_gb(K, 1, T, S, streamed=True)[1]
    peak, out = traced_peak(lambda: run(
        tc.features, data, order=1, cfun=tc.wisfc, rfun=tc.WeightedStrength(signed=True),
        delta_rfun=None, max_memory_gb=budget, weights_params={"var": 100},
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak <= budget * 1e9, (peak, budget)


@pytest.mark.parametrize("combine", [tc.corrmean_combine, tc.mean_combine, tc.tstat_combine],
                         ids=["corrmean", "mean", "tstat"])
@pytest.mark.parametrize("K, T, S", [(60, 300, 6), (100, 200, 6), (80, 400, 5)])
def test_step_estimate_counts_the_builtin_combine_functions(combine, K, T, S):
    # the combine function runs on every subject's vectorized correlations;
    # its temporaries are counted in the estimate and the guard (before,
    # corrmean_combine traced 2.0 times the estimate at K 300, T 300, 3
    # subjects)
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((T, K)) for _ in range(S)]
    budget = ho.estimate_memory_gb(K, 1, T, S, combine=combine)[1]
    peak, out = traced_peak(lambda: run(
        tc.features, data, order=1, cfun=tc.isfc, combine=combine, rfun=None,
        delta_rfun=None, max_memory_gb=budget, weights_params={"var": 100},
    ))
    assert not isinstance(out[0], str), out[0]
    assert peak <= budget * 1e9, (peak, budget)
    # the guard uses the same estimate
    with pytest.raises(MemoryError, match=f"need about {budget:.3g} GB"):
        tc.features(data, order=1, cfun=tc.isfc, combine=combine, rfun=None,
                    delta_rfun=None, max_memory_gb=0.99 * budget,
                    weights_params={"var": 100})


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


# --------------------------------------- kernel classification of the guards


def test_a_custom_cfun_counts_its_t_by_t_kernels():
    # timecorr passes a correlation function of the user's T x T kernel
    # arrays (only autofc, isfc and wisfc read a built-in kernel in chunks):
    # the guard must count them and refuse before anything is computed
    x = np.random.default_rng(0).normal(size=(10_000, 2))
    seen = []

    def wrapped(data, weights):
        seen.append(weights.nbytes)
        return tc.autofc(data, weights)

    with pytest.raises(MemoryError, match="max_memory_gb=0.1"):
        tc.features(x, order=1, rfun=None, delta_rfun=None, cfun=wrapped,
                    max_memory_gb=0.1)
    assert seen == []
    # it does receive a T x T array
    tc.timecorr(x[:2_000], cfun=wrapped)
    assert seen == [2_000 * 2_000 * 8]
    est = ho.estimate_memory_gb(2, 1, 10_000, cfun=wrapped)[1]
    assert est > 3 * 10_000 ** 2 * 8 / 1e9 > ho.estimate_memory_gb(2, 1, 10_000)[1]


def test_an_unused_feeding_kernel_is_not_counted():
    # order 1 without return_delta runs no feeding path: a feeding kernel
    # function of the user's must not count (the review measured a refusal
    # at 2.4 GB for a run that needs 0.078 GB)
    x = np.random.default_rng(1).normal(size=(10_000, 2))
    custom = as_array_function(tc.gaussian_weights)
    est = ho.estimate_memory_gb(2, 1, 10_000, delta_weights_function=custom)
    assert est == ho.estimate_memory_gb(2, 1, 10_000)
    out = tc.features(x, order=1, rfun=None, delta_rfun=None,
                      delta_weights_function=custom, delta_weights_params={"var": 25},
                      max_memory_gb=0.2)
    assert out[1].shape == (10_000, 3)
    # with return_delta the feeding path runs and its T x T arrays count
    assert (ho.estimate_memory_gb(2, 1, 10_000, return_delta=True,
                                  delta_weights_function=custom)[1]
            > 3 * 10_000 ** 2 * 8 / 1e9)
    with pytest.raises(MemoryError):
        tc.features(x, order=1, rfun=None, delta_rfun=None, return_delta=True,
                    delta_weights_function=custom, delta_weights_params={"var": 25},
                    max_memory_gb=0.2)


def test_a_streamed_path_is_not_charged_the_other_paths_event_kernels():
    # per-subject events with a feeding kernel of the user's: the streamed
    # analysis step reads a built-in kernel and holds no T x T arrays, so a
    # refusal comes from the feeding step, whose kernels are T x T arrays
    T = 3_000
    rng = np.random.default_rng(2)
    data = [rng.normal(size=(T, 6)) for _ in range(3)]
    custom = as_array_function(tc.gaussian_weights)
    with pytest.raises(MemoryError, match="streamed feeding step"):
        tc.two_path_step(data, cfun=tc.autofc, rfun=tc.WeightedStrength(signed=True),
                         delta_rfun=tc.WeightedStrength(signed=True),
                         events=[[1000], [1500], [2000]],
                         delta_weights_function=custom,
                         delta_weights_params={"var": 25}, max_memory_gb=0.15)


# ----------------------------------------------- kernel objects and options


@pytest.mark.parametrize("fn", [tc.uniform_weights, tc.eye_weights],
                         ids=["uniform", "eye"])
def test_dense_uniform_and_eye_kernels_allocate_only_their_output(fn):
    # the T x T output is built directly, without a T x T matrix of lags first
    T = 2_000
    out_bytes = T * T * 8
    peak, w = traced_peak(lambda: fn(T))
    assert w.shape == (T, T) and w.dtype == np.float64
    assert peak < 1.1 * out_bytes
    np.testing.assert_array_equal(w, np.eye(T) if fn is tc.eye_weights
                                  else np.ones((T, T)))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan], ids=["inf", "-inf", "nan"])
@pytest.mark.parametrize("builtin", [True, False], ids=["builtin", "array"])
def test_non_finite_exclude_timepoints_is_rejected_for_every_kernel(value, builtin,
                                                                   subjects):
    fn = tc.gaussian_weights if builtin else as_array_function(tc.gaussian_weights)
    with pytest.raises(ValueError, match="exclude_timepoints must be a finite number"):
        tc.timecorr(subjects, weights_function=fn, weights_params={"var": 4},
                    cfun=tc.autofc, exclude_timepoints=value)


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_lag_kernel_rejects_a_non_finite_exclude_when_it_is_made(value):
    with pytest.raises(ValueError, match="exclude_timepoints must be a finite number"):
        LagKernel(np.ones(5), exclude=value)


def test_lag_kernel_with_events_none_keeps_its_events():
    k = LagKernel(np.ones(6), labels=[0, 0, 0, 1, 1, 1])
    kept = k.with_events(None)
    np.testing.assert_array_equal(kept.dense(), k.dense())
    assert kept.dense()[0, 5] == 0
    assert LagKernel(np.ones(6)).with_events(None).labels is None


def test_lag_kernel_labels_are_runs_of_consecutive_values():
    # a label that returns after another starts a new block (the runs that
    # span(), key() and the boundaries of block_labels assume)
    k = LagKernel(np.ones(4), labels=[0, 1, 1, 0])
    expect = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1.0]])
    np.testing.assert_array_equal(k.dense(), expect)
    for t0, t1 in [(0, 4), (1, 3), (3, 4)]:
        c0, c1 = k.span(t0, t1)
        np.testing.assert_array_equal(k.block(t0, t1, 0, 4)[:, :c0], 0)
        np.testing.assert_array_equal(k.block(t0, t1, 0, 4)[:, c1:], 0)
    # labels of a single block are no restriction, and the same kernel
    assert LagKernel(np.ones(4), labels=[0, 0, 0, 0]).same_as(LagKernel(np.ones(4)))
    assert LagKernel(np.ones(4), labels=[2, 2, 5, 5]).same_as(
        LagKernel(np.ones(4), labels=[0, 0, 1, 1]))
    with pytest.raises(ValueError, match="length T=4"):
        LagKernel(np.ones(4), labels=[0, 1])


def test_lag_kernel_with_no_timepoints():
    k = LagKernel(np.ones(0), labels=np.zeros(0, dtype=int))
    assert k.dense().shape == (0, 0)
    kept = k.with_events(np.zeros(0, dtype=int))
    assert kept.dense().shape == (0, 0)
    assert kept.labels is None or kept.labels.shape == (0,)


def test_profiles_square_int32_lags_without_overflow():
    # int32 d**2 overflows from d = 46,341
    from timecorr import _kernels as K
    d = np.array([0, 46_341, 60_000], dtype=np.int32)
    params = {"var": 1e10, "df": 3, "sigma": 1e5, "scale": 1e5}
    for profile in (K.gaussian_profile, K.t_profile, K.mexican_hat_profile,
                    K.laplace_profile):
        np.testing.assert_array_equal(profile(d, params),
                                      profile(d.astype(np.int64), params))
    assert K.lags(3).dtype == np.int64


def test_kernel_rows_compute_each_block_once_and_count_it_once():
    # the blocks _time_slices reads while preparing are kept for the
    # products, and a block cached under two keys is counted once
    from timecorr import helpers as H
    T = 1_000
    k = _shaped_kernel(T, tc.gaussian_weights, {"var": 25})
    calls = []
    block = k.block

    def counted(t0, t1, c0, c1):
        calls.append((t0, t1, c0, c1))
        return block(t0, t1, c0, c1)

    k.block = counted
    rows = H._KernelRows(k)
    before = len(calls)
    rows.zero_rows(rows.support < 3)
    rows.prepare(True)
    x = np.random.default_rng(0).normal(size=(T, 3))
    out = {kind: (rows.product(x, kind), rows.row_sum(kind)) for kind in "was"}
    computed = calls[before:]
    assert len(computed) == len(set(computed))
    distinct = {id(v): v.size for v in rows._cache.values()}
    assert rows._cached == sum(distinct.values())
    # the same values as the T x T array
    dense = H._KernelRows(k.dense())
    dense.zero_rows(dense.support < 3)
    dense.prepare(True)
    for kind in "was":
        np.testing.assert_array_equal(out[kind][0], dense.product(x, kind))
        np.testing.assert_array_equal(out[kind][1], dense.row_sum(kind))


def test_kernel_rows_cache_keys_include_the_zeroed_rows():
    # interior chunks share a key; rows zeroed in one of them only must not
    # reach the other through a cached block
    from timecorr import helpers as H
    T = 1_000
    k = _shaped_kernel(T, tc.gaussian_weights, {"var": 25})
    rows = H._KernelRows(k)
    mask = np.zeros(T, dtype=bool)
    mask[300:310] = True
    rows.zero_rows(mask)
    rows.prepare(True)
    w = k.dense()
    w[mask] = 0
    x = np.random.default_rng(1).normal(size=(T, 2))
    for kind, m in (("w", w), ("a", np.abs(w)), ("s", (w != 0) * 1.0)):
        np.testing.assert_allclose(rows.product(x, kind), m @ x, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(rows.row_sum(kind), m.sum(axis=1), rtol=1e-12)


def _messages(f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = f()
    return out, [str(x.message) for x in w]


@pytest.mark.parametrize("case", ["constant", "sparse"])
def test_streamed_static_connectome_warns_as_wcorr_does(case):
    # the wisfc subject weights computed in blocks (streamed summaries) give
    # the warnings of the one wcorr call they replace
    from timecorr import helpers as H
    rng = np.random.default_rng(5)
    d = rng.normal(size=(50, 6))
    if case == "constant":
        d[:] = 3.0
    else:
        d[2:, 4] = np.nan  # two observed samples
    iu = np.triu_indices(6, 1)
    whole, expected = _messages(lambda: H._static_connectome(d, iu, None))
    assert expected
    for b in (1, 2, 4, 6):
        blocked, got = _messages(lambda: H._static_connectome(d, iu, b))
        assert got == expected
        np.testing.assert_allclose(blocked, whole, rtol=1e-12, atol=1e-15)
