"""
The timepoints at which the shift null of the automatic number of communities
is evaluated (``summaries._shift_null_threshold``).

The surrogates' squared correlations are averaged over every ``s``-th
timepoint, where ``s`` is the largest step at which two rows of the kernel
still overlap by ``exp(-1/4)`` (less a margin of 1e-6 for round-off; the
standard deviation, rounded down, for a Gaussian).  ``features`` and
``two_path_step`` do this for built-in kernels without events, one-sided
restriction or ``exclude_timepoints`` (``summaries._stationary``); for any
other kernel (arrays, functions of your own, restricted kernels), whose rows
can differ, the null is evaluated at every timepoint.

Everything is computed for real on simulated data; nothing is mocked.
"""

import warnings

import numpy as np
import pytest

import timecorr as tc
import timecorr.higher_order as ho
import timecorr.summaries as sm


def lag_kernel(T, function=tc.gaussian_weights, params=None, **kwargs):
    return ho._step_kernels(T, 1, function, params, **kwargs)


def overlap_stride(row, level=np.exp(-0.25) * (1 - 1e-6)):
    """Independent computation: the largest step d such that the row shifted
    by every step up to d keeps a normalized overlap of at least level."""
    stride = 1
    for d in range(1, len(row)):
        if row[d:] @ row[:-d] / (row @ row) < level:
            break
        stride = d
    return stride


@pytest.mark.parametrize("var, expected", [(1, 1), (9, 3), (25, 5), (100, 10), (30.25, 5)])
def test_stride_of_a_gaussian_kernel_is_its_standard_deviation(var, expected):
    assert sm._null_row_stride(lag_kernel(600, params={"var": var})) == expected


def test_stride_of_other_kernels():
    T = 400
    assert sm._null_row_stride(lag_kernel(T, tc.eye_weights)) == 1
    for function, params in ((tc.laplace_weights, {"scale": 20}),
                             (tc.boxcar_weights, {"width": 30}),
                             (tc.mexican_hat_weights, {"sigma": 12}),
                             (tc.t_weights, {"df": 3})):
        row = function(T, params)[T // 2]
        assert sm._null_row_stride(lag_kernel(T, function, params)) == overlap_stride(row)
    # one-sided kernels: the row as restricted
    row = tc.gaussian_weights(T, {"var": 100})[T // 2].copy()
    row[T // 2 + 1:] = 0
    backward = lag_kernel(T, params={"var": 100}, include_timepoints="backward")
    assert sm._null_row_stride(backward) == overlap_stride(row)
    assert 1 < sm._null_row_stride(backward) < 10
    # events: the middle row as the events restrict it (here timepoint 200,
    # 50 timepoints after the boundary: the full row to within round-off)
    with_events = ho._step_kernels(T, 1, tc.gaussian_weights, {"var": 100}, events=[150])
    assert sm._null_row_stride(with_events) == 10
    at_boundary = ho._step_kernels(T, 1, tc.gaussian_weights, {"var": 100}, events=[200])
    dense = tc.gaussian_weights(T, {"var": 100})[T // 2].copy()
    dense[:T // 2] = 0
    assert sm._null_row_stride(at_boundary) == overlap_stride(dense[T // 2:])
    assert sm._null_row_stride(at_boundary) < 10


def test_stride_of_an_array_kernel_is_that_of_the_same_built_in_kernel():
    assert sm._null_row_stride(tc.gaussian_weights(200, {"var": 25})) == 5
    assert sm._null_row_stride(tc.gaussian_weights(200, {"var": 100})) == 10
    assert sm._null_row_stride(np.eye(50)) == 1
    assert sm._null_row_stride(tc.laplace_weights(300, {"scale": 20})) == sm._null_row_stride(
        lag_kernel(300, tc.laplace_weights, {"scale": 20}))


def test_a_kernel_function_of_your_own_gets_the_null_at_every_timepoint():
    # changed with the second outside review (controller ruling): the rows of
    # a kernel function of your own can differ, so its null is evaluated at
    # every timepoint; the same kernel built in is evaluated at every 5th.
    # This test asserted before that both gave the threshold at every 5th
    # timepoint.
    def own(T, params=None):
        return tc.gaussian_weights(T, {"var": 25})

    rng = np.random.default_rng(5)
    data = [rng.standard_normal((300, 8)) for _ in range(2)]
    recorded = []
    for kernel in (dict(), dict(delta_weights_function=own)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            (_, info) = tc.features(data, order=1, cfun=tc.autofc, return_delta=True,
                                    return_info=True, weights_params={"var": 100},
                                    **kernel)
        recorded.append([e for e in info if e["path"] == "feeding"][0]["null_threshold"])
    dense = tc.gaussian_weights(300, {"var": 25})
    # built in: the null at every 5th timepoint; your own: at every timepoint
    assert recorded[0] == pytest.approx(
        reference_threshold(data, dense, 5, 5.0, np.arange(2, 300, 5)), rel=1e-9)
    assert recorded[1] == pytest.approx(
        reference_threshold(data, dense, 5, 5.0, np.arange(300)), rel=1e-9)
    assert recorded[0] != recorded[1]


def reference_threshold(data, kernel, n_null, bandwidth, rows):
    """The shift null computed from full correlation arrays, averaged over
    the timepoints ``rows`` (lags drawn as the package draws them)."""
    T, K = data[0].shape
    L = int(min(T // 4, max(10, 10 * bandwidth)))
    rng = np.random.default_rng(0)
    idx = np.arange(T)[:, None]
    cols = np.arange(K)
    tops = []
    for _ in range(n_null):
        shifted = []
        for x in data:
            lags = L + rng.choice(T - 2 * L + 1, size=K, replace=False)
            shifted.append(x[(idx - lags[None, :]) % T, cols])
        weights = []
        for v in tc.autofc(shifted, kernel):
            M = tc.vec2mat(v)[:, :, rows]
            w = (M ** 2).mean(axis=2)
            np.fill_diagonal(w, 0)
            weights.append(w)
        wbar = np.mean(weights, axis=0)
        C = wbar - np.median(wbar[np.triu_indices(K, 1)])
        np.fill_diagonal(C, 0)
        tops.append(np.linalg.eigvalsh(C)[-1])
    return float(max(tops))


@pytest.mark.parametrize("var, nan", [(25, False), (49, False), (25, True)])
def test_threshold_is_the_null_at_every_sth_timepoint(var, nan):
    T, K = 600, 12
    rng = np.random.default_rng(3)
    data = [rng.standard_normal((T, K)) for _ in range(2)]
    if nan:
        data[0][100:110, 2] = np.nan
    sd = float(np.sqrt(var))
    stride = int(sd)
    dense = tc.gaussian_weights(T, {"var": var})
    rows = np.arange(stride // 2, T, stride)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        got = sm._shift_null_threshold(data, lag_kernel(T, params={"var": var}),
                                       n_null=3, bandwidth=sd)
        full = sm._shift_null_threshold(data, dense, n_null=3, bandwidth=sd)
        assert got == pytest.approx(reference_threshold(data, dense, 3, sd, rows), rel=1e-9)
        # the array kernel: every timepoint
        assert full == pytest.approx(reference_threshold(data, dense, 3, sd, np.arange(T)),
                                     rel=1e-9)
    # the two differ, by little
    assert got != full
    assert got == pytest.approx(full, rel=0.1)


def test_thresholds_agree_on_average_over_draws():
    # 20 draws of noise (K 24, T 300, sd 5): the threshold at every 5th
    # timepoint is the threshold at every timepoint within 2 % on average
    ratios = []
    lag = lag_kernel(300, params={"var": 25})
    dense = tc.gaussian_weights(300, {"var": 25})
    for draw in range(20):
        rng = np.random.default_rng([7, draw])
        data = [rng.standard_normal((300, 24)) for _ in range(2)]
        ratios.append(sm._shift_null_threshold(data, lag) / sm._shift_null_threshold(data, dense))
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.02)
    assert np.max(np.abs(np.array(ratios) - 1)) < 0.15


def test_features_record_the_threshold_of_their_kernel():
    data, _ = tc.simulate_nested(S=2, T=400, n_supergroups=2, modules_per_supergroup=3,
                                 nodes_per_module=4, set_random_seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (_, info) = tc.features(data, order=1, cfun=tc.autofc, return_delta=True,
                                return_info=True)
    feeding = [e for e in info if e["path"] == "feeding"][0]
    kernels = ho._step_kernels(400, 2, tc.gaussian_weights, {"var": 25})
    expected = sm._shift_null_threshold(data, kernels, bandwidth=5.0)
    assert feeding["null_threshold"] == pytest.approx(expected, rel=1e-12)
    assert feeding["n_communities"] == 6


# second outside review: rows built, heterogeneous kernels, when the step is computed


def test_every_nth_row_computes_only_the_rows_it_keeps():
    # a chunk of 100 kept rows of a (10000, 10000) kernel at stride 100 is a
    # (100, 10000) block (8 MB); the rows between the kept ones (0.8 GB) are
    # not built
    import tracemalloc
    k = ho._step_kernels(10000, 1, tc.gaussian_weights, {"var": 10000})
    sub = sm._every_nth_row(k, 100)
    i, j, c0, c1, _ = next(sub.chunks(128))
    tracemalloc.start()
    block = sub.block(i, j, c0, c1)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert block.shape == (100, 10000)
    assert peak < 1.5 * block.nbytes
    rows = np.arange(50, 10000, 100)
    np.testing.assert_array_equal(block, k.dense()[rows][:, c0:c1])


def _heterogeneous(kind, phase=0):
    def kernel(T, params=None):
        w = tc.gaussian_weights(T, {"var": 25})
        rows = np.arange(2 + phase, T, 5)
        w[rows] = 0
        if kind == "singleton":
            w[rows, rows] = 1.0
        return w
    return kernel


def _recorded_null(data, kernel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, info = tc.features(data, order=1, cfun=tc.autofc, return_delta=True,
                              return_info=True, rfun="weighted_strength",
                              delta_weights_function=kernel)
    return [e for e in info if e["path"] == "feeding"][0]["null_threshold"]


def test_a_kernel_with_zero_rows_gets_the_null_at_every_timepoint():
    # every 5th row is zero: sampling every 5th row could keep only zero
    # rows ("Every row of the kernel is zero"); the null of such a kernel is
    # evaluated at every timepoint
    x = np.random.default_rng(1).normal(size=(300, 8))
    kernel = _heterogeneous("zero")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tc.two_path_step(x, cfun=tc.autofc, rfun="weighted_strength",
                         delta_weights_function=kernel)
    got = _recorded_null([x], kernel)
    # bandwidth: the function has no "var" default, so the default 5.0
    assert got == pytest.approx(reference_threshold([x], kernel(300), 5, 5.0,
                                                    np.arange(300)), rel=1e-9)


@pytest.mark.parametrize("phase", [0, 1])
def test_a_kernel_with_single_weight_rows_keeps_its_estimator(phase):
    # every 5th row has one weight: in the full kernel those rows are
    # undefined (0); a sample of only such rows would switch the estimator to
    # co-fluctuations.  The null is the full one, whatever the rows' phase
    rng = np.random.default_rng(4)
    data = [rng.standard_normal((300, 12)) for _ in range(2)]
    kernel = _heterogeneous("singleton", phase)
    got = _recorded_null(data, kernel)
    full = reference_threshold(data, kernel(300), 5, 5.0, np.arange(300))
    assert got == pytest.approx(full, rel=1e-9)
    assert got < 0.5  # 2.19 when every 5th row (the single-weight rows) was sampled


@pytest.mark.parametrize("kw", [dict(include_timepoints="backward"),
                                dict(events=[150])])
def test_restricted_built_in_kernels_get_the_null_at_every_timepoint(kw):
    T = 300
    rng = np.random.default_rng(6)
    data = [rng.standard_normal((T, 8)) for _ in range(2)]
    k = ho._step_kernels(T, 2, tc.gaussian_weights, {"var": 25}, **kw)
    assert not sm._stationary(k)
    dense = k.dense()
    assert sm._shift_null_threshold(data, k) == pytest.approx(
        reference_threshold(data, dense, 5, 5.0, np.arange(T)), rel=1e-9)


def test_stationary_built_in_kernels_still_sample_every_sth_timepoint():
    T, var = 600, 25
    rng = np.random.default_rng(3)
    data = [rng.standard_normal((T, 12)) for _ in range(2)]
    k = lag_kernel(T, params={"var": var})
    assert sm._stationary(k) and sm._null_stride([k]) == 5
    rows = np.arange(2, T, 5)
    assert sm._shift_null_threshold(data, k, n_null=3) == pytest.approx(
        reference_threshold(data, tc.gaussian_weights(T, {"var": var}), 3, 5.0, rows),
        rel=1e-9)


def test_the_step_is_computed_only_when_the_null_runs():
    # a streamed WeightedStrength or a fixed number of communities runs no
    # shift null, so no step is computed (it is a search over the kernel row)
    calls = []
    original = sm._null_row_stride

    def counted(kernel):
        calls.append(kernel)
        return original(kernel)

    sm._null_row_stride = counted
    had = hasattr(ho, "_null_row_stride")
    if had:
        ho._null_row_stride = counted
    try:
        rng = np.random.default_rng(2)
        x = [rng.standard_normal((300, 6)) for _ in range(2)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for delta_rfun in (tc.WeightedStrength(signed=True), tc.Communities(n=2)):
                tc.features(x, order=1, cfun=tc.autofc, rfun="weighted_strength",
                            return_delta=True, delta_rfun=delta_rfun)
            assert calls == []
            tc.features(x, order=1, cfun=tc.autofc, rfun="weighted_strength",
                        return_delta=True, delta_rfun=tc.Communities())
        assert len(calls) >= 1
    finally:
        sm._null_row_stride = original
        if had:
            ho._null_row_stride = original


def test_the_step_of_a_uniform_kernel_is_cheap():
    # the overlaps come from one FFT of the row: at T 1,000,000 the uniform
    # kernel's step (T // 8) takes under 2 seconds (the loop of dot
    # products took O(T**2) operations)
    import time
    k = lag_kernel(1_000_000, tc.uniform_weights)
    start = time.perf_counter()
    assert sm._null_row_stride(k) == 1_000_000 // 8
    assert time.perf_counter() - start < 2.0
