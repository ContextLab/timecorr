"""
The timepoints at which the shift null of the automatic number of communities
is evaluated (``summaries._shift_null_threshold``).

The surrogates' squared correlations are averaged over every ``s``-th
timepoint, where ``s`` is the largest step at which two rows of the kernel
still overlap by ``exp(-1/4)`` (less a margin of 1e-6 for round-off; the
standard deviation, rounded down, for a Gaussian).  ``features`` and
``two_path_step`` do this for every kernel, built in or a function of your
own; ``_shift_null_threshold`` called with arrays and no ``row_stride`` uses
every timepoint.

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


def test_a_kernel_function_of_your_own_gives_the_threshold_of_the_built_in_one():
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
    assert recorded[0] == pytest.approx(recorded[1], rel=1e-12)
    # and it is the null at every 5th timepoint, also for the array
    dense = tc.gaussian_weights(300, {"var": 25})
    assert recorded[1] == pytest.approx(
        sm._shift_null_threshold(data, dense, bandwidth=5.0, row_stride=5), rel=1e-12)
    assert recorded[1] != sm._shift_null_threshold(data, dense, bandwidth=5.0)


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
