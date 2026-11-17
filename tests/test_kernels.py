import numpy as np
import pytest

import timecorr as tc
from timecorr.helpers import (
    boxcar_weights,
    eye_weights,
    gaussian_weights,
    laplace_weights,
    mexican_hat_weights,
    t_weights,
    uniform_weights,
)

KERNELS = [
    gaussian_weights,
    laplace_weights,
    eye_weights,
    uniform_weights,
    t_weights,
    mexican_hat_weights,
    boxcar_weights,
]
T = 7


@pytest.mark.parametrize("kernel", KERNELS)
def test_backward_kernel(kernel):
    full = kernel(T)
    back = kernel(T, direction="backward")
    future = np.triu(np.ones([T, T]), k=1).astype(bool)
    assert np.all(back[future] == 0)
    assert np.array_equal(back[~future], full[~future])


@pytest.mark.parametrize("kernel", KERNELS)
def test_forward_kernel_mirrors_backward(kernel):
    full = kernel(T)
    fwd = kernel(T, direction="forward")
    past = np.tril(np.ones([T, T]), k=-1).astype(bool)
    assert np.all(fwd[past] == 0)
    assert np.array_equal(fwd[~past], full[~past])


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("direction", [None, "backward", "forward"])
def test_include_zero_false_zeroes_diagonal(kernel, direction):
    w = kernel(T, direction=direction, include_zero=False)
    ref = kernel(T, direction=direction)
    assert np.all(np.diag(w) == 0)
    off = ~np.eye(T, dtype=bool)
    assert np.array_equal(w[off], ref[off])


@pytest.mark.parametrize("kernel", KERNELS)
def test_bad_direction_raises(kernel):
    with pytest.raises(ValueError):
        kernel(T, direction="sideways")


def test_one_sided_directly():
    w = np.arange(1.0, 17.0).reshape(4, 4)
    assert np.array_equal(tc.one_sided(w, "backward"), np.tril(w))
    assert np.array_equal(tc.one_sided(w, "forward"), np.triu(w))
    assert np.array_equal(
        tc.one_sided(w, "backward", include_zero=False), np.tril(w, -1)
    )
    assert np.array_equal(tc.one_sided(w, "forward", include_zero=False), np.triu(w, 1))
    with pytest.raises(ValueError):
        tc.one_sided(w, "pre")


def _flip_data():
    rng = np.random.RandomState(0)
    n = 200
    x = rng.randn(n)
    noise = rng.randn(n) * 0.3
    y = np.where(np.arange(n) < 100, x, -x) + noise
    return np.stack([x, y], axis=1)


def _xy_at(data, t, include):
    # the first (pre) or last (post) timepoint has a single-sample kernel
    with pytest.warns(UserWarning, match="fewer than 3 non-zero kernel weights"):
        out = np.asarray(
            tc.timecorr(
                data,
                weights_function=gaussian_weights,
                weights_params={"var": 2000},
                include_timepoints=include,
            )
        )
    return out[t, 2]  # columns: vec of 2x2 corr matrix; index 2 is the x-y entry


def test_pre_post_sign_flip_behavior():
    data = _flip_data()
    assert _xy_at(data, 101, "pre") > 0
    assert _xy_at(data, 98, "post") < 0


class _Recorder:
    """A real cfun that records the weights it receives and computes isfc."""

    def __init__(self):
        self.weights = None

    def __call__(self, data, weights):
        self.weights = np.array(weights, copy=True)
        return tc.isfc(data, weights)


@pytest.mark.parametrize("flag", ["pre", "backward"])
def test_timecorr_backward_weights_passed_to_cfun(flag):
    rec = _Recorder()
    with pytest.warns(UserWarning, match="fewer than 3 non-zero kernel weights"):
        tc.timecorr(
            _flip_data(), weights_params={"var": 50}, include_timepoints=flag, cfun=rec
        )
    w = rec.weights
    full = gaussian_weights(200, {"var": 50})
    assert np.all(w[np.triu_indices(200, k=1)] == 0)
    low = np.tril_indices(200)
    assert np.array_equal(w[low], full[low])


@pytest.mark.parametrize("flag", ["post", "forward"])
def test_timecorr_forward_weights_passed_to_cfun(flag):
    rec = _Recorder()
    with pytest.warns(UserWarning, match="fewer than 3 non-zero kernel weights"):
        tc.timecorr(
            _flip_data(), weights_params={"var": 50}, include_timepoints=flag, cfun=rec
        )
    w = rec.weights
    full = gaussian_weights(200, {"var": 50})
    assert np.all(w[np.tril_indices(200, k=-1)] == 0)
    up = np.triu_indices(200)
    assert np.array_equal(w[up], full[up])


def test_timecorr_include_zero_false_and_exclude_compose():
    rec = _Recorder()
    # backward without zero keeps tau < t; exclude_timepoints=-3 also drops lags
    # 1-3, so row t has max(0, t - 3) non-zero weights (tau = 0..t-4).  Rows
    # 0-5 have fewer than 3 (R12), so 6 timepoints are undefined.
    with pytest.warns(UserWarning, match="6 of 200 timepoints"):
        tc.timecorr(
            _flip_data(),
            weights_params={"var": 50},
            include_timepoints="backward",
            include_zero=False,
            exclude_timepoints=-3,
            cfun=rec,
        )
    w = rec.weights
    idx = np.arange(200)
    lag = idx[:, None] - idx[None, :]  # t - tau
    assert np.all(w[lag < 4] == 0)  # exclude_timepoints=-3 drops lags within 3 samples
    full = gaussian_weights(200, {"var": 50})
    assert np.array_equal(w[lag >= 4], full[lag >= 4])


def test_timecorr_bad_include_timepoints():
    with pytest.raises(ValueError):
        tc.timecorr(_flip_data(), include_timepoints="sideways")


def test_wcorr_reads_kernel_by_row():
    rng = np.random.RandomState(1)
    a = rng.randn(30, 3)
    b = rng.randn(30, 3)
    w = rng.rand(30, 30)
    assert not np.allclose(tc.wcorr(a, b, w), tc.wcorr(a, b, w.T))
