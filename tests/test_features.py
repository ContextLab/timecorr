"""Tests for the two-path higher-order pipeline (timecorr.higher_order)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import timecorr as tc
from timecorr.higher_order import estimate_memory_gb, two_path_step

T = 60
K = 4
S = 3


def make_data(seed=0, n_subjects=None, T=T, K=K):
    rng = np.random.RandomState(seed)
    if n_subjects is None:
        return rng.randn(T, K)
    return [rng.randn(T, K) for _ in range(n_subjects)]


def assert_list_equal(a, b):
    assert isinstance(a, list) and isinstance(b, list)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------- shapes


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_shapes_single_array_pca(order):
    data = make_data()
    X, D = tc.features(data, order=order, weights_params={"var": 10}, return_delta=True)
    assert isinstance(X, list) and isinstance(D, list)
    assert len(X) == order + 1 and len(D) == order + 1
    for x, d in zip(X, D):
        assert isinstance(x, np.ndarray) and isinstance(d, np.ndarray)
        assert x.shape == (T, K)
        assert d.shape == (T, K)
    assert np.array_equal(X[0], data)
    assert np.array_equal(D[0], data)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_shapes_list_of_subjects_pca(order):
    data = make_data(n_subjects=S)
    X, D = tc.features(data, order=order, weights_params={"var": 10}, return_delta=True)
    assert len(X) == order + 1 and len(D) == order + 1
    for x, d in zip(X, D):
        assert isinstance(x, list) and isinstance(d, list)
        assert len(x) == S and len(d) == S
        for xs, ds in zip(x, d):
            assert xs.shape == (T, K)
            assert ds.shape == (T, K)
    assert_list_equal(X[0], data)


def test_return_delta_false_returns_only_x():
    X = tc.features(make_data(), order=2, weights_params={"var": 10})
    assert isinstance(X, list) and len(X) == 3
    assert all(isinstance(x, np.ndarray) and x.shape == (T, K) for x in X)


@pytest.mark.parametrize("n_subjects", [None, 3])
def test_return_delta_does_not_change_x(n_subjects):
    # without return_delta the last order's delta step is skipped; X is unchanged
    data = make_data(n_subjects=n_subjects) if n_subjects else make_data()
    for kwargs in ({"rfun": None}, {"ndims": 3}):
        X = tc.features(data, order=2, weights_params={"var": 10}, **kwargs)
        Xd, _ = tc.features(
            data, order=2, weights_params={"var": 10}, return_delta=True, **kwargs
        )
        for x, xd in zip(X, Xd):
            for a, b in zip(x if n_subjects else [x], xd if n_subjects else [xd]):
                assert np.array_equal(a, b)


def test_unreduced_widths_grow():
    X, D = tc.features(make_data(), order=2, rfun=None, return_delta=True)
    assert [x.shape for x in X] == [(T, 4), (T, 10), (T, 55)]
    assert [d.shape for d in D] == [(T, 4), (T, 10), (T, 55)]


def test_ndims_sets_reduced_width():
    X, D = tc.features(make_data(n_subjects=S), order=2, ndims=2, return_delta=True)
    for k in (1, 2):
        assert all(x.shape == (T, 2) for x in X[k])
        assert all(d.shape == (T, 2) for d in D[k])


def test_single_subject_list_stays_list():
    data = make_data(n_subjects=1)
    X, D = tc.features(data, order=2, return_delta=True)
    for x, d in zip(X, D):
        assert isinstance(x, list) and len(x) == 1 and x[0].shape == (T, K)
        assert isinstance(d, list) and len(d) == 1 and d[0].shape == (T, K)


def test_combine_applies_to_analysis_path_only():
    data = make_data(n_subjects=S)
    X, D = tc.features(
        data, order=2, combine=tc.corrmean_combine, return_delta=True,
        weights_params={"var": 10},
    )
    for k in (1, 2):
        assert isinstance(X[k], np.ndarray) and X[k].shape == (T, K)
        assert isinstance(D[k], list) and len(D[k]) == S


# ---------------------------------------------------------------- equivalences


def test_x1_unreduced_equals_timecorr_single():
    data = make_data()
    X = tc.features(data, order=1, rfun=None, weights_params={"var": 10})
    expected = tc.timecorr(data, weights_params={"var": 10})
    assert np.array_equal(X[1], expected)


def test_x1_unreduced_equals_timecorr_list():
    data = make_data(n_subjects=S)
    X = tc.features(data, order=1, rfun=None, weights_params={"var": 10})
    expected = tc.timecorr(data, weights_params={"var": 10})
    assert_list_equal(X[1], expected)


def test_x1_unreduced_equals_timecorr_with_kernel_options():
    data = make_data(n_subjects=S)
    kwargs = dict(
        weights_function=tc.laplace_weights,
        weights_params={"scale": 5},
        include_timepoints="backward",
        include_zero=False,
    )
    # the first three rows of a backward kernel without zero have < 3 weights
    with pytest.warns(UserWarning, match="3 of 60 timepoints"):
        X = tc.features(data, order=1, rfun=None, **kwargs)
    with pytest.warns(UserWarning, match="3 of 60 timepoints"):
        expected = tc.timecorr(data, **kwargs)
    assert_list_equal(X[1], expected)


def test_x1_unreduced_equals_timecorr_with_combine():
    data = make_data(n_subjects=S)
    kwargs = dict(weights_function=tc.laplace_weights, weights_params={"scale": 5},
                  combine=tc.corrmean_combine)
    X = tc.features(data, order=1, rfun=None, **kwargs)
    assert np.array_equal(X[1], tc.timecorr(data, **kwargs))


def test_d1_unreduced_equals_delta_timecorr():
    for data in (make_data(), make_data(n_subjects=S)):
        _, D = tc.features(data, order=1, rfun=None, return_delta=True)
        expected = tc.timecorr(data, weights_function=tc.eye_weights, cfun=tc.autofc)
        if isinstance(data, list):
            assert_list_equal(D[1], expected)
        else:
            assert np.array_equal(D[1], expected)


def test_x1_pca_is_joint_reduction_of_timecorr():
    data = make_data(n_subjects=S)
    X = tc.features(data, order=1, weights_params={"var": 10})
    expected = tc.reduce(tc.timecorr(data, weights_params={"var": 10}), rfun="PCA")
    assert_list_equal(X[1], expected)


def test_x2_is_analysis_kernel_applied_to_d1():
    """Order 2 analysis features come from the unblurred D_1, not from X_1."""
    data = make_data(n_subjects=S)
    X, D = tc.features(data, order=2, rfun=None, weights_params={"var": 10},
                       return_delta=True)
    expected = tc.timecorr(D[1], weights_params={"var": 10})
    assert_list_equal(X[2], expected)
    blurred_twice = tc.timecorr(X[1], weights_params={"var": 10})
    assert not np.allclose(X[2][0], blurred_twice[0])


def test_widening_analysis_kernel_changes_x2_not_delta_path():
    data = make_data(n_subjects=S)
    Xn, Dn = tc.features(data, order=2, weights_params={"var": 5}, return_delta=True)
    Xw, Dw = tc.features(data, order=2, weights_params={"var": 500}, return_delta=True)
    for k in (1, 2):
        assert_list_equal(Dn[k], Dw[k])
    assert not np.allclose(Xn[2][0], Xw[2][0])


def test_features_equals_repeated_two_path_step():
    data = make_data(n_subjects=S)
    X, D = tc.features(data, order=3, weights_params={"var": 10}, return_delta=True)
    prev = data
    for k in (1, 2, 3):
        x, d = two_path_step(prev, weights_params={"var": 10})
        assert_list_equal(x, X[k])
        assert_list_equal(d, D[k])
        prev = d


def test_two_path_step_single_array():
    data = make_data()
    x, d = two_path_step(data, rfun=None, weights_params={"var": 10})
    assert np.array_equal(x, tc.timecorr(data, weights_params={"var": 10}))
    assert np.array_equal(
        d, tc.timecorr(data, weights_function=tc.eye_weights, cfun=tc.autofc)
    )


def test_delta_rfun_applies_to_delta_path_only():
    data = make_data(n_subjects=S)

    def first_two_columns(arrays):
        return [a[:, :2] for a in arrays]

    X, D = tc.features(data, order=2, rfun="PCA", delta_rfun=first_two_columns,
                       return_delta=True, weights_params={"var": 10})
    raw_d1 = tc.timecorr(data, weights_function=tc.eye_weights, cfun=tc.autofc)
    assert_list_equal(D[1], [a[:, :2] for a in raw_d1])
    # analysis path at order 1 is reduced with PCA back to K columns
    assert all(x.shape == (T, K) for x in X[1])
    # order 2 runs on the two selected columns: 3 vectorized columns -> 2 after PCA
    assert all(x.shape == (T, 2) for x in X[2])
    assert all(d.shape == (T, 2) for d in D[2])


def test_delta_rfun_none_defaults_to_rfun():
    data = make_data(n_subjects=S)
    _, D1 = tc.features(data, order=2, return_delta=True)
    _, D2 = tc.features(data, order=2, delta_rfun="PCA", return_delta=True)
    for k in (1, 2):
        assert_list_equal(D1[k], D2[k])


# ---------------------------------------------------------------- input types


@pytest.mark.parametrize("convert", [pd.DataFrame, pl.DataFrame])
def test_dataframe_inputs_match_numpy(convert):
    data = make_data(n_subjects=S)
    X, D = tc.features(data, order=2, weights_params={"var": 10}, return_delta=True)
    Xc, Dc = tc.features([convert(d) for d in data], order=2,
                         weights_params={"var": 10}, return_delta=True)
    for k in range(3):
        assert_list_equal(Xc[k], X[k])
        assert_list_equal(Dc[k], D[k])

    single = data[0]
    Xs = tc.features(single, order=2, weights_params={"var": 10})
    Xsc = tc.features(convert(single), order=2, weights_params={"var": 10})
    for a, b in zip(Xs, Xsc):
        assert isinstance(b, np.ndarray)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("order", [-1, 1.5, 2.0, "2", None, True])
def test_bad_order_raises(order):
    with pytest.raises(ValueError, match="order"):
        tc.features(make_data(), order=order)


def test_numpy_integer_order_accepted():
    X = tc.features(make_data(), order=np.int64(1))
    assert len(X) == 2


# ---------------------------------------------------------------- memory


def test_estimate_memory_gb_values():
    est = estimate_memory_gb(4, 2, 100)
    assert len(est) == 3
    assert est[0] == 0.0
    # order 1: 4 -> 10 columns; order 2: 10 -> 55 columns
    assert est[1] == pytest.approx((100 * 4**2 * 8 + 100 * 10 * 8) / 1e9)
    assert est[2] == pytest.approx((100 * 10**2 * 8 + 100 * 55 * 8) / 1e9)
    assert estimate_memory_gb(4, 0, 100) == [0.0]


def largest_k_brute_force(order, T, limit):
    k = 1
    while max(estimate_memory_gb(k + 1, order, T)) <= limit:
        k += 1
    return k


def test_unreduced_memory_guard_raises_with_estimate_and_fitting_k():
    data = make_data(K=10, T=100)
    est = estimate_memory_gb(10, 3, 100)
    assert est[2] < 1.0 < est[3]
    with pytest.raises(MemoryError) as info:
        tc.features(data, order=3, rfun=None, max_memory_gb=1.0)
    msg = str(info.value)
    assert f"{est[3]:.3g} GB" in msg
    fit = largest_k_brute_force(3, 100, 1.0)
    assert max(estimate_memory_gb(fit, 3, 100)) <= 1.0
    assert max(estimate_memory_gb(fit + 1, 3, 100)) > 1.0
    assert f"K={fit}" in msg


def test_memory_guard_allows_runs_that_fit():
    data = make_data(K=10, T=100)
    X = tc.features(data, order=2, rfun=None, max_memory_gb=1.0)
    assert X[2].shape == (100, 55 * 56 // 2)


def test_memory_guard_with_reduced_delta_path():
    """With the delta path reduced, each step costs about the same; the guard
    still refuses a step whose allocation exceeds the limit."""
    data = make_data(K=50, T=100)
    step = estimate_memory_gb(50, 1, 100)[1]
    with pytest.raises(MemoryError) as info:
        tc.features(data, order=2, rfun=None, delta_rfun="PCA",
                    max_memory_gb=step / 2)
    assert f"{step:.3g} GB" in str(info.value)


def test_backward_corrmean_pipeline_is_finite_and_warns_only_for_undefined_rows():
    import warnings

    data = make_data(n_subjects=S)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        X = tc.features(data, order=1, rfun=None, include_timepoints="backward",
                        combine=tc.corrmean_combine, weights_params={"var": 10})
    assert np.all(np.isfinite(X[1]))
    messages = {(r.category, str(r.message)) for r in rec}
    assert messages == {(
        UserWarning,
        "2 of 60 timepoints have fewer than 3 non-zero kernel weights; their "
        "correlations are undefined and set to 0.",
    )}
