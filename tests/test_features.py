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
    # order 1: 4 -> 10 columns; order 2: 10 -> 55 columns.  Terms (float64
    # elements): the (d, d, T) array, the results of every subject, 3 copies of
    # the inputs, 3 T x T kernels, and the larger of 6 block temporaries (one
    # block is all d x d pairs at these sizes) and the delta-path results
    # (order 1 computes both paths; order 2 is the last and only analyzed).
    assert est[1] == pytest.approx(
        (100 * 4**2 + 100 * 10 + 3 * 100 * 4 + 3 * 100**2
         + max(6 * 4 * 100 * 4, 100 * 10)) * 8 / 1e9
    )
    assert est[2] == pytest.approx(
        (100 * 10**2 + 100 * 55 + 3 * 100 * 10 + 3 * 100**2
         + max(6 * 10 * 100 * 10, 0)) * 8 / 1e9
    )
    assert estimate_memory_gb(4, 0, 100) == [0.0]
    # more subjects hold more results; the delta path of the last order counts
    # only when it is returned
    three = estimate_memory_gb(4, 2, 100, n_subjects=3)
    assert three[2] == pytest.approx(
        (100 * 10**2 + 3 * 100 * 55 + 3 * 3 * 100 * 10 + 3 * 100**2
         + 6 * 10 * 100 * 10) * 8 / 1e9
    )
    assert estimate_memory_gb(4, 2, 100, return_delta=True)[2] >= est[2]


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
    # R-A: row 2 of the backward kernel has weights 1, exp(-1/20), exp(-4/20),
    # effective sample size (sum w)**2 / sum w**2 = 2.98 < 3; row 3 has 3.90
    assert messages == {(
        UserWarning,
        "2 of 60 timepoints have fewer than 3 non-zero kernel weights; their "
        "correlations are undefined and set to 0. 1 of 60 timepoints have an "
        "effective sample size ((sum |w|)**2 / sum w**2 over the kernel row) "
        "below 3; the kernel is too narrow to estimate correlations reliably "
        "there.",
    )}


@pytest.mark.parametrize("cfun_name", ["isfc", "autofc"])
@pytest.mark.parametrize("K,T", [(100, 200), (300, 300)])
def test_memory_estimate_matches_traced_peak_for_three_subjects(cfun_name, K, T):
    """The guard's estimate is honest for the default (isfc) list path: the
    traced peak of a 3-subject unreduced order-1 run is within 1.25x of it."""
    import tracemalloc

    S = 3
    data = make_data(K=K, T=T, n_subjects=S)
    est = estimate_memory_gb(K, 1, T, n_subjects=S)[1]
    tracemalloc.start()
    try:
        X = tc.features(data, order=1, rfun=None, cfun=getattr(tc, cfun_name),
                        weights_params={"var": 100})
        peak = tracemalloc.get_traced_memory()[1] / 1e9
    finally:
        tracemalloc.stop()
    assert len(X[1]) == S and X[1][0].shape == (T, K * (K + 1) // 2)
    assert est / 1.25 <= peak <= 1.25 * est, (peak, est)


def test_memory_guard_counts_subjects():
    # order 2 of K=20, T=300: about 0.56 GB for one subject, 0.67 GB for three
    one = estimate_memory_gb(20, 2, 300)[2]
    three = estimate_memory_gb(20, 2, 300, n_subjects=3)[2]
    assert one < 0.6 < three
    data = make_data(K=20, T=300, n_subjects=3)
    with pytest.raises(MemoryError) as info:
        tc.features(data, order=2, rfun=None, max_memory_gb=0.6)
    assert f"{three:.3g} GB" in str(info.value)
    assert "3 subjects" in str(info.value)
    X = tc.features(data[0], order=2, rfun=None, max_memory_gb=0.6)
    assert X[2].shape == (300, 210 * 211 // 2)


# ------------------------------------------------- undefined-row warning


def _undefined_warnings(rec):
    return [r for r in rec if "non-zero kernel weights" in str(r.message)]


def test_kernel_with_fewer_than_three_weights_everywhere_warns_all_rows():
    # a backward boxcar of width 3 keeps t-1 and t: at most 2 weights per row
    import warnings

    x = make_data(T=30, K=3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = tc.timecorr(x, weights_function=tc.boxcar_weights,
                          weights_params={"width": 3},
                          include_timepoints="backward")
    found = _undefined_warnings(rec)
    assert len(found) == 1
    assert str(found[0].message).startswith("30 of 30 timepoints have fewer than 3")
    assert np.array_equal(out, np.zeros((30, 6)))


@pytest.mark.parametrize("cfun_name", ["isfc", "autofc", "wisfc"])
def test_undefined_row_warning_once_per_call_at_the_users_line(cfun_name):
    import warnings

    data = make_data(n_subjects=4)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tc.timecorr(data, cfun=getattr(tc, cfun_name), include_timepoints="backward",
                    weights_params={"var": 10})
    found = _undefined_warnings(rec)
    assert len(found) == 1
    assert found[0].filename == __file__


def test_undefined_row_warning_points_at_the_users_line_through_features():
    import warnings

    data = make_data(n_subjects=3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tc.features(data, order=2, include_timepoints="backward",
                    weights_params={"var": 10})
    found = _undefined_warnings(rec)
    # one analysis call per order, each warning once
    assert len(found) == 2
    assert all(r.filename == __file__ for r in found)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tc.wcorr(data[0], data[1], tc.one_sided(tc.gaussian_weights(T), "backward"))
    found = _undefined_warnings(rec)
    assert len(found) == 1 and found[0].filename == __file__


# ------------------------------------------- deferred triage: missing tests


def test_list_returning_combine_on_single_array_input():
    data = make_data()

    def as_list(corrs):  # a combine that returns a list (like null_combine)
        return [np.array(c, copy=True) for c in corrs]

    X, D = tc.features(data, order=2, combine=as_list, return_delta=True,
                       weights_params={"var": 10})
    ref_X, ref_D = tc.features(data, order=2, return_delta=True,
                               weights_params={"var": 10})
    for k in (0, 1, 2):
        assert isinstance(X[k], np.ndarray) and X[k].shape == (T, K)
        assert isinstance(D[k], np.ndarray) and D[k].shape == (T, K)
        assert np.array_equal(X[k], ref_X[k])
        assert np.array_equal(D[k], ref_D[k])


@pytest.mark.parametrize(
    "option", [{"include_timepoints": "backward"}, {"include_zero": False}]
)
def test_kernel_options_reach_the_analysis_path_only(option):
    import warnings

    data = make_data(n_subjects=S)
    common = dict(order=2, rfun=None, return_delta=True,
                  weights_function=tc.laplace_weights, weights_params={"scale": 5})
    ref_X, ref_D = tc.features(data, **common)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        X, D = tc.features(data, **common, **option)
    # the only warnings: a backward kernel leaves its first rows undefined
    assert all("non-zero kernel weights" in str(r.message) for r in rec)
    assert len(rec) == (2 if "include_timepoints" in option else 0)
    for k in (1, 2):
        assert_list_equal(D[k], ref_D[k])  # delta path unchanged
        assert all(not np.array_equal(x, r) for x, r in zip(X[k], ref_X[k]))


# ------------------------------------- reducer validation and ndims routing


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"delta_rfun": "bogus"}, ValueError),
        ({"rfun": "not_a_model"}, ValueError),
        ({"rfun": 3}, TypeError),
        ({"delta_rfun": {"params": {"whiten": True}}}, ValueError),
        ({"delta_rfun": {"model": "nonsense"}}, ValueError),
    ],
)
def test_reducers_are_validated_up_front(kwargs, error):
    # R-Q: a bad delta_rfun fails even when the last delta step is skipped
    with pytest.raises(error, match="rfun"):
        tc.features(make_data(), order=1, **kwargs)


@pytest.mark.parametrize(
    "rfun", [None, "PCA", "strength", {"model": "PCA"}, lambda xs: xs]
)
def test_valid_reducers_pass_validation(rfun):
    X = tc.features(make_data(), order=1, rfun=rfun, weights_params={"var": 10})
    assert X[1].shape[0] == T


def test_ndims_reaches_model_reducers_only():
    # R-I: with ndims, a callable delta reducer and a graph measure still run;
    # ndims sets the width of the model-based (PCA) reduction only
    keep = lambda xs: [x[:, :3] for x in xs]  # noqa: E731
    X, D = tc.features(make_data(), order=2, rfun="PCA", delta_rfun=keep, ndims=2,
                       weights_params={"var": 10}, return_delta=True)
    assert X[1].shape == (T, 2) and X[2].shape == (T, 2)
    assert D[1].shape == (T, 3) and D[2].shape == (T, 3)
    X = tc.features(make_data(), order=1, rfun="strength", ndims=2,
                    weights_params={"var": 10})
    assert X[1].shape == (T, K)


# ------------------------------------------- NaN beyond order 1 (Ruling R-E)


def _with_nans(seed=11, n_subjects=None):
    data = make_data(seed=seed, n_subjects=n_subjects)
    arrays = data if isinstance(data, list) else [data]
    for i, x in enumerate(arrays):
        x[5 + i, 1] = np.nan
        x[20 + i, [0, 3]] = np.nan
    return data


def test_unreduced_delta_path_marks_co_fluctuations_of_nan_samples_as_nan():
    data = _with_nans()
    X, D = tc.features(data, order=2, rfun=None, weights_params={"var": 10},
                       return_delta=True)
    D1 = D[1]
    nan_samples = np.isnan(data)
    # mat2vec layout: diagonal first, then the upper triangle row by row
    pairs = [(i, i) for i in range(K)] + [
        (i, j) for i in range(K) for j in range(i + 1, K)
    ]
    for col, (i, j) in enumerate(pairs):
        expected = nan_samples[:, i] | nan_samples[:, j]
        assert np.array_equal(np.isnan(D1[:, col]), expected), (i, j)
    # observed co-fluctuations are unchanged: the z-scored products
    clean = tc.features(np.where(nan_samples, 0.0, data), order=1, rfun=None,
                        return_delta=True)[1][1]
    z = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
    i, j = 0, 2
    col = pairs.index((i, j))
    ok = ~(nan_samples[:, i] | nan_samples[:, j])
    np.testing.assert_allclose(D1[ok, col], (z[:, i] * z[:, j])[ok], atol=1e-12)
    assert clean.shape == D1.shape
    # the next order ignores those samples pairwise: X_2 is finite
    assert np.all(np.isfinite(X[2]))


def test_order_two_ignores_nan_co_fluctuations_pairwise():
    # X_2 from data with NaN samples equals the analysis path applied to D_1
    # with those co-fluctuations removed, not set to 0
    data = _with_nans()
    X, D = tc.features(data, order=2, rfun=None, weights_params={"var": 10},
                       return_delta=True)
    direct = tc.timecorr(D[1], weights_params={"var": 10})
    np.testing.assert_allclose(X[2], direct, atol=1e-12)
    zeroed = tc.timecorr(np.nan_to_num(D[1]), weights_params={"var": 10})
    assert not np.allclose(X[2], zeroed)


def test_callable_delta_reducer_receives_the_nan_co_fluctuations():
    data = _with_nans()
    seen = []

    def keep(xs):
        seen.append([np.isnan(x).sum() for x in xs])
        return [x[:, :4] for x in xs]

    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("error")
        X, D = tc.features(data, order=2, rfun=None, delta_rfun=keep,
                           weights_params={"var": 10}, return_delta=True)
    assert seen and seen[0][0] > 0
    assert np.isnan(D[1]).any()
    assert np.all(np.isfinite(X[2]))


@pytest.mark.parametrize("rfun", ["PCA", "strength"])
def test_model_or_graph_delta_reducer_fills_nan_with_one_warning(rfun):
    data = _with_nans(n_subjects=S)
    with pytest.warns(UserWarning, match="approximate from order 2") as rec:
        X, D = tc.features(data, order=3, rfun=None, delta_rfun=rfun,
                           weights_params={"var": 10}, return_delta=True)
    assert len([r for r in rec if "approximate" in str(r.message)]) == 1
    for d in D[1:]:
        assert all(np.all(np.isfinite(x)) for x in d)


def test_data_without_nan_do_not_warn_about_nan_handling():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tc.features(make_data(n_subjects=S), order=3, weights_params={"var": 10},
                    return_delta=True)
