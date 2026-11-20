"""
The feeding path's kernel, per-order reducers and streamed summaries in
``features`` and ``two_path_step`` (the defaults are tested in
``tests/test_features_defaults.py``; the original pipeline is pinned by
``tests/test_legacy_pipeline.py``).

Everything is computed for real on simulated data; nothing is mocked.
"""

import tracemalloc
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import summaries as sm
from timecorr.higher_order import estimate_memory_gb, two_path_step

T = 300
FEED = dict(delta_weights_function=tc.gaussian_weights,
            delta_weights_params={"var": 25})
WIDE = {"var": 100}


def nested(S=2, T=T, seed=3):
    """24 nodes in 6 modules (2 supergroups x 3 modules x 4 nodes)."""
    return tc.simulate_nested(
        S=S, T=T, n_supergroups=2, modules_per_supergroup=3, nodes_per_module=4,
        set_random_seed=seed,
    )


def noise(S=2, T=120, K=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, K)) for _ in range(S)]


def assert_lists_close(a, b, rtol=1e-9, atol=1e-12):
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return f(*args, **kwargs)


def budget_for(T, K, S, b):
    """max_memory_gb that gives streamed steps of exactly b rows per block."""
    return 8 * sm._streamed_peak_elements(T, K, S, b) / 1e9


# ------------------------------------------------------------ feeding kernel


def test_feeding_kernel_gives_summarized_kernel_correlations():
    data, _ = nested()
    X, D = tc.features(
        data, order=2, cfun=tc.autofc, weights_params=WIDE, rfun=None,
        delta_rfun="weighted_strength", return_delta=True, **FEED,
    )
    corrs = tc.timecorr(data, weights_params={"var": 25}, cfun=tc.autofc, rfun=None)
    assert_lists_close(D[1], tc.WeightedStrength()(corrs))
    for s in range(2):  # unsplit weighted strength has no shared fit
        assert_lists_close(D[1][s], tc.WeightedStrength()(corrs[s]))
    expected = tc.timecorr(D[1], weights_params=WIDE, cfun=tc.autofc, rfun=None)
    assert_lists_close(X[2], expected)
    assert D[1][0].shape == (T, 24)


def test_gaussian_feeding_kernel_is_the_default():
    data = noise()
    common = dict(order=2, cfun=tc.autofc, weights_params={"var": 10},
                  delta_rfun="weighted_strength", return_delta=True)
    X0, D0 = quiet(tc.features, data, **common)
    X1, D1 = quiet(tc.features, data, **common, **FEED)
    for k in range(3):
        assert_lists_close(X0[k], X1[k], rtol=0, atol=0)
        assert_lists_close(D0[k], D1[k], rtol=0, atol=0)


def test_feeding_kernel_params_none_uses_the_function_defaults():
    data = noise()
    common = dict(order=1, cfun=tc.autofc, rfun=None, delta_rfun=None,
                  return_delta=True, delta_weights_function=tc.laplace_weights)
    _, D_none = tc.features(data, **common)
    _, D_dflt = tc.features(data, delta_weights_params=tc.laplace_params, **common)
    assert_lists_close(D_none[1], D_dflt[1], rtol=0, atol=0)
    expected = tc.timecorr(data, weights_function=tc.laplace_weights, cfun=tc.autofc)
    assert_lists_close(D_none[1], expected)


@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc], ids=["autofc", "isfc"])
def test_streamed_equals_materialized_through_the_pipeline(cfun):
    data, _ = nested()
    K, S = 24, 2
    common = dict(order=2, cfun=cfun, weights_params=WIDE,
                  rfun=tc.WeightedStrength(signed=True),
                  delta_rfun="weighted_strength", return_delta=True, **FEED)
    X_big, D_big = quiet(tc.features, data, max_memory_gb=None, **common)
    # two rows per block at order 1 (order 2 has 48 inputs and gets more)
    X_small, D_small = quiet(
        tc.features, data, max_memory_gb=budget_for(T, K, S, 2), **common
    )
    for k in (1, 2):
        assert_lists_close(X_small[k], X_big[k])
        assert_lists_close(D_small[k], D_big[k])
    # materialized: the correlation function's whole output, then the summary
    corrs = tc.timecorr(data, weights_params=WIDE, cfun=cfun, rfun=None)
    assert_lists_close(X_big[1], quiet(tc.WeightedStrength(signed=True), corrs))
    assert X_big[2][0].shape == (T, 2 * D_big[1][0].shape[1])


def test_include_timepoints_backward_shapes_the_feeding_kernel():
    data = noise(T=120, K=5)
    t0 = 70
    changed = [x.copy() for x in data]
    rng = np.random.default_rng(9)
    for x in changed:
        x[t0:] = 3 * rng.standard_normal(x[t0:].shape)
    feed = dict(delta_weights_function=tc.gaussian_weights,
                delta_weights_params={"var": 4})
    common = dict(order=2, cfun=tc.autofc, weights_params={"var": 4}, rfun=None,
                  delta_rfun=None, return_delta=True, **feed)
    A = quiet(tc.features, data, include_timepoints="backward", **common)
    B = quiet(tc.features, changed, include_timepoints="backward", **common)
    for out in (0, 1):  # X, D
        for k in (1, 2):
            for a, b in zip(A[out][k], B[out][k]):
                # equal up to round-off: the estimator's centering and scaling
                # of each column use every timepoint
                np.testing.assert_allclose(a[:t0], b[:t0], rtol=1e-7, atol=1e-9)
    # with a two-sided kernel the rows just before t0 do change
    A = quiet(tc.features, data, **common)
    B = quiet(tc.features, changed, **common)
    assert not np.allclose(A[1][1][0][t0 - 2], B[1][1][0][t0 - 2])
    assert not np.allclose(A[0][2][0][t0 - 2], B[0][2][0][t0 - 2])


def test_events_restrict_the_feeding_kernel():
    data = noise(T=120, K=5)
    t0 = 60
    changed = [x.copy() for x in data]
    rng = np.random.default_rng(10)
    for x in changed:
        x[t0:] = 3 * rng.standard_normal(x[t0:].shape)
    common = dict(order=1, cfun=tc.autofc, weights_params={"var": 10}, rfun=None,
                  delta_rfun=None, return_delta=True, **FEED)
    _, DA = tc.features(data, events=[t0], **common)
    _, DB = tc.features(changed, events=[t0], **common)
    for a, b in zip(DA[1], DB[1]):
        np.testing.assert_allclose(a[:t0], b[:t0], rtol=1e-7, atol=1e-9)
    _, DA = tc.features(data, **common)
    _, DB = tc.features(changed, **common)
    assert not np.allclose(DA[1][0][t0 - 2], DB[1][0][t0 - 2])
    # the events of order 1 restrict the feeding kernel as timecorr does
    _, D = tc.features(data, events={1: [t0]}, **common)
    expected = tc.timecorr(data, weights_params={"var": 25}, cfun=tc.autofc,
                           events=[t0])
    assert_lists_close(D[1], expected)


def test_events_restrict_the_streamed_feeding_kernel():
    data = noise(T=120, K=6)
    common = dict(order=1, cfun=tc.autofc, weights_params={"var": 10}, rfun=None,
                  delta_rfun="weighted_strength", return_delta=True, **FEED)
    _, D = quiet(tc.features, data, events=[[50], [40, 80]], **common)
    corrs = tc.timecorr(data, weights_params={"var": 25}, cfun=tc.autofc,
                        events=[[50], [40, 80]])
    assert_lists_close(D[1], quiet(tc.WeightedStrength(), corrs))


# ------------------------------------------------------------------ grammar


def test_order_map_and_auto_resolution():
    data, _ = nested()
    X, D = quiet(
        tc.features, data, order=2, cfun=tc.autofc, weights_params=WIDE,
        rfun={1: "PCA", 2: tc.WeightedStrength(signed=True)},
        delta_rfun="weighted_strength", return_delta=True, **FEED,
    )
    assert X[1][0].shape == (T, 24)  # PCA back to K columns
    assert X[2][0].shape == (T, 2 * D[1][0].shape[1])
    # missing orders use "auto": weighted strength (signed) at order 2
    X2 = quiet(
        tc.features, data, order=2, cfun=tc.autofc, weights_params=WIDE,
        rfun={1: "PCA"}, delta_rfun="weighted_strength", **FEED,
    )
    for a, b in zip(X[2], X2[2]):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_auto_reducers_give_the_documented_widths():
    data, truth = nested()
    X, D = quiet(
        tc.features, data, order=2, cfun=tc.autofc, weights_params=WIDE,
        rfun="auto", delta_rfun="auto", return_delta=True, **FEED,
    )
    assert X[1][0].shape == (T, 24)
    assert D[1][0].shape == (T, 6)  # one column per true module
    assert X[2][0].shape == (T, 2 * D[1][0].shape[1])


@pytest.mark.parametrize(
    "rfun",
    [{"params": {"whiten": True}}, {1: "PCA", "x": "PCA"}, {3: "PCA"}, {0: "PCA"},
     {1: {2: "PCA"}}, {1: "bogus"}, {True: "PCA"}],
)
@pytest.mark.parametrize("name", ["rfun", "delta_rfun"])
def test_bad_order_maps_raise(rfun, name):
    with pytest.raises(ValueError, match="rfun"):
        tc.features(noise(), order=2, **{name: rfun})


def test_delta_rfun_none_copies_the_reducer_of_that_order():
    data = noise(T=120, K=6)
    common = dict(order=2, cfun=tc.autofc, weights_params={"var": 10},
                  return_delta=True, **FEED)
    X, D = quiet(tc.features, data, rfun={1: "weighted_strength", 2: "PCA"},
                 delta_rfun=None, **common)
    Xe, De = quiet(tc.features, data, rfun={1: "weighted_strength", 2: "PCA"},
                   delta_rfun={1: "weighted_strength", 2: "PCA"}, **common)
    for k in (1, 2):
        assert_lists_close(D[k], De[k], rtol=0, atol=0)
        assert_lists_close(X[k], Xe[k], rtol=0, atol=0)
    assert D[1][0].shape == (120, 6)


def test_features_equals_repeated_two_path_step_under_an_order_map():
    data = noise(T=100, K=6)
    kw = dict(cfun=tc.autofc, weights_params={"var": 10},
              rfun={1: "PCA", 2: "weighted_strength", 3: tc.WeightedStrength()},
              delta_rfun={1: "weighted_strength", 2: "PCA"}, **FEED)
    X, D = quiet(tc.features, data, order=3, return_delta=True, **kw)
    prev = data
    for k in (1, 2, 3):
        x, d = quiet(two_path_step, prev, order=k, **kw)
        assert_lists_close(x, X[k], rtol=0, atol=0)
        assert_lists_close(d, D[k], rtol=0, atol=0)
        prev = d


def test_two_path_step_order_resolves_auto():
    data = noise(T=100, K=6)
    x1, _ = quiet(two_path_step, data, order=1, cfun=tc.autofc,
                  weights_params={"var": 10}, rfun="auto",
                  delta_rfun="weighted_strength", **FEED)
    x2, _ = quiet(two_path_step, data, order=2, cfun=tc.autofc,
                  weights_params={"var": 10}, rfun="auto",
                  delta_rfun="weighted_strength", **FEED)
    assert x1[0].shape == (100, 6)  # PCA
    assert x2[0].shape == (100, 12)  # signed weighted strength
    with pytest.raises(ValueError, match="order"):
        two_path_step(data, order=0)


# -------------------------------------------------------- instances and fits_


def test_fits_are_recorded_and_cleared_and_the_count_is_automatic():
    data, truth = nested()
    analysis = tc.WeightedStrength(signed=True)
    feeding = tc.Communities()
    # an earlier fit, removed at the start of the call
    feeding.n = 3
    quiet(feeding, tc.timecorr(data, cfun=tc.autofc, weights_params=WIDE)[0])
    feeding.n = None
    analysis(tc.timecorr(data, cfun=tc.autofc, weights_params=WIDE)[0])
    assert len(feeding.fits_) == 1 and len(analysis.fits_) == 1
    for _ in range(2):
        X, D = quiet(
            tc.features, data, order=2, cfun=tc.autofc, weights_params=WIDE,
            rfun={1: "PCA", 2: analysis}, delta_rfun=feeding, return_delta=True,
            **FEED,
        )
        # cleared at the start of each call: one feeding fit per computed order
        assert [(f["order"], f["path"]) for f in feeding.fits_] == [
            (1, "feeding"), (2, "feeding")]
        assert [(f["order"], f["path"]) for f in analysis.fits_] == [(2, "analysis")]
    fit = feeding.fits_[0]
    assert fit["n"] == 6 and fit["null_threshold"] is not None
    labels, modules = fit["labels"], truth["module"]
    # the same partition as the true modules
    pairs_found = labels[:, None] == labels[None, :]
    pairs_true = modules[:, None] == modules[None, :]
    assert np.array_equal(pairs_found, pairs_true)
    assert D[1][0].shape == (T, 6)


def test_direct_n_none_still_raises():
    data, _ = nested()
    corrs = tc.timecorr(data, cfun=tc.autofc, weights_params=WIDE)
    with pytest.raises(ValueError, match="number of communities n"):
        tc.Communities()(corrs)


def test_communities_on_isfc_analysis_needs_n():
    data, _ = nested()
    with pytest.raises(ValueError, match="number of communities n.*cfun=autofc "
                       "and combine=null_combine"):
        tc.features(data, order=1, cfun=tc.isfc, weights_params=WIDE,
                    rfun=tc.Communities())


@pytest.mark.parametrize("settings", [
    dict(cfun=tc.isfc, rfun="communities"),
    dict(cfun=tc.wisfc, rfun=tc.WeightedStrength(split=True)),
    dict(cfun=tc.autofc, combine=tc.corrmean_combine, rfun=tc.Communities()),
], ids=["isfc", "wisfc", "corrmean_combine"])
def test_automatic_count_error_names_its_conditions(settings):
    data, _ = nested()
    with pytest.raises(ValueError) as err:
        tc.features(data, order=1, weights_params=WIDE, **settings)
    message = str(err.value)
    assert "cfun=autofc and combine=null_combine" in message
    assert "feeding path" in message
    assert "available only inside features" not in message


def test_communities_on_autofc_analysis_counts_automatically():
    data, _ = nested()
    comm = tc.Communities()
    X = quiet(tc.features, data, order=1, cfun=tc.autofc, weights_params=WIDE,
              rfun=comm)
    assert comm.fits_[0]["path"] == "analysis" and comm.fits_[0]["n"] >= 2
    assert X[1][0].shape == (T, comm.fits_[0]["n"])


def test_automatic_count_needs_forty_timepoints():
    data = noise(T=30, K=8)
    with pytest.raises(ValueError, match="number of communities n"):
        tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 10},
                    rfun="PCA", delta_rfun=tc.Communities(), **FEED)


def test_eye_feeding_path_counts_communities_too():
    data, _ = nested()
    comm = tc.Communities()
    _, D = quiet(tc.features, data, order=1, cfun=tc.autofc, weights_params=WIDE,
                 rfun="PCA", delta_rfun=comm, return_delta=True)
    assert comm.fits_[0]["path"] == "feeding"
    assert comm.fits_[0]["null_threshold"] is not None
    assert D[1][0].shape[1] == comm.fits_[0]["n"]


def test_string_summaries_are_fresh_per_call():
    data = noise(T=100, K=6)
    a = quiet(tc.features, data, order=2, cfun=tc.autofc, weights_params={"var": 10},
              rfun="weighted_strength", **FEED)
    b = quiet(tc.features, data, order=2, cfun=tc.autofc, weights_params={"var": 10},
              rfun="weighted_strength", **FEED)
    for k in (1, 2):
        assert_lists_close(a[k], b[k], rtol=0, atol=0)


# ------------------------------------------------------------------- ndims


def test_ndims_with_a_summary_on_a_computed_step_raises():
    data = noise(T=100, K=6)
    with pytest.raises(ValueError, match="ndims"):
        tc.features(data, order=2, rfun="auto", ndims=3, **FEED)
    with pytest.raises(ValueError, match="ndims"):
        tc.features(data, order=1, rfun="PCA", delta_rfun="weighted_strength",
                    ndims=3, return_delta=True, **FEED)
    with pytest.raises(ValueError, match="ndims"):
        two_path_step(data, rfun="PCA", delta_rfun="weighted_strength", ndims=3)


def test_reduce_ndims_message_names_summaries():
    corrs = tc.timecorr(noise(T=100, K=6), cfun=tc.autofc, weights_params={"var": 10})
    with pytest.raises(ValueError, match="graph measure, summary or callable"):
        tc.reduce(corrs, rfun="weighted_strength", ndims=3)


def test_ndims_with_a_summary_only_at_an_uncomputed_step_does_not_raise():
    data = noise(T=100, K=6)
    X = tc.features(data, order=1, cfun=tc.autofc, weights_params={"var": 10},
                    rfun="auto", delta_rfun="auto", ndims=3, **FEED)
    assert X[1][0].shape == (100, 3)
    X = tc.features(data, order=1, cfun=tc.autofc, weights_params={"var": 10},
                    rfun={1: "PCA"}, delta_rfun="communities", ndims=2)
    assert X[1][0].shape == (100, 2)


# --------------------------------------------------------------- NaN samples


def _nan_data():
    data = noise(T=120, K=5, seed=4)
    rng = np.random.default_rng(6)
    for x in data:
        x[rng.random(x.shape) < 0.05] = np.nan
    return data


@pytest.mark.parametrize("delta_rfun", [None, "PCA", "weighted_strength"])
def test_nan_with_a_kernel_feeding_path_gives_finite_features(delta_rfun):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        X, D = tc.features(_nan_data(), order=2, cfun=tc.autofc,
                           weights_params={"var": 10}, rfun=None,
                           delta_rfun=delta_rfun, return_delta=True, **FEED)
    for k in (1, 2):
        assert all(np.isfinite(d).all() for d in D[k])
        assert all(np.isfinite(x).all() for x in X[k])
    assert not any("co-fluctuations" in str(r.message) for r in rec)


def test_nan_with_the_eye_feeding_path_keeps_the_old_message():
    with pytest.warns(UserWarning, match="co-fluctuations"):
        tc.features(_nan_data(), order=2, cfun=tc.autofc,
                    weights_params={"var": 10}, rfun="PCA", delta_rfun=None,
                    delta_weights_function=tc.eye_weights)


# ---------------------------------------------------------------- warnings


@pytest.mark.parametrize("streamed", [False, True], ids=["materialized", "streamed"])
def test_one_estimator_warning_per_step_and_path(streamed):
    data = noise(T=100, K=5)
    reducers = (dict(rfun=tc.WeightedStrength(signed=True),
                     delta_rfun="weighted_strength") if streamed
                else dict(rfun=None, delta_rfun=None))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 10},
                    include_timepoints="backward", return_delta=True,
                    delta_weights_function=tc.gaussian_weights,
                    delta_weights_params={"var": 4}, **reducers)
    estimator = [r for r in rec if "non-zero kernel weights" in str(r.message)]
    # X_1, D_1, X_2, D_2: the first rows of a backward kernel are undefined
    assert len(estimator) == 4
    assert all(r.filename == __file__ for r in estimator)


# ------------------------------------------------------------------- memory


def test_streamed_memory_estimate():
    T_, K_ = 200, 300
    streamed = estimate_memory_gb(K_, 1, T_, streamed=True)
    full = estimate_memory_gb(K_, 1, T_)
    assert streamed[0] == 0.0 and full[0] == 0.0
    assert streamed[1] < full[1]
    assert streamed[1] == 8 * sm._streamed_peak_elements(T_, K_, 1, 1) / 1e9
    two = estimate_memory_gb(K_, 2, T_, n_subjects=3, streamed=True)
    assert two[1] == 8 * sm._streamed_peak_elements(T_, K_, 3, 1) / 1e9


def recording_kernel(calls):
    """A Gaussian kernel function that appends T to ``calls`` when called:
    every correlation step builds its kernel before computing anything."""

    def kernel(T, params=None):
        calls.append(T)
        return tc.gaussian_weights(T, tc.gaussian_params if params is None else params)

    return kernel


@pytest.mark.parametrize("path", ["analysis", "feeding"])
def test_streamed_budget_too_small_raises_before_computing(path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2000, 2000))
    need = estimate_memory_gb(2000, 1, 2000, streamed=True)[1]
    calls = []
    kernel = recording_kernel(calls)
    kw = (dict(rfun=tc.WeightedStrength(signed=True)) if path == "analysis"
          else dict(rfun=tc.WeightedStrength(), delta_rfun="weighted_strength",
                    return_delta=True, delta_weights_function=kernel,
                    delta_weights_params={"var": 25}))
    with pytest.raises(MemoryError, match="streamed"):
        tc.features(data, order=1, cfun=tc.autofc, max_memory_gb=0.9 * need,
                    weights_function=kernel, **kw)
    # no kernel was built, so no correlation was computed
    assert calls == []


def test_streamed_budget_that_fits_runs():
    data = noise(T=100, K=6)
    need = estimate_memory_gb(6, 1, 100, n_subjects=2, streamed=True)[1]
    X = quiet(tc.features, data, order=1, cfun=tc.autofc, weights_params={"var": 10},
              rfun=tc.WeightedStrength(), max_memory_gb=need)
    assert X[1][0].shape == (100, 6)


# ------------------------------------------------------- default feeding kernel


def test_delta_weights_function_must_be_callable():
    with pytest.raises(TypeError, match="delta_weights_function"):
        tc.features(noise(), order=1, delta_weights_function="gaussian")


def test_default_feeding_selection_gives_the_documented_kernels():
    # _feeding_kernel(None, ...) selects the kernel named by _DEFAULT_FEEDING
    # ("delta" or "gaussian"); the selection is passed here as an argument
    from timecorr.higher_order import _DEFAULT_FEEDING, _feeding_kernel

    data = noise(T=100, K=5)
    common = dict(order=1, cfun=tc.autofc, rfun=None, delta_rfun=None,
                  return_delta=True)
    _, explicit = tc.features(data, **common, **FEED)
    function, params = _feeding_kernel(None, None, default="gaussian")
    assert function is tc.gaussian_weights and params == {"var": 25}
    _, selected = tc.features(data, delta_weights_function=function,
                              delta_weights_params=params, **common)
    assert_lists_close(selected[1], explicit[1], rtol=0, atol=0)
    assert _feeding_kernel(None, {"var": 4}, default="gaussian") == (
        tc.gaussian_weights, {"var": 4})

    assert _DEFAULT_FEEDING == "gaussian"
    assert _feeding_kernel(None, None) == (tc.gaussian_weights, {"var": 25})
    assert _feeding_kernel(None, None, default="delta") == (tc.eye_weights, None)
    _, default = tc.features(data, **common)
    assert_lists_close(default[1], explicit[1], rtol=0, atol=0)
    assert_lists_close(
        default[1],
        tc.timecorr(data, weights_function=tc.gaussian_weights,
                    weights_params={"var": 25}, cfun=tc.autofc),
        rtol=0, atol=0,
    )


# review follow-ups: memory held across the two paths, argument check -------

def _held_sizes(d=10, T=2000, S=2):
    """GB of one streamed step at one row per block, and of S (T, d(d+1)/2)
    results held while it runs."""
    need = estimate_memory_gb(d, 1, T, S, streamed=True)[1]
    held = S * T * (d * (d + 1) // 2) * 8 / 1e9
    return need, held


def test_streamed_feeding_counts_the_unreduced_analysis_results():
    # rfun=None materializes X unreduced; the streamed feeding step runs while
    # X is held, so the budget must cover both
    data = noise(S=2, T=2000, K=10)
    need, held = _held_sizes()
    common = dict(order=1, cfun=tc.autofc, rfun=None, return_delta=True,
                  delta_rfun="weighted_strength", **FEED)
    assert estimate_memory_gb(10, 1, 2000, 2)[1] < need + held / 2  # analysis fits
    with pytest.raises(MemoryError, match="results held"):
        tc.features(data, max_memory_gb=need + held / 2, **common)
    X, D = tc.features(data, max_memory_gb=need + 2 * held, **common)
    assert X[1][0].shape == (2000, 55) and D[1][0].shape == (2000, 10)


def test_shift_null_on_the_eye_path_is_checked_before_computing():
    # The budget lets the materialized analysis step, whose kernel function
    # is the user's (T x T arrays, estimate_memory_gb(..., weights_function=)),
    # run and leaves too little for the shift null (one streamed row per
    # block on the eye kernel, read in chunks, while the co-fluctuations are
    # held).  T is 1,000 (the controller accepted the change from T 2,000):
    # at T 2,000 the analysis step alone is estimated at 0.118 GB, more than
    # the shift null's 0.095 GB, so no budget lets the first run and stops
    # the second there.
    T = 1000
    data = noise(S=2, T=T, K=10)
    need, held = _held_sizes(T=T)
    calls = []
    kernel = recording_kernel(calls)
    analysis = estimate_memory_gb(10, 1, T, 2, weights_function=kernel)[1]
    budget = (analysis + need + held) / 2
    assert analysis < budget < need + held
    common = dict(order=1, cfun=tc.autofc, rfun="PCA", return_delta=True,
                  delta_rfun=tc.Communities(), weights_function=kernel,
                  delta_weights_function=tc.eye_weights)
    with pytest.raises(MemoryError, match="shift null of the feeding step"):
        tc.features(data, max_memory_gb=budget, **common)
    # the analysis kernel was not built, so no correlation was computed
    assert calls == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, D = tc.features(data, max_memory_gb=need + 2 * held + 0.2, **common)
    assert len(D[1]) == 2


def test_delta_weights_params_alone_sets_the_default_gaussian():
    data = noise()
    common = dict(cfun=tc.autofc, rfun=None, delta_rfun=None)
    _, D = tc.features(data, order=1, return_delta=True,
                       delta_weights_params={"var": 4}, **common)
    expected = tc.timecorr(data, weights_function=tc.gaussian_weights,
                           weights_params={"var": 4}, cfun=tc.autofc)
    assert_lists_close(D[1], expected)
    _, step = two_path_step(data, delta_weights_params={"var": 4}, **common)
    assert_lists_close(step, D[1], rtol=0, atol=0)
    # the "delta" selection (eye_weights) takes no parameters
    from timecorr.higher_order import _feeding_kernel

    with pytest.raises(ValueError, match="eye_weights takes no parameters"):
        _feeding_kernel(None, {"var": 4}, default="delta")


def test_explicit_eye_feeding_kernel_takes_no_params():
    data = noise()
    with pytest.raises(ValueError, match="eye_weights takes no parameters"):
        tc.features(data, order=1, cfun=tc.autofc, delta_weights_function=tc.eye_weights,
                    delta_weights_params={"var": 4})
    with pytest.raises(ValueError, match="eye_weights takes no parameters"):
        two_path_step(data, cfun=tc.autofc, delta_weights_function=tc.eye_weights,
                      delta_weights_params={})


# final review fixes ---------------------------------------------------------


def test_streamed_step_without_a_budget_holds_less_than_one_cube():
    # max_memory_gb=None: the streamed step uses the estimator's block size
    # (2**23 correlations), not one block of all K rows
    K_, T_ = 700, 300
    data = np.random.default_rng(11).standard_normal((T_, K_))
    cube = 8 * K_ * K_ * T_
    tracemalloc.start()
    try:
        X = quiet(tc.features, data, order=1, cfun=tc.autofc,
                  weights_params={"var": 10}, rfun=tc.WeightedStrength(),
                  max_memory_gb=None)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert X[1].shape == (T_, K_)
    assert peak < cube, (peak / 1e6, cube / 1e6)


def test_streamed_steps_are_checked_before_order_one():
    # order 1 holds its correlations and fits the budget; the streamed order-2
    # step (input width unknown in advance: K is used) does not
    calls = []

    def recording_pca(corrs):
        calls.append(len(corrs))
        return tc.reduce(corrs, rfun="PCA")

    data = noise(S=2, T=2000, K=10)
    first = estimate_memory_gb(10, 2, 2000, 2)[1]
    streamed = estimate_memory_gb(10, 2, 2000, 2, streamed=True)[2]
    assert first < streamed
    with pytest.raises(MemoryError, match="streamed analysis step at order 2"):
        tc.features(data, order=2, cfun=tc.autofc,
                    rfun={1: recording_pca, 2: tc.WeightedStrength()},
                    delta_rfun=None, delta_weights_function=tc.eye_weights,
                    max_memory_gb=(first + streamed) / 2)
    assert calls == []


def test_streamed_feeding_steps_are_checked_before_order_one():
    # a known width: unreduced order-1 features have d(d+1)/2 = 55 columns
    data = noise(S=2, T=2000, K=10)
    first = estimate_memory_gb(10, 2, 2000, 2)[1]
    width_k = estimate_memory_gb(10, 1, 2000, 2, streamed=True)[1]
    width_55 = estimate_memory_gb(55, 1, 2000, 2, streamed=True)[1]
    # fits order 1 and a streamed step on K = 10 inputs, not one on 55 inputs
    budget = (width_k + width_55) / 2
    assert first < budget and width_k < budget < width_55
    calls = []
    with pytest.raises(MemoryError, match="streamed feeding step at order 2"):
        tc.features(data, order=2, cfun=tc.autofc, rfun={1: None, 2: "PCA"},
                    delta_rfun={1: None, 2: "weighted_strength"}, return_delta=True,
                    delta_weights_function=recording_kernel(calls),
                    delta_weights_params={"var": 25}, max_memory_gb=budget)
    assert calls == []


def test_direct_calls_keep_the_latest_fit_and_fit_appends():
    data = noise(T=100, K=6)
    corrs = tc.timecorr(data, cfun=tc.autofc, weights_params={"var": 10})
    ws = tc.WeightedStrength()
    for _ in range(3):
        quiet(ws, corrs)
    assert len(ws.fits_) == 1
    for _ in range(2):
        quiet(tc.reduce, corrs, rfun=ws)
    assert len(ws.fits_) == 1
    quiet(tc.timecorr, data, cfun=tc.autofc, weights_params={"var": 10}, rfun=ws)
    assert len(ws.fits_) == 1
    ws.fit(corrs)
    ws.fit(corrs, order=2, path="analysis")
    assert len(ws.fits_) == 3 and ws.fits_[-1]["order"] == 2
    comm = tc.Communities(n=2)
    comm(corrs)
    comm(corrs[0])
    assert len(comm.fits_) == 1 and len(comm.fits_[0]["weights"]) == 1


def test_two_path_step_appends_one_fit_per_step():
    data = noise(T=100, K=6)
    ws = tc.WeightedStrength()
    for k in (1, 2):
        quiet(two_path_step, data, cfun=tc.autofc, weights_params={"var": 10},
              rfun=ws, delta_rfun="PCA", order=k, **FEED)
    assert [(f["order"], f["path"]) for f in ws.fits_] == [
        (1, "analysis"), (2, "analysis")]


def test_order_map_missing_order_uses_auto_on_the_feeding_path():
    # delta_rfun={2: ...} leaves order 1 to "auto": "communities" (automatic
    # count) on the feeding path, while rfun is a plain reducer
    data, _ = nested()
    common = dict(order=2, cfun=tc.autofc, weights_params=WIDE, rfun="PCA",
                  return_delta=True, **FEED)
    X, D = quiet(tc.features, data, delta_rfun={2: "PCA"}, **common)
    comm = tc.Communities()
    X_ref, D_ref = quiet(tc.features, data, delta_rfun={1: comm, 2: "PCA"}, **common)
    assert D[1][0].shape == (T, comm.fits_[0]["n"]) and comm.fits_[0]["n"] == 6
    for k in (1, 2):
        assert_lists_close(D[k], D_ref[k], rtol=0, atol=0)
        assert_lists_close(X[k], X_ref[k], rtol=0, atol=0)


# review items 3 and 4: the shift null in the pipeline -----------------------


@pytest.mark.parametrize("feeding", ["gaussian", "eye"])
def test_n_null_reaches_the_shift_null_of_the_feeding_path(feeding):
    data, _ = nested()
    kw = FEED if feeding == "gaussian" else dict(delta_weights_function=tc.eye_weights)
    thresholds = []
    for m in (1, 7):
        comm = tc.Communities(n_null=m)
        quiet(tc.features, data, order=1, cfun=tc.autofc, weights_params=WIDE,
              rfun="PCA", delta_rfun=comm, return_delta=True, **kw)
        assert comm.fits_[0]["n_null"] == m
        thresholds.append(comm.fits_[0]["null_threshold"])
    assert thresholds[0] <= thresholds[1]


def test_more_features_than_shift_lags_runs():
    # T 40 gives 21 distinct lags in [10, 30] for 24 features: the shift
    # null reuses lags (each used once or twice); deterministic
    data, _ = nested(T=40)
    runs = []
    for _ in range(2):
        with pytest.warns(UserWarning, match="share a lag"):
            runs.append(tc.features(data, order=1, cfun=tc.autofc, weights_params=WIDE,
                                    rfun="PCA", delta_rfun=tc.Communities(),
                                    return_delta=True, **FEED))
    for a, b in zip(runs[0][1][1], runs[1][1][1]):
        assert a.shape[0] == 40 and np.all(np.isfinite(a))
        np.testing.assert_array_equal(a, b)


# review item 1: what the pipeline itself holds in a streamed step -----------


def test_distinct_event_kernels_are_budgeted_before_they_are_built():
    # 10 subjects with 10 distinct event sets: 10 T x T kernels (0.08 GB at
    # T 1000) are built for the streamed step; the budget leaves 0.04 GB
    # beyond one streamed row per block, so the step cannot hold them
    S, T_, K_ = 10, 1000, 4
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((T_, K_)) for _ in range(S)]
    events = [[100 + 10 * s] for s in range(S)]
    need = estimate_memory_gb(K_, 1, T_, S, streamed=True)[1]
    kernels_gb = S * T_ * T_ * 8 / 1e9
    calls = []
    common = dict(order=1, cfun=tc.autofc, rfun=tc.WeightedStrength(),
                  weights_function=recording_kernel(calls),
                  weights_params={"var": 25}, events=events)
    with pytest.raises(MemoryError, match="event kernels"):
        tc.features(data, max_memory_gb=need + kernels_gb / 2, **common)
    # the check ran before any kernel was built
    assert calls == []
    X = quiet(tc.features, data, max_memory_gb=need + 2 * kernels_gb, **common)
    assert len(X[1]) == S and X[1][0].shape == (T_, K_)


def test_distinct_event_kernels_of_the_streamed_feeding_step_are_budgeted():
    S, T_, K_ = 10, 1000, 4
    rng = np.random.default_rng(1)
    data = [rng.standard_normal((T_, K_)) for _ in range(S)]
    events = [[100 + 10 * s] for s in range(S)]
    need = estimate_memory_gb(K_, 1, T_, S, streamed=True)[1]
    kernels_gb = S * T_ * T_ * 8 / 1e9
    calls = []
    common = dict(order=1, cfun=tc.autofc, rfun="PCA", return_delta=True,
                  delta_rfun="weighted_strength",
                  delta_weights_function=recording_kernel(calls),
                  delta_weights_params={"var": 25}, events=events)
    with pytest.raises(MemoryError, match="event kernels"):
        tc.features(data, max_memory_gb=need + kernels_gb / 2, **common)
    assert calls == []


def test_streamed_feeding_counts_a_callable_analysis_output():
    # an identity callable keeps the unreduced (T, d(d+1)/2) correlations;
    # they are held while the streamed feeding step runs
    def identity(corrs):
        return corrs

    data = noise(S=2, T=2000, K=10)
    need, held = _held_sizes()
    common = dict(order=1, cfun=tc.autofc, rfun=identity, return_delta=True,
                  delta_rfun="weighted_strength", **FEED)
    assert estimate_memory_gb(10, 1, 2000, 2)[1] < need + held / 2  # analysis fits
    with pytest.raises(MemoryError, match="results held"):
        tc.features(data, max_memory_gb=need + held / 2, **common)
    X, D = tc.features(data, max_memory_gb=need + 2 * held, **common)
    assert X[1][0].shape == (2000, 55) and D[1][0].shape == (2000, 10)


def test_retained_fits_are_counted():
    # a summary used at both orders keeps the order-1 fit's (K, K) arrays per
    # unit while order 2 runs; they are counted against the budget
    from timecorr.higher_order import _fits_elements

    data, _ = nested()
    ws = tc.WeightedStrength(signed=True)
    quiet(tc.features, data, order=2, cfun=tc.autofc, weights_params=WIDE,
          rfun=ws, delta_rfun="weighted_strength", **FEED)
    expected = sum(w.size for f in ws.fits_ for w in f["weights"] + f["mean_r"])
    assert len(ws.fits_) == 2 and _fits_elements(ws) == expected
    assert _fits_elements("weighted_strength", None, "PCA") == 0


# review item 2: up-front checks use only widths known in advance -------------


def test_compress_then_expand_completes_under_a_budget_that_fits_every_step():
    # widths 24 -> 2 (Communities(n=2) on the feeding path) -> 3 (unreduced
    # order 2) -> weighted strength at order 3 on 3 inputs.  Substituting K
    # for the unknown width after the communities would give 24 -> 300 and
    # check order 3 on 300 inputs, which this budget does not fit
    data, _ = nested(T=2000)
    K_, T_ = 24, 2000
    first = estimate_memory_gb(K_, 1, T_, 2)[1]
    on_k = estimate_memory_gb(K_, 1, T_, 2, streamed=True)[1]
    on_guess = estimate_memory_gb(K_ * (K_ + 1) // 2, 1, T_, 2, streamed=True)[1]
    budget = 2 * max(first, on_k)
    assert budget < on_guess
    comm = tc.Communities(n=2)
    X = quiet(tc.features, data, order=3, cfun=tc.autofc, weights_params=WIDE,
              rfun={1: "PCA", 2: None, 3: tc.WeightedStrength()},
              delta_rfun={1: comm, 2: None}, max_memory_gb=budget, **FEED)
    assert comm.fits_[0]["n"] == 2
    assert X[2][0].shape == (T_, 3) and X[3][0].shape == (T_, 3)


def test_weighted_strength_width_is_known_in_advance():
    from timecorr.higher_order import _known_width

    assert _known_width(tc.WeightedStrength(), 7, None) == 7
    assert _known_width(tc.WeightedStrength(signed=True, split=True, n=2), 7, None) == 28
    assert _known_width("weighted_strength", 7, None) == 7
    assert _known_width("weighted_strength_split", 7, None) == 14
    assert _known_width(tc.Communities(n=2), 7, None) is None
    assert _known_width(lambda c: c, 7, None) is None


def test_up_front_check_uses_the_original_k_after_a_callable():
    # widths 24 -> 2 (PCA with ndims=2 on the feeding path at order 1); the
    # callable at order 2 receives 2 features, but the up-front check takes
    # the original K = 24 as its output width, so order 3 is checked on 24
    # inputs, and a budget that fits 2 inputs but not 24 raises there
    from timecorr.higher_order import _check_streamed_steps

    K_, T_, S_ = 24, 2000, 2
    on_two = estimate_memory_gb(2, 1, T_, S_, streamed=True)[1]
    on_k = estimate_memory_gb(K_, 1, T_, S_, streamed=True)[1]
    assert on_two < on_k
    analysis = {1: "PCA", 2: "PCA", 3: tc.WeightedStrength()}
    feeding = {1: "PCA", 2: lambda corrs: corrs, 3: "PCA"}
    common = dict(cfun=tc.autofc, combine=tc.null_combine,
                  feeding_function=tc.gaussian_weights, ndims=2)
    with pytest.raises(MemoryError, match=r"at order 3 .*24 input features"):
        _check_streamed_steps(K_, T_, S_, (on_two + on_k) / 2, analysis, feeding,
                              [1, 2], **common)
    _check_streamed_steps(K_, T_, S_, 2 * on_k, analysis, feeding, [1, 2], **common)


# Task 8 review: delta_weights_params alone without "var" ---------------------


@pytest.mark.parametrize("params", [{"scale": 5}, {}], ids=["scale", "empty"])
def test_default_gaussian_feeding_params_need_var(params):
    # gaussian_weights itself needs "var" (only None selects its defaults)
    with pytest.raises(KeyError):
        tc.gaussian_weights(10, params)
    data = noise()
    with pytest.raises(ValueError, match='default Gaussian feeding kernel needs "var"'):
        tc.features(data, order=1, cfun=tc.autofc, return_delta=True,
                    delta_rfun="weighted_strength", delta_weights_params=params)
    with pytest.raises(ValueError, match='default Gaussian feeding kernel needs "var"'):
        two_path_step(data, cfun=tc.autofc, delta_rfun="weighted_strength",
                      delta_weights_params=params)


# second review, F1: storage retained by the whole run ------------------------


def _long_run_data():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(40, 4)) for _ in range(2)]


LONG_RUN = dict(order=100, cfun=tc.autofc, delta_rfun="weighted_strength")


def _retained_bytes(outputs):
    seen, total = set(), 0
    for out in outputs:
        for a in (out if isinstance(out, list) else [out]):
            if id(a) not in seen:
                seen.add(id(a))
                total += a.nbytes
    return total


def test_outputs_kept_by_the_run_are_counted_against_the_budget():
    # every local streamed step fits 0.0003 GB, but the outputs the run keeps
    # (X_1 .. X_100) grow past it; the run raises instead of returning them
    data = _long_run_data()
    with pytest.raises(MemoryError, match=r"about .* GB") as info:
        quiet(tc.features, data, max_memory_gb=0.0003, **LONG_RUN)
    assert "held while it runs" in str(info.value) or "retained" in str(info.value)
    # with a budget that fits, it completes and the outputs exceed 0.0003 GB
    X = quiet(tc.features, data, max_memory_gb=0.01, **LONG_RUN)
    assert len(X) == 101
    assert _retained_bytes(X[1:]) > 0.0003 * 1e9


def test_feeding_outputs_not_returned_are_released():
    # return_delta=False keeps only the feeding output the next order needs:
    # the same X, and the run fits a budget that the returned D_1 .. D_k
    # exceed.  Smallest budgets that complete (bisection, order 30): 0.000331
    # GB without D and 0.000408 GB with D (D_1 .. D_30: 76,800 bytes)
    data = _long_run_data()
    common = dict(order=30, cfun=tc.autofc, delta_rfun="weighted_strength")
    X = quiet(tc.features, data, max_memory_gb=None, **common)
    X_d, D = quiet(tc.features, data, max_memory_gb=None, return_delta=True, **common)
    for a, b in zip(X, X_d):
        assert_lists_close(a, b, rtol=0, atol=0)
    assert _retained_bytes(D[1:]) == 76_800
    X_b = quiet(tc.features, data, max_memory_gb=0.00037, **common)
    # smaller blocks change the round-off, which 30 orders amplify (largest
    # relative difference measured: 1.2e-9)
    for a, b in zip(X, X_b):
        assert_lists_close(a, b, rtol=1e-6, atol=1e-9)
    with pytest.raises(MemoryError):
        quiet(tc.features, data, max_memory_gb=0.00037, return_delta=True, **common)


def test_retained_fits_are_enforced_by_the_budget():
    # two_path_step keeps the fits_ a summary object already holds; 50
    # earlier fits of a 24-node summary on 2 units retain 57,600 values,
    # counted against the budget of the step
    data, _ = nested(T=200)
    corrs = tc.timecorr(data, cfun=tc.autofc, weights_params=WIDE)
    S_, T_, K_ = 2, 200, 24
    need = estimate_memory_gb(K_, 1, T_, S_, streamed=True)[1]
    # the step alone: one streamed row per block, the analysis output held
    # while the feeding step runs, and this step's own fit
    base = need + 8 * (S_ * T_ * K_ + 2 * S_ * K_ * K_) / 1e9
    retained = 50 * S_ * K_ * K_
    common = dict(cfun=tc.autofc, weights_params=WIDE, delta_rfun="weighted_strength",
                  order=2)

    def prefitted():
        ws = tc.WeightedStrength()
        for _ in range(50):
            ws.fit(corrs)
        return ws

    quiet(two_path_step, data, rfun=tc.WeightedStrength(), max_memory_gb=base, **common)
    with pytest.raises(MemoryError, match="held while it runs"):
        quiet(two_path_step, data, rfun=prefitted(),
              max_memory_gb=base + 8 * retained / 2 / 1e9, **common)
    quiet(two_path_step, data, rfun=prefitted(),
          max_memory_gb=base + 8 * 2 * retained / 1e9, **common)


# re-review: a materialized step's own input is counted once -------------------


def test_materialized_step_input_is_not_counted_twice():
    # order 2 of the original pipeline on 2 x (80, 10): the step estimate
    # (0.0010496 GB) already counts its input D_1; with X_1 retained
    # (1.28e-5 GB) the run needs 0.0010624 GB.  The step estimate is
    # (80 * 10**2 + 2 * 80 * 55 + 3 * 2 * 80 * 10 + 3 * 80**2 + 3 * 80 * 10
    #  + max(7 * 10 * 80 * 10, 11 * 10**2 * 80)) * 8 / 1e9: the Fisher z
    # temporaries of two subjects are larger than the 7 estimator blocks
    rng = np.random.default_rng(0)
    data = [rng.normal(size=(80, 10)) for _ in range(2)]
    common = dict(order=2, rfun="PCA", delta_rfun=None,
                  delta_weights_function=tc.eye_weights)
    # PCA keeps 10 columns, so step 2 is a one-output step on 10 inputs
    step = estimate_memory_gb(10, 1, 80, 2, delta_weights_function=tc.eye_weights)[1]
    x1 = 2 * 80 * 10 * 8 / 1e9
    assert step == pytest.approx(0.0010496) and x1 == pytest.approx(1.28e-5)
    X = quiet(tc.features, data, max_memory_gb=0.0010688, **common)
    assert len(X) == 3
    with pytest.raises(MemoryError, match="retained by the run"):
        quiet(tc.features, data, max_memory_gb=step + x1 - 1e-7, **common)


def test_views_of_one_array_are_counted_once():
    from timecorr.higher_order import _output_elements

    a = np.zeros((10, 4))
    assert _output_elements([a, a[:5], a[:, :2]]) == 40
    assert _output_elements([a[:5], a[5:]]) == 40
    assert _output_elements([a, np.zeros(3)]) == 43


# ------------------------------------------------ PageRank on |r| as a reducer


@pytest.mark.parametrize("path", ["rfun", "delta_rfun"])
def test_abs_pagerank_works_as_a_reducer_in_features(path):
    data = noise(S=2, T=80, K=5, seed=4)
    kwargs = {"rfun": "PCA", "delta_rfun": "weighted_strength"}
    kwargs[path] = "pagerank_centrality_abs"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X, D = tc.features(data, order=2, return_delta=True, **kwargs)
    assert len(X) == 3 and len(D) == 3
    for k in (1, 2):
        for x in X[k] + D[k]:
            assert x.shape == (80, 5) and np.all(np.isfinite(x))
    # one PageRank value per node, summing to 1
    pageranked = X[1] + X[2] if path == "rfun" else D[1] + D[2]
    for x in pageranked:
        assert np.all(x > 0)
        np.testing.assert_allclose(x.sum(axis=1), 1.0, rtol=0, atol=1e-12)
