"""
The feeding path's kernel, per-order reducers and streamed summaries in
``features`` and ``two_path_step`` (opt-in arguments; the defaults are those
of the legacy pipeline, pinned by ``tests/test_legacy_pipeline.py``).

Everything is computed for real on simulated data; nothing is mocked.
"""

import time
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


def test_eye_feeding_kernel_is_the_default():
    data = noise()
    common = dict(order=2, cfun=tc.autofc, weights_params={"var": 10},
                  return_delta=True)
    X0, D0 = tc.features(data, **common)
    X1, D1 = tc.features(data, delta_weights_function=tc.eye_weights, **common)
    for k in range(3):
        assert_lists_close(X0[k], X1[k], rtol=0, atol=0)
        assert_lists_close(D0[k], D1[k], rtol=0, atol=0)


def test_feeding_kernel_params_none_uses_the_function_defaults():
    data = noise()
    common = dict(order=1, cfun=tc.autofc, rfun=None, return_delta=True,
                  delta_weights_function=tc.laplace_weights)
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
    with pytest.raises(ValueError, match="number of communities n"):
        tc.features(data, order=1, cfun=tc.isfc, weights_params=WIDE,
                    rfun=tc.Communities())


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
                    weights_params={"var": 10}, rfun="PCA")


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


@pytest.mark.parametrize("path", ["analysis", "feeding"])
def test_streamed_budget_too_small_raises_before_computing(path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2000, 2000))
    need = estimate_memory_gb(2000, 1, 2000, streamed=True)[1]
    kw = (dict(rfun=tc.WeightedStrength(signed=True)) if path == "analysis"
          else dict(rfun=tc.WeightedStrength(), delta_rfun="weighted_strength",
                    return_delta=True, **FEED))
    start = time.perf_counter()
    with pytest.raises(MemoryError, match="streamed"):
        tc.features(data, order=1, cfun=tc.autofc, max_memory_gb=0.9 * need, **kw)
    assert time.perf_counter() - start < 2.0


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


def test_default_feeding_constant_selects_the_gaussian_kernel(monkeypatch):
    # the one switch for the later change of default: with "gaussian", None
    # means gaussian_weights with {"var": 25}
    from timecorr import higher_order

    data = noise(T=100, K=5)
    common = dict(order=1, cfun=tc.autofc, rfun=None, return_delta=True)
    _, explicit = tc.features(data, **common, **FEED)
    monkeypatch.setattr(higher_order, "_DEFAULT_FEEDING", "gaussian")
    _, default = tc.features(data, **common)
    assert_lists_close(default[1], explicit[1], rtol=0, atol=0)
    monkeypatch.setattr(higher_order, "_DEFAULT_FEEDING", "delta")
    _, eye = tc.features(data, **common)
    assert_lists_close(
        eye[1], tc.timecorr(data, weights_function=tc.eye_weights, cfun=tc.autofc),
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
    data = noise(S=2, T=2000, K=10)
    need, held = _held_sizes()
    common = dict(order=1, cfun=tc.autofc, rfun="PCA", return_delta=True,
                  delta_rfun=tc.Communities())
    start = time.perf_counter()
    with pytest.raises(MemoryError, match="shift null of the feeding step"):
        tc.features(data, max_memory_gb=need + held / 2, **common)
    assert time.perf_counter() - start < 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, D = tc.features(data, max_memory_gb=need + 2 * held + 0.2, **common)
    assert len(D[1]) == 2


def test_delta_weights_params_needs_delta_weights_function():
    data = noise()
    with pytest.raises(ValueError, match="delta_weights_params needs delta_weights_function"):
        tc.features(data, order=1, cfun=tc.autofc, delta_weights_params={"var": 25})
    with pytest.raises(ValueError, match="delta_weights_params needs delta_weights_function"):
        two_path_step(data, cfun=tc.autofc, delta_weights_params={"var": 25})
