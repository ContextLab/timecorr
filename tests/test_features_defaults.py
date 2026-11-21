"""
Defaults of the higher-order pipeline (``features`` and ``two_path_step``).

The defaults are ``rfun="auto"`` (PCA at order 1, signed weighted strength at
orders >= 2), ``delta_rfun="auto"`` (communities) and Gaussian kernels whose
widths are set from each step's input when their parameters are None
(``tests/test_features_auto_kernels.py``; on white data of up to 500
timepoints these are the fixed kernels of variance 100 and 25).  Tests that
need the fixed kernels where the automatic ones differ give them
(``FIXED_FEEDING``, ``FIXED_KERNELS``).  The original pipeline is
``rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights``
(pinned by ``tests/test_legacy_pipeline.py``).

Everything is computed for real on simulated data; nothing is mocked.
"""

import inspect
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.higher_order import two_path_step

LEGACY = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
EXPLICIT = dict(rfun="auto", delta_rfun="auto",
                delta_weights_function=tc.gaussian_weights,
                delta_weights_params={"var": 25})
SETTINGS = dict(cfun=tc.autofc, weights_params={"var": 100})
# the fixed kernels (Gaussian, variance 100 and 25 at every order) that were
# the defaults before kernel parameters of None became automatic
FIXED_FEEDING = dict(delta_weights_params={"var": 25})
FIXED_KERNELS = dict(weights_params={"var": 100}, **FIXED_FEEDING)


def nested(S=2, T=300, seed=5):
    """24 nodes in 4 modules (2 supergroups x 2 modules x 6 nodes)."""
    data, _ = tc.simulate_nested(
        S=S, T=T, n_supergroups=2, modules_per_supergroup=2, nodes_per_module=6,
        set_random_seed=seed,
    )
    return data


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return f(*args, **kwargs)


def assert_identical(a, b):
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("function", [tc.features, two_path_step])
def test_signature_defaults(function):
    params = inspect.signature(function).parameters
    assert params["rfun"].default == "auto"
    assert params["delta_rfun"].default == "auto"
    assert params["delta_weights_function"].default is None
    assert params["delta_weights_params"].default is None


def test_default_call_equals_the_explicit_call():
    x = nested()
    X, D = quiet(tc.features, x, order=2, return_delta=True, **SETTINGS)
    Xe, De = quiet(tc.features, x, order=2, return_delta=True, **SETTINGS, **EXPLICIT)
    for k in range(3):
        assert_identical(X[k], Xe[k])
        assert_identical(D[k], De[k])
    # signed weighted strength at order 2: two columns per community column
    assert X[2][0].shape[1] == 2 * D[1][0].shape[1]


def test_step_default_equals_the_explicit_step():
    x = nested()
    X1, D1 = quiet(two_path_step, x, **SETTINGS)
    Xe, De = quiet(two_path_step, x, **SETTINGS, **EXPLICIT)
    assert_identical(X1, Xe)
    assert_identical(D1, De)
    X2, D2 = quiet(two_path_step, D1, order=2, **SETTINGS)
    Xe2, De2 = quiet(two_path_step, D1, order=2, **SETTINGS, **EXPLICIT)
    assert_identical(X2, Xe2)
    assert_identical(D2, De2)


def test_order_one_output_is_unchanged():
    x = nested()
    X = tc.features(x, order=1, **SETTINGS)
    legacy = tc.features(x, order=1, **SETTINGS, **LEGACY)
    assert_identical(X[1], legacy[1])
    assert_identical(X[1], tc.timecorr(x, rfun="PCA", **SETTINGS))
    # also with isfc, the default cfun
    X = tc.features(x, order=1)
    assert_identical(X[1], tc.features(x, order=1, **LEGACY)[1])


def test_delta_weights_params_alone_selects_a_gaussian():
    x = nested()
    _, D = quiet(tc.features, x, order=1, return_delta=True, rfun=None,
                 delta_rfun=None, delta_weights_params={"var": 9}, **SETTINGS)
    expected = tc.timecorr(x, cfun=tc.autofc, weights_function=tc.gaussian_weights,
                           weights_params={"var": 9})
    np.testing.assert_allclose(np.stack(D[1]), np.stack(expected), rtol=1e-12, atol=1e-12)
    _, step = quiet(two_path_step, x, rfun=None, delta_rfun=None,
                    delta_weights_params={"var": 9}, **SETTINGS)
    assert_identical(step, D[1])


def test_explicit_eye_weights_with_params_still_raises():
    x = nested()
    with pytest.raises(ValueError, match="eye_weights takes no parameters"):
        tc.features(x, order=1, delta_weights_function=tc.eye_weights,
                    delta_weights_params={"var": 9})
    with pytest.raises(ValueError, match="eye_weights takes no parameters"):
        two_path_step(x, delta_weights_function=tc.eye_weights,
                      delta_weights_params={"var": 9})


@pytest.mark.parametrize("shape", [(300, 3), (30, 6)], ids=["K3", "T30"])
def test_default_order_two_needs_four_features_and_forty_timepoints(shape):
    rng = np.random.default_rng(0)
    x = [rng.standard_normal(shape) for _ in range(2)]
    for call in (
        lambda: tc.features(x, order=2, weights_params={"var": 10}),
        lambda: two_path_step(x, weights_params={"var": 10}),
    ):
        with pytest.raises(ValueError) as info:
            call()
        message = str(info.value)
        assert 'delta_rfun="weighted_strength"' in message
        assert 'rfun="PCA", delta_rfun=None' in message
        assert "delta_weights_function=tc.eye_weights" in message
    # both alternatives named in the message run
    X = quiet(tc.features, x, order=2, weights_params={"var": 10},
              delta_rfun="weighted_strength")
    assert X[2][0].shape == (shape[0], 2 * shape[1])
    X = tc.features(x, order=2, weights_params={"var": 10}, **LEGACY)
    assert X[2][0].shape == (shape[0], shape[1])


def test_default_order_one_without_the_feeding_step_takes_any_shape():
    rng = np.random.default_rng(0)
    x = [rng.standard_normal((30, 3)) for _ in range(2)]
    X = tc.features(x, order=1, weights_params={"var": 10})
    assert X[1][0].shape == (30, 3)


def test_ndims_with_the_default_order_two_pipeline_raises():
    x = nested()
    with pytest.raises(ValueError, match="ndims=2 applies to model-based reduction only"):
        tc.features(x, order=2, ndims=2, **SETTINGS)
    # order 1 with the analysis path only keeps working
    X = tc.features(x, order=1, ndims=2, **SETTINGS)
    assert X[1][0].shape == (300, 2)


def test_legacy_recipe_is_the_original_pipeline():
    # the legacy arguments give eye_weights co-fluctuations on the feeding path
    x = nested()
    _, D = tc.features(x, order=1, return_delta=True, rfun=None, cfun=tc.autofc,
                       delta_rfun=None, delta_weights_function=tc.eye_weights)
    expected = tc.timecorr(x, weights_function=tc.eye_weights, cfun=tc.autofc)
    assert_identical(D[1], expected)


@pytest.mark.parametrize("cfun", [tc.isfc, tc.autofc], ids=["isfc", "autofc"])
def test_memory_estimate_counts_the_kernel_feeding_path(cfun):
    # unreduced, both paths materialized: the Gaussian feeding path has block
    # temporaries of its own, so its estimate is larger than eye_weights'
    import tracemalloc

    from timecorr.higher_order import estimate_memory_gb

    K, T, S = 100, 200, 3
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((T, K)) for _ in range(S)]
    est = estimate_memory_gb(K, 1, T, S, return_delta=True)[1]
    eye = estimate_memory_gb(K, 1, T, S, return_delta=True,
                             delta_weights_function=tc.eye_weights)[1]
    assert est > eye
    tracemalloc.start()
    try:
        quiet(tc.features, data, order=1, rfun=None, delta_rfun=None, cfun=cfun,
              weights_params={"var": 100}, return_delta=True, max_memory_gb=None)
        peak = tracemalloc.get_traced_memory()[1] / 1e9
    finally:
        tracemalloc.stop()
    assert est / 1.25 <= peak <= 1.25 * est, (peak, est)
    # the guard uses the same estimate: 0.1795 GB (estimate_memory_gb's
    # formula: 7 block temporaries of 100 x 200 x 100 elements, plus the
    # Gaussian feeding path's results and the (S + 1) T d stacked subjects),
    # printed to 3 significant digits
    with pytest.raises(MemoryError, match="Unreduced features at order 1 need about 0.18 GB"):
        tc.features(data, order=1, rfun=None, delta_rfun=None, cfun=cfun,
                    weights_params={"var": 100}, return_delta=True,
                    max_memory_gb=(est + eye) / 2)


# second review, F2: the default pipeline after community compression ---------


def _feeding(info, k):
    """The feeding entry of order k in features' return_info."""
    (entry,) = [e for e in info if e["order"] == k and e["path"] == "feeding"]
    return entry


def _auto_warning(message, k):
    return message.startswith(f"Automatic feeding reducer at order {k}")


def _alternatives_named(message):
    assert 'delta_rfun="weighted_strength"' in message
    assert 'rfun="PCA", delta_rfun=None' in message
    assert "delta_weights_function=tc.eye_weights" in message


@pytest.mark.parametrize("kw", [dict(order=3), dict(order=2, return_delta=True)],
                         ids=["order3", "order2-return-delta"])
def test_default_feeding_after_community_compression_keeps_the_width(kw):
    # 4 features: communities would keep at most 4 // 2 = 2 columns, so every
    # automatic feeding step emits weighted strength of its 4 input features
    # instead (recorded in info; a warning only where the shift null found
    # no structure), and the run completes
    x = np.random.default_rng(0).normal(size=(100, 4))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, info = tc.features(x, return_info=True, **kw)
    fed = range(1, kw["order"] + (1 if kw.get("return_delta") else 0))
    for k in fed:
        entry = _feeding(info, k)
        assert entry["fallback"] in ("too_few_community_columns", "no_null_exceedance")
        assert entry["output_width"] == 4
        warned = sum(_auto_warning(str(r.message), k) for r in rec)
        assert warned == (entry["fallback"] == "no_null_exceedance"), k
    if kw.get("return_delta"):
        X, D = out
        assert [d.shape for d in D] == [(100, 4)] * 3
    else:
        X = out
    assert X[-1].shape == (100, 8)


def test_default_order_two_without_the_last_feeding_step_completes():
    x = np.random.default_rng(0).normal(size=(100, 4))
    X = quiet(tc.features, x, order=2)
    assert len(X) == 3


def test_alternatives_complete_after_community_compression():
    x = np.random.default_rng(0).normal(size=(100, 4))
    X = quiet(tc.features, x, order=3, delta_rfun="weighted_strength")
    assert X[3].shape == (100, 8)
    X = quiet(tc.features, x, order=3, **LEGACY)
    assert X[3].shape == (100, 4)


def test_default_feeding_runs_with_more_features_than_shift_lags():
    # T 40, standard deviation 5: L 10, so 21 distinct lags in [10, 30] for
    # 22 features: the shift null reuses lags (each used once or twice)
    # and the result is deterministic
    x = [np.random.default_rng(1).normal(size=(40, 22)) for _ in range(2)]
    with pytest.warns(UserWarning, match="share a lag"):
        first = tc.features(x, order=1, return_delta=True, weights_params={"var": 10})
    again = quiet(tc.features, x, order=1, return_delta=True,
                  weights_params={"var": 10})
    for part in (0, 1):
        for a, b in zip(first[part][1], again[part][1]):
            np.testing.assert_array_equal(a, b)
    assert [d.shape[0] for d in first[1][1]] == [40, 40]
    X = quiet(tc.features, [a[:, :21] for a in x], order=1, return_delta=True,
              weights_params={"var": 10})
    assert len(X) == 2


def test_defaults_run_with_more_features_than_shift_lags():
    # 2 subjects, T 120, K 150 (default feeding kernel: L 30, 61 distinct
    # lags in [30, 90]): this raised ValueError before lags could be reused
    x = [np.random.default_rng(s).standard_normal((120, 150)) for s in range(2)]
    with pytest.warns(UserWarning, match="share a lag"):
        X, D = tc.features(x, order=2, return_delta=True)
    assert len(X) == 3 and len(D) == 3
    for k in (1, 2):
        assert len(X[k]) == 2 and len(D[k]) == 2
        for a in X[k] + D[k]:
            assert a.shape[0] == 120 and np.all(np.isfinite(a))
    # order 1: PCA to as many features as the data have
    assert all(a.shape == (120, 150) for a in X[1])
    # order 2: WeightedStrength(signed=True), two columns per feeding column
    assert all(a.shape == (120, 2 * D[1][0].shape[1]) for a in X[2])
    X_again = quiet(tc.features, x, order=2)
    for a, b in zip(X[2], X_again[2]):
        np.testing.assert_array_equal(a, b)


# wave 4, D1: the automatic feeding reducer keeps the width -------------------

FALLBACK = dict(split=False, signed=False, center=False)
GAUSS25 = dict(weights_function=tc.gaussian_weights, weights_params={"var": 25})


def noise(S, shape, seed):
    """S subjects of independent standard normal noise of the given shape."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape) for _ in range(S)]


def recorded(f, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = f(*args, **kwargs)
    return out, [str(r.message) for r in rec]


def auto_messages(messages, k):
    return [m for m in messages if _auto_warning(m, k)]


def nested_modules(sg, per, nodes, S=2, T=300, seed=0):
    data, _ = tc.simulate_nested(S=S, T=T, n_supergroups=sg, modules_per_supergroup=per,
                                 nodes_per_module=nodes, set_random_seed=seed)
    return data


def assert_close(a, b, rtol=1e-9, atol=1e-12):
    # the tolerance of the streamed-versus-materialized tests
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize("K", [4, 5, 6, 7])
def test_four_to_seven_features_complete_with_the_width_kept(K):
    # communities can emit at most K // 2 <= 3 columns, so every automatic
    # feeding step falls back to weighted strength of its K inputs
    x = noise(2, (100, K), K)
    (X, info), messages = recorded(tc.features, x, order=3, return_info=True)
    assert X[3][0].shape == (100, 2 * K)
    for k in (1, 2):
        entry = _feeding(info, k)
        assert entry["fallback"] is not None and entry["n_communities"] < 4
        assert (entry["input_width"], entry["output_width"]) == (K, K)
        found = auto_messages(messages, k)
        assert len(found) == (entry["fallback"] == "no_null_exceedance"), (k, messages)
        for m in found:
            assert m.endswith(f"uses WeightedStrength on its {K} input features.")
    assert not auto_messages(messages, 3)  # the order-3 feeding step is not computed
    assert not [e for e in info if e["order"] == 3 and e["path"] == "feeding"]
    (X, D), messages = recorded(tc.features, x, order=2, return_delta=True)
    assert [d[0].shape for d in D] == [(100, K)] * 3


@pytest.mark.parametrize("modules, columns", [((1, 2, 6), 2), ((1, 3, 6), 3)],
                         ids=["2-columns", "3-columns"])
@pytest.mark.parametrize("feeding", ["gaussian", "eye"])
def test_two_or_three_community_columns_fall_back(modules, columns, feeding):
    # 2 or 3 modules of 6 nodes: the communities would emit 2 or 3 columns;
    # the step emits weighted strength of the same correlations instead,
    # equal to the materialized computation (streamed with the Gaussian
    # feeding kernel; materialized with eye_weights)
    data = nested_modules(*modules)
    K = data[0].shape[1]
    kw = {} if feeding == "gaussian" else dict(delta_weights_function=tc.eye_weights)
    ((X, D), info), messages = recorded(tc.features, data, order=1, return_delta=True,
                                        return_info=True, **SETTINGS, **kw)
    # communities were found: the fallback is silent and recorded in info
    assert auto_messages(messages, 1) == []
    entry = _feeding(info, 1)
    assert entry["fallback"] == "too_few_community_columns"
    assert entry["n_communities"] == columns and entry["n_exceed"] >= 1
    if feeding == "gaussian":
        corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None, **GAUSS25)
    else:
        corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None,
                            weights_function=tc.eye_weights)
    expected = quiet(tc.WeightedStrength(**FALLBACK), corrs)
    assert D[1][0].shape == (300, K)
    assert_close(D[1], expected)
    # the explicit reducer keeps its communities
    comm = tc.Communities()
    _, D_explicit = quiet(tc.features, data, order=1, return_delta=True,
                          delta_rfun=comm, **SETTINGS, **kw)
    assert D_explicit[1][0].shape == (300, columns) and comm.fits_[0]["n"] == columns


def test_one_community_column_after_an_exceedance_falls_back():
    # 4 noise features (seed 6): one eigenvalue above the null gives two
    # clusters, one of them a single node, so the communities emit 1 column
    x = np.random.default_rng(6).normal(size=(100, 4))
    ((X, D), info), messages = recorded(tc.features, x, order=1, return_delta=True,
                                        return_info=True)
    entry = _feeding(info, 1)
    assert entry["fallback"] == "too_few_community_columns"
    assert entry["n_communities"] == 1 and entry["n_exceed"] >= 1
    assert auto_messages(messages, 1) == []
    corrs = tc.timecorr(x, cfun=tc.autofc, rfun=None, **GAUSS25)
    assert_close(D[1], quiet(tc.WeightedStrength(**FALLBACK), corrs))


def test_no_exceedance_on_noise_falls_back():
    # independent noise: no eigenvalue above the shift null; the automatic
    # reducer says so and keeps the width, without the explicit reducer's
    # "one community" warning
    x = noise(2, (200, 10), 7)  # no exceedance with this seed (5 of seeds 0-7)
    ((X, D), info), messages = recorded(tc.features, x, order=1, return_delta=True,
                                        return_info=True)
    found = auto_messages(messages, 1)
    assert len(found) == 1
    assert found[0].startswith("Automatic feeding reducer at order 1: the shift null "
                               "found no structure (no eigenvalue above null_threshold=")
    assert found[0].endswith("); this order uses WeightedStrength on its 10 input "
                             "features.")
    entry = _feeding(info, 1)
    assert entry["fallback"] == "no_null_exceedance"
    assert entry["n_exceed"] == 0 and entry["n_communities"] == 1
    assert not any("no community structure" in m for m in messages)
    assert D[1][0].shape == (200, 10)
    corrs = tc.timecorr(x, cfun=tc.autofc, rfun=None, **GAUSS25)
    assert_close(D[1], quiet(tc.WeightedStrength(**FALLBACK), corrs))
    # the explicit reducer keeps its warning and its one community
    comm = tc.Communities()
    (_, D_explicit), messages = recorded(tc.features, x, order=1, return_delta=True,
                                         delta_rfun=comm)
    assert any("no community structure above the shift null" in m for m in messages)
    assert not auto_messages(messages, 1)
    assert D_explicit[1][0].shape == (200, 1)


@pytest.mark.parametrize("modules", [(2, 2, 6), (1, 4, 4), (2, 3, 4)],
                         ids=["4-modules", "4-modules-of-4", "6-modules"])
def test_four_or_more_community_columns_are_unchanged(modules):
    # at least 4 columns: the automatic reducer emits the communities,
    # identical to an explicit Communities(), without the fallback warning
    data = nested_modules(*modules)
    ((X, D), info), messages = recorded(tc.features, data, order=1, return_delta=True,
                                        return_info=True, **SETTINGS)
    assert not auto_messages(messages, 1)
    comm = tc.Communities()
    _, D_explicit = quiet(tc.features, data, order=1, return_delta=True,
                          delta_rfun=comm, **SETTINGS)
    assert comm.fits_[0]["n"] >= 4
    entry = _feeding(info, 1)
    assert entry["fallback"] is None and entry["n_communities"] == comm.fits_[0]["n"]
    assert entry["effective"] == repr(tc.Communities())
    assert_identical(D[1], D_explicit[1])


def test_explicit_communities_still_raise_after_compression():
    # explicit Communities() keep their behavior: order 1 on 4 features keeps
    # at most 2 columns, and order 2 then raises (K >= 4 needed)
    x = np.random.default_rng(0).normal(size=(100, 4))
    with pytest.raises(ValueError, match="needs K >= 4 nodes"):
        quiet(tc.features, x, order=2, return_delta=True, delta_rfun=tc.Communities())
    # explicit Communities at order 1, the automatic reducer at order 2: the
    # error names the communities found at order 1
    with pytest.raises(ValueError, match="at order 2") as info:
        quiet(tc.features, x, order=2, return_delta=True,
              delta_rfun={1: tc.Communities()})
    _alternatives_named(str(info.value))
    assert "communities found at order 1" in str(info.value)


@pytest.mark.parametrize("shape, match", [((100, 3), "got 3 features"),
                                          ((39, 6), "T = 39")],
                         ids=["K3", "T39"])
def test_original_input_errors_stay(shape, match):
    # the original input: fewer than 4 features or T < 40 raise at order 1,
    # before anything is computed, whenever a feeding step is required
    x = noise(2, shape, 2)
    for kw in (dict(order=3), dict(order=2), dict(order=1, return_delta=True)):
        with pytest.raises(ValueError, match=match) as info:
            tc.features(x, weights_params={"var": 10}, **kw)
        assert "at order 1" in str(info.value)
        _alternatives_named(str(info.value))
    # without a feeding step, order 1 is unaffected
    assert tc.features(x, order=1, weights_params={"var": 10})[1][0].shape[0] == shape[0]


def test_original_input_with_more_features_than_shift_lags_runs():
    # T 40, 22 features (21 distinct shift-null lags in [10, 30]) raised
    # at order 1 when lags could not be reused; every order now runs, and
    # the result is deterministic
    x = noise(2, (40, 22), 2)
    for kw in (dict(order=3), dict(order=2), dict(order=1, return_delta=True)):
        with pytest.warns(UserWarning, match="share a lag"):
            first = tc.features(x, weights_params={"var": 10}, **kw)
        again = quiet(tc.features, x, weights_params={"var": 10}, **kw)
        X = first[0] if kw.get("return_delta") else first
        X_again = again[0] if kw.get("return_delta") else again
        for k in range(1, kw["order"] + 1):
            for a, b in zip(X[k], X_again[k]):
                assert a.shape[0] == 40
                np.testing.assert_array_equal(a, b)


def test_one_no_exceedance_warning_per_order_not_per_subject():
    # 4 subjects; the warning is issued once per order whose shift null
    # found no structure, and never otherwise
    x = noise(4, (200, 10), 7)
    (_, info), messages = recorded(tc.features, x, order=3, return_delta=True,
                                   return_info=True)
    reasons = [_feeding(info, k)["fallback"] for k in (1, 2, 3)]
    assert "no_null_exceedance" in reasons
    assert [len(auto_messages(messages, k)) for k in (1, 2, 3)] == [
        int(r == "no_null_exceedance") for r in reasons]
    _, messages = recorded(two_path_step, x)
    assert len(auto_messages(messages, 1)) == int(reasons[0] == "no_null_exceedance")


def test_prefix_consistency_across_requested_orders():
    # the orders a shorter run computes are the same in a longer run
    x = noise(2, (120, 6), 3)
    X2, D2 = quiet(tc.features, x, order=2, return_delta=True)
    X3, D3 = quiet(tc.features, x, order=3, return_delta=True)
    X3_only = quiet(tc.features, x, order=3)
    for k in range(3):
        assert_identical(X2[k], X3[k])
        assert_identical(D2[k], D3[k])
        assert_identical(X2[k], X3_only[k])
    assert_identical(quiet(tc.features, x, order=2)[2], X2[2])


def test_memory_accounting_after_a_fallback():
    # 6 noise features, order 3: every automatic feeding step keeps 6
    # columns.  The smallest budget that completes (bisection from [0, 0.5]
    # GB until the bounds are adjacent floats, with the built-in kernels read
    # in chunks of rows): 0.090026624 GB, of which 0.00077 GB are held while
    # the order-3 streamed analysis step runs on the 6 fallback columns (X_1,
    # X_2 and D_2: 96,000 values, with the fits); just below it, that step
    # raises before it computes anything (measured: "The streamed analysis
    # step at order 3 needs at least about 0.09 GB"; the match below accepts a MemoryError
    # at order 2 or 3 on the 6 input features, so it also holds if a change
    # of accounting moves the refusal one order earlier)
    # FIXED_KERNELS: every order at every timepoint, as measured (with
    # kernel parameters of None, orders 2 and 3 of a recording of 2000
    # timepoints are computed at every 2nd timepoint and need less)
    x = noise(2, (2000, 6), 4)
    budget = 0.090026624
    X = quiet(tc.features, x, order=3, max_memory_gb=budget, **FIXED_KERNELS)
    assert X[3][0].shape == (2000, 12)
    with pytest.raises(MemoryError, match=r"at order [23] .*6 input features"):
        quiet(tc.features, x, order=3, max_memory_gb=budget * (1 - 1e-6),
              **FIXED_KERNELS)


def test_two_nodes_give_identical_weighted_strength_columns():
    # with 2 features the weighted strength of each node is the same pair's
    # a_12 r_12(t)**2, so the two columns are identical
    x = noise(2, (100, 2), 5)
    _, D = quiet(tc.features, x, order=1, return_delta=True,
                 delta_rfun="weighted_strength")
    for d in D[1]:
        assert d.shape == (100, 2)
        np.testing.assert_allclose(d[:, 0], d[:, 1], rtol=1e-12, atol=0)
        assert np.ptp(d[:, 0]) > 0


# return_info ------------------------------------------------------------------

ANALYSIS_KEYS = {"order", "path", "requested", "effective", "input_width", "output_width"}
AUTO_KEYS = ANALYSIS_KEYS | {"n_communities", "n_exceed", "null_threshold", "n_null",
                             "fallback"}


def _no_arrays(info):
    for entry in info:
        for key, value in entry.items():
            assert isinstance(key, str)
            assert value is None or type(value) in (int, float, str), (key, value)


def test_return_info_shapes_of_the_return():
    x = noise(2, (100, 6), 3)
    X = quiet(tc.features, x, order=2)
    X_i, info = quiet(tc.features, x, order=2, return_info=True)
    for a, b in zip(X, X_i):
        assert_identical(a, b)
    (X_d, D), info_d = quiet(tc.features, x, order=2, return_delta=True,
                             return_info=True)
    assert len(X_d) == 3 and len(D) == 3
    # return_delta adds the feeding entry of the last order
    assert [(e["order"], e["path"]) for e in info] == [
        (1, "analysis"), (1, "feeding"), (2, "analysis")]
    assert [(e["order"], e["path"]) for e in info_d] == [
        (1, "analysis"), (1, "feeding"), (2, "analysis"), (2, "feeding")]
    X0, info0 = tc.features(x, order=0, return_info=True)
    assert info0 == [] and len(X0) == 1
    _no_arrays(info_d)


def test_return_info_content():
    # FIXED_FEEDING in both calls: entries of automatic kernels also have
    # "kernel_sd" and "sampling" (tests/test_features_auto_kernels.py)
    data = nested_modules(2, 2, 6)  # 24 nodes in 4 modules
    ws = tc.WeightedStrength(signed=True)
    (X, D), info = quiet(tc.features, data, order=2, return_delta=True,
                         return_info=True, rfun={2: ws}, **SETTINGS, **FIXED_FEEDING)
    _no_arrays(info)
    a1, f1, a2, f2 = info
    assert set(a1) == ANALYSIS_KEYS and set(a2) == ANALYSIS_KEYS
    assert set(f1) == AUTO_KEYS and set(f2) == AUTO_KEYS
    assert a1 == {"order": 1, "path": "analysis", "requested": "auto",
                  "effective": "PCA", "input_width": 24, "output_width": 24}
    assert a2["requested"] == repr(ws) and a2["effective"] == repr(ws)
    assert (a2["input_width"], a2["output_width"]) == (D[1][0].shape[1], X[2][0].shape[1])
    assert f1["requested"] == "auto" and f1["input_width"] == 24
    assert f1["output_width"] == D[1][0].shape[1]
    assert f1["n_null"] == 5 and f1["null_threshold"] > 0
    assert f2["input_width"] == D[1][0].shape[1] and f2["output_width"] == D[2][0].shape[1]
    # explicit reducers: no automatic-reducer keys
    (_, D), info = quiet(tc.features, data, order=1, return_delta=True, return_info=True,
                         rfun=None, delta_rfun="weighted_strength", **SETTINGS,
                         **FIXED_FEEDING)
    assert info[0]["requested"] is None and info[0]["effective"] is None
    assert info[0]["output_width"] == 24 * 25 // 2
    assert set(info[1]) == ANALYSIS_KEYS
    assert info[1]["requested"] == "weighted_strength"


def test_return_info_records_the_three_fallback_values():
    found = {}
    for data in (nested_modules(2, 2, 6), nested_modules(1, 2, 6), noise(2, (200, 10), 7)):
        (_, D), info = quiet(tc.features, data, order=1, return_delta=True,
                             return_info=True, **SETTINGS)
        entry = _feeding(info, 1)
        found[entry["fallback"]] = entry
        if entry["fallback"] is None:
            assert entry["output_width"] == entry["n_communities"] >= 4
        else:
            assert entry["output_width"] == entry["input_width"]
            assert entry["effective"] == repr(tc.WeightedStrength(**FALLBACK))
    assert set(found) == {None, "too_few_community_columns", "no_null_exceedance"}
    assert found["no_null_exceedance"]["n_exceed"] == 0
    assert found["too_few_community_columns"]["n_exceed"] >= 1
    assert found["too_few_community_columns"]["n_communities"] == 2


# wave 4, D6: the MemoryError of a default order-1 step names streaming --------


def test_default_order_one_memory_error_names_a_streamed_reducer():
    # the default order-1 analysis step (PCA) holds its correlations; under a
    # budget between its estimate and that of a streamed step, the error
    # names the summary reducers that stream it, and both suggestions run
    from timecorr.higher_order import estimate_memory_gb

    K, T, S = 60, 400, 2
    x = noise(S, (T, K), 8)
    materialized = estimate_memory_gb(K, 1, T, S)[1]
    streamed = estimate_memory_gb(K, 1, T, S, streamed=True)[1]
    budget = (materialized + 2 * streamed) / 2
    assert 2 * streamed < budget < materialized
    with pytest.raises(MemoryError) as info:
        tc.features(x, order=1, max_memory_gb=budget)
    message = str(info.value)
    assert "rfun=tc.WeightedStrength(signed=True)" in message
    assert "rfun={1: tc.WeightedStrength(signed=True)}" in message
    X = tc.features(x, order=1, max_memory_gb=budget,
                    rfun=tc.WeightedStrength(signed=True))
    assert X[1][0].shape == (T, 2 * K)
    X = tc.features(x, order=1, max_memory_gb=budget,
                    rfun={1: tc.WeightedStrength(signed=True)})
    assert X[1][0].shape == (T, 2 * K)


def test_round_off_constant_feeding_columns_do_not_amplify_noise():
    # Columns [a, a, b, b] of two independent series (default_rng(10),
    # T 100): the order-1 feeding step falls back to weighted strength, and
    # its four columns are nearly identical, so the order-2 feeding columns
    # differ from constants only by round-off (range about 1e-13 on values
    # about 3 before the fix, and order 3 then varied by about 1.2).  They
    # are set to their means and reported by the constant-output warning;
    # order 3 then meets exact zero variance, whose correlations the
    # estimator sets to 0, so its weighted strength is exactly 0.
    rng = np.random.default_rng(10)
    a, b = rng.standard_normal(100), rng.standard_normal(100)
    x = np.column_stack([a, a, b, b])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _, D = tc.features(x, order=3, rfun=None, return_delta=True)
    messages = [str(w.message) for w in rec]
    assert any("constant over time" in m for m in messages), messages
    assert np.ptp(D[1], axis=0).min() > 1e-3
    assert np.all(np.ptp(D[2], axis=0) == 0)
    assert np.all(np.isfinite(D[3]))
    np.testing.assert_array_equal(D[3], 0)


# the streamed-reducer hint of the MemoryError ------------------------------

HINT = "A summary reducer streams the analysis step"


def _hint_subjects(K=120, T=600, S=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, K)) for _ in range(S)]


@pytest.mark.parametrize("frac, fits", [(0.3, 59), (0.5, 78), (0.7, 92)])
def test_original_pipeline_memory_error_has_no_hint(frac, fits):
    # K 120, T 600, 2 subjects: the order-1 step was estimated at 0.552 GB
    # (the budgets below stay fractions of that figure).
    # The co-fluctuation feeding step is materialized, so a summary reducer
    # on the analysis path would not make the call fit: no hint, and the
    # message is the one of commit ad50637 (before wave 4), verbatim except
    # for its figures.  The GB figure (and with it the largest K that fits)
    # was corrected because the earlier estimate was below the measured
    # peak: this call traces 0.6001 GB with tracemalloc against 0.552 GB
    # estimated then and 0.620 GB now.  The figures are literals (0.62 GB,
    # and K = 59, 78 and 92 for the three budgets), so the expectation does
    # not depend on the code under test.
    budget = 0.552 * frac
    kw = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
    with pytest.raises(MemoryError) as err:
        tc.features(_hint_subjects(), order=2, max_memory_gb=budget, **kw)
    assert str(err.value) == (
        "The correlation step at order 1 needs about 0.62 GB (120 input "
        f"features, T=600, 2 subjects), more than max_memory_gb={budget}. The "
        f"largest number of input features that fits one step is K={fits}. "
        "Reduce to fewer features (ndims, rfun, delta_rfun) or raise "
        "max_memory_gb."
    )
    # following the hint would indeed not have helped
    with pytest.raises(MemoryError):
        tc.features(_hint_subjects(), order=2, max_memory_gb=budget,
                    rfun=tc.WeightedStrength(signed=True), delta_rfun=None,
                    delta_weights_function=tc.eye_weights)


def test_default_call_memory_error_hint_names_order_one_and_works():
    # default call: the order-1 PCA step does not fit, the feeding step is
    # streamed; the hint names order 1 and following it completes
    subjects = _hint_subjects()
    with pytest.raises(MemoryError, match=HINT) as err:
        tc.features(subjects, order=1, max_memory_gb=0.3)
    assert "rfun={1: tc.WeightedStrength(signed=True)} for order 1 only" in str(err.value)
    X = quiet(tc.features, subjects, order=1, max_memory_gb=0.3,
              rfun={1: tc.WeightedStrength(signed=True)})
    assert X[1][0].shape == (600, 240)


# The text of commit ad50637 with its figures corrected: the GB figure
# (0.515 GB then) and with it the largest K that fits (89 then) were raised
# because the earlier estimate was below the measured peak (this call,
# computed without the budget, traces 0.5617 GB with tracemalloc, 0.5675 GB
# at commit 6b3d23a; the estimate is now 0.583 GB, and K = 84 fits).  The
# figures are literals, independent of the code under test.  Every other
# character is that commit's.
AD50637_SINGLE = (
    "The correlation step at order 1 needs about 0.583 GB (120 input "
    "features, T=600, 1 subject), more than max_memory_gb=0.3. The largest "
    "number of input features that fits one step is K=84. Reduce to fewer "
    "features (ndims, rfun, delta_rfun) or raise max_memory_gb."
)


@pytest.mark.parametrize("order, return_delta", [(1, False), (1, True), (2, False), (2, True)])
def test_original_pipeline_single_subject_memory_error_is_unchanged(order, return_delta):
    # one (600, 120) subject, max_memory_gb=0.3: the text that commit ad50637
    # (before wave 4) raised for each of these calls, verbatim except for
    # the corrected figures (see AD50637_SINGLE); at order=1
    # without return_delta the step runs without the feeding path
    x = np.random.default_rng(0).standard_normal((600, 120))
    with pytest.raises(MemoryError) as err:
        tc.features(x, order=order, return_delta=return_delta, rfun="PCA",
                    delta_rfun=None, delta_weights_function=tc.eye_weights,
                    max_memory_gb=0.3)
    assert str(err.value) == AD50637_SINGLE


@pytest.mark.parametrize("rfun", ["PCA", {1: "PCA"}])
def test_no_hint_with_an_explicit_rfun(rfun):
    # the hint is given only for a reducer the caller did not choose
    with pytest.raises(MemoryError) as err:
        tc.features(_hint_subjects(), order=1, max_memory_gb=0.3, rfun=rfun)
    assert HINT not in str(err.value)


def test_hint_with_an_order_map_that_leaves_order_one_automatic():
    # rfun={2: ...} leaves order 1 at "auto" (PCA): the hint names order 1
    with pytest.raises(MemoryError, match=HINT) as err:
        tc.features(_hint_subjects(), order=2, max_memory_gb=0.3,
                    rfun={2: tc.WeightedStrength(signed=True)})
    assert "at order 1" in str(err.value)
    assert "rfun={1: tc.WeightedStrength(signed=True)} for order 1 only" in str(err.value)


def test_no_hint_when_the_feeding_reducer_inherits_a_non_summary():
    # delta_rfun=None takes rfun ("PCA"): the hint would change the feeding
    # reducer too, so none is given
    with pytest.raises(MemoryError) as err:
        tc.features(_hint_subjects(), order=2, max_memory_gb=0.3,
                    rfun="PCA", delta_rfun=None)
    assert HINT not in str(err.value)


def test_default_memory_accounting_after_a_fallback():
    # the default call (automatic kernels and reducers) on 6 noise features,
    # order 3, T 1500 (below 2000 rows, so every order runs on all rows):
    # both feeding steps fall back to 6 weighted-strength columns (no null
    # exceedance at order 1, too few community columns at order 2).  The
    # smallest budget that completes (bisection from [0, 0.5] GB until the
    # bounds are adjacent floats, measured at the commit of this test):
    # 0.084298624 GB; just below it the order-3 streamed analysis step on
    # the 6 fallback columns raises before it computes anything (measured:
    # "The streamed analysis step at order 3 needs at least about 0.0843
    # GB"; the match accepts order 2 or 3, as the fixed-kernel version of
    # this test does)
    x = noise(2, (1500, 6), 4)
    budget = 0.084298624
    X, info = quiet(tc.features, x, order=3, max_memory_gb=budget, return_info=True)
    assert X[3][0].shape == (1500, 12)
    assert [e["fallback"] for e in info if e["path"] == "feeding"] == [
        "no_null_exceedance", "too_few_community_columns"]
    assert all("kernel_sd" in e for e in info)
    with pytest.raises(MemoryError, match=r"at order [23] .*6 input features"):
        quiet(tc.features, x, order=3, max_memory_gb=budget * (1 - 1e-6))
