"""
Defaults of the higher-order pipeline (``features`` and ``two_path_step``).

The defaults are ``rfun="auto"`` (PCA at order 1, signed weighted strength at
orders >= 2), ``delta_rfun="auto"`` (communities) and a Gaussian feeding kernel
with variance 25 (standard deviation 5 samples).  The original pipeline is
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
    # the guard uses the same estimate
    with pytest.raises(MemoryError, match="Unreduced features at order 1 need about 0.163 GB"):
        tc.features(data, order=1, rfun=None, delta_rfun=None, cfun=cfun,
                    weights_params={"var": 100}, return_delta=True,
                    max_memory_gb=(est + eye) / 2)


# second review, F2: the default pipeline after community compression ---------


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
    # instead, with a warning naming the order, and the run completes
    x = np.random.default_rng(0).normal(size=(100, 4))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = tc.features(x, **kw)
    fed = range(1, kw["order"] + (1 if kw.get("return_delta") else 0))
    for k in fed:
        assert sum(_auto_warning(str(r.message), k) for r in rec) == 1, k
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


def test_default_feeding_needs_k_distinct_shift_lags():
    # T 40, standard deviation 5: L 10, so 21 distinct lags in [10, 30]; 22
    # features need 22.  The check runs before anything is computed
    x = [np.random.default_rng(1).normal(size=(40, 22)) for _ in range(2)]
    with pytest.raises(ValueError, match="distinct lags") as info:
        tc.features(x, order=1, return_delta=True, weights_params={"var": 10})
    _alternatives_named(str(info.value))
    X = quiet(tc.features, [a[:, :21] for a in x], order=1, return_delta=True,
              weights_params={"var": 10})
    assert len(X) == 2


# wave 4, D1: the automatic feeding reducer keeps the width -------------------

FALLBACK = dict(split=False, signed=False, center=True)
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
    X, messages = recorded(tc.features, x, order=3)
    assert X[3][0].shape == (100, 2 * K)
    for k in (1, 2):
        found = auto_messages(messages, k)
        assert len(found) == 1, (k, messages)
        assert found[0].endswith(f"uses WeightedStrength on its {K} input features.")
    assert not auto_messages(messages, 3)  # the order-3 feeding step is not computed
    (X, D), messages = recorded(tc.features, x, order=2, return_delta=True)
    assert [d[0].shape for d in D] == [(100, K)] * 3
    assert [len(auto_messages(messages, k)) for k in (1, 2)] == [1, 1]


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
    (X, D), messages = recorded(tc.features, data, order=1, return_delta=True,
                                **SETTINGS, **kw)
    found = auto_messages(messages, 1)
    assert found == [
        f"Automatic feeding reducer at order 1 found {columns} community columns; "
        f"fewer than 4, so this order uses WeightedStrength on its {K} input features."
    ]
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
    # order 2 of 4 noise features: one eigenvalue above the null gives two
    # clusters, one of them a single node, so the communities emit 1 column
    x = np.random.default_rng(0).normal(size=(100, 4))
    (X, D), messages = recorded(tc.features, x, order=2, return_delta=True)
    assert auto_messages(messages, 2) == [
        "Automatic feeding reducer at order 2 found 1 community column; fewer than "
        "4, so this order uses WeightedStrength on its 4 input features."
    ]
    corrs = tc.timecorr(D[1], cfun=tc.autofc, rfun=None, **GAUSS25)
    assert_close(D[2], quiet(tc.WeightedStrength(**FALLBACK), corrs))


def test_no_exceedance_on_noise_falls_back():
    # independent noise: no eigenvalue above the shift null; the automatic
    # reducer says so and keeps the width, without the explicit reducer's
    # "one community" warning
    x = noise(2, (200, 10), 7)  # no exceedance with this seed (5 of seeds 0-7)
    (X, D), messages = recorded(tc.features, x, order=1, return_delta=True)
    found = auto_messages(messages, 1)
    assert len(found) == 1
    assert found[0].startswith("Automatic feeding reducer at order 1: the shift null "
                               "found no structure (no eigenvalue above null_threshold=")
    assert found[0].endswith("so communities would give 1 column; fewer than 4, so "
                             "this order uses WeightedStrength on its 10 input features.")
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
    (X, D), messages = recorded(tc.features, data, order=1, return_delta=True,
                                **SETTINGS)
    assert not auto_messages(messages, 1)
    comm = tc.Communities()
    _, D_explicit = quiet(tc.features, data, order=1, return_delta=True,
                          delta_rfun=comm, **SETTINGS)
    assert comm.fits_[0]["n"] >= 4
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
                                          ((39, 6), "T = 39"),
                                          ((40, 22), "distinct lags")],
                         ids=["K3", "T39", "lags"])
def test_original_input_errors_stay(shape, match):
    # the original input: fewer than 4 features, T < 40, or fewer distinct
    # shift-null lags than features raise at order 1, before anything is
    # computed, whenever a feeding step is required
    x = noise(2, shape, 2)
    for kw in (dict(order=3), dict(order=2), dict(order=1, return_delta=True)):
        with pytest.raises(ValueError, match=match) as info:
            tc.features(x, weights_params={"var": 10}, **kw)
        assert "at order 1" in str(info.value)
        _alternatives_named(str(info.value))
    # without a feeding step, order 1 is unaffected
    assert tc.features(x, order=1, weights_params={"var": 10})[1][0].shape[0] == shape[0]


def test_one_fallback_warning_per_order_not_per_subject():
    x = [np.random.default_rng(s).normal(size=(100, 5)) for s in range(4)]
    _, messages = recorded(tc.features, x, order=3, return_delta=True)
    assert [len(auto_messages(messages, k)) for k in (1, 2, 3)] == [1, 1, 1]
    _, messages = recorded(two_path_step, x, order=2)
    assert len(auto_messages(messages, 2)) == 1


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
    # columns.  The smallest budget that completes (bisection to 1e-17 GB):
    # 0.23053376 GB, of which 0.00077 GB are held while the order-3 streamed
    # analysis step runs on the 6 fallback columns (X_1, X_2 and D_2: 96,000
    # values, with the fits); just below it, that step raises before it
    # computes anything
    x = noise(2, (2000, 6), 4)
    budget = 0.23053376
    X = quiet(tc.features, x, order=3, max_memory_gb=budget)
    assert X[3][0].shape == (2000, 12)
    with pytest.raises(MemoryError, match=r"at order [23] .*6 input features"):
        quiet(tc.features, x, order=3, max_memory_gb=budget * (1 - 1e-6))


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
