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


def _alternatives_named(message):
    assert 'delta_rfun="weighted_strength"' in message
    assert 'rfun="PCA", delta_rfun=None' in message
    assert "delta_weights_function=tc.eye_weights" in message


@pytest.mark.parametrize("kw", [dict(order=3), dict(order=2, return_delta=True)],
                         ids=["order3", "order2-return-delta"])
def test_default_feeding_after_community_compression_raises(kw):
    # 4 features: the order-1 communities keep at most 4 // 2 = 2 columns, so
    # the order-2 feeding step gets fewer than 4 features
    x = np.random.default_rng(0).normal(size=(100, 4))
    with pytest.raises(ValueError, match="at order 2") as info:
        quiet(tc.features, x, **kw)
    message = str(info.value)
    _alternatives_named(message)
    assert "communities found at order 1" in message


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


# wave 4, D6: the MemoryError of a default order-1 step names streaming --------


def noise(S, shape, seed):
    """S subjects of independent standard normal noise of the given shape."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape) for _ in range(S)]


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
