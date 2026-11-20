"""
Kernels keyed by order in ``features``: ``weights_params`` and
``delta_weights_params`` as order maps (dicts whose keys are all integer
orders in ``1..order``).

``features`` with order maps is compared with ``two_path_step`` chained by
hand with the same kernel at each order.  Both run with
``max_memory_gb=None``: a streamed step's block size is the largest that fits
the budget left after the outputs a run retains, which differ between one
``features`` call and a chain of steps; with no budget both use the
estimator's default block size, so the two run the same arithmetic in the
same order and the outputs are compared with ``np.array_equal``.
"""

import warnings

import numpy as np
import pytest

import timecorr as tc
import timecorr.higher_order as ho
from timecorr.higher_order import two_path_step


def nested(S=2, T=300, seed=3):
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
        assert np.array_equal(x, y)


def chain(data, order, analysis, feeding, return_last_delta=True, **kwargs):
    """``two_path_step`` order by order; ``analysis[k]`` / ``feeding[k]`` are
    the kernel parameters of order k (None: the default)."""
    X, D, d = [data], [data], data
    for k in range(1, order + 1):
        x, dk = quiet(
            two_path_step, d, weights_params=analysis.get(k),
            delta_weights_params=feeding.get(k), order=k, max_memory_gb=None,
            **kwargs,
        )
        X.append(x)
        D.append(dk)
        d = dk
    return X, D


ANALYSIS = {1: {"var": 100}, 2: {"var": 400}, 3: {"var": 900}}
FEEDING = {1: {"var": 9}, 2: {"var": 49}, 3: {"var": 144}}


@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc], ids=["autofc", "isfc"])
def test_order_maps_equal_a_chain_of_steps(cfun):
    x = nested()
    X, D = quiet(tc.features, x, order=3, cfun=cfun, weights_params=ANALYSIS,
                 delta_weights_params=FEEDING, return_delta=True, max_memory_gb=None)
    Xc, Dc = chain(x, 3, ANALYSIS, FEEDING, cfun=cfun)
    for k in range(1, 4):
        assert_identical(X[k], Xc[k])
        assert_identical(D[k], Dc[k])
    # the kernels differ by order: a single kernel gives other features
    Xs = quiet(tc.features, x, order=3, cfun=cfun, weights_params={"var": 100},
               delta_weights_params={"var": 9}, max_memory_gb=None)
    assert_identical(X[1], Xs[1])
    assert not np.array_equal(X[2][0], Xs[2][0])


def test_one_path_by_order_the_other_plain():
    x = nested()
    X, D = quiet(tc.features, x, order=3, cfun=tc.autofc, weights_params=ANALYSIS,
                 return_delta=True, max_memory_gb=None)
    Xc, Dc = chain(x, 3, ANALYSIS, {}, cfun=tc.autofc)
    for k in range(1, 4):
        assert_identical(X[k], Xc[k])
        assert_identical(D[k], Dc[k])
    X, D = quiet(tc.features, x, order=3, cfun=tc.autofc, weights_params={"var": 100},
                 delta_weights_params=FEEDING, return_delta=True, max_memory_gb=None)
    Xc, Dc = chain(x, 3, {k: {"var": 100} for k in (1, 2, 3)}, FEEDING, cfun=tc.autofc)
    for k in range(1, 4):
        assert_identical(X[k], Xc[k])
        assert_identical(D[k], Dc[k])


def test_missing_orders_use_the_defaults():
    x = nested()
    common = dict(order=3, cfun=tc.autofc, return_delta=True, max_memory_gb=None)
    # analysis: the kernel function's own default (gaussian_params, var 100);
    # feeding: the default feeding kernel (Gaussian, var 25)
    X, D = quiet(tc.features, x, weights_params={2: {"var": 400}},
                 delta_weights_params={3: {"var": 144}}, **common)
    Xe, De = quiet(tc.features, x, weights_params={1: None, 2: {"var": 400}, 3: None},
                   delta_weights_params={1: None, 2: None, 3: {"var": 144}}, **common)
    Xc, Dc = chain(x, 3, {2: {"var": 400}}, {3: {"var": 144}}, cfun=tc.autofc)
    for k in range(1, 4):
        assert_identical(X[k], Xe[k])
        assert_identical(D[k], De[k])
        assert_identical(X[k], Xc[k])
        assert_identical(D[k], Dc[k])
    # orders left out equal passing nothing at all for them
    X0, D0 = quiet(tc.features, x, **common)
    assert_identical(X[1], X0[1])
    assert_identical(D[1], D0[1])
    # explicit defaults
    Xd, Dd = quiet(tc.features, x, weights_params={2: tc.gaussian_params},
                   delta_weights_params={1: {"var": 25}}, **common)
    for k in range(1, 4):
        assert_identical(Xd[k], X0[k])
        assert_identical(Dd[k], D0[k])


def test_an_empty_map_and_plain_params_keep_their_meaning():
    x = nested()
    common = dict(order=2, cfun=tc.autofc, max_memory_gb=None)
    plain = quiet(tc.features, x, weights_params={"var": 400}, **common)
    as_map = quiet(tc.features, x, weights_params={1: {"var": 400}, 2: {"var": 400}},
                   **common)
    for k in range(3):
        assert_identical(plain[k], as_map[k])
    # an empty dict is passed to the kernel as today: gaussian_weights needs "var"
    with pytest.raises(KeyError):
        quiet(tc.features, x, weights_params={}, **common)
    # and an empty delta_weights_params is a Gaussian without "var"
    with pytest.raises(ValueError, match='needs "var"'):
        tc.features(x, delta_weights_params={}, **common)


def test_eye_weights_feeding_with_a_map_of_none():
    x = nested()
    legacy = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
    X = tc.features(x, order=2, weights_params={1: {"var": 100}, 2: {"var": 400}},
                    delta_weights_params={1: None, 2: None}, **legacy)
    Xc, _ = chain(x, 2, {1: {"var": 100}, 2: {"var": 400}}, {}, **legacy)
    for k in range(1, 3):
        assert_identical(X[k], Xc[k])
    with pytest.raises(ValueError, match="eye_weights takes no parameters"):
        tc.features(x, order=2, delta_weights_params={2: {"var": 9}}, **legacy)


@pytest.mark.parametrize("name", ["weights_params", "delta_weights_params"])
@pytest.mark.parametrize(
    "value, match",
    [
        ({0: {"var": 9}}, r"integers in 1\.\.2; got \[0\]"),
        ({3: {"var": 9}}, r"integers in 1\.\.2; got \[3\]"),
        ({1: {"var": 9}, -1: None}, r"integers in 1\.\.2; got \[-1\]"),
        ({True: {"var": 9}}, "order map"),
        ({1: {"var": 9}, "var": 9}, "order map"),
        ({1: {"var": 9}, False: None}, "order map"),
        ({1: {2: {"var": 9}}}, "cannot be nested"),
        ({1: 9}, r"must be a dict of kernel parameters or None"),
    ],
    ids=["zero", "above", "negative", "bool", "mixed", "mixed-bool", "nested", "value"],
)
def test_order_map_validation(name, value, match):
    x = nested(T=100)
    with pytest.raises(ValueError, match=match):
        tc.features(x, order=2, **{name: value})


@pytest.mark.parametrize("value", [{True: {"var": 9}}, {1: {"var": 9}, False: None}])
def test_boolean_keys_are_refused_as_orders(value):
    x = nested(T=100)
    with pytest.raises(ValueError, match="booleans are not accepted as orders"):
        tc.features(x, order=2, weights_params=value)


def test_order_map_messages_say_what_an_order_map_is():
    x = nested(T=100)
    with pytest.raises(ValueError) as info:
        tc.features(x, order=2, weights_params={1: {"var": 9}, "var": 9})
    message = str(info.value)
    assert "weights_params" in message and "integer orders" in message
    assert "kernel parameters" in message
    with pytest.raises(ValueError, match="order map"):
        tc.features(x, order=0, weights_params={1: {"var": 9}})


@pytest.mark.parametrize("name", ["weights_params", "delta_weights_params"])
def test_two_path_step_refuses_an_order_map(name):
    x = nested(T=100)
    with pytest.raises(ValueError, match="features") as info:
        two_path_step(x, **{name: {1: {"var": 9}}})
    assert "order map" in str(info.value)
    # plain parameters are still accepted
    quiet(two_path_step, x, cfun=tc.autofc, **{name: {"var": 9}})


def test_events_with_kernels_by_order():
    x = nested()
    events = {2: [150], 3: [100, 200]}
    X, D = quiet(tc.features, x, order=3, cfun=tc.autofc, weights_params=ANALYSIS,
                 delta_weights_params=FEEDING, events=events, return_delta=True,
                 max_memory_gb=None)
    Xc, Dc, d = [x], [x], x
    for k in range(1, 4):
        xk, dk = quiet(two_path_step, d, cfun=tc.autofc, weights_params=ANALYSIS[k],
                       delta_weights_params=FEEDING[k], events=events.get(k),
                       order=k, max_memory_gb=None)
        Xc.append(xk)
        Dc.append(dk)
        d = dk
    for k in range(1, 4):
        assert_identical(X[k], Xc[k])
        assert_identical(D[k], Dc[k])


def test_feeding_shift_null_check_uses_each_orders_bandwidth():
    # T 100: L = min(25, max(10, 10 * sd)); the default feeding kernel (sd 5)
    # gives L 25 and 51 distinct lags, fewer than 60 features; sd 1 gives
    # L 10 and 81 lags.  The order-1 feeding step reads order 1's kernel
    rng = np.random.default_rng(4)
    x = [rng.normal(size=(100, 60)) for _ in range(2)]
    with pytest.raises(ValueError, match="L = 25 give 51 distinct lags"):
        tc.features(x, order=1, return_delta=True, cfun=tc.autofc)
    with pytest.raises(ValueError, match="L = 25 give 51 distinct lags"):
        tc.features(x, order=2, cfun=tc.autofc, delta_weights_params={2: {"var": 1}})
    (_, D) = quiet(tc.features, x, order=1, return_delta=True, cfun=tc.autofc,
                   delta_weights_params={1: {"var": 1}})
    assert D[1][0].shape[0] == 100
    # the analysis bandwidth sets the shift null of an automatic analysis
    # Communities with autofc, order by order (as in the chain test above)
    assert ho._bandwidth(tc.gaussian_weights, ANALYSIS[3]) == 30.0


def test_streamed_memory_check_does_not_depend_on_kernel_parameters():
    # the memory estimates do not depend on kernel parameters (this test
    # does not show that each order's kernel reaches the memory check):
    # a budget that refuses order 1's streamed step refuses it with plain
    # parameters and with order maps of different widths alike, and the
    # MemoryError names the order whose step does not fit
    x = nested(T=300)
    need = ho.estimate_memory_gb(24, 1, 300, 2, streamed=True)[1]
    budget = need / 2
    for params in ({"var": 100}, {1: {"var": 400}}, {1: {"var": 16}}):
        with pytest.raises(MemoryError, match="at order 1"):
            tc.features(x, order=2, cfun=tc.autofc,
                        rfun=tc.WeightedStrength(signed=True),
                        delta_weights_params=params, max_memory_gb=budget)


def test_two_path_step_is_exported_at_the_top_level():
    assert tc.two_path_step is ho.two_path_step
    assert "two_path_step" in dir(tc)


def test_analysis_communities_use_the_kernel_of_their_order():
    # automatic analysis communities at order 2 (autofc, T 100, 60 feeding
    # columns): with the default analysis kernel (Gaussian, var 100: sd 10,
    # L = 25, 51 lags in [25, 75]) the shift null of order 2 had fewer
    # distinct lags than columns; weights_params={2: {"var": 1}} (sd 1,
    # L = 10) is used at order 2 only, for the kernel and for the null's L
    x = [np.random.RandomState(s).randn(100, 60) for s in range(2)]
    summary = tc.Communities()
    X, D = quiet(
        tc.features, x, order=2, cfun=tc.autofc, rfun={2: summary},
        delta_rfun="weighted_strength", weights_params={2: {"var": 1}},
        return_delta=True,
    )
    assert [a.shape[0] for a in X[2]] == [100, 100]
    fit = summary.fits_[-1]
    kernels = ho._step_kernels(100, 2, tc.gaussian_weights, {"var": 1})
    from timecorr.summaries import _shift_null_threshold
    expected = _shift_null_threshold(
        [np.ascontiguousarray(d, dtype=np.float64) for d in D[1]], kernels,
        n_null=summary.n_null, bandwidth=1.0,
    )
    assert fit["null_threshold"] == pytest.approx(expected, rel=1e-12)
    # order 1 kept the default analysis kernel: its output equals a run
    # without the order map
    X_default, _ = quiet(
        tc.features, x, order=1, cfun=tc.autofc,
        delta_rfun="weighted_strength", return_delta=True,
    )
    for a, b in zip(X[1], X_default[1]):
        assert_identical(a, b)
