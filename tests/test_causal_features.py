"""
``causal=True`` in ``features`` and ``two_path_step``, with reducers that
need no fit (None and graph measures).

The guarantee: row t of every ``X[k]`` and ``D[k]`` is unchanged (to 1e-9,
NaN rows included) when the input rows after t are replaced by other finite
values, by NaN, by inf, or removed; with ``include_zero=False`` also when
row t itself changes.  Warm-up rows are NaN and are missing data for the
next order, so they accumulate over the orders.  Real computations; nothing
is mocked.
"""

import functools
import time
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.higher_order import estimate_memory_gb

from .test_causal_estimator import (
    SUBJECT_WEIGHTS,
    _as_list,
    _expected_undefined,
    _max_difference,
    _perturb,
    _quiet,
    _series,
)
from .test_estimator_causality import _data as _conditioned_data

ATOL = 1e-9
WISFC = functools.partial(tc.wisfc, subject_weights=SUBJECT_WEIGHTS)


# ----------------------------------------------------------------------------
# the estimator's rule for columns that are constant up to round-off
# ----------------------------------------------------------------------------


def _nearly_constant(mean, relative, n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4))
    noise = rng.standard_normal(n)
    x[:, 1] = mean * (1 + relative * noise)
    return x, noise


@pytest.mark.parametrize("mean", [1.0, 1e6, -3.0, 2e-5])
def test_a_column_constant_to_relative_round_off_has_zero_variance(mean):
    x, _ = _nearly_constant(mean, 1e-12)
    kw = dict(cfun=tc.autofc, weights_params={"var": 25})
    out = _quiet(tc.timecorr, x, causal=True, **kw)
    assert np.isnan(out[:2]).all() and not np.isnan(out[2:]).any()
    full = tc.vec2mat(out[2:])
    # 0, not correlations of round-off noise; the other features keep theirs
    assert np.all(full[1, :, :] == 0) and np.all(full[:, 1, :] == 0)
    assert np.all(full[[0, 2, 3], [0, 2, 3], :] == 1)
    assert np.all(np.abs(full[0, 2, 5:]) > 0)
    # identically when later rows are removed
    for n in (3, 40, 128, 129, 257):
        part = _quiet(tc.timecorr, x[:n], causal=True, **kw)
        assert _max_difference(out, part, n) <= ATOL
        assert np.all(tc.vec2mat(part[2:])[1] == 0)
    # the rule is the causal estimator's: without causal=True the column's
    # round-off noise is correlated like any signal
    plain = tc.vec2mat(_quiet(tc.timecorr, x, include_timepoints="backward", **kw)[2:])
    assert np.mean(plain[1, 0, :] != 0) > 0.9


@pytest.mark.parametrize("mean", [1.0, 1e6])
def test_a_column_with_relative_variation_1e_6_is_not_constant(mean):
    x, noise = _nearly_constant(mean, 1e-6)
    kw = dict(causal=True, cfun=tc.autofc, weights_params={"var": 25})
    out = tc.vec2mat(_quiet(tc.timecorr, x, **kw)[2:])
    # a correlation does not depend on offset and scale: those of the noise
    y = x.copy()
    y[:, 1] = noise
    expected = tc.vec2mat(_quiet(tc.timecorr, y, **kw)[2:])
    assert np.all(out[1, 1, :] == 1)
    assert np.mean(out[1, 0, :] != 0) == 1
    assert np.max(np.abs(out - expected)) < 1e-6


def test_the_rule_applies_to_the_next_order_of_features():
    # a feeding column that is constant up to round-off (strength of a node
    # whose correlations are all 0 or 1) gives zeros at the next order
    x, _ = _nearly_constant(1.0, 1e-12, n=200)
    kw = dict(causal=True, cfun=tc.autofc, rfun=None, delta_rfun=None,
              weights_params={"var": 25}, delta_weights_params={"var": 25})
    X = _quiet(tc.features, x, order=1, **kw)
    full = tc.vec2mat(X[1][2:])
    assert np.all(full[1, :, :] == 0)
    for n in (40, 129):
        part = _quiet(tc.features, x[:n], order=1, **kw)
        assert _max_difference(X[1], part[1], n) <= ATOL


def _constant_feeding_column(relative, seed=3):
    """A feeding reducer whose last output column is 3 to ``relative``."""
    noise = np.random.default_rng(seed).standard_normal(400)

    def reducer(arrays):
        return [np.column_stack([a[:, 4:7], 3.0 * (1 + relative * noise[:len(a)])])
                for a in arrays]

    return reducer


@pytest.mark.parametrize("mode", ["finite", "nan", "inf", "remove"])
def test_a_feeding_column_constant_to_round_off_is_zero_at_the_next_order(mode):
    # the order-1 feeding features: three correlations and a column that is
    # 3 up to a relative 1e-12 (the round-off of a summary that is not set to
    # its mean in causal mode).  At order 2 its correlations are 0, not
    # correlations of that noise (which they are when the estimator's rule,
    # helpers._causal_zero_variance, is taken out: 27% of them non-zero, up
    # to 0.28), whatever comes later.
    x = np.random.default_rng(4).standard_normal((200, 4))
    kw = dict(order=2, causal=True, cfun=tc.autofc, rfun=None,
              delta_rfun=_constant_feeding_column(1e-12), return_delta=True,
              weights_params={"var": 25}, delta_weights_params={"var": 25})
    X, D = _quiet(tc.features, x, **kw)
    column = D[1][2:, 3]
    assert 0 < np.ptp(column) < 1e-10 and np.all(np.abs(column - 3) < 1e-10)
    second = tc.vec2mat(X[2][4:])
    assert np.all(second[3, :, :] == 0) and np.all(second[:, 3, :] == 0)
    # the other feeding features keep their correlations
    assert np.all(second[[0, 1, 2], [0, 1, 2], :] == 1)
    assert np.mean(second[0, 1, :] != 0) > 0.99
    rng = np.random.default_rng(5)
    for n in (5, 40, 129, 199):
        part = _quiet(tc.features, _perturb(x, n, mode, rng), **kw)
        assert _worst((X, D), part, n, (1, 2)) <= ATOL
        if n > 5:
            assert np.all(tc.vec2mat(part[0][2][4:n])[3] == 0)
    # a column that varies by a relative 1e-6 is a signal: correlated
    kw["delta_rfun"] = _constant_feeding_column(1e-6)
    varying = tc.vec2mat(_quiet(tc.features, x, **kw)[0][2][4:])
    assert np.mean(varying[3, 0, :] != 0) > 0.99


def test_duplicated_features_give_a_constant_feeding_column():
    # a real case: feature 1 is an affine function of feature 0, so their
    # order-1 correlation is 1 up to round-off at every timepoint; at order
    # 2 that feeding column has no correlation with any other
    x = np.random.default_rng(6).standard_normal((200, 4))
    x[:, 1] = 2 * x[:, 0] + 3
    kw = dict(order=2, causal=True, cfun=tc.autofc, rfun=None, delta_rfun=None,
              return_delta=True, weights_params={"var": 25},
              delta_weights_params={"var": 25})
    X, D = _quiet(tc.features, x, **kw)
    pair = 4  # mat2vec: the 4 diagonal entries, then the pair (0, 1)
    assert np.max(np.abs(D[1][2:, pair] - 1)) < 1e-12
    second = tc.vec2mat(X[2][4:])
    assert np.all(np.delete(second[pair], pair, axis=0) == 0)
    rng = np.random.default_rng(7)
    for mode in MODES:
        for n in (40, 129):
            part = _quiet(tc.features, _perturb(x, n, mode, rng), **kw)
            assert _worst((X, D), part, n, (1, 2)) <= ATOL


# ----------------------------------------------------------------------------
# the guarantee
# ----------------------------------------------------------------------------

T = 300
CUTS = (3, 40, 128, 129, 200, 299)
PARAMS = dict(weights_params={"var": 25}, delta_weights_params={"var": 16})
UNREDUCED = dict(rfun=None, delta_rfun=None)
STRENGTH = dict(rfun="strength", delta_rfun="strength")

# name: (number of subjects, features, highest order, arguments)
CONFIGS = {
    "unreduced": (1, 4, 3, dict(cfun=tc.autofc, **UNREDUCED, **PARAMS)),
    "strength": (1, 6, 4, dict(cfun=tc.autofc, **STRENGTH, **PARAMS)),
    "eigenvector": (1, 6, 3, dict(cfun=tc.autofc, rfun="eigenvector_centrality",
                                  delta_rfun="strength", **PARAMS)),
    "mixed": (1, 5, 3, dict(cfun=tc.autofc, rfun={1: None, 2: "strength", 3: None},
                            delta_rfun={1: "strength", 2: None, 3: "strength"},
                            **PARAMS)),
    "laplace": (1, 5, 3, dict(cfun=tc.autofc, **STRENGTH,
                              weights_function=tc.laplace_weights,
                              weights_params={"scale": 6},
                              delta_weights_function=tc.laplace_weights,
                              delta_weights_params={"scale": 4})),
    "boxcar": (1, 5, 3, dict(cfun=tc.autofc, **STRENGTH,
                             weights_function=tc.boxcar_weights,
                             weights_params={"width": 30},
                             delta_weights_function=tc.boxcar_weights,
                             delta_weights_params={"width": 16})),
    "order-maps": (1, 5, 3, dict(
        cfun=tc.autofc, **STRENGTH,
        weights_params={1: {"var": 25}, 2: {"var": 49}, 3: {"var": 9}},
        delta_weights_params={1: {"var": 16}, 2: {"var": 36}, 3: {"var": 16}})),
    "events": (1, 5, 3, dict(cfun=tc.autofc, **STRENGTH, **PARAMS, events=[90, 210])),
    "events-by-order": (1, 5, 3, dict(cfun=tc.autofc, **STRENGTH, **PARAMS,
                                      events={1: [150], 3: [60, 100]})),
    "min-samples": (1, 5, 3, dict(cfun=tc.autofc, **STRENGTH, **PARAMS, min_samples=6)),
    "events-short": (1, 5, 3, dict(cfun=tc.autofc, **STRENGTH, **PARAMS,
                                   events={1: [1, 2, 4, 7, 8, 150], 2: [1, 100]})),
    "events-per-subject": (3, 5, 3, dict(cfun=tc.isfc, **STRENGTH, **PARAMS,
                                         events=[[1, 60], None, [130, 131, 250]])),
    "isfc": (3, 5, 3, dict(cfun=tc.isfc, **STRENGTH, **PARAMS)),
    "isfc-unreduced": (3, 4, 2, dict(cfun=tc.isfc, **UNREDUCED, **PARAMS)),
    "isfc-corrmean": (3, 5, 3, dict(cfun=tc.isfc, combine=tc.corrmean_combine,
                                    **STRENGTH, **PARAMS)),
    "isfc-mean": (3, 5, 3, dict(cfun=tc.isfc, combine=tc.mean_combine, **STRENGTH,
                                **PARAMS)),
    "isfc-tstat": (3, 4, 2, dict(cfun=tc.isfc, combine=tc.tstat_combine,
                                 **UNREDUCED, **PARAMS)),
    "wisfc": (3, 5, 3, dict(cfun=WISFC, **STRENGTH, **PARAMS)),
    "wisfc-corrmean": (3, 5, 2, dict(cfun=WISFC, combine=tc.corrmean_combine,
                                     **UNREDUCED, **PARAMS)),
    "autofc-subjects": (3, 5, 3, dict(cfun=tc.autofc, **STRENGTH, **PARAMS)),
}
MODES = ("finite", "nan", "inf", "remove")
# Removing rows changes the shapes of the estimator's last sums and so their
# round-off (replacement changes no shape and no bit).  A t statistic over 3
# subjects divides by their spread, which amplifies it (measured: 7.8e-12).
REMOVE_ATOL = {}


def _data(config, seed=0, n=T):
    n_subjects, K = CONFIGS[config][:2]
    data = [_series(seed + s, n=n, K=K) for s in range(n_subjects)]
    return data[0] if n_subjects == 1 else data


def _map(data, f):
    return [f(x) for x in data] if isinstance(data, list) else f(data)


def _within(events, n):
    """Boundaries of a series cut to ``n`` rows (the later ones are not in it)."""
    if isinstance(events, dict):
        return {k: _within(v, n) for k, v in events.items()}
    if any(e is None or isinstance(e, list) for e in events):  # one entry per subject
        return [None if e is None else _within(e, n) for e in events]
    kept = [b for b in events if b < n]
    return kept or None


def _run(data, config, order=None, all_events=False, **kwargs):
    highest, kw = CONFIGS[config][2:]
    if "events" in kw and not all_events:
        kw = {**kw, "events": _within(kw["events"], len(_as_list(data)[0]))}
    return _quiet(tc.features, data, order=highest if order is None else order,
                  causal=True, return_delta=True, **{**kw, **kwargs})


def _worst(full, part, rows, orders):
    """Largest difference over ``X[k]`` and ``D[k]``, k in ``orders``, rows ``:rows``."""
    return max(_max_difference(f[k], p[k], rows)
               for f, p in zip(full, part) for k in orders)


def _defined_rows(out):
    return int(np.sum(~np.isnan(np.asarray(_as_list(out)[0])).all(axis=1)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_rows_do_not_depend_on_later_rows(config, include_zero, mode):
    data = _data(config)
    highest = CONFIGS[config][2]
    orders = range(1, highest + 1)
    full = _run(data, config, include_zero=include_zero)
    assert len(full[0]) == highest + 1 and len(full[1]) == highest + 1
    # the highest order has defined rows before the last cut
    assert _defined_rows(full[0][highest]) > T - CUTS[-2]
    rng = np.random.default_rng(7)
    atol = REMOVE_ATOL.get(config, ATOL) if mode == "remove" else ATOL
    for n in CUTS:
        part = _run(_map(data, lambda x: _perturb(x, n, mode, rng)), config,
                    include_zero=include_zero)
        # without the diagonal, row n itself may change too
        rows = n if include_zero or mode == "remove" else n + 1
        assert _worst(full, part, rows, orders) <= atol


@pytest.mark.parametrize("mode", ["finite", "nan", "inf"])
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("config", ["unreduced", "strength", "eigenvector", "isfc",
                                    "wisfc", "events", "min-samples"])
def test_replacing_later_rows_changes_no_bit(config, include_zero, mode):
    data = _data(config, seed=20)
    full = _run(data, config, include_zero=include_zero)
    rng = np.random.default_rng(8)
    for n in CUTS:
        part = _run(_map(data, lambda x: _perturb(x, n, mode, rng)), config,
                    include_zero=include_zero)
        rows = n if include_zero else n + 1
        for f, p in zip(full, part):
            for fk, pk in zip(f[1:], p[1:]):
                for a, b in zip(_as_list(fk), _as_list(pk)):
                    assert np.array_equal(a[:rows], b[:rows], equal_nan=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("include_zero", [True, False])
def test_order_four_unreduced(include_zero, mode):
    # 3 features: 6, 21, 231 and 26,796 columns at orders 1 to 4
    n = 150
    rng = np.random.default_rng(5)
    x = rng.standard_normal((n, 3))
    x[:, 0] += 1e3
    x[:, 1] = np.cumsum(x[:, 1])
    x[rng.random(x.shape) < 0.02] = np.nan
    kw = dict(order=4, causal=True, cfun=tc.autofc, include_zero=include_zero,
              **UNREDUCED, **PARAMS)
    full = _quiet(tc.features, x, **kw)
    assert [f.shape[1] for f in full] == [3, 6, 21, 231, 26796]
    assert _defined_rows(full[4]) > 100
    for cut in (60, 129):
        part = _quiet(tc.features, _perturb(x, cut, mode, rng), **kw)
        rows = cut if include_zero or mode == "remove" else cut + 1
        assert max(_max_difference(full[k], part[k], rows) for k in range(1, 5)) <= ATOL


@pytest.mark.parametrize("config", ["unreduced", "strength", "eigenvector", "laplace",
                                    "boxcar", "isfc", "isfc-corrmean", "wisfc",
                                    "min-samples", "events", "events-by-order",
                                    "events-short", "events-per-subject"])
@pytest.mark.parametrize("include_zero", [True, False])
def test_every_prefix_runs_and_equals_the_first_rows(config, include_zero):
    data = _data(config, seed=30)
    orders = range(1, 4) if CONFIGS[config][2] >= 3 else range(1, 3)
    full = _run(data, config, order=orders[-1], include_zero=include_zero)
    atol = REMOVE_ATOL.get(config, ATOL)
    for n in list(range(1, 13)) + [20, 64, 91, 130, 211, 257]:
        # every prefix gets the events of the whole series: the boundaries
        # it has not reached are ignored
        part = _run(_map(data, lambda x: x[:n]), config, order=orders[-1],
                    include_zero=include_zero, all_events=True)
        for out in part:
            assert all(a.shape[0] == n for k in orders for a in _as_list(out[k]))
        assert _worst(full, part, n, orders) <= atol


def test_two_path_step():
    x = _series(40, K=5)
    kw = dict(causal=True, cfun=tc.autofc, **STRENGTH, **PARAMS)
    X, D = _quiet(tc.features, x, order=2, return_delta=True, **kw)
    X1, D1 = _quiet(tc.two_path_step, x, **kw)
    assert np.array_equal(X1, X[1], equal_nan=True)
    assert np.array_equal(D1, D[1], equal_nan=True)
    X2, D2 = _quiet(tc.two_path_step, D1, order=2, **kw)
    assert np.array_equal(X2, X[2], equal_nan=True)
    assert np.array_equal(D2, D[2], equal_nan=True)
    assert np.isnan(X2[:4]).all() and not np.isnan(X2[4:]).any()
    # rows do not depend on later rows, with min_samples too
    rng = np.random.default_rng(41)
    full = _quiet(tc.two_path_step, x, min_samples=5, **kw)
    assert 5 < _defined_rows(full[0]) < T - 2
    for mode in MODES:
        for n in (3, 129, 200):
            part = _quiet(tc.two_path_step, _perturb(x, n, mode, rng), min_samples=5,
                          **kw)
            assert max(_max_difference(f, p, n) for f, p in zip(full, part)) <= ATOL
    # list input
    X1s, D1s = _quiet(tc.two_path_step, [x, x[::-1].copy()], **kw)
    assert len(X1s) == 2 and len(D1s) == 2
    assert np.array_equal(D1s[0], D1, equal_nan=True)


# ----------------------------------------------------------------------------
# warm-up rows accumulate over the orders
# ----------------------------------------------------------------------------


def _kernel(function, params, n, include_zero, boundaries=()):
    w = function(n, params, direction="backward", include_zero=include_zero)
    block = np.searchsorted(np.asarray(boundaries), np.arange(n), side="right")
    return w * (block[:, np.newaxis] == block[np.newaxis, :])


@pytest.mark.parametrize("config", ["unreduced", "strength", "laplace", "boxcar",
                                    "order-maps", "events", "events-by-order", "isfc",
                                    "isfc-corrmean", "wisfc", "autofc-subjects"])
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("min_samples", [None, 3, 4.5])
def test_warm_up_rows_accumulate_over_the_orders(config, include_zero, min_samples):
    data = _data(config, seed=50)
    highest, kw = CONFIGS[config][2:]
    (X, D), info = _run(data, config, include_zero=include_zero,
                        min_samples=min_samples, return_info=True)
    analysis = kw.get("weights_function", tc.gaussian_weights)
    feeding = kw.get("delta_weights_function", tc.gaussian_weights)
    events = kw.get("events")
    observed = np.ones(T)  # no row of the data is missing altogether
    for k in range(1, highest + 1):
        boundaries = events.get(k, ()) if isinstance(events, dict) else (events or ())
        params = [kw[name][k] if 1 in kw[name] else kw[name]
                  for name in ("weights_params", "delta_weights_params")]
        expected = [
            _expected_undefined(_kernel(f, p, T, include_zero, boundaries), observed,
                                min_samples)
            for f, p in zip((analysis, feeding), params)
        ]
        for out, exp, path in zip((X[k], D[k]), expected, ("analysis", "feeding")):
            for a in _as_list(out):
                assert np.array_equal(np.isnan(a).all(axis=1), exp)
                assert not np.isnan(a[~exp]).any()
            entry = [e for e in info if e["order"] == k and e["path"] == path][0]
            count = int(exp.sum())
            if isinstance(out, list):
                assert entry["warmup_rows"] == [count] * len(out)
            else:
                assert entry["warmup_rows"] == count
            assert entry["weights_params"] == params[path == "feeding"]
        # the next order reads the feeding features: their NaN rows are missing
        observed = (~expected[1]).astype(float)
    counts = [e["warmup_rows"] for e in info if e["path"] == "analysis"]
    counts = [c[0] if isinstance(c, list) else c for c in counts]
    assert counts[0] >= (2 if include_zero else 3)
    if not isinstance(events, dict) and 1 not in kw["weights_params"]:
        # the same kernels and events at every order: more undefined rows at each
        assert all(b > a for a, b in zip(counts, counts[1:]))


def test_missing_rows_of_the_data_are_missing_at_every_order():
    x = _series(60, K=5)
    x[100:112] = np.nan
    kw = dict(causal=True, cfun=tc.autofc, return_delta=True, **STRENGTH, **PARAMS)
    X, D = _quiet(tc.features, x, order=2, min_samples=4, **kw)
    observed = (~np.isnan(x)).any(axis=1).astype(float)
    wa = _kernel(tc.gaussian_weights, {"var": 25}, T, True)
    wf = _kernel(tc.gaussian_weights, {"var": 16}, T, True)
    assert np.array_equal(np.isnan(X[1]).all(axis=1), _expected_undefined(wa, observed, 4))
    d1 = _expected_undefined(wf, observed, 4)
    assert np.array_equal(np.isnan(D[1]).all(axis=1), d1)
    # the first missing rows still have 4 effective samples before them
    assert not d1[100] and d1[106:112].all() and not d1[130:].any()
    x2 = _expected_undefined(wa, (~d1).astype(float), 4)
    assert np.array_equal(np.isnan(X[2]).all(axis=1), x2)
    assert np.flatnonzero(~x2)[0] > np.flatnonzero(~d1)[0]


# ----------------------------------------------------------------------------
# agreement with causal=False
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("reducers", ["unreduced", "strength", "eigenvector"])
@pytest.mark.parametrize("cfun", ["autofc", "isfc"])
def test_agrees_with_backward_kernels_of_causal_false(reducers, cfun):
    # boxcar kernels over the lags 0..14 (analysis) and 0..7 (feeding).
    # causal=False returns undefined rows as rows of 0 and reads them as data
    # at the next order, where causal=True reads them as missing: the two
    # agree on the rows that weight none of them, at any order before
    subjects = [_conditioned_data(seed=s)[:, :4] for s in range(3)]
    n = subjects[0].shape[0]
    kw = dict(order=3, cfun={"autofc": tc.autofc, "isfc": tc.isfc}[cfun],
              return_delta=True,
              weights_function=tc.boxcar_weights, weights_params={"width": 30},
              delta_weights_function=tc.boxcar_weights,
              delta_weights_params={"width": 16},
              **{"unreduced": UNREDUCED, "strength": STRENGTH,
                 "eigenvector": dict(rfun="eigenvector_centrality",
                                     delta_rfun="strength")}[reducers])
    causal = _quiet(tc.features, subjects, causal=True, **kw)
    plain = _quiet(tc.features, subjects, include_timepoints="backward", **kw)
    clean = 0  # the feeding features of the previous order agree from this row on
    for k in range(1, 4):
        starts = (max(2, clean + 14), max(2, clean + 7))
        for c_out, p_out, start in zip(causal, plain, starts):
            for c, p in zip(c_out[k], p_out[k]):
                assert c.shape == p.shape
                assert not np.isnan(c[start:]).any()
                assert np.max(np.abs(c[start:] - p[start:])) <= ATOL
        clean = starts[1]
    assert starts[0] < n // 4


# ----------------------------------------------------------------------------
# return_info, constant columns, memory
# ----------------------------------------------------------------------------


def test_return_info_of_a_causal_call():
    x = _series(70, K=5)
    (X, D), info = _quiet(
        tc.features, x, order=2, causal=True, cfun=tc.autofc, return_delta=True,
        return_info=True, **STRENGTH,
        weights_params={1: {"var": 25}, 2: {"var": 49}},
        delta_weights_params={"var": 16})
    assert [(e["order"], e["path"]) for e in info] == [
        (1, "analysis"), (1, "feeding"), (2, "analysis"), (2, "feeding")]
    assert [e["weights_params"] for e in info] == [
        {"var": 25}, {"var": 16}, {"var": 49}, {"var": 16}]
    assert [e["warmup_rows"] for e in info] == [2, 2, 4, 4]
    assert all(type(e["warmup_rows"]) is int for e in info)
    for e in info:
        out = (X if e["path"] == "analysis" else D)[e["order"]]
        assert e["warmup_rows"] == int(np.isnan(out).all(axis=1).sum())
        assert e["requested"] == "strength" and e["effective"] == "strength"
    # the dicts are copies: changing them does not change the caller's
    info[0]["weights_params"]["var"] = 1
    # without return_delta the last feeding step is not computed
    _, info = _quiet(tc.features, [x, x + 1], order=1, causal=True, return_info=True,
                     **UNREDUCED, **PARAMS)
    assert len(info) == 1 and info[0]["warmup_rows"] == [2, 2]


def test_constant_columns_ignore_the_warm_up_rows():
    rng = np.random.default_rng(71)
    x = rng.standard_normal((120, 5))
    x[:, 2] = 4.0  # zero variance: correlations 0, strength 0 at every defined row
    (X, D), info = _quiet(tc.features, x, order=2, causal=True, cfun=tc.autofc,
                          return_delta=True, return_info=True, **STRENGTH, **PARAMS)
    for e in info:
        out = (X if e["path"] == "analysis" else D)[e["order"]]
        defined = ~np.isnan(out).all(axis=1)
        assert 0 < (~defined).sum() < 10
        expected = np.flatnonzero(np.ptp(out[defined], axis=0) == 0).tolist()
        assert e["constant_columns"] == expected
        assert 2 in expected
    # an output without a defined row has no constant column
    _, info = _quiet(tc.features, x[:2], order=1, causal=True, cfun=tc.autofc,
                     return_info=True, **STRENGTH, **PARAMS)
    assert info[0]["constant_columns"] == [] and info[0]["warmup_rows"] == 2


def test_info_of_calls_without_causal_is_unchanged():
    x = np.random.default_rng(72).standard_normal((80, 4))
    _, info = _quiet(tc.features, x, order=1, rfun=None, delta_rfun=None,
                     return_info=True, **PARAMS)
    assert set(info[0]) == {"order", "path", "requested", "effective", "input_width",
                            "output_width", "constant_columns"}


def test_memory_limit_applies():
    x = np.random.default_rng(73).standard_normal((400, 30))
    kw = dict(order=2, causal=True, cfun=tc.autofc, **PARAMS)
    with pytest.raises(MemoryError, match="max_memory_gb"):
        tc.features(x, max_memory_gb=0.01, **UNREDUCED, **kw)
    with pytest.raises(MemoryError, match="max_memory_gb"):
        tc.features(x, max_memory_gb=0.001, **STRENGTH, **kw)
    with pytest.raises(MemoryError, match="max_memory_gb"):
        tc.two_path_step(x, causal=True, max_memory_gb=0.001, **STRENGTH, **PARAMS)
    needed = estimate_memory_gb(30, 1, 400)[1]
    X = _quiet(tc.features, x, max_memory_gb=needed * 1.5, **STRENGTH,
               **{**kw, "order": 1})
    assert X[1].shape == (400, 30)
    # wisfc with fixed weights is read like the built-in function (kernels in
    # chunks of rows, not T x T arrays)
    data = [np.random.default_rng(s).standard_normal((3000, 6)) for s in range(3)]
    limit = estimate_memory_gb(6, 1, 3000, n_subjects=3, cfun=tc.wisfc)[1] * 1.2
    assert limit < estimate_memory_gb(6, 1, 3000, n_subjects=3, cfun=WISFC)[1] / 2
    X = _quiet(tc.features, data, order=1, causal=True, cfun=WISFC,
               max_memory_gb=limit, **STRENGTH, **PARAMS)
    assert len(X[1]) == 3


# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------

X0 = np.random.default_rng(0).standard_normal((60, 5))
OK = dict(causal=True, cfun=tc.autofc, **STRENGTH, **PARAMS)
FUNCTIONS = {
    "features": lambda x, **kw: tc.features(x, order=2, **kw),
    "two_path_step": tc.two_path_step,
}
both = pytest.mark.parametrize("function", sorted(FUNCTIONS))


@both
def test_backward_kernels_are_implied(function):
    f = FUNCTIONS[function]
    out = _quiet(f, X0, **OK)
    for side in ("all", "backward", "pre", "BACKWARD"):
        other = _quiet(f, X0, include_timepoints=side, **OK)
        for a, b in zip(out, other):
            assert np.array_equal(a, b, equal_nan=True)


@both
@pytest.mark.parametrize("side", ["forward", "post", "Forward"])
def test_forward_kernels_raise(function, side):
    with pytest.raises(ValueError, match="backward"):
        FUNCTIONS[function](X0, include_timepoints=side, **OK)


@both
@pytest.mark.parametrize("missing", ["weights_params", "delta_weights_params"])
def test_kernel_parameters_are_required_on_both_paths(function, missing):
    kw = {k: v for k, v in OK.items() if k != missing}
    with pytest.raises(ValueError, match=rf"causal=True.*\b{missing}"):
        FUNCTIONS[function](X0, **kw)
    # also when another kernel function is given
    names = {"weights_params": "weights_function",
             "delta_weights_params": "delta_weights_function"}
    with pytest.raises(ValueError, match=rf"causal=True.*\b{missing}"):
        FUNCTIONS[function](X0, **{names[missing]: tc.laplace_weights}, **kw)


@pytest.mark.parametrize("name", ["weights_params", "delta_weights_params"])
def test_order_maps_of_kernel_parameters_must_cover_every_order(name):
    kw = {**OK, name: {1: {"var": 25}, 3: {"var": 25}}}
    with pytest.raises(ValueError, match=rf"causal=True.*{name}.*order 2"):
        tc.features(X0, order=3, **kw)
    kw = {**OK, name: {1: {"var": 25}, 2: None}}
    with pytest.raises(ValueError, match=rf"causal=True.*{name}.*order 2"):
        tc.features(X0, order=2, **kw)
    kw = {**OK, name: {1: {"var": 25}, 2: {"var": 9}}}
    assert len(_quiet(tc.features, X0, order=2, **kw)) == 3


@both
def test_delta_kernels_raise(function):
    f = FUNCTIONS[function]
    kw = {k: v for k, v in OK.items() if k != "delta_weights_params"}
    with pytest.raises(ValueError, match="causal=True.*eye_weights"):
        f(X0, delta_weights_function=tc.eye_weights, **kw)
    kw = {k: v for k, v in OK.items() if k != "weights_params"}
    with pytest.raises(ValueError, match="causal=True.*eye_weights"):
        f(X0, weights_function=tc.eye_weights, **kw)
    with pytest.raises(ValueError, match="causal=True.*eye_weights"):
        f(X0, weights_function=tc.eye_weights, weights_params={}, **kw)


@both
def test_wisfc_needs_fixed_subject_weights(function):
    f = FUNCTIONS[function]
    data = [X0, X0[::-1].copy(), X0**2]
    kw = {**OK}
    for cfun in (tc.wisfc, functools.partial(tc.wisfc),
                 functools.partial(tc.wisfc, subject_weights=None)):
        kw["cfun"] = cfun
        with pytest.raises(ValueError, match="causal=True.*subject_weights"):
            f(data, **kw)
    kw["cfun"] = WISFC
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(data, **kw)
    assert not [c for c in caught if "guarantee" in str(c.message)]
    assert len(out[-1]) == 3


@both
def test_events_must_be_boundaries(function):
    f = FUNCTIONS[function]
    feats = np.random.default_rng(1).standard_normal((60, 3))
    with pytest.raises(ValueError, match="causal=True.*boundaries"):
        f(X0, events=feats, **OK)
    with pytest.raises(ValueError, match="causal=True.*boundaries"):
        f([X0, X0 + 1], events=[[30], feats], **OK)
    out = _quiet(f, X0, events=[30], **OK)
    assert np.isnan(np.asarray(out[-1])[30:32]).all()
    if function == "features":
        with pytest.raises(ValueError, match="causal=True.*boundaries"):
            tc.features(X0, order=2, events={2: feats}, **OK)
        with pytest.raises(ValueError, match="causal=True.*boundaries"):
            tc.features(X0, order=2, events={1: [30], 2: feats}, **OK)
        assert len(_quiet(tc.features, X0, order=2, events={2: [30]}, **OK)) == 3


FITTED = ["auto", "PCA", {"model": "PCA"}, "weighted_strength",
          "weighted_strength_split", "communities", tc.WeightedStrength(),
          tc.Communities(n=2)]


@both
@pytest.mark.parametrize("rfun", FITTED, ids=str)
@pytest.mark.parametrize("name", ["rfun", "delta_rfun"])
def test_fitted_reducers_need_fit_timepoints(function, name, rfun):
    kw = {**OK, name: rfun}
    with pytest.raises(ValueError, match=rf"{name}.*fit_timepoints"):
        FUNCTIONS[function](X0, **kw)


@both
def test_the_default_reducers_are_fitted(function):
    kw = {k: v for k, v in OK.items() if k not in ("rfun", "delta_rfun")}
    with pytest.raises(ValueError, match="fit_timepoints"):
        FUNCTIONS[function](X0, **kw)
    with pytest.raises(ValueError, match="fit_timepoints"):
        FUNCTIONS[function](X0, rfun="strength", **kw)
    with pytest.raises(ValueError, match="fit_timepoints"):
        FUNCTIONS[function](X0, delta_rfun="strength", **kw)


def test_fitted_reducers_in_order_maps_need_fit_timepoints():
    kw = {**OK, "rfun": {1: "strength", 2: "PCA"}}
    with pytest.raises(ValueError, match=r"rfun.*order 2.*fit_timepoints"):
        tc.features(X0, order=2, **kw)
    # an order missing from a map is "auto"
    kw = {**OK, "delta_rfun": {1: "strength"}}
    with pytest.raises(ValueError, match=r"delta_rfun.*order 2.*fit_timepoints"):
        tc.features(X0, order=2, **kw)
    kw = {**OK, "rfun": {1: "strength", 2: None}, "delta_rfun": {1: None, 2: "strength"}}
    assert len(_quiet(tc.features, X0, order=2, **kw)) == 3


@both
def test_functions_of_the_users_warn_once(function):
    f = FUNCTIONS[function]

    def my_cfun(data, weights):
        return tc.autofc(data, weights)

    def my_rfun(arrays):
        return [np.nan_to_num(a[:, :3]) for a in arrays]

    for kw, names in (({"cfun": my_cfun}, ["cfun"]), ({"rfun": my_rfun}, ["rfun"]),
                      ({"delta_rfun": my_rfun}, ["delta_rfun"]),
                      ({"cfun": my_cfun, "rfun": my_rfun}, ["cfun", "rfun"])):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f(X0, **{**OK, **kw})
        messages = [str(c.message) for c in caught if "guarantee" in str(c.message)]
        assert len(messages) == 1
        assert all(name in messages[0] for name in names)


@both
@pytest.mark.parametrize("bad", [-1, 0, 0.5, np.nan, np.inf, "3", True])
def test_min_samples_must_be_none_or_a_number_from_1(function, bad):
    with pytest.raises(ValueError, match="min_samples"):
        FUNCTIONS[function](X0, min_samples=bad, **OK)


@both
def test_min_samples_needs_causal_and_a_kernel_that_reaches_it(function):
    f = FUNCTIONS[function]
    kw = {k: v for k, v in OK.items() if k != "causal"}
    with pytest.raises(ValueError, match="causal=True"):
        f(X0, min_samples=5, **kw)
    # the feeding kernel (variance 16) has at most 7.6 effective samples: the
    # error comes before anything is computed, whatever the length
    for x in (X0, X0[:3]):
        with pytest.raises(ValueError, match="min_samples=8.*7.5"):
            f(x, min_samples=8, **OK)
    assert np.isnan(np.asarray(_quiet(f, X0[:3], min_samples=7, **OK)[-1])).all()


@both
def test_new_arguments_are_keyword_only_and_documented(function):
    import inspect

    f = getattr(tc, function)
    parameters = inspect.signature(f).parameters
    for name in ("causal", "min_samples"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert f"\n    {name} : " in f.__doc__
    assert parameters["causal"].default is False
    assert parameters["min_samples"].default is None
    assert "fit_timepoints" in f.__doc__


def test_default_is_not_causal():
    x = np.random.default_rng(2).standard_normal((80, 4))
    kw = dict(order=2, cfun=tc.autofc, return_delta=True, **STRENGTH, **PARAMS)
    a = _quiet(tc.features, x, **kw)
    b = _quiet(tc.features, x, causal=False, min_samples=None, **kw)
    for p, q in zip(a, b):
        for u, v in zip(p, q):
            assert np.array_equal(u, v)
    assert not np.isnan(a[0][2]).any()


# ----------------------------------------------------------------------------
# speed
# ----------------------------------------------------------------------------


def test_order_three_of_5033_by_49_takes_seconds():
    x = np.random.default_rng(0).standard_normal((5033, 49))
    kw = dict(order=3, rfun="strength", delta_rfun="strength",
              weights_params={"var": 25}, delta_weights_params={"var": 25})
    start = time.perf_counter()
    X = _quiet(tc.features, x, causal=True, **kw)
    causal = time.perf_counter() - start
    start = time.perf_counter()
    P = _quiet(tc.features, x, include_timepoints="backward", **kw)
    plain = time.perf_counter() - start
    print(f"\n5033 x 49, order 3: causal {causal:.2f} s, causal=False backward "
          f"{plain:.2f} s")
    assert X[3].shape == (5033, 49)
    assert np.array_equal(np.flatnonzero(np.isnan(X[3]).all(axis=1)), np.arange(6))
    assert not np.isnan(X[3][6:]).any()
    # the rows far from the start agree with the kernels of causal=False
    assert np.max(np.abs(X[3][800:] - P[3][800:])) <= ATOL
    assert causal < 15
