"""
``refit_every`` of ``features`` with ``causal=True``: the reducers are fitted
on the first ``fit_timepoints`` rows and again every ``refit_every`` rows,
each time on every row before.

What is checked: rows ``[s_j, s_{j+1})`` of the output are those of a call
with ``fit_timepoints=s_j`` alone (exactly, where no rule changes a fit; up
to the matching of columns otherwise); the structure of the first fit is
kept; the columns of a fit are matched to those of the fit before; and the
guarantee of the causal mode holds (a row is unchanged when later rows are
replaced or removed).  Real computations; nothing is mocked.
"""

import pickle
import re
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.helpers import _FittedModel, _FitUnusable
from timecorr.higher_order import two_path_step
from timecorr.model import _assignment, _match_labels, _match_model

from .test_causal_estimator import _as_list, _max_difference, _perturb, _quiet, _series
from .test_causal_fit import _identical, _map, _modular, _worst

ATOL = 1e-9
T, N, M = 200, 100, 40
ROWS = [100, 140, 180]  # the fits: N + j * M below T
SEGMENTS = [(100, 140), (140, 180), (180, 200)]
PARAMS = dict(causal=True, weights_params={"var": 25},
              delta_weights_params={"var": 16})

def _drifting(seed, n=T, modules=4, per=3, weak=3):
    """Modules of nodes that share a signal, and ``weak`` nodes that load on
    a module with a strength that drifts: a weak node is a cluster of its
    own in one fit and a member of a community in the next, so the number
    of clusters does not fix the number of communities of 2 or more nodes."""
    rng = np.random.default_rng(31 + seed)
    signals = rng.standard_normal((n, modules))
    x = np.repeat(signals, per, axis=1) + 0.5 * rng.standard_normal((n, modules * per))
    rng.choice([1.0, 0.0], size=(1, weak), p=[0.5, 0.5])
    load = np.abs(np.cumsum(rng.standard_normal((n, weak)) * 0.08, axis=0) + 0.45)
    extra = load * signals[:, rng.integers(0, modules, weak)]
    return np.hstack([x, extra + rng.standard_normal((n, weak))])


# name: (subjects, data, features, highest order, arguments)
CONFIGS = {
    # the defaults: PCA, then signed weighted strength; communities counted
    # with the shift null on the feeding path
    "auto": (1, _modular, 12, 3, dict(cfun=tc.autofc)),
    "auto-isfc": (3, _modular, 12, 2, dict()),
    "pca": (1, _series, 5, 3, dict(cfun=tc.autofc, rfun="PCA", delta_rfun="PCA")),
    "pca-subjects": (3, _series, 5, 2, dict(rfun="PCA", delta_rfun="PCA", ndims=3)),
    "strength": (1, _series, 6, 3, dict(
        cfun=tc.autofc, rfun=tc.WeightedStrength(signed=True),
        delta_rfun=tc.WeightedStrength())),
    "strength-isfc": (3, _series, 5, 2, dict(rfun="weighted_strength",
                                             delta_rfun="weighted_strength")),
    "split": (1, _modular, 12, 2, dict(
        cfun=tc.autofc, rfun=tc.WeightedStrength(split=True, n=3, center="median"),
        delta_rfun="weighted_strength")),
    "mixed": (1, _series, 5, 3, dict(
        cfun=tc.autofc, rfun={1: None, 2: "weighted_strength", 3: "strength"},
        delta_rfun={1: "strength", 2: "weighted_strength", 3: None})),
    "communities": (1, _modular, 12, 2, dict(
        cfun=tc.autofc, rfun={1: tc.Communities(n=4), 2: "weighted_strength"},
        delta_rfun={1: tc.Communities(), 2: "weighted_strength"})),
    # six clusters give 5 communities of 2 or more nodes on the first 100
    # rows and 4 on the first 140 and 180, where five clusters give 5
    "drifting": (1, _drifting, 15, 2, dict(
        cfun=tc.autofc, rfun="weighted_strength",
        delta_rfun={1: tc.Communities(n=6), 2: "weighted_strength"})),
}
# no rule changes a fit: nothing is matched, no structure is carried over
EXACT = ["strength", "strength-isfc", "split", "mixed"]
# (every configuration, and both values of include_zero, run in every
# selection; the second value of three configurations is slow)
_SLOW = pytest.mark.slow
GUARANTEED = [pytest.param("auto", True, marks=_SLOW), ("auto", False),
              ("auto-isfc", True), ("pca", True),
              pytest.param("pca", False, marks=_SLOW), ("pca-subjects", True),
              pytest.param("strength", True, marks=_SLOW), ("strength", False),
              ("strength-isfc", False), ("drifting", True), ("drifting", False)]


def _data(config, seed=0, n=T):
    n_subjects, make, K = CONFIGS[config][:3]
    kw = {"K": K} if make is _series else {}
    data = [make(seed + s, n=n, **kw) for s in range(n_subjects)]
    return data[0] if n_subjects == 1 else data


def _run(data, config, **kwargs):
    order, kw = CONFIGS[config][3:]
    return _quiet(tc.features, data, **{
        **dict(order=order, fit_timepoints=N, refit_every=M, return_delta=True),
        **PARAMS, **kw, **kwargs})


def _rows(out, start, stop):
    """Rows ``start:stop`` of every ``X[k]`` and ``D[k]``, k >= 1."""
    return [[[np.asarray(a)[start:stop] for a in _as_list(o)] for o in part[1:]]
            for part in out]


def _same_rows(a, b, start, stop):
    return all(np.array_equal(u, v, equal_nan=True)
               for p, q in zip(_rows(a, start, stop), _rows(b, start, stop))
               for x, y in zip(p, q) for u, v in zip(x, y))


def _rows_difference(a, b, start, stop):
    return max(_max_difference(u, v, None)
               for p, q in zip(_rows(a, start, stop), _rows(b, start, stop))
               for x, y in zip(p, q) for u, v in zip(x, y))


def _cosines(a, b):
    return (a @ b.T) / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))


# ----------------------------------------------------------------------------
# the definition: each segment is a call with that fit alone
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("config", EXACT)
def test_each_segment_is_the_call_with_that_fit_alone(config):
    """Exactly: these reducers are neither matched nor given a structure."""
    data = _data(config)
    out, info = _run(data, config, return_info=True)
    fitted = [e for e in info if "fit_timepoints" in e]
    assert fitted
    for entry in fitted:
        assert entry["fit_timepoints"] == N
        assert entry["refit_rows"] == ROWS
        assert entry["refit_kept_previous"] == []
    for j, (start, stop) in enumerate(SEGMENTS):
        alone = _run(data, config, fit_timepoints=start, refit_every=None)
        assert _same_rows(out, alone, start, stop)
        if j == 0:
            # the rows before the first fit's row are the first fit's
            assert _same_rows(out, alone, 0, N)
        else:
            assert not _same_rows(out, alone, 0, start)


@pytest.mark.parametrize("config", ["pca", "auto", "communities", "auto-isfc"])
def test_each_segment_is_the_model_of_the_fits_so_far(config):
    """
    With matched columns a call with one fit is not the reference (its
    columns are in its own order).  The model of the call on the rows before
    ``s_{j+1}`` holds fit ``j`` as matched to the fits before it: its
    transform of all the data gives rows ``[s_j, s_{j+1})`` (to 1e-9: the
    rows after were removed for that call).
    """
    data = _data(config)
    out, model = _run(data, config, return_model=True)
    assert model.refits == ROWS
    for start, stop in SEGMENTS:
        _, prefix = _run(_map(data, lambda x: x[:stop]), config, return_model=True)
        assert prefix.refits == [r for r in ROWS if r < stop]
        applied = _quiet(prefix.transform, data, return_delta=True)
        assert _rows_difference(out, applied, start, stop) <= ATOL
    # the model of the whole call holds the last fit: the same arrays
    applied = _quiet(model.transform, data, return_delta=True)
    assert _same_rows(out, applied, ROWS[-1], T)


def test_model_columns_are_those_of_single_fits_up_to_order_and_sign():
    """PCA of the correlations of unmatched feeding features (weighted
    strength): at every order, each column of a segment is a column of the
    call with that fit alone, or its negative."""
    x = _series(3, n=T, K=5, missing=0.0)
    kw = dict(order=2, cfun=tc.autofc, rfun="PCA", delta_rfun="weighted_strength",
              fit_timepoints=N, **PARAMS)
    X = _quiet(tc.features, x, refit_every=M, **kw)
    moved = 0
    for start, stop in SEGMENTS:
        alone = _quiet(tc.features, x, **{**kw, "fit_timepoints": start})
        for k in (1, 2):
            a, b = X[k][start:stop], alone[k][start:stop]
            assert not np.isnan(a).any()
            # which column of the single fit each column is
            distance = np.minimum(
                np.abs(a[:, :, None] - b[:, None, :]).max(axis=0),
                np.abs(a[:, :, None] + b[:, None, :]).max(axis=0))
            source = distance.argmin(axis=1)
            assert sorted(source) == list(range(a.shape[1]))
            assert distance[np.arange(a.shape[1]), source].max() <= ATOL
            moved += not np.array_equal(a, b)
    assert moved  # the matching did something in this data


# ----------------------------------------------------------------------------
# matching
# ----------------------------------------------------------------------------


def _swapping(seed=2, n=T):
    """Two pairs of columns whose correlations vary; the second pair's vary
    most after row 100, so the leading components of later fits change."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    a = 0.6 * np.sin(t / 9.0)
    b = np.where(t < 100, 0.15 * np.sin(t / 7.0), 0.95 * np.sign(np.sin(t / 6.0)))
    z, e = rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
    return np.column_stack([
        z[:, 0], a * z[:, 0] + np.sqrt(1 - a ** 2) * e[:, 0],
        z[:, 1], b * z[:, 1] + np.sqrt(1 - b ** 2) * e[:, 1]])


def test_components_keep_their_columns_and_signs_across_refits():
    x = _swapping()
    kw = dict(order=1, cfun=tc.autofc, rfun="PCA", delta_rfun="weighted_strength",
              ndims=2, **PARAMS)
    key = (1, "analysis")

    def components(n, **more):
        _, model = _quiet(tc.features, x[:n + 1], return_model=True,
                          **{**kw, "fit_timepoints": n, **more})
        return model.fits[key].model_.components_

    # the fits alone: the second fit's components are the first's in the
    # other order, one of them negated
    alone = _cosines(components(140), components(100))
    assert list(_assignment(np.abs(alone))) == [1, 0]
    assert alone[0, 1] > 0.5 and alone[1, 0] < -0.5
    # matched: each stays where it was, with its sign
    first = components(100)
    second = components(140, **{"fit_timepoints": 100, "refit_every": 40})
    matched = _cosines(second, first)
    assert list(_assignment(np.abs(matched))) == [0, 1]
    assert matched[0, 0] > 0.5 and matched[1, 1] > 0.5
    # the same vectors as the fit alone, reordered and negated
    assert np.allclose(np.abs(_cosines(second, components(140))),
                       [[0, 1], [1, 0]], atol=1e-12)
    # so each column goes on measuring what it measured: over the second
    # fit's rows it follows the first fit's column, which the second column
    # of the fit alone does not
    X = _quiet(tc.features, x, fit_timepoints=100, refit_every=40, **kw)[1]
    before = _quiet(tc.features, x, fit_timepoints=100, **kw)[1]
    unmatched = _quiet(tc.features, x, fit_timepoints=140, **kw)[1]

    def follows(a, column):
        return np.corrcoef(a[140:180, column], before[140:180, column])[0, 1]

    assert follows(X, 0) > 0.6 and follows(X, 1) > 0.6
    assert follows(unmatched, 1) < 0.3


@pytest.mark.parametrize("rfun", [
    "PCA", {"model": "PCA", "kwargs": {"whiten": True}}, "IncrementalPCA",
    "FactorAnalysis", "TruncatedSVD", "FastICA"], ids=str)
def test_a_matched_model_gives_the_reordered_columns(rfun):
    from timecorr.timecorr import _fit_reducer

    rng = np.random.default_rng(0)
    rows = rng.standard_normal((60, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    new = _quiet(_fit_reducer, rows, rfun, 3)
    assert isinstance(new, _FittedModel)
    # an earlier fit with these components in another order, one negated
    previous = pickle.loads(pickle.dumps(new))
    estimator = previous.model.model_
    estimator.components_ = estimator.components_[[2, 0, 1]] * np.array([[1], [-1], [1]])
    matched = _match_model(new, previous, 1, "analysis")
    assert matched is not new
    assert np.allclose(np.diag(_cosines(matched.model.model_.components_,
                                        estimator.components_)), 1, atol=1e-12)
    other = rng.standard_normal((20, 6))
    before, after = new.apply([other])[0], matched.apply([other])[0]
    assert np.allclose(after, before[:, [2, 0, 1]] * np.array([1, -1, 1]),
                       rtol=0, atol=1e-9 * np.abs(before).max())
    # nothing to move: the fit itself
    assert _match_model(new, pickle.loads(pickle.dumps(new)), 1, "analysis") is new


def test_frozen_constant_columns_follow_their_components():
    from timecorr.timecorr import _fit_reducer

    rng = np.random.default_rng(1)
    rows = rng.standard_normal((40, 6))
    rows[:, 4] = rows[:, 5] = rows[:, 3]  # rank 4: the fifth component is round-off
    new = _quiet(_fit_reducer, rows, "PCA", 5)
    assert list(new.constants[0]) == [4]
    previous = pickle.loads(pickle.dumps(new))
    estimator = previous.model.model_
    estimator.components_ = estimator.components_[[4, 1, 2, 3, 0]]
    matched = _match_model(new, previous, 1, "analysis")
    assert list(matched.constants[0]) == [0]
    assert np.array_equal(matched.apply([rows])[0][:, 0], new.apply([rows])[0][:, 4])
    # padded columns stay last; another number of them is another structure
    padded = _quiet(_fit_reducer, rows[:3], "PCA", 5)
    assert padded.pad == 2
    with pytest.raises(_FitUnusable):
        _match_model(new, padded, 1, "analysis")


def test_labels_take_the_columns_of_the_communities_they_share_most_nodes_with():
    previous = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    # node 0 moved: numbered by smallest member, every community has another label
    new = np.array([0, 1, 1, 2, 2, 2, 0, 0, 0])
    assert list(_match_labels(new, previous)) == [2, 0, 0, 1, 1, 1, 2, 2, 2]
    assert list(_match_labels(previous, previous)) == list(previous)
    # single-node communities have no column and keep their labels
    assert list(_match_labels(np.array([0, 1, 1, 2, 3, 3]),
                              np.array([4, 3, 3, 2, 0, 0]))) == [0, 3, 3, 2, 1, 1]
    with pytest.raises(_FitUnusable, match="communities of 2 or more"):
        _match_labels(np.array([0, 0, 0, 0, 1, 2]), np.array([0, 0, 1, 1, 2, 2]))


def _moving_node(n=T):
    """Three modules of three nodes; from row 100 node 0 follows the third."""
    rng = np.random.default_rng(1)
    signals = rng.standard_normal((n, 3))
    x = np.repeat(signals, 3, axis=1) + 0.5 * rng.standard_normal((n, 9))
    x[100:, 0] = 3 * (signals[100:, 2] + 0.2 * rng.standard_normal(n - 100))
    return x


def test_communities_keep_their_columns_across_refits():
    x = _moving_node()
    kw = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
              delta_rfun={1: tc.Communities(n=3), 2: "weighted_strength"},
              return_delta=True, return_info=True, **PARAMS)

    def run(**more):
        (X, D), info = _quiet(tc.features, x, **kw, **more)
        return X, D, info[1]

    _, D100, first = run(fit_timepoints=100)
    _, D180, alone = run(fit_timepoints=180)
    assert first["labels"] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # alone, the fit at row 180 numbers the communities in another order
    assert alone["labels"] == [0, 1, 1, 2, 2, 2, 0, 0, 0]
    X, D, entry = run(fit_timepoints=100, refit_every=40)
    assert entry["refit_rows"] == ROWS and entry["refit_kept_previous"] == []
    # the labels of the last fit: node 0 with the third module, in its column
    assert entry["labels"] == [2, 0, 0, 1, 1, 1, 2, 2, 2]
    # the same columns as the fit alone, reordered, at order 1 ...
    assert np.array_equal(D[1][180:], D180[1][180:][:, [1, 2, 0]])
    assert np.array_equal(D[1][:140], D100[1][:140], equal_nan=True)
    # ... and order 2 was fitted on the reordered columns
    alone2 = _quiet(tc.features, x, **{**kw, "fit_timepoints": 180})[0][0][2]
    assert _max_difference(X[2][180:], alone2[180:][:, [1, 2, 0]], None) <= ATOL


# ----------------------------------------------------------------------------
# the structure is the first fit's
# ----------------------------------------------------------------------------


def _feeding(x, **kwargs):
    (X, D), info = _quiet(tc.features, x, order=1, cfun=tc.autofc, return_delta=True,
                          return_info=True, **PARAMS, **kwargs)
    return D[1], info[1]


def test_a_fallback_of_the_first_fit_stays():
    # noise, then modules: later fits alone would emit their communities
    x = np.vstack([np.random.default_rng(2).standard_normal((100, 12)),
                   _modular(3, n=200)])
    D0, first = _feeding(x, fit_timepoints=100)
    assert first["fallback"] == "no_null_exceedance" and D0.shape[1] == 12
    D200, alone = _feeding(x, fit_timepoints=200)
    assert alone["fallback"] is None and D200.shape[1] == 4
    D, entry = _feeding(x, fit_timepoints=100, refit_every=100)
    assert entry["refit_rows"] == [100, 200] and entry["refit_kept_previous"] == []
    assert entry["fallback"] == "no_null_exceedance" and entry["labels"] is None
    assert entry["effective"] == first["effective"] and D.shape == (300, 12)
    assert np.array_equal(D[:200], D0[:200], equal_nan=True)
    # from row 200: weighted strength with the pair weights of rows 0..199
    again, _ = _feeding(x, fit_timepoints=200, delta_rfun=tc.WeightedStrength(
        split=False, signed=False, center=False))
    assert np.array_equal(D[200:], again[200:])
    assert not np.array_equal(D[200:], D0[200:])


def test_communities_of_the_first_fit_stay():
    # weak modules, then noise: later fits alone fall back to weighted strength
    rng = np.random.default_rng(0)
    modules = np.repeat(rng.standard_normal((100, 4)), 3, axis=1)
    x = np.vstack([modules + rng.standard_normal((100, 12)),
                   np.random.default_rng(9).standard_normal((200, 12))])
    D0, first = _feeding(x, fit_timepoints=100)
    assert first["fallback"] is None and first["n_communities"] == 4
    D200, alone = _feeding(x, fit_timepoints=200)
    assert alone["fallback"] == "too_few_community_columns" and D200.shape[1] == 12
    D, entry = _feeding(x, fit_timepoints=100, refit_every=100)
    assert entry["fallback"] is None and entry["n_communities"] == 4
    assert D.shape == (300, 4) and len(set(entry["labels"])) >= 4
    assert entry["null_threshold"] == first["null_threshold"]  # no second null
    assert np.array_equal(D[:200], D0[:200], equal_nan=True)


def _merging(n=T):
    """Three pairs of nodes; from row 100 four nodes share one signal and
    two are noise, and three clusters leave fewer than three pairs."""
    rng = np.random.default_rng(5)
    x = np.repeat(rng.standard_normal((n, 3)), 2, axis=1)
    x = x + rng.standard_normal((n, 6))
    common = rng.standard_normal(n)
    x[100:, :4] = 4 * common[100:, None] + 0.3 * rng.standard_normal((n - 100, 4))
    x[100:, 4:] = rng.standard_normal((n - 100, 2))
    return x


def test_a_refit_without_the_structure_keeps_the_fit_before():
    x = _merging()
    kw = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
              delta_rfun={1: tc.Communities(n=3), 2: "weighted_strength"},
              return_delta=True, return_info=True, **PARAMS)
    first, info0 = _quiet(tc.features, x, fit_timepoints=100, **kw)
    _, alone = _quiet(tc.features, x, fit_timepoints=140, **kw)
    assert info0[1]["output_width"] == 3 and alone[1]["output_width"] == 2
    out, info, model = _quiet(tc.features, x, fit_timepoints=100, refit_every=40,
                              return_model=True, **kw)
    for entry in info:
        assert entry["refit_rows"] == ROWS
        # of every order: the whole fit is left out
        assert entry["refit_kept_previous"] == [140, 180]
        assert entry["fit_timepoints"] == 100
    assert info[1]["labels"] == info0[1]["labels"] == [0, 0, 1, 1, 2, 2]
    assert _same_rows(out, first, 0, T)
    assert model.refits == ROWS
    assert _same_rows(_quiet(model.transform, x, return_delta=True), first, 0, T)
    # whether a fit is kept is decided by the rows before it
    cut, cut_info = _quiet(tc.features, x[:150], fit_timepoints=100, refit_every=40,
                           **kw)
    assert cut_info[0]["refit_kept_previous"] == [140]
    assert _rows_difference(out, cut, 0, 150) <= ATOL


# ----------------------------------------------------------------------------
# the guarantee
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("config,include_zero", GUARANTEED)
def test_rows_do_not_depend_on_later_rows(config, include_zero):
    data = _data(config)
    full = _run(data, config, include_zero=include_zero)
    order = CONFIGS[config][3]
    last = np.asarray(_as_list(full[0][order])[0])
    assert np.isnan(last[0]).all() and not np.isnan(last[N:]).any()
    rng = np.random.default_rng(7)
    # replaced from the first fit's row (the rows before it depend on
    # nothing after them), at a refit row and inside segments; removed also
    # just before and just after a refit row
    cuts = {"finite": (N, 140, 185), "nan": (140,), "inf": (120,),
            "remove": (N, 139, 140, 141, 181, T - 1)}
    for mode in cuts:
        for m in cuts[mode]:
            part = _run(_map(data, lambda x: _perturb(x, m, mode, rng)), config,
                        include_zero=include_zero)
            # without the diagonal, row m itself may change too
            rows = m if include_zero or mode == "remove" else m + 1
            assert _worst(full, part, rows) <= ATOL
            if mode != "remove":
                # replacing later rows changes no bit
                assert _identical(full, part, rows)


def _kept_warnings(f, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*args, **kwargs)
    return out, [str(w.message) for w in caught
                 if str(w.message).startswith("refit_every:")]


def test_a_refit_looks_for_the_columns_of_the_first_fit_not_its_clusters():
    x = _data("drifting")
    kw = dict(return_info=True, return_model=True)
    widths = []
    for n in ROWS:  # single fits, each with six clusters
        _, info, model = _run(x, "drifting", fit_timepoints=n, refit_every=None, **kw)
        assert model.fits[(1, "feeding")]["n"] == 6
        widths.append(info[1]["output_width"])
    assert widths == [5, 4, 4]
    (out, info, model), messages = _kept_warnings(_run, x, "drifting", **kw)
    # no fit is left out, and nothing warns
    assert messages == []
    for entry in info:
        assert entry["refit_rows"] == ROWS and entry["refit_kept_previous"] == []
    assert out[1][1].shape == (T, 5)
    fit = model.fits[(1, "feeding")]
    # the last fit: n0 + 1 = 7 clusters give 4 such communities, n0 - 1 = 5
    # give the 5 of the first fit
    assert fit["n"] == 5 and int(np.sum(np.bincount(fit["labels"]) >= 2)) == 5
    assert len(set(info[1]["labels"])) == 5 and info[1]["output_width"] == 5
    first = _run(x, "drifting", refit_every=None)
    assert _same_rows(out, first, 0, ROWS[1])
    for start, stop in SEGMENTS[1:]:
        assert not _same_rows(out, first, start, stop)
    # each segment is the model of the fits so far (as for any matched fit)
    for start, stop in SEGMENTS:
        _, prefix = _run(x[:stop], "drifting", return_model=True)
        applied = _quiet(prefix.transform, x, return_delta=True)
        assert _rows_difference(out, applied, start, stop) <= ATOL


def test_a_fit_that_is_left_out_warns_once_with_its_rows_and_step():
    x = _merging()
    kw = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
              delta_rfun={1: tc.Communities(n=3), 2: "weighted_strength"}, **PARAMS)
    (_, info), messages = _kept_warnings(
        tc.features, x, fit_timepoints=100, refit_every=40, return_info=True, **kw)
    assert info[0]["refit_kept_previous"] == [140, 180]
    assert len(messages) == 1
    assert re.search(r"refit_every: 2 of the 2 fits after the first .*"
                     r"row 140 \(order 1, feeding path\): 2 communities of 2 or more "
                     r"nodes.*never the 3 of the fit before.*"
                     r"row 180 \(order 1, feeding path\)", messages[0])
    assert "refit_kept_previous" in messages[0]
    # the fit at row 140 alone: one row, and no warning without a refit
    (_, info), messages = _kept_warnings(
        tc.features, x[:150], fit_timepoints=100, refit_every=40, return_info=True, **kw)
    assert info[0]["refit_kept_previous"] == [140]
    assert len(messages) == 1 and "row 140 (order 1, feeding path)" in messages[0]
    assert _kept_warnings(tc.features, x, fit_timepoints=100, **kw)[1] == []
    for text in ("n0 + 1", "n0 - 5", "communities of 2 or more nodes",
                 "one ``UserWarning`` for the call names"):
        assert text in " ".join(tc.features.__doc__.split())


@pytest.mark.slow
def test_the_defaults_refit_on_the_industry_returns():
    # 5033 x 49 daily returns (a 4 MB download, cached).  With the defaults
    # every refit was left out: the first fit's 14 clusters hold 10
    # communities of 2 or more industries, and 14 clusters of later rows 9
    x = tc.datasets.industry_returns("2005-01-01", "2024-12-31").values
    assert x.shape == (5033, 49)
    kw = dict(order=2, causal=True, fit_timepoints=2517,
              weights_params={"var": 400}, delta_weights_params={"var": 400})
    (X, info, model), messages = _kept_warnings(
        tc.features, x, refit_every=1000, return_info=True, return_model=True, **kw)
    kept = info[0]["refit_kept_previous"]
    print(f"\nindustry returns: refit_rows {info[0]['refit_rows']}, "
          f"refit_kept_previous {kept}, clusters of the last fit "
          f"{model.fits[(1, 'feeding')]['n']}, columns {info[1]['output_width']}")
    assert info[0]["refit_rows"] == [2517, 3517, 4517]
    assert len(kept) < 2  # at least one refit is used
    assert len(messages) == (1 if kept else 0)
    assert all(str(row) in messages[0] for row in kept)
    assert info[1]["output_width"] == 10
    one = _quiet(tc.features, x, **kw)
    last = [r for r in info[0]["refit_rows"][1:] if r not in kept][-1]
    assert np.array_equal(X[2][:3517], one[2][:3517], equal_nan=True) or 3517 not in kept
    assert not np.array_equal(X[2][last:], one[2][last:], equal_nan=True)


def test_the_fits_made_depend_on_the_length_only_through_their_rows():
    data = _data("pca")
    kw = dict(return_info=True, return_model=True)
    for n, rows in ((140, [100]), (141, [100, 140]), (180, [100, 140]),
                    (181, [100, 140, 180])):
        _, info, model = _run(data[:n], "pca", **kw)
        assert model.refits == rows and info[0]["refit_rows"] == rows
        assert model.fit_timepoints == N
    # no row for a second fit: the call with one fit
    one = _run(data[:140], "pca")
    assert _same_rows(one, _run(data[:140], "pca", refit_every=None), 0, 140)
    assert _same_rows(one, _run(data[:140], "pca", refit_every=1000), 0, 140)


def test_refit_every_row():
    x = _series(4, n=70, K=4, missing=0.0)
    kw = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
              delta_rfun="weighted_strength", **PARAMS)
    X, info = _quiet(tc.features, x, fit_timepoints=60, refit_every=1,
                     return_info=True, **kw)
    assert info[0]["refit_rows"] == list(range(60, 70))
    for t in (60, 64, 69):
        alone = _quiet(tc.features, x, fit_timepoints=t, **kw)
        assert np.array_equal(X[2][t], alone[2][t])


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["auto", "pca", "strength"])
def test_the_model_holds_the_last_fit(config):
    data = _data(config)
    out, info, model = _run(data, config, return_info=True, return_model=True)
    assert isinstance(model, tc.FeaturesModel)
    assert model.fit_timepoints == N and model.refits == ROWS
    assert model.info is info
    applied, applied_info = _quiet(model.transform, data, return_delta=True,
                                   return_info=True)
    # from the last fit's row on: what features returned
    assert _same_rows(out, applied, ROWS[-1], T)
    # before it, features returned the rows of earlier fits ...
    assert not _same_rows(out, applied, N, ROWS[-1])
    assert not _same_rows(out, applied, 0, N)
    # ... which differ only through the fit: the last fit's are those of the
    # model of that fit alone, up to the matching of its columns (none for
    # weighted strength)
    if config == "strength":
        alone = _run(data, config, fit_timepoints=ROWS[-1], refit_every=None)
        assert _same_rows(applied, alone, 0, T)
    fitted = [e for e in applied_info if "fit_timepoints" in e]
    assert fitted and all(e["refit_rows"] == ROWS for e in fitted)
    # other data, causally
    longer = _data(config, seed=20, n=T + 30)
    more = _quiet(model.transform, longer, return_delta=True)
    part = _quiet(model.transform, longer[:T], return_delta=True)
    assert _rows_difference(more, part, 0, T) <= ATOL
    # a pickled model gives the same arrays
    copy = pickle.loads(pickle.dumps(model))
    assert copy.refits == ROWS and copy.fit_timepoints == N
    assert _same_rows(_quiet(copy.transform, longer, return_delta=True), more, 0, T + 30)


def test_info_and_model_without_refits():
    x = _series(5, n=T, K=5)
    kw = dict(order=2, cfun=tc.autofc, rfun={1: "PCA", 2: "strength"},
              delta_rfun="strength", **PARAMS)
    X, info, model = _quiet(tc.features, x, fit_timepoints=N, return_info=True,
                            return_model=True, **kw)
    assert info[0]["refit_rows"] == [N] and info[0]["refit_kept_previous"] == []
    assert all("refit_rows" not in e for e in info[1:])  # nothing fitted there
    assert model.refits == [N]
    # reducers without a fit: nothing to fit again
    kw["rfun"] = "strength"
    X, info, model = _quiet(tc.features, x, fit_timepoints=N, refit_every=M,
                            return_info=True, return_model=True, **kw)
    assert model.refits == [] and all("refit_rows" not in e for e in info)
    alone = _quiet(tc.features, x, **kw)
    assert np.array_equal(X[2], alone[2], equal_nan=True)


def test_reducer_objects_keep_the_first_fit():
    x = _series(6, n=T, K=5, missing=0.0)
    strength = tc.WeightedStrength()
    kw = dict(order=1, cfun=tc.autofc, rfun=strength, delta_rfun="strength",
              fit_timepoints=N, **PARAMS)
    _, model = _quiet(tc.features, x, refit_every=M, return_model=True, **kw)
    assert len(strength.fits_) == 1
    refitted = strength.fits_[0]["weights"][0].copy()
    _quiet(tc.features, x, **kw)
    assert np.array_equal(strength.fits_[0]["weights"][0], refitted)
    assert not np.array_equal(model.fits[(1, "analysis")]["weights"][0], refitted)


# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------

OK = dict(order=1, cfun=tc.autofc, rfun="weighted_strength", delta_rfun="strength",
          **PARAMS)
X0 = np.random.default_rng(0).standard_normal((80, 4))


def test_refit_every_needs_causal_and_fit_timepoints():
    with pytest.raises(ValueError, match=r"refit_every=10.*fit_timepoints"):
        tc.features(X0, refit_every=10, **OK)
    plain = {k: v for k, v in OK.items() if k != "causal"}
    with pytest.raises(ValueError, match=r"refit_every=10.*causal=True"):
        tc.features(X0, refit_every=10, **plain)
    with pytest.raises(TypeError):
        two_path_step(X0, refit_every=10, fit_timepoints=40, **{
            k: v for k, v in OK.items() if k != "order"})
    with pytest.raises(TypeError):
        tc.timecorr(X0, causal=True, weights_params={"var": 25}, fit_timepoints=40,
                    rfun="weighted_strength", refit_every=10)


@pytest.mark.parametrize("value", [0, -3, 2.5, 10.0, "10", True, [10], np.nan])
def test_refit_every_is_a_positive_integer(value):
    with pytest.raises(ValueError, match="refit_every must be None or a positive"):
        tc.features(X0, fit_timepoints=40, refit_every=value, **OK)


def test_refit_every_takes_numpy_integers_and_none():
    a = _quiet(tc.features, X0, fit_timepoints=40, refit_every=np.int64(20), **OK)
    b = _quiet(tc.features, X0, fit_timepoints=40, refit_every=20, **OK)
    assert np.array_equal(a[1], b[1], equal_nan=True)
    c = _quiet(tc.features, X0, fit_timepoints=40, refit_every=None, **OK)
    assert not np.array_equal(a[1], c[1], equal_nan=True)


def test_new_argument_is_keyword_only_and_documented():
    import inspect

    parameter = inspect.signature(tc.features).parameters["refit_every"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert "\n    refit_every : " in tc.features.__doc__
    assert "refit_every" not in inspect.signature(two_path_step).parameters
    assert "refit_every" not in inspect.signature(tc.timecorr).parameters
    for name in ("refit_rows", "refit_kept_previous"):
        assert name in tc.features.__doc__
    assert "refits" in tc.FeaturesModel.__doc__


def test_warnings_are_not_repeated_for_every_fit():
    x = _series(8, n=T, K=5)
    kw = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
              delta_rfun="weighted_strength", fit_timepoints=N, **PARAMS)
    counts = []
    for more in ({}, {"refit_every": M}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tc.features(x, **kw, **more)
        counts.append(len(caught))
    assert counts[0] == counts[1]
