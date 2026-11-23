"""
Reducers fitted on the first rows and then frozen: ``fit_timepoints`` in
``timecorr``, ``features`` and ``two_path_step`` with ``causal=True``, and
``tc.FeaturesModel`` (``return_model=True``, ``transform``).

The guarantee: the reducers are fitted on rows ``[0, fit_timepoints)`` only
(a fit does not change, bit for bit, whatever follows those rows), and every
output row t is unchanged (to 1e-9, NaN rows included; bit for bit under
replacement) when the input rows after t and at or after ``fit_timepoints``
are replaced by other finite values, by NaN, by inf, or removed.  Real
computations; nothing is mocked.
"""

import inspect
import pickle
import re
import subprocess
import sys
import warnings

import hypertools as hyp
import numpy as np
import pytest

import timecorr as tc
from timecorr.higher_order import two_path_step

from .test_causal_estimator import (
    _as_list,
    _grid,
    _max_difference,
    _perturb,
    _quiet,
    _series,
)
from .test_causal_features import WISFC

ATOL = 1e-9
T, N = 240, 140  # timepoints, and the rows the reducers are fitted on
PARAMS = dict(weights_params={"var": 25}, delta_weights_params={"var": 16})
MODES = ("finite", "nan", "inf", "remove")


def _modular(seed, n=T, modules=4, per=3):
    """Modules of nodes that share a signal: communities for the feeding path."""
    rng = np.random.default_rng(seed)
    shared = np.repeat(rng.standard_normal((n, modules)), per, axis=1)
    return shared + 0.5 * rng.standard_normal((n, modules * per))


# name: (subjects, data, features, highest order, arguments)
CONFIGS = {
    # the defaults: PCA, then signed weighted strength; communities counted
    # with the shift null on the feeding path (found at order 1, and the
    # fallback to weighted strength at order 2)
    "auto": (1, _modular, 12, 3, dict(cfun=tc.autofc)),
    "auto-noise": (1, _series, 6, 3, dict(cfun=tc.autofc)),
    "auto-isfc": (3, _modular, 12, 2, dict()),
    "pca": (1, _series, 5, 3, dict(cfun=tc.autofc, rfun="PCA", delta_rfun=None)),
    "pca-order-4": (1, _series, 4, 4, dict(cfun=tc.autofc, rfun="PCA",
                                           delta_rfun=None)),
    "models": (3, _series, 5, 2, dict(rfun={"model": "IncrementalPCA"},
                                      delta_rfun="FactorAnalysis", ndims=3)),
    "strength": (1, _series, 6, 3, dict(
        cfun=tc.autofc, rfun=tc.WeightedStrength(signed=True),
        delta_rfun=tc.WeightedStrength())),
    "strength-split": (1, _modular, 12, 2, dict(
        cfun=tc.autofc, rfun=tc.WeightedStrength(split=True, n=3, center="median"),
        delta_rfun="weighted_strength")),
    "communities-2": (1, _modular, 12, 2, dict(
        cfun=tc.autofc, rfun="weighted_strength",
        delta_rfun={1: tc.Communities(n=2, signed=True), 2: "weighted_strength"})),
    "communities": (1, _modular, 12, 2, dict(
        cfun=tc.autofc, rfun={1: tc.Communities(n=4), 2: "weighted_strength"},
        delta_rfun={1: tc.Communities(), 2: "weighted_strength"})),
    "isfc": (3, _series, 5, 3, dict(rfun="weighted_strength",
                                    delta_rfun="weighted_strength")),
    "wisfc": (3, _series, 5, 3, dict(cfun=WISFC, rfun=tc.WeightedStrength(signed=True),
                                     delta_rfun="weighted_strength")),
    "wisfc-corrmean": (3, _series, 5, 2, dict(
        cfun=WISFC, combine=tc.corrmean_combine, rfun="PCA",
        delta_rfun="weighted_strength")),
    "autofc-subjects": (3, _series, 5, 2, dict(
        cfun=tc.autofc, rfun="weighted_strength",
        delta_rfun={1: tc.Communities(n=2), 2: "weighted_strength"})),
    "events": (1, _series, 6, 2, dict(
        cfun=tc.autofc, rfun="PCA", delta_rfun="weighted_strength",
        events=[60, 170], min_samples=5)),
    "mixed": (1, _series, 5, 3, dict(
        cfun=tc.autofc, rfun={1: None, 2: "PCA", 3: "strength"},
        delta_rfun={1: "strength", 2: "weighted_strength", 3: None})),
}
WITHOUT_ZERO = ["auto", "pca", "strength", "wisfc", "events", "communities"]


def _data(config, seed=0, n=T):
    n_subjects, make, K = CONFIGS[config][:3]
    kw = {} if make is _modular else {"K": K}
    data = [make(seed + s, n=n, **kw) for s in range(n_subjects)]
    return data[0] if n_subjects == 1 else data


def _map(data, f):
    return [f(x) for x in data] if isinstance(data, list) else f(data)


def _run(data, config, **kwargs):
    order, kw = CONFIGS[config][3:]
    return _quiet(tc.features, data, causal=True,
                  **{**dict(order=order, fit_timepoints=N, return_delta=True),
                     **PARAMS, **kw, **kwargs})


def _worst(full, part, rows):
    """Largest difference over every ``X[k]`` and ``D[k]``, rows ``:rows``."""
    return max(_max_difference(f[k], p[k], rows)
               for f, p in zip(full, part) for k in range(1, len(f)))


def _identical(full, part, rows=None):
    return all(np.array_equal(np.asarray(a)[:rows], np.asarray(b)[:rows],
                              equal_nan=True)
               for f, p in zip(full, part) for fk, pk in zip(f[1:], p[1:])
               for a, b in zip(_as_list(fk), _as_list(pk)))


def _same_fits(a, b):
    """Whether two models hold the same fits, bit for bit."""
    assert sorted(a.fits) == sorted(b.fits)
    for key in a.fits:
        p, q = a.fits[key], b.fits[key]
        if isinstance(p, dict):  # a summary's record
            assert sorted(p) == sorted(q)
            for name in p:
                u, v = p[name], q[name]
                if isinstance(u, (list, np.ndarray)):
                    if not all(np.array_equal(i, j) for i, j in zip(u, v)):
                        return False
                elif u != v:
                    return False
        elif hasattr(p, "model_"):  # a fitted hypertools model
            for name, u in vars(p.model_).items():
                if isinstance(u, np.ndarray) and not np.array_equal(
                        u, getattr(q.model_, name), equal_nan=True):
                    return False
        elif p is not q and p != q:
            return False
    return True


# ----------------------------------------------------------------------------
# the guarantee
# ----------------------------------------------------------------------------

# every configuration runs in every selection, those of WITHOUT_ZERO without
# the diagonal; with it they are slow (other configurations have the diagonal)
CASES = [pytest.param(c, True, marks=pytest.mark.slow if c in WITHOUT_ZERO else ())
         for c in sorted(CONFIGS)] + [(c, False) for c in WITHOUT_ZERO]


@pytest.mark.parametrize("config,include_zero", CASES)
def test_rows_do_not_depend_on_later_rows(config, include_zero):
    data = _data(config)
    full = _run(data, config, include_zero=include_zero)
    order = CONFIGS[config][3]
    assert len(full[0]) == order + 1 and len(full[1]) == order + 1
    last = np.asarray(_as_list(full[0][order])[0])
    # defined after the training rows (but for the warm-up of an event)
    assert np.isnan(last[0]).all() and np.isnan(last[N:]).all(axis=1).sum() < 20
    rng = np.random.default_rng(7)
    for mode in MODES:
        # rows before fit_timepoints are in-sample: they change with the
        # rows the reducers are fitted on, and with nothing after those
        for m in (N, N + 1, 200, T - 1) if mode == "remove" else (N, 200):
            part = _run(_map(data, lambda x: _perturb(x, m, mode, rng)), config,
                        include_zero=include_zero)
            # without the diagonal, row m itself may change too
            rows = m if include_zero or mode == "remove" else m + 1
            assert _worst(full, part, rows) <= ATOL
            if mode != "remove":
                # replacing later rows changes no bit
                assert _identical(full, part, rows)


@pytest.mark.parametrize("config", ["auto", "pca", "communities", "wisfc", "models"])
def test_the_fit_reads_the_first_rows_only(config):
    data = _data(config, seed=10)
    full, model = _run(data, config, return_model=True)
    assert model.fit_timepoints == N
    assert any(isinstance(f, dict) or hasattr(f, "model_") for f in model.fits.values())
    # the same fits from the first rows alone, and whatever follows them
    _, alone = _run(_map(data, lambda x: x[:N]), config, return_model=True)
    assert _same_fits(model, alone)
    rng = np.random.default_rng(11)
    for mode in ("finite", "nan", "inf"):
        _, other = _run(_map(data, lambda x: _perturb(x, N, mode, rng)), config,
                        return_model=True)
        assert _same_fits(model, other)
    # another row in the training span: other fits
    _, more = _run(data, config, return_model=True, fit_timepoints=N + 25)
    assert not _same_fits(model, more)


def test_the_fit_is_made_of_the_defined_training_rows():
    x = _series(12, n=T, K=6)
    kw = dict(causal=True, cfun=tc.autofc, min_samples=5, events=[40], **PARAMS)
    strength = tc.WeightedStrength(signed=True)
    (X, D), model = _quiet(tc.features, x, order=1, rfun="PCA", delta_rfun=strength,
                           fit_timepoints=N, return_delta=True, return_model=True, **kw)
    corrs = [_quiet(tc.timecorr, x[:N], causal=True, cfun=tc.autofc, min_samples=5,
                    events=[40], weights_params=PARAMS[name])
             for name in ("weights_params", "delta_weights_params")]
    defined = [~np.isnan(c).all(axis=1) for c in corrs]
    assert 5 < (~defined[0]).sum() < 30  # the warm-up of both blocks is left out
    # weighted strength: the mean squared (and mean) correlations of those rows
    r = tc.vec2mat(corrs[1][defined[1]])
    w = np.mean(r**2, axis=2)
    np.fill_diagonal(w, 0)
    m = np.mean(r, axis=2)
    np.fill_diagonal(m, 0)
    fit = model.fits[(1, "feeding")]
    assert np.allclose(fit["weights"][0], w, atol=1e-12, rtol=0)
    assert np.allclose(fit["mean_r"][0], m, atol=1e-12, rtol=0)
    assert strength.fits_[0]["weights"][0] is not None  # the caller's object too
    # PCA: hypertools' model of those rows, applied to every defined row
    _, pca = hyp.reduce(corrs[0][defined[0]], reduce="PCA", ndims=6, random_state=0,
                        return_model=True)
    every = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc, min_samples=5,
                   events=[40], **{"weights_params": PARAMS["weights_params"]})
    rows = ~np.isnan(every).all(axis=1)
    expected = np.full((T, 6), np.nan)
    expected[rows] = hyp.reduce(every[rows], reduce=pca)
    assert _max_difference(X[1], expected, T) <= ATOL
    assert np.array_equal(np.isnan(X[1]).all(axis=1), ~rows)


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["auto", "pca", "strength", "wisfc", "events",
                                    "communities-2", "models"])
def test_transform_of_the_fitted_data_is_the_output(config):
    data = _data(config, seed=20)
    full, info, model = _run(data, config, return_info=True, return_model=True)
    assert isinstance(model, tc.FeaturesModel) and model.info is info
    events = CONFIGS[config][4].get("events")
    again, again_info = _quiet(model.transform, data, return_delta=True,
                               return_info=True, events=events)
    assert _identical(full, again) and again_info == info
    # without the feeding features: X alone, the same arrays
    X = _quiet(model.transform, data, events=events)
    assert _identical((full[0],), (X,))
    # a pickled model gives the same output
    copy = pickle.loads(pickle.dumps(model))
    assert repr(copy) == repr(model) and _same_fits(copy, model)
    assert _identical(full, _quiet(copy.transform, data, return_delta=True,
                                   events=events))
    # rows do not depend on later rows: prefixes of every length, and
    # later rows replaced
    rng = np.random.default_rng(21)
    for n in (1, 2, 3, 7, 30, N - 1, N, N + 1, 150, 170, 171, 199, 223, T - 1):
        part = _quiet(model.transform, _map(data, lambda x: x[:n]),
                      return_delta=True, events=events)
        assert all(np.asarray(a).shape[0] == n for a in _as_list(part[0][-1]))
        assert _worst(full, part, n) <= ATOL
    for mode in ("finite", "nan", "inf"):
        for n in (5, 100, 200):
            part = _quiet(model.transform, _map(data, lambda x: _perturb(x, n, mode, rng)),
                          return_delta=True, events=events)
            assert _identical(full, part, n)


def test_transform_of_other_data():
    data = _data("auto", seed=30)
    X, model = _run(data, "auto", return_model=True, return_delta=False)
    other = _modular(31, n=90)
    out, info = _quiet(model.transform, other, return_info=True)
    assert [a.shape for a in out] == [(90,) + a.shape[1:] for a in X]
    assert np.array_equal(out[0], other)
    assert [e["warmup_rows"] for e in info] == [2, 2, 4, 4, 6]
    assert not np.isnan(out[3][6:]).any()
    # the frozen fits: the same columns as a call that fits on these rows too
    # would not give (other weights), and no fit is changed by transform
    before = pickle.dumps(model)
    _quiet(model.transform, other)
    assert pickle.dumps(model) == before
    # a list of one subject, and frames
    import pandas as pd

    as_list = _quiet(model.transform, [other])
    assert isinstance(as_list[1], list) and np.array_equal(as_list[3][0], out[3],
                                                           equal_nan=True)
    frame = _quiet(model.transform, pd.DataFrame(other))
    assert np.array_equal(frame[3], out[3], equal_nan=True)


def test_transform_checks_its_data():
    data = _data("wisfc", seed=32)
    X, model = _run(data, "wisfc", return_model=True, return_delta=False)
    assert (model.n_features, model.n_subjects, model.order) == (5, 3, 3)
    with pytest.raises(ValueError, match="5 features; got 4"):
        model.transform([x[:, :4] for x in data])
    # summaries hold pair weights per subject
    with pytest.raises(ValueError, match="pair weights of 3 subjects.*got 2"):
        _quiet(model.transform, data[:2])
    # the last feeding step was not computed by the call that made the model
    with pytest.raises(ValueError, match="return_delta=True"):
        model.transform(data, return_delta=True)
    with pytest.raises(ValueError, match="causal=True.*boundaries"):
        model.transform(data, events=np.zeros((T, 2)))
    with pytest.raises(MemoryError, match="max_memory_gb"):
        model.max_memory_gb = 1e-6
        model.transform(data)
    # a pipeline without summaries takes any number of subjects
    x = _series(33, n=T, K=5)
    _, pca = _quiet(tc.features, x, order=2, causal=True, cfun=tc.autofc, rfun="PCA",
                    delta_rfun="PCA", fit_timepoints=N, return_model=True, **PARAMS)
    out = _quiet(pca.transform, [x[:50], x[50:100]])
    assert len(out[2]) == 2 and out[2][0].shape == (50, 5)


def test_a_model_without_fitted_reducers():
    x = _series(34, n=T, K=5)
    kw = dict(order=2, causal=True, cfun=tc.autofc, rfun="strength",
              delta_rfun="strength", **PARAMS)
    X = _quiet(tc.features, x, **kw)
    same, model = _quiet(tc.features, x, return_model=True, **kw)
    assert model.fit_timepoints is None and _identical((X,), (same,))
    assert model.fits == {(1, "analysis"): "strength", (1, "feeding"): "strength",
                          (2, "analysis"): "strength"}
    assert _identical((X,), (_quiet(model.transform, x),))
    # fit_timepoints is not used by reducers that need no fit
    other, info = _quiet(tc.features, x, fit_timepoints=50, return_info=True, **kw)
    assert _identical((X,), (other,))
    assert all("fit_timepoints" not in e for e in info)
    zero, model = _quiet(tc.features, x, order=0, causal=True, return_model=True)
    assert len(zero) == 1 and model.order == 0 and model.fits == {}
    assert np.array_equal(model.transform(x[:9])[0], x[:9], equal_nan=True)


def test_the_model_describes_the_call():
    x = _modular(35)
    kw = dict(order=2, causal=True, cfun=tc.autofc, fit_timepoints=N,
              weights_params={1: {"var": 25}, 2: {"var": 49}},
              delta_weights_params={"var": 16}, min_samples=4, include_zero=False)
    X, model = _quiet(tc.features, x, return_model=True, **kw)
    assert repr(model) == ("FeaturesModel(order=2, fit_timepoints=140, n_features=12, "
                           "n_subjects=1)")
    assert model.weights_function is tc.gaussian_weights
    assert model.delta_weights_function is tc.gaussian_weights
    assert model.weights_params == {1: {"var": 25}, 2: {"var": 49}}
    assert model.delta_weights_params == {1: {"var": 16}, 2: {"var": 16}}
    assert model.cfun is tc.autofc and model.combine is tc.null_combine
    assert (model.min_samples, model.include_zero, model.ndims) == (4.0, False, None)
    assert model.max_memory_gb == 4.0
    assert sorted(model.fits) == [(1, "analysis"), (1, "feeding"), (2, "analysis")]
    # a hypertools model, a record of the communities, a record of the weights
    assert model.fits[(1, "analysis")].model_.components_.shape == (12, 78)
    assert model.fits[(1, "feeding")]["labels"].tolist() == np.repeat(
        np.arange(4), 3).tolist()
    assert model.fits[(2, "analysis")]["weights"][0].shape == (4, 4)
    # the caller's dictionaries are not the model's
    kw["weights_params"][1]["var"] = 1
    assert model.weights_params[1] == {"var": 25}
    # a model that holds a lambda cannot be pickled
    _, local = _quiet(tc.features, x, order=1, causal=True, fit_timepoints=N,
                      cfun=lambda data, weights: tc.autofc(data, weights),
                      return_model=True, **PARAMS)
    with pytest.raises(Exception, match="(?i)pickle|lambda"):
        pickle.dumps(local)


SCRIPT = """
import sys, warnings
import numpy as np
import timecorr as tc
warnings.simplefilter("ignore")
rng = np.random.default_rng(40)
shared = np.repeat(rng.standard_normal((240, 4)), 3, axis=1)
x = shared + 0.5 * rng.standard_normal((240, 12))
(X, D), model = tc.features(x, order=3, causal=True, cfun=tc.autofc,
                            weights_params={"var": 25},
                            delta_weights_params={"var": 16}, fit_timepoints=140,
                            return_delta=True, return_model=True)
Y = model.transform(x[:200])
np.savez(sys.argv[1], *X, *D, *Y)
"""


def test_the_same_arrays_in_two_processes(tmp_path):
    files = [str(tmp_path / f"run{i}.npz") for i in range(2)]
    for path in files:
        subprocess.run([sys.executable, "-c", SCRIPT, path], check=True)
    first, second = (np.load(path) for path in files)
    assert len(first.files) == 12 and first.files == second.files
    for name in first.files:
        assert np.array_equal(first[name], second[name], equal_nan=True)
    assert first["arr_3"].shape == (240, 8) and not np.isnan(first["arr_3"][6:]).any()


# ----------------------------------------------------------------------------
# what is fitted, and what is reported
# ----------------------------------------------------------------------------


def test_return_info_names_the_fitted_steps():
    x = _modular(50)
    kw = dict(order=2, causal=True, cfun=tc.autofc, return_delta=True, **PARAMS)
    (X, D), info = _quiet(tc.features, x, fit_timepoints=N, return_info=True,
                          rfun={1: "PCA", 2: "strength"},
                          delta_rfun={1: "auto", 2: None}, **kw)
    assert [(e["order"], e["path"], e.get("fit_timepoints")) for e in info] == [
        (1, "analysis", N), (1, "feeding", N), (2, "analysis", None),
        (2, "feeding", None)]
    assert all("weights_params" in e and "warmup_rows" in e for e in info)
    # the automatic feeding step: decided on the training rows
    step = info[1]
    assert step["fallback"] is None and step["n_communities"] == 4
    assert step["labels"] == np.repeat(np.arange(4), 3).tolist()
    assert step["effective"] == "Communities(n=None, signed=False, n_null=5)"
    assert D[1].shape == (T, 4)
    # unstructured data: the fallback, decided on the training rows too, with
    # the one warning of the fit
    noise = np.random.default_rng(51).standard_normal((T, 6))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (X, D), info = tc.features(noise, fit_timepoints=N, return_info=True, **kw)
    assert info[1]["fallback"] in ("no_null_exceedance", "too_few_community_columns")
    assert info[1]["effective"].startswith("WeightedStrength(") and D[1].shape == (T, 6)
    fallbacks = [c for c in caught if "Automatic feeding reducer" in str(c.message)]
    assert len(fallbacks) == sum(e.get("fallback") == "no_null_exceedance" for e in info)
    # the estimator's warning is issued once per step, for every row
    rows = [str(c.message) for c in caught if "timepoints have" in str(c.message)]
    assert len(rows) == 4 and all(f"of {T} timepoints" in r for r in rows)
    # calls without causal have no such keys
    _, info = _quiet(tc.features, x, order=1, return_info=True, **PARAMS)
    assert not {"fit_timepoints", "weights_params", "warmup_rows"} & set(info[0])


def test_constant_columns_are_reported_and_not_set_to_their_mean():
    # features 0 and 1 are equal up to a factor: their summaries agree to
    # round-off at every timepoint.  Without causal such a column of the
    # difference would be set to its mean, which reads every row.
    rng = np.random.default_rng(52)
    x = rng.standard_normal((T, 5))
    x[:, 4] = 3.0  # zero variance: weighted strength 0 at every row
    kw = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
              delta_rfun="weighted_strength", return_delta=True, return_info=True,
              **PARAMS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (X, D), info = tc.features(x, causal=True, fit_timepoints=N, **kw)
    assert not [c for c in caught if "constant over time" in str(c.message)]
    for e in info:
        out = (X if e["path"] == "analysis" else D)[e["order"]]
        defined = ~np.isnan(out).all(axis=1)
        assert e["constant_columns"] == np.flatnonzero(
            np.ptp(out[defined], axis=0) == 0).tolist()
        assert 4 in e["constant_columns"]
    assert np.all(X[1][2:, 4] == 0)
    # values are those of the frozen weights applied to each row's
    # correlations: nothing is replaced afterwards
    fit = tc.WeightedStrength()
    corrs = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc,
                   weights_params={"var": 25})
    fit.fit(corrs[2:N])
    expected = _quiet(fit.transform, corrs[2:])
    assert np.max(np.abs(X[1][2:] - expected)) <= 1e-12


def test_errors_of_the_training_rows():
    x = _modular(53)
    kw = dict(order=2, causal=True, cfun=tc.autofc, **PARAMS)
    # the default feeding reducer: 40 rows to be fitted on
    with pytest.raises(ValueError, match=r"40 timepoints.*fit_timepoints = 30"):
        tc.features(x, fit_timepoints=30, **kw)
    # the shift null of a later order: 40 rows of its input, which starts
    # with the warm-up rows of the order before
    later = {1: "weighted_strength", 2: tc.Communities()}
    with pytest.raises(ValueError, match="shift null needs T >= 40.*training span"):
        _quiet(tc.features, x, fit_timepoints=41, delta_rfun=later, **{**kw, "order": 3})
    assert len(_quiet(tc.features, x, fit_timepoints=42, delta_rfun=later,
                      **{**kw, "order": 3})) == 4
    # no defined row to fit: the first rows of every order are warm-up
    for reducers in (dict(rfun="PCA", delta_rfun="strength"),
                     dict(rfun="weighted_strength", delta_rfun="strength"),
                     dict(rfun="strength", delta_rfun="weighted_strength"),
                     dict(rfun="strength", delta_rfun=tc.Communities(n=2))):
        with pytest.raises(ValueError, match="fit_timepoints.*Raise fit_timepoints"):
            _quiet(tc.features, x, fit_timepoints=2, **reducers, **kw)
    with pytest.raises(ValueError, match="fit_timepoints.*Raise fit_timepoints"):
        _quiet(tc.features, x, fit_timepoints=4, rfun={1: "strength", 2: "PCA"},
               delta_rfun="strength", **kw)
    assert len(_quiet(tc.features, x, fit_timepoints=8, rfun="PCA",
                      delta_rfun="weighted_strength", **kw)) == 3


@pytest.mark.parametrize("model", ["MDS", "SpectralEmbedding", "LocallyLinearEmbedding"])
def test_models_that_cannot_embed_later_rows_raise(model):
    x = np.random.default_rng(54).standard_normal((80, 5))
    kw = dict(causal=True, cfun=tc.autofc, fit_timepoints=50,
              weights_params={"var": 25})
    with pytest.raises(ValueError, match=f"causal=True: the model '{model}'"):
        _quiet(tc.timecorr, x, rfun={"model": model}, **kw)
    with pytest.raises(ValueError, match=f"causal=True: the model '{model}'.*order 1"):
        _quiet(tc.features, x, order=1, rfun=model, delta_weights_params={"var": 16},
               **kw)
    # without causal they reduce the recording as before
    assert _quiet(tc.timecorr, x, cfun=tc.autofc, rfun=model).shape == (80, 5)


def test_model_columns_constant_on_the_training_rows_are_frozen():
    # a feature that is constant has correlation 0 with every other, so the
    # correlations of the training rows span fewer dimensions than PCA has
    # components: the last component's output is round-off (1e-16).  The
    # next order would correlate that noise, which is not reproducible when
    # rows are removed; the values are returned as the column's training
    # mean, exactly.
    x = _series(56, n=T, K=4)  # its feature 2 is 2 up to round-off
    kw = dict(causal=True, cfun=tc.autofc, rfun="PCA", delta_rfun="PCA",
              fit_timepoints=N, return_delta=True, **PARAMS)
    (X, D), info = _quiet(tc.features, x, order=1, return_info=True, **kw)
    assert info[1]["constant_columns"] == [3] and info[0]["constant_columns"] == [3]
    assert np.ptp(D[1][2:, 3]) == 0 and abs(D[1][2, 3]) < 1e-12
    assert np.all(np.ptp(D[1][2:, :3], axis=0) > 1)
    # at the next orders that column has zero variance at every row, and
    # the outputs do not depend on later rows
    full = _quiet(tc.features, x, order=3, **kw)
    rng = np.random.default_rng(57)
    for mode in MODES:
        for m in (N, 141, 200):
            part = _quiet(tc.features, _perturb(x, m, mode, rng), order=3, **kw)
            assert _worst(full, part, m) <= ATOL
    # the rule itself: values within round-off of the training mean are that
    # mean; any other value is the model's
    from timecorr.helpers import _model_reduce

    rng = np.random.default_rng(58)
    rows = rng.standard_normal((50, 3)) @ rng.standard_normal((3, 6))  # rank 3
    _, fitted = _quiet(_model_reduce, rows, "PCA", 5, return_model=True)
    plain = hyp.reduce(rows, reduce=fitted.model)
    assert 0 < np.ptp(plain[:, 3:], axis=0).max() < 1e-12
    fitted.freeze_constants([plain])
    frozen = fitted.apply([rows])[0]
    assert np.all(np.ptp(frozen[:, 3:], axis=0) == 0)
    assert np.array_equal(frozen[:, :3], plain[:, :3])
    other = rng.standard_normal((10, 6))  # rows outside that subspace
    assert np.array_equal(fitted.apply([other])[0], hyp.reduce(other, reduce=fitted.model))
    assert np.ptp(fitted.apply([other])[0][:, 3:], axis=0).min() > 0.1


def test_pca_is_fitted_with_an_exact_solver():
    # 520 training rows of 55 correlations: scikit-learn's default solver for
    # 10 components is the randomized one, whose fit_transform is not the
    # projection that transform computes
    x = np.random.default_rng(55).standard_normal((600, 10))
    kw = dict(causal=True, cfun=tc.autofc, weights_params={"var": 25},
              fit_timepoints=522)
    out = _quiet(tc.timecorr, x, rfun="PCA", **kw)
    corrs = _quiet(tc.timecorr, x, **{**kw, "fit_timepoints": None})
    rows = corrs[2:522]
    centered = rows - rows.mean(axis=0)
    # the principal components of the training rows: decreasing variances
    # that are the eigenvalues of their covariance
    variances = np.var(out[2:522], axis=0, ddof=1)
    expected = np.linalg.eigvalsh(centered.T @ centered / 519)[::-1][:10]
    assert np.allclose(variances, expected, rtol=1e-9, atol=0)
    same = _quiet(tc.timecorr, x, rfun={"model": "PCA", "kwargs": {"whiten": False}},
                  **kw)
    assert np.array_equal(out, same, equal_nan=True)
    with pytest.raises(ValueError, match="does not reproduce the rows"):
        _quiet(tc.timecorr, x, **kw,
               rfun={"model": "PCA", "kwargs": {"svd_solver": "randomized",
                                                "iterated_power": 0}})


# ----------------------------------------------------------------------------
# two_path_step and timecorr
# ----------------------------------------------------------------------------


def test_two_path_step_fits_on_the_first_rows():
    x = _modular(60)
    kw = dict(causal=True, cfun=tc.autofc, fit_timepoints=N, **PARAMS)
    X, D = _quiet(tc.features, x, order=2, return_delta=True, **kw)
    X1, D1 = _quiet(tc.two_path_step, x, **kw)
    assert np.array_equal(X1, X[1], equal_nan=True)
    assert np.array_equal(D1, D[1], equal_nan=True)
    # the next order: the fit skips the warm-up rows of its input (features
    # fits it on the feeding features of the first rows alone, which equal
    # these to round-off)
    X2, D2 = _quiet(tc.two_path_step, D1, order=2, **kw)
    assert np.isnan(X2[:4]).all() and not np.isnan(X2[4:]).any()
    assert _max_difference(X2, X[2], T) <= ATOL
    assert _max_difference(D2, D[2], T) <= ATOL
    rng = np.random.default_rng(61)
    for mode in MODES:
        for m in (N, 200):
            part = _quiet(tc.two_path_step, _perturb(x, m, mode, rng), **kw)
            assert max(_max_difference(f, p, m) for f, p in zip((X1, D1), part)) <= ATOL
    with pytest.raises(ValueError, match="more than the 240 timepoints"):
        tc.two_path_step(x, **{**kw, "fit_timepoints": T + 1})


TIMECORR_REDUCERS = {
    "PCA": lambda: "PCA",
    "model-dict": lambda: {"model": "FactorAnalysis", "params": {"n_components": 3}},
    "strength": lambda: tc.WeightedStrength(signed=True),
    "split": lambda: tc.WeightedStrength(split=True, n=2),
    "communities": lambda: tc.Communities(n=2),
    "name": lambda: "weighted_strength",
}


@pytest.mark.parametrize("subjects,reducer,include_zero", _grid(
    [1, 3], sorted(TIMECORR_REDUCERS), [True, False]))
def test_timecorr_fits_its_reducer_on_the_first_rows(subjects, reducer, include_zero):
    data = [_series(70 + s, n=T, K=5) for s in range(subjects)]
    data = data[0] if subjects == 1 else data
    rfun = TIMECORR_REDUCERS[reducer]()
    kw = dict(causal=True, weights_params={"var": 25}, rfun=rfun, fit_timepoints=N,
              include_zero=include_zero,
              cfun=tc.autofc if subjects == 1 else tc.isfc)
    full = _quiet(tc.timecorr, data, **kw)
    assert all(np.isnan(f[0]).all() and not np.isnan(f[N:]).any()
               for f in _as_list(full))
    rng = np.random.default_rng(71)
    for mode in MODES:
        for m in (N, 200) if mode != "remove" else (N, N + 1, T - 1):
            part = _quiet(tc.timecorr, _map(data, lambda x: _perturb(x, m, mode, rng)),
                          **kw)
            rows = m if include_zero or mode == "remove" else m + 1
            assert _max_difference(full, part, rows) <= ATOL
    if isinstance(rfun, tc.WeightedStrength) and subjects == 1:
        # the caller's object holds the fit of the training rows, to use again
        corrs = _quiet(tc.timecorr, data, **{**kw, "rfun": None})
        assert len(rfun.fits_) == 1
        again = _quiet(rfun.transform, corrs[5:])
        assert np.max(np.abs(again - full[5:])) <= ATOL


def test_timecorr_with_every_row_in_the_training_span():
    x = _series(72, n=T, K=5)
    kw = dict(causal=True, cfun=tc.autofc, weights_params={"var": 25}, rfun="PCA")
    out = _quiet(tc.timecorr, x, fit_timepoints=T, **kw)
    assert out.shape == (T, 5) and np.isnan(out[:2]).all()
    # the fit of every row is not causal for the rows before the last
    assert _max_difference(out, _quiet(tc.timecorr, x[:200], fit_timepoints=200, **kw),
                           200) > 1e-3
    # smoothing has no reducer: fit_timepoints is not used
    kw = dict(causal=True, cfun=None, weights_params={"var": 25})
    assert np.array_equal(_quiet(tc.timecorr, x, fit_timepoints=9, **kw),
                          _quiet(tc.timecorr, x, **kw), equal_nan=True)


# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------

X0 = np.random.default_rng(0).standard_normal((60, 5))
CALLS = {
    "timecorr": lambda **kw: tc.timecorr(X0, cfun=tc.autofc, rfun="PCA",
                                         weights_params={"var": 25}, **kw),
    "features": lambda **kw: tc.features(X0, order=2, cfun=tc.autofc, **PARAMS, **kw),
    "two_path_step": lambda **kw: tc.two_path_step(X0, cfun=tc.autofc, **PARAMS, **kw),
}
each = pytest.mark.parametrize("function", sorted(CALLS))


def _rows(out):
    """Rows of the last output (an array, or the highest order of a list)."""
    return (out if isinstance(out, np.ndarray) else np.asarray(out[-1])).shape[0]


@each
def test_fitted_reducers_need_fit_timepoints(function):
    with pytest.raises(ValueError, match="causal=True.*needs fit_timepoints=n"):
        CALLS[function](causal=True)
    assert _rows(_quiet(CALLS[function], causal=True, fit_timepoints=50)) == 60
    # every row in the rows the reducers are fitted on
    assert _rows(_quiet(CALLS[function], causal=True, fit_timepoints=60)) == 60


@each
@pytest.mark.parametrize("bad", [0, -3, 2.0, "10", True, np.nan])
def test_fit_timepoints_is_a_positive_integer(function, bad):
    with pytest.raises(ValueError, match="fit_timepoints must be None or a positive"):
        CALLS[function](causal=True, fit_timepoints=bad)


@each
def test_fit_timepoints_needs_causal_and_rows(function):
    with pytest.raises(ValueError, match="fit_timepoints=30 applies with causal=True"):
        CALLS[function](fit_timepoints=30)
    with pytest.raises(ValueError, match="fit_timepoints=61 is more than the 60"):
        CALLS[function](causal=True, fit_timepoints=61)
    assert _rows(_quiet(CALLS[function], causal=True,
                        fit_timepoints=np.int64(50))) == 60


def test_return_model_needs_causal():
    with pytest.raises(ValueError, match="return_model=True applies with causal=True"):
        tc.features(X0, order=1, return_model=True)
    out = _quiet(tc.features, X0, order=1, causal=True, cfun=tc.autofc,
                 fit_timepoints=50, return_delta=True, return_info=True,
                 return_model=True, **PARAMS)
    (X, D), info, model = out
    assert len(X) == 2 and len(D) == 2 and len(info) == 2
    assert isinstance(model, tc.FeaturesModel)
    assert np.array_equal(_quiet(model.transform, X0, return_delta=True)[1][1], D[1],
                          equal_nan=True)


@pytest.mark.parametrize("function,names", [
    (tc.timecorr, ["fit_timepoints"]),
    (tc.features, ["fit_timepoints", "refit_every", "return_model"]),
    (two_path_step, ["fit_timepoints"]),
])
def test_new_arguments_are_keyword_only_and_documented(function, names):
    parameters = inspect.signature(function).parameters
    for name in names:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert f"\n    {name} : " in function.__doc__
    assert parameters["fit_timepoints"].default is None
    assert list(parameters)[-len(names):] == names
    for section in ("Parameters", "Returns", "Examples"):
        assert section in tc.FeaturesModel.transform.__doc__
    assert "Examples" in tc.FeaturesModel.__doc__
    assert tc.FeaturesModel.__module__ == "timecorr.model"


# ----------------------------------------------------------------------------
# models: only linear maps applied row by row
# ----------------------------------------------------------------------------

LINEAR = ["PCA", "IncrementalPCA", "TruncatedSVD", "FastICA"]


@pytest.mark.parametrize("model", ["UMAP", "KernelPCA", "Isomap"])
def test_models_without_components_raise(model):
    # reproducing its training rows does not make a model's transform one
    # of each row alone (UMAP's depends on the number of rows it is given):
    # only models with components_, a linear map, are taken
    x = np.random.default_rng(60).standard_normal((100, 4))
    kw = dict(causal=True, cfun=tc.autofc, fit_timepoints=40,
              weights_params={"var": 25})
    rfun = {"model": model, "kwargs": {"n_neighbors": 5}} if model != "KernelPCA" else model
    with pytest.raises(ValueError, match=f"causal=True: the model '{model}'.*components_"):
        tc.timecorr(x, rfun=rfun, **kw)
    kw["ndims"] = 2
    for name in ("rfun", "delta_rfun"):
        other = "delta_rfun" if name == "rfun" else "rfun"
        with pytest.raises(ValueError, match=f"causal=True: the model '{model}' "
                                             rf"\({name}.*order 1.*components_"):
            tc.features(x, order=1, delta_weights_params={"var": 16},
                        **{name: rfun, other: "strength"}, **kw)
        with pytest.raises(ValueError, match=f"causal=True: the model '{model}'"):
            two_path_step(x, delta_weights_params={"var": 16},
                          **{name: rfun, other: "strength"}, **kw)
    with pytest.raises(ValueError, match="components_"):
        tc.features(x, order=1, delta_weights_params={"var": 16}, rfun=rfun,
                    delta_rfun="strength", refit_every=20, **kw)


def test_a_model_whose_transform_is_not_row_by_row_raises():
    from sklearn.decomposition import PCA

    class BatchCentered(PCA):
        """Components, but an output that depends on the other rows given."""

        def transform(self, X):
            out = super().transform(X)
            return out - out.mean(axis=0)

    class Unfitted(PCA):
        """No components_ once fitted."""

        def fit(self, X, y=None):
            super().fit(X, y)
            self.axes_ = self.components_
            del self.components_
            return self

        def fit_transform(self, X, y=None):
            return self.fit(X).transform(X)

        def transform(self, X):
            return (X - self.mean_) @ self.axes_.T

    x = np.random.default_rng(61).standard_normal((100, 4))
    kw = dict(causal=True, cfun=tc.autofc, fit_timepoints=40,
              weights_params={"var": 25})
    with pytest.raises(ValueError, match="row by row"):
        _quiet(tc.timecorr, x, rfun={"model": BatchCentered}, **kw)
    with pytest.raises(ValueError, match=r"order 1.*row by row"):
        _quiet(tc.features, x, order=1, rfun={"model": BatchCentered}, ndims=2,
               delta_rfun="strength", delta_weights_params={"var": 16}, **kw)
    with pytest.raises(ValueError, match="components_"):
        _quiet(tc.timecorr, x, rfun={"model": Unfitted}, **kw)
    # without causal they reduce the recording
    assert _quiet(tc.timecorr, x, cfun=tc.autofc,
                  rfun={"model": BatchCentered}).shape == (100, 4)


GUARANTEED_MODELS = LINEAR + ["FactorAnalysis"]
NOT_COVERED = "causal=True: the model .* is not one of the models the guarantee covers"
SOME_NOT_COVERED = "causal=True: the models .* not among the models the guarantee covers"


def _model_warnings(f, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*args, **kwargs)
    return out, [str(w.message) for w in caught
                 if "the models the guarantee covers" in str(w.message)]


def test_a_model_outside_the_list_runs_with_one_warning():
    from sklearn.decomposition import PCA

    class BatchPCA(PCA):
        """Passes every probe of the fit (they reduce training rows only)
        and reads the rows it is given together when there are more."""

        def transform(self, X):
            out = super().transform(X)
            if len(X) > self.n_samples_:
                out -= out.mean(axis=0)
            return out

    class SamePCA(PCA):
        """A subclass that changes nothing is still not the listed class."""

    x = np.random.default_rng(61).standard_normal((100, 4))
    kw = dict(causal=True, cfun=tc.autofc, fit_timepoints=40,
              weights_params={"var": 25})
    listed = ("PCA, IncrementalPCA, TruncatedSVD, FastICA and FactorAnalysis")
    for model in (BatchPCA, SamePCA, "MiniBatchDictionaryLearning"):
        out, messages = _model_warnings(tc.timecorr, x, rfun={"model": model}, **kw)
        # (rows of BatchPCA may depend on later rows: that is what the
        # warning says, and nothing is asserted about them)
        assert out.shape == (100, 4)
        assert len(messages) == 1
        assert re.search(NOT_COVERED, messages[0]) and listed in messages[0]
        assert getattr(model, "__name__", model) in messages[0]
        assert "rfun" in messages[0]
    # a model of another name that cannot reduce one row alone: no probe
    # shows that it reduces rows one by one
    with pytest.raises(ValueError, match="'SparsePCA'.*row by row.*one row alone"):
        _quiet(tc.timecorr, x, rfun="SparsePCA", **kw)
    # features and two_path_step: once for the call, whatever the number of
    # orders, paths and fits
    more = dict(delta_weights_params={"var": 16}, ndims=2, **kw)
    # (a refit cannot match the components of BatchPCA to the fit before;
    # those of SamePCA, a class that cannot be pickled, it reorders in a copy)
    for first, extra in ((BatchPCA, {}), (SamePCA, {}), (SamePCA, {"refit_every": 20})):
        out, messages = _model_warnings(
            tc.features, x, order=2, rfun={"model": first},
            delta_rfun={"model": SamePCA}, **more, **extra)
        assert out[2].shape == (100, 2)
        assert len(messages) == 1
        assert re.search(SOME_NOT_COVERED if first is BatchPCA else NOT_COVERED,
                         messages[0])
        for name in (first.__name__, "SamePCA", "rfun at order 1",
                     "delta_rfun at order 2"):
            assert name in messages[0]
    out, messages = _model_warnings(
        two_path_step, x, rfun="PCA", delta_rfun={"model": BatchPCA}, **more)
    assert len(messages) == 1 and "delta_rfun" in messages[0]
    assert "(rfun" not in messages[0]
    # without causal: no warning
    assert _model_warnings(tc.timecorr, x, cfun=tc.autofc,
                           rfun={"model": BatchPCA})[1] == []


@pytest.mark.parametrize("model", GUARANTEED_MODELS)
def test_the_listed_models_do_not_warn(model):
    import sklearn.decomposition

    cls = getattr(sklearn.decomposition, model)
    x = np.random.default_rng(65).standard_normal((100, 4))
    kw = dict(causal=True, cfun=tc.autofc, fit_timepoints=40,
              weights_params={"var": 25})
    for rfun in (model, {"model": model}, {"model": cls}):
        assert _model_warnings(tc.timecorr, x, rfun=rfun, **kw)[1] == []
    (out, fitted), messages = _model_warnings(
        tc.features, x, order=2, rfun=model, delta_rfun={"model": model}, ndims=3,
        delta_weights_params={"var": 16}, refit_every=30, return_model=True, **kw)
    assert messages == []
    # the list is of exact classes: what hypertools fits for these names
    assert type(fitted.fits[(1, "analysis")].model_) is cls
    for name in ("timecorr", "features", "two_path_step"):
        doc = {"timecorr": tc.timecorr, "features": tc.features,
               "two_path_step": two_path_step}[name].__doc__
        assert f'``"{model}"``' in doc


@pytest.mark.parametrize("model", GUARANTEED_MODELS)
def test_linear_models_are_taken_and_guaranteed(model):
    x = np.random.default_rng(62).standard_normal((T, 5))
    kw = dict(order=2, causal=True, cfun=tc.autofc, rfun=model, delta_rfun=model,
              ndims=3, fit_timepoints=N, return_delta=True, **PARAMS)
    full, fitted = _quiet(tc.features, x, return_model=True, **kw)
    assert hasattr(fitted.fits[(1, "analysis")].model_, "components_")
    rng = np.random.default_rng(63)
    worst = 0.0
    for mode in MODES:
        for m in (N, 190):
            part = _quiet(tc.features, _perturb(x, m, mode, rng), **kw)
            worst = max(worst, _worst(full, part, m))
            if mode != "remove":
                assert _identical(full, part, m)
    assert worst <= ATOL
    # the model on a series and on its first rows
    z = np.random.default_rng(64).standard_normal((300, 5))
    whole = _quiet(fitted.transform, z, return_delta=True)
    part = _quiet(fitted.transform, z[:120], return_delta=True)
    assert _worst(whole, part, 120) <= ATOL


# ----------------------------------------------------------------------------
# a subject without a defined row keeps its place
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("rfun", [tc.WeightedStrength(), "PCA", "strength"], ids=str)
def test_a_subject_without_defined_rows_keeps_its_place(rfun):
    import functools

    x = np.random.default_rng(3).normal(size=(80, 4))
    # a combine function of the user's: the step is not streamed
    kw = dict(order=1, causal=True, cfun=tc.autofc, rfun=rfun, delta_rfun=rfun,
              combine=functools.partial(tc.null_combine), fit_timepoints=40,
              return_delta=True, **PARAMS)
    _, model = _quiet(tc.features, [x, 2 * x], return_model=True, **kw)
    y = x.copy()
    y[:60] = np.nan
    some = _quiet(model.transform, [y, x], return_delta=True)
    gone = y.copy()
    gone[60:] = np.nan  # later rows missing too: no defined row is left
    none = _quiet(model.transform, [gone, x], return_delta=True)
    for a, b in zip(some, none):
        assert len(b[1]) == 2 and b[1][0].shape == a[1][0].shape
        assert np.isnan(b[1][0]).all()
        # replacing later rows by NaN changes no bit of the rows before
        assert np.array_equal(a[1][0][:60], b[1][0][:60], equal_nan=True)
        assert np.array_equal(a[1][1], b[1][1], equal_nan=True)
        assert not np.isnan(a[1][1][5:]).any()


def test_features_with_a_subject_without_defined_rows():
    x = np.random.default_rng(3).normal(size=(80, 4))
    y = x.copy()
    y[:60] = np.nan
    gone = np.full_like(x, np.nan)
    # a model is fitted to the rows of all subjects: one of them suffices
    kw = dict(order=2, causal=True, cfun=tc.autofc, rfun="PCA", delta_rfun="PCA",
              ndims=3, fit_timepoints=40, return_delta=True, **PARAMS)
    some = _quiet(tc.features, [y, x], **kw)
    none = _quiet(tc.features, [gone, x], **kw)
    for a, b in zip(some, none):
        for k in (1, 2):
            assert np.isnan(b[k][0]).all() and b[k][0].shape == a[k][0].shape
            assert np.array_equal(a[k][0][:60], b[k][0][:60], equal_nan=True)
            assert np.array_equal(a[k][1], b[k][1], equal_nan=True)
    # a summary is fitted to each subject: no defined row, no fit
    kw.update(rfun="weighted_strength", delta_rfun="weighted_strength", ndims=None)
    with pytest.raises(ValueError, match="fit_timepoints"):
        _quiet(tc.features, [gone, x], **kw)
    kw["combine"] = __import__("functools").partial(tc.null_combine)
    with pytest.raises(ValueError, match="fit_timepoints"):
        _quiet(tc.features, [gone, x], **kw)


# ----------------------------------------------------------------------------
# a model and the reducer objects of the caller are independent
# ----------------------------------------------------------------------------


def _snapshot(summary):
    import copy

    return copy.deepcopy(summary.fits_)


def _same_records(a, b):
    assert len(a) == len(b)
    for p, q in zip(a, b):
        assert sorted(p) == sorted(q)
        for name in p:
            u, v = p[name], q[name]
            if isinstance(u, (list, np.ndarray)):
                if not all(np.array_equal(i, j) for i, j in zip(u, v)):
                    return False
            elif u != v:
                return False
    return True


@pytest.mark.parametrize("streamed", [True, False])
@pytest.mark.parametrize("refit", [None, 50])
def test_a_model_does_not_share_its_fits_with_the_reducer_objects(streamed, refit):
    import copy
    import functools

    x = _modular(70)
    strength, communities = tc.WeightedStrength(signed=True), tc.Communities(n=3)
    kw = dict(order=2, causal=True, cfun=tc.autofc, rfun=strength,
              delta_rfun={1: communities, 2: "weighted_strength"},
              fit_timepoints=N, refit_every=refit, return_delta=True, **PARAMS)
    if not streamed:
        kw["combine"] = functools.partial(tc.null_combine)
    out, info, model = _quiet(tc.features, x, return_info=True, return_model=True, **kw)
    assert model.info is info
    kept_info = copy.deepcopy(info)
    before = _quiet(model.transform, x, return_delta=True)
    # transforming with the model does not touch the objects
    records = _snapshot(strength), _snapshot(communities)
    _quiet(model.transform, x[:150], return_delta=True)
    assert _same_records(strength.fits_, records[0])
    assert _same_records(communities.fits_, records[1])
    # changing the objects' records changes nothing in the model
    for record in strength.fits_:
        record["weights"][0][:] = 0
        record["mean_r"][0][:] = -1
    for record in communities.fits_:
        record["weights"][0][:] = 0
        record["labels"][:] = 0
    assert _identical(before, _quiet(model.transform, x, return_delta=True))
    # nor does fitting them again, here or in another call
    strength.fit(_quiet(tc.timecorr, _modular(71), cfun=tc.autofc, rfun=None,
                        weights_params={"var": 9}))
    _quiet(tc.features, _modular(72), **{**kw, "fit_timepoints": 60})
    assert _identical(before, _quiet(model.transform, x, return_delta=True))
    assert model.info == kept_info and info == kept_info
    assert _identical(before, _quiet(pickle.loads(pickle.dumps(model)).transform, x,
                                     return_delta=True))


def test_timecorr_does_not_share_a_fit_with_the_reducer_object():
    x = _modular(73)
    strength = tc.WeightedStrength()
    kw = dict(causal=True, cfun=tc.autofc, rfun=strength, fit_timepoints=N,
              weights_params={"var": 25})
    out = _quiet(tc.timecorr, x, **kw)
    record = _snapshot(strength)
    assert np.array_equal(_quiet(tc.timecorr, x, **kw), out, equal_nan=True)
    assert _same_records(strength.fits_, record)


# ----------------------------------------------------------------------------
# reductions that change nothing
# ----------------------------------------------------------------------------


def test_a_model_with_as_many_components_as_columns_is_the_identity():
    # hypertools returns such data as they are, and fits no model
    x = np.random.default_rng(80).standard_normal((120, 4))
    one = dict(causal=True, cfun=tc.autofc, fit_timepoints=60,
               weights_params={"var": 25})
    plain = _quiet(tc.timecorr, x, rfun=None, **one)
    assert plain.shape == (120, 10)
    two = dict(order=1, delta_rfun="strength", delta_weights_params={"var": 16}, **one)
    assert np.array_equal(_quiet(tc.features, x, rfun="PCA", ndims=10, **two)[1], plain,
                          equal_nan=True)
    # more components than columns: zero columns are appended, as without causal
    padded = _quiet(tc.features, x, rfun="PCA", ndims=12, **two)[1]
    assert padded.shape == (120, 12)
    assert np.array_equal(padded[:, :10], plain, equal_nan=True)
    defined = ~np.isnan(plain).all(axis=1)
    assert np.all(padded[defined, 10:] == 0) and np.isnan(padded[~defined]).all()
    usual = _quiet(tc.features, x, order=1, cfun=tc.autofc, rfun="PCA", ndims=12,
                   delta_rfun="strength", weights_params={"var": 25})[1]
    assert usual.shape == (120, 12) and np.all(usual[:, 10:] == 0)
    # one column: one correlation, itself
    single = _quiet(tc.timecorr, x[:, :1], rfun="PCA", **one)
    assert np.array_equal(single, _quiet(tc.timecorr, x[:, :1], rfun=None, **one),
                          equal_nan=True)
    assert single.shape == (120, 1)


@pytest.mark.parametrize("refit", [None, 20])
@pytest.mark.parametrize("columns,ndims", [(4, 10), (1, None)])
def test_identity_reductions_through_the_orders(columns, ndims, refit):
    x = np.random.default_rng(81).standard_normal((120, columns))
    kw = dict(order=2, causal=True, cfun=tc.autofc, ndims=ndims, fit_timepoints=60,
              refit_every=refit, return_delta=True, return_info=True,
              return_model=True, **PARAMS)
    out, info, model = _quiet(tc.features, x, rfun="PCA", delta_rfun="PCA", **kw)
    # order 1 reduces 10 columns to 10 (1 to 1): nothing; order 2 reduces 55
    # columns to 10 with a fitted model (1 to 1: nothing again)
    same, same_info, _ = _quiet(tc.features, x, **kw, **(
        dict(rfun={1: None, 2: "PCA"}, delta_rfun={1: None, 2: "PCA"})
        if columns == 4 else dict(rfun=None, delta_rfun=None)))
    assert _identical(out, same)
    assert [e["output_width"] for e in info] == [e["output_width"] for e in same_info]
    assert info[0]["output_width"] == (10 if columns == 4 else 1)
    rows = [60] if refit is None else [60, 80, 100]
    assert info[0]["fit_timepoints"] == 60 and info[0]["refit_rows"] == rows
    assert info[0]["refit_kept_previous"] == []
    assert model.refits == rows
    assert _identical(_quiet(model.transform, x, return_delta=True),
                      _quiet(pickle.loads(pickle.dumps(model)).transform, x,
                             return_delta=True))
    if refit is None:
        assert _identical(out, _quiet(model.transform, x, return_delta=True))
    # the guarantee
    cut = _quiet(tc.features, x[:90], rfun="PCA", delta_rfun="PCA", **kw)[0]
    assert _worst(out, cut, 90) <= ATOL
