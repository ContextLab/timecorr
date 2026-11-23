"""
Causal features whose reducers are fitted once and then frozen.

``features(..., causal=True, fit_timepoints=n)`` runs two passes of the
two-path pipeline, both with the causal estimator:

1. the fit: the pipeline on the first ``n`` rows alone, every order.  Each
   reducer that is fitted to data (a hypertools model, ``WeightedStrength``,
   ``Communities`` with its shift null, the automatic feeding reducer and its
   fallback) is fitted to the correlations of those rows, and each order's
   feeding features of those rows, reduced with the new fit, are the input
   of the next order's fit.  Later rows are not read, so the fits of a
   series and of any longer series with the same first ``n`` rows are equal
   bit for bit.
2. the output: the pipeline on every row, with the fits frozen.

``FeaturesModel`` holds the settings and the fits of pass 1;
``FeaturesModel.transform`` is pass 2 on other data.  ``_single_fit`` is
pass 1 and ``_causal_pass`` either pass.

``refit_every=m`` repeats both passes for the fits at rows ``s_j = n + j m``
(``_refit``): fit ``j`` is pass 1 on the first ``s_j`` rows, with the
structure of the first fit (``_refit_settings``) and every reducer's fit
matched to the fit before as soon as it is made (``_Matched``), so that
the next order of that fit is fitted on the matched columns; rows
``[s_j, s_{j+1})`` of the output are those of pass 2 on every row with fit
``j``.  The structure is the output columns: a communities step looks for
a clustering with the first fit's number of communities of 2 or more nodes
(``_same_width``).  A fit that cannot have the structure of the first one
(``helpers._FitUnusable``) is left out: the fit before stays for its rows,
and one warning for the call names the rows left out.
"""

import copy

import numpy as np
from scipy.optimize import linear_sum_assignment

from .events import _has_features, _parse_spec, resolve_events_by_order
from .helpers import (
    _SUMMARY_NAMES,
    _FittedModel,
    _FitUnusable,
    _warn_user,
    format_data,
    gaussian_weights,
)
from .higher_order import (
    _constant_columns,
    _describe,
    _fit_pass,
    _output_elements,
    _step_info,
    _two_path_step,
    _warmup_rows,
)
from .summaries import (
    Communities,
    _auto_fallback,
    _find_communities,
    _FittedSummary,
    _from_name,
    _Summary,
)
from .timecorr import _needs_fit

__all__ = ["FeaturesModel"]


def _is_fit(frozen):
    """True for what a fitted reducer leaves (a frozen summary or model)."""
    return isinstance(frozen, (_FittedSummary, _FittedModel))


def _events_by_order(events, T, n_subjects, order):
    """The boundaries of each order for a causal pass over T rows (those at
    or beyond T are left out); ``ValueError`` for events given as features."""
    if order == 0:
        return {}
    specs = list(events.values()) if isinstance(events, dict) else [events]
    for spec in specs:
        if _has_features(_parse_spec(spec, T, n_subjects, True)):
            raise ValueError(
                "causal=True: events given as features are segmented from the "
                "whole recording.  Pass the boundaries (the timepoints where a "
                "new block starts), for example events=[100, 250]."
            )
    return resolve_events_by_order(events, T, n_subjects, order, None, causal=True)


def _effective(reducer, fitted):
    """
    The ``"effective"`` entry of a step reduced by a ``Communities`` object
    with a given ``n`` whose fit ``fitted`` has another number of clusters
    (a refit that took a number near the first fit's: ``_same_width``):
    the reducer described with the number of the fit.  None otherwise (the
    entry of ``_step_info`` stays).
    """
    if not (isinstance(fitted, _FittedSummary) and isinstance(reducer, Communities)
            and reducer.n is not None and fitted.fit.get("n") is not None
            and int(fitted.fit["n"]) != int(reducer.n)):
        return None
    used = copy.copy(reducer)
    used.n = int(fitted.fit["n"])
    return _describe(used)


def _causal_pass(settings, subjects, events, *, steps=None, record=None,
                 keep_delta=True, single=False, match=None):
    """
    One causal pass of the two-path pipeline over ``subjects`` (a list of
    (T, K) arrays): ``(X, D, info)``, the lists of outputs by order
    (``D[k]`` None for a feeding step that is not computed) and the
    ``return_info`` entries.

    ``steps`` None: every reducer is fitted to the rows given (pass 1 of the
    module docstring when they are the training rows; reducers without a
    fit are simply applied), and ``record`` (a dict) receives, by order,
    what reduced each path (``{"analysis": ..., "feeding": ...}``: a
    ``_FittedSummary``, a ``_FittedModel``, or the reducer itself).  With
    ``steps`` (such a dict), nothing is fitted: pass 2.  ``events``: the
    specification, resolved for these rows.  ``keep_delta``: whether the
    feeding features of every order are kept (counted against
    ``max_memory_gb``); ``single``: the data were one array (the entries of
    ``info`` then describe arrays, not lists).  ``match`` (with
    ``record``): the fits of the fit before, to which each new fit is
    matched when it is made (a refit; see ``_Matched``).
    """
    s = settings
    order = s["order"]
    T = subjects[0].shape[0]
    resolved = _events_by_order(events, T, len(subjects), order)
    held = tuple({id(r): r for r in list(s["rfun"].values())
                  + list(s["delta_rfun"].values()) if isinstance(r, _Summary)}.values())
    X, D, info = [subjects], [subjects], []
    fed_by = subjects
    previous_communities = None
    for k in range(1, order + 1):
        kept = X[1:] + ([d for d in D[1:] if d is not None] if keep_delta
                        else ([] if k == 1 else [fed_by]))
        used = {} if match is None else _Matched(match[k], k)
        x, delta = _two_path_step(
            fed_by,
            delta_rfun=s["delta_rfun"][k],
            auto_analysis=s["auto"][k],
            max_memory_gb=s["max_memory_gb"],
            order=k,
            feed=k in s["fed"],
            feeding_function=s["delta_weights_function"],
            feeding_params=s["delta_weights_params"][k],
            weights_function=s["weights_function"],
            weights_params=s["weights_params"][k],
            cfun=s["cfun"],
            rfun=s["rfun"][k],
            ndims=s["ndims"],
            combine=s["combine"],
            include_timepoints="backward",
            include_zero=s["include_zero"],
            events=resolved[k],
            retained_elements=_output_elements(*kept),
            input_elements=0 if k == 1 else _output_elements(fed_by),
            reducers=held,
            previous_communities=previous_communities,
            causal=True,
            min_samples=s["min_samples"],
            frozen=None if steps is None else steps[k],
            record=used,
        )
        if record is not None:
            record[k] = dict(used)
        X.append(x)
        D.append(delta)
        feeding = used.get("feeding")
        feeding_fit = feeding.fit if isinstance(feeding, _FittedSummary) else None
        # explicit Communities only (see features)
        previous_communities = (
            feeding_fit["n"] if feeding_fit is not None
            and isinstance(s["delta_rfun"][k], Communities)
            and not getattr(s["delta_rfun"][k], "_from_auto", False) else None
        )
        entries = _step_info(k, fed_by, x, delta, s["rfun"][k], s["delta_rfun"][k],
                             *s["requested"][k], feeding_fit=feeding_fit)
        for entry, params in zip(entries, (s["weights_params"][k],
                                           s["delta_weights_params"][k])):
            out = x if entry["path"] == "analysis" else delta
            if single and isinstance(out, list):
                out = out[0]
            entry["weights_params"] = dict(params)  # a copy
            described = _effective(s["rfun" if entry["path"] == "analysis"
                                     else "delta_rfun"][k], used.get(entry["path"]))
            if described is not None:
                entry["effective"] = described
            entry["constant_columns"] = _constant_columns(out, True)
            entry["warmup_rows"] = _warmup_rows(out)
            if _is_fit(used.get(entry["path"])):
                entry["fit_timepoints"] = s["fit_timepoints"]
                entry["refit_rows"] = list(s.get("refits", [s["fit_timepoints"]]))
                entry["refit_kept_previous"] = list(s.get("refit_kept_previous", []))
        info.extend(entries)
        if delta is not None:
            fed_by = delta
    return X, D, info


def _single_fit(settings, subjects, events):
    """
    Pass 1: the fits of every order from the first ``fit_timepoints`` rows
    of ``subjects`` alone, as ``{order: {"analysis": ..., "feeding": ...}}``
    (see ``_causal_pass``), or None when no reducer of a computed step is
    fitted to data.  The estimator's warnings about these rows are not
    issued (pass 2 issues them for every row).
    """
    s = settings
    orders = range(1, s["order"] + 1)
    needed = any(_needs_fit(s["rfun"][k]) for k in orders) or any(
        _needs_fit(s["delta_rfun"][k]) for k in s["fed"])
    if not needed:
        return None
    n = s["fit_timepoints"]
    steps = {}
    with _fit_pass():
        _causal_pass(s, [x[:n] for x in subjects], events, record=steps,
                     keep_delta=False)
    return steps


# -- refits ------------------------------------------------------------------

# the attributes of a fitted scikit-learn model that hold one value per
# component (reordered with ``components_``), and the one with one column
# per component
_PER_COMPONENT = ("explained_variance_", "explained_variance_ratio_",
                  "singular_values_")
_COLUMN_PER_COMPONENT = "mixing_"
# a model with reordered components must give the reordered outputs of the
# model as fitted to this (relative to the largest of them, at least 1)
_MATCH_RTOL = 1e-9


def _assignment(score):
    """For each row of a square score matrix, the column of the one-to-one
    assignment with the largest total score."""
    rows, columns = linear_sum_assignment(-np.asarray(score, dtype=np.float64))
    out = np.empty(len(rows), dtype=int)
    out[rows] = columns
    return out


def _match_model(new, previous, order, path):
    """
    ``new`` (a ``_FittedModel``) with its components in the order and with
    the signs of those of ``previous``: each component is assigned to one
    of ``previous`` (the one-to-one assignment with the largest sum of
    |cosine| between component vectors), takes its place, and is negated
    when the cosine is negative; the columns frozen as constant
    (``constants``) follow, and the padding stays last.  ``new`` itself
    when nothing moves, and for the identity (a model with as many
    components as columns).  Every model of a causal fit has components
    (``timecorr._fit_reducer``).  ``_FitUnusable`` when the numbers of
    components or of padded columns differ; ``ValueError`` when the
    reordered model does not give the reordered outputs.
    """
    a, b = new.components, previous.components
    if a is None and b is None and new.pad == previous.pad:
        return new  # the identity (no more columns than components) stays
    if a is None or b is None or a.shape != b.shape or new.pad != previous.pad:
        shapes = [None if c is None else c.shape[0] for c in (a, b)]
        raise _FitUnusable(
            f"refit at order {order} ({path}): the model has {shapes[0]} "
            f"components and {new.pad} padded columns; the fit before "
            f"{shapes[1]} and {previous.pad}."
        )
    c = a.shape[0]
    norms = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(norms > 0, (a @ b.T) / norms, 0.0)
    cosine = np.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0)
    place = _assignment(np.abs(cosine))  # place[q]: where component q goes
    source = np.empty(c, dtype=int)
    source[place] = np.arange(c)  # source[p]: the component that comes to p
    signs = np.where(cosine[source, np.arange(c)] < 0, -1.0, 1.0)
    if np.array_equal(source, np.arange(c)) and np.all(signs > 0):
        return new
    matched = _FittedModel(new.model, new.n_features, new.pad)  # a copy
    if matched.model is new.model:
        # (a model that cannot be pickled, as one of a class defined in a
        # function, was taken as it is: reorder a copy, not the fit itself)
        matched.model = copy.deepcopy(new.model)
    estimator = matched.estimator
    fitted = np.asarray(estimator.components_)
    estimator.components_ = np.ascontiguousarray(signs[:, None] * fitted[source])
    for name in _PER_COMPONENT:
        value = getattr(estimator, name, None)
        if isinstance(value, np.ndarray) and value.shape[:1] == (c,):
            setattr(estimator, name, np.ascontiguousarray(value[source]))
    value = getattr(estimator, _COLUMN_PER_COMPONENT, None)
    if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == c:
        setattr(estimator, _COLUMN_PER_COMPONENT,
                np.ascontiguousarray(value[:, source] * signs))
    # as a pickle restores it, like every fitted model (see _FittedModel)
    matched = _FittedModel(matched.model, new.n_features, new.pad)
    if new.constants is not None:
        columns, means, tol = (np.array(v) for v in new.constants)
        real = columns < c
        columns[real] = place[columns[real]]
        means[real] = means[real] * signs[columns[real]]
        at = np.argsort(columns)
        matched.constants = (columns[at], means[at], tol[at])
    # the reordered model on rows of its own: the reordered outputs
    probe = np.random.RandomState(0).standard_normal((4, new.n_features))
    probe = probe * np.max(np.abs(fitted), initial=1.0)
    before, after = new.apply([probe])[0], matched.apply([probe])[0]
    expected = before.copy()
    expected[:, :c] = before[:, source] * signs
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    if not np.max(np.abs(after - expected), initial=0.0) <= _MATCH_RTOL * scale:
        raise ValueError(
            f"refit_every: the components of the model at order {order} ({path}) "
            "could not be put in the order of the fit before (the reordered "
            "model does not give the reordered outputs).  Use 'PCA', "
            "'IncrementalPCA', 'FactorAnalysis' or 'TruncatedSVD', or a summary."
        )
    return matched


def _match_labels(labels, previous):
    """
    The community labels of a refit with the communities of 2 or more
    members (one output column each, in the order of their labels) in the
    order of those of the fit before: each is assigned to one of the
    earlier ones (the one-to-one assignment with the largest number of
    shared nodes) and takes the label that puts its column at that one's
    place.  Single-node communities keep their labels.  ``_FitUnusable``
    when the numbers of such communities differ (``labels`` are those of
    ``_same_width``, which looked for a clustering with that number).
    """
    labels, previous = np.asarray(labels), np.asarray(previous)
    kept = np.flatnonzero(np.bincount(labels) >= 2)
    kept_before = np.flatnonzero(np.bincount(previous) >= 2)
    if len(kept) != len(kept_before) or len(labels) != len(previous):
        raise _FitUnusable(
            f"{len(kept)} communities of 2 or more nodes, and with up to "
            f"{_REFIT_CLUSTER_RANGE} clusters more or fewer never the "
            f"{len(kept_before)} of the fit before."
        )
    shared = np.array([[np.sum((labels == q) & (previous == p)) for p in kept_before]
                       for q in kept]).reshape(len(kept), len(kept))
    place = _assignment(shared)  # place[q]: the column community kept[q] takes
    out = labels.copy()
    for q, label in enumerate(kept):
        out[labels == label] = kept[place[q]]
    return out


# a refit of a communities step tries the numbers of clusters within this
# distance of the number its settings give (the first fit's)
_REFIT_CLUSTER_RANGE = 5


def _n_columns(labels):
    """The communities of 2 or more nodes: one output column each."""
    return int(np.sum(np.bincount(np.asarray(labels)) >= 2))


def _same_width(fit, previous):
    """
    Community labels for the record ``fit`` of a refit of a communities
    step that give as many communities of 2 or more nodes (output columns)
    as the labels ``previous`` of the fit before, and so as the first fit.

    The number of clusters of the first fit does not fix that number: a
    weakly attached node is a cluster of its own in one fit and a member of
    a community in the next.  So the pair weights of the refit (those of
    its training rows) are clustered with ``n0``, the number of clusters
    the refit was made with (the first fit's; these are the labels of
    ``fit``), then ``n0 + 1``, ``n0 - 1``, ``n0 + 2``, ``n0 - 2``, ... up to
    ``n0 +- _REFIT_CLUSTER_RANGE``, between 2 and the number of nodes, and
    the first clustering with the columns of the fit before is taken.  The
    labels of ``fit`` when none has them (``_match_labels`` then raises
    ``_FitUnusable``: the refit is left out).
    """
    labels = np.asarray(fit["labels"])
    wanted = _n_columns(previous)
    if _n_columns(labels) == wanted or len(labels) != len(previous):
        return labels
    wbar = np.mean(fit["weights"], axis=0)
    n0, K = int(fit["n"]), len(labels)
    for step in range(1, _REFIT_CLUSTER_RANGE + 1):
        for n in (n0 + step, n0 - step):
            if 2 <= n <= K:
                other = _find_communities(wbar, n)
                if _n_columns(other) == wanted:
                    return other
    return labels


# what an automatic feeding step records about its choice (made by the first fit)
_AUTO_KEYS = ("n_columns", "fallback", "n_exceed", "null_threshold", "n_null")


def _match(new, previous, order, path):
    """
    The fit ``new`` of a refit, matched to ``previous``, the fit before of
    the same step (see ``_match_model`` and ``_match_labels``; the fits of
    ``WeightedStrength`` have nothing to match); ``new`` itself when
    nothing changes.  ``_FitUnusable`` when ``new`` has not the structure
    of ``previous``.
    """
    if isinstance(new, _FittedModel) and isinstance(previous, _FittedModel):
        return _match_model(new, previous, order, path)
    if not (isinstance(new, _FittedSummary) and isinstance(previous, _FittedSummary)):
        return new
    if "fallback" in previous.fit:
        # the choice of the automatic feeding reducer is the first fit's
        new.fit.update({key: previous.fit[key] for key in _AUTO_KEYS})
    if not (isinstance(new.summary, Communities)
            and isinstance(previous.summary, Communities)):
        return new
    found = _same_width(new.fit, previous.fit["labels"])
    labels = _match_labels(found, previous.fit["labels"])
    if np.array_equal(labels, new.fit["labels"]):
        return new
    return _FittedSummary(new.summary, {**new.fit, "labels": labels,
                                        "n": int(len(np.unique(labels)))})


class _Matched(dict):
    """
    The record of one order of a refit (``record`` of ``_two_path_step``):
    a fit stored under its path is replaced by that fit matched to the fit
    before (``previous``: that order's record of the fit before), and the
    step is then reduced with what the record holds
    (``higher_order._apply_reducer`` and ``_run_streamed`` read it back).
    """

    def __init__(self, previous, order):
        super().__init__()
        self.previous = previous
        self.order = order

    def __setitem__(self, path, fit):
        try:
            fit = _match(fit, self.previous[path], self.order, path)
        except _FitUnusable as err:
            err.step = f"order {self.order}, {path} path"
            raise
        super().__setitem__(path, fit)


def _refit_settings(settings, steps):
    """
    The settings of the fits after the first, whose fits are ``steps``:
    the structure is the first fit's.  A summary that chose its number of
    communities with the shift null gets that number (no null runs again);
    an automatic feeding step is what it chose, those communities or the
    weighted strength it fell back to.  The summaries are copies, so the
    objects of the caller keep the records of the first fit.
    """
    s = FeaturesModel._detached(settings)
    for name, path in (("rfun", "analysis"), ("delta_rfun", "feeding")):
        reducers = dict(s[name])
        for k, step in steps.items():
            fitted = step.get(path)
            if not isinstance(fitted, _FittedSummary):
                continue
            reducer = reducers[k]
            if isinstance(reducer, str) and reducer in _SUMMARY_NAMES:
                reducer = _from_name(reducer)
            if fitted.fit.get("fallback"):
                reducers[k] = _auto_fallback()
            elif reducer._needs_communities and reducer.n is None:
                reducer = copy.copy(reducer)
                reducer.n = int(fitted.fit["n"])
                reducer._from_auto = False
                reducers[k] = reducer
        s[name] = reducers
    return s


def _summaries(settings):
    return {id(r): r for name in ("rfun", "delta_rfun")
            for r in settings[name].values() if isinstance(r, _Summary)}.values()


def _stitch(into, part, start):
    """Rows ``start:`` of the outputs ``part`` (arrays, lists of arrays or
    None, by order) copied into ``into``."""
    for a, b in zip(into[1:], part[1:]):
        if a is None:
            continue
        for u, v in zip(a if isinstance(a, list) else [a],
                        b if isinstance(b, list) else [b]):
            u[start:] = v[start:]


def _refit(model, settings, subjects, single, events, return_delta, refit_every,
           X, D, info):
    """
    The fits after the first of ``refit_every`` (module docstring): the
    outputs ``X`` and ``D`` of pass 2 with the first fit get, from each row
    ``s_j`` on, the rows of pass 2 with fit ``j``; ``model`` gets the last
    fit.  Returns the ``return_info`` entries (those of the last fit's
    pass, with the constant columns and warm-up rows of the outputs as
    returned, the rows of the fits and those whose fit was left out).
    """
    steps = model._steps
    n, T = settings["fit_timepoints"], subjects[0].shape[0]
    rows = [n] + list(range(n + refit_every, T, refit_every))
    fit_settings = _refit_settings(settings, steps)
    kept, reasons = [], {}
    for s_j in rows[1:]:
        for summary in _summaries(fit_settings):
            summary.fits_ = []
        new = {}
        try:
            with _fit_pass():
                _causal_pass(dict(fit_settings, fit_timepoints=s_j),
                             [x[:s_j] for x in subjects], events, record=new,
                             keep_delta=False, match=steps)
        except _FitUnusable as err:
            # not the structure of the first fit: the fit before stays, and
            # with it the rows already there
            kept.append(s_j)
            reason = (getattr(err, "step", None), str(err))
            reasons.setdefault(reason, []).append(s_j)
            continue
        steps = new
        with _fit_pass():
            X_j, D_j, info = _causal_pass(model._settings, subjects, events,
                                          steps=steps, keep_delta=return_delta,
                                          single=single)
        _stitch(X, X_j, s_j)
        _stitch(D, D_j, s_j)
        del X_j, D_j
    if kept:
        _warn_user(
            f"refit_every: {len(kept)} of the {len(rows) - 1} fits after the "
            "first could not have the structure of the first fit and were left "
            "out; the fit before stays for their rows ('refit_kept_previous' "
            "of return_info).  "
            + "  ".join(
                f"Fit{'' if len(at) == 1 else 's'} at row"
                f"{'' if len(at) == 1 else 's'} {', '.join(map(str, at))}"
                f"{'' if step is None else f' ({step})'}: {message}"
                for (step, message), at in reasons.items())
        )
    model._steps = steps
    model._settings["refits"] = rows
    model._settings["refit_kept_previous"] = kept
    for entry in info:
        out = (X if entry["path"] == "analysis" else D)[entry["order"]]
        if single and isinstance(out, list):
            out = out[0]
        entry["constant_columns"] = _constant_columns(out, True)
        entry["warmup_rows"] = _warmup_rows(out)
        if "refit_rows" in entry:
            entry["refit_rows"] = list(rows)
            entry["refit_kept_previous"] = list(kept)
    return info


def _causal_features(subjects, single, events, settings, *, fit_timepoints,
                     refit_every, return_delta, return_info, return_model):
    """``features(..., causal=True)`` on formatted subjects, after its checks."""
    settings = dict(settings, fit_timepoints=fit_timepoints)
    steps = _single_fit(settings, subjects, events)
    model = FeaturesModel(settings, steps, n_features=subjects[0].shape[1],
                          n_subjects=len(subjects))
    X, D, info = model._pass(subjects, single, events, return_delta)
    if refit_every is not None and steps is not None:
        info = _refit(model, settings, subjects, single, events, return_delta,
                      refit_every, X, D, info)
    result = model._returned(X, D, single, return_delta)
    model.info = info
    out = (result,) + ((info,) if return_info else ()) + (
        (model,) if return_model else ())
    return out[0] if len(out) == 1 else out


class FeaturesModel:
    """
    The fitted two-path pipeline of a causal ``features`` call.

    Returned by ``features(..., causal=True, return_model=True)``; not made
    directly.  It holds what the call computed with: the order, the kernel
    function and parameters of both paths at every order, ``cfun``,
    ``combine``, ``include_zero``, ``min_samples``, ``ndims``,
    ``max_memory_gb``, and the reducers of every order and path as they
    were fitted on the first ``fit_timepoints`` rows (the components of a
    model, the pair weights and communities of a summary, the choice an
    automatic feeding step made).  ``transform`` applies that pipeline to
    other data without fitting anything.  Events are not kept: ``transform``
    takes its own.

    Parameters
    ----------
    settings : dict
        The resolved arguments of the ``features`` call.
    steps : dict or None
        The fits by order and path (None: no reducer is fitted to data).
    n_features : int
        The number of columns of the data.
    n_subjects : int
        The number of subjects of the data.

    Attributes
    ----------
    order : int
        The highest order.
    fit_timepoints : int or None
        The number of first rows the reducers were fitted on (None: the
        call gave none, and no reducer needed a fit).
    refits : list of int
        The rows of the fits: ``[fit_timepoints]``, and with
        ``refit_every`` every row at which the reducers were fitted again
        (each fit read the rows before its row only); empty when no
        reducer was fitted.  The model holds the last fit.
    n_features, n_subjects : int
        The numbers of columns and of subjects of the data of the fit.
    info : list of dict
        The ``return_info`` entries of the ``features`` call.
    fits : dict
        ``{(order, path): fit}`` for every computed step, ``path`` being
        ``"analysis"`` or ``"feeding"``: for a summary its record (a dict
        as in ``fits_``, with the pair weights ``"weights"``, one (K, K)
        array per subject, ``"mean_r"``, ``"labels"``, ...), for a model
        the fitted hypertools model, and for a reducer without a fit the
        reducer itself (None, a graph measure, a function).
    weights_function, delta_weights_function : function
        The kernel functions of the analysis and feeding paths.
    weights_params, delta_weights_params : dict
        The kernel parameters of each path, by order.
    cfun, combine : function
        The correlation and combine functions of the analysis path.
    include_zero : bool
    min_samples : float or None
    ndims : int or None
    max_memory_gb : float or None
        As given to ``features``; ``transform`` uses them (assign another
        ``max_memory_gb`` to change the limit of ``transform``).

    Notes
    -----
    With ``refit_every`` the model holds the last fit (its columns matched
    to those of the fits before), and ``transform`` applies it to every
    row.  The ``features`` call returned, before the last row of
    ``refits``, rows made with earlier fits: ``transform`` of the same
    data equals that output from ``refits[-1]`` on (and on every row
    when that fit is the only one) and, before it, differs from it as the
    fits differ.

    A model can be pickled when the arguments of the call can: a ``cfun``,
    reducer or kernel function that is a lambda or a local function cannot.
    Unpickling needs the same versions of timecorr, hypertools and
    scikit-learn.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> x = np.random.RandomState(0).randn(150, 6)
    >>> X, model = tc.features(x, order=2, causal=True, cfun=tc.autofc,
    ...                        weights_params={"var": 25},
    ...                        delta_weights_params={"var": 25},
    ...                        delta_rfun="weighted_strength",
    ...                        fit_timepoints=100, return_model=True)
    >>> model
    FeaturesModel(order=2, fit_timepoints=100, n_features=6, n_subjects=1)
    >>> sorted(model.fits)
    [(1, 'analysis'), (1, 'feeding'), (2, 'analysis')]
    >>> later = model.transform(x[:120])
    >>> bool(np.allclose(later[2], X[2][:120], atol=1e-9, equal_nan=True))
    True
    """

    def __init__(self, settings, steps, *, n_features, n_subjects):
        self._settings = self._detached(settings)
        self._steps = steps
        self.n_features = int(n_features)
        self.n_subjects = int(n_subjects)
        self.info = []

    @staticmethod
    def _detached(settings):
        """The settings with copies of the summary objects among the
        reducers, without their fits: they only say how a step is computed
        (the fits are in ``steps``), and the caller's objects are fitted
        again by later calls."""
        copies = {}

        def detach(reducer):
            if not isinstance(reducer, _Summary):
                return reducer
            if id(reducer) not in copies:
                clone = copy.copy(reducer)
                clone.fits_ = []
                copies[id(reducer)] = clone
            return copies[id(reducer)]

        settings = dict(settings)
        for name in ("rfun", "delta_rfun"):
            settings[name] = {k: detach(r) for k, r in settings[name].items()}
        for name in ("weights_params", "delta_weights_params"):
            settings[name] = {k: copy.deepcopy(p) for k, p in settings[name].items()}
        return settings

    # -- what the call computed with -----------------------------------------

    order = property(lambda self: self._settings["order"])
    fit_timepoints = property(lambda self: self._settings["fit_timepoints"])

    @property
    def refits(self):
        if self._steps is None or not any(
                _is_fit(f) for step in self._steps.values() for f in step.values()):
            return []
        return list(self._settings.get("refits", [self.fit_timepoints]))
    weights_function = property(lambda self: self._settings["weights_function"])
    weights_params = property(lambda self: self._settings["weights_params"])
    delta_weights_function = property(
        lambda self: self._settings["delta_weights_function"] or gaussian_weights)
    delta_weights_params = property(
        lambda self: self._settings["delta_weights_params"])
    cfun = property(lambda self: self._settings["cfun"])
    combine = property(lambda self: self._settings["combine"])
    include_zero = property(lambda self: self._settings["include_zero"])
    min_samples = property(lambda self: self._settings["min_samples"])
    ndims = property(lambda self: self._settings["ndims"])

    @property
    def max_memory_gb(self):
        return self._settings["max_memory_gb"]

    @max_memory_gb.setter
    def max_memory_gb(self, value):
        self._settings["max_memory_gb"] = value

    @property
    def fits(self):
        s = self._settings
        out = {}
        for k in range(1, s["order"] + 1):
            for path, reducers in (("analysis", s["rfun"]), ("feeding", s["delta_rfun"])):
                if path == "feeding" and k not in s["fed"]:
                    continue
                step = reducers[k] if self._steps is None else self._steps[k][path]
                if isinstance(step, _FittedSummary):
                    step = step.fit
                elif isinstance(step, _FittedModel):
                    step = step.model
                out[(k, path)] = step
        return out

    def __repr__(self):
        return (
            f"FeaturesModel(order={self.order}, fit_timepoints={self.fit_timepoints}, "
            f"n_features={self.n_features}, n_subjects={self.n_subjects})"
        )

    # -- applying it -----------------------------------------------------------

    def transform(self, data, return_delta=False, return_info=False, *, events=None):
        """
        Features of ``data`` from the fitted pipeline, without fitting.

        Every row is computed causally, as by the ``features`` call that
        made the model: row ``t`` of every output uses rows ``0..t`` of
        ``data`` and the frozen fits only, so it is unchanged (to 1e-9 for
        features built from correlations; see ``features``) when later rows
        are replaced or removed, and
        ``transform`` of the data of the fit returns what that call
        returned.

        Parameters
        ----------
        data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
            One timepoints-by-features matrix per subject, with the number
            of features of the fit (``n_features``) and any number of
            timepoints (one or more).  The number of subjects must be that
            of the fit (``n_subjects``) when a summary was fitted (each
            subject is summarized with the pair weights fitted to the
            subject at its position); a pipeline without summaries takes
            any number.
        return_delta : bool
            If True, also return the feeding features ``[D_0, ...,
            D_order]``; the ``features`` call must have computed the last
            of them (``return_delta=True`` there).  Default: False.
        return_info : bool
            If True, also return the list of dicts ``features`` returns
            with ``return_info=True`` and ``causal=True``, for this data.
            Default: False.
        events : None, dict, array-like, or list
            Event boundaries that restrict the kernels (keyword-only), as
            in ``features`` with ``causal=True`` (boundaries only; a dict
            by order; boundaries at or beyond the end of ``data`` are
            ignored).  The events of the ``features`` call are not kept:
            give them again, extended to this data, if the fit used any.
            Default: None (no restriction).

        Returns
        -------
        X : list
            ``[X_0, ..., X_order]``, as ``features`` returns it (arrays for
            array input, lists of arrays for list input unless ``combine``
            returns one array).  The first rows of every order are NaN
            (see ``min_samples`` of ``features``).
        D : list
            Only with ``return_delta=True``: ``[D_0, ..., D_order]``; the
            call then returns ``(X, D)``.
        info : list of dict
            Only with ``return_info=True``; the call then returns ``(X,
            info)`` or ``((X, D), info)``.

        Raises
        ------
        ValueError
            If the data have another number of features than the fit, or
            another number of subjects when a summary was fitted; if
            ``return_delta=True`` and the fit did not compute the last
            feeding step; for events given as features or malformed.
        MemoryError
            As ``features``, with ``max_memory_gb`` of the model.

        Examples
        --------
        >>> import numpy as np
        >>> import timecorr as tc
        >>> rng = np.random.RandomState(0)
        >>> x = rng.randn(150, 6)
        >>> X, model = tc.features(x, order=2, causal=True, cfun=tc.autofc,
        ...                        weights_params={"var": 25},
        ...                        delta_weights_params={"var": 25},
        ...                        delta_rfun="weighted_strength",
        ...                        fit_timepoints=100, return_model=True)
        >>> same = model.transform(x)
        >>> bool(np.array_equal(same[2], X[2], equal_nan=True))
        True

        Other data with 6 features, of any length:

        >>> new = model.transform(rng.randn(40, 6))
        >>> new[1].shape, new[2].shape
        ((40, 6), (40, 12))
        """
        single = not isinstance(data, (list, tuple))
        subjects = format_data(data, fill_nan=False)
        K = subjects[0].shape[1]
        if K != self.n_features:
            raise ValueError(
                f"the model was fitted to data with {self.n_features} features; "
                f"got {K}."
            )
        if return_delta and self.order and self.order not in self._settings["fed"]:
            raise ValueError(
                f"return_delta=True: the feeding features of order {self.order} "
                "were not computed by the features call that made this model; "
                "call features with return_delta=True."
            )
        X, D, info = self._pass(subjects, single, events, return_delta)
        result = self._returned(X, D, single, return_delta)
        return (result, info) if return_info else result

    def _pass(self, subjects, single, events, return_delta):
        """Pass 2 on formatted subjects: ``(X, D, info)`` of ``_causal_pass``."""
        return _causal_pass(self._settings, subjects, events, steps=self._steps,
                            keep_delta=return_delta, single=single)

    @staticmethod
    def _returned(X, D, single, return_delta):
        """The outputs of a pass as ``features`` returns them."""
        if single:
            X = [x[0] if isinstance(x, list) else x for x in X]
            D = [d if d is None else d[0] for d in D]
        return (X, D) if return_delta else X
