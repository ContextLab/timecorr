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
"""

import copy

from .events import _has_features, _parse_spec, resolve_events_by_order
from .helpers import _FittedModel, format_data, gaussian_weights
from .higher_order import (
    _constant_columns,
    _fit_pass,
    _output_elements,
    _step_info,
    _two_path_step,
    _warmup_rows,
)
from .summaries import Communities, _FittedSummary, _Summary
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


def _causal_pass(settings, subjects, events, *, steps=None, record=None,
                 keep_delta=True, single=False):
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
    ``info`` then describe arrays, not lists).
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
        used = {}
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
            record[k] = used
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
            entry["constant_columns"] = _constant_columns(out, True)
            entry["warmup_rows"] = _warmup_rows(out)
            if _is_fit(used.get(entry["path"])):
                entry["fit_timepoints"] = s["fit_timepoints"]
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


def _causal_features(subjects, single, events, settings, *, fit_timepoints,
                     return_delta, return_info, return_model):
    """``features(..., causal=True)`` on formatted subjects, after its checks."""
    settings = dict(settings, fit_timepoints=fit_timepoints)
    steps = _single_fit(settings, subjects, events)
    model = FeaturesModel(settings, steps, n_features=subjects[0].shape[1],
                          n_subjects=len(subjects))
    result, info = model._transform(subjects, single, events, return_delta)
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
        ``data`` and the frozen fits only, so it is unchanged (to 1e-9; see
        ``features``) when later rows are replaced or removed, and
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
        result, info = self._transform(subjects, single, events, return_delta)
        return (result, info) if return_info else result

    def _transform(self, subjects, single, events, return_delta):
        """Pass 2 on formatted subjects: ``(result, info)``."""
        X, D, info = _causal_pass(self._settings, subjects, events,
                                  steps=self._steps, keep_delta=return_delta,
                                  single=single)
        if single:
            X = [x[0] if isinstance(x, list) else x for x in X]
            D = [d if d is None else d[0] for d in D]
        return ((X, D) if return_delta else X), info
