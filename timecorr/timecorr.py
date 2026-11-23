# coding: utf-8

import functools
import numbers
import warnings

import numpy as np

from ._kernels import CHUNK, as_kernel, blocks, check_exclude
from .events import (
    _has_features,
    _parse_spec,
    distinct_event_sets,
    _mask_kernel,
    resolve_events,
)
from .helpers import (
    _GRAPH_MEASURES,
    _MIN_SAMPLES_RTOL,
    _SUMMARY_NAMES,
    _COMPONENT_MODELS,
    _FittedModel,
    _FitUnusable,
    _is_model,
    _model_reduce,
    _abs_weights,
    _as_kernel_array,
    _causal_clean,
    _check_reachable,
    _check_causal_kernel,
    _check_min_samples,
    _check_rfun,
    _lag_kernel,
    _warn_user,
    autofc,
    eye_weights,
    format_data,
    gaussian_weights,
    isfc,
    null_combine,
    one_sided,
    reduce,
    wisfc,
)

# correlation functions that accept one kernel per subject
_KERNEL_LIST_CFUNS = (wisfc, isfc, autofc)


def _causal_smooth(x, weights, min_samples):
    """
    ``_kernel_smooth`` for ``causal=True``: the same kernel-weighted means of
    a kernel without weight on later timepoints, each chunk of rows over the
    columns where it has weight.  The samples of a chunk are centered on the
    weighted mean of its first row (which weights no later sample) and the
    mean is added back, so the round-off of the sums is relative to the
    variation of the data in the chunk, not to its level; the values
    themselves are still rounded at their own magnitude.  Infinite samples and samples above 1e150
    in magnitude are missing, like NaN.  An entry whose feature has no
    weighted sample or, with ``min_samples`` (not None), an effective sample
    size ``(sum |w|)**2 / sum w**2`` (over the feature's observed samples in
    the kernel row) below it is NaN; one warning reports how many timepoints
    have one.  ``ValueError`` when the kernel cannot reach ``min_samples``
    with any number of earlier timepoints (``_check_reachable``).
    """
    if min_samples is not None:
        _check_reachable(weights, min_samples)
    x = _causal_clean(x)
    valid = ~np.isnan(x)
    observed = np.where(valid, x, 0.0)
    valid = valid.astype(np.float64)
    out = np.full(observed.shape, np.nan)
    for t0, t1, c0, c1, _, w in blocks(as_kernel(weights), CHUNK, _abs_weights):
        if c1 <= c0:
            continue
        # each row scaled by its largest weight, so squares do not underflow
        peak = w.max(axis=1, keepdims=True, initial=0.0)
        with np.errstate(invalid="ignore"):
            w = np.divide(w, peak, out=np.zeros_like(w), where=peak != 0)
            den = w @ valid[c0:c1]
            defined = den > 0
            # the chunk's samples are centered on its first row's weighted
            # mean (of samples that row weights: none after it), so the
            # round-off of the sums is relative to the variation of the
            # data, not to its level
            first = np.zeros(den.shape[1])
            np.divide(w[0] @ observed[c0:c1], den[0], out=first, where=defined[0])
            centered = observed[c0:c1] - first
            centered *= valid[c0:c1]
            num = w @ centered
            if min_samples is not None:
                ess = np.divide(den**2, (w * w) @ valid[c0:c1],
                                out=np.zeros_like(den), where=defined)
                defined &= ess >= min_samples * (1 - _MIN_SAMPLES_RTOL)
        np.divide(num, den, out=out[t0:t1], where=defined)
        out[t0:t1] += first
    undefined = np.isnan(out).any(axis=1)
    if undefined.any():
        warnings.warn(
            f"Smoothing is undefined at {int(undefined.sum())} timepoint(s) (for "
            "some feature, no weighted, observed sample"
            + ("" if min_samples is None else
               f" or an effective sample size below min_samples={min_samples:g}")
            + "); those values are NaN (causal=True)."
        )
    return out


def _kernel_smooth(x, weights):
    """
    Kernel-weighted mean of the rows of x: out[t] = sum_j |w[t, j]| x[j] / sum_j |w[t, j]|.

    Row t of ``weights`` is the kernel used at timepoint t.  NaN samples get no
    weight.  Entries with no weighted, observed sample are undefined; they are
    set to 0 and one warning reports how many timepoints were affected.
    ``weights`` is an array or a ``LagKernel``, read a chunk of rows at a
    time over the columns where the chunk can have weight.  As in the
    estimator, an infinite sample enters every row's sums (``0 * inf`` is
    NaN), so with one every chunk spans all columns.
    """
    valid = ~np.isnan(x)
    observed = np.where(valid, x, 0.0)
    valid = valid.astype(np.float64)
    num = np.zeros(observed.shape)
    den = np.zeros(observed.shape)
    full = not np.all(np.isfinite(observed))
    for t0, t1, c0, c1, _, w in blocks(as_kernel(weights), CHUNK, np.abs, full=full):
        if c1 > c0:
            num[t0:t1] = w @ observed[c0:c1]
            den[t0:t1] = w @ valid[c0:c1]
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    undefined = np.any(den <= 0, axis=1)
    if undefined.any():
        warnings.warn(
            f"Smoothing is undefined at {int(undefined.sum())} timepoint(s) (no "
            "weighted, observed samples); those values are set to 0."
        )
    return out


def _restricted(resolved):
    """Resolved events with single-block boundary sets (no restriction) replaced by None."""

    def one(b):
        return None if b is None or len(b) <= 1 else b

    if isinstance(resolved, list):
        return [one(b) for b in resolved]
    return one(resolved)


def _mask_for_subjects(weights, boundaries, subjects, causal=False):
    """``mask_kernel`` for one event set; an error names the subjects that use it."""
    try:
        return _mask_kernel(weights, boundaries, not causal)
    except ValueError as err:
        raise ValueError(f"per-subject events of subjects {subjects}: {err}") from err


def _temporal_filter(T, k):
    """
    Band of half-width |k| around the diagonal (k >= 0), or its complement;
    k is rounded to the nearest integer (as for the built-in kernels).
    """
    k = int(check_exclude(k))
    filt = np.eye(T)
    for i in np.arange(1, np.min([np.abs(k) + 1, T])):
        filt = filt + np.eye(T, k=np.abs(i)) + np.eye(T, k=-np.abs(i))
    if k < 0:
        return 1 - filt
    else:
        return filt


def _shaped_kernel(
    T,
    weights_function,
    weights_params,
    include_timepoints="all",
    include_zero=True,
    exclude_timepoints=None,
    causal=False,
):
    """
    The T x T kernel of ``timecorr`` before the events restriction:
    ``weights_function(T, weights_params)`` shaped by ``include_timepoints``,
    ``include_zero`` and ``exclude_timepoints``.  For a built-in kernel
    function it is a ``LagKernel``, whose rows are computed when they are
    read (no T x T array); for any other function, the array it returns
    (with ``causal=True`` that array must have no weight above its
    diagonal: ``ValueError``).
    """
    exclude_timepoints = check_exclude(exclude_timepoints)
    include_timepoints = include_timepoints.lower()
    sides = {"all": "all", "backward": "backward", "pre": "backward",
             "forward": "forward", "post": "forward"}
    if include_timepoints not in sides:
        raise ValueError(
            f"Invalid option for include_timepoints: '{include_timepoints}'.  "
            "Must be one of: 'all', 'backward', 'pre', 'forward', or 'post'."
        )
    lazy = _lag_kernel(T, weights_function, weights_params, sides[include_timepoints],
                       include_zero, exclude_timepoints)
    if lazy is not None:
        return lazy

    weights = weights_function(T, weights_params)
    if causal:
        _check_causal_kernel(weights, "the kernel that weights_function returned")
    if include_timepoints == "all":
        if not include_zero:
            weights = weights.copy()
            np.fill_diagonal(weights, 0)
    elif include_timepoints in ("backward", "pre"):
        weights = one_sided(weights, "backward", include_zero)
    else:
        weights = one_sided(weights, "forward", include_zero)

    if not (exclude_timepoints is None):
        weights = np.multiply(_temporal_filter(T, exclude_timepoints), weights)
    return weights


def _event_kernels(weights, resolved, n_subjects, causal=False):
    """
    ``weights`` restricted to the RESOLVED events as ``timecorr`` restricts
    it: one masked kernel when every subject has the same events, otherwise
    a list of ``n_subjects`` kernels (subjects that share events share one
    array).  ``causal``: events that leave no row with 3 weights are not an
    error (a short prefix of a series; its rows are undefined).
    """
    groups = distinct_event_sets(_restricted(resolved), n_subjects)
    if len(groups) == 1:
        return _mask_kernel(weights, groups[0][0], not causal)
    kernels = [None] * n_subjects
    for boundaries, subjects in groups:
        masked = _mask_for_subjects(weights, boundaries, subjects, causal)
        for s in subjects:
            kernels[s] = masked
    return kernels


def _per_subject_cfun(cfun, data, kernels, groups):
    """
    Correlations of a custom ``cfun`` with per-subject kernels.

    ``kernels`` holds one masked kernel per subject (from ``_event_kernels``;
    subjects of one group share one array) and ``groups`` the
    ``(boundaries, subjects)`` pairs of ``distinct_event_sets``.  ``cfun`` is
    called once per group with the full data and that group's kernel, and
    each subject's output is taken from the call made with its kernel.  (The
    built-in correlation functions take the list of kernels in one call and
    do not come here.)
    """
    S = len(data)
    out = [None] * S
    for _, subjects in groups:
        result = cfun(data, kernels[subjects[0]])
        if not isinstance(result, list) or len(result) != S:
            raise ValueError(
                "with per-subject events, cfun must return a list with one array per "
                f"subject ({S}); got "
                + (
                    f"a list of {len(result)}"
                    if isinstance(result, list)
                    else type(result).__name__
                )
            )
        for s in subjects:
            out[s] = result[s]
        del result
    return out


def _fixed_weights_wisfc(cfun):
    """
    True for ``functools.partial(wisfc, subject_weights=...)`` with weights
    given (and nothing else bound): the built-in ``wisfc`` with fixed
    subject weights.
    """
    return (
        isinstance(cfun, functools.partial)
        and cfun.func is wisfc
        and not cfun.args
        and set(cfun.keywords) == {"subject_weights"}
        and cfun.keywords["subject_weights"] is not None
    )


def _needs_fit(rfun):
    """True for a reducer that is fitted to data: a model or a summary."""
    from .summaries import _Summary  # timecorr.summaries imports helpers

    if isinstance(rfun, (dict, _Summary)):
        return True
    return isinstance(rfun, str) and rfun not in _GRAPH_MEASURES


def _check_fit_timepoints(fit_timepoints, causal, T=None):
    """
    ``fit_timepoints`` as None or an int: ``ValueError`` unless it is None
    or an integer in ``1..T`` (``T`` None: any integer >= 1), and when it is
    given without ``causal=True``.
    """
    if fit_timepoints is None:
        return None
    if (
        isinstance(fit_timepoints, (bool, np.bool_))
        or not isinstance(fit_timepoints, numbers.Integral)
        or fit_timepoints < 1
    ):
        raise ValueError(
            "fit_timepoints must be None or a positive integer (the number of "
            f"first rows the reducers are fitted on); got {fit_timepoints!r}."
        )
    if not causal:
        raise ValueError(
            f"fit_timepoints={fit_timepoints!r} applies with causal=True only; "
            "pass causal=True, or leave fit_timepoints out."
        )
    if T is not None and fit_timepoints > T:
        raise ValueError(
            f"fit_timepoints={fit_timepoints} is more than the {T} "
            f"timepoint{'s' if T != 1 else ''} of the data; pass at most {T} "
            "(every row is then in the rows the reducers are fitted on)."
        )
    return int(fit_timepoints)


def _defined_rows(arrays):
    """For each array, the mask of its rows that are not entirely NaN."""
    return [~np.isnan(np.asarray(a)).all(axis=1) for a in arrays]


def _on_defined(arrays, f):
    """
    ``f`` (a list of arrays in, a list of arrays out, row by row) applied to
    the rows of ``arrays`` that are not entirely NaN; the other rows stay
    rows of NaN.  Every array is passed, at its position, also one without
    such a row (a fit that holds one unit per subject is applied to the
    subject at each position).
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    defined = _defined_rows(arrays)
    # every row is passed, the undefined ones as rows of zeros: a row is then
    # reduced at the same position of an array of the same shape whatever
    # the later rows hold, so replacing later rows (by NaN too) changes no
    # bit of it (the round-off of a matrix product can depend on its shape)
    reduced = f([np.where(d[:, None], a, 0.0) for a, d in zip(arrays, defined)])
    out = []
    for r, d in zip(reduced, defined):
        r = np.array(r, dtype=np.float64)
        r[~d] = np.nan
        out.append(r)
    return out


# a fitted model must reproduce its own training rows to this (relative to
# the largest of them, at least 1) to be used on other rows
_REFIT_ATOL = 1e-9


def _fit_reducer(corrs, rfun, ndims=None, *, order=None, path=None, append_fit=False,
                 name="rfun"):
    """
    ``rfun`` fitted to the rows of ``corrs`` (an array or a list of arrays
    of vectorized correlations) that are not entirely NaN, for
    ``causal=True``: what ``_apply_frozen`` applies to these or other rows.

    A summary (object or name) gives a ``_FittedSummary`` (the object's
    ``fits_`` gains the fit, with ``order`` and ``path``; ``append_fit`` as
    in ``higher_order._apply_reducer``), a model a ``helpers._FittedModel``
    (the identity when hypertools returns the rows as they are: no more
    columns than components); None, a graph measure and a function of the
    user's are returned as they are (nothing to fit).

    A model is taken only when its fitted estimator has ``components_`` (a
    linear map), reproduces its training rows (to ``_REFIT_ATOL``) and
    reduces a few of them to the same values one at a time and in another
    batch: together these show that it reduces a row by itself.

    ``ValueError`` otherwise, and when the rows hold no defined row to fit
    (a model: fewer than 2); ``helpers._FitUnusable``, a ``ValueError``,
    for what can differ from one fit to the next (no defined row, a model
    that does not reproduce its rows).
    """
    from .summaries import _auto_emitter, _FittedSummary, _from_name, _Summary

    summary = rfun
    if isinstance(rfun, str) and rfun in _SUMMARY_NAMES:
        summary = _from_name(rfun)
    is_summary = isinstance(summary, _Summary)
    if not is_summary and not _is_model(rfun):
        return rfun
    arrays = corrs if isinstance(corrs, list) else [corrs]
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    rows = [a[d] for a, d in zip(arrays, _defined_rows(arrays))]
    at = "" if order is None else f" at order {order}"
    fewest = min(len(r) for r in rows) if is_summary else sum(len(r) for r in rows)
    if fewest < (1 if is_summary else 2):
        raise _FitUnusable(
            f"causal=True: the rows {name} is fitted on (fit_timepoints){at} "
            f"hold {'no row' if is_summary else 'fewer than 2 rows'} with "
            "defined correlations (the first rows of every order are "
            "undefined: see min_samples).  Raise fit_timepoints."
        )
    if is_summary:
        summary.fit(rows, order=order, path=path)
        fit = summary.fits_[-1]
        K = fit["weights"][0].shape[0]
        _auto_emitter(summary, fit, K, order)
        if not append_fit:
            del summary.fits_[:-1]
        return _FittedSummary(summary, fit)
    rfun = _exact_pca(rfun)
    # one model for all subjects: those without a defined row add nothing
    rows = [r for r in rows if len(r)]
    reduced, fitted = _model_reduce(rows, rfun, ndims, return_model=True)
    label = _model_label(rfun)
    where = f"{name}{at}"
    if fitted.model is None:
        # hypertools fits no model to data with no more columns than the
        # components asked for: it returns them as they are (and reduce
        # appends its zero columns), which the identity does for any row
        if all(np.array_equal(np.asarray(r)[:, :fitted.n_features], a)
               for r, a in zip(reduced, rows)):
            return fitted
        raise _FitUnusable(
            f"causal=True: the model {label!r} ({where}) cannot reduce rows it "
            "was not fitted on (hypertools returned no fitted model), so it "
            "cannot be fitted on the first rows and applied to later ones.  "
            + _LINEAR_MODELS
        )
    if fitted.components is None:
        raise ValueError(_not_linear_message(label, where))
    try:
        again = fitted.apply(rows)
    except (NotImplementedError, AttributeError, TypeError) as err:
        raise _FitUnusable(
            f"causal=True: the model {label!r} ({where}) cannot reduce rows it "
            f"was not fitted on ({err}), so it cannot be fitted on the first "
            "rows and applied to later ones.  " + _LINEAR_MODELS
        ) from err
    worst = max(float(np.max(np.abs(np.asarray(r) - a), initial=0.0))
                for r, a in zip(reduced, again))
    scale = max(1.0, max(float(np.max(np.abs(a), initial=0.0)) for a in again))
    if not worst <= _REFIT_ATOL * scale:
        raise _FitUnusable(
            f"causal=True: the model {label!r} ({where}) does not reproduce "
            f"the rows it was fitted on when it reduces them again (they differ "
            f"by up to {worst:.3g}), so the rows it is fitted on and later rows "
            "would not be comparable.  " + _LINEAR_MODELS + "  Use an exact "
            "solver (a randomized one, as svd_solver='randomized' of PCA, does "
            "not reproduce its rows)."
        )
    # each row alone, and in other company: the same output
    few, batch = np.vstack(rows)[:4], np.vstack(again)[:4]
    try:
        alone = np.vstack([fitted.apply([few[i:i + 1]])[0] for i in range(len(few))])
        turned = np.vstack(fitted.apply([few[1:], few[:1]])[::-1])
    except Exception as err:
        raise ValueError(
            f"causal=True: the model {label!r} ({where}) does not reduce its "
            f"rows row by row: it cannot reduce one row alone ({err}), so "
            "nothing shows that a row does not depend on the rows after it.  "
            + _LINEAR_MODELS
        ) from err
    worst = max(float(np.max(np.abs(other - batch), initial=0.0))
                for other in (alone, turned))
    if not worst <= _REFIT_ATOL * scale:
        raise ValueError(
            f"causal=True: the model {label!r} ({where}) does not reduce its "
            "rows row by row: a row reduced alone or with other rows differs "
            f"(by up to {worst:.3g}) from the row as reduced with the rows of "
            "the fit, so a row would depend on the rows after it.  "
            + _LINEAR_MODELS
        )
    fitted.freeze_constants(again)
    return fitted


# what the errors about a model in causal mode recommend
_LINEAR_MODELS = (
    "Use a model that is a linear map of the correlations of a row, with "
    "components_: 'PCA', 'IncrementalPCA', 'TruncatedSVD', 'FastICA' or "
    "'FactorAnalysis' (the models the guarantee covers); or a summary such "
    "as 'weighted_strength'."
)


# the models the causal guarantee covers: scikit-learn's classes of these
# names, which hypertools fits for them.  Their ``transform`` is a fixed
# linear map of each row.  A fitted estimator is one of them by its exact
# class: a subclass can pass every probe of ``_fit_reducer`` (they reduce
# training rows only) and still reduce a row differently with later rows.
_GUARANTEED_MODELS = ("PCA", "IncrementalPCA", "TruncatedSVD", "FastICA",
                      "FactorAnalysis")
_GUARANTEED_TEXT = (
    f"scikit-learn's {', '.join(_GUARANTEED_MODELS[:-1])} and "
    f"{_GUARANTEED_MODELS[-1]}, by exact class: a subclass is not one"
)


def _uncovered_model(rfun):
    """
    The label of a model reducer (a name or a dict) that is not one of
    ``_GUARANTEED_MODELS``: another name, or a class or object that is not
    exactly one of those scikit-learn classes.  None for a model of the
    list and for a reducer that is not a model.
    """
    if not _is_model(rfun):
        return None
    model = rfun.get("model") if isinstance(rfun, dict) else rfun
    if isinstance(model, str):
        covered = model in _GUARANTEED_MODELS
    else:
        import sklearn.decomposition

        classes = [getattr(sklearn.decomposition, name) for name in _GUARANTEED_MODELS]
        cls = model if isinstance(model, type) else type(model)
        covered = any(cls is c for c in classes)
    return None if covered else _model_label(rfun)


def _warn_uncovered_models(found):
    """
    One ``UserWarning`` for the models of a causal call that are outside
    ``_GUARANTEED_MODELS``: ``found`` is a list of ``(label, where)``, as
    ``(_uncovered_model(rfun), "rfun at order 1")``; entries with a label
    of None are models of the list.  Such a model is used when it passes
    the probes of ``_fit_reducer``, like a function of the user's.
    """
    places = {}
    for label, where in found:
        if label is not None:
            places.setdefault(label, []).append(where)
    if not places:
        return
    named = [f"{label!r} ({', '.join(where)})" for label, where in places.items()]
    one = len(named) == 1
    _warn_user(
        f"causal=True: the model{'' if one else 's'} "
        f"{', '.join(named[:-1])}{'' if one else ' and '}{named[-1]} "
        f"{'is not one of' if one else 'are not among'} the models the guarantee "
        f"covers ({_GUARANTEED_TEXT}).  "
        f"{'It is' if one else 'They are'} used as given: the guarantee that row "
        "t does not depend on later rows does not cover what "
        f"{'its transform does' if one else 'their transforms do'}."
    )


def _model_label(rfun):
    """The name of a model reducer (a name or a dict), for messages."""
    label = rfun.get("model") if isinstance(rfun, dict) else rfun
    return getattr(label, "__name__", None) or str(label)


def _not_linear_message(label, where):
    return (
        f"causal=True: the model {label!r} ({where}) has no components_ (it is "
        "not a linear map of the correlations of a row), so nothing shows "
        "that it reduces a row in the same way whatever rows come with it "
        "(reproducing the rows it was fitted on does not: the transform of "
        "'UMAP' depends on the number of rows it is given).  " + _LINEAR_MODELS
    )


def _check_linear_model(rfun, where):
    """
    ``ValueError`` for a model reducer named in ``rfun`` (a name or a dict
    with one) that is known to have no ``components_``, before anything is
    computed (``causal=True``; a model given as an object is checked when
    it has been fitted: ``_fit_reducer``).
    """
    if not _is_model(rfun):
        return
    label = rfun.get("model") if isinstance(rfun, dict) else rfun
    if isinstance(label, str) and label not in _COMPONENT_MODELS:
        raise ValueError(_not_linear_message(label, where))


def _exact_pca(rfun):
    """
    ``"PCA"`` (a name, or a dict that gives no ``svd_solver``) with
    ``svd_solver="full"``; any other reducer unchanged.  scikit-learn's
    default chooses a randomized solver for large inputs, whose
    ``fit_transform`` is not the projection ``transform`` computes, so the
    rows of the fit would not be reproduced.
    """
    if isinstance(rfun, str):
        rfun = {"model": rfun}
    if not (isinstance(rfun, dict) and isinstance(rfun.get("model"), str)
            and rfun["model"] == "PCA"):
        return rfun
    given = {**(rfun.get("kwargs") or {}), **(rfun.get("params") or {})}
    if "svd_solver" in given:
        return rfun
    return {**rfun, "kwargs": {**(rfun.get("kwargs") or {}), "svd_solver": "full"}}


def _apply_frozen(corrs, frozen):
    """
    The reduction of ``corrs`` (an array or a list of arrays of vectorized
    correlations of the causal estimator) by ``frozen``, a result of
    ``_fit_reducer``: a fitted summary or model and a graph measure reduce
    the rows that are not entirely NaN (the others stay rows of NaN); None
    returns ``corrs``; a function of the user's is called as ``reduce``
    calls it.  The same container type as ``corrs``.
    """
    from .summaries import _FittedSummary

    if isinstance(frozen, str):
        return _reduce_defined(corrs, frozen)
    if not isinstance(frozen, (_FittedModel, _FittedSummary)):
        return reduce(corrs, rfun=frozen)
    arrays = corrs if isinstance(corrs, list) else [corrs]
    if isinstance(frozen, _FittedSummary):
        K = frozen.fit["weights"][0].shape[0]
        if arrays and np.asarray(arrays[0]).shape[1] == K * (K + 1) // 2:
            frozen.check_units(len(arrays), K)
    out = _on_defined(arrays, frozen.apply)
    return out if isinstance(corrs, list) else out[0]


def _causal_arguments(weights_function, weights_params, include_timepoints, cfun,
                      rfun, events, T, n_subjects, fit_timepoints=None):
    """
    The checks of ``timecorr(..., causal=True)``.  Returns
    ``(include_timepoints, cfun)``: ``'backward'``, and the correlation
    function to call (a built-in one is called with ``causal=True``).
    Raises ``ValueError`` for what has no causal form, and warns once when
    ``cfun`` or ``rfun`` is a function of the user's.
    """
    side = str(include_timepoints).lower()
    if side in ("forward", "post"):
        raise ValueError(
            f"causal=True: include_timepoints={include_timepoints!r} weights "
            "later timepoints.  Leave include_timepoints out (causal=True uses "
            "'backward'), or pass causal=False."
        )
    if side in ("all", "backward", "pre"):
        include_timepoints = "backward"
    if weights_function is eye_weights:
        raise ValueError(
            "causal=True: eye_weights gives co-fluctuations of series z-scored "
            "over the whole recording, which use later timepoints.  Pass a "
            "kernel with a width, for example weights_function="
            "tc.gaussian_weights, weights_params={'var': 25}."
        )
    if weights_params is None:
        raise ValueError(
            "causal=True: weights_params must be given (the kernel's width is "
            "not chosen for you), for example weights_params={'var': 25} for "
            "gaussian_weights; pass {} for a kernel without parameters."
        )
    if cfun is wisfc or (
        isinstance(cfun, functools.partial) and cfun.func is wisfc
        and not _fixed_weights_wisfc(cfun)
    ):
        raise ValueError(
            "causal=True: the default subject_weights of wisfc are computed "
            "from every timepoint of the recording.  Pass cfun="
            "functools.partial(tc.wisfc, subject_weights=...) with weights of "
            "your own and nothing else bound, or cfun=tc.isfc."
        )
    if _has_features(_parse_spec(events, T, n_subjects, True)):
        raise ValueError(
            "causal=True: events given as features are segmented from the whole "
            "recording.  Pass the boundaries (the timepoints where a new block "
            "starts), for example events=[100, 250]."
        )
    _check_rfun(rfun)
    _check_linear_model(rfun, "rfun")
    if _needs_fit(rfun) and fit_timepoints is None:
        raise ValueError(
            f"causal=True: rfun={rfun!r} is fitted to data and needs "
            "fit_timepoints=n: it is then fitted on the first n rows and applied "
            "to every row.  Or use rfun=None or a graph measure such as "
            "'strength', which need no fit."
        )
    builtin = cfun is None or cfun in (isfc, autofc) or _fixed_weights_wisfc(cfun)
    from .summaries import _Summary  # timecorr.summaries imports helpers

    theirs = [name for name, f, ours in (
        ("cfun", cfun, builtin),
        ("rfun", rfun, not callable(rfun) or isinstance(rfun, _Summary)),
    ) if not ours]
    if theirs:
        _warn_user(
            f"causal=True: {' and '.join(theirs)} "
            f"{'is a function' if len(theirs) == 1 else 'are functions'} of your "
            "own; the guarantee that row t does not depend on later rows does "
            "not cover what "
            f"{'it does' if len(theirs) == 1 else 'they do'}."
        )
    _warn_uncovered_models([(_uncovered_model(rfun), "rfun")])
    return include_timepoints, cfun


def _reduce_defined(corrs, rfun):
    """
    ``reduce(corrs, rfun)`` for a graph measure on the output of the causal
    estimator: the measure is computed for the defined rows; rows of NaN
    stay rows of NaN.
    """
    arrays = corrs if isinstance(corrs, list) else [corrs]
    out = []
    for c in arrays:
        c = np.asarray(c)
        defined = ~np.isnan(c).all(axis=1)
        if not defined.any():
            width = np.asarray(reduce(np.nan_to_num(c[:1]), rfun=rfun)).shape[1]
            out.append(np.full((c.shape[0], width), np.nan))
            continue
        reduced = np.asarray(reduce(c[defined], rfun=rfun))
        full = np.full((c.shape[0], reduced.shape[1]), np.nan)
        full[defined] = reduced
        out.append(full)
    return out if isinstance(corrs, list) else out[0]


def timecorr(
    data,
    weights_function=gaussian_weights,
    weights_params=None,
    include_timepoints="all",
    exclude_timepoints=None,
    combine=null_combine,
    cfun=isfc,
    rfun=None,
    include_zero=True,
    *,
    events=None,
    events_params=None,
    causal=False,
    min_samples=None,
    fit_timepoints=None,
):
    """
    Compute dynamic (moment-by-moment) correlations in one or more datasets.

    For each timepoint ``t``, row ``t`` of the kernel matrix
    (``weights[t, :]``) weights every timepoint's contribution, and ``cfun``
    computes kernel-weighted correlations (see ``helpers.wcorr``).  The
    estimator weights the covariance and variance sums as well as the means;
    it differs from Eq. 4 of Owen, Chang & Manning (2021) and from timecorr
    0.2.0, where the kernel set only the means, so numerical results differ
    from 0.2.0.  A feature whose kernel-weighted variance at a timepoint is
    zero, or zero up to round-off relative to its kernel-weighted mean
    square (``wcorr`` gives the rule; it does not depend on the units of the
    data), has correlation 0 with every feature there.  For higher-order
    correlations computed as in the paper (Fig. 10), see
    ``timecorr.features``.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
        A timepoints-by-features matrix, or a list of them (one per subject,
        all the same shape).  A 1-D array or a pandas or polars Series is one
        feature (shape ``(T, 1)``); arrays with more than two dimensions raise
        ``ValueError``.  Inputs are not modified.  NaN samples carry no
        weight in the correlations.
    weights_function : function
        Called as ``weights_function(T, weights_params)``; returns a T x T
        kernel matrix whose row ``t`` is the kernel used at timepoint ``t``.
        Options: ``gaussian_weights`` (default), ``laplace_weights``,
        ``t_weights``, ``mexican_hat_weights``, ``boxcar_weights``,
        ``uniform_weights``, ``eye_weights`` (a delta kernel, which gives
        co-fluctuations: products of the z-scored series), or a function of
        your own with the same signature.  The built-in kernel functions are
        not called: the same weights are computed 128 rows at a time, over
        the columns where those rows can have weight, so with ``cfun``
        ``autofc``, ``isfc``, ``wisfc`` or None no T x T array is allocated
        and memory grows linearly in T (see Notes).  A function of your own
        is called, and its T x T array is used; a ``cfun`` of your own
        receives T x T arrays.
    weights_params : dict or None
        Parameters of the kernel, e.g. ``{"var": 100}`` for
        ``gaussian_weights``.  Default: None (the kernel's defaults:
        ``gaussian_params``, ``laplace_params``, ``t_params``, ``eye_params``,
        ``mexican_hat_params``, ...).
    include_timepoints : str
        Which timepoints may contribute to the estimate at each timepoint:
        "all" (default), "backward" (alias "pre": only timepoints at or before
        ``t``), or "forward" (alias "post": only timepoints at or after ``t``).
        Applied to the kernel after it is built (see ``one_sided``).  The
        estimator centers each feature on its mean over all timepoints, so
        an estimate whose kernel-weighted variance is a small fraction (near
        1e-10, see ``wcorr``) of the mean square of the centered feature can
        change, by round-off or between 0 and a non-zero value, when
        timepoints outside its kernel row change; other estimates do not.
    exclude_timepoints : int or None
        Further restricts the kernel by distance from ``t``.  A positive value
        ``x`` keeps only the timepoints at most ``x`` samples from ``t``; a
        negative value ``-x`` removes them (``t`` itself included), keeping
        only timepoints more than ``x`` samples away.  Real values are rounded
        to the nearest integer; a non-finite value (inf or NaN) raises
        ``ValueError``.  Default: None (no restriction).
    combine : function
        Applied to the per-subject results: ``null_combine`` (default; one
        result per subject), ``corrmean_combine`` (Fisher z average),
        ``mean_combine``, or ``tstat_combine``.
    cfun : function or None
        Correlation function, called as ``cfun(data, weights)``:

        - ``isfc`` (default): with a list, each subject's features versus the
          mean of the other subjects' features (inter-subject functional
          connectivity; Simony et al., 2016); with one array, its own
          correlations.
        - ``wisfc``: versus a weighted mean of the other subjects.
        - ``autofc``: each subject's own correlations.
        - None: the data are smoothed instead of correlated: the output at
          timepoint ``t`` is the mean of the samples weighted by
          ``abs(weights[t, :])`` (the same kernel rows, after
          ``include_timepoints``, ``include_zero`` and
          ``exclude_timepoints``), one timepoints-by-features array per
          subject.  NaN samples get no weight.
    rfun : None, str, dict, or callable
        Dimensionality reduction of the vectorized correlations (see
        ``helpers.reduce``): a hypertools / scikit-learn model name such as
        "PCA", "FastICA", "FactorAnalysis" or "IncrementalPCA"; a dict such as
        ``{"model": "PCA", "params": {"whiten": True}}``; a graph measure
        ("eigenvector_centrality", "pagerank_centrality",
        "pagerank_centrality_abs", "strength"); a
        squared-correlation summary (``tc.WeightedStrength``,
        ``tc.Communities``, or the strings "weighted_strength",
        "weighted_strength_split", "communities"); or a callable
        ``f(list_of_arrays) -> list_of_arrays``.  Model-based
        reduction returns as many features as the data have.  Default: None
        (no reduction).
    include_zero : bool
        If False, the timepoint being estimated is left out of its own
        estimate (the diagonal of the kernel is zeroed).  Default: True.
    events : None, array-like, or list
        Restricts the kernel to events (blocks of consecutive timepoints).
        A boundary set is a 1-D sequence of integer timepoint indices at which
        a new block starts (a list or tuple of integers, a 1-D integer numpy
        array, or a pandas or polars Series); index 0 is implied and an empty
        sequence means a single block (no restriction).  A boundary set is
        shared by all subjects.  A list or tuple with one entry per subject,
        each ``None`` (no restriction) or a boundary set, gives per-subject
        events.  The kernel weight ``weights[t, u]`` is set to 0 whenever
        ``t`` and ``u`` lie in different blocks, after ``include_timepoints``,
        ``include_zero`` and ``exclude_timepoints`` are applied (the estimator
        normalizes each row, so no renormalization is needed).  The smoothing
        path (``cfun=None``) uses the same restricted kernel.  A 2-D array or a
        DataFrame with T rows (shared, or as a per-subject entry) holds features
        that are segmented into events with ``segment`` (with
        ``events_params``) before any correlations are computed; the same
        features object is segmented once.  See
        ``timecorr.events.resolve_events`` for the full grammar.  Default:
        None (no restriction).
    events_params : dict or None
        Keyword arguments for the segmentation of features entries in
        ``events`` (keys ``n``, ``kind``, ``max_events``, ``random_state``).
        Unknown keys, or giving it when no entry of ``events`` holds features,
        raise ``ValueError``.  Default: None.
    causal : bool
        True: row ``t`` of the output uses no sample after timepoint ``t``.
        It is unchanged (to 1e-9) when the rows after ``t`` are replaced by
        any values, NaN and inf included, or removed; with
        ``include_zero=False`` also when row ``t`` itself changes.
        Replacing later rows changes no bit, with every supported ``rfun``.
        Removing them changes the correlations by round-off, so the
        outputs agree to 1e-9 where they depend continuously on the
        correlations (no reduction, ``"strength"``, the summaries, and
        linear models such as PCA once fitted); the eigenvector-based
        graph measures (``"eigenvector_centrality"``, the PageRank
        measures) can differ by more at a correlation matrix whose leading
        eigenvalues coincide.  Smoothed values (``cfun=None``) are on the
        scale of the data, not bounded like correlations: when later rows
        are removed they agree to round-off relative to the magnitude of
        the data, that is to 1e-9 or 8 units in the last place of the
        largest magnitude of the data, whichever is larger (1.2e-7 for
        data near 1e8); replacement still changes no bit.  Keyword only.
        Default: False.  With ``causal=True``:

        - the kernel is backward: ``include_timepoints`` 'all' (the
          default) means 'backward', and 'forward' / 'post' raise
          ``ValueError``;
        - ``weights_params`` must be given, and ``weights_function`` cannot
          be ``eye_weights`` (its z-scores use every timepoint); a kernel
          function of your own must return a matrix without weight above
          its diagonal (for example ``tc.one_sided(w, 'backward')``), and
          removing rows is covered only if its weights do not depend on T;
        - ``cfun`` is ``isfc``, ``autofc``, None, or
          ``functools.partial(tc.wisfc, subject_weights=...)`` with weights
          of your own (the default weights of ``wisfc`` use every
          timepoint, so ``cfun=tc.wisfc`` raises ``ValueError``).  Any
          other function is called as without ``causal``, with a
          ``UserWarning`` that the guarantee does not cover it; the same
          holds for a callable ``rfun``, and for a model outside the list
          under ``fit_timepoints``;
        - ``events`` must be boundaries; features to segment raise
          ``ValueError``.  That the boundaries themselves were chosen
          without later data is the caller's responsibility.  The warm-up
          below starts again at each boundary.  Boundaries at or beyond
          the end of the data are ignored (times the series has not
          reached), and events that leave no timepoint with 3 samples give
          rows of NaN, not an error, so a prefix of a series runs with the
          events of the whole series;
        - an ``rfun`` that is fitted to data (a model name or dict, a
          summary) needs ``fit_timepoints``: it is fitted on the first
          ``fit_timepoints`` rows and then applied, unchanged, to every
          row.  None, a graph measure (computed for the defined rows) and
          a callable need no fit;
        - infinite samples and samples above 1e150 in magnitude are missing
          data, like NaN;
        - the estimator centers each feature, a chunk of 128 timepoints at
          a time, on the mean of its earlier samples that the chunk
          weights, and its zero-variance rule is relative to that reference
          (see ``wcorr``);
        - timepoints with fewer than 3 samples are rows of NaN, not 0 (see
          ``min_samples``).
    min_samples : float or None
        With ``causal=True``, a timepoint with fewer than 3 observed
        timepoints with non-zero weight in its kernel row is a row of NaN
        (observed: the data, and for ``isfc`` and ``wisfc`` the reference,
        have at least one sample there): the rows that are 0 without
        ``causal``.  A number (>= 1) is also the effective sample size
        ``(sum |w|)**2 / sum w**2`` a timepoint needs over those
        timepoints, compared with a relative tolerance of 1e-12 (three
        equal weights are 3 samples); rows below it are NaN too.  A number
        that a built-in kernel cannot reach with any number of earlier
        timepoints raises ``ValueError`` (a backward Gaussian of variance
        25 has at most 9.4 effective samples), whatever the length of the
        data; the rows of a short recording that do not reach a number the
        kernel can reach are NaN, as are all rows when a kernel function
        of your own never reaches it.  With
        ``cfun=None`` a value is NaN when its feature has no weighted
        sample, or fewer effective samples than the number.  Features with
        zero variance still give correlations of 0.  Keyword only.
        Default: None.
    fit_timepoints : int or None
        With ``causal=True``, the number ``n`` of first rows that ``rfun``
        is fitted on (an integer from 1 to the number of timepoints), for
        an ``rfun`` that is fitted to data: a hypertools model (the
        components of ``"PCA"``) or a summary (the pair weights and
        communities of ``tc.WeightedStrength`` and ``tc.Communities``; an
        object given as ``rfun`` holds that fit in ``fits_`` afterwards).
        It is fitted to the correlations of rows ``0..n-1`` that are
        defined (not NaN: see ``min_samples``; a model needs 2 such rows,
        a summary 1) and then reduces every row, so row ``t >= n`` is
        unchanged (to 1e-9; bit for bit under replacement) when rows after
        ``t`` are replaced or removed, and a row ``t < n`` when rows from
        ``n`` on are.  ``n`` equal to the number of timepoints fits on
        every row.  A model must be a linear map applied to each row by itself.
        The guarantee covers ``"PCA"``, ``"IncrementalPCA"``,
        ``"TruncatedSVD"``, ``"FastICA"`` and ``"FactorAnalysis"``
        (scikit-learn's classes of those names, by exact class: a subclass is
        not one).  Any other model whose fitted estimator has ``components_``,
        reproduces its training rows to 1e-9 and reduces rows one at a time as
        it reduces them together (``"MiniBatchDictionaryLearning"``, a class of
        your own) is used with one ``UserWarning`` per call that the guarantee
        does not cover what its transform does: those checks read a few
        training rows and cannot show how it reduces later rows.  A model
        without ``components_``, or that fails a check, raises ``ValueError``
        (``"UMAP"``, ``"KernelPCA"``, ``"Isomap"``, ``"MDS"``, ``"TSNE"``,
        ``"SpectralEmbedding"``, ``"LocallyLinearEmbedding"``: the transform of
        some of them depends on the rows it is given together).
        A model asked for as many components as its input
        has columns, or more, changes nothing (zero columns are appended
        for the components beyond them), as without ``causal``.
        ``"PCA"`` is
        fitted with ``svd_solver="full"`` unless a model dict gives a
        solver, because the randomized solver scikit-learn chooses for
        large inputs does not reproduce its rows.  An output column of
        a model that is constant on the training rows up to round-off (a
        range of at most 1e-9 of its largest magnitude, or 1e-12 of the
        largest magnitude of the output) is frozen: at every row, a value
        within that tolerance of the column's training mean is returned as
        exactly that mean.  Output columns
        of a summary that are constant up to round-off are not set to
        their mean, and no warning reports them.  Required
        (``ValueError``) for such an ``rfun``; not used by any other, nor
        with ``cfun=None``.  Given without ``causal=True``, it raises
        ``ValueError``.  Keyword only.  Default: None.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        One row per timepoint, each a vectorized correlation matrix (see
        ``mat2vec``: the K diagonal entries, then the K(K-1)/2 entries above
        the diagonal), so each array has shape (T, K * (K + 1) / 2); or the
        reduced features when ``rfun`` is given ((T, K) for model-based
        reduction and graph measures); or the smoothed (T, K) data when
        ``cfun`` is None.  A list (one entry per subject)
        when ``combine`` returns one; a single array otherwise.  With
        ``causal=True`` the timepoints with too few samples (the first ones,
        and those after long gaps or event boundaries) are rows of NaN.

    Warns
    -----
    UserWarning
        From the correlation estimator (see ``wcorr``), one per call, naming
        each that applies: timepoints whose kernel row has fewer than 3
        non-zero kernel weights (for example the first timepoints of a backward
        kernel; their correlations are undefined and set to 0); other
        timepoints whose effective sample size ``(sum |w|)**2 / sum w**2`` is
        below 2 (the kernel is too narrow to estimate correlations reliably
        there); timepoints where, because of NaN samples, some pairs of
        features have fewer than 3 observed samples with non-zero kernel weight
        (those correlations are undefined and set to 0); and timepoints with
        zero kernel-weighted variance in every feature (their correlations are
        set to 0).  Never for a delta kernel.  With per-subject events, a
        user-supplied ``cfun`` is called once per distinct event set, so it may
        warn once per set.
    UserWarning
        With ``cfun=None``, when some timepoints have no weighted, observed
        sample: the smoothed values there are undefined and set to 0.  With
        per-subject events this warning is issued per subject, as it is
        without events.

    Raises
    ------
    ValueError
        For an unknown ``include_timepoints``, subjects of different shapes,
        non-numeric columns, or a kernel whose rows are all zero; for a
        malformed ``events`` specification; when ``events`` leave every kernel
        row with fewer than 3 non-zero weights (no block is long enough to
        estimate a correlation); and, with per-subject events, when a custom
        ``cfun`` does not return a list with one array per subject; and as
        raised by ``segment`` for features entries of ``events``.  With
        ``causal=True``, for the arguments listed under ``causal`` that have
        no causal form, and for a ``min_samples`` that is not None or a
        finite number >= 1, or that a built-in kernel cannot reach; for
        ``min_samples`` given without ``causal=True``; for an ``rfun``
        that is fitted to data without ``fit_timepoints``; for a
        ``fit_timepoints`` that is
        not an integer from 1 to the number of timepoints, or given
        without ``causal=True``; when the first ``fit_timepoints`` rows
        hold no row with defined correlations to fit (2 for a model); and
        for a model that is not a linear map applied row by row (see
        ``fit_timepoints``).
    RuntimeError, ImportError
        As raised by ``segment`` for features entries of ``events``.

    Notes
    -----
    A feature is treated as constant within a kernel row (correlation 0, see
    ``wcorr``) when its kernel-weighted variance is at most 1e-10 of its
    centered mean square, or when its ratio of kernel-weighted standard
    deviation to root mean square is at most 48 eps (1.07e-14, the
    round-off this library's own arithmetic produces with a margin): a
    feature on an offset of more than about 9.4e13 of its local standard
    deviations counts as constant.

    With ``events``, the kernel gives no weight to timepoints outside the
    current block.  This is not statistical independence from the rest of the
    recording: the subject weights of ``wisfc`` and the reductions (``rfun``)
    still use all timepoints, and the estimator centers each feature over all
    timepoints, so outputs inside a block can change by floating-point
    rounding when data outside the block change.

    With per-subject events, subject s's output at timepoint ``t`` is computed
    with subject s's restricted kernel row, applied both to subject s's data
    and to the reference it is correlated with (the average of the other
    subjects for ``isfc``, the weighted average for ``wisfc``, subject s itself
    for ``autofc``); the reference is not restricted by the other subjects'
    events.  Subjects with identical boundaries share one kernel.  ``isfc``,
    ``wisfc`` and ``autofc`` receive the list of per-subject kernels in one
    call (with a kernel function of your own, one T x T array per distinct
    event set is held in memory).  Any other ``cfun`` is called once per
    distinct event set with the full data and that event set's T x T kernel
    array (the arrays of all distinct event sets are built before the first
    call and held until the last); it must return a list with one array per
    subject, and subject s's output is taken from the call made with subject
    s's kernel.

    Kernels and memory: with a built-in kernel function and ``cfun`` one of
    ``isfc``, ``wisfc``, ``autofc`` or None, the kernel is never built as a
    T x T array.  The estimator reads it 128 rows at a time, each chunk
    over the columns where its rows have non-zero weight, and keeps at most
    32 MB of such blocks.  A kernel function of your own is read the same
    way, from its T x T array, so the correlations of a built-in kernel and
    of the same kernel as an array are equal.  The smoothed values
    (``cfun=None``) are computed a chunk of rows at a time too; an infinite
    sample enters every row's sums, as in the full product.  The time grows
    linearly in T once T is well above the kernel's span (the
    largest lag with a non-zero weight in float64: 192 samples for a
    Gaussian of variance 25, 385 at variance 100, 73,983 for the Laplace
    kernel at its default scale of 100, 15,823 for the t kernel at its
    default df of 100); the uniform kernel has weight at every lag, so its
    time grows with ``T**2`` while its memory still grows linearly.  A
    custom ``cfun`` receives T x T arrays.

    With per-subject events and a cross-subject correlation function (``isfc``,
    ``wisfc``), the timepoints that enter subject s's correlations at time t are those
    in subject s's current block, for both subject s's data and the other subjects'
    data. If the subjects' events occur at different times, the other subjects' data
    inside that window may come from different events. Cross-subject correlations
    assume that all subjects experience the same external stimulus at the same times;
    per-subject events that are not aligned across subjects are better suited to
    within-subject correlations (``cfun=autofc``).

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> data = np.random.RandomState(0).randn(100, 5)
    >>> corrs = tc.timecorr(data, weights_params={"var": 50})
    >>> corrs.shape
    (100, 15)
    >>> subjects = [np.random.RandomState(s).randn(100, 5) for s in range(3)]
    >>> group = tc.timecorr(subjects, cfun=tc.isfc, combine=tc.corrmean_combine)
    >>> group.shape
    (100, 15)
    >>> tc.timecorr(data, rfun="PCA").shape
    (100, 5)

    Restrict the kernel to three events (timepoints 0-39, 40-69 and 70-99):

    >>> blocks = tc.timecorr(data, weights_params={"var": 50}, events=[40, 70])
    >>> blocks.shape
    (100, 15)
    >>> full = tc.gaussian_weights(100, {"var": 50})
    >>> alone = tc.timecorr(data[40:70], weights_function=lambda T, p: full[40:70, 40:70])
    >>> bool(np.allclose(blocks[40:70], alone))
    True
    >>> per_subject = tc.timecorr(subjects, cfun=tc.autofc, events=[[40], [50, 80], None])
    >>> len(per_subject), per_subject[0].shape
    (3, (100, 15))

    Causal estimates: row 60 does not change when later rows do.  The first
    2 rows have fewer than 3 samples and are NaN (a ``UserWarning`` says
    so).

    >>> past = tc.timecorr(data, causal=True, weights_params={"var": 50})
    >>> bool(np.isnan(past[:2]).all()), bool(np.isnan(past[2:]).any())
    (True, False)
    >>> later = data.copy()
    >>> later[61:] = 1000.0
    >>> changed = tc.timecorr(later, causal=True, weights_params={"var": 50})
    >>> bool(np.array_equal(changed[60], past[60]))
    True

    A reducer that is fitted (here PCA) is fitted on the first
    ``fit_timepoints`` rows and applied to every row:

    >>> kw = dict(causal=True, weights_params={"var": 50}, rfun="PCA",
    ...           fit_timepoints=50)
    >>> reduced = tc.timecorr(data, **kw)
    >>> reduced.shape
    (100, 5)
    >>> bool(np.array_equal(tc.timecorr(later, **kw)[60], reduced[60]))
    True
    """

    # NaN samples are kept; both the correlation functions and the smoothing
    # path (cfun=None) give them no weight.
    data = format_data(data, fill_nan=False)
    T = data[0].shape[0]

    min_samples = _check_min_samples(min_samples, causal)
    fit_timepoints = _check_fit_timepoints(fit_timepoints, causal, T)
    if causal:
        include_timepoints, cfun = _causal_arguments(
            weights_function, weights_params, include_timepoints, cfun, rfun,
            events, T, len(data), fit_timepoints,
        )
    # the built-in correlation functions: one kernel per subject, never T x T
    builtin = cfun in _KERNEL_LIST_CFUNS or (causal and _fixed_weights_wisfc(cfun))
    estimator = {"causal": True, "min_samples": min_samples} if causal and builtin else {}

    weights = _shaped_kernel(
        T,
        weights_function,
        weights_params,
        include_timepoints,
        include_zero,
        exclude_timepoints,
        causal,
    )

    # event restriction: applied after the include/exclude options; subjects that
    # share events share one masked kernel (one array, or a list per subject)
    resolved = resolve_events(events, T, len(data), events_params, causal=causal)
    kernels = _event_kernels(weights, resolved, len(data), causal)
    per_subject = isinstance(kernels, list)

    if cfun:
        if not builtin:
            # a correlation function of the user's receives T x T arrays
            if per_subject:
                arrays = {}
                for k in kernels:  # subjects that share events share one array
                    if id(k) not in arrays:
                        arrays[id(k)] = _as_kernel_array(k)
                kernels = [arrays[id(k)] for k in kernels]
            else:
                kernels = _as_kernel_array(kernels)
        if not per_subject or builtin:
            corrs = cfun(data, kernels, **estimator)
        else:
            groups = distinct_event_sets(_restricted(resolved), len(data))
            corrs = _per_subject_cfun(cfun, data, kernels, groups)
        if causal:
            # a reducer that is fitted: on the first fit_timepoints rows only
            corrs = combine(corrs)
            first = ([c[:fit_timepoints] for c in corrs] if isinstance(corrs, list)
                     else corrs[:fit_timepoints])
            corrs = _apply_frozen(corrs, _fit_reducer(first, rfun))
        else:
            corrs = reduce(combine(corrs), rfun=rfun)

    else:
        # smoothing path: kernel-weighted mean of the samples, using the same
        # kernel rows (after include_timepoints, include_zero, exclude_timepoints
        # and events) as the correlation path
        if causal:
            subject_kernels = kernels if per_subject else [kernels] * len(data)
            smoothed = [_causal_smooth(x, k, min_samples)
                        for x, k in zip(data, subject_kernels)]
        elif per_subject:
            smoothed = [_kernel_smooth(x, k) for x, k in zip(data, kernels)]
        else:
            smoothed = [_kernel_smooth(x, kernels) for x in data]
        corrs = combine(smoothed)

    if isinstance(corrs, list) and len(corrs) == 1:
        return corrs[0]
    return corrs
