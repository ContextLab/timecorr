# coding: utf-8
"""
Higher-order features through two paths per order (Owen, Chang & Manning, 2021, Fig. 10).

At each order ``k >= 1`` the previous order's feeding features ``D_{k-1}``
(``D_0`` is the data) feed two computations:

- the analysis path, ``X_k = reduce(combine(timecorr(D_{k-1}, analysis kernel,
  cfun)), rfun[k])``, which gives that order's output features, and
- the feeding path, ``D_k = reduce(timecorr(D_{k-1}, feeding kernel, autofc),
  delta_rfun[k])``, whose output is passed to the next order.

The feeding path is called the delta path in argument names (``delta_rfun``,
``return_delta``, ``delta_weights_function``) for historical reasons: in the
original pipeline its kernel is ``eye_weights``, which gives unblurred
co-fluctuations (products of z-scored series), so temporal blur is applied
once, at the order being analyzed.

Defaults: ``rfun="auto"`` (``"PCA"`` at order 1,
``WeightedStrength(signed=True)`` at orders 2 and above),
``delta_rfun="auto"`` (``Communities()``, with the number of communities
counted with a shift null) and a Gaussian feeding kernel with variance 25
(standard deviation 5 samples), so the feeding features are kernel-weighted
correlations summarized by community.  The feeding kernel is shaped by
``include_timepoints``, ``include_zero`` and the events of each order, as
the analysis kernel is.  The default feeding reducer needs at least 4
features and 40 timepoints (``ValueError`` otherwise);
``delta_rfun="weighted_strength"`` takes any number of features and
timepoints.  When the communities of a default feeding step would emit
fewer than 4 columns, that step emits ``WeightedStrength(split=False,
signed=False, center=False)`` of the same correlations instead (one column
per input feature; the pair weights are used as they are, without
centering or clipping); a ``UserWarning`` is issued only when the shift null
found no structure.  The original two-path pipeline (Owen, Chang & Manning, 2021,
Fig. 10) is ``rfun="PCA", delta_rfun=None, delta_weights_function=eye_weights``.

A step whose reducer is a squared-correlation summary (``WeightedStrength``
or ``Communities``) and whose correlations come from ``autofc``, ``isfc`` or
``wisfc`` with ``null_combine`` (every feeding step with a kernel other than
``eye_weights`` qualifies) is streamed: its correlations are computed in row
blocks sized to ``max_memory_gb`` (the estimator's default block size when it
is None), and the full (d, d, T) array of a subject is held only when one
block holds all d rows.
"""

import inspect
import numbers

import numpy as np

from .helpers import (
    _SUMMARY_NAMES,
    _check_ndims,
    _check_rfun,
    _is_model,
    _wcorr_block_rows,
    _warn_user,
    autofc,
    eye_weights,
    format_data,
    gaussian_weights,
    isfc,
    null_combine,
    reduce,
    wisfc,
)
from .events import (
    distinct_event_sets,
    _check_params_keys,
    _has_features,
    _parse_spec,
    resolve_events,
    resolve_events_by_order,
)
from .summaries import (
    Communities,
    WeightedStrength,
    _block_rows_for,
    _from_name,
    _shift_null_threshold,
    _streamed_peak_elements,
    _streamed_summary,
    _Summary,
    _auto_emitter,
    _auto_fallback,
)
from .timecorr import _event_kernels, _restricted, _shaped_kernel, timecorr

_BYTES_PER_GB = 1e9

# The feeding kernel used when delta_weights_function is None (read by
# _feeding_kernel): "gaussian" (gaussian_weights with _DEFAULT_FEEDING_PARAMS
# unless delta_weights_params is given) or "delta" (eye_weights,
# co-fluctuations; delta_weights_params then raises ValueError, since
# eye_weights takes no parameters).
_DEFAULT_FEEDING = "gaussian"
_DEFAULT_FEEDING_PARAMS = {"var": 25}
# kernel standard deviation assumed by the shift null for kernels without a
# "var" parameter
_DEFAULT_BANDWIDTH = 5.0
# correlation functions whose steps can be streamed
_STREAMED_CFUNS = (autofc, isfc, wisfc)

# Block temporaries of the correlation step, in multiples of one block
# (``_wcorr_blocks`` keeps about this many block-sized arrays alive at once;
# measured with tracemalloc for isfc and autofc).
_WORKSPACE_BLOCKS = 6


def _step_gb(d, T, n_subjects=1, n_outputs=1, kernel_feeding=False):
    """
    Peak memory (GB) of one unreduced correlation step on ``n_subjects`` (T, d)
    inputs that holds ``n_outputs`` (T, d(d+1)/2) results per subject.
    ``kernel_feeding``: the feeding path (``n_outputs=2``) uses a kernel other
    than ``eye_weights``, so it has block temporaries of its own while both
    paths' results are held.
    """
    d, T, S = int(d), int(T), int(n_subjects)
    v = d * (d + 1) // 2
    block = min(d, _wcorr_block_rows(T, d)) * T * d
    elements = (
        d * d * T  # the (d, d, T) correlation array of one subject
        + S * T * v  # the analysis path's vectorized results, every subject
        + 3 * S * T * d  # the inputs (and the copies made while formatting)
        + 3 * T * T  # the kernel, its working copy and its absolute value
        + _workspace(_WORKSPACE_BLOCKS * block, (n_outputs - 1) * S * T * v,
                     kernel_feeding)
    )
    return elements * 8 / _BYTES_PER_GB


def _workspace(blocks, feeding, kernel_feeding):
    """
    Elements held besides the correlation array, inputs, kernels and the
    analysis results: block temporaries (``blocks``) and the feeding results
    (``feeding``).  With ``eye_weights`` the feeding path has no block
    temporaries and runs after the analysis path's are freed, so the peak
    holds the larger of the two; with another kernel the feeding path's own
    block temporaries are held together with its results.
    """
    return blocks + feeding if kernel_feeding else max(blocks, feeding)


def estimate_memory_gb(K, order, T, n_subjects=1, return_delta=False, streamed=False,
                       *, delta_weights_function=None):
    """
    Estimate the peak memory of each order of an unreduced or streamed pipeline.

    Without reduction the number of features grows from ``d`` to ``d(d+1)/2`` at
    every order, starting from ``K``.  The correlation step at order ``k``
    allocates one ``(d_{k-1}, d_{k-1}, T)`` array at a time, holds a
    ``(T, d_k)`` vectorized result for every subject (two per subject once
    the delta path has run too), and the analysis path (and the delta path,
    unless its kernel is ``eye_weights``) uses block temporaries of bounded
    size, all in float64.  The estimate is the peak of
    ``tc.features(..., rfun=None, delta_rfun=None)`` with the same
    ``delta_weights_function`` and the default ``combine`` (``null_combine``)
    for data without NaNs; other ``combine`` functions and reducers allocate
    more.  With per-subject events, the kernels of the
    distinct event sets (one T x T array each) are not included in the
    estimate.

    With ``streamed=True`` each entry is instead the smallest peak of one
    streamed step (one row of correlations per block; see ``features``) on
    ``n_subjects`` (T, K) inputs: the step's block of correlations, the
    estimator's temporaries, the kernel copies, the accumulated pair weights
    and the outputs, without the inputs themselves.  A streamed step's input
    width at orders above 1 is the width of the previous summary, which is
    not known in advance; every entry uses K.

    Parameters
    ----------
    K : int
        Number of features in the input data.
    order : int
        Highest order computed.
    T : int
        Number of timepoints.
    n_subjects : int, optional
        Number of subjects.  Default: 1.
    return_delta : bool, optional
        Whether the delta path of the last order is computed too (as
        ``features(..., return_delta=True)`` does).  Not used with
        ``streamed=True``.  Default: False.
    streamed : bool, optional
        If True, estimate streamed steps (see above).  Default: False.
    delta_weights_function : function or None, optional
        The feeding (delta) kernel of the run (keyword-only), as in
        ``features``: None (the default Gaussian kernel) or any kernel other
        than ``eye_weights`` counts the feeding path's block temporaries;
        ``eye_weights`` (the original pipeline's co-fluctuations) does not.
        Not used with ``streamed=True``.  Default: None.

    Returns
    -------
    list of float
        ``order + 1`` estimates in gigabytes (1e9 bytes), indexed by order; entry
        0 is 0.0 because order 0 is the input itself.

    Examples
    --------
    >>> from timecorr.higher_order import estimate_memory_gb
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, n_subjects=3)]
    [0.0, 0.163, 102.484]
    >>> import timecorr as tc
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, n_subjects=3,
    ...                                          delta_weights_function=tc.eye_weights)]
    [0.0, 0.139, 102.484]
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, 3, streamed=True)]
    [0.0, 0.014, 0.014]
    """
    K = int(K)
    T = int(T)
    if streamed:
        step = 8 * _streamed_peak_elements(T, K, n_subjects, 1) / _BYTES_PER_GB
        return [0.0] + [step] * int(order)
    kernel_feeding = delta_weights_function is not eye_weights
    estimates = [0.0]
    d = K
    for k in range(1, int(order) + 1):
        both_paths = k < order or return_delta
        estimates.append(
            _step_gb(d, T, n_subjects, 2 if both_paths else 1, kernel_feeding)
        )
        d = d * (d + 1) // 2
    return estimates


def _largest_k(order, T, max_memory_gb, n_subjects=1, return_delta=False,
               streamed=False, kernel_feeding=False):
    """Largest K whose estimates up to ``order`` all fit (0 if none)."""
    feeding = gaussian_weights if kernel_feeding else eye_weights

    def fits(k):
        est = estimate_memory_gb(k, order, T, n_subjects, return_delta, streamed,
                                 delta_weights_function=feeding)
        return max(est) <= max_memory_gb

    if not fits(1):
        return 0
    lo, hi = 1, 2
    while fits(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


# named in the MemoryError of a features step that holds its correlations
# although a summary reducer would stream it
def _stream_hint(order):
    """The MemoryError hint for a materialized analysis step at ``order``."""
    k = 1 if order is None else int(order)
    return (
        " A summary reducer streams the analysis step in blocks of rows: "
        "rfun=tc.WeightedStrength(signed=True), or a dict by order such as "
        f"rfun={{{k}: tc.WeightedStrength(signed=True)}} for order {k} only."
    )


def _check_step_memory(d, T, n_subjects, n_outputs, max_memory_gb, order=None,
                       kernel_feeding=False, held_elements=0, hint=""):
    """
    Raise MemoryError before an unreduced correlation step on ``n_subjects``
    (T, d) inputs whose estimate (``_step_gb``), plus ``held_elements``
    float64 values retained by the caller (earlier outputs and fits of a
    ``features`` run), exceeds ``max_memory_gb``.  ``max_memory_gb=None``
    checks nothing.  ``hint`` is appended to the message.  Shared by ``features``, the path functions and the
    decoders, so all refuse the same steps.
    """
    if max_memory_gb is None:
        return
    S = int(n_subjects)
    step = _step_gb(d, T, S, n_outputs, kernel_feeding)
    held = 8 * int(held_elements) / _BYTES_PER_GB
    if step + held > max_memory_gb:
        fit = _largest_k(1, T, max_memory_gb - held, S, n_outputs > 1,
                         kernel_feeding=kernel_feeding)
        where = "" if order is None else f" at order {order}"
        retained = (
            f", plus {held:.3g} GB of outputs and fits retained by the run"
            if held_elements else ""
        )
        raise MemoryError(
            f"The correlation step{where} needs about {step:.3g} GB "
            f"({d} input features, T={T}, {S} subject{'s' if S > 1 else ''})"
            f"{retained}, more than max_memory_gb={max_memory_gb}. The largest "
            f"number of input features that fits one step is K={fit}. Reduce "
            "to fewer features (ndims, rfun, delta_rfun) or raise max_memory_gb."
            + hint
        )


def _streamed_block_elements(d, T, n_subjects, max_memory_gb, order=None,
                             path="analysis"):
    """
    ``max_block_elements`` for a streamed step on ``n_subjects`` (T, d)
    inputs: ``b * d * T`` with b the largest number of rows per block whose
    peak (``summaries._streamed_peak_elements``) fits in ``max_memory_gb``
    (None: no limit; the step then uses the estimator's default block
    size, ``helpers._wcorr_block_rows``).  Raise MemoryError, before the step
    computes anything, when one row per block does not fit.  The only place
    that converts the byte budget of a streamed step to a block size.
    """
    if max_memory_gb is None:
        return None
    S = int(n_subjects)
    b = _block_rows_for(T, d, S, max_memory_gb * _BYTES_PER_GB)
    if b < 1:
        need = estimate_memory_gb(d, 1, T, S, streamed=True)[1]
        fit = _largest_k(1, T, max_memory_gb, S, streamed=True)
        where = "" if order is None else f" at order {order}"
        raise MemoryError(
            f"The streamed {path} step{where} needs at least about {need:.3g} GB "
            f"(one row of correlations per block; {d} input features, T={T}, "
            f"{S} subject{'s' if S > 1 else ''}), more than "
            f"max_memory_gb={max_memory_gb}. The largest number of input "
            f"features that fits one streamed step is K={fit}. Reduce to fewer "
            "features (rfun, delta_rfun) or raise max_memory_gb."
        )
    return b * int(d) * int(T)


def _budget_left(max_memory_gb, held_elements, d, T, n_subjects, order, what):
    """
    The budget (GB) left for a streamed computation on ``n_subjects`` (T, d)
    inputs while ``held_elements`` float64 values are held (None: no limit).
    Raise MemoryError, before anything is computed, when what is held plus one
    row of correlations per block exceeds ``max_memory_gb``.
    """
    if max_memory_gb is None:
        return None
    S = int(n_subjects)
    held = 8 * int(held_elements) / _BYTES_PER_GB
    need = 8 * _streamed_peak_elements(T, d, S, 1) / _BYTES_PER_GB
    if held + need > max_memory_gb:
        where = "" if order is None else f" at order {order}"
        raise MemoryError(
            f"The {what}{where} needs at least about {held + need:.3g} GB "
            f"({held:.3g} GB of results held while it runs, with retained fits "
            f"and event kernels, plus {need:.3g} GB "
            f"for one row of correlations per block; {d} input features, T={T}, "
            f"{S} subject{'s' if S > 1 else ''}), more than "
            f"max_memory_gb={max_memory_gb}. Reduce to fewer features (rfun, "
            "delta_rfun), give the number of communities n, or raise "
            "max_memory_gb."
        )
    return max_memory_gb - held


def _event_kernel_elements(T, n_subjects, events):
    """
    Elements of the kernels a step builds for RESOLVED ``events``: one T x T
    array per distinct event set when the subjects' events differ, 0 when
    one kernel is shared by every subject (as without events).
    """
    groups = distinct_event_sets(_restricted(events), int(n_subjects))
    return len(groups) * int(T) * int(T) if len(groups) > 1 else 0


def _fits_elements(*reducers):
    """
    Elements of the arrays kept in the ``fits_`` of the summary objects
    among ``reducers`` (pair weights and mean correlations, (K, K) per unit
    per fit); names, models, callables and None hold none.
    """
    total, seen = 0, set()
    for r in reducers:
        if not isinstance(r, _Summary) or id(r) in seen:
            continue
        seen.add(id(r))
        for f in r.fits_:
            for key in ("weights", "mean_r"):
                if f.get(key) is not None:
                    total += sum(int(np.asarray(a).size) for a in f[key])
    return total


def _output_elements(*outputs):
    """
    Elements of outputs (arrays, or lists of arrays).  An array that appears
    in several outputs is counted once; views of one base array are counted
    by their sizes, at most the size of the base (overlapping views are not
    counted twice beyond it; the part of a base that no view covers is not
    counted).
    """
    seen, by_root = set(), {}
    for out in outputs:
        for a in _as_list(out):
            a = np.asarray(a)
            if id(a) in seen:
                continue
            seen.add(id(a))
            root = a
            while isinstance(root.base, np.ndarray):
                root = root.base
            size, cap = by_root.get(id(root), (0, int(root.size)))
            by_root[id(root)] = (size + int(a.size), cap)
    return sum(min(size, cap) for size, cap in by_root.values())


def _needs_shift_null(rfun):
    """True for a reducer whose number of communities comes from the shift null."""
    summary = _as_summary(rfun)
    return summary is not None and summary._needs_communities and summary.n is None


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _reduce(x, rfun, ndims):
    """``reduce``, passing ``ndims`` only to model-based reducers (it sets their width)."""
    return reduce(x, rfun=rfun, ndims=ndims if _is_model(rfun) else None)


def _is_summary(rfun):
    """True for a summary object or a summary name (``helpers._SUMMARY_NAMES``)."""
    return isinstance(rfun, _Summary) or (
        isinstance(rfun, str) and rfun in _SUMMARY_NAMES
    )


def _as_summary(rfun):
    """The summary object of ``rfun`` (a fresh one for a name), or None."""
    if isinstance(rfun, _Summary):
        return rfun
    if isinstance(rfun, str) and rfun in _SUMMARY_NAMES:
        return _from_name(rfun)
    return None


def _apply_reducer(corrs, rfun, ndims, order=None, path=None, null_threshold=None,
                   append_fit=False):
    """
    ``_reduce``; a summary is fit (recording ``order`` and ``path`` in its
    ``fits_``) and then transforms the same correlations.  With
    ``append_fit`` False, ``fits_`` keeps only the new fit (as a direct call
    of the summary does); True appends it (the steps of ``features`` and
    ``two_path_step``).
    """
    summary = _as_summary(rfun)
    if summary is None:
        return _reduce(corrs, rfun, ndims)
    summary.fit(corrs, order=order, path=path, null_threshold=null_threshold)
    fit = summary.fits_[-1]
    if null_threshold is not None:
        # the threshold came from the shift null with the summary's n_null
        fit["n_null"] = summary.n_null
    K = fit["weights"][0].shape[0]
    out = _auto_emitter(summary, fit, K, order)._transform_fit(fit, corrs)
    if not append_fit:
        del summary.fits_[:-1]
    return out


def _streams(rfun, cfun, combine, groups=None):
    """Whether an analysis step is streamed (see the module docstring)."""
    return (
        groups is None
        and combine is null_combine
        and any(cfun is f for f in _STREAMED_CFUNS)
        and _is_summary(rfun)
    )


def _bandwidth(weights_function, weights_params):
    """
    Standard deviation of a kernel for the shift null: ``sqrt(var)`` when the
    parameters (or, for None, the function's default parameters) have a
    ``"var"``; otherwise ``_DEFAULT_BANDWIDTH``.
    """
    params = weights_params
    if params is None:
        try:
            default = inspect.signature(weights_function).parameters.get("params")
        except (TypeError, ValueError):
            default = None
        if default is not None and default.default is not inspect.Parameter.empty:
            params = default.default
    if isinstance(params, dict) and "var" in params:
        return float(np.sqrt(float(params["var"])))
    return _DEFAULT_BANDWIDTH


def _step_kernels(T, n_subjects, weights_function, weights_params,
                  include_timepoints="all", include_zero=True, events=None):
    """
    The kernel(s) ``timecorr`` builds for these arguments and RESOLVED events:
    one T x T kernel, or one per subject when the subjects' events differ.
    """
    weights = _shaped_kernel(
        T, weights_function, weights_params, include_timepoints, include_zero
    )
    resolved = resolve_events(events, T, n_subjects, None)
    return _event_kernels(weights, resolved, n_subjects)


def _run_streamed(subjects, summary, cfun, kernels, *, max_memory_gb, order, path,
                  auto_n, bandwidth, append_fit=False):
    """
    ``summaries._streamed_summary`` under a memory budget; issues the
    estimator's warning.  ``append_fit`` as in ``_apply_reducer``.
    """
    # C-ordered float copies: the blocked arithmetic's round-off then does not
    # depend on the memory layout of the inputs (reducers can return views)
    subjects = [np.ascontiguousarray(x, dtype=np.float64) for x in subjects]
    T, d = subjects[0].shape
    max_block = _streamed_block_elements(
        d, T, len(subjects), max_memory_gb, order, path
    )
    out, message = _streamed_summary(
        subjects,
        kernels,
        summary,
        cfun,
        max_block_elements=max_block,
        order=order,
        path=path,
        auto_n=auto_n,
        bandwidth=bandwidth,
    )
    if not append_fit:
        del summary.fits_[:-1]
    if message is not None:
        _warn_user(message)
    return out


def _analysis_path(
    subjects,
    weights_function=gaussian_weights,
    weights_params=None,
    cfun=isfc,
    rfun="PCA",
    ndims=None,
    combine=null_combine,
    include_timepoints="all",
    include_zero=True,
    groups=None,
    max_memory_gb=None,
    n_outputs=1,
    order=None,
    events=None,
    append_fit=False,
    kernel_feeding=False,
    held_elements=0,
    reducers=(),
    input_elements=0,
):
    """
    Analysis path of one order: ``reduce(combine(timecorr(subjects, kernel, cfun)))``.

    ``held_elements``: float64 values the caller retains (earlier outputs of
    a ``features`` run), counted against ``max_memory_gb``; ``reducers``: the
    reducer objects the caller holds, whose ``fits_`` arrays are counted too.
    ``input_elements``: the part of ``held_elements`` that is this step's
    input, left out of a materialized check, whose estimate (``_step_gb``)
    already counts the input.

    ``subjects`` is a list of (T, d) arrays.  With ``groups=None`` all subjects
    are analyzed together and the result is what ``combine`` returns, reduced.
    With ``groups`` (a sequence of sequences of subject indices), ``cfun`` and
    ``combine`` see only one group's subjects at a time, the combined results of
    all groups are reduced jointly (one model, one shared space), and a list with
    one entry per group is returned.

    Before computing, the step is checked against ``max_memory_gb`` (see
    ``_check_step_memory``; ``n_outputs=2`` when the delta path of the same
    order follows and its results are held together).

    ``events`` is a RESOLVED entry (``None``, one boundary array shared by all
    subjects, or a list with one boundary array or ``None`` per subject; see
    ``timecorr.events.resolve_events``) and restricts the analysis kernel.
    With ``groups``, a per-subject list is subset by member index, so each
    subject keeps its own events.

    Without ``groups``, a summary reducer with ``combine=null_combine`` and
    ``cfun`` ``autofc``, ``isfc`` or ``wisfc`` is streamed (checked against
    ``max_memory_gb`` with the streamed estimate); with ``autofc``, a
    ``Communities`` (or split ``WeightedStrength``) with ``n=None`` gets the
    number of communities from the shift null.  Summaries record ``order``
    and ``path="analysis"`` in their ``fits_`` (appended with
    ``append_fit``, otherwise the only entry; see ``_apply_reducer``).
    """
    T, d = subjects[0].shape
    if _streams(rfun, cfun, combine, groups):
        summary = _as_summary(rfun)
        # the distinct event kernels and earlier fits are held while the step
        # runs: checked before the kernels are built
        budget = _budget_left(
            max_memory_gb,
            held_elements + _event_kernel_elements(T, len(subjects), events)
            + _fits_elements(summary, *reducers),
            d, T, len(subjects), order, "streamed analysis step",
        )
        kernels = _step_kernels(
            T, len(subjects), weights_function, weights_params,
            include_timepoints, include_zero, events,
        )
        return _run_streamed(
            subjects,
            summary,
            cfun,
            kernels,
            max_memory_gb=budget,
            order=order,
            path="analysis",
            auto_n=cfun is autofc,
            bandwidth=_bandwidth(weights_function, weights_params),
            append_fit=append_fit,
        )
    _check_step_memory(d, T, len(subjects), n_outputs, max_memory_gb, order,
                       kernel_feeding,
                       held_elements - input_elements + _fits_elements(*reducers))

    def analyze(members, member_events):
        return combine(
            _as_list(
                timecorr(
                    members,
                    weights_function=weights_function,
                    weights_params=weights_params,
                    include_timepoints=include_timepoints,
                    combine=null_combine,
                    cfun=cfun,
                    rfun=None,
                    include_zero=include_zero,
                    events=member_events,
                )
            )
        )

    if groups is None:
        return _apply_reducer(
            analyze(subjects, events), rfun, ndims, order, "analysis",
            append_fit=append_fit,
        )

    def group_events(g):
        if isinstance(events, list):
            return [events[j] for j in g]
        return events

    combined = [analyze([subjects[j] for j in g], group_events(g)) for g in groups]
    # flatten every group's output into one list so a single reduction model is
    # fit to all of them, then restore each group's container type
    pieces, sizes = [], []
    for c in combined:
        if isinstance(c, list):
            pieces.extend(c)
            sizes.append(len(c))
        else:
            pieces.append(c)
            sizes.append(None)
    reduced = _as_list(
        _apply_reducer(pieces, rfun, ndims, order, "analysis", append_fit=append_fit)
    )
    out, start = [], 0
    for n in sizes:
        if n is None:
            out.append(reduced[start])
            start += 1
        else:
            out.append(list(reduced[start : start + n]))
            start += n
    return out


_NAN_FILL_MESSAGE = (
    "NaN samples: their co-fluctuations are set to 0 before the delta path's "
    "{kind} reduction ({rfun!r}), so NaN handling is approximate from order 2 on "
    "under that reducer (at order 1, NaN samples carry no weight).  Use "
    "delta_rfun=None or a callable to keep them out of later orders."
)


def _nan_pairs(x):
    """
    (T, d(d+1)/2) mask of the feature pairs with a NaN sample in ``x``, in
    ``mat2vec`` layout (the diagonal, then the upper triangle row by row).
    """
    d = x.shape[1]
    iu = np.triu_indices(d, 1)
    rows = np.concatenate([np.arange(d), iu[0]])
    cols = np.concatenate([np.arange(d), iu[1]])
    missing = np.isnan(x)
    return missing[:, rows] | missing[:, cols]


def _delta_path(
    subjects,
    rfun="PCA",
    ndims=None,
    max_memory_gb=None,
    n_outputs=1,
    order=None,
    nan_warned=None,
    weights_function=eye_weights,
    weights_params=None,
    include_timepoints="all",
    include_zero=True,
    events=None,
    held_elements=0,
    append_fit=False,
    retained_elements=0,
):
    """
    Feeding (delta) path of one order: per-subject correlations with the
    feeding kernel and ``autofc``, never combined, reduced jointly across
    subjects.  ``subjects`` is a list of (T, d) arrays; a list of the same
    length is returned.  The step is checked against ``max_memory_gb`` first
    (see ``_check_step_memory``, or ``_streamed_block_elements`` for a
    streamed step).  Summaries record ``order`` and ``path="feeding"`` in
    their ``fits_`` (appended with ``append_fit``, otherwise the only entry;
    see ``_apply_reducer``); a ``Communities`` (or split
    ``WeightedStrength``) with ``n=None`` gets the number of communities
    from the shift null.

    With ``weights_function=eye_weights`` (the default) the features are
    co-fluctuations; ``weights_params``, ``include_timepoints``,
    ``include_zero`` and ``events`` are not used.  A co-fluctuation
    involving a NaN sample is NaN (not 0), so the next order's correlations
    leave it out pairwise.  Unreduced, and for a callable reducer that is not
    a summary, the NaNs are kept.  Before a model-based reduction, a graph
    measure or a summary they are set to 0 with one ``UserWarning``;
    ``nan_warned`` (a one-element list shared across the steps of one call)
    keeps it to one warning per call.

    With another kernel, the kernel is built as ``timecorr`` builds it
    (``include_timepoints``, ``include_zero``, and the RESOLVED ``events``
    entry), NaN samples are left out pairwise by the estimator, and the
    estimator's warning is issued at most once.  A summary reducer is
    streamed.
    """
    T, d = subjects[0].shape
    S = len(subjects)
    summary = _as_summary(rfun)
    if weights_function is not eye_weights:
        if summary is not None:
            # what the caller holds (``held_elements``: results and retained
            # fits) and the distinct event kernels: checked before the
            # kernels are built
            budget = _budget_left(
                max_memory_gb,
                held_elements + _event_kernel_elements(T, S, events),
                d, T, S, order, "streamed feeding step",
            )
            kernels = _step_kernels(
                T, S, weights_function, weights_params, include_timepoints,
                include_zero, events,
            )
            return _run_streamed(
                subjects,
                summary,
                autofc,
                kernels,
                max_memory_gb=budget,
                order=order,
                path="feeding",
                auto_n=True,
                bandwidth=_bandwidth(weights_function, weights_params),
                append_fit=append_fit,
            )
        _check_step_memory(d, T, S, n_outputs, max_memory_gb, order, True,
                           retained_elements)
        corrs = _as_list(
            timecorr(
                subjects,
                weights_function=weights_function,
                weights_params=weights_params,
                include_timepoints=include_timepoints,
                include_zero=include_zero,
                combine=null_combine,
                cfun=autofc,
                rfun=None,
                events=events,
            )
        )
        return _as_list(
            _apply_reducer(corrs, rfun, ndims, order, "feeding", append_fit=append_fit)
        )

    _check_step_memory(d, T, S, n_outputs, max_memory_gb, order,
                       held_elements=retained_elements)
    null_budget = None
    if _needs_shift_null(rfun):
        # the shift null runs while the co-fluctuations (and whatever the
        # caller holds, ``held_elements``) are held: checked before computing
        null_budget = _budget_left(
            max_memory_gb, held_elements + S * T * (d * (d + 1) // 2), d, T, S,
            order, "shift null of the feeding step",
        )
    delta = _as_list(
        timecorr(
            subjects,
            weights_function=eye_weights,
            combine=null_combine,
            cfun=autofc,
            rfun=None,
        )
    )
    missing = [_nan_pairs(x) if np.isnan(x).any() else None for x in subjects]
    if any(m is not None for m in missing):
        delta = [
            c if m is None else np.where(m, np.nan, c) for c, m in zip(delta, missing)
        ]
        if rfun is not None and (summary is not None or not callable(rfun)):
            if nan_warned is None or not nan_warned[0]:
                if _is_model(rfun):
                    kind = "model-based"
                elif summary is not None:
                    kind = "summary"
                else:
                    kind = "graph-measure"
                _warn_user(_NAN_FILL_MESSAGE.format(kind=kind, rfun=rfun))
                if nan_warned is not None:
                    nan_warned[0] = True
            delta = [np.nan_to_num(c, nan=0.0) for c in delta]
    null_threshold = None
    if summary is not None and summary._needs_communities and summary.n is None:
        # the number of communities from the shift null of the delta kernel
        summary._check_fit(d, 0.0)
        null_threshold = _shift_null_threshold(
            subjects,
            eye_weights(T),
            n_null=summary.n_null,
            bandwidth=_DEFAULT_BANDWIDTH,
            max_block_elements=_streamed_block_elements(
                d, T, S, null_budget, order, "feeding"
            ),
        )
    return _as_list(
        _apply_reducer(
            delta, rfun, ndims, order, "feeding", null_threshold, append_fit=append_fit
        )
    )


# ----------------------------------------------------------------------------
# reducers by order
# ----------------------------------------------------------------------------


def _is_auto(value):
    return isinstance(value, str) and value == "auto"


def _auto_at(rfun, k):
    """Whether order ``k``'s analysis reducer comes from ``"auto"`` (the
    caller did not choose it): ``rfun`` is ``"auto"``, or an order map that
    leaves order ``k`` out or maps it to ``"auto"``.  Called after
    ``_resolve_reducers`` has checked ``rfun``."""
    if isinstance(rfun, dict) and "model" not in rfun:
        return _is_auto(rfun.get(k, "auto"))
    return _is_auto(rfun)


def _check_reducer(value, name):
    """Raise for a value that is not "auto" or a reducer ``reduce`` accepts."""
    if _is_auto(value):
        return
    if isinstance(value, dict) and "model" not in value:
        raise ValueError(
            f"{name} must be a reducer; a dict here needs a 'model' key (order "
            f"maps cannot be nested); got keys {sorted(map(repr, value))}."
        )
    _check_rfun(value, name)


def _order_map(value, name, max_order):
    """
    ``{order: value}`` for an order map (a dict without a ``'model'`` key whose
    keys are all integers in ``1..max_order``; ``max_order=None``: any
    positive integer), after checking its values; None for any other value.
    """
    if not isinstance(value, dict) or "model" in value:
        return None
    keys = list(value)
    integral = all(
        isinstance(k, numbers.Integral) and not isinstance(k, bool) for k in keys
    )
    if not keys or not integral:
        raise ValueError(
            f"A dict {name} is either a model configuration with a 'model' key "
            "(for example {'model': 'PCA', 'kwargs': {...}}) or an order map "
            "whose keys are integer orders (for example {1: 'PCA', 2: "
            f"'weighted_strength'}}); got keys {sorted(map(repr, keys))}."
        )
    span = "positive integers" if max_order is None else f"integers in 1..{max_order}"
    bad = [k for k in keys if k < 1 or (max_order is not None and k > max_order)]
    if bad:
        raise ValueError(
            f"The keys of the order map {name} must be {span}; got "
            f"{sorted(int(k) for k in bad)}."
        )
    for k, v in value.items():
        _check_reducer(v, f"{name}[{int(k)}]")
    return {int(k): v for k, v in value.items()}


def _resolve_auto(value, k, path):
    """``"auto"``: PCA at order 1 and ``WeightedStrength(signed=True)`` above
    on the analysis path, ``Communities()`` on the feeding path (marked with
    ``_from_auto`` so that ``_check_auto_feeding`` can name the
    alternatives)."""
    if not _is_auto(value):
        return value
    if path == "feeding":
        summary = Communities()
        summary._from_auto = True
        return summary
    return "PCA" if k == 1 else WeightedStrength(signed=True)


# smallest input of the default feeding reducer (communities with the
# automatic count): K >= 4 features, T >= 40 timepoints for the shift null
_AUTO_FEEDING_MIN_K = 4
_AUTO_FEEDING_MIN_T = 40


def _check_auto_feeding(delta_rfun, d, T, order, previous=None):
    """
    Raise ``ValueError`` naming the alternatives when the feeding reducer
    resolved from ``delta_rfun="auto"`` (communities) cannot be fitted to
    (T, d) inputs: fewer than 4 features or fewer than 40 timepoints (the
    shift null then has at least one lag in ``[L, T - L]``, and lags are
    reused when d exceeds their number; see ``summaries._shift_lags``).
    ``previous``: the number of communities found at the
    previous order when the d inputs are the community columns of an
    explicit ``Communities`` there (the automatic reducer keeps at least 4
    columns; see ``summaries._auto_emitter``).
    """
    if not getattr(delta_rfun, "_from_auto", False):
        return
    at = "" if order is None else f" at order {order}"
    alternatives = (
        'Pass delta_rfun="weighted_strength" (any number of features and '
        "timepoints), or the original pipeline's settings "
        'rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights.'
    )
    if d < _AUTO_FEEDING_MIN_K or T < _AUTO_FEEDING_MIN_T:
        source = ""
        if previous is not None and order is not None:
            source = (
                f" (the community columns of order {order - 1}; communities "
                f"found at order {order - 1}: {previous}, at most half the "
                "features of that order)"
            )
        raise ValueError(
            f'The default feeding reducer (delta_rfun="auto": communities, counted '
            f"with a shift null){at} needs at least {_AUTO_FEEDING_MIN_K} features "
            f"and {_AUTO_FEEDING_MIN_T} timepoints; got {d} feature"
            f"{'' if d == 1 else 's'}{source} and T = {T}. " + alternatives
        )


def _resolve_reducers(rfun, delta_rfun, orders, max_order):
    """
    Check ``rfun`` and ``delta_rfun`` and resolve them for each order in
    ``orders``: ``(analysis, feeding)``, two dicts keyed by order.  Orders
    missing from an order map use ``"auto"``; ``"auto"`` resolves as in
    ``_resolve_auto`` (a new ``WeightedStrength`` per order and call); a
    ``delta_rfun`` (or order map value) of None takes the analysis reducer of
    that order.
    """
    maps = {}
    for name, value in (("rfun", rfun), ("delta_rfun", delta_rfun)):
        maps[name] = _order_map(value, name, max_order)
        if maps[name] is None:
            _check_reducer(value, name)
    analysis, feeding = {}, {}
    for k in orders:
        a = rfun if maps["rfun"] is None else maps["rfun"].get(k, "auto")
        d = maps["delta_rfun"]
        f = delta_rfun if d is None else d.get(k, "auto")
        analysis[k] = _resolve_auto(a, k, "analysis")
        feeding[k] = analysis[k] if f is None else _resolve_auto(f, k, "feeding")
    return analysis, feeding


def _summary_instances(*values):
    """The summary objects among ``values`` and the values of order maps."""
    found = []
    for value in values:
        is_map = isinstance(value, dict) and "model" not in value
        items = value.values() if is_map else [value]
        found.extend(v for v in items if isinstance(v, _Summary))
    return found


def _check_ndims_steps(ndims, steps):
    """
    Raise when ``ndims`` is given and a computed step's reducer is a summary;
    ``steps`` holds ``(reducer, name, order)``.
    """
    if ndims is None:
        return
    for rfun, name, k in steps:
        if _is_summary(rfun):
            raise ValueError(
                f"ndims={ndims} applies to model-based reduction only; {name} at "
                f"order {k} is the summary {rfun!r}, which defines its own output "
                "width.  Leave ndims as None."
            )


def _known_width(rfun, d, ndims):
    """
    Number of columns ``rfun`` returns for (T, d(d+1)/2) correlations when it
    is known in advance: d(d+1)/2 unreduced, ``ndims`` (d when None) for a
    model, d for a graph measure, d times 2 with ``split`` times 2 with
    ``signed`` for weighted strength; None for ``Communities`` (the number
    of communities kept is found by the fit) and for a callable.
    """
    if rfun is None:
        return d * (d + 1) // 2
    summary = _as_summary(rfun)
    if isinstance(summary, WeightedStrength):
        return summary._n_columns(None, d)
    if _is_model(rfun):
        return d if ndims is None else int(ndims)
    if isinstance(rfun, str) and not _is_summary(rfun):
        return d
    return None


def _check_streamed_steps(K, T, S, max_memory_gb, analysis, feeding, fed, *, cfun,
                          combine, feeding_function, ndims):
    """
    Raise MemoryError, before anything is computed, for a streamed step of
    ``features`` that does not fit ``max_memory_gb`` with one row per block.
    Input widths follow ``_known_width`` from K.  After ``Communities``,
    whose output width is found by the fit, the width is unknown until it
    runs, and the later steps whose width follows from it are not checked
    here.  After a callable that is not a summary, the original K (the
    number of features of the data, not the callable's input width) is
    used (as ``estimate_memory_gb(..., streamed=True)`` uses K).  Every step is checked
    again, with its actual width, when it runs.
    """
    d = int(K)
    for k in sorted(analysis):
        if d is not None:
            if _streams(analysis[k], cfun, combine):
                _streamed_block_elements(d, T, S, max_memory_gb, k, "analysis")
            if (k in fed and feeding_function is not eye_weights
                    and _is_summary(feeding[k])):
                _streamed_block_elements(d, T, S, max_memory_gb, k, "feeding")
            width = _known_width(feeding[k], d, ndims)
            if width is None and not _is_summary(feeding[k]):
                width = int(K)  # a callable: the original K, as estimate_memory_gb uses
            d = width


_PARAMS_MAP_EXAMPLE = "{1: {'var': 100}, 2: {'var': 400}}"


def _is_bool(key):
    """True for a Python or NumPy boolean."""
    return isinstance(key, (bool, np.bool_))


def _has_integer_keys(value):
    """True for a dict with at least one integer key (Python and NumPy
    booleans included)."""
    return isinstance(value, dict) and any(
        isinstance(k, numbers.Integral) or _is_bool(k) for k in value
    )


def _params_by_order(value, name, max_order):
    """
    ``{order: params}`` for an order map of kernel parameters (a non-empty
    dict whose keys are all integers in ``1..max_order``; booleans are not
    orders), after checking its values (a dict of kernel parameters or
    None); None for any other value (kernel parameters for every order: a
    dict without integer keys, an empty dict, None).  A dict that mixes
    integer and other keys, or has boolean keys (Python or NumPy), raises
    ``ValueError``.  Only the outer dict is examined: each value is passed
    to the kernel unchanged, so a custom kernel's parameters may themselves
    have integer keys.
    """
    if not _has_integer_keys(value):
        return None
    keys = list(value)
    if any(_is_bool(k) for k in keys):
        raise ValueError(
            f"The keys of an order map {name} are integer orders in "
            f"1..{max_order}; booleans are not accepted as orders; got keys "
            f"{sorted(map(repr, keys))}."
        )
    if not all(isinstance(k, numbers.Integral) for k in keys):
        raise ValueError(
            f"A dict {name} is either kernel parameters (keys that are not "
            "integers, for example {'var': 100}, used at every order) or an "
            "order map: a dict whose keys are all integer orders in "
            f"1..{max_order} and whose values are kernel parameters or None "
            f"(for example {_PARAMS_MAP_EXAMPLE}); got keys "
            f"{sorted(map(repr, keys))}."
        )
    bad = [k for k in keys if k < 1 or k > max_order]
    if bad:
        raise ValueError(
            f"The keys of the order map {name} must be integers in "
            f"1..{max_order}; got {sorted(int(k) for k in bad)}."
        )
    for k, v in value.items():
        if v is not None and not isinstance(v, dict):
            raise ValueError(
                f"{name}[{int(k)}] must be a dict of kernel parameters or None "
                f"(the default of order {int(k)}); got {v!r}."
            )
    return {int(k): v for k, v in value.items()}


def _refuse_params_map(value, name):
    """Raise ``ValueError`` for a dict with integer keys (Python or NumPy
    booleans included) given to ``two_path_step``, which computes one order;
    the message names the dict's actual problem (an order map, boolean keys,
    or integer keys mixed with other keys)."""
    if not _has_integer_keys(value):
        return
    keys = sorted(map(repr, value))
    start = (
        f"two_path_step computes one order, so {name} holds the kernel "
        "parameters of that order"
    )
    if any(_is_bool(k) for k in value):
        raise ValueError(
            f"{start}; got a dict with boolean keys ({keys}), which is "
            "neither kernel parameters nor an order map (booleans are not "
            "accepted as integer keys or orders)."
        )
    if not all(isinstance(k, numbers.Integral) for k in value):
        raise ValueError(
            f"{start}; got a dict that mixes integer and other keys ({keys}), "
            "which is neither kernel parameters nor an order map (a dict "
            "whose keys are all integer orders)."
        )
    raise ValueError(
        f"{start}; got an order map (a dict keyed by integer orders, keys "
        f"{keys}).  Order maps of kernel parameters are accepted by features, "
        f"which computes every order: tc.features(..., "
        f"{name}={_PARAMS_MAP_EXAMPLE})."
    )


def _feeding_kernel(weights_function, weights_params, default=None):
    """
    ``(function, params)`` of the feeding kernel: None selects the kernel
    named by ``default`` (None: ``_DEFAULT_FEEDING``): ``"gaussian"`` gives
    ``gaussian_weights`` with ``_DEFAULT_FEEDING_PARAMS`` unless params are
    given, ``"delta"`` gives ``eye_weights``.  ``eye_weights`` (given or
    selected) with params raises ``ValueError``.
    """
    default = _DEFAULT_FEEDING if default is None else default
    if weights_function is None:
        if default not in ("delta", "gaussian"):
            raise ValueError(
                f"default must be 'delta' or 'gaussian'; got {default!r}."
            )
        if default == "delta":
            weights_function = eye_weights
        elif weights_params is None:
            return gaussian_weights, dict(_DEFAULT_FEEDING_PARAMS)
        elif not isinstance(weights_params, dict) or "var" not in weights_params:
            raise ValueError(
                'delta_weights_params without delta_weights_function: the default '
                'Gaussian feeding kernel needs "var" (its variance in squared '
                f"samples, for example {{'var': 25}}); got {weights_params!r}. "
                "Pass delta_weights_function to use another kernel's parameters."
            )
        else:
            return gaussian_weights, weights_params
    if not callable(weights_function):
        raise TypeError(
            "delta_weights_function must be None or a kernel function called as "
            f"f(T, params); got {type(weights_function).__name__}."
        )
    if weights_function is eye_weights and weights_params is not None:
        raise ValueError(
            "delta_weights_params is not used with delta_weights_function="
            "eye_weights: eye_weights takes no parameters; leave "
            "delta_weights_params as None."
        )
    return weights_function, weights_params


def two_path_step(
    data,
    weights_function=gaussian_weights,
    weights_params=None,
    cfun=isfc,
    rfun="auto",
    delta_rfun="auto",
    ndims=None,
    combine=null_combine,
    include_timepoints="all",
    include_zero=True,
    max_memory_gb=4.0,
    *,
    events=None,
    events_params=None,
    delta_weights_function=None,
    delta_weights_params=None,
    order=1,
):
    """
    Compute one order of the two-path pipeline from the previous feeding features.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
        The previous order's feeding features ``D_{k-1}`` (the raw data for
        ``k = 1``), one timepoints-by-features matrix per subject.
    weights_function : function
        Analysis kernel, called as ``weights_function(T, weights_params)``; row
        ``t`` of the kernel (``weights[t, :]``) is the kernel used at timepoint
        ``t``.  Default: ``gaussian_weights``.
    weights_params : dict or None
        Parameters of the analysis kernel.  A dict with integer keys
        raises ``ValueError`` (an order map, a dict mixing integer and
        other keys, or a dict with Python or NumPy boolean keys; the message
        says which): order maps of kernel parameters are accepted by
        ``features``.  Default: None (the kernel's
        defaults, for example ``gaussian_params``).
    cfun : function
        Analysis-path correlation function (``isfc``, ``wisfc``, ``autofc``).
        Default: ``isfc``.
    rfun : None, str, dict, or callable
        Reducer for the analysis path: any reducer ``helpers.reduce`` accepts
        (including the summaries ``WeightedStrength`` and ``Communities``),
        ``"auto"``, or an order map (see ``features``); ``"auto"`` and order
        maps are resolved with ``order`` (``"auto"``: ``"PCA"`` at order 1,
        ``WeightedStrength(signed=True)`` at orders 2 and above).  Default:
        "auto".
    delta_rfun : None, str, dict, or callable
        Reducer for the feeding (delta) path, as ``rfun``; ``"auto"`` is
        ``Communities()`` (at least 4 features and 40 timepoints; when its
        communities would emit fewer than 4 columns, the step emits
        unsigned weighted strength of the same correlations instead, with a
        ``UserWarning`` only when the shift null found no structure; see
        ``features``), None is the same as ``rfun`` at
        this order.  Default: "auto".
    ndims : int, optional
        Output width of model-based reduction (a model name or dict) on
        either path (see ``helpers.reduce``); it is not passed to a graph
        measure or a callable, which set their own width.  Given together
        with a summary on either path it raises ``ValueError``.  Default
        (None): ``K`` columns for ``K(K+1)/2`` inputs.
    combine : function
        Applied to the analysis path only (see ``timecorr``).  Default:
        ``null_combine``.
    include_timepoints : str
        "all", "backward"/"pre", or "forward"/"post" (see ``timecorr``).
        Shapes the analysis kernel and, unless it is ``eye_weights``, the
        feeding kernel.  Default: "all".
    include_zero : bool
        Whether the estimated timepoint contributes to its own estimate.
        Shapes the analysis kernel and, unless it is ``eye_weights``, the
        feeding kernel.  Default: True.
    max_memory_gb : float or None
        Upper limit, in gigabytes, on the estimated peak memory of each path
        of the step for all subjects together (see ``features`` and
        ``estimate_memory_gb``); both paths are checked before either
        computes anything.  With per-subject events, the kernels of the
        distinct event sets (one T x T array each) are not included in the
        estimate of a materialized step; a streamed step counts them, the
        analysis results held while the feeding path runs, and the (K, K)
        arrays kept in summaries' ``fits_``, before its kernels are built.
        None checks nothing.  Default: 4.0.
    events : None, array-like, or list
        Restricts the analysis kernel of this order, and the feeding kernel
        unless it is ``eye_weights``, to events (keyword-only; see
        ``timecorr``): boundaries shared by all subjects, features with T
        rows segmented with ``segment``, or a list with one such entry (or
        None) per subject.  A dict keyed by order is not accepted here (see
        ``features``).  Default: None (no restriction).
    events_params : dict or None
        Keyword arguments for ``segment`` when ``events`` holds features
        (keyword-only; see ``timecorr``).  Default: None.
    delta_weights_function : function or None
        Feeding kernel (keyword-only; see ``features``).  Default: None
        (``gaussian_weights``).
    delta_weights_params : dict or None
        Parameters of the feeding kernel (keyword-only).  With
        ``delta_weights_function=None`` they are the parameters of
        ``gaussian_weights`` (None: ``{"var": 25}``; a dict without
        ``"var"`` raises ``ValueError``); with a given function,
        None means that function's own defaults; with ``eye_weights``, which
        takes no parameters, they must be None (``ValueError`` otherwise).
        A dict with integer keys raises ``ValueError``, as for
        ``weights_params``.  Default: None.
    order : int
        The order this step computes (keyword-only), used to resolve
        ``"auto"`` and order maps (whose keys may be any positive integers
        here) and recorded in summaries' ``fits_``.  Default: 1.

    Returns
    -------
    X_next : numpy.ndarray or list of numpy.ndarray
        Analysis features: ``reduce(combine(timecorr(data, analysis kernel)))``.
        One array per subject for list input unless ``combine`` returns a
        single array; a single array for array input.
    D_next : numpy.ndarray or list of numpy.ndarray
        Feeding features: per-subject correlations with the feeding kernel
        and ``autofc``, never combined across subjects, reduced jointly.
        With ``eye_weights`` they are co-fluctuations; a co-fluctuation
        involving a NaN sample is NaN (kept when the reducer is None or a
        callable other than a summary; set to 0, with a ``UserWarning``,
        before a model-based, graph-measure or summary reduction).  With
        another kernel, NaN samples are left out pairwise and the features
        are finite.  A list for list input, an array for array input.

    Warns
    -----
    UserWarning
        From the correlation estimator (see ``wcorr``), at most one per path,
        naming each that applies: timepoints whose kernel row has fewer than 3
        non-zero kernel weights (for example the first timepoints of a
        backward kernel; their correlations are undefined and set to 0);
        other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there); timepoints where, because of NaN
        samples, some pairs of features have fewer than 3 observed samples with
        non-zero kernel weight (those correlations are undefined and set to 0);
        and timepoints with zero kernel-weighted variance in every feature
        (their correlations are set to 0).  Never for a feeding path with
        ``eye_weights``.
    UserWarning
        From the summaries (see ``WeightedStrength`` and ``Communities``).
    UserWarning
        When the default feeding reducer found no eigenvalue above the shift
        null and the step emits weighted strength instead (see
        ``features``).

    Raises
    ------
    MemoryError
        Before the step allocates its arrays, if the estimate of either path
        exceeds ``max_memory_gb``.
    ValueError
        If ``order`` is not a positive integer; if ``weights_params`` or
        ``delta_weights_params`` is a dict with integer keys (an order map,
        mixed keys or boolean keys; see ``features``); if ``rfun`` or ``delta_rfun``
        is an unknown model, graph measure or summary name, a dict that is
        neither a model configuration (with a ``'model'`` key) nor an order
        map, or an order map with a key that is not a positive integer or a
        value that is not a reducer; if ``ndims`` is not a positive integer,
        or is given with a summary on either path (with the defaults, at
        every order: the feeding reducer is ``Communities()`` at order 1 and
        the analysis reducer a summary from order 2); if the default
        feeding reducer
        (``delta_rfun="auto"``) gets fewer than 4 features or fewer than 40
        timepoints (the message names ``delta_rfun="weighted_strength"`` and
        the original pipeline's settings); for a ``Communities`` (or split
        ``WeightedStrength``) without ``n`` where the number of
        communities cannot be counted automatically (analysis path with a
        ``cfun`` other than ``autofc`` or a ``combine`` other than
        ``null_combine``; T < 40); and as in ``timecorr`` (subjects of
        different shapes, non-numeric columns, an unknown
        ``include_timepoints``, an all-zero kernel, a malformed ``events``
        specification, a dict ``events``, or events that leave no kernel row
        with 3 or more non-zero weights).  The events are resolved (and
        features segmented) before any correlations are computed.
    RuntimeError, ImportError
        As raised by ``segment`` for features entries of ``events``.
    TypeError
        If ``rfun`` or ``delta_rfun`` is not None, a string, a dict or a
        callable, or ``delta_weights_function`` is not None or callable.

    Examples
    --------
    >>> import timecorr as tc
    >>> from timecorr.higher_order import two_path_step
    >>> data, truth = tc.simulate_nested(S=2, T=300, n_supergroups=2,
    ...                                  modules_per_supergroup=3,
    ...                                  nodes_per_module=4, set_random_seed=0)

    With the defaults, order 1 gives PCA-reduced correlations and one feeding
    column per community (6 modules of 4 nodes here):

    >>> X1, D1 = two_path_step(data, cfun=tc.autofc, weights_params={"var": 100})
    >>> len(X1), X1[0].shape, len(D1), D1[0].shape
    (2, (300, 24), 2, (300, 6))

    At order 2 the default analysis reducer is signed weighted strength (two
    columns per input feature):

    >>> X2, D2 = two_path_step(D1, cfun=tc.autofc, weights_params={"var": 100},
    ...                        delta_rfun="weighted_strength", order=2)
    >>> X2[0].shape, D2[0].shape
    ((300, 12), (300, 6))

    The original pipeline's settings (co-fluctuations on the feeding path,
    PCA on both paths), with the analysis kernel restricted to two events
    (timepoints 0-149 and 150-299):

    >>> X1, D1 = two_path_step(data, weights_params={"var": 100}, events=[150],
    ...                        rfun="PCA", delta_rfun=None,
    ...                        delta_weights_function=tc.eye_weights)
    >>> X1[0].shape, D1[0].shape
    ((300, 24), (300, 24))
    """
    if (
        not isinstance(order, numbers.Integral)
        or isinstance(order, bool)
        or order < 1
    ):
        raise ValueError(f"order must be a positive integer; got {order!r}.")
    order = int(order)
    _refuse_params_map(weights_params, "weights_params")
    _refuse_params_map(delta_weights_params, "delta_weights_params")
    analysis, feeding = _resolve_reducers(rfun, delta_rfun, [order], None)
    ndims = _check_ndims(ndims)
    _check_ndims_steps(
        ndims, [(analysis[order], "rfun", order), (feeding[order], "delta_rfun", order)]
    )
    feeding_function, feeding_params = _feeding_kernel(
        delta_weights_function, delta_weights_params
    )
    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    resolved = resolve_events(
        events, subjects[0].shape[0], len(subjects), events_params
    )
    X_next, D_next = _two_path_step(
        subjects,
        weights_function=weights_function,
        weights_params=weights_params,
        cfun=cfun,
        rfun=analysis[order],
        delta_rfun=feeding[order],
        auto_analysis=_auto_at(rfun, order),
        feeding_function=feeding_function,
        feeding_params=feeding_params,
        order=order,
        ndims=ndims,
        combine=combine,
        include_timepoints=include_timepoints,
        include_zero=include_zero,
        max_memory_gb=max_memory_gb,
        events=resolved,
    )
    if single:
        if isinstance(X_next, list):
            X_next = X_next[0]
        D_next = D_next[0]
    return X_next, D_next


def _two_path_step(
    subjects,
    delta_rfun=None,
    max_memory_gb=None,
    order=None,
    nan_warned=None,
    *,
    feed=True,
    feeding_function=eye_weights,
    feeding_params=None,
    retained_elements=0,
    reducers=(),
    previous_communities=None,
    input_elements=0,
    auto_analysis=False,
    **analysis
):
    """
    One order on formatted subjects with RESOLVED reducers; returns lists
    ``(X_next, D_next)`` (``D_next`` is None with ``feed=False``, which skips
    the feeding path).  ``delta_rfun=None`` takes ``analysis["rfun"]``.

    Both paths are checked against ``max_memory_gb`` before either computes
    anything; when both are materialized each check counts the other path's
    unreduced results (``n_outputs=2``), as the paths hold them together.
    ``retained_elements`` (float64 values of the outputs a ``features`` run
    keeps from earlier orders) and the ``fits_`` arrays of ``reducers`` (the
    reducer objects the run holds), with this step's own reducers, are
    counted against ``max_memory_gb`` by every check of both paths.
    ``input_elements``: the part of ``retained_elements`` that is this
    step's input (the previous order's feeding features); a materialized
    check leaves it out, as its estimate (``_step_gb``) counts the input,
    while a streamed check, whose estimate excludes the inputs, keeps it.
    ``auto_analysis``: the analysis reducer comes from ``rfun="auto"``; only
    then may a MemoryError of the analysis step carry the streamed-reducer
    hint.
    """
    if delta_rfun is None:
        delta_rfun = analysis.get("rfun", "PCA")
    T, d = subjects[0].shape
    if feed:
        _check_auto_feeding(delta_rfun, d, T, order, previous_communities)
    analysis_streamed = _streams(
        analysis.get("rfun", "PCA"), analysis.get("cfun", isfc),
        analysis.get("combine", null_combine),
    )
    feeding_streamed = feeding_function is not eye_weights and _is_summary(delta_rfun)
    n_outputs = 2 if feed and not (analysis_streamed or feeding_streamed) else 1
    kernel_feeding = feed and feeding_function is not eye_weights
    S = len(subjects)
    v = d * (d + 1) // 2
    analysis_rfun = analysis.get("rfun", "PCA")
    events = analysis.get("events")
    kernel_elements = _event_kernel_elements(T, S, events)
    reducers = tuple(reducers)
    retained = int(retained_elements)
    # materialized checks: the step's input is already in _step_gb
    retained_m = retained - int(input_elements)
    # the analysis results are held while the feeding path runs: before
    # anything is computed, their size where it is known in advance (an
    # unreduced, model, graph-measure or weighted-strength output; 0
    # otherwise), and their actual size once computed
    width = _known_width(analysis_rfun, d, analysis.get("ndims"))
    held = 0 if width is None else S * T * width
    fits = _fits_elements(analysis_rfun, delta_rfun, *reducers)
    if max_memory_gb is not None:
        if analysis_streamed:
            _streamed_block_elements(
                d, T, S,
                _budget_left(max_memory_gb, retained + kernel_elements + fits,
                             d, T, S, order, "streamed analysis step"),
                order, "analysis",
            )
        else:
            # the hint is given only for an analysis reducer from "auto" (an
            # explicitly given rfun, the original pipeline's included, keeps
            # the message without it) and only when following it (a summary
            # as this order's analysis reducer) would stream this step and
            # leave the feeding step able to run: absent, streamed, or
            # materialized but fitting alone; not when the feeding reducer
            # is inherited from rfun (the hint would change it too)
            streamable = auto_analysis and _streams(
                "weighted_strength", analysis.get("cfun", isfc),
                analysis.get("combine", null_combine),
            )
            if streamable and feed:
                if delta_rfun is analysis_rfun:
                    streamable = False
                elif not feeding_streamed:
                    alone = _step_gb(d, T, S, 1, kernel_feeding)
                    held_gb = 8 * (retained_m + fits) / _BYTES_PER_GB
                    streamable = alone + held_gb <= max_memory_gb
            _check_step_memory(d, T, S, n_outputs, max_memory_gb, order,
                               kernel_feeding, retained_m + fits,
                               _stream_hint(order) if streamable else "")
        if feed and feeding_streamed:
            _streamed_block_elements(
                d, T, S,
                _budget_left(max_memory_gb, retained + held + kernel_elements + fits,
                             d, T, S, order, "streamed feeding step"),
                order, "feeding",
            )
        elif feed:
            _check_step_memory(d, T, S, n_outputs, max_memory_gb, order,
                               kernel_feeding, retained_m + fits)
            if _needs_shift_null(delta_rfun):
                _budget_left(
                    max_memory_gb, retained + held + fits + S * T * v, d, T, S, order,
                    "shift null of the feeding step",
                )
    X_next = _analysis_path(
        subjects, max_memory_gb=max_memory_gb, n_outputs=n_outputs, order=order,
        append_fit=True, kernel_feeding=kernel_feeding, held_elements=retained,
        reducers=(delta_rfun,) + reducers, input_elements=int(input_elements),
        **analysis
    )
    if not feed:
        return X_next, None
    fits = _fits_elements(analysis_rfun, delta_rfun, *reducers)
    D_next = _delta_path(
        subjects,
        rfun=delta_rfun,
        ndims=analysis.get("ndims"),
        max_memory_gb=max_memory_gb,
        n_outputs=n_outputs,
        order=order,
        nan_warned=nan_warned,
        weights_function=feeding_function,
        weights_params=feeding_params,
        include_timepoints=analysis.get("include_timepoints", "all"),
        include_zero=analysis.get("include_zero", True),
        events=analysis.get("events"),
        held_elements=retained + _output_elements(X_next) + fits,
        append_fit=True,
        retained_elements=retained_m + fits,
    )
    return X_next, D_next


def _requested(value, k):
    """The reducer given for order ``k`` (an order map's value, or ``"auto"``)."""
    if isinstance(value, dict) and "model" not in value:
        return value.get(k, "auto")
    return value


def _describe(reducer):
    """A reducer as a string (None stays None), without arrays."""
    if reducer is None or isinstance(reducer, str):
        return reducer
    if isinstance(reducer, _Summary):
        return repr(reducer)
    if isinstance(reducer, dict):
        return repr({key: reducer[key] for key in sorted(reducer, key=str)})
    return getattr(reducer, "__name__", None) or repr(reducer)


def _width(out):
    """Number of columns of an output (an array or a list of arrays)."""
    return int(np.asarray(_as_list(out)[0]).shape[1])


def _step_info(k, inputs, x, delta, analysis_rfun, feeding_rfun, requested,
               requested_delta):
    """
    The ``return_info`` entries of order ``k`` of ``features``: one dict for
    the analysis path and, when it was computed, one for the feeding path.
    """
    d = _width(inputs)
    entries = [{
        "order": k, "path": "analysis", "requested": _describe(requested),
        "effective": _describe(analysis_rfun), "input_width": d,
        "output_width": _width(x),
    }]
    if delta is None:
        return entries
    entry = {
        "order": k, "path": "feeding", "requested": _describe(requested_delta),
        "effective": _describe(feeding_rfun), "input_width": d,
        "output_width": _width(delta),
    }
    if getattr(feeding_rfun, "_from_auto", False):
        fit = feeding_rfun.fits_[-1]
        if fit["fallback"] is not None:
            entry["effective"] = _describe(_auto_fallback())
        entry.update(
            n_communities=int(fit["n_columns"]),
            n_exceed=int(fit["n_exceed"]),
            null_threshold=float(fit["null_threshold"]),
            n_null=int(fit["n_null"]),
            fallback=fit["fallback"],
        )
    entries.append(entry)
    return entries


def _check_events_order_zero(events, T, n_subjects, events_params):
    """
    Check ``events`` / ``events_params`` for ``features(order=0)``, which has no
    kernel to restrict: a dict with any key raises, and a non-dict
    specification is parsed as for one order (features are not segmented).
    """
    if isinstance(events, dict):
        if events:
            raise ValueError(
                "order is 0, so there is no kernel for events to restrict; "
                f"got events for order(s) {list(events)}"
            )
        events = None
    _check_params_keys(events_params)
    parsed = _parse_spec(events, T, n_subjects)
    if events_params and not _has_features(parsed):
        raise ValueError("events_params was given but no events entry holds features")


def features(
    data,
    order=1,
    weights_function=gaussian_weights,
    weights_params=None,
    cfun=isfc,
    rfun="auto",
    delta_rfun="auto",
    ndims=None,
    combine=null_combine,
    include_timepoints="all",
    include_zero=True,
    return_delta=False,
    max_memory_gb=4.0,
    *,
    events=None,
    events_params=None,
    delta_weights_function=None,
    delta_weights_params=None,
    return_info=False,
):
    """
    Compute features of every order from 0 to ``order`` with the two-path pipeline.

    Order 1 is the dynamic correlations of the data; order ``k`` is the
    dynamic correlations of the order-``(k-1)`` feeding features.

    ``X_0 = D_0 = data``.  For ``k >= 1``,
    ``X_k = reduce(combine(timecorr(D_{k-1}, analysis kernel, cfun)), rfun[k])``
    and ``D_k = reduce(timecorr(D_{k-1}, feeding kernel, autofc),
    delta_rfun[k])``.  The path that gives ``D_k`` is the feeding path; its
    arguments are named ``delta_*`` for historical reasons.  Reductions of
    lists are fit jointly across subjects, so all subjects share one space.

    With the defaults, the feeding kernel is a Gaussian with variance 25
    (standard deviation 5 samples), each ``D_k`` holds one column per
    community of features (``Communities()``: the mean squared correlation
    within the community over time, with the number of communities counted
    with a shift null), ``X_1`` is reduced with PCA and ``X_k`` for ``k >=
    2`` with ``WeightedStrength(signed=True)`` (two columns per input
    feature).  The default feeding reducer needs at least 4 features and 40
    timepoints at every order whose feeding step is computed (orders below
    ``order``, and ``order`` itself with ``return_delta=True``); otherwise
    ``ValueError`` names the alternatives: ``delta_rfun="weighted_strength"``
    (one column per feature, any number of features and timepoints) or the
    original pipeline.  At a default feeding step whose communities would
    emit fewer than 4 columns, the step emits weighted strength of the same
    correlations instead (one column per input feature; see
    ``delta_rfun``), so the feeding width never drops below 4.  This also
    changes calls that completed before the fallback was added: on 4
    independent noise features (T 100), ``features(x, order=1,
    return_delta=True)`` returned 1 community column in ``D[1]`` and now
    returns 4 weighted-strength columns, and ``features(x, order=2)``
    returns 8 columns in ``X[2]`` instead of 2.

    The original two-path pipeline (Owen, Chang & Manning, 2021, Fig. 10) is
    ``rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights``:
    the feeding features are then co-fluctuations (products of z-scored
    series), each order's features are blurred by the analysis kernel
    exactly once, and the order-``k`` features (without reduction) are
    correlations among products of up to ``2**(k-1)`` of the original
    features.

    A step whose reducer is a summary (``WeightedStrength`` or
    ``Communities``, as an object or a name) is streamed when its
    correlations come from ``autofc``, ``isfc`` or ``wisfc`` with
    ``null_combine``; every feeding step with a kernel other than
    ``eye_weights`` qualifies.  A streamed step computes its correlations a
    block of rows at a time, twice (fit, then output); its block size is the
    largest that fits in ``max_memory_gb`` (when ``max_memory_gb`` is None,
    the estimator's default: at most ``2**23`` correlations per block, a
    tenth of that with NaN samples, and at least one row), so the full (d,
    d, T) array of a subject is held only when one block holds all d rows.
    Other steps hold their correlations in full and then call the reducer.
    The summaries' weights and communities are fitted on all timepoints and
    all events of a step (the fit is not causal, also with
    ``include_timepoints="backward"``); ``fit`` and ``transform`` of an
    instance allow fitting on other data.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
        One timepoints-by-features matrix per subject.  A 1-D array or a
        pandas or polars Series is one feature (shape ``(T, 1)``); arrays
        with more than two dimensions raise ``ValueError``.  NaN samples
        carry no weight at order 1.  With ``eye_weights`` as the feeding
        kernel, their co-fluctuations are NaN, so later orders leave them
        out pairwise as well when the feeding path is unreduced
        (``delta_rfun=None`` with ``rfun=None``) or reduced by a callable
        other than a summary (which receives the NaNs).  A model-based
        reducer, a graph measure or a summary cannot take NaNs: before one,
        they are set to 0 and a ``UserWarning`` says that NaN handling is
        approximate from order 2 on.  With another feeding kernel, the
        estimator leaves NaN samples out pairwise (undefined correlations
        are 0, with its warning), so the feeding features are finite and
        that warning is not issued.  ``X_0`` (and ``D_0``) keep the NaNs.
    order : int
        Highest order to compute (non-negative integer).  Default: 1.
    weights_function, weights_params : function, dict or None
        Analysis kernel and its parameters (see ``timecorr``); row ``t`` of the
        kernel (``weights[t, :]``) is the kernel used at timepoint ``t``.
        ``weights_params`` may also be an order map: a non-empty dict whose
        keys are all integers in ``1..order`` and whose values are the
        kernel parameters of that order (a dict, or None for the kernel's
        defaults), for example ``{1: {"var": 100}, 2: {"var": 400}}``;
        orders missing from it use the kernel's defaults.  A dict without
        integer keys (or an empty dict) is the parameters of every order; a
        dict that mixes integer and other keys, has boolean keys (Python or
        NumPy) or keys outside ``1..order``, or maps an order to a value that
        is neither a dict nor None, raises ``ValueError``.  A dict whose keys
        are all integers is always read as an order map, so the parameters
        of a custom kernel that are keyed by integers must be wrapped in one
        (for example ``{1: params, 2: params}``).  Only the outer dict is
        examined: each value is passed to the kernel unchanged, whatever its
        keys.  Default:
        ``gaussian_weights`` with ``weights_params=None`` (its defaults,
        ``gaussian_params``).
    cfun : function
        Analysis-path correlation function (``isfc``, ``wisfc``, ``autofc``).
        Default: ``isfc``, which correlates each subject's features with the
        other subjects' and so reflects structure shared across subjects;
        ``autofc`` computes within-subject correlations.  On
        ``simulate_nested`` data (108 nodes, T 1000, 6 subjects in one call,
        12 seeds, other arguments default), ``h`` was decoded from ``X[2]``
        with ``autofc`` (score .559 +- .022, control -.008 +- .025) but not
        with ``isfc`` (-.035 +- .029, control .055 +- .019) when the order-1
        couplings are drawn per subject (``shared="order2"``), and with both
        when they are shared (``shared="order1"``: ``isfc`` .638 +- .028,
        control .074 +- .009; ``autofc`` .702 +- .012, control -.008 +-
        .025); see ``tests/test_nested_recovery.py``.  The feeding path
        does not use ``cfun``.
    rfun : None, str, dict, or callable
        Analysis-path reducer: anything ``helpers.reduce`` accepts (None, a
        model name, a model dict with a ``'model'`` key, a graph measure, a
        callable, a summary name ``"weighted_strength"``,
        ``"weighted_strength_split"`` or ``"communities"``, or a
        ``WeightedStrength`` or ``Communities`` object); ``"auto"``, which is
        ``"PCA"`` at order 1 and ``WeightedStrength(signed=True)`` at orders
        2 and above; or an order map, a dict whose keys are all integers in
        ``1..order`` and whose values are any of the above except an order
        map (orders missing from it use ``"auto"``).  A dict with a
        ``'model'`` key is a model configuration; any other dict that is not
        an order map raises ``ValueError``.  Reducers are resolved once per
        call, before anything is computed.  Summary names and ``"auto"``
        create new summary objects in every call; to read ``fits_``, pass an
        object.  Default: "auto".
    delta_rfun : None, str, dict, or callable
        Feeding-path reducer, with the same forms as ``rfun``; here
        ``"auto"`` is ``Communities()`` at every order (automatic count; at
        least 4 features and 40 timepoints, otherwise ``ValueError`` naming
        the alternatives; with more features than distinct shift-null lags
        in ``[L, T - L]``, lags are reused, see Notes).  At each feeding step it computes, the candidate
        communities are fitted to that step's correlations (with the shift
        null) and their emitted columns counted (communities with at least
        2 members; at most half the step's input features).  With 4 or more
        columns, the step emits the communities, as an explicit
        ``Communities()`` does.  With fewer (including the case where no
        eigenvalue exceeds the shift null, which gives one community), it
        emits ``WeightedStrength(split=False, signed=False, center=False)``
        of the same correlations (from the same pair weights, used as
        they are, without centering or clipping; no second null), one column per input feature (recorded in ``return_info``;
        one ``UserWarning`` per order only when no eigenvalue exceeds the
        shift null); the feeding width then stays at least 4 (for example
        with 4 to 7 input features, every default feeding step emits
        weighted strength).  An explicit ``Communities()`` or
        ``"communities"`` keeps its communities whatever their number.
        None (also as an order map value) is the same as ``rfun`` at that
        order.  A
        delta-kernel co-fluctuation matrix ``z z^T`` has rank one, so with
        ``eye_weights`` eigenvector centrality equals ``|z| / ||z||`` and the
        graph measures carry no network structure.  Default: "auto".
    ndims : int, optional
        Output width of model-based reduction (a model name or dict) on
        either path; it is not passed to a graph measure or a callable, which
        set their own width.  Given together with a summary on a step that
        is computed, it raises ``ValueError``.  Default (None): ``K`` columns
        for ``K(K+1)/2`` inputs.
    combine : function
        Applied to the analysis path only (see ``timecorr``).  Default:
        ``null_combine``.
    include_timepoints : str
        "all", "backward"/"pre", or "forward"/"post".  Shapes the analysis
        kernel and, unless it is ``eye_weights``, the feeding kernel.
        Default: "all".
    include_zero : bool
        Whether the estimated timepoint contributes to its own estimate.
        Shapes the analysis kernel and, unless it is ``eye_weights``, the
        feeding kernel.  Default: True.
    return_delta : bool
        If True, also return ``[D_0, ..., D_order]``.  If False, the feeding
        step of the last order (``D_order``), which feeds no later order, is
        not computed, so ``delta_rfun`` is not called at that step (and is not
        called at all when ``order`` is 1).  Default: False.
    max_memory_gb : float or None
        Upper limit, in gigabytes (1e9 bytes), on the estimated peak memory of
        any one correlation step for all subjects together.  Both paths of
        an order are checked before either computes anything.  For a step
        that holds its correlations, the estimate (see
        ``estimate_memory_gb``) covers the correlation step and the
        unreduced results it holds, not the memory of reducers or of
        ``combine`` functions other than ``null_combine``; it is checked on
        every step, because the (d, d, T) correlation array is allocated
        whether or not the result is then reduced; for large K, raising it
        or streaming the step is the remedy.  A streamed step uses blocks of
        as many rows as fit; if one row per block does not fit
        (``estimate_memory_gb(..., streamed=True)``), it raises.  With
        per-subject events, the kernels of the distinct event sets (one T x
        T array each) are not included in the estimate of a step that holds
        its correlations; a streamed step counts them and is checked before
        its kernels are built.  Every step, streamed or not, on both paths,
        also counts what the run retains: the outputs of earlier orders
        (``X_1 .. X_{k-1}``; ``D_1 .. D_{k-1}`` when ``return_delta=True``,
        otherwise only the feeding features the order reads, as the others
        are released), each array once (views of one base array by their
        sizes, at most the base's), and the (K, K) arrays kept in the
        ``fits_`` of the summary objects the run holds; a streamed feeding
        step also counts the analysis results of its order.  A step that
        holds its correlations does not count its own input (the previous
        order's feeding features) twice: its estimate already includes it.  The input data
        and the memory of reducers and ``combine`` functions themselves are
        not counted.  None checks nothing (a streamed step then uses the
        estimator's default block size).  Default: 4.0.
    events : None, dict, array-like, or list
        Restricts kernels to events (keyword-only; see ``timecorr`` for one
        specification: boundaries shared by all subjects, features with T
        rows segmented with ``segment``, or a list with one such entry, or
        None, per subject).  A dict maps orders (integers in ``1..order``)
        to specifications, and orders missing from it are unrestricted; any
        other value applies to every order.  At order ``k`` the analysis
        kernel and, unless it is ``eye_weights``, the feeding kernel are
        restricted with order ``k``'s events; with ``eye_weights`` the
        events of order ``k`` change ``X_k`` only.  Summaries are fitted
        across all events.  The events are resolved once, before any
        correlations are computed, and a features object used for several
        orders is segmented once (per distinct ``events_params``).  With
        ``order=0`` there is no kernel to restrict: a non-dict specification
        is checked (without segmenting features) and the data are returned
        unchanged, and a dict with any key raises ``ValueError``.  Default:
        None (no restriction).
    events_params : dict or None
        Keyword arguments for ``segment`` when ``events`` holds features
        (keyword-only; keys ``n``, ``kind``, ``max_events``,
        ``random_state``).  When ``events`` is a dict, this may instead be a
        dict keyed by order whose values are such dicts.  Unknown keys, or
        giving it when no entry of ``events`` holds features, raise
        ``ValueError``.  Default: None.
    delta_weights_function : function or None
        Feeding kernel (keyword-only), called as
        ``delta_weights_function(T, delta_weights_params)`` and shaped like
        the analysis kernel (``include_timepoints``, ``include_zero``,
        events).  ``eye_weights`` gives co-fluctuations by the
        co-fluctuation code path of the original pipeline (NaN handling and
        message as described under ``data``).  Default: None
        (``gaussian_weights``).
    delta_weights_params : dict or None
        Parameters of the feeding kernel (keyword-only).  With
        ``delta_weights_function=None`` they are the parameters of
        ``gaussian_weights`` (None: ``{"var": 25}``, standard deviation 5
        samples; a dict without ``"var"`` raises ``ValueError``); with a
        given function, None means that function's own
        defaults; with ``eye_weights``, which takes no parameters, they must
        be None (``ValueError`` otherwise).  An order map, as for
        ``weights_params``, sets the feeding kernel's parameters order by
        order; orders missing from it (or mapped to None) use what
        ``delta_weights_params=None`` gives (``{"var": 25}`` for the default
        Gaussian; with ``eye_weights`` every value must be None).  The
        shift null of a feeding step reads that order's standard deviation.
        Default: None.
    return_info : bool
        If True (keyword-only), return ``(result, info)``, where ``result``
        is what the call returns otherwise (``X``, or ``(X, D)`` with
        ``return_delta=True``) and ``info`` is a list with one dict per
        computed step and path, in the order computed: ``"order"``,
        ``"path"`` (``"analysis"`` or ``"feeding"``), ``"requested"`` (the
        reducer given for that order and path, for example ``"auto"``;
        objects as their ``repr``, functions by name), ``"effective"`` (the
        reducer that produced the output, as a string; None when
        unreduced), ``"input_width"`` and ``"output_width"`` (numbers of
        columns).  Default feeding steps (``delta_rfun="auto"``) also have
        ``"n_communities"`` (the columns the candidate communities emit),
        ``"n_exceed"`` (eigenvalues above the shift-null threshold),
        ``"null_threshold"``, ``"n_null"`` and ``"fallback"`` (None,
        ``"too_few_community_columns"`` or ``"no_null_exceedance"``).  No
        arrays.  Default: False.

    Returns
    -------
    X : list
        ``[X_0, ..., X_order]``.  For array input each entry is an array; for list
        input each entry is a list with one array per subject, unless ``combine``
        returns a single array.  ``X_0`` is the input as float arrays.
    D : list
        Only with ``return_delta=True``: ``[D_0, ..., D_order]``, with the same
        container types (never combined across subjects).
    info : list of dict
        Only with ``return_info=True`` (see ``return_info``); the call then
        returns ``(X, info)`` or ``((X, D), info)``.

    Warns
    -----
    UserWarning
        From the correlation estimator (see ``wcorr``), at most one per
        correlation step and path, naming each that applies: timepoints whose
        kernel row has fewer than 3 non-zero kernel weights (their correlations
        are undefined and set to 0); other timepoints whose effective sample
        size ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow
        to estimate correlations reliably there); timepoints where, because of
        NaN samples, some pairs of features have fewer than 3 observed samples
        with non-zero kernel weight (those correlations are undefined and set
        to 0); and timepoints with zero kernel-weighted variance in every
        feature (their correlations are set to 0).  Never for a feeding path
        with ``eye_weights``.
    UserWarning
        Once per call, when NaN co-fluctuations (``eye_weights``) are set to
        0 before a model-based, graph-measure or summary reduction.
    UserWarning
        From the summaries: no community structure above the shift null
        (one community; explicit ``Communities`` and split
        ``WeightedStrength``), or output columns constant over time.
    UserWarning
        Once per order whose default feeding step (``delta_rfun="auto"``)
        found no eigenvalue above the shift null and so emits weighted
        strength; the message gives the threshold and the number of input
        features.  Communities that were found but emit fewer than 4 columns
        fall back without a warning (see ``return_info``).

    Raises
    ------
    ValueError
        If ``order`` is not a non-negative integer; if ``rfun`` or
        ``delta_rfun`` is an unknown model, graph measure or summary name, a
        dict that is neither a model configuration (with a ``'model'`` key)
        nor an order map, or an order map with keys outside ``1..order`` or
        a value that is not a reducer (all are checked before anything is
        computed, even when a reducer would not be called); for a malformed
        order map in ``weights_params`` or ``delta_weights_params`` (see
        ``weights_params``; checked before anything is computed); if ``ndims`` is
        not a positive integer, or is given with a summary on a computed
        step (with the defaults, whenever ``order >= 2``, and at
        ``order=1`` with ``return_delta=True``, which computes the order-1
        feeding step and its reducer ``Communities()``); if the default
        feeding reducer (``delta_rfun="auto"``) gets fewer than 4 features
        or fewer than 40 timepoints at a computed feeding step: the input data, or the output
        of an explicit reducer of the previous order (for example an
        explicit ``Communities()``, whose communities the message then
        names); the message names ``delta_rfun="weighted_strength"`` and
        the original pipeline's settings; for a ``Communities`` (or split ``WeightedStrength``) without
        ``n`` where the number of communities cannot be counted (analysis
        steps with a ``cfun`` other than ``autofc`` or a ``combine`` other
        than ``null_combine``; T < 40) and for K < 4 nodes; for a malformed
        ``events`` specification or ``events_params`` (including dict keys
        outside ``1..order``), and when the events of an order leave no
        kernel row with 3 or more non-zero weights; and as in ``timecorr``
        (subjects of different shapes, non-numeric columns, an unknown
        ``include_timepoints``).  The events are checked (and features
        segmented) before anything is computed.
    RuntimeError, ImportError
        As raised by ``segment`` for features entries of ``events``.
    TypeError
        If ``rfun`` or ``delta_rfun`` is not None, a string, a dict or a
        callable, or ``delta_weights_function`` is not None or callable.
    MemoryError
        Before a correlation step allocates its arrays, if that step's
        estimate exceeds ``max_memory_gb`` (with unreduced feeding features,
        every order is checked before anything is computed; for a streamed
        step, if one row per block does not fit).  Every streamed step is
        also checked before anything is computed, with its input width where
        it is known in advance (K at order 1; ``d(d+1)/2`` after an
        unreduced feeding step, ``ndims`` or ``d`` after a model-based one,
        ``d`` after a graph measure, ``d`` times 2 with ``split`` times 2
        with ``signed`` after weighted strength) and K after a callable
        (this up-front check takes K, the number of features of the input
        data, as the output width of a callable reducer that is not a
        summary, whatever the callable's own input width; its actual width
        is checked when the next step runs), as
        ``estimate_memory_gb(..., streamed=True)`` uses K.  After
        ``Communities``, whose width is found by the fit, the steps whose
        width follows from it are checked when they run.
        The message gives the estimate and the largest number of features
        that would fit; for an analysis step that holds its correlations
        although a summary would stream it (for example the default
        order-1 PCA step, with ``cfun`` ``isfc``, ``wisfc`` or ``autofc``
        and ``combine=null_combine``), it also names
        ``rfun=tc.WeightedStrength(signed=True)`` and the order map
        ``rfun={1: tc.WeightedStrength(signed=True)}``.

    Notes
    -----
    The ``WeightedStrength`` and ``Communities`` objects passed in ``rfun``
    or ``delta_rfun`` (also as order map values) have their ``fits_`` cleared
    at the start of the call; each step they reduce appends one entry, with
    ``"order"`` and ``"path"`` (``"analysis"`` or ``"feeding"``).  With
    ``n=None``, the number of communities is counted with the shift null
    (see ``Communities``) on the feeding path and on analysis steps with
    ``cfun=autofc`` and ``combine=null_combine``; the null uses the step's
    kernel (with its events), and circular shifts of at least
    ``L = min(T // 4, max(10, 10 * sd))`` samples, where sd is the kernel's
    standard deviation (``sqrt(var)`` for kernels whose parameters, or
    whose function's default parameters when they are None, have a
    ``"var"``; otherwise 5, whatever the kernel's own width parameter, for
    example ``scale`` of ``laplace_weights``, and 5 for ``eye_weights``),
    distinct within a subject when the n integers
    in ``[L, T - L]`` are at least the d input features (drawn without
    replacement); with d > n, each lag is used ``d // n`` or
    ``ceil(d / n)`` times within a subject, and columns that share a lag
    (a fraction [n q (q - 1) + 2 r q] / [d (d - 1)] of the column pairs,
    d = q n + r) keep their dependence in that surrogate, with one
    ``UserWarning`` per fit; and the summary's ``n_null`` surrogates.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_nested(S=2, T=300, n_supergroups=2,
    ...                                  modules_per_supergroup=3,
    ...                                  nodes_per_module=4, set_random_seed=0)

    With the defaults, ``X_1`` is PCA-reduced (24 columns for 24 nodes),
    ``D_1`` has one column per community (6 modules) and ``X_2`` is signed
    weighted strength of ``D_1`` (two columns per community):

    >>> X = tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 100})
    >>> len(X), len(X[2]), X[1][0].shape, X[2][0].shape
    (3, 2, (300, 24), (300, 12))

    Restrict the order-2 kernels to two events (timepoints 0-149 and
    150-299); ``X_1`` is unrestricted:

    >>> X = tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 100},
    ...                 events={2: [150]})
    >>> X[2][0].shape
    (300, 12)

    Kernels by order: a wider analysis kernel at order 2 than at order 1,
    and a feeding kernel of variance 64 at order 1 (orders missing from a
    map use the defaults):

    >>> X = tc.features(data, order=2, cfun=tc.autofc,
    ...                 weights_params={1: {"var": 100}, 2: {"var": 400}},
    ...                 delta_weights_params={1: {"var": 64}})
    >>> X[1][0].shape, X[2][0].shape
    ((300, 24), (300, 12))

    Weighted strength on the feeding path (one column per feature; any
    number of features and timepoints):

    >>> rng = np.random.RandomState(0)
    >>> small = [rng.randn(50, 3) for _ in range(3)]
    >>> X = tc.features(small, order=2, cfun=tc.autofc, weights_params={"var": 10},
    ...                 delta_rfun="weighted_strength")
    >>> X[1][0].shape, X[2][0].shape
    ((50, 3), (50, 6))

    The original pipeline (co-fluctuations on the feeding path, PCA at every
    order), and the same feeding kernel without reduction (order 1 then has
    K(K+1)/2 = 6 columns):

    >>> X = tc.features(small, order=2, weights_params={"var": 10},
    ...                 rfun="PCA", delta_rfun=None,
    ...                 delta_weights_function=tc.eye_weights)
    >>> len(X), len(X[2]), X[2][0].shape
    (3, 3, (50, 3))
    >>> X, D = tc.features(small[0], order=1, rfun=None, delta_rfun=None,
    ...                    delta_weights_function=tc.eye_weights, return_delta=True)
    >>> X[1].shape, D[1].shape
    ((50, 6), (50, 6))
    """
    if (
        not isinstance(order, numbers.Integral)
        or isinstance(order, bool)
        or order < 0
    ):
        raise ValueError(f"order must be a non-negative integer; got {order!r}.")
    order = int(order)
    # every reducer is checked and resolved here, including ones a short run
    # never calls
    orders = range(1, order + 1)
    analysis_rfun, feeding_rfun = _resolve_reducers(rfun, delta_rfun, orders, order)
    ndims = _check_ndims(ndims)
    fed = [k for k in orders if k < order or return_delta]
    _check_ndims_steps(
        ndims,
        [(analysis_rfun[k], "rfun", k) for k in orders]
        + [(feeding_rfun[k], "delta_rfun", k) for k in fed],
    )
    analysis_map = _params_by_order(weights_params, "weights_params", order)
    feeding_map = _params_by_order(delta_weights_params, "delta_weights_params", order)
    # kernel parameters of each order: an order map's value (None, the
    # default, for orders missing from it), or the same parameters at every
    # order
    analysis_params = {
        k: weights_params if analysis_map is None else analysis_map.get(k)
        for k in orders
    }
    if feeding_map is None:
        feeding_function, plain = _feeding_kernel(
            delta_weights_function, delta_weights_params
        )
        feeding_params = {k: plain for k in orders}
    else:
        feeding_function, _ = _feeding_kernel(delta_weights_function, None)
        feeding_params = {
            k: _feeding_kernel(delta_weights_function, feeding_map.get(k))[1]
            for k in orders
        }
    for summary in _summary_instances(rfun, delta_rfun):
        summary.fits_ = []

    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    T, K = subjects[0].shape
    delta_unreduced = all(feeding_rfun[k] is None for k in orders)

    S = len(subjects)
    # events are resolved (and features segmented) once, before anything is computed
    if order == 0:
        _check_events_order_zero(events, T, S, events_params)
        resolved = {}
    else:
        resolved = resolve_events_by_order(events, T, S, order, events_params)

    if delta_unreduced and max_memory_gb is not None:
        # widths are known in advance: refuse before computing anything
        estimates = estimate_memory_gb(
            K, order, T, S, return_delta, delta_weights_function=feeding_function
        )
        for k in range(1, order + 1):
            if estimates[k] > max_memory_gb:
                fit = _largest_k(order, T, max_memory_gb, S, return_delta,
                                 kernel_feeding=feeding_function is not eye_weights)
                raise MemoryError(
                    f"Unreduced features at order {k} need about {estimates[k]:.3g} GB "
                    f"(K={K}, T={T}, {S} subject{'s' if S > 1 else ''}), more than "
                    f"max_memory_gb={max_memory_gb}. The largest number of features "
                    f"that fits orders up to {order} is K={fit}. Pass a reducer "
                    "(rfun, delta_rfun), lower the order, or raise max_memory_gb."
                )
    if max_memory_gb is not None:
        # streamed steps of later orders are checked before order 1 runs
        _check_streamed_steps(
            K, T, S, max_memory_gb, analysis_rfun, feeding_rfun, fed, cfun=cfun,
            combine=combine, feeding_function=feeding_function, ndims=ndims,
        )

    X = [subjects]
    D = [subjects]
    fed_by = subjects  # the feeding features the next order reads
    # the reducer objects the run holds (their fits_ are retained)
    held_reducers = tuple(
        r for r in list(analysis_rfun.values()) + list(feeding_rfun.values())
        if isinstance(r, _Summary)
    )
    nan_warned = [False]  # one NaN-filling warning per call
    info = []  # with return_info: one dict per computed step and path
    previous_communities = None  # communities found by the previous feeding step
    for k in orders:
        # the outputs kept so far (each array once; the input data are the
        # caller's): X_1 .. X_{k-1}, and D_1 .. D_{k-1} when returned,
        # otherwise only the feeding features this order reads
        kept = X[1:] + (D[1:] if return_delta else ([] if k == 1 else [fed_by]))
        # the last order's feeding features feed nothing and, unless returned,
        # are not computed (unreduced, they are as large as X_order)
        x, delta = _two_path_step(
            fed_by,
            delta_rfun=feeding_rfun[k],
            auto_analysis=_auto_at(rfun, k),
            max_memory_gb=max_memory_gb,
            order=k,
            nan_warned=nan_warned,
            feed=k in fed,
            feeding_function=feeding_function,
            feeding_params=feeding_params[k],
            weights_function=weights_function,
            weights_params=analysis_params[k],
            cfun=cfun,
            rfun=analysis_rfun[k],
            ndims=ndims,
            combine=combine,
            include_timepoints=include_timepoints,
            include_zero=include_zero,
            events=resolved[k],
            retained_elements=_output_elements(*kept),
            input_elements=0 if k == 1 else _output_elements(fed_by),
            reducers=held_reducers,
            previous_communities=previous_communities,
        )
        X.append(x)
        previous = feeding_rfun[k]
        # explicit Communities only: the automatic feeding reducer keeps at
        # least 4 columns (see summaries._auto_emitter)
        previous_communities = (
            previous.fits_[-1]["n"]
            if isinstance(previous, Communities) and previous.fits_
            and not getattr(previous, "_from_auto", False) else None
        )
        if return_info:
            info.extend(_step_info(
                k, fed_by, x, delta, analysis_rfun[k], feeding_rfun[k],
                _requested(rfun, k), _requested(delta_rfun, k),
            ))
        if delta is not None:
            fed_by = delta
            if return_delta:
                D.append(delta)

    if single:
        X = [x[0] if isinstance(x, list) else x for x in X]
        D = [d[0] for d in D]
    result = (X, D) if return_delta else X
    if return_info:
        return result, info
    return result
