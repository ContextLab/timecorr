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
counted with a shift null) and Gaussian kernels on both paths whose widths
are set from each step's input (``weights_params=None``,
``delta_weights_params=None``: the feeding kernel's standard deviation is 5
autocorrelation widths of the input, at most a hundredth of its rows
but at least 5 rows (which wins below 500 rows), and the analysis kernel's is that or 10 rows,
whichever is larger; see the Notes of ``features``), so the feeding
features are kernel-weighted correlations summarized by community.  When no
kernel parameters and no events are given, ``features`` passes the feeding
features of long recordings on at a lower rate and returns every output at
all timepoints by linear interpolation.  Kernel parameters that are given
(for example ``weights_params={"var": 100}``,
``delta_weights_params={"var": 25}``, the fixed kernels of earlier
versions) are used as they are, at every timepoint.
The feeding kernel is shaped by
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

import contextlib
import functools
import inspect
import numbers
import warnings

import numpy as np

from .helpers import (
    _SUMMARY_NAMES,
    _check_min_samples,
    _check_ndims,
    _check_reachable,
    _check_rfun,
    _LAG_PROFILES,
    _is_model,
    _fisher_workspace,
    _kernel_work_elements,
    _lag_kernel,
    _wcorr_block_rows,
    _warn_user,
    _zscore_columns,
    autofc,
    corrmean_combine,
    eye_weights,
    format_data,
    gaussian_weights,
    isfc,
    mean_combine,
    null_combine,
    reduce,
    tstat_combine,
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
    _FittedSummary,
)
from .timecorr import (
    _apply_frozen,
    _builtin_combine,
    _check_fit_timepoints,
    _check_linear_model,
    _event_kernels,
    _fit_reducer,
    _fixed_weights_wisfc,
    _needs_fit,
    _restricted,
    _shaped_kernel,
    _uncovered_model,
    _warn_uncovered_models,
    timecorr,
)

_BYTES_PER_GB = 1e9
# outputs computed at fewer timepoints are interpolated to all T timepoints
# in chunks of rows of at most this many elements (a few such chunks of work
# space beside the full-length result)
_EXPAND_ELEMENTS = 2 ** 17

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
# measured with tracemalloc for isfc and autofc: about 7.3 to 8 blocks
# including the kernel's blocks, which are counted separately).
_WORKSPACE_BLOCKS = 7


def _builtin_kernel(weights_function):
    """True for a built-in kernel function (read in chunks of rows, no T x T array)."""
    try:
        return weights_function is None or weights_function in _LAG_PROFILES
    except TypeError:
        return False


def _dense_analysis(weights_function, cfun=isfc, causal=False):
    """
    True when an analysis step holds T x T kernel arrays: for a kernel
    function of your own, and for a correlation function of your own, which
    ``timecorr`` passes T x T arrays (only ``autofc``, ``isfc``, ``wisfc``
    and the smoothing of ``cfun=None`` read a built-in kernel in chunks of
    rows; with ``causal``, also ``functools.partial(wisfc,
    subject_weights=...)``, which ``timecorr`` then calls as ``wisfc``).
    """
    # cfun=None smooths the data with the kernel, also in chunks of rows
    builtin_cfun = (
        cfun is None or any(cfun is f for f in _STREAMED_CFUNS)
        or (causal and _fixed_weights_wisfc(cfun))
    )
    return not (_builtin_kernel(weights_function) and builtin_cfun)


def _combine_elements(combine, S, T, v):
    """
    Float64 elements a built-in ``combine`` function allocates on S (T, v)
    results, its output included (measured with ``tracemalloc`` for S 2 to
    10: ``corrmean_combine`` 5.26 to 5.31 S T v, ``mean_combine`` (S + 1) T v,
    ``tstat_combine`` 7.65 to 9.75 S T v, with margin); 0 for
    ``null_combine`` and for a function of your own, which is not counted.
    """
    if combine is corrmean_combine:
        return 11 * S * T * v // 2 if S > 1 else 0
    if combine is mean_combine:
        return (S + 1) * T * v
    if combine is tstat_combine:
        return (15 * S + 10) * T * v // 2
    return 0


def _step_gb(d, T, n_subjects=1, n_outputs=1, kernel_feeding=False, dense=False,
             combine=None):
    """
    Peak memory (GB) of one unreduced correlation step on ``n_subjects`` (T, d)
    inputs that holds ``n_outputs`` (T, d(d+1)/2) results per subject.
    ``kernel_feeding``: the feeding path (``n_outputs=2``) uses a kernel other
    than ``eye_weights``, so it has block temporaries of its own while both
    paths' results are held.  ``dense``: a kernel of the step is a T x T
    array (a kernel function that is not built in), not read in chunks.
    ``combine``: the analysis path's combine function; a built-in one is
    counted (``_combine_elements``), after the correlations are freed.
    """
    d, T, S = int(d), int(T), int(n_subjects)
    v = d * (d + 1) // 2
    block = min(d, _wcorr_block_rows(T, d)) * T * d
    # the kernel, its working copy and its absolute value as T x T arrays;
    # a built-in kernel holds at most that and a bounded set of row blocks
    kernel = 3 * T * T if dense else _kernel_work_elements(T, 3 * T * T)
    correlation = (
        d * d * T  # the (d, d, T) correlation array of one subject
        # the stacked subjects and one subject's reference (wisfc, which
        # also computes isfc and autofc, stacks the subjects)
        + (S + 1) * T * d
        # the estimator's blocks, or the Fisher z symmetrization of isfc and
        # wisfc with several subjects, which runs after them (one subject is
        # its own reference and is not symmetrized)
        + _workspace(max(_WORKSPACE_BLOCKS * block, _fisher_workspace(d, T) if S > 1 else 0),
                     (n_outputs - 1) * S * T * v, kernel_feeding)
    )
    elements = (
        S * T * v  # the analysis path's vectorized results, every subject
        + 3 * S * T * d  # the inputs (and the copies made while formatting)
        + kernel
        # the correlation step, or the combine function, which runs after it
        + max(correlation, _combine_elements(combine, S, T, v))
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
                       *, delta_weights_function=None, weights_function=None,
                       combine=null_combine, cfun=isfc):
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
    ``delta_weights_function`` and ``combine`` for data without NaNs;
    reducers allocate more.  The built-in kernel functions are read a chunk of rows at a time:
    their kernel term is ``3 T**2`` elements up to T = 1,270 and then
    grows linearly in T (at most 32 MB plus 4 blocks of 128 rows by T
    columns); a kernel function of your own gives T x T arrays (``3 T**2``),
    as does a correlation function (``cfun``) of your own, which receives
    T x T arrays.  The feeding kernel counts only at the orders whose feeding
    path runs (every order but the last, and the last with
    ``return_delta=True``).  It counts 7 block temporaries of the estimator (or the temporaries of
    the Fisher z symmetrization of ``isfc`` and ``wisfc`` with several
    subjects, when larger) and what the estimator holds besides the
    correlations (the stacked subjects and one subject's reference).
    With per-subject events, the kernels of the distinct event sets (one
    T x T array each for a kernel function of your own) are not included in
    the estimate.

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
        Number of timepoints.  Every order is estimated with T rows; when
        ``features`` passes feeding features on at a lower rate (no kernel
        parameters and no events given, T of 2000 or more), orders 2 and
        above run on fewer rows (at least 1000) and need less, so the
        estimate is an upper limit for them (``estimate_memory_gb(d, 1,
        rows, ...)`` gives one such step).
    n_subjects : int, optional
        Number of subjects.  Default: 1.
    return_delta : bool, optional
        Whether the delta path of the last order is computed too (as
        ``features(..., return_delta=True)`` does).  With
        ``streamed=True`` it only decides whether a feeding kernel function
        of your own counts at the last order.  Default: False.
    streamed : bool, optional
        If True, estimate streamed steps (see above).  Default: False.
    delta_weights_function : function or None, optional
        The feeding (delta) kernel of the run (keyword-only), as in
        ``features``: None (the default Gaussian kernel) or any kernel other
        than ``eye_weights`` counts the feeding path's block temporaries;
        ``eye_weights`` (the original pipeline's co-fluctuations) does not.
        Default: None.
    weights_function : function or None, optional
        The analysis kernel of the run (keyword-only), as in ``features``:
        None (the default Gaussian kernel) or a built-in kernel function is
        read in chunks of rows; a function of your own returns T x T arrays,
        which are counted as such (as is a ``delta_weights_function`` of your
        own).  Default: None.
    combine : function, optional
        The combine function of the run (keyword-only), as in ``features``.
        The temporaries of the built-in ones (``corrmean_combine``,
        ``mean_combine``, ``tstat_combine``) are counted; a function of your
        own cannot be counted and is estimated as ``null_combine``.  Not used
        with ``streamed=True``.  Default: ``null_combine``.
    cfun : function, optional
        The correlation function of the run (keyword-only), as in
        ``features``: ``autofc``, ``isfc`` and ``wisfc`` read a built-in
        kernel in chunks of rows; a function of your own receives T x T
        kernel arrays, which are counted as such.  Default: ``isfc``.

    Returns
    -------
    list of float
        ``order + 1`` estimates in gigabytes (1e9 bytes), indexed by order; entry
        0 is 0.0 because order 0 is the input itself.

    Examples
    --------
    >>> from timecorr.higher_order import estimate_memory_gb
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, n_subjects=3)]
    [0.0, 0.18, 104.372]
    >>> import timecorr as tc
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, n_subjects=3,
    ...                                          delta_weights_function=tc.eye_weights)]
    [0.0, 0.155, 104.372]
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, 3, streamed=True)]
    [0.0, 0.014, 0.014]
    """
    return _estimates(K, order, T, n_subjects, return_delta, streamed,
                      delta_weights_function is not eye_weights,
                      _dense_analysis(weights_function, cfun), combine,
                      not _builtin_kernel(delta_weights_function))


def _estimates(K, order, T, n_subjects, return_delta, streamed, kernel_feeding,
               dense, combine=None, dense_feeding=None, later_T=None):
    """``estimate_memory_gb`` with the kernels given by ``kernel_feeding`` and
    ``dense`` (see ``_step_gb``): ``dense`` for the analysis path,
    ``dense_feeding`` (None: the same as ``dense``) for the feeding path,
    which counts only at the orders where it runs (every order but the last,
    and the last with ``return_delta``, streamed or not).  ``later_T``: the
    rows of orders 2 and above (None: T)."""
    K = int(K)
    T = int(T)
    if dense_feeding is None:
        dense_feeding = dense
    estimates = [0.0]
    d = K
    for k in range(1, int(order) + 1):
        rows = T if k == 1 or later_T is None else int(later_T)
        both_paths = k < order or return_delta
        dense_k = dense or (both_paths and dense_feeding)
        if streamed:
            estimates.append(
                8 * _streamed_peak_elements(rows, K, n_subjects, 1, dense_k)
                / _BYTES_PER_GB
            )
            continue
        estimates.append(
            _step_gb(d, rows, n_subjects, 2 if both_paths else 1, kernel_feeding,
                     dense_k, combine)
        )
        d = d * (d + 1) // 2
    return estimates


def _largest_k(order, T, max_memory_gb, n_subjects=1, return_delta=False,
               streamed=False, kernel_feeding=False, dense=False, combine=None,
               dense_feeding=None, later_T=None):
    """Largest K whose estimates up to ``order`` all fit (0 if none)."""

    def fits(k):
        est = _estimates(k, order, T, n_subjects, return_delta, streamed,
                         kernel_feeding, dense, combine, dense_feeding, later_T)
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
                       kernel_feeding=False, held_elements=0, hint="", dense=False,
                       combine=None):
    """
    Raise MemoryError before an unreduced correlation step on ``n_subjects``
    (T, d) inputs whose estimate (``_step_gb``), plus ``held_elements``
    float64 values retained by the caller (earlier outputs and fits of a
    ``features`` run), exceeds ``max_memory_gb``.  ``max_memory_gb=None``
    checks nothing.  ``hint`` is appended to the message.  ``dense``: a
    kernel of the step is a T x T array, and ``combine`` the analysis
    path's combine function (see ``_step_gb``).  Shared by
    ``features``, the path functions and the decoders, so all refuse the
    same steps.
    """
    if max_memory_gb is None:
        return
    S = int(n_subjects)
    step = _step_gb(d, T, S, n_outputs, kernel_feeding, dense, combine)
    held = 8 * int(held_elements) / _BYTES_PER_GB
    if step + held > max_memory_gb:
        fit = _largest_k(1, T, max_memory_gb - held, S, n_outputs > 1,
                         kernel_feeding=kernel_feeding, dense=dense, combine=combine)
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
                             path="analysis", dense=False):
    """
    ``max_block_elements`` for a streamed step on ``n_subjects`` (T, d)
    inputs: ``b * d * T`` with b the largest number of rows per block whose
    peak (``summaries._streamed_peak_elements``) fits in ``max_memory_gb``
    (None: no limit; the step then uses the estimator's default block
    size, ``helpers._wcorr_block_rows``).  Raise MemoryError, before the step
    computes anything, when one row per block does not fit.  The only place
    that converts the byte budget of a streamed step to a block size.
    ``dense``: the step's kernels are T x T arrays (see ``_step_gb``).
    """
    if max_memory_gb is None:
        return None
    S = int(n_subjects)
    b = _block_rows_for(T, d, S, max_memory_gb * _BYTES_PER_GB, dense)
    if b < 1:
        need = _estimates(d, 1, T, S, False, True, True, dense)[1]
        fit = _largest_k(1, T, max_memory_gb, S, streamed=True, dense=dense)
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


def _budget_left(max_memory_gb, held_elements, d, T, n_subjects, order, what,
                 dense=False):
    """
    The budget (GB) left for a streamed computation on ``n_subjects`` (T, d)
    inputs while ``held_elements`` float64 values are held (None: no limit).
    Raise MemoryError, before anything is computed, when what is held plus one
    row of correlations per block exceeds ``max_memory_gb``.  ``dense``: the
    kernels are T x T arrays (see ``_step_gb``).
    """
    if max_memory_gb is None:
        return None
    S = int(n_subjects)
    held = 8 * int(held_elements) / _BYTES_PER_GB
    need = 8 * _streamed_peak_elements(T, d, S, 1, dense) / _BYTES_PER_GB
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


# float64 elements (per timepoint) of a LagKernel: its profile, its
# symmetric profile, and the block labels and bounds of its events
_LAG_KERNEL_PER_T = 6


def _event_kernel_elements(T, n_subjects, events, dense=False):
    """
    Elements of the kernels a step builds for RESOLVED ``events`` when the
    subjects' events differ: per distinct event set, one T x T array for a
    kernel function that is not built in (``dense``), and
    ``_LAG_KERNEL_PER_T * T`` for a built-in one; 0 when one kernel is
    shared by every subject (as without events).
    """
    groups = distinct_event_sets(_restricted(events), int(n_subjects))
    if len(groups) == 1:
        return 0
    T = int(T)
    return len(groups) * (T * T if dense else _LAG_KERNEL_PER_T * T)


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
                   append_fit=False, causal=False, frozen=None, record=None):
    """
    ``_reduce``; a summary is fit (recording ``order`` and ``path`` in its
    ``fits_``) and then transforms the same correlations.  With
    ``append_fit`` False, ``fits_`` keeps only the new fit (as a direct call
    of the summary does); True appends it (the steps of ``features`` and
    ``two_path_step``).

    ``causal``: the correlations are those of the causal estimator, whose
    undefined rows are NaN: they, and any row with a NaN or infinite value,
    are left out of every fit and are rows of NaN in the output (see ``timecorr._fit_reducer`` and
    ``timecorr._apply_frozen``).  The reducer is fitted to ``corrs`` (the
    training rows of a causal fit) unless ``frozen`` holds a fit made
    earlier (a ``_FittedSummary`` or ``_FittedModel``), which then reduces
    ``corrs``.  ``record`` (a dict): ``record[path]`` is set to what
    reduced the step (the fit, or the reducer itself when it has none).
    """
    if causal:
        if frozen is None:
            frozen = _fit_reducer(
                corrs, rfun, ndims, order=order, path=path, append_fit=append_fit,
                name="rfun" if path == "analysis" else "delta_rfun",
            )
        if record is not None:
            record[path] = frozen
            # a record of a refit matches the fit to the one before
            # (model._Matched): what it holds reduces the step
            frozen = record[path]
        return _apply_frozen(corrs, frozen)
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


class _causal_step:
    """
    Context of one ``timecorr`` call of a step: the keyword arguments of the
    causal estimator (none without ``causal``).  ``timecorr``'s warning about
    a correlation or combine function of the user's is not repeated at every step
    (``_check_causal`` issued it once for the call).
    """

    def __init__(self, causal, min_samples):
        self.causal = causal
        self.min_samples = min_samples
        self._caught = None

    def __enter__(self):
        if not self.causal:
            return {}
        self._caught = warnings.catch_warnings()
        self._caught.__enter__()
        warnings.filterwarnings(
            "ignore", message=r"causal=True: (cfun|combine|cfun and combine) "
                              r"(is a function|are functions) of your own"
        )
        return {"causal": True, "min_samples": self.min_samples}

    def __exit__(self, *exc):
        if self._caught is not None:
            self._caught.__exit__(*exc)
        return False


def _streams(rfun, cfun, combine, groups=None, causal=False):
    """Whether an analysis step is streamed (see the module docstring);
    with ``causal``, ``wisfc`` with fixed subject weights is streamed too."""
    return (
        groups is None
        and combine is null_combine
        and (any(cfun is f for f in _STREAMED_CFUNS)
             or (causal and _fixed_weights_wisfc(cfun)))
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
                  include_timepoints="all", include_zero=True, events=None,
                  causal=False):
    """
    The kernel(s) ``timecorr`` builds for these arguments and RESOLVED events:
    one kernel, or one per subject when the subjects' events differ
    (``LagKernel`` objects for a built-in kernel function, T x T arrays
    otherwise).  ``causal``: as ``timecorr(..., causal=True)`` builds them.
    """
    weights = _shaped_kernel(
        T, weights_function, weights_params, include_timepoints, include_zero,
        causal=causal,
    )
    resolved = resolve_events(events, T, n_subjects, None, causal=causal)
    return _event_kernels(weights, resolved, n_subjects, causal)


def _run_streamed(subjects, summary, cfun, kernels, *, max_memory_gb, order, path,
                  auto_n, bandwidth, append_fit=False, dense=False, estimator=None,
                  frozen=None, record=None):
    """
    ``summaries._streamed_summary`` under a memory budget; issues the
    estimator's warning.  ``append_fit`` as in ``_apply_reducer``.
    ``estimator``: the keyword arguments of the causal estimator (None
    without ``causal``); ``frozen`` and ``record`` as in ``_apply_reducer``.
    """
    # C-ordered float copies: the blocked arithmetic's round-off then does not
    # depend on the memory layout of the inputs (reducers can return views)
    subjects = [np.ascontiguousarray(x, dtype=np.float64) for x in subjects]
    T, d = subjects[0].shape
    max_block = _streamed_block_elements(
        d, T, len(subjects), max_memory_gb, order, path, dense
    )
    step = dict(
        max_block_elements=max_block,
        order=order,
        path=path,
        auto_n=auto_n,
        bandwidth=bandwidth,
        # the shift null (when it runs) chooses its timepoints: every s-th
        # for a stationary built-in kernel, every timepoint otherwise
        null_row_stride=None,
        estimator=estimator,
    )
    out, message = _streamed_summary(subjects, kernels, summary, cfun,
                                     fitted=frozen, **step)
    if frozen is None:
        if record is not None:
            made = _FittedSummary(summary, summary.fits_[-1])
            record[path] = made
            if record[path] is not made:
                # a refit, matched to the fit before (model._Matched): the
                # matched fit summarizes the step
                out, _ = _streamed_summary(subjects, kernels, summary, cfun,
                                           fitted=record[path], **step)
        if not append_fit:
            del summary.fits_[:-1]
    elif record is not None:
        record[path] = frozen
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
    causal=False,
    min_samples=None,
    frozen=None,
    record=None,
):
    """
    Analysis path of one order: ``reduce(combine(timecorr(subjects, kernel, cfun)))``.

    ``causal`` and ``min_samples`` are ``timecorr``'s (the arguments were
    checked by ``_check_causal``); ``frozen`` and ``record`` as in
    ``_apply_reducer`` (with ``causal`` a reducer is fitted to the rows
    given, the training rows of a causal fit, or ``frozen`` is applied).

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
    dense = _dense_analysis(weights_function, cfun, causal)
    estimator = {"causal": True, "min_samples": min_samples} if causal else None
    if _streams(rfun, cfun, combine, groups, causal):
        summary = _as_summary(rfun)
        # the distinct event kernels and earlier fits are held while the step
        # runs: checked before the kernels are built
        budget = _budget_left(
            max_memory_gb,
            held_elements + _event_kernel_elements(T, len(subjects), events, dense)
            + _fits_elements(summary, *reducers),
            d, T, len(subjects), order, "streamed analysis step", dense,
        )
        kernels = _step_kernels(
            T, len(subjects), weights_function, weights_params,
            include_timepoints, include_zero, events, causal,
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
            dense=dense,
            estimator=estimator,
            frozen=frozen,
            record=record,
        )
    _check_step_memory(d, T, len(subjects), n_outputs, max_memory_gb, order,
                       kernel_feeding,
                       held_elements - input_elements + _fits_elements(*reducers),
                       dense=dense, combine=combine)

    def analyze(members, member_events):
        with _causal_step(causal, min_samples) as estimator:
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
                        **estimator,
                    )
                )
            )

    if groups is None:
        return _apply_reducer(
            analyze(subjects, events), rfun, ndims, order, "analysis",
            append_fit=append_fit, causal=causal, frozen=frozen, record=record,
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
    causal=False,
    min_samples=None,
    frozen=None,
    record=None,
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
    streamed.  ``causal``, ``min_samples``, ``frozen`` and ``record`` as in
    ``_analysis_path``.
    """
    T, d = subjects[0].shape
    S = len(subjects)
    summary = _as_summary(rfun)
    dense = not _builtin_kernel(weights_function)
    if weights_function is not eye_weights:
        if summary is not None:
            # what the caller holds (``held_elements``: results and retained
            # fits) and the distinct event kernels: checked before the
            # kernels are built
            budget = _budget_left(
                max_memory_gb,
                held_elements + _event_kernel_elements(T, S, events, dense),
                d, T, S, order, "streamed feeding step", dense,
            )
            kernels = _step_kernels(
                T, S, weights_function, weights_params, include_timepoints,
                include_zero, events, causal,
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
                dense=dense,
                estimator=({"causal": True, "min_samples": min_samples}
                           if causal else None),
                frozen=frozen,
                record=record,
            )
        _check_step_memory(d, T, S, n_outputs, max_memory_gb, order, True,
                           retained_elements, dense=dense)
        with _causal_step(causal, min_samples) as estimator:
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
                    **estimator,
                )
            )
        return _as_list(
            _apply_reducer(corrs, rfun, ndims, order, "feeding",
                           append_fit=append_fit, causal=causal, frozen=frozen,
                           record=record)
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
            _lag_kernel(T, eye_weights, None),
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
    on the analysis path; ``Communities()`` on the feeding path (marked with
    ``_from_auto`` so that ``_check_auto_feeding`` can name the alternatives)."""
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


def _check_auto_feeding(delta_rfun, d, T, order, previous=None, training=False):
    """
    Raise ``ValueError`` naming the alternatives when the feeding reducer
    resolved from ``delta_rfun="auto"`` (communities) cannot be fitted to
    (T, d) inputs: fewer than 4 features or fewer than 40 timepoints (the
    shift null then has at least one lag in ``[L, T - L]``, and lags are
    reused when d exceeds their number; see ``summaries._shift_lags``).
    ``previous``: the number of communities found at the
    previous order when the d inputs are the community columns of an
    explicit ``Communities`` there (the automatic reducer keeps at least 4
    columns; see ``summaries._auto_emitter``).  ``training``: the T rows
    are the rows a causal call fits its reducers on (``fit_timepoints``).
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
            f"and {_AUTO_FEEDING_MIN_T} timepoints"
            f"{' to be fitted on (fit_timepoints)' if training else ''}; got "
            f"{d} feature{'' if d == 1 else 's'}{source} and "
            f"{'fit_timepoints' if training else 'T'} = {T}. " + alternatives
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
                          combine, feeding_function, ndims, weights_function=None,
                          later_T=None, causal=False):
    """
    Raise MemoryError, before anything is computed, for a streamed step of
    ``features`` that does not fit ``max_memory_gb`` with one row per block.
    Input widths follow ``_known_width`` from K.  After ``Communities``,
    whose output width is found by the fit, the width is unknown until it
    runs, and the later steps whose width follows from it are not checked
    here.  After a callable that is not a summary, the original K (the
    number of features of the data, not the callable's input width) is
    used (as ``estimate_memory_gb(..., streamed=True)`` uses K).  Every step is checked
    again, with its actual width, when it runs.  ``later_T``: the number of
    rows assumed for orders 2 and above (None: T); ``features`` gives the
    fewest rows such an order can have when feeding features are passed on
    at a lower rate, so that a step is refused here only if it cannot fit.
    ``causal``: the call has ``causal=True``, where ``wisfc`` with fixed
    subject weights is streamed too (``_streams``).
    """
    d = int(K)
    rows = {k: T if k == 1 or later_T is None else later_T for k in analysis}
    dense_analysis = not _builtin_kernel(weights_function)
    dense_feeding = not _builtin_kernel(feeding_function)
    for k in sorted(analysis):
        if d is not None:
            if _streams(analysis[k], cfun, combine, causal=causal):
                _streamed_block_elements(d, rows[k], S, max_memory_gb, k, "analysis",
                                         dense_analysis)
            if (k in fed and feeding_function is not eye_weights
                    and _is_summary(feeding[k])):
                _streamed_block_elements(d, rows[k], S, max_memory_gb, k, "feeding",
                                         dense_feeding)
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


# Automatic kernels (kernel parameters None with Gaussian kernels on both
# paths): the feeding kernel's standard deviation is this many
# autocorrelation widths of the step's input,
_AUTO_WIDTHS = 5
# at most this fraction of the input's rows (a Gaussian kernel of standard
# deviation sd spans about 3.5 sd rows, so the input holds at least about 28
# windows that do not overlap) and at least this many rows (the fixed
# feeding kernel of earlier versions);
_AUTO_MAX_FRACTION = 1 / 100
_AUTO_FEEDING_SD = 5.0
# the analysis kernel's is that or the Gaussian kernel's default (10 rows),
# whichever is larger;
_AUTO_ANALYSIS_SD = 10.0
# and features passes the feeding features on at every floor(sd / 2)-th row
# while at least this many rows remain
_AUTO_MIN_ROWS = 1000


def _autocorrelation_width(subjects):
    """
    Autocorrelation width of the columns, in rows: the lag at which the
    autocorrelation, averaged over every column of every subject, first
    falls below ``exp(-1/2)``, linearly interpolated between lags and at
    least 1 (white noise gives 1; a Gaussian autocorrelation with standard
    deviation w gives w).  Constant columns (the estimator's rule,
    ``helpers._zscore_columns``: constant up to round-off, whatever their
    value) are left out and NaN samples count as the column's mean.  Each
    column is divided by its largest absolute value first, so the result
    does not depend on the units of the data (no square overflows or
    underflows).  Lags beyond the largest automatic kernel
    are not examined: the result is at most ``T * _AUTO_MAX_FRACTION /
    _AUTO_WIDTHS + 1``.
    """
    threshold = np.exp(-0.5)
    T = subjects[0].shape[0]
    columns = []
    for x in subjects:
        x = np.asarray(x, dtype=float)
        # the largest absolute value of each column, NaN samples left out
        # (an all-NaN column gives -inf, without numpy's warning of nanmax)
        a = np.abs(x)
        top = (np.where(np.isnan(a), -np.inf, a).max(axis=0) if x.shape[0]
               else np.zeros(x.shape[1]))
        top = np.where(np.isfinite(top) & (top > 0), top, 1.0)
        # z-scores: NaN samples and constant columns give 0
        z = _zscore_columns(x / top)
        scale = np.sqrt((z * z).sum(axis=0))
        keep = scale > 0
        columns.append(z[:, keep] / scale[keep])
    n = sum(c.shape[1] for c in columns)
    last = int(T * _AUTO_MAX_FRACTION / _AUTO_WIDTHS) + 1
    if n == 0:
        return 1.0
    previous = 1.0
    for lag in range(1, min(last, T - 1) + 1):
        acf = sum(float(np.einsum("tk,tk->", c[lag:], c[:-lag])) for c in columns) / n
        if acf < threshold:
            return max(1.0, lag - 1 + (previous - threshold) / (previous - acf))
        previous = acf
    return float(last)


def _auto_sd(subjects):
    """
    Standard deviation, in rows, of the automatic feeding kernel of a step
    with these inputs: ``_AUTO_WIDTHS`` autocorrelation widths, at most
    ``_AUTO_MAX_FRACTION`` of the number of rows and at least
    ``_AUTO_FEEDING_SD``.
    """
    T = subjects[0].shape[0]
    longest = max(_AUTO_FEEDING_SD, T * _AUTO_MAX_FRACTION)
    return float(min(_AUTO_WIDTHS * _autocorrelation_width(subjects), longest))


def _automatic_paths(weights_function, delta_weights_function):
    """
    ``(analysis, feeding)``: whether kernel parameters of None on that path
    mean a Gaussian kernel whose width is set from each step's input
    (``_auto_sd``).  Feeding: the feeding kernel is ``gaussian_weights``
    (``delta_weights_function`` None or ``gaussian_weights``).  Analysis:
    ``weights_function`` is ``gaussian_weights`` and the feeding kernel is
    not ``eye_weights`` (the original pipeline keeps the kernel's default
    parameters).  On other paths None means the kernel function's own
    defaults.
    """
    feeding_function, _ = _feeding_kernel(delta_weights_function, None)
    return (
        weights_function is gaussian_weights and feeding_function is not eye_weights,
        feeding_function is gaussian_weights,
    )


def _auto_params(subjects, analysis, feeding):
    """
    ``(analysis params, feeding params)`` of a step with automatic kernels
    on the paths flagged True: Gaussian variances from ``_auto_sd`` of the
    step's inputs (None for a path that is not automatic).
    """
    if not (analysis or feeding):
        return None, None
    sd = _auto_sd(subjects)
    wide = max(sd, _AUTO_ANALYSIS_SD)
    return (
        {"var": wide ** 2} if analysis else None,
        {"var": sd ** 2} if feeding else None,
    )


def _at_every_timepoint(out, rows, T, chunk_elements=None):
    """
    An output computed at the timepoints ``rows`` (increasing indices),
    linearly interpolated to all ``T`` timepoints; after the last of
    ``rows`` its value is held (before the first, the first).  A computed
    row keeps its value exactly, and a timepoint strictly between two
    computed rows is NaN only where one of them is NaN (a NaN row never
    reaches a timepoint where it has no weight).  ``out``: an array or a
    list of arrays; a list is expanded in place, one array at a time, so
    each computed array is released once its full-length copy exists.

    The result is written into one preallocated array, ``chunk_elements``
    (default ``_EXPAND_ELEMENTS``) elements of rows at a time, so the work
    space beside it is a few such chunks (see ``_expansion_bytes``).
    """
    if isinstance(out, list):
        for n in range(len(out)):
            out[n] = _at_every_timepoint(out[n], rows, T, chunk_elements)
        return out
    out = np.asarray(out)
    rows = np.asarray(rows)
    if len(rows) == T:
        return out
    if len(rows) == 1:
        return np.repeat(out, T, axis=0)
    result = np.empty((T,) + out.shape[1:], dtype=np.result_type(out, np.float64))
    first, last = int(rows[0]), int(rows[-1])
    # held ends
    result[: first + 1] = out[0]
    result[last:] = out[-1]
    # strictly between two computed rows: both have non-zero weight
    width = max(1, int(np.prod(out.shape[1:], dtype=np.int64)))
    step = max(1, int(chunk_elements or _EXPAND_ELEMENTS) // width)
    shape = (-1,) + (1,) * (out.ndim - 1)
    for a in range(first + 1, last, step):
        ti = np.arange(a, min(a + step, last))
        i = np.searchsorted(rows, ti, side="right") - 1
        w = ((ti - rows[i]) / (rows[i + 1] - rows[i])).reshape(shape)
        chunk = result[a: a + len(ti)]
        np.multiply(out[i], 1 - w, out=chunk)
        chunk += out[i + 1] * w
    # the computed rows themselves, exactly
    result[rows] = out
    return result


def _expansion_bytes(outputs, computed_at, T, fits_elements=0):
    """
    Peak bytes while ``outputs`` (lists of arrays, one per order, computed at
    the timepoints ``computed_at``) are returned at all ``T`` timepoints by
    ``_at_every_timepoint``: every output at full length, the largest
    computed array not yet released, the interpolation's work space (4
    chunks) and ``fits_elements`` (the retained ``fits_`` and the formatted
    data, which the run holds to the end).  Returns (total bytes,
    bytes of the outputs at full length).
    """
    full = largest = 0
    seen = set()
    for out, at in zip(outputs, computed_at):
        for a in _as_list(out):
            a = np.asarray(a)
            if id(a) in seen:
                continue
            seen.add(id(a))
            width = a.size // max(1, a.shape[0])
            full += int(T) * width
            if len(at) != T:
                largest = max(largest, a.size)
    total = 8 * (full + largest + 4 * _EXPAND_ELEMENTS + int(fits_elements))
    return total, 8 * full


def _check_expansion(outputs, computed_at, T, max_memory_gb, fits_elements=0,
                     lower_bound=False):
    """MemoryError when returning ``outputs`` at all ``T`` timepoints would
    need more than ``max_memory_gb`` (see ``_expansion_bytes``)."""
    if max_memory_gb is None or all(len(at) == T for at in computed_at):
        return
    total, full = _expansion_bytes(outputs, computed_at, T, fits_elements)
    if total / _BYTES_PER_GB > max_memory_gb:
        raise MemoryError(_expansion_message(T, total, full, max_memory_gb, lower_bound))


def _expansion_message(T, total, full, max_memory_gb, lower_bound=False):
    return (
        f"Returning the features at all {T} timepoints needs "
        f"{'at least ' if lower_bound else ''}about {total / _BYTES_PER_GB:.3g} GB "
        f"({full / _BYTES_PER_GB:.3g} GB of outputs at full length, plus the "
        "rows being interpolated and the interpolation's work space), more than "
        f"max_memory_gb={max_memory_gb}. Orders 2 and above were computed at "
        "fewer timepoints (automatic kernels) and are interpolated to every "
        "timepoint. Reduce the outputs (rfun, delta_rfun), lower the order, or "
        "raise max_memory_gb."
    )


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
    causal=False,
    min_samples=None,
    fit_timepoints=None,
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
        Parameters of the analysis kernel.  None with ``gaussian_weights``:
        the variance is set from ``data`` (standard deviation ``max(sd,
        10)`` rows, with ``sd`` as for ``delta_weights_params``); None with
        another ``weights_function``, or with
        ``delta_weights_function=eye_weights``: the kernel's defaults (for
        example ``gaussian_params``).  A
        dict with integer keys raises ``ValueError`` (an order map, a dict
        mixing integer and other keys, or a dict with Python or NumPy
        boolean keys; the message says which): order maps of kernel
        parameters are accepted by ``features``.  Default: None.
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
        distinct event sets (one T x T array each for a kernel function of
        your own) are not included in the estimate of a materialized step; a streamed step counts them, the
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
        Parameters of the feeding kernel (keyword-only).  None with a
        Gaussian feeding kernel (``delta_weights_function`` None or
        ``gaussian_weights``): standard deviation ``sd = min(5 w, max(5, T
        / 100))`` rows, with ``w`` the autocorrelation width of ``data``
        and ``T`` its number of rows (see "Automatic kernels" in the Notes
        of ``features``); None with any other feeding kernel: that
        function's own defaults.  With ``delta_weights_function=None`` a
        dict holds the parameters of ``gaussian_weights`` (a dict without
        ``"var"`` raises ``ValueError``); with ``eye_weights``, which
        takes no parameters, they must be None (``ValueError``
        otherwise).  A dict with integer keys raises ``ValueError``, as for
        ``weights_params``.  This function computes one order at every row
        of ``data``; passing feeding features on at a lower rate is done by
        ``features``.  Default: None.
    order : int
        The order this step computes (keyword-only), used to resolve
        ``"auto"`` and order maps (whose keys may be any positive integers
        here) and recorded in summaries' ``fits_``.  Default: 1.
    causal : bool
        If True (keyword-only), row ``t`` of every output is computed from
        rows ``0..t`` of the data only: it is unchanged when later rows
        are replaced by any values, NaN and inf included, or removed, and
        with ``include_zero=False`` also when row ``t`` itself changes.
        Replacing later rows changes no bit, with every supported reducer.
        With the built-in kernel functions, removing them changes no bit
        either (every reducer and ``combine`` function and ``cfun=None``
        included): a row is computed with the same operations whatever
        the length of the series and whatever ``max_memory_gb`` (tested,
        also in configurations drawn at random, not proven: it holds when
        the linear-algebra library computes a row of a matrix product of
        a given shape from that row's operands alone).  With a kernel function of your own the
        columns of a chunk of kernel rows are found from its weights and
        can depend on the length; removal is then documented to 1e-9
        where the outputs depend continuously on the correlations (no
        reduction, ``"strength"``, the summaries, and linear models such
        as PCA once fitted); the eigenvector-based graph measures
        (``"eigenvector_centrality"``, the PageRank measures) can then
        differ by more at a correlation matrix whose leading eigenvalues
        coincide, and the t statistics of ``tstat_combine``, which divide
        by the spread of the subjects' correlations, where that spread is
        close to zero, agree to about 1e-9 of their magnitude instead;
        with ``cfun=None`` the steps smooth their input, and smoothed
        values then agree to round-off relative to the magnitude of the
        data (see ``timecorr``).  Default: False.  With
        ``causal=True``:

        - both paths use backward kernels (``include_timepoints`` may be
          left out; ``"forward"``/``"post"`` raises ``ValueError``), and
          the correlations are those of the causal estimator (see
          ``timecorr`` and ``wcorr``): infinite samples and samples above
          1e150 in magnitude are missing, like NaN;
        - the kernel parameters of both paths must be given
          (``weights_params`` and ``delta_weights_params``):
          widths are not chosen from the data.  ``eye_weights`` on either
          path raises ``ValueError`` (co-fluctuations use series z-scored
          over the whole recording);
        - a reducer that is fitted to data (a model such as ``"PCA"``, a
          summary, and ``"auto"``, the default of ``rfun`` and
          ``delta_rfun``) needs ``fit_timepoints``: it is fitted on the
          first ``fit_timepoints`` rows and then applied, unchanged, to
          every row.  None and the graph measures (``"strength"``,
          ``"eigenvector_centrality"``, ``"pagerank_centrality"``,
          ``"pagerank_centrality_abs"``) need no fit.  Every reducer (a
          model, a summary, a graph measure) reads the rows without any
          NaN or infinite value only: any other row (the undefined rows
          of the estimator; a smoothed row of ``cfun=None`` that is
          partly NaN) is left out of the fit and is a row of NaN in the
          reduced output.  A function of your
          own (as ``cfun``, ``combine``, ``rfun`` or ``delta_rfun``) is called as
          without ``causal``, with one ``UserWarning`` that the guarantee
          does not cover what it does; so is a model outside the list
          under ``fit_timepoints``, with one ``UserWarning`` of its own;
        - ``cfun=tc.wisfc`` raises ``ValueError`` (its default subject
          weights use the whole recording);
          ``functools.partial(tc.wisfc, subject_weights=...)`` is accepted;
        - ``events`` must be boundaries (features to segment raise
          ``ValueError``); that the boundaries are known at their own time
          is the caller's responsibility, and the warm-up starts again at
          each boundary.  Boundaries at or beyond the end of the data are
          ignored (times the series has not reached), and events that
          leave no timepoint with 3 samples give rows of NaN, not an
          error, so a prefix of a series runs with the events of the whole
          series;
        - a row with too few earlier samples (see ``min_samples``) is NaN
          in every column, not 0.  The NaN rows of an order's feeding
          features are missing data for the next order, so each order has
          more of them than the one before; graph measures are computed
          for the other rows.  A series of any length runs, and its output
          is the first rows of the output of a longer series;
        - a feature whose kernel-weighted standard deviation at a timepoint
          is below 1e-9 of the magnitude of its kernel-weighted mean has
          correlation 0 with every feature there (a column that is constant
          up to round-off; see ``wcorr``).
    min_samples : float or None
        With ``causal=True`` (keyword-only), a timepoint with fewer than 3
        observed timepoints in its kernel row is undefined (NaN).  A number
        (>= 1) is also the effective sample size ``(sum |w|)**2 / sum
        w**2`` a row needs over the observed timepoints it weights, on both
        paths (see ``timecorr``); a number that a built-in
        kernel of the call cannot reach with any number of earlier
        timepoints raises ``ValueError`` before anything is computed.
        Given without ``causal=True``, it raises ``ValueError``.  Default:
        None.
    fit_timepoints : int or None
        With ``causal=True`` (keyword-only), the number ``n`` of first rows
        that the reducers are fitted on: an integer from 1 to the number
        of timepoints.  Every reducer that is fitted to data is fitted to
        the correlations of rows ``0..n-1`` alone and then frozen:
        hypertools models (the components of ``"PCA"``), the pair weights
        and communities of ``WeightedStrength`` and ``Communities`` (with
        ``n=None``, the shift null runs on those rows), and the automatic
        reducers (``"auto"``), whose feeding step chooses between
        communities and weighted strength on those rows.  The rows
        without defined correlations (the warm-up, see ``min_samples``) are
        left out of the fits; a model needs 2 rows with defined
        correlations there, a summary 1, the shift null 40 rows of
        input.  The fits do not change, bit for bit, whatever follows row
        ``n-1``.  Every row is then reduced with them: row ``t >= n`` is
        unchanged (bit for bit; under removal with the built-in kernel
        functions, see ``causal``) when rows after
        ``t`` are replaced or removed, and a row ``t < n`` (in sample: the
        fits have read rows after it) when rows from ``n`` on are replaced
        or removed.  ``n`` equal to the number of timepoints fits on every
        row.  A model must be a linear map applied to each row by itself.  The
        guarantee covers ``"PCA"``, ``"IncrementalPCA"``, ``"TruncatedSVD"``,
        ``"FastICA"`` and ``"FactorAnalysis"`` (scikit-learn's classes of those
        names, by exact class: a subclass is not one).  Any other model whose
        fitted estimator has ``components_``, reproduces its training rows to
        1e-9 and reduces rows one at a time as it reduces them together
        (``"MiniBatchDictionaryLearning"``, a class of your own) is used with
        one ``UserWarning`` per call that the guarantee does not cover what its
        transform does: those checks read a few training rows and cannot show
        how it reduces later rows.  A model without ``components_``, or that
        fails a check, raises ``ValueError`` (``"UMAP"``, ``"KernelPCA"``,
        ``"Isomap"``, ``"MDS"``, ``"TSNE"``, ``"SpectralEmbedding"``,
        ``"LocallyLinearEmbedding"``: the transform of some of them depends on
        the rows it is given together).
        A model asked for as many components as its input
        has columns, or more, changes nothing (zero columns are appended
        up to ``ndims``), whatever the number of training
        rows; the number asked for is ``n_components`` given with the
        model (in its ``kwargs``, or of a model object; it decides
        whether a model is fitted and the width of a fitted model's
        output, not the number of zero columns: the widths are those of
        the call without ``causal``), otherwise ``ndims``.  Asked for
        fewer components than columns
        but more than there are defined training rows, a model is fitted
        with as many components as those rows, and zero columns are
        appended for the others, with a warning.
        ``"FastICA"`` (when it is fitted: asked for fewer components than
        columns) is given the columns that vary over the training rows
        first and is asked for no more components than the rank of the
        centered training rows (zero columns are appended for the
        others, with a warning; ``ValueError`` when no column varies):
        scikit-learn sets the sign of a component from its loading on
        the first column, which for vectorized correlations is a
        diagonal entry, 1 in every row, and the fit held NaN where that
        loading was exactly 0; and it divides by the singular values of
        the rows, so a component beyond their rank did not reproduce the
        training rows.
        ``"PCA"`` is fitted with ``svd_solver="full"``
        unless a model dict gives a solver, because the randomized solver
        scikit-learn chooses for large inputs does not reproduce its
        rows.  An output column of a model that is constant on the
        training rows up to round-off (a range of at most 1e-9 of its
        largest magnitude, or 1e-12 of the largest magnitude of the
        output: for example a principal component along which those rows
        do not vary) is frozen: at every row, a value within that
        tolerance of the column's training mean is returned as exactly
        that mean, so the next order does not correlate round-off.  Output
        columns of a summary that are constant up to round-off
        are not set to their mean (see ``WeightedStrength``), and no
        warning reports them.  Not used by reducers that need no fit;
        required (``ValueError``) when a reducer is fitted.  Given without
        ``causal=True``, it raises ``ValueError``.  Default: None.

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
    ValueError
        With ``causal=True``, before anything is computed: for the
        arguments listed under ``causal`` that have no causal form, and
        for a ``min_samples`` that is not None or a finite number >= 1 or
        that a built-in kernel of the call cannot reach; for
        ``min_samples`` given without ``causal=True``.
    ValueError
        With ``causal=True``, for an ``rfun`` or ``delta_rfun`` that is
        fitted to data (the defaults included) when ``fit_timepoints`` is
        not given; for a ``fit_timepoints`` that is not an integer from 1
        to the number of timepoints, or given without ``causal=True``;
        when the first ``fit_timepoints`` rows hold no row with defined
        correlations for a fit (2 for a model), or fewer than 40 rows for
        the default feeding reducer or a shift null; and for a model that
        is not a linear map applied row by row (see ``fit_timepoints``).
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

    A causal step (row ``t`` uses rows ``0..t`` only; the first 2 rows have
    fewer than 3 samples and are NaN):

    >>> import numpy as np
    >>> X1, D1 = two_path_step(data[0], causal=True, cfun=tc.autofc,
    ...                        rfun="strength", delta_rfun="strength",
    ...                        weights_params={"var": 100},
    ...                        delta_weights_params={"var": 25})
    >>> X1.shape, D1.shape, int(np.isnan(D1).all(axis=1).sum())
    ((300, 24), (300, 24), 2)

    With the default reducers, fitted on the first 200 rows and applied to
    every row:

    >>> X1, D1 = two_path_step(data[0], causal=True, cfun=tc.autofc,
    ...                        weights_params={"var": 100},
    ...                        delta_weights_params={"var": 25},
    ...                        fit_timepoints=200)
    >>> X1.shape, D1.shape
    ((300, 24), (300, 6))
    """
    if (
        not isinstance(order, numbers.Integral)
        or isinstance(order, bool)
        or order < 1
    ):
        raise ValueError(f"order must be a positive integer; got {order!r}.")
    order = int(order)
    min_samples = _check_min_samples(min_samples, causal)
    _refuse_params_map(weights_params, "weights_params")
    _refuse_params_map(delta_weights_params, "delta_weights_params")
    analysis, feeding = _resolve_reducers(rfun, delta_rfun, [order], None)
    ndims = _check_ndims(ndims)
    _check_ndims_steps(
        ndims, [(analysis[order], "rfun", order), (feeding[order], "delta_rfun", order)]
    )
    auto_analysis, auto_feeding = _automatic_paths(weights_function, delta_weights_function)
    feeding_function, feeding_params = _feeding_kernel(
        delta_weights_function, delta_weights_params
    )
    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    fit_timepoints = _check_fit_timepoints(fit_timepoints, causal,
                                           subjects[0].shape[0])
    estimator = {}
    if causal:
        _check_causal(
            function="two_path_step", include_timepoints=include_timepoints,
            weights_function=weights_function,
            delta_weights_function=delta_weights_function, cfun=cfun, rfun=rfun,
            delta_rfun=delta_rfun, analysis_rfun=analysis, feeding_rfun=feeding,
            analysis_params={order: weights_params},
            feeding_params={order: delta_weights_params}, events=[events],
            T=subjects[0].shape[0], n_subjects=len(subjects),
            include_zero=include_zero, min_samples=min_samples,
            fit_timepoints=fit_timepoints, combine=combine,
        )
        include_timepoints = "backward"
        estimator = {"causal": True, "min_samples": min_samples}
    analysis_auto, feeding_auto = _auto_params(
        subjects,
        auto_analysis and weights_params is None,
        auto_feeding and delta_weights_params is None,
    )
    if analysis_auto is not None:
        weights_params = analysis_auto
    if feeding_auto is not None:
        feeding_params = feeding_auto
    resolved = resolve_events(
        events, subjects[0].shape[0], len(subjects), events_params, causal=causal
    )
    step = dict(
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
        **estimator,
    )
    frozen = None
    if causal and (_needs_fit(analysis[order]) or _needs_fit(feeding[order])):
        # the reducers are fitted on the first fit_timepoints rows alone,
        # then applied to every row
        frozen = {}
        with _fit_pass():
            _two_path_step(
                [x[:fit_timepoints] for x in subjects],
                events=resolve_events(events, fit_timepoints, len(subjects), None,
                                      causal=True),
                record=frozen, **step,
            )
    X_next, D_next = _two_path_step(subjects, events=resolved, frozen=frozen, **step)
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
    frozen=None,
    record=None,
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

    With ``causal`` (in ``analysis``): the reducers are fitted to the rows
    given, which are the training rows of a causal fit, and ``record`` (a
    dict) receives what reduced each path (``record["analysis"]``,
    ``record["feeding"]``: see ``_apply_reducer``); or ``frozen``, such a
    dict of an earlier call, reduces these rows without any fit.
    """
    if delta_rfun is None:
        delta_rfun = analysis.get("rfun", "PCA")
    T, d = subjects[0].shape
    causal = analysis.get("causal", False)
    frozen = frozen or {}
    if feed and "feeding" not in frozen:
        _check_auto_feeding(delta_rfun, d, T, order, previous_communities,
                            training=causal)
    analysis_streamed = _streams(
        analysis.get("rfun", "PCA"), analysis.get("cfun", isfc),
        analysis.get("combine", null_combine), causal=causal,
    )
    feeding_streamed = feeding_function is not eye_weights and _is_summary(delta_rfun)
    n_outputs = 2 if feed and not (analysis_streamed or feeding_streamed) else 1
    kernel_feeding = feed and feeding_function is not eye_weights
    dense_analysis = _dense_analysis(analysis.get("weights_function"),
                                     analysis.get("cfun", isfc), causal)
    dense_feeding = feed and not _builtin_kernel(feeding_function)
    dense = dense_analysis or dense_feeding
    S = len(subjects)
    v = d * (d + 1) // 2
    analysis_rfun = analysis.get("rfun", "PCA")
    events = analysis.get("events")
    # each path's event kernels, as that path's kernel is held (T x T arrays
    # only for a kernel or correlation function of your own)
    analysis_kernels = _event_kernel_elements(T, S, events, dense_analysis)
    feeding_kernels = _event_kernel_elements(T, S, events, dense_feeding)
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
                _budget_left(max_memory_gb, retained + analysis_kernels + fits,
                             d, T, S, order, "streamed analysis step",
                             dense_analysis),
                order, "analysis", dense_analysis,
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
                    alone = _step_gb(d, T, S, 1, kernel_feeding, dense)
                    held_gb = 8 * (retained_m + fits) / _BYTES_PER_GB
                    streamable = alone + held_gb <= max_memory_gb
            _check_step_memory(d, T, S, n_outputs, max_memory_gb, order,
                               kernel_feeding, retained_m + fits,
                               _stream_hint(order) if streamable else "", dense=dense,
                               combine=analysis.get("combine", null_combine))
        if feed and feeding_streamed:
            _streamed_block_elements(
                d, T, S,
                _budget_left(max_memory_gb, retained + held + feeding_kernels + fits,
                             d, T, S, order, "streamed feeding step", dense_feeding),
                order, "feeding", dense_feeding,
            )
        elif feed:
            _check_step_memory(d, T, S, n_outputs, max_memory_gb, order,
                               kernel_feeding, retained_m + fits, dense=dense)
            if _needs_shift_null(delta_rfun):
                _budget_left(
                    max_memory_gb, retained + held + fits + S * T * v, d, T, S, order,
                    "shift null of the feeding step",
                )
    X_next = _analysis_path(
        subjects, max_memory_gb=max_memory_gb, n_outputs=n_outputs, order=order,
        append_fit=True, kernel_feeding=kernel_feeding, held_elements=retained,
        reducers=(delta_rfun,) + reducers, input_elements=int(input_elements),
        frozen=frozen.get("analysis"), record=record,
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
        causal=causal,
        min_samples=analysis.get("min_samples"),
        frozen=frozen.get("feeding"),
        record=record,
    )
    return X_next, D_next


@contextlib.contextmanager
def _fit_pass():
    """
    The pass of a causal call that fits the reducers on the first
    ``fit_timepoints`` rows: the estimator's warnings about its rows are
    not issued (the pass over every row, which follows, issues them).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(Subjects? [0-9, ]+: )?[0-9]+ of [0-9]+ timepoints have",
        )
        yield


def _check_causal(*, function, include_timepoints, weights_function,
                  delta_weights_function, cfun, rfun, delta_rfun, analysis_rfun,
                  feeding_rfun, analysis_params, feeding_params, events, T,
                  n_subjects, include_zero, min_samples, fit_timepoints=None,
                  combine=null_combine):
    """
    The checks of ``features`` and ``two_path_step`` with ``causal=True``
    (``function``: the name of the caller, for the messages), before
    anything is computed.  ``analysis_rfun`` and ``feeding_rfun`` are the
    resolved reducers by order, ``analysis_params`` and ``feeding_params``
    the kernel parameters GIVEN for each order (None: not given),
    ``events`` the specifications to check (one per order that has one).

    Raises ``ValueError`` for what has no causal form (a forward kernel,
    ``eye_weights``, kernel parameters that are not given, the default
    subject weights of ``wisfc``, events given as features, a
    ``min_samples`` no row of a built-in kernel can reach, a reducer that
    is fitted to data without ``fit_timepoints``); warns once when ``cfun``
    or a reducer is a function of the user's.
    """
    side = str(include_timepoints).lower()
    if side in ("forward", "post"):
        raise ValueError(
            f"causal=True: include_timepoints={include_timepoints!r} weights "
            "later timepoints.  Leave include_timepoints out (causal=True uses "
            "'backward' on both paths), or pass causal=False."
        )
    for name, kernel in (("weights_function", weights_function),
                         ("delta_weights_function", delta_weights_function)):
        if kernel is eye_weights:
            raise ValueError(
                f"causal=True: {name}=eye_weights gives co-fluctuations of series "
                "z-scored over the whole recording, which use later timepoints.  "
                f"Pass a kernel with a width, for example {name}="
                f"tc.gaussian_weights with {name.replace('function', 'params')}="
                "{'var': 25}."
            )
    for name, params in (("weights_params", analysis_params),
                         ("delta_weights_params", feeding_params)):
        for k in sorted(params):
            if params[k] is None:
                raise ValueError(
                    f"causal=True: {name} must be given for every order (kernel "
                    "widths are not chosen from the data, which would read later "
                    f"timepoints); order {k} has none.  Pass for example "
                    f"{name}={{'var': 25}} for a Gaussian kernel, {{}} for a "
                    "kernel without parameters, or in features an order map "
                    "that names every order."
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
    for spec in events:
        if _has_features(_parse_spec(spec, T, n_subjects, True)):
            raise ValueError(
                "causal=True: events given as features are segmented from the "
                "whole recording.  Pass the boundaries (the timepoints where a "
                "new block starts), for example events=[100, 250]."
            )
    for name, given, resolved in (("rfun", rfun, analysis_rfun),
                                  ("delta_rfun", delta_rfun, feeding_rfun)):
        for k in sorted(resolved):
            _check_linear_model(resolved[k], f"{name} at order {k}")
            if _needs_fit(resolved[k]) and fit_timepoints is None:
                raise ValueError(
                    f"causal=True: {name}={_requested(given, k)!r} (order {k}) is "
                    "fitted to data and needs fit_timepoints=n: the reducers "
                    "are then fitted on the first n rows and applied to every "
                    "row.  Or pass rfun and delta_rfun as None or a graph "
                    "measure such as 'strength', which need no fit (the "
                    "defaults are fitted)."
                )
    if min_samples is not None:
        # a number no kernel row can reach, whatever the length of the data
        for kernel, params in ((weights_function, analysis_params),
                               (delta_weights_function, feeding_params)):
            kernel = gaussian_weights if kernel is None else kernel
            for k in sorted(params):
                lag_kernel = _lag_kernel(2, kernel, params[k], "backward", include_zero)
                if lag_kernel is not None:
                    _check_reachable(lag_kernel, min_samples)
    builtin = cfun is None or cfun in (isfc, autofc) or _fixed_weights_wisfc(cfun)
    theirs = [name for name, ours in (
        ("cfun", builtin),
        ("combine", _builtin_combine(combine)),
        ("rfun", not any(_theirs(r) for r in analysis_rfun.values())),
        ("delta_rfun", not any(_theirs(r) for r in feeding_rfun.values())),
    ) if not ours]
    if theirs:
        one = len(theirs) == 1
        _warn_user(
            f"causal=True: {', '.join(theirs[:-1])}{'' if one else ' and '}"
            f"{theirs[-1]} {'is a function' if one else 'are functions'} of your "
            "own; the guarantee that row t does not depend on later rows does "
            f"not cover what {'it does' if one else 'they do'}."
        )
    _warn_uncovered_models([
        (_uncovered_model(resolved[k]), f"{name} at order {k}")
        for name, resolved in (("rfun", analysis_rfun), ("delta_rfun", feeding_rfun))
        for k in sorted(resolved)
    ])


def _check_refit_every(refit_every, causal, fit_timepoints):
    """
    ``refit_every`` as None or an int: ``ValueError`` unless it is None or
    an integer >= 1 given with ``causal=True`` and ``fit_timepoints``.
    """
    if refit_every is None:
        return None
    if (
        isinstance(refit_every, (bool, np.bool_))
        or not isinstance(refit_every, numbers.Integral)
        or refit_every < 1
    ):
        raise ValueError(
            "refit_every must be None or a positive integer (the number of "
            f"rows between two fits of the reducers); got {refit_every!r}."
        )
    if not causal or fit_timepoints is None:
        raise ValueError(
            f"refit_every={refit_every!r} applies with causal=True and "
            "fit_timepoints=n only (the reducers are fitted on the first n "
            f"rows, then again every {refit_every} rows, each time on all "
            "rows before); pass both, or leave refit_every out."
        )
    return int(refit_every)


def _theirs(reducer):
    """True for a reducer that is a function of the user's (not a summary)."""
    return callable(reducer) and not isinstance(reducer, _Summary)


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
               requested_delta, feeding_fit=None):
    """
    The ``return_info`` entries of order ``k`` of ``features``: one dict for
    the analysis path and, when it was computed, one for the feeding path.
    ``feeding_fit``: the fit record of the feeding step when it is not the
    last of ``feeding_rfun.fits_`` for this order (a frozen fit).
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
        "output_width": _width(delta), "labels": None,
    }
    fit = feeding_fit
    if isinstance(feeding_rfun, Communities):
        # this step's fit (the same object may also reduce the analysis path)
        if fit is None:
            fit = next(f for f in reversed(feeding_rfun.fits_)
                       if f["order"] == k and f["path"] == "feeding")
        if fit.get("fallback") is None:
            entry["labels"] = fit["labels"].tolist()
    if getattr(feeding_rfun, "_from_auto", False):
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


def _constant_columns(out, causal=False):
    """
    The columns of an output that are constant over time
    (``np.ptp(x, axis=0) == 0``), as a list of ints; for a list of arrays,
    one such list per array.  ``causal``: over the rows that are not
    entirely NaN (the warm-up rows of the causal estimator are left out; an
    output without any other row has no constant column).
    """
    if isinstance(out, list):
        return [_constant_columns(x, causal) for x in out]
    x = np.asarray(out)
    if causal:
        x = x[~np.isnan(x).all(axis=1)]
        if not x.shape[0]:
            return []
    return np.flatnonzero(np.ptp(x, axis=0) == 0).tolist()


def _warmup_rows(out):
    """
    The number of rows of an output that are entirely NaN (the rows the
    causal estimator leaves undefined), as an int; for a list of arrays,
    one int per array.
    """
    if isinstance(out, list):
        return [_warmup_rows(x) for x in out]
    return int(np.isnan(np.asarray(out)).all(axis=1).sum())


def _check_events_order_zero(events, T, n_subjects, events_params, causal=False):
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
    parsed = _parse_spec(events, T, n_subjects, causal)
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
    causal=False,
    min_samples=None,
    fit_timepoints=None,
    refit_every=None,
    return_model=False,
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

    With the defaults, the kernels of each order are Gaussians whose width
    is set from that order's input: the feeding kernel's standard deviation
    is 5 autocorrelation widths of the input (at most a hundredth of its
    rows, but at least 5 rows, which wins below 500 rows) and the analysis kernel's is that or 10 rows,
    whichever is larger (see "Automatic kernels" in the Notes).  Each
    ``D_k`` holds one column per community of that order's input features
    (``Communities()``: the mean squared correlation within the community,
    with the number of communities counted with a shift null), ``X_1`` is
    reduced with PCA and ``X_k`` for ``k >= 2`` with
    ``WeightedStrength(signed=True)`` (two columns per input feature).  The
    default feeding reducer needs at least 4 features and 40 timepoints at
    every order whose feeding step is computed (orders below
    ``order``, and ``order`` itself with ``return_delta=True``); otherwise
    ``ValueError`` names the alternatives: ``delta_rfun="weighted_strength"``
    (one column per feature, any number of features and timepoints) or the
    original pipeline.  At a default feeding step whose communities would
    emit fewer than 4 columns, the step emits weighted strength of the same
    correlations instead (one column per input feature; see
    ``delta_rfun``), so the feeding width never drops below 4.  To keep the
    community columns of an order whatever their number, give that order's
    reducer (for example ``delta_rfun={3: "communities"}``); on inputs
    without structure it then emits two communities about 1 time in 5 (17
    to 19 of 96 noise inputs had an eigenvalue above the null).  This also
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
    instance allow fitting on other data.  With ``causal=True`` and
    ``fit_timepoints=n`` every reducer is fitted on the first ``n`` rows
    only: the pipeline first runs on those rows alone, order by order (each
    order's fit reads the feeding features of those rows), and then on
    every row with the fits frozen (see ``fit_timepoints`` and
    ``FeaturesModel``).

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
        kernel parameters of that order (a dict, or None for the default),
        for example ``{1: {"var": 100}, 2: {"var": 400}}``; orders missing
        from it use the default.  A dict without
        integer keys (or an empty dict) is the parameters of every order; a
        dict that mixes integer and other keys, has boolean keys (Python or
        NumPy) or keys outside ``1..order``, or maps an order to a value that
        is neither a dict nor None, raises ``ValueError``.  A dict whose keys
        are all integers is always read as an order map, so the parameters
        of a custom kernel that are keyed by integers must be wrapped in one
        (for example ``{1: params, 2: params}``).  Only the outer dict is
        examined: each value is passed to the kernel unchanged, whatever its
        keys.  None (also for an order missing from an order map, or mapped
        to None) with ``gaussian_weights`` sets the variance of that order
        from its input (see "Automatic kernels" in the Notes); with another
        ``weights_function``, or with
        ``delta_weights_function=tc.eye_weights`` (the original pipeline),
        None is the kernel's default parameters (``gaussian_params``,
        variance 100, for ``gaussian_weights``).  Default:
        ``gaussian_weights`` with ``weights_params=None``.
    cfun : function
        Analysis-path correlation function (``isfc``, ``wisfc``, ``autofc``).
        Default: ``isfc``, which correlates each subject's features with the
        other subjects' and so reflects structure shared across subjects;
        ``autofc`` computes within-subject correlations.  On
        ``simulate_nested`` data (108 nodes, T 1000, 6 subjects in one call,
        12 seeds, other arguments default), ``h`` was decoded from ``X[2]``
        with ``autofc`` (score .552 +- .023, control -.008 +- .025) but not
        with ``isfc`` (-.031 +- .033, control .059 +- .020) when the order-1
        couplings are drawn per subject (``shared="order2"``), and with both
        when they are shared (``shared="order1"``: ``isfc`` .652 +- .026,
        control .078 +- .010; ``autofc`` .700 +- .012, control -.008 +-
        .025); see ``tests/test_nested_recovery.py``.  The feeding path
        does not use ``cfun``: ``D[2]`` gave .560 +- .020 (control .002 +-
        .025) and, with shared order-1 couplings, .596 +- .018 (control
        .010 +- .025).
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
        Default: "all".  A one-sided kernel's well-conditioned estimates do
        not depend on the timepoints it excludes; estimates near the
        estimator's zero-variance threshold can (see ``timecorr``).
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
        T array each for a kernel function of your own) are not included in
        the estimate of a step that holds its correlations; a streamed step counts them and is checked before
        its kernels are built.  Every step, streamed or not, on both paths,
        also counts what the run retains: the outputs of earlier orders
        (``X_1 .. X_{k-1}``; ``D_1 .. D_{k-1}`` when ``return_delta=True``,
        otherwise only the feeding features the order reads, as the others
        are released), each array once (views of one base array by their
        sizes, at most the base's), and the (K, K) arrays kept in the
        ``fits_`` of the summary objects the run holds; a streamed feeding
        step also counts the analysis results of its order.  A step that
        holds its correlations does not count its own input (the previous
        order's feeding features) twice: its estimate already includes it.  The
        temporaries of the built-in ``combine`` functions are counted; the
        memory of a ``combine`` function of your own cannot be counted.  The
        input data and the memory of reducers are not counted.  None checks nothing (a streamed step then uses the
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
        Parameters of the feeding kernel (keyword-only).  None with a
        Gaussian feeding kernel (``delta_weights_function`` None or
        ``gaussian_weights``): at each order the variance is set from that
        order's input, as described under "Automatic kernels" in the Notes;
        None with any other feeding kernel: that function's own defaults.
        With ``delta_weights_function=None`` a dict holds the parameters of
        ``gaussian_weights`` (for example ``{"var": 25}``, standard
        deviation 5 samples, at every order; a dict without ``"var"``
        raises ``ValueError``); with ``eye_weights``, which takes no
        parameters, they must be None (``ValueError`` otherwise).  An order
        map, as for ``weights_params``, sets the feeding kernel's parameters
        order by order; orders missing from it (or mapped to None) use the
        default (with ``eye_weights`` every value must be None).  The shift
        null of a feeding step reads that order's standard deviation.
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
        ``"too_few_community_columns"`` or ``"no_null_exceedance"``).
        Feeding steps have ``"labels"``: when the step's output columns are
        the communities of a ``Communities`` object (the default, or an
        object given in ``delta_rfun``), a list with the community (an int)
        of each input column, the same for every subject; communities are
        numbered in the order of their first column, and the output has
        one column per community of two or more input columns, in that
        order.  ``"labels"`` is None when the step fell back
        (``"fallback"``), was reduced in another way, or was reduced with
        the name ``"communities"`` (whose fit is not kept).  Every entry
        has ``"constant_columns"``: the columns of the step's output that
        are constant over time (``np.ptp(x, axis=0) == 0``) in the arrays
        the call returns (for outputs computed at a lower rate, after the
        interpolation to all timepoints), as a list of ints, or one such
        list per subject when the output is a list; for feeding features
        that are not returned (``return_delta=False``), in the rows
        computed.  No arrays: numbers, strings, None and lists of ints.
        With ``causal=True`` every entry also has ``"weights_params"`` (a
        dict: the kernel parameters of that step) and ``"warmup_rows"`` (the
        number of rows of the step's output that are entirely NaN, an int,
        or one int per subject when the output is a list), and
        ``"constant_columns"`` is taken over the other rows; the entry of
        a step whose reducer was fitted has ``"fit_timepoints"`` (the
        number of rows it was fitted on), ``"refit_rows"`` (the rows of
        the fits: ``[fit_timepoints]``, and with ``refit_every`` every
        row at which the reducers were fitted again) and
        ``"refit_kept_previous"`` (those of them whose fit could not
        have the output columns of the first fit, so that the fit before
        was kept, of every order; one ``UserWarning`` names them; see
        ``refit_every``), and
        what describes a fit (``"labels"``, ``"n_communities"``,
        ``"fallback"``, ...) is that of the fit on those rows (with
        ``refit_every``: of the last fit).  When a later fit took another
        number of clusters than the ``n`` of a ``Communities`` object (see
        ``refit_every``), ``"effective"`` shows the object with that
        number.
        Default: False.
    causal : bool
        If True (keyword-only), row ``t`` of every output is computed from
        rows ``0..t`` of the data only: it is unchanged when later rows
        are replaced by any values, NaN and inf included, or removed, and
        with ``include_zero=False`` also when row ``t`` itself changes.
        Replacing later rows changes no bit, with every supported reducer.
        With the built-in kernel functions, removing them changes no bit
        either (every reducer and ``combine`` function and ``cfun=None``
        included): a row is computed with the same operations whatever
        the length of the series and whatever ``max_memory_gb`` (tested,
        also in configurations drawn at random, not proven: it holds when
        the linear-algebra library computes a row of a matrix product of
        a given shape from that row's operands alone).  With a kernel function of your own the
        columns of a chunk of kernel rows are found from its weights and
        can depend on the length; removal is then documented to 1e-9
        where the outputs depend continuously on the correlations (no
        reduction, ``"strength"``, the summaries, and linear models such
        as PCA once fitted); the eigenvector-based graph measures
        (``"eigenvector_centrality"``, the PageRank measures) can then
        differ by more at a correlation matrix whose leading eigenvalues
        coincide, and the t statistics of ``tstat_combine``, which divide
        by the spread of the subjects' correlations, where that spread is
        close to zero, agree to about 1e-9 of their magnitude instead;
        with ``cfun=None`` the steps smooth their input, and smoothed
        values then agree to round-off relative to the magnitude of the
        data (see ``timecorr``).  Default: False.  With
        ``causal=True``:

        - both paths use backward kernels (``include_timepoints`` may be
          left out; ``"forward"``/``"post"`` raises ``ValueError``), and
          the correlations are those of the causal estimator (see
          ``timecorr`` and ``wcorr``): infinite samples and samples above
          1e150 in magnitude are missing, like NaN;
        - the kernel parameters of both paths must be given
          (``weights_params`` and ``delta_weights_params``, for every order when they are order maps):
          widths are not chosen from the data.  ``eye_weights`` on either
          path raises ``ValueError`` (co-fluctuations use series z-scored
          over the whole recording);
        - a reducer that is fitted to data (a model such as ``"PCA"``, a
          summary, and ``"auto"``, the default of ``rfun`` and
          ``delta_rfun``) needs ``fit_timepoints``: it is fitted on the
          first ``fit_timepoints`` rows and then applied, unchanged, to
          every row.  None and the graph measures (``"strength"``,
          ``"eigenvector_centrality"``, ``"pagerank_centrality"``,
          ``"pagerank_centrality_abs"``) need no fit.  Every reducer (a
          model, a summary, a graph measure) reads the rows without any
          NaN or infinite value only: any other row (the undefined rows
          of the estimator; a smoothed row of ``cfun=None`` that is
          partly NaN) is left out of the fit and is a row of NaN in the
          reduced output.  A function of your
          own (as ``cfun``, ``combine``, ``rfun`` or ``delta_rfun``) is called as
          without ``causal``, with one ``UserWarning`` that the guarantee
          does not cover what it does; so is a model outside the list
          under ``fit_timepoints``, with one ``UserWarning`` of its own;
        - ``cfun=tc.wisfc`` raises ``ValueError`` (its default subject
          weights use the whole recording);
          ``functools.partial(tc.wisfc, subject_weights=...)`` is accepted;
        - ``events`` must be boundaries (features to segment raise
          ``ValueError``); that the boundaries are known at their own time
          is the caller's responsibility, and the warm-up starts again at
          each boundary.  Boundaries at or beyond the end of the data are
          ignored (times the series has not reached), and events that
          leave no timepoint with 3 samples give rows of NaN, not an
          error, so a prefix of a series runs with the events of the whole
          series;
        - a row with too few earlier samples (see ``min_samples``) is NaN
          in every column, not 0.  The NaN rows of an order's feeding
          features are missing data for the next order, so each order has
          more of them than the one before; graph measures are computed
          for the other rows.  A series of any length runs, and its output
          is the first rows of the output of a longer series;
        - a feature whose kernel-weighted standard deviation at a timepoint
          is below 1e-9 of the magnitude of its kernel-weighted mean has
          correlation 0 with every feature there (a column that is constant
          up to round-off; see ``wcorr``).
    min_samples : float or None
        With ``causal=True`` (keyword-only), a timepoint with fewer than 3
        observed timepoints in its kernel row is undefined (NaN).  A number
        (>= 1) is also the effective sample size ``(sum |w|)**2 / sum
        w**2`` a row needs over the observed timepoints it weights, on both
        paths and at every order (see ``timecorr``); a number that a built-in
        kernel of the call cannot reach with any number of earlier
        timepoints raises ``ValueError`` before anything is computed.
        Given without ``causal=True``, it raises ``ValueError``.  Default:
        None.
    fit_timepoints : int or None
        With ``causal=True`` (keyword-only), the number ``n`` of first rows
        that the reducers are fitted on: an integer from 1 to the number
        of timepoints.  Every reducer that is fitted to data is fitted to
        the correlations of rows ``0..n-1`` alone and then frozen:
        hypertools models (the components of ``"PCA"``), the pair weights
        and communities of ``WeightedStrength`` and ``Communities`` (with
        ``n=None``, the shift null runs on those rows), and the automatic
        reducers (``"auto"``), whose feeding step chooses between
        communities and weighted strength on those rows.  The rows
        without defined correlations (the warm-up, see ``min_samples``) are
        left out of the fits; a model needs 2 rows with defined
        correlations there, a summary 1, the shift null 40 rows of
        input.  The fits do not change, bit for bit, whatever follows row
        ``n-1``.  Every row is then reduced with them: row ``t >= n`` is
        unchanged (bit for bit; under removal with the built-in kernel
        functions, see ``causal``) when rows after
        ``t`` are replaced or removed, and a row ``t < n`` (in sample: the
        fits have read rows after it) when rows from ``n`` on are replaced
        or removed.  ``n`` equal to the number of timepoints fits on every
        row.  A model must be a linear map applied to each row by itself.  The
        guarantee covers ``"PCA"``, ``"IncrementalPCA"``, ``"TruncatedSVD"``,
        ``"FastICA"`` and ``"FactorAnalysis"`` (scikit-learn's classes of those
        names, by exact class: a subclass is not one).  Any other model whose
        fitted estimator has ``components_``, reproduces its training rows to
        1e-9 and reduces rows one at a time as it reduces them together
        (``"MiniBatchDictionaryLearning"``, a class of your own) is used with
        one ``UserWarning`` per call that the guarantee does not cover what its
        transform does: those checks read a few training rows and cannot show
        how it reduces later rows.  A model without ``components_``, or that
        fails a check, raises ``ValueError`` (``"UMAP"``, ``"KernelPCA"``,
        ``"Isomap"``, ``"MDS"``, ``"TSNE"``, ``"SpectralEmbedding"``,
        ``"LocallyLinearEmbedding"``: the transform of some of them depends on
        the rows it is given together).
        A model asked for as many components as its input
        has columns, or more, changes nothing (zero columns are appended
        up to ``ndims``), whatever the number of training
        rows; the number asked for is ``n_components`` given with the
        model (in its ``kwargs``, or of a model object; it decides
        whether a model is fitted and the width of a fitted model's
        output, not the number of zero columns: the widths are those of
        the call without ``causal``), otherwise ``ndims``.  Asked for
        fewer components than columns
        but more than there are defined training rows, a model is fitted
        with as many components as those rows, and zero columns are
        appended for the others, with a warning.
        ``"FastICA"`` (when it is fitted: asked for fewer components than
        columns) is given the columns that vary over the training rows
        first and is asked for no more components than the rank of the
        centered training rows (zero columns are appended for the
        others, with a warning; ``ValueError`` when no column varies):
        scikit-learn sets the sign of a component from its loading on
        the first column, which for vectorized correlations is a
        diagonal entry, 1 in every row, and the fit held NaN where that
        loading was exactly 0; and it divides by the singular values of
        the rows, so a component beyond their rank did not reproduce the
        training rows.
        ``"PCA"`` is fitted with ``svd_solver="full"``
        unless a model dict gives a solver, because the randomized solver
        scikit-learn chooses for large inputs does not reproduce its
        rows.  An output column of a model that is constant on the
        training rows up to round-off (a range of at most 1e-9 of its
        largest magnitude, or 1e-12 of the largest magnitude of the
        output: for example a principal component along which those rows
        do not vary) is frozen: at every row, a value within that
        tolerance of the column's training mean is returned as exactly
        that mean, so the next order does not correlate round-off.  Output
        columns of a summary that are constant up to round-off
        are not set to their mean (see ``WeightedStrength``), and no
        warning reports them.  Not used by reducers that need no fit;
        required (``ValueError``) when a reducer is fitted.  Given without
        ``causal=True``, it raises ``ValueError``.  Default: None.
    refit_every : int or None
        With ``causal=True`` and ``fit_timepoints=n`` (keyword-only): the
        number of rows ``m >= 1`` after which the reducers are fitted
        again, each time on every row before.  The fits are at rows
        ``s_j = n + j * m`` for ``j = 0, 1, ...`` while ``s_j`` is less
        than the number of timepoints; fit ``j`` reads rows ``0..s_j - 1``
        only.  Rows ``s_j`` to ``s_{j+1} - 1`` of every output (to the
        last row for the last fit) are the rows that this call returns
        with ``fit_timepoints=s_j`` and no ``refit_every``, up to the two
        rules below; rows before ``n`` are those of fit 0.  So the
        guarantee of ``fit_timepoints`` holds: a row ``t >= n`` is
        unchanged when rows after ``t`` are replaced or removed, and a
        row ``t < n`` when rows from ``n`` on are.  Every fit is a pass
        over its rows and one over all rows, so the time grows with the
        number of fits.

        - The structure is that of fit 0, so that every output has the
          same columns at every row.  Fit 0 fixes: what an automatic
          feeding step chose (communities or weighted strength; the
          shift null runs in fit 0 only); the number of output columns
          of a communities step, which is its number of communities of
          2 or more nodes; and the number of components of a model
          (``ndims``).  The number of clusters is not fixed: a weakly
          attached node is a cluster of its own in one fit and a member
          of a community in the next, so a later fit clusters its rows
          with fit 0's number of clusters ``n0`` and then, while the
          communities of 2 or more nodes are not as many as fit 0's,
          with ``n0 + 1``, ``n0 - 1``, ``n0 + 2``, ``n0 - 2``, ... up to
          ``n0 + 5`` and ``n0 - 5`` (between 2 and the number of
          features), and takes the first clustering that has them.  A
          later fit that cannot have the structure (none of those
          clusterings has fit 0's number of communities of 2 or more
          nodes, its model has another number of components or does not
          reproduce its rows, or its rows hold no defined correlations)
          is not used: the fit before, of every order, stays for its
          rows, the row is listed in ``"refit_kept_previous"`` of
          ``return_info``, and one ``UserWarning`` for the call names
          those rows and the step (order and path) that could not be
          fitted.  All of this reads the rows before the fit's row only.
        - The columns of a fit are matched to those of the fit before,
          order by order, before the next order of that fit is fitted:
          the components of a model (the one-to-one assignment with the
          largest sum of absolute cosines between component vectors;
          a component is negated when its cosine is negative), and the
          communities of ``Communities`` (the assignment with the most
          shared nodes).  The
          columns of ``WeightedStrength`` belong to the nodes and are not
          matched; with ``split=True`` a node that changes community
          changes which pairs its own and other columns sum.

        Summary objects passed as reducers keep the records of fit 0 in
        ``fits_``; the last fit is in the model (``return_model``).
        Default: None (one fit).
    return_model : bool
        If True (keyword-only; needs ``causal=True``), also return the
        fitted pipeline as a ``tc.FeaturesModel``, last: ``(X, model)``,
        ``((X, D), model)``, ``(X, info, model)`` or ``((X, D), info,
        model)``.  ``model.transform(data)`` computes the features of
        other data (the same series with later rows, for example) with the
        reducers as fitted here, causally and without fitting anything;
        ``model.transform`` of this data returns these outputs.  With
        ``refit_every`` the model holds the last fit, which
        ``model.transform`` applies to every row: it returns these
        outputs from the last row of ``model.refits`` on, and the rows
        before as the last fit gives them, not as returned here (they
        came from earlier fits).  The model
        can be pickled (see ``FeaturesModel``).  Default: False.

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
    model : FeaturesModel
        Only with ``return_model=True`` (see ``return_model``), after the
        other items.

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
    UserWarning
        Once per call with ``causal=True``: for a model outside the list
        the guarantee covers (see ``fit_timepoints``), and when a fit of
        ``refit_every`` could not have the structure of the first fit and
        was left out (the message names the rows and the step).

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
    ValueError
        With ``causal=True``, before anything is computed: for the
        arguments listed under ``causal`` that have no causal form, and
        for a ``min_samples`` that is not None or a finite number >= 1 or
        that a built-in kernel of the call cannot reach; for
        ``min_samples`` given without ``causal=True``.
    ValueError
        With ``causal=True``, for an ``rfun`` or ``delta_rfun`` that is
        fitted to data (the defaults included) when ``fit_timepoints`` is
        not given; for a ``fit_timepoints`` that is not an integer from 1
        to the number of timepoints, or given without ``causal=True``;
        when the first ``fit_timepoints`` rows hold no row with defined
        correlations for a fit (2 for a model), or fewer than 40 rows for
        the default feeding reducer or a shift null; for a model that
        is not a linear map applied row by row (see ``fit_timepoints``);
        and for a ``refit_every`` that is not None or an integer >= 1,
        or that is given without ``causal=True`` and ``fit_timepoints``.
    TypeError
        If ``rfun`` or ``delta_rfun`` is not None, a string, a dict or a
        callable, or ``delta_weights_function`` is not None or callable.
    MemoryError
        Before a correlation step allocates its arrays, if that step's
        estimate exceeds ``max_memory_gb`` (with unreduced feeding features,
        every order is checked before anything is computed; for a streamed
        step, if one row per block does not fit).  Every streamed step is
        also checked before anything is computed (orders 2 and above with
        ``min(T, 1000)`` rows, the fewest they can have, when feeding
        features are passed on at a lower rate; see "Automatic kernels"),
        with its input width where
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
    Automatic kernels.  With ``delta_weights_params=None`` and a Gaussian
    feeding kernel, the feeding kernel of each order has standard deviation
    ``sd = min(5 w, max(5, T' / 100))`` rows, where ``T'`` is the number of
    rows of that order's input and ``w`` is its autocorrelation width: the
    lag at which the autocorrelation, averaged over all columns of all
    subjects, first falls below ``exp(-1/2)``, linearly interpolated
    between lags and at least 1 (lags beyond ``T' / 500 + 1`` are not
    examined; constant columns are left out and NaN samples count as the
    column's mean).  With ``weights_params=None`` and ``gaussian_weights``
    (and a feeding kernel other than ``eye_weights``), the analysis kernel
    has standard deviation ``max(sd, 10)`` rows.  A Gaussian kernel of
    standard deviation ``sd`` spans about ``3.5 sd`` rows, so the limit of
    ``T' / 100`` leaves at least about 28 windows that do not overlap in
    the recording.  On ``simulate_nested`` data (T 1000, 12 seeds) the
    limit of a hundredth gives ``D[2]`` a decoding score of .560 +- .020
    with a control of .002 +- .025 whose spread over seeds is .086; a limit
    of a two-hundredth gives the fixed kernels' .199 +- .021 with a control
    spread of .051, and on a two-level design of 108 nodes (T 6000, 4
    recordings) lowers the decoding from ``X[2]`` from .775 +- .013 to
    .717 +- .012.  When the mean lag-1 autocorrelation of the data is
    below ``exp(-1/2)`` (about 0.61), ``w`` is 1, so the order-1 kernels
    have standard deviations 10 and 5 samples (variances 100 and 25, the
    fixed kernels of earlier versions).  At orders 2 and above the input
    was itself computed with a kernel and is smoother than the data, so the
    kernels widen from order to order until they reach the limit (measured
    on simulated hierarchies with two-sided kernels: 5, 46 and 470 to 526
    samples at orders 1 to 3, from input widths of 1, 9 and 94 to 105
    samples).  Recordings of up to 1000 timepoints use kernels of at most
    10 samples at every order.  The widths are read from the whole
    recording, also with a one-sided kernel.

    When no kernel parameters and no events are given at any order and
    ``include_timepoints`` is ``"all"``, the
    feeding features of each order are passed on at every ``s``-th row,
    ``s = max(1, min(floor(sd / 2), T' // 1000))``, so later orders of
    recordings of 2000 or more timepoints are computed at fewer timepoints;
    ``X[k]`` and ``D[k]`` are then returned at all T timepoints by linear
    interpolation between the timepoints they were computed at (after the
    last of these, its value is held), and ``X[1]`` and ``D[1]`` are
    computed at every timepoint.  With ``return_info=True``, the entries of
    automatic kernels have ``"kernel_sd"`` (in rows of that order's input)
    and ``"sampling"`` (timepoints of the data per row of that order's
    input); an entry without ``"sampling"`` was computed at every
    timepoint.  ``max_memory_gb`` also covers returning the outputs at all T
    timepoints: each is interpolated into one array, a chunk of rows at a
    time, and a run whose outputs at full length (with the rows being
    interpolated and the work space) would not fit raises ``MemoryError``
    naming their size, before order 1 when the widths are known in advance
    (unreduced on both paths) and otherwise after the order whose outputs
    exceed it.  With kernel parameters given on either path at any order
    (also the values the default would choose: ``"kernel_sd"`` times
    ``"sampling"``, in timepoints of the data), with events, or with a
    one-sided kernel (interpolating between computed rows would read rows
    on the side the kernel excludes), every order is computed at every
    timepoint.  That takes longer and changes orders 2 and above slightly:
    on 2 subjects of 12,000 x 72 at order 3, giving ``weights_params={1:
    {"var": 100}}`` (the value the default chose) took 15.9 s instead of
    8.4 s (one run each), and ``X[2]`` and ``X[3]`` differed by at most
    .021 and .010 (correlations above .9999).

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
    ``UserWarning`` per fit; and the summary's ``n_null`` surrogates, whose
    squared correlations are averaged over every ``s``-th timepoint
    (``s``: the kernel's standard deviation, rounded down, for
    ``gaussian_weights``; see ``Communities``).

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

    >>> X = tc.features(data, order=2, cfun=tc.autofc)
    >>> len(X), len(X[2]), X[1][0].shape, X[2][0].shape
    (3, 2, (300, 24), (300, 12))

    The kernels the defaults chose (standard deviations in rows; these data
    are white and short, so they are the fixed kernels of variance 100 and
    25 at order 1, and 10 rows at order 2):

    >>> X, info = tc.features(data, order=2, cfun=tc.autofc, return_info=True)
    >>> [(e["order"], e["path"], e["kernel_sd"]) for e in info]
    [(1, 'analysis', 10.0), (1, 'feeding', 5.0), (2, 'analysis', 10.0)]

    The community of each node at the feeding step of order 1 (the 6
    modules of 4 nodes), and the columns of each subject's ``X_2`` that are
    constant over time (none):

    >>> info[1]["labels"]
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    >>> info[2]["constant_columns"]
    [[], []]

    The same kernels given explicitly (these data have fewer than 2000
    timepoints, so no order is computed at a lower rate either way):

    >>> Xe = tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 100},
    ...                  delta_weights_params={"var": 25})
    >>> all(np.array_equal(a, b) for a, b in zip(X[2], Xe[2]))
    True

    Restrict the order-2 kernels to two events (timepoints 0-149 and
    150-299); ``X_1`` is unrestricted:

    >>> X = tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 100},
    ...                 events={2: [150]})
    >>> X[2][0].shape
    (300, 12)

    Kernels by order: a wider analysis kernel at order 2 than at order 1,
    and a feeding kernel of variance 64 at order 1 (orders missing from a
    map use the default, the automatic kernel):

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

    Causal features: row ``t`` uses rows ``0..t`` of the data only.  The
    kernel parameters of both paths and reducers that are not fitted are
    given; the first rows have too few earlier samples and are NaN, more of
    them at each order (a ``UserWarning`` per step says so):

    >>> x = np.random.RandomState(1).randn(200, 5)
    >>> kw = dict(order=2, causal=True, cfun=tc.autofc, rfun="strength",
    ...           delta_rfun="strength", weights_params={"var": 25},
    ...           delta_weights_params={"var": 25})
    >>> X, info = tc.features(x, return_info=True, **kw)
    >>> [(e["order"], e["path"], e["warmup_rows"]) for e in info]
    [(1, 'analysis', 2), (1, 'feeding', 2), (2, 'analysis', 4)]
    >>> later = x.copy()
    >>> later[101:] = 1000.0
    >>> changed = tc.features(later, **kw)
    >>> bool(np.array_equal(changed[2][:101], X[2][:101], equal_nan=True))
    True

    Reducers that are fitted (here PCA and weighted strength) are fitted
    on the first ``fit_timepoints`` rows and then frozen; the model applies
    them to other data, such as the same series with more rows:

    >>> kw = dict(order=2, causal=True, cfun=tc.autofc, rfun="PCA",
    ...           delta_rfun="weighted_strength", weights_params={"var": 25},
    ...           delta_weights_params={"var": 25}, fit_timepoints=120)
    >>> X, model = tc.features(x[:150], return_model=True, **kw)
    >>> model
    FeaturesModel(order=2, fit_timepoints=120, n_features=5, n_subjects=1)
    >>> longer = model.transform(x)
    >>> longer[2].shape
    (200, 5)
    >>> bool(np.allclose(longer[2][:150], X[2], atol=1e-9, equal_nan=True))
    True

    With ``refit_every`` the reducers are fitted again every so many rows,
    each time on all rows before; the rows from a fit to the next are those
    of a call with that fit alone (the columns of a model or of
    communities are also matched to those of the fit before):

    >>> kw["rfun"] = "weighted_strength"
    >>> X, info = tc.features(x, refit_every=40, return_info=True, **kw)
    >>> info[0]["refit_rows"], info[0]["refit_kept_previous"]
    ([120, 160], [])
    >>> alone = tc.features(x, **{**kw, "fit_timepoints": 160})
    >>> bool(np.array_equal(X[2][160:], alone[2][160:]))
    True
    """
    if (
        not isinstance(order, numbers.Integral)
        or isinstance(order, bool)
        or order < 0
    ):
        raise ValueError(f"order must be a non-negative integer; got {order!r}.")
    order = int(order)
    min_samples = _check_min_samples(min_samples, causal)
    if return_model and not causal:
        raise ValueError(
            "return_model=True applies with causal=True only (the model holds "
            "the reducers fitted on the first fit_timepoints rows); pass "
            "causal=True, or leave return_model out."
        )
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
    # kernel parameters given for each order: an order map's value (None,
    # the default, for orders missing from it), or the same parameters at
    # every order
    analysis_params = {
        k: weights_params if analysis_map is None else analysis_map.get(k)
        for k in orders
    }
    given_feeding = {
        k: delta_weights_params if feeding_map is None else feeding_map.get(k)
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
            k: _feeding_kernel(delta_weights_function, given_feeding[k])[1]
            for k in orders
        }
    # paths on which parameters of None are set from each order's input
    auto_analysis, auto_feeding = _automatic_paths(weights_function, delta_weights_function)
    auto_analysis = {k: auto_analysis and analysis_params[k] is None for k in orders}
    auto_feeding = {k: auto_feeding and given_feeding[k] is None for k in orders}
    for summary in _summary_instances(rfun, delta_rfun):
        summary.fits_ = []

    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    T, K = subjects[0].shape
    delta_unreduced = all(feeding_rfun[k] is None for k in orders)

    S = len(subjects)
    fit_timepoints = _check_fit_timepoints(fit_timepoints, causal, T)
    refit_every = _check_refit_every(refit_every, causal, fit_timepoints)
    if causal:
        _check_causal(
            function="features", include_timepoints=include_timepoints,
            weights_function=weights_function,
            delta_weights_function=delta_weights_function, cfun=cfun, rfun=rfun,
            delta_rfun=delta_rfun, analysis_rfun=analysis_rfun,
            feeding_rfun=feeding_rfun, analysis_params=analysis_params,
            feeding_params=given_feeding,
            events=list(events.values()) if isinstance(events, dict) else [events],
            T=T, n_subjects=S, include_zero=include_zero, min_samples=min_samples,
            fit_timepoints=fit_timepoints, combine=combine,
        )
        include_timepoints = "backward"
    # events are resolved (and features segmented) once, before anything is computed
    if order == 0:
        _check_events_order_zero(events, T, S, events_params, causal)
        resolved = {}
    else:
        resolved = resolve_events_by_order(events, T, S, order, events_params,
                                           causal=causal)

    # with both kernels automatic at every order, no events and two-sided
    # kernels, each order's feeding features are passed on at a lower rate
    # (with a one-sided kernel, interpolating between the computed rows
    # would read rows on the side the kernel excludes)
    thin = include_timepoints == "all" and all(
        auto_analysis[k] and auto_feeding[k] and resolved[k] is None for k in orders
    )
    # at a lower rate, orders 2 and above have at least min(T, _AUTO_MIN_ROWS)
    # rows (how many depends on the data): the checks before order 1 runs
    # use that, so that a run is refused here only if it cannot fit; each
    # step is checked again with its actual rows when it runs
    later_T = min(T, _AUTO_MIN_ROWS) if thin else None
    if delta_unreduced and max_memory_gb is not None:
        # widths are known in advance: refuse before computing anything
        kernels = dict(
            kernel_feeding=feeding_function is not eye_weights,
            dense=_dense_analysis(weights_function, cfun, causal),
            combine=combine,
            dense_feeding=not _builtin_kernel(feeding_function),
        )
        estimates = _estimates(K, order, T, S, return_delta, False,
                               later_T=later_T, **kernels)
        for k in range(1, order + 1):
            if estimates[k] > max_memory_gb:
                fit = _largest_k(order, T, max_memory_gb, S, return_delta,
                                 later_T=later_T, **kernels)
                raise MemoryError(
                    f"Unreduced features at order {k} need about {estimates[k]:.3g} GB "
                    f"(K={K}, T={T}, {S} subject{'s' if S > 1 else ''}), more than "
                    f"max_memory_gb={max_memory_gb}. The largest number of features "
                    f"that fits orders up to {order} is K={fit}. Pass a reducer "
                    "(rfun, delta_rfun), lower the order, or raise max_memory_gb."
                )
    if (
        thin and max_memory_gb is not None and order >= 2
        and T >= 2 * _AUTO_MIN_ROWS
        and delta_unreduced and all(analysis_rfun[k] is None for k in orders)
    ):
        # the outputs returned at all T timepoints, widths known in advance
        # (a lower bound: orders 2 and above have at least later_T rows)
        d, full = K, 0
        for k in orders:
            d = d * (d + 1) // 2
            # X[k], and D[k] (as wide, unreduced) when returned
            full += S * T * d * (2 if return_delta else 1)
        # plus the formatted data, which the run holds to the end
        total = 8 * (full + later_T * d + 4 * _EXPAND_ELEMENTS + S * T * K)
        if total / _BYTES_PER_GB > max_memory_gb:
            raise MemoryError(_expansion_message(T, total, 8 * full, max_memory_gb, True))
    if max_memory_gb is not None:
        # streamed steps of later orders are checked before order 1 runs
        _check_streamed_steps(
            K, T, S, max_memory_gb, analysis_rfun, feeding_rfun, fed, cfun=cfun,
            combine=combine, feeding_function=feeding_function, ndims=ndims,
            weights_function=weights_function, later_T=later_T, causal=causal,
        )

    if causal:
        from .model import _causal_features  # timecorr.model imports this module

        return _causal_features(
            subjects, single, events,
            dict(
                order=order, weights_function=weights_function,
                weights_params=analysis_params, cfun=cfun, combine=combine,
                rfun=analysis_rfun, delta_rfun=feeding_rfun, ndims=ndims,
                delta_weights_function=feeding_function,
                delta_weights_params=feeding_params, include_zero=include_zero,
                min_samples=min_samples, max_memory_gb=max_memory_gb,
                requested={k: (_describe(_requested(rfun, k)),
                               _describe(_requested(delta_rfun, k))) for k in orders},
                auto={k: _auto_at(rfun, k) for k in orders},
                fed=list(fed),
            ),
            fit_timepoints=fit_timepoints, refit_every=refit_every,
            return_delta=return_delta, return_info=return_info,
            return_model=return_model,
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
    # rows: the timepoints (indices into the T of the data) of the rows of
    # the current feeding features
    rows = np.arange(T)
    computed_at = [rows]  # the timepoints of the rows of X[k] and D[k]
    for k in orders:
        analysis_auto, feeding_auto = _auto_params(
            fed_by, auto_analysis[k], auto_feeding[k]
        )
        if analysis_auto is not None:
            analysis_params[k] = analysis_auto
        if feeding_auto is not None:
            feeding_params[k] = feeding_auto
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
        computed_at.append(rows)
        previous = feeding_rfun[k]
        # explicit Communities only: the automatic feeding reducer keeps at
        # least 4 columns (see summaries._auto_emitter)
        previous_communities = (
            previous.fits_[-1]["n"]
            if isinstance(previous, Communities) and previous.fits_
            and not getattr(previous, "_from_auto", False) else None
        )
        if return_info:
            entries = _step_info(
                k, fed_by, x, delta, analysis_rfun[k], feeding_rfun[k],
                _requested(rfun, k), _requested(delta_rfun, k),
            )
            # automatic kernels: their standard deviation, in rows of this
            # order's input, and the timepoints of the data per row
            for entry, params in zip(entries, (analysis_auto, feeding_auto)):
                if params is not None:
                    entry["kernel_sd"] = _bandwidth(gaussian_weights, params)
                    entry["sampling"] = int(rows[1] - rows[0]) if len(rows) > 1 else 1
            if delta is not None and not return_delta:
                # feeding features that are not returned: as computed
                computed = delta[0] if single else delta
                entries[-1]["constant_columns"] = _constant_columns(computed)
            info.extend(entries)
        if delta is not None:
            fed_by = delta
            if return_delta:
                D.append(delta)
            if thin:
                sd = _bandwidth(gaussian_weights, feeding_auto)
                stride = max(1, min(int(sd // 2), len(rows) // _AUTO_MIN_ROWS))
                fed_by = [d[::stride] for d in delta]
                rows = rows[::stride]
        if thin:
            # the outputs so far, at all T timepoints: a lower bound of what
            # returning them needs
            _check_expansion(
                X[1:] + D[1:], computed_at[1:] + computed_at[1: len(D)], T,
                max_memory_gb, _fits_elements(*held_reducers) + S * T * K,
                lower_bound=k < order,
            )

    if thin:
        # release the references to computed arrays, then expand each output
        # in place (each computed array is released once expanded)
        x = delta = fed_by = kept = None
        X[1:] = [_at_every_timepoint(x, at, T) for x, at in zip(X[1:], computed_at[1:])]
        D[1:] = [_at_every_timepoint(d, at, T) for d, at in zip(D[1:], computed_at[1:])]
    if single:
        X = [x[0] if isinstance(x, list) else x for x in X]
        D = [d[0] for d in D]
    result = (X, D) if return_delta else X
    if return_info:
        # the constant columns of the returned arrays
        for entry in info:
            if entry["path"] == "analysis" or return_delta:
                out = (X if entry["path"] == "analysis" else D)[entry["order"]]
                entry["constant_columns"] = _constant_columns(out)
        return result, info
    return result
