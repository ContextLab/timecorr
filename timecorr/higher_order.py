# coding: utf-8
"""
Higher-order features through two paths per order (Owen, Chang & Manning, 2021, Fig. 10).

At each order ``k >= 1`` the previous order's delta features ``D_{k-1}`` feed two
computations:

- the analysis path, ``X_k = reduce(timecorr(D_{k-1}, analysis kernel, cfun, combine))``,
  which gives that order's output features, and
- the delta path, ``D_k = reduce(timecorr(D_{k-1}, eye_weights, cfun=autofc))``, which
  gives unblurred co-fluctuations that are passed to the next order.

Temporal blur is therefore applied once, at the order being analyzed.
"""

import numbers

import numpy as np

from .helpers import (
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
)
from .timecorr import timecorr

_BYTES_PER_GB = 1e9

# Block temporaries of the correlation step, in multiples of one block
# (``_wcorr_blocks`` keeps about this many block-sized arrays alive at once;
# measured with tracemalloc for isfc and autofc).
_WORKSPACE_BLOCKS = 6


def _step_gb(d, T, n_subjects=1, n_outputs=1):
    """
    Peak memory (GB) of one unreduced correlation step on ``n_subjects`` (T, d)
    inputs that holds ``n_outputs`` (T, d(d+1)/2) results per subject.
    """
    d, T, S = int(d), int(T), int(n_subjects)
    v = d * (d + 1) // 2
    block = min(d, _wcorr_block_rows(T, d)) * T * d
    elements = (
        d * d * T  # the (d, d, T) correlation array of one subject
        + S * T * v  # the analysis path's vectorized results, every subject
        + 3 * S * T * d  # the inputs (and the copies made while formatting)
        + 3 * T * T  # the kernel, its working copy and its absolute value
        # the analysis path's block temporaries, or (when the delta path is
        # computed too: it has no block temporaries) the delta results
        + max(_WORKSPACE_BLOCKS * block, (n_outputs - 1) * S * T * v)
    )
    return elements * 8 / _BYTES_PER_GB


def estimate_memory_gb(K, order, T, n_subjects=1, return_delta=False):
    """
    Estimate the peak memory of each order of an unreduced feature pipeline.

    Without reduction the number of features grows from ``d`` to ``d(d+1)/2`` at
    every order, starting from ``K``.  The correlation step at order ``k``
    allocates one ``(d_{k-1}, d_{k-1}, T)`` array at a time, holds a
    ``(T, d_k)`` vectorized result for every subject (two per subject once
    the delta path has run too), and the analysis path uses block temporaries
    of bounded size, all in float64.  The estimate is the peak of
    ``tc.features(..., rfun=None)`` with the default ``combine``
    (``null_combine``) for data without NaNs; other ``combine`` functions and
    reducers allocate more.

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
        ``features(..., return_delta=True)`` does).  Default: False.

    Returns
    -------
    list of float
        ``order + 1`` estimates in gigabytes (1e9 bytes), indexed by order; entry
        0 is 0.0 because order 0 is the input itself.

    Examples
    --------
    >>> from timecorr.higher_order import estimate_memory_gb
    >>> [round(x, 3) for x in estimate_memory_gb(100, 2, 200, n_subjects=3)]
    [0.0, 0.139, 102.484]
    """
    K = int(K)
    T = int(T)
    estimates = [0.0]
    d = K
    for k in range(1, int(order) + 1):
        both_paths = k < order or return_delta
        estimates.append(_step_gb(d, T, n_subjects, 2 if both_paths else 1))
        d = d * (d + 1) // 2
    return estimates


def _largest_k(order, T, max_memory_gb, n_subjects=1, return_delta=False):
    """Largest K whose unreduced estimates up to ``order`` all fit (0 if none)."""

    def fits(k):
        est = estimate_memory_gb(k, order, T, n_subjects, return_delta)
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


def _check_step_memory(d, T, n_subjects, n_outputs, max_memory_gb, order=None):
    """
    Raise MemoryError before an unreduced correlation step on ``n_subjects``
    (T, d) inputs whose estimate (``_step_gb``) exceeds ``max_memory_gb``.
    ``max_memory_gb=None`` checks nothing.  Shared by ``features``, the path
    functions and the decoders, so all refuse the same steps.
    """
    if max_memory_gb is None:
        return
    S = int(n_subjects)
    step = _step_gb(d, T, S, n_outputs)
    if step > max_memory_gb:
        fit = _largest_k(1, T, max_memory_gb, S, n_outputs > 1)
        where = "" if order is None else f" at order {order}"
        raise MemoryError(
            f"The correlation step{where} needs about {step:.3g} GB "
            f"({d} input features, T={T}, {S} subject{'s' if S > 1 else ''}), "
            f"more than max_memory_gb={max_memory_gb}. The largest number of "
            f"input features that fits one step is K={fit}. Reduce to fewer "
            "features (ndims, rfun, delta_rfun) or raise max_memory_gb."
        )


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _reduce(x, rfun, ndims):
    """``reduce``, passing ``ndims`` only to model-based reducers (it sets their width)."""
    return reduce(x, rfun=rfun, ndims=ndims if _is_model(rfun) else None)


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
):
    """
    Analysis path of one order: ``reduce(combine(timecorr(subjects, kernel, cfun)))``.

    ``subjects`` is a list of (T, d) arrays.  With ``groups=None`` all subjects
    are analyzed together and the result is what ``combine`` returns, reduced.
    With ``groups`` (a sequence of sequences of subject indices), ``cfun`` and
    ``combine`` see only one group's subjects at a time, the combined results of
    all groups are reduced jointly (one model, one shared space), and a list with
    one entry per group is returned.

    Before computing, the step is checked against ``max_memory_gb`` (see
    ``_check_step_memory``; ``n_outputs=2`` when the delta path of the same
    order follows and its results are held together).
    """
    T, d = subjects[0].shape
    _check_step_memory(d, T, len(subjects), n_outputs, max_memory_gb, order)

    def analyze(members):
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
                )
            )
        )

    if groups is None:
        return _reduce(analyze(subjects), rfun, ndims)

    combined = [analyze([subjects[j] for j in g]) for g in groups]
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
    reduced = _as_list(_reduce(pieces, rfun, ndims))
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
):
    """
    Delta path of one order: per-subject co-fluctuations (``eye_weights``,
    ``autofc``), never combined, reduced jointly across subjects.  ``subjects`` is
    a list of (T, d) arrays; a list of the same length is returned.  The step
    is checked against ``max_memory_gb`` first (see ``_check_step_memory``).

    A co-fluctuation involving a NaN sample is NaN (not 0), so the next
    order's correlations leave it out pairwise.  Unreduced, and for a callable
    reducer, the NaNs are kept.  Before a model-based reduction or a graph
    measure they are set to 0 with one ``UserWarning``; ``nan_warned`` (a
    one-element list shared across the steps of one call) keeps it to one
    warning per call.
    """
    T, d = subjects[0].shape
    _check_step_memory(d, T, len(subjects), n_outputs, max_memory_gb, order)
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
        if rfun is not None and not callable(rfun):
            if nan_warned is None or not nan_warned[0]:
                kind = "model-based" if _is_model(rfun) else "graph-measure"
                _warn_user(_NAN_FILL_MESSAGE.format(kind=kind, rfun=rfun))
                if nan_warned is not None:
                    nan_warned[0] = True
            delta = [np.nan_to_num(c, nan=0.0) for c in delta]
    return _as_list(_reduce(delta, rfun, ndims))


def two_path_step(
    data,
    weights_function=gaussian_weights,
    weights_params=None,
    cfun=isfc,
    rfun="PCA",
    delta_rfun=None,
    ndims=None,
    combine=null_combine,
    include_timepoints="all",
    include_zero=True,
    max_memory_gb=4.0,
):
    """
    Compute one order of the two-path pipeline from the previous delta features.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
        The previous order's delta features ``D_{k-1}`` (the raw data for
        ``k = 1``), one timepoints-by-features matrix per subject.
    weights_function : function
        Analysis kernel, called as ``weights_function(T, weights_params)``; row
        ``t`` of the kernel (``weights[t, :]``) is the kernel used at timepoint
        ``t``.  Default: ``gaussian_weights``.
    weights_params : dict or None
        Parameters of the analysis kernel.  Default: None (the kernel's
        defaults, for example ``gaussian_params``).
    cfun : function
        Analysis-path correlation function (``isfc``, ``wisfc``, ``autofc``).
        Default: ``isfc``.
    rfun : None, str, dict, or callable
        Reducer for the analysis path (see ``helpers.reduce``).  Default: "PCA".
    delta_rfun : None, str, dict, or callable
        Reducer for the delta path.  Default (None): the same as ``rfun``.
    ndims : int, optional
        Output width of model-based reduction (a model name or dict) on
        either path (see ``helpers.reduce``); it is not passed to a graph
        measure or a callable, which set their own width.  Default (None):
        ``K`` columns for ``K(K+1)/2`` inputs.
    combine : function
        Applied to the analysis path only (see ``timecorr``).  Default:
        ``null_combine``.
    include_timepoints : str
        "all", "backward"/"pre", or "forward"/"post" (see ``timecorr``).
        Shapes the analysis kernel only, not the delta path.  Default: "all".
    include_zero : bool
        Whether the estimated timepoint contributes to its own estimate.
        Shapes the analysis kernel only, not the delta path.  Default: True.
    max_memory_gb : float or None
        Upper limit, in gigabytes, on the estimated peak memory of the step
        for all subjects together (see ``features`` and
        ``estimate_memory_gb``); None checks nothing.  Default: 4.0.

    Returns
    -------
    X_next : numpy.ndarray or list of numpy.ndarray
        Analysis features: ``reduce(combine(timecorr(data, analysis kernel)))``.
        One array per subject for list input unless ``combine`` returns a
        single array; a single array for array input.
    D_next : numpy.ndarray or list of numpy.ndarray
        Delta features: per-subject co-fluctuations (``eye_weights``,
        ``autofc``), never combined across subjects, reduced jointly.  A
        co-fluctuation involving a NaN sample is NaN (kept when ``delta_rfun``
        is None or a callable; set to 0, with a ``UserWarning``, before a
        model-based or graph-measure reduction).  A list
        for list input, an array for array input.

    Warns
    -----
    UserWarning
        From the correlation estimator on the analysis path (see ``wcorr``),
        one per call, naming each that applies: timepoints whose kernel row
        has fewer than 3 non-zero kernel weights (for example the first
        timepoints of a backward kernel; their correlations are undefined and
        set to 0); other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there);
        timepoints where, because of NaN samples, some pairs of features
        have fewer than 3 observed samples with non-zero kernel weight
        (those correlations are undefined and set to 0); and timepoints
        with zero kernel-weighted variance in every feature (their correlations are set
        to 0).  Never for the delta path.

    Raises
    ------
    MemoryError
        Before the step allocates its arrays, if its estimate exceeds
        ``max_memory_gb``.
    ValueError
        If ``rfun`` or ``delta_rfun`` is an unknown model or graph measure
        name or a dict without a ``'model'``, if ``ndims`` is not a positive
        integer, and as in ``timecorr`` (subjects of different shapes,
        non-numeric columns, an unknown ``include_timepoints``, or an all-zero
        kernel).
    TypeError
        If ``rfun`` or ``delta_rfun`` is not None, a string, a dict or a
        callable.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.higher_order import two_path_step
    >>> rng = np.random.RandomState(0)
    >>> data = [rng.randn(40, 4) for _ in range(3)]
    >>> X1, D1 = two_path_step(data, weights_params={"var": 10})
    >>> len(X1), X1[0].shape, len(D1), D1[0].shape
    (3, (40, 4), 3, (40, 4))
    """
    _check_rfun(rfun, "rfun")
    _check_rfun(delta_rfun, "delta_rfun")
    ndims = _check_ndims(ndims)
    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    X_next, D_next = _two_path_step(
        subjects,
        weights_function=weights_function,
        weights_params=weights_params,
        cfun=cfun,
        rfun=rfun,
        delta_rfun=delta_rfun,
        ndims=ndims,
        combine=combine,
        include_timepoints=include_timepoints,
        include_zero=include_zero,
        max_memory_gb=max_memory_gb,
    )
    if single:
        if isinstance(X_next, list):
            X_next = X_next[0]
        D_next = D_next[0]
    return X_next, D_next


def _two_path_step(
    subjects, delta_rfun=None, max_memory_gb=None, order=None, nan_warned=None,
    **analysis
):
    """``two_path_step`` on formatted subjects; returns lists (X_next, D_next)."""
    if delta_rfun is None:
        delta_rfun = analysis.get("rfun", "PCA")
    X_next = _analysis_path(
        subjects, max_memory_gb=max_memory_gb, n_outputs=2, order=order, **analysis
    )
    D_next = _delta_path(
        subjects,
        rfun=delta_rfun,
        ndims=analysis.get("ndims"),
        max_memory_gb=max_memory_gb,
        n_outputs=2,
        order=order,
        nan_warned=nan_warned,
    )
    return X_next, D_next


def features(
    data,
    order=1,
    weights_function=gaussian_weights,
    weights_params=None,
    cfun=isfc,
    rfun="PCA",
    delta_rfun=None,
    ndims=None,
    combine=null_combine,
    include_timepoints="all",
    include_zero=True,
    return_delta=False,
    max_memory_gb=4.0,
):
    """
    Compute features of every order from 0 to ``order`` with the two-path pipeline.

    Order 1 is the dynamic correlations of the data; order ``k`` is the
    dynamic correlations of the order-``(k-1)`` co-fluctuations, so the
    order-``k`` features (without reduction) are correlations among products
    of up to ``2**(k-1)`` of the original features.

    ``X_0 = D_0 = data``.  For ``k >= 1``,
    ``X_k = reduce(timecorr(D_{k-1}, analysis kernel, cfun, combine), rfun)`` and
    ``D_k = reduce(timecorr(D_{k-1}, eye_weights, cfun=autofc), delta_rfun)``, so
    each order's features are blurred by the analysis kernel exactly once
    (Owen, Chang & Manning, 2021, Fig. 10).  Reductions of lists are fit jointly
    across subjects, so all subjects share one space.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
        One timepoints-by-features matrix per subject.  A 1-D array or a
        pandas or polars Series is one feature (shape ``(T, 1)``); arrays
        with more than two dimensions raise ``ValueError``.  NaN samples
        carry no weight at order 1, and their co-fluctuations on the delta
        path are NaN, so later orders leave them out pairwise as well when
        the delta path is unreduced (``delta_rfun=None``) or reduced by a
        callable (which receives the NaNs).  A model-based reducer or a graph
        measure cannot take NaNs: before one, they are set to 0 and a
        ``UserWarning`` says that NaN handling is approximate from order 2
        on.  ``X_0`` (and ``D_0``) keep the NaNs.
    order : int
        Highest order to compute (non-negative integer).  Default: 1.
    weights_function, weights_params : function, dict or None
        Analysis kernel and its parameters (see ``timecorr``); row ``t`` of the
        kernel (``weights[t, :]``) is the kernel used at timepoint ``t``.
        Default: ``gaussian_weights`` with ``weights_params=None`` (its
        defaults, ``gaussian_params``).
    cfun : function
        Analysis-path correlation function (``isfc``, ``wisfc``, ``autofc``).
        Default: ``isfc``.
    rfun : None, str, dict, or callable
        Analysis-path reducer (see ``helpers.reduce``).  Default: "PCA".
    delta_rfun : None, str, dict, or callable
        Delta-path reducer.  Default (None): the same as ``rfun``.  A
        delta-kernel co-fluctuation matrix ``z z^T`` has rank one, so on the
        delta path eigenvector centrality equals ``|z| / ||z||`` and the graph
        measures carry no network structure.
    ndims : int, optional
        Output width of model-based reduction (a model name or dict) on
        either path; it is not passed to a graph measure or a callable, which
        set their own width.  Default (None): ``K`` columns for ``K(K+1)/2``
        inputs.
    combine : function
        Applied to the analysis path only (see ``timecorr``).  Default:
        ``null_combine``.
    include_timepoints : str
        "all", "backward"/"pre", or "forward"/"post".  Shapes the analysis
        kernel only, not the delta path.  Default: "all".
    include_zero : bool
        Whether the estimated timepoint contributes to its own estimate.
        Shapes the analysis kernel only, not the delta path.  Default: True.
    return_delta : bool
        If True, also return ``[D_0, ..., D_order]``.  If False, the delta
        step of the last order (``D_order``), which feeds no later order, is
        not computed, so ``delta_rfun`` is not called at that step (and is not
        called at all when ``order`` is 1).  Default: False.
    max_memory_gb : float or None
        Upper limit, in gigabytes (1e9 bytes), on the estimated peak memory of
        any one correlation step for all subjects together (see
        ``estimate_memory_gb``; the estimate covers the correlation step and
        the unreduced results it holds, not the memory of reducers or of
        ``combine`` functions other than ``null_combine``).  It is checked on
        every step, because the (d, d, T) correlation array is allocated
        whether or not the result is then reduced; for large K, raising it is
        the remedy.  None checks nothing.  Default: 4.0.

    Returns
    -------
    X : list
        ``[X_0, ..., X_order]``.  For array input each entry is an array; for list
        input each entry is a list with one array per subject, unless ``combine``
        returns a single array.  ``X_0`` is the input as float arrays.
    D : list
        Only with ``return_delta=True``: ``[D_0, ..., D_order]``, with the same
        container types (never combined across subjects).

    Warns
    -----
    UserWarning
        From the correlation estimator on the analysis path (see ``wcorr``),
        one per correlation step, naming each that applies: timepoints whose
        kernel row has fewer than 3 non-zero kernel weights (their
        correlations are undefined and set to 0); other timepoints whose
        effective sample size ``(sum |w|)**2 / sum w**2`` is below 2 (the
        kernel is too narrow to estimate correlations reliably there);
        timepoints where, because of NaN samples, some pairs of features
        have fewer than 3 observed samples with non-zero kernel weight
        (those correlations are undefined and set to 0); and timepoints
        with zero kernel-weighted variance in every feature (their
        correlations are set to 0).  Never for the delta path.
    UserWarning
        Once per call, when NaN co-fluctuations are set to 0 before a
        model-based or graph-measure delta reduction.

    Raises
    ------
    ValueError
        If ``order`` is not a non-negative integer; if ``rfun`` or
        ``delta_rfun`` is an unknown model or graph measure name or a dict
        without a ``'model'`` (both are checked before anything is computed,
        even when ``delta_rfun`` would not be called); if ``ndims`` is not a
        positive integer; and as in ``timecorr`` (subjects of different
        shapes, non-numeric columns, an unknown ``include_timepoints``).
    TypeError
        If ``rfun`` or ``delta_rfun`` is not None, a string, a dict or a
        callable.
    MemoryError
        Before a correlation step allocates its arrays, if that step's
        estimate exceeds ``max_memory_gb`` (with unreduced delta features,
        every order is checked before anything is computed).  The message
        gives the estimate and the largest number of features that would fit.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> rng = np.random.RandomState(0)
    >>> data = [rng.randn(50, 5) for _ in range(3)]
    >>> X = tc.features(data, order=2, weights_params={"var": 10})
    >>> len(X), len(X[2]), X[2][0].shape
    (3, 3, (50, 5))
    >>> X, D = tc.features(data[0], order=1, rfun=None, return_delta=True)
    >>> X[1].shape, D[1].shape
    ((50, 15), (50, 15))
    """
    if (
        not isinstance(order, numbers.Integral)
        or isinstance(order, bool)
        or order < 0
    ):
        raise ValueError(f"order must be a non-negative integer; got {order!r}.")
    order = int(order)
    # every reducer is checked here, including ones a short run never calls
    _check_rfun(rfun, "rfun")
    _check_rfun(delta_rfun, "delta_rfun")
    ndims = _check_ndims(ndims)

    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    T, K = subjects[0].shape
    delta_unreduced = (rfun if delta_rfun is None else delta_rfun) is None

    S = len(subjects)
    if delta_unreduced and max_memory_gb is not None:
        # widths are known in advance: refuse before computing anything
        estimates = estimate_memory_gb(K, order, T, S, return_delta)
        for k in range(1, order + 1):
            if estimates[k] > max_memory_gb:
                fit = _largest_k(order, T, max_memory_gb, S, return_delta)
                raise MemoryError(
                    f"Unreduced features at order {k} need about {estimates[k]:.3g} GB "
                    f"(K={K}, T={T}, {S} subject{'s' if S > 1 else ''}), more than "
                    f"max_memory_gb={max_memory_gb}. The largest number of features "
                    f"that fits orders up to {order} is K={fit}. Pass a reducer "
                    "(rfun, delta_rfun), lower the order, or raise max_memory_gb."
                )

    X = [subjects]
    D = [subjects]
    nan_warned = [False]  # one NaN-filling warning per call
    for k in range(1, order + 1):
        analysis = dict(
            weights_function=weights_function,
            weights_params=weights_params,
            cfun=cfun,
            rfun=rfun,
            ndims=ndims,
            combine=combine,
            include_timepoints=include_timepoints,
            include_zero=include_zero,
        )
        if k == order and not return_delta:
            # the last order's delta features feed nothing and are not returned:
            # skip them (unreduced, they are as large as X_order)
            X.append(
                _analysis_path(D[-1], max_memory_gb=max_memory_gb, order=k, **analysis)
            )
            break
        x, delta = _two_path_step(
            D[-1], delta_rfun=delta_rfun, max_memory_gb=max_memory_gb, order=k,
            nan_warned=nan_warned, **analysis
        )
        X.append(x)
        D.append(delta)

    if single:
        X = [x[0] if isinstance(x, list) else x for x in X]
        D = [d[0] for d in D]
    if return_delta:
        return X, D
    return X
