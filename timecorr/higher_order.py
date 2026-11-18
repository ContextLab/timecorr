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

from .helpers import (
    _wcorr_block_rows,
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


def _as_list(x):
    return x if isinstance(x, list) else [x]


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
):
    """
    Analysis path of one order: ``reduce(combine(timecorr(subjects, kernel, cfun)))``.

    ``subjects`` is a list of (T, d) arrays.  With ``groups=None`` all subjects
    are analyzed together and the result is what ``combine`` returns, reduced.
    With ``groups`` (a sequence of sequences of subject indices), ``cfun`` and
    ``combine`` see only one group's subjects at a time, the combined results of
    all groups are reduced jointly (one model, one shared space), and a list with
    one entry per group is returned.
    """

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
        return reduce(analyze(subjects), rfun=rfun, ndims=ndims)

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
    reduced = _as_list(reduce(pieces, rfun=rfun, ndims=ndims))
    out, start = [], 0
    for n in sizes:
        if n is None:
            out.append(reduced[start])
            start += 1
        else:
            out.append(list(reduced[start : start + n]))
            start += n
    return out


def _delta_path(subjects, rfun="PCA", ndims=None):
    """
    Delta path of one order: per-subject co-fluctuations (``eye_weights``,
    ``autofc``), never combined, reduced jointly across subjects.  ``subjects`` is
    a list of (T, d) arrays; a list of the same length is returned.
    """
    delta = _as_list(
        timecorr(
            subjects,
            weights_function=eye_weights,
            combine=null_combine,
            cfun=autofc,
            rfun=None,
        )
    )
    return _as_list(reduce(delta, rfun=rfun, ndims=ndims))


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
        Output width for model-based reduction on both paths (see
        ``helpers.reduce``).  Default (None): ``K`` columns for ``K(K+1)/2``
        inputs.
    combine : function
        Applied to the analysis path only (see ``timecorr``).  Default:
        ``null_combine``.
    include_timepoints : str
        "all", "backward"/"pre", or "forward"/"post" (see ``timecorr``).
        Shapes the analysis kernel only, not the delta path.  Default: "all".
    include_zero : bool
        Whether the estimated timepoint contributes to its own estimate.
        Shapes the analysis kernel only, not the delta path.  Default: True.

    Returns
    -------
    X_next : numpy.ndarray or list of numpy.ndarray
        Analysis features: ``reduce(combine(timecorr(data, analysis kernel)))``.
        One array per subject for list input unless ``combine`` returns a
        single array; a single array for array input.
    D_next : numpy.ndarray or list of numpy.ndarray
        Delta features: per-subject co-fluctuations (``eye_weights``,
        ``autofc``), never combined across subjects, reduced jointly.  A list
        for list input, an array for array input.

    Warns
    -----
    UserWarning
        When some rows of the analysis kernel have fewer than 3 non-zero
        weights (for example the first timepoints of a backward kernel): the
        correlations there are undefined and set to 0.

    Raises
    ------
    ValueError
        As in ``timecorr`` (subjects of different shapes, non-numeric columns,
        an unknown ``include_timepoints``, or an all-zero kernel).

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
    if delta_rfun is None:
        delta_rfun = rfun
    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)

    X_next = _analysis_path(
        subjects,
        weights_function=weights_function,
        weights_params=weights_params,
        cfun=cfun,
        rfun=rfun,
        ndims=ndims,
        combine=combine,
        include_timepoints=include_timepoints,
        include_zero=include_zero,
    )
    D_next = _delta_path(subjects, rfun=delta_rfun, ndims=ndims)

    if single:
        if isinstance(X_next, list):
            X_next = X_next[0]
        D_next = D_next[0]
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
        with more than two dimensions raise ``ValueError``.  NaN handling
        (NaN samples carry no weight) applies to the input data at order 1;
        ``X_0`` keeps the NaNs.
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
        Output width of model-based reductions on both paths.  Default (None):
        ``K`` columns for ``K(K+1)/2`` inputs.
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
    max_memory_gb : float
        Upper limit, in gigabytes (1e9 bytes), on the estimated peak memory of
        any one correlation step for all subjects together (see
        ``estimate_memory_gb``; the estimate covers the correlation step and
        the unreduced results it holds, not the memory of reducers or of
        ``combine`` functions other than ``null_combine``).  It is checked on
        every step, because the (d, d, T) correlation array is allocated
        whether or not the result is then reduced; for large K, raising it is
        the remedy.  Default: 4.0.

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
        When some rows of the analysis kernel have fewer than 3 non-zero
        weights: the correlations there are undefined and set to 0.

    Raises
    ------
    ValueError
        If ``order`` is not a non-negative integer, and as in ``timecorr``
        (subjects of different shapes, non-numeric columns, an unknown
        ``include_timepoints``).
    MemoryError
        Before a correlation step allocates its arrays, if that step's
        estimate exceeds ``max_memory_gb`` (with unreduced delta features,
        every order is checked before anything is computed).  The message gives the estimate and the largest
        number of features that would fit.

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

    single = not isinstance(data, (list, tuple))
    subjects = format_data(data, fill_nan=False)
    T, K = subjects[0].shape
    delta_unreduced = (rfun if delta_rfun is None else delta_rfun) is None

    S = len(subjects)
    if delta_unreduced:
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
    for k in range(1, order + 1):
        d = D[-1][0].shape[1]
        both_paths = k < order or return_delta
        step = _step_gb(d, T, S, 2 if both_paths else 1)
        if step > max_memory_gb:
            fit = _largest_k(1, T, max_memory_gb, S, both_paths)
            raise MemoryError(
                f"The correlation step at order {k} needs about {step:.3g} GB "
                f"({d} input features, T={T}, {S} subject{'s' if S > 1 else ''}), "
                f"more than max_memory_gb={max_memory_gb}. The largest number of "
                f"input features that fits one step is K={fit}. Reduce to fewer "
                "features (ndims, rfun, delta_rfun) or raise max_memory_gb."
            )
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
            X.append(_analysis_path(D[-1], **analysis))
            break
        x, delta = two_path_step(D[-1], delta_rfun=delta_rfun, **analysis)
        X.append(x)
        D.append(delta)

    if single:
        X = [x[0] if isinstance(x, list) else x for x in X]
        D = [d[0] for d in D]
    if return_delta:
        return X, D
    return X
