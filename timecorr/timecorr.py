# coding: utf-8

import warnings

import numpy as np

from ._kernels import CHUNK, as_kernel, blocks
from .events import distinct_event_sets, mask_kernel, resolve_events
from .helpers import (
    _as_kernel_array,
    _lag_kernel,
    autofc,
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


def _mask_for_subjects(weights, boundaries, subjects):
    """``mask_kernel`` for one event set; an error names the subjects that use it."""
    try:
        return mask_kernel(weights, boundaries)
    except ValueError as err:
        raise ValueError(f"per-subject events of subjects {subjects}: {err}") from err


def _temporal_filter(T, k):
    """
    Band of half-width |k| around the diagonal (k >= 0), or its complement;
    k is rounded to the nearest integer (as for the built-in kernels).
    """
    k = int(np.round(k))
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
):
    """
    The T x T kernel of ``timecorr`` before the events restriction:
    ``weights_function(T, weights_params)`` shaped by ``include_timepoints``,
    ``include_zero`` and ``exclude_timepoints``.  For a built-in kernel
    function it is a ``LagKernel``, whose rows are computed when they are
    read (no T x T array); for any other function, the array it returns.
    """
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


def _event_kernels(weights, resolved, n_subjects):
    """
    ``weights`` restricted to the RESOLVED events as ``timecorr`` restricts
    it: one masked kernel when every subject has the same events, otherwise
    a list of ``n_subjects`` kernels (subjects that share events share one
    array).
    """
    groups = distinct_event_sets(_restricted(resolved), n_subjects)
    if len(groups) == 1:
        return mask_kernel(weights, groups[0][0])
    kernels = [None] * n_subjects
    for boundaries, subjects in groups:
        masked = _mask_for_subjects(weights, boundaries, subjects)
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
):
    """
    Compute dynamic (moment-by-moment) correlations in one or more datasets.

    For each timepoint ``t``, row ``t`` of the kernel matrix
    (``weights[t, :]``) weights every timepoint's contribution, and ``cfun``
    computes kernel-weighted correlations (see ``helpers.wcorr``).  The
    estimator weights the covariance and variance sums as well as the means;
    it differs from Eq. 4 of Owen, Chang & Manning (2021) and from timecorr
    0.2.0, where the kernel set only the means, so numerical results differ
    from 0.2.0.  For higher-order correlations computed as in the paper
    (Fig. 10), see ``timecorr.features``.

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
        the columns where those rows can have weight, so no T x T array is
        allocated and memory grows linearly in T (see Notes).  A function of
        your own is called, and its T x T array is used.
    weights_params : dict or None
        Parameters of the kernel, e.g. ``{"var": 100}`` for
        ``gaussian_weights``.  Default: None (the kernel's defaults:
        ``gaussian_params``, ``laplace_params``, ``t_params``, ``eye_params``,
        ``mexican_hat_params``, ...).
    include_timepoints : str
        Which timepoints may contribute to the estimate at each timepoint:
        "all" (default), "backward" (alias "pre": only timepoints at or before
        ``t``), or "forward" (alias "post": only timepoints at or after ``t``).
        Applied to the kernel after it is built (see ``one_sided``).
    exclude_timepoints : int or None
        Further restricts the kernel by distance from ``t``.  A positive value
        ``x`` keeps only the timepoints at most ``x`` samples from ``t``; a
        negative value ``-x`` removes them (``t`` itself included), keeping
        only timepoints more than ``x`` samples away.  Real values are rounded
        to the nearest integer.  Default: None (no restriction).
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

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        One row per timepoint, each a vectorized correlation matrix (see
        ``mat2vec``: the K diagonal entries, then the K(K-1)/2 entries above
        the diagonal), so each array has shape (T, K * (K + 1) / 2); or the
        reduced features when ``rfun`` is given ((T, K) for model-based
        reduction and graph measures); or the smoothed (T, K) data when
        ``cfun`` is None.  A list (one entry per subject)
        when ``combine`` returns one; a single array otherwise.

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
        raised by ``segment`` for features entries of ``events``.
    RuntimeError, ImportError
        As raised by ``segment`` for features entries of ``events``.

    Notes
    -----
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
    """

    # NaN samples are kept; both the correlation functions and the smoothing
    # path (cfun=None) give them no weight.
    data = format_data(data, fill_nan=False)
    T = data[0].shape[0]

    weights = _shaped_kernel(
        T,
        weights_function,
        weights_params,
        include_timepoints,
        include_zero,
        exclude_timepoints,
    )

    # event restriction: applied after the include/exclude options; subjects that
    # share events share one masked kernel (one array, or a list per subject)
    resolved = resolve_events(events, T, len(data), events_params)
    kernels = _event_kernels(weights, resolved, len(data))
    per_subject = isinstance(kernels, list)

    if cfun:
        if cfun not in _KERNEL_LIST_CFUNS:
            # a correlation function of the user's receives T x T arrays
            if per_subject:
                arrays = {}
                for k in kernels:  # subjects that share events share one array
                    if id(k) not in arrays:
                        arrays[id(k)] = _as_kernel_array(k)
                kernels = [arrays[id(k)] for k in kernels]
            else:
                kernels = _as_kernel_array(kernels)
        if not per_subject or cfun in _KERNEL_LIST_CFUNS:
            corrs = cfun(data, kernels)
        else:
            groups = distinct_event_sets(_restricted(resolved), len(data))
            corrs = _per_subject_cfun(cfun, data, kernels, groups)
        corrs = reduce(combine(corrs), rfun=rfun)

    else:
        # smoothing path: kernel-weighted mean of the samples, using the same
        # kernel rows (after include_timepoints, include_zero, exclude_timepoints
        # and events) as the correlation path
        if per_subject:
            smoothed = [_kernel_smooth(x, k) for x, k in zip(data, kernels)]
        else:
            smoothed = [_kernel_smooth(x, kernels) for x in data]
        corrs = combine(smoothed)

    if isinstance(corrs, list) and len(corrs) == 1:
        return corrs[0]
    return corrs
