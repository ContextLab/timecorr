# coding: utf-8

import warnings

import numpy as np

from .events import distinct_event_sets, mask_kernel, resolve_events
from .helpers import (
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
    """
    w = np.abs(weights)
    valid = ~np.isnan(x)
    num = w @ np.where(valid, x, 0.0)
    den = w @ valid.astype(np.float64)
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


def _per_subject_cfun(cfun, data, weights, groups):
    """
    Correlations with one masked kernel per group of subjects sharing events.

    The built-in correlation functions take the list of kernels in one call; a
    custom ``cfun`` is called once per group with the full data, and each subject's
    output is taken from the call made with that subject's kernel.
    """
    S = len(data)
    if cfun in _KERNEL_LIST_CFUNS:
        kernels = [None] * S
        for boundaries, subjects in groups:
            masked = mask_kernel(weights, boundaries)
            for s in subjects:
                kernels[s] = masked
        return cfun(data, kernels)

    out = [None] * S
    for boundaries, subjects in groups:
        result = cfun(data, mask_kernel(weights, boundaries))
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
        your own with the same signature.
    weights_params : dict or None
        Parameters of the kernel, e.g. ``{"var": 100}`` for
        ``gaussian_weights``.  Default: None (the kernel's defaults:
        ``gaussian_params``, ``laplace_params``, ``t_params``, ``eye_params``,
        ``mexican_hat_params``, ...).
    include_timepoints : str
        Which timepoints may contribute to the estimate at each timepoint:
        "all" (default), "backward" (alias "pre": only timepoints at or before
        ``t``), or "forward" (alias "post": only timepoints at or after ``t``).
        Applied to the kernel matrix after it is built (see ``one_sided``).
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
        ("eigenvector_centrality", "pagerank_centrality", "strength"); or a
        callable ``f(list_of_arrays) -> list_of_arrays``.  Model-based
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
        path (``cfun=None``) uses the same restricted kernel.  2-D arrays and
        DataFrames (features to segment into events) raise
        ``NotImplementedError`` in this version.  See
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
        non-zero kernel weights (for example the first timepoints of a
        backward kernel; their correlations are undefined and set to 0);
        other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there);
        timepoints where, because of NaN samples, some pairs of features
        have fewer than 3 observed samples with non-zero kernel weight
        (those correlations are undefined and set to 0); and timepoints
        with zero kernel-weighted variance in every feature (their correlations are set
        to 0).  Never for a delta kernel.
    UserWarning
        With ``cfun=None``, when some timepoints have no weighted, observed
        sample: the smoothed values there are undefined and set to 0.

        With per-subject events, a user-supplied ``cfun`` is called once per
        distinct event set (so it may warn once per set), and the smoothing
        path (``cfun=None``) warns per subject, as it does without events.

    Raises
    ------
    ValueError
        For an unknown ``include_timepoints``, subjects of different shapes,
        non-numeric columns, or a kernel whose rows are all zero; for a
        malformed ``events`` specification; when ``events`` leave every kernel
        row with fewer than 3 non-zero weights (no block is long enough to
        estimate a correlation); and, with per-subject events, when a custom
        ``cfun`` does not return a list with one array per subject.
    NotImplementedError
        If an entry of ``events`` holds features to segment.

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
    call, which holds one T x T kernel per distinct event set in memory.  Any
    other ``cfun`` is called once per distinct event set with the full data
    and one kernel (one T x T kernel at a time); it must return a list with
    one array per subject, and subject s's output is taken from the call made
    with subject s's kernel.

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

    def temporal_filter(T, k):
        k = np.round(k)
        filt = np.eye(T)
        for i in np.arange(1, np.min([np.abs(k) + 1, T])):
            filt = filt + np.eye(T, k=np.abs(i)) + np.eye(T, k=-np.abs(i))
        if k < 0:
            return 1 - filt
        else:
            return filt

    # NaN samples are kept; both the correlation functions and the smoothing
    # path (cfun=None) give them no weight.
    data = format_data(data, fill_nan=False)
    T = data[0].shape[0]

    weights = weights_function(T, weights_params)

    include_timepoints = include_timepoints.lower()
    if include_timepoints == "all":
        if not include_zero:
            weights = weights.copy()
            np.fill_diagonal(weights, 0)
    elif include_timepoints in ("backward", "pre"):
        weights = one_sided(weights, "backward", include_zero)
    elif include_timepoints in ("forward", "post"):
        weights = one_sided(weights, "forward", include_zero)
    else:
        raise ValueError(
            f"Invalid option for include_timepoints: '{include_timepoints}'.  "
            "Must be one of: 'all', 'backward', 'pre', 'forward', or 'post'."
        )

    if not (exclude_timepoints is None):
        weights = np.multiply(temporal_filter(T, exclude_timepoints), weights)

    # event restriction: applied after the include/exclude options; subjects that
    # share events share one masked kernel
    resolved = resolve_events(events, T, len(data), events_params)
    groups = distinct_event_sets(_restricted(resolved), len(data))
    if len(groups) == 1:
        weights = mask_kernel(weights, groups[0][0])

    if cfun:
        if len(groups) == 1:
            corrs = cfun(data, weights)
        else:
            corrs = _per_subject_cfun(cfun, data, weights, groups)
        corrs = reduce(combine(corrs), rfun=rfun)

    else:
        # smoothing path: kernel-weighted mean of the samples, using the same
        # kernel rows (after include_timepoints, include_zero, exclude_timepoints
        # and events) as the correlation path
        if len(groups) == 1:
            smoothed = [_kernel_smooth(x, weights) for x in data]
        else:
            smoothed = [None] * len(data)
            for boundaries, subjects in groups:
                masked = mask_kernel(weights, boundaries)
                for s in subjects:
                    smoothed[s] = _kernel_smooth(data[s], masked)
        corrs = combine(smoothed)

    if isinstance(corrs, list) and len(corrs) == 1:
        return corrs[0]
    return corrs
