# coding: utf-8

import warnings

import numpy as np

from .helpers import (
    format_data,
    gaussian_weights,
    isfc,
    null_combine,
    one_sided,
    reduce,
)


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

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        One row per timepoint, each a vectorized correlation matrix (see
        ``mat2vec``: the K diagonal entries, then the K(K-1)/2 entries above
        the diagonal), or the reduced features when ``rfun`` is given, or the
        smoothed data when ``cfun`` is None.  A list (one entry per subject)
        when ``combine`` returns one; a single array otherwise.

    Warns
    -----
    UserWarning
        When some kernel rows have fewer than 3 non-zero weights (for example
        the first timepoints of a backward kernel): the correlations there are
        undefined and set to 0.

    Raises
    ------
    ValueError
        For an unknown ``include_timepoints``, subjects of different shapes,
        non-numeric columns, or a kernel whose rows are all zero.

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

    if cfun:
        corrs = reduce(combine(cfun(data, weights)), rfun=rfun)

    else:
        # smoothing path: kernel-weighted mean of the samples, using the same
        # kernel rows (after include_timepoints, include_zero and
        # exclude_timepoints) as the correlation path
        corrs = combine([_kernel_smooth(x, weights) for x in data])

    if isinstance(corrs, list) and len(corrs) == 1:
        return corrs[0]
    return corrs
