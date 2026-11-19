# coding: utf-8
import inspect
import math
import numbers
import os
import sys
import warnings

import datawrangler as dw
import hypertools as hyp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.linalg import toeplitz
from scipy.special import gamma
from scipy.stats import ttest_1samp as ttest

try:  # polars is optional
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

try:  # hypertools' registry of reducer names (read to validate names up front)
    from hypertools.reduce.common import AUTOENCODER_NAMES as _HYP_AUTOENCODERS
    from hypertools.reduce.common import REDUCERS as _HYP_REDUCERS
except ImportError:  # pragma: no cover - other hypertools layouts
    _HYP_AUTOENCODERS, _HYP_REDUCERS = (), {}

# Default parameters of the kernel functions below.  Each kernel function uses
# its dictionary when ``params`` is omitted or None.  The ``#:`` comments are
# the documentation that Sphinx shows for each dictionary.

#: Default ``params`` of :func:`gaussian_weights`: ``var``, the variance of the
#: Gaussian in squared samples.
gaussian_params = {"var": 100}
#: Default ``params`` of :func:`laplace_weights`: ``scale``, the scale of the
#: Laplace density in samples.
laplace_params = {"scale": 100}
#: Default ``params`` of :func:`eye_weights` (the delta kernel has no parameters).
eye_params = {}
#: Default ``params`` of :func:`t_weights`: ``df``, the degrees of freedom of
#: Student's t density.
t_params = {"df": 100}
#: Default ``params`` of :func:`mexican_hat_weights`: ``sigma``, the width of
#: the wavelet in samples.
mexican_hat_params = {"sigma": 10}
#: Default ``params`` of :func:`uniform_weights` (the uniform kernel has no
#: parameters).
uniform_params = {}
#: Default ``params`` of :func:`boxcar_weights`: ``width``, the total width of
#: the window in samples.
boxcar_params = {"width": 10}


def one_sided(weights, direction, include_zero=True):
    """
    Restrict a T x T kernel to the past or the future of each timepoint.

    Row ``t`` of the kernel (``weights[t, :]``) is the kernel used to estimate
    at timepoint ``t``.

    Parameters
    ----------
    weights : numpy.ndarray
        T x T kernel matrix.
    direction : {'backward', 'forward'}
        'backward' keeps ``weights[t, tau]`` for ``tau <= t``; 'forward' keeps
        it for ``tau >= t``.  All other entries are set to zero.
    include_zero : bool, optional
        If False, the diagonal is also zeroed (``tau < t`` for 'backward',
        ``tau > t`` for 'forward').  Default: True

    Returns
    -------
    numpy.ndarray
        T x T one-sided kernel (a new array).

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> w = tc.one_sided(np.ones([4, 4]), "backward")
    >>> print(w[1])
    [1. 1. 0. 0.]
    >>> print(tc.one_sided(np.ones([4, 4]), "forward", include_zero=False)[1])
    [0. 0. 1. 1.]
    """
    weights = np.asarray(weights)
    if direction not in ("backward", "forward"):
        raise ValueError(
            f"Invalid direction: {direction!r}.  Must be 'backward' or 'forward'."
        )
    k = 0 if include_zero else (-1 if direction == "backward" else 1)
    if direction == "backward":
        return np.tril(weights, k)
    return np.triu(weights, k)


def _finish_kernel(w, direction, include_zero):
    """Apply the optional direction and diagonal handling shared by all kernels."""
    if direction is not None:
        return one_sided(w, direction, include_zero)
    if not include_zero:
        w = w.copy()
        np.fill_diagonal(w, 0)
    return w


def gaussian_weights(T, params=gaussian_params, direction=None, include_zero=True):
    """
    Generate Gaussian weighting function for dynamic correlations.

    This function creates a time-varying weighting matrix where each timepoint
    receives weights according to a Gaussian distribution centered at that timepoint.
    Useful for computing smooth dynamic correlations.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Kernel parameters.  Default: ``gaussian_params`` (``{'var': 100}``);
        None also uses ``gaussian_params``.

        - ``var`` : float, the variance of the Gaussian in squared samples
          (its standard deviation is ``sqrt(var)`` samples).
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T matrix of Gaussian weights, where weights[i,j] represents
        the weight given to timepoint j when computing correlations at timepoint i

    Raises
    ------
    KeyError
        If ``params`` lacks the ``var`` key.

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.gaussian_weights(50, {'var': 10})
    >>> print(weights.shape)
    (50, 50)
    """
    if params is None:
        params = gaussian_params

    c1 = np.divide(1, np.sqrt(2 * math.pi * params["var"]))
    c2 = np.divide(-1, 2 * params["var"])
    sqdiffs = toeplitz(np.arange(T) ** 2)
    return _finish_kernel(c1 * np.exp(c2 * sqdiffs), direction, include_zero)


def laplace_weights(T, params=laplace_params, direction=None, include_zero=True):
    """
    Generate Laplace (exponential) weighting function for dynamic correlations.

    This function creates a time-varying weighting matrix where each timepoint
    receives weights according to a Laplace distribution centered at that timepoint.
    Provides sharper temporal localization compared to Gaussian weights.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Kernel parameters.  Default: ``laplace_params`` (``{'scale': 100}``);
        None also uses ``laplace_params``.

        - ``scale`` : float, the scale ``b`` of the Laplace density, in
          samples: ``weights[t, tau] = exp(-|t - tau| / b) / (2 b)``.
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T matrix of Laplace weights, where weights[i,j] represents
        the weight given to timepoint j when computing correlations at timepoint i

    Raises
    ------
    KeyError
        If ``params`` lacks the ``scale`` key.

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.laplace_weights(50, {'scale': 5})
    >>> print(weights.shape)
    (50, 50)
    """
    if params is None:
        params = laplace_params

    absdiffs = toeplitz(np.arange(T))
    w = np.multiply(
        np.divide(1, 2 * params["scale"]), np.exp(-np.divide(absdiffs, params["scale"]))
    )
    return _finish_kernel(w, direction, include_zero)


def eye_weights(T, params=eye_params, direction=None, include_zero=True):
    """
    Generate identity (delta) weighting function for dynamic correlations.

    This function creates an identity matrix for weighting, where each timepoint
    only receives weight from itself. Useful for computing instantaneous correlations
    without temporal smoothing.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Unused (the delta kernel has no parameters); accepted so that every
        kernel function has the same signature.  Default: ``eye_params``
        (``{}``, no keys).
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T identity matrix where weights[i,j] = 1 if i==j, else 0

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.eye_weights(50, {})
    >>> print(weights.shape)
    (50, 50)
    >>> print(weights[0, 0])
    1.0
    >>> print(weights[0, 1])
    0.0
    """

    return _finish_kernel(np.eye(T), direction, include_zero)


def uniform_weights(T, params=uniform_params, direction=None, include_zero=True):
    """
    Generate a uniform weighting function for dynamic correlations.

    Every timepoint receives the same weight (1) at every timepoint, so the
    resulting correlations are static (non-dynamic) unless a direction is given.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Unused (the uniform kernel has no parameters); accepted so that every
        kernel function has the same signature.  Default: ``uniform_params``
        (``{}``, no keys).
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T matrix of ones (restricted according to ``direction`` and
        ``include_zero``), where weights[i,j] is the weight given to timepoint j
        when computing correlations at timepoint i

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.uniform_weights(5)
    >>> print(weights.shape)
    (5, 5)
    >>> print(tc.uniform_weights(3, direction="backward"))
    [[1. 0. 0.]
     [1. 1. 0.]
     [1. 1. 1.]]
    """
    return _finish_kernel(np.ones([T, T]), direction, include_zero)


def t_weights(T, params=t_params, direction=None, include_zero=True):
    """
    Generate Student's t weighting function for dynamic correlations.

    The weight given to timepoint ``tau`` when estimating at timepoint ``t`` is
    Student's t density (unit scale) evaluated at the lag ``t - tau``:

    ``weights[t, tau] = c * (1 + (t - tau)**2 / df) ** (-(df + 1) / 2)``,

    with ``c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))``
    (``scipy.stats.t.pdf(t - tau, df)``).  Smaller ``df`` gives heavier tails
    (``df=1`` is the Cauchy density); as ``df`` grows the kernel approaches the
    standard normal density.  Because the scale is fixed at one sample, this
    kernel is narrow: most of its weight lies within a few samples of ``t``.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Kernel parameters.  Default: ``t_params`` (``{'df': 100}``); None also
        uses ``t_params``.

        - ``df`` : float, the degrees of freedom of the t density (distances
          in samples).  Smaller values give heavier tails.
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T matrix of t weights, where weights[i,j] is the weight given to
        timepoint j when computing correlations at timepoint i

    Raises
    ------
    KeyError
        If ``params`` lacks the ``df`` key.

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.t_weights(50, {'df': 10})
    >>> print(weights.shape)
    (50, 50)
    """
    if params is None:
        params = t_params

    c1 = np.divide(
        gamma((params["df"] + 1) / 2),
        np.sqrt(params["df"] * math.pi) * gamma(params["df"] / 2),
    )
    c2 = np.divide(-(params["df"] + 1), 2)

    sqdiffs = toeplitz(np.arange(T) ** 2)
    w = np.multiply(c1, np.power(1 + np.divide(sqdiffs, params["df"]), c2))
    return _finish_kernel(w, direction, include_zero)


def mexican_hat_weights(
    T, params=mexican_hat_params, direction=None, include_zero=True
):
    """
    Generate Mexican Hat (Ricker) weighting function for dynamic correlations.

    This function creates a time-varying weighting matrix where each timepoint
    receives weights according to a Mexican Hat wavelet centered at that timepoint.
    Useful for capturing temporal dynamics and transitions in correlations.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Kernel parameters.  Default: ``mexican_hat_params``
        (``{'sigma': 10}``); None also uses ``mexican_hat_params``.

        - ``sigma`` : float, the width of the wavelet in samples.  Weights
          are positive within ``sigma`` samples of ``t``, zero at exactly
          ``sigma`` samples, and negative beyond.
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T matrix of Mexican Hat weights, where weights[i,j] represents
        the weight given to timepoint j when computing correlations at timepoint i

    Raises
    ------
    KeyError
        If ``params`` lacks the ``sigma`` key.

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.mexican_hat_weights(50, {'sigma': 5})
    >>> print(weights.shape)
    (50, 50)
    """
    if params is None:
        params = mexican_hat_params

    absdiffs = toeplitz(np.arange(T))
    sqdiffs = toeplitz(np.arange(T) ** 2)

    a = np.divide(2, np.sqrt(3 * params["sigma"]) * np.power(math.pi, 0.25))
    b = 1 - np.power(np.divide(absdiffs, params["sigma"]), 2)
    c = np.exp(-np.divide(sqdiffs, 2 * np.power(params["sigma"], 2)))

    return _finish_kernel(np.multiply(a, np.multiply(b, c)), direction, include_zero)


def boxcar_weights(T, params=boxcar_params, direction=None, include_zero=True):
    """
    Generate a boxcar (moving window) weighting function for dynamic correlations.

    Timepoints fewer than ``width / 2`` samples from each timepoint receive
    weight 1; all others receive weight 0.

    Parameters
    ----------
    T : int
        Number of timepoints in the timeseries
    params : dict or None, optional
        Kernel parameters.  Default: ``boxcar_params`` (``{'width': 10}``);
        None also uses ``boxcar_params``.

        - ``width`` : float, the total width of the window in samples:
          ``weights[t, tau]`` is 1 where ``|t - tau| < width / 2``.
    direction : {None, 'backward', 'forward'}, optional
        If 'backward', only timepoints at or before each timepoint receive
        weight (``weights[t, tau] = 0`` for ``tau > t``); if 'forward', only
        timepoints at or after it do.  Default: None (two-sided kernel).
    include_zero : bool, optional
        If False, the diagonal is zeroed so that timepoint t does not
        contribute to its own estimate.  Default: True

    Returns
    -------
    numpy.ndarray
        T x T matrix of zeros and ones, where weights[i,j] is the weight given
        to timepoint j when computing correlations at timepoint i

    Raises
    ------
    KeyError
        If ``params`` lacks the ``width`` key.

    Examples
    --------
    >>> import timecorr as tc
    >>> weights = tc.boxcar_weights(6, {'width': 4})
    >>> print(weights[2])
    [0. 1. 1. 1. 0. 0.]
    """
    if params is None:
        params = boxcar_params

    w = np.multiply(toeplitz(np.arange(T)) < params["width"] / 2.0, 1.0)
    return _finish_kernel(w, direction, include_zero)


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _warn_user(message, category=UserWarning):
    """
    Warn with the location of the first caller outside the timecorr package,
    so the warning points at the user's line however deep the call is.
    """
    frame = sys._getframe(1)
    level = 2
    while frame is not None and (
        os.path.dirname(os.path.abspath(frame.f_code.co_filename)) == _PACKAGE_DIR
    ):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)


def _kernel_row_problems(weights):
    """
    Masks ``(undefined, low)`` of the kernel's rows, or None for a delta or an
    all-zero kernel (NaN weights count as 0).

    ``undefined``: fewer than 3 non-zero weights (two samples always give
    r = +-1).  ``low``: the remaining rows whose effective sample size
    ``(sum|w|)**2 / sum(w**2)`` is below 2.
    """
    w = np.abs(np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0))
    support = np.count_nonzero(w, axis=1)
    if np.all(support == 0) or np.all(support == 1):
        return None
    undefined = support < 3
    ess = _safe_divide(w.sum(axis=1) ** 2, (w**2).sum(axis=1))
    return undefined, ~undefined & (ess < 2)


def _estimator_message(weights, cancelled=None, sparse=None):
    """
    The one warning of a correlation call, or None.

    It names the kernel rows with fewer than 3 non-zero weights (undefined, set
    to 0), the other rows whose effective sample size is below 2, the
    timepoints (``sparse``, a boolean mask) where some pairs of features have
    fewer than 3 observed samples with non-zero weight because of NaN samples
    (undefined, set to 0), and the timepoints (``cancelled``, a boolean mask)
    where every correlation was set to 0 because the kernel-weighted variances
    there are zero.
    """
    T = np.shape(weights)[0]
    parts = []
    problems = _kernel_row_problems(weights)
    if problems is not None:
        undefined, low = problems
        if undefined.any():
            parts.append(
                f"{int(undefined.sum())} of {T} timepoints have fewer than 3 "
                "non-zero kernel weights; their correlations are undefined and "
                "set to 0."
            )
        if low.any():
            parts.append(
                f"{int(low.sum())} of {T} timepoints have an effective sample size "
                "((sum |w|)**2 / sum w**2 over the kernel row) below 2; the kernel "
                "is too narrow to estimate correlations reliably there."
            )
    if sparse is not None and np.any(sparse):
        parts.append(
            f"{int(np.sum(sparse))} of {T} timepoints have, for some pairs of "
            "features, fewer than 3 observed (non-NaN) samples with non-zero "
            "kernel weight; those correlations are undefined and set to 0."
        )
    if cancelled is not None and np.any(cancelled):
        parts.append(
            f"{int(np.sum(cancelled))} of {T} timepoints have zero kernel-weighted "
            "variance in every feature (the data are constant within the kernel, "
            "or the kernel is too narrow); their correlations are set to 0."
        )
    return " ".join(parts) if parts else None


def _weights_alias(name, value, alias, alias_value):
    """``value``, or ``alias_value`` when only the alias was given; both raise TypeError."""
    if alias_value is None:
        return value
    if value is not None:
        raise TypeError(f"Pass {name} or its alias {alias}, not both.")
    return alias_value


def _is_polars(x):
    return pl is not None and isinstance(x, (pl.DataFrame, pl.Series))


def _check_numeric(x):
    """Raise ValueError naming the first non-numeric column of a DataFrame."""
    if isinstance(x, pd.DataFrame):
        cols = list(zip(x.columns, x.dtypes))
        numeric = [
            pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt)
            for _, dt in cols
        ]
    elif isinstance(x, pd.Series):
        cols = [(x.name, x.dtype)]
        numeric = [pd.api.types.is_numeric_dtype(x.dtype)]
    elif _is_polars(x):
        if isinstance(x, pl.Series):
            cols = [(x.name, x.dtype)]
        else:
            cols = list(zip(x.columns, x.dtypes))
        numeric = [dt.is_numeric() or dt == pl.Boolean for _, dt in cols]
    else:
        return
    for (name, dt), ok in zip(cols, numeric):
        if not ok:
            raise ValueError(
                f"Column {name!r} has non-numeric type {dt}; timecorr requires "
                "numeric data."
            )


def _as_two_dimensional(x):
    """Turn one subject's input into something 2-D, or raise ValueError."""
    if isinstance(x, pd.Series) or (pl is not None and isinstance(x, pl.Series)):
        return x.to_frame()
    if isinstance(x, pd.DataFrame) or _is_polars(x):
        return x
    a = np.asarray(x)
    if a.ndim == 0:
        raise ValueError(
            "Each subject's data must be a timepoints-by-features matrix (or a "
            f"1-D array of timepoints); got a 0-dimensional input {x!r}."
        )
    if a.ndim == 1:
        return a[:, np.newaxis]
    if a.ndim > 2:
        raise ValueError(
            "Each subject's data must be a timepoints-by-features matrix; got an "
            f"array with {a.ndim} dimensions, shape {a.shape}.  Pass a list "
            "with one 2-D array per subject instead."
        )
    return a


def format_data(data, fill_nan=True):
    """
    Convert input data to a list of writable float64 timepoints-by-features arrays.

    Inputs are standardized with pydata-wrangler, so numpy arrays, pandas
    DataFrames, polars DataFrames, and lists of these (possibly mixed) are
    supported.  A 1-D array or a pandas or polars Series is one feature
    observed over its timepoints and becomes a (timepoints, 1) array.  The
    caller's data are never modified.  By default NaNs in the output are
    replaced with zeros.  Every returned array is C-contiguous, whatever the
    input's memory layout, so equal values give bit-identical results.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, Series, or list of these
        One timepoints-by-features observations matrix per subject (or one
        1-D array or Series of timepoints per subject).
    fill_nan : bool, optional
        If True (default), replace NaNs with zeros.  If False, keep them (the
        correlation functions give NaN samples no weight).

    Returns
    -------
    list of numpy.ndarray
        One new float64 (timepoints, features) array per subject.

    Raises
    ------
    ValueError
        If ``data`` is an empty list, a DataFrame has a non-numeric column,
        an input is 0-dimensional or has more than two dimensions, or
        subjects differ in their number of timepoints or features.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from timecorr.helpers import format_data
    >>> out = format_data(pd.DataFrame(np.random.randn(10, 3)))
    >>> out[0].shape
    (10, 3)
    >>> bool(np.isnan(format_data(np.array([[np.nan, 1.0]]), fill_nan=False)[0][0, 0]))
    True
    >>> format_data(pd.Series(np.arange(5.0)))[0].shape
    (5, 1)
    """
    items = list(data) if isinstance(data, (list, tuple)) else [data]
    if not items:
        raise ValueError("data must contain at least one subject; got an empty list.")
    for x in items:
        _check_numeric(x)

    out = []
    for x in items:
        w = dw.wrangle(_as_two_dimensional(x))
        if _is_polars(w):
            w = w.to_numpy()
        # order="C": the memory layout changes floating-point rounding in the
        # matrix products downstream, and pandas, polars and numpy inputs can
        # arrive in different layouts.
        a = np.array(w, dtype=np.float64, order="C", copy=True)
        if a.ndim != 2:
            raise ValueError(
                "Each subject's data must be a timepoints-by-features matrix; "
                f"got shape {a.shape} after conversion."
            )
        if fill_nan:
            a[np.isnan(a)] = 0
        out.append(a)

    shapes = [a.shape for a in out]
    if any(s != shapes[0] for s in shapes):
        raise ValueError(
            "All subjects must have the same number of timepoints and features; "
            f"got shapes {shapes}."
        )
    return out


def _is_empty(dict):
    if not bool(dict):
        return True
    return False


# Relative tolerance below which a kernel-weighted variance counts as zero.  The
# variance is compared with the kernel-weighted mean square of the (globally
# centered) feature, so the test does not depend on the units of the data.
_ZERO_VARIANCE_TOL = 1e-10

# Size, in float64 elements, of each temporary block used by wcorr (64 MB).  The
# estimator holds a few such blocks at once, so peak extra memory stays well
# under 1 GB regardless of the number of features.
_WCORR_BLOCK_ELEMENTS = 2**23

# Relative round-off of removing a column's mean (a few float64 spacings).
_ROUND_OFF = 64 * np.finfo(np.float64).eps

# Values at most this far outside [-1, 1] are correlations with round-off error
# (clipped to +-1), not co-fluctuations.
_R_TOLERANCE = 1e-12


def _safe_divide(num, den):
    """Elementwise num / den, with 0 wherever den is 0."""
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _nan_center(x):
    """Subtract each column's mean over its non-NaN samples (0 for all-NaN columns)."""
    valid = ~np.isnan(x)
    total = np.where(valid, x, 0.0).sum(axis=0)
    return x - _safe_divide(total, valid.sum(axis=0))


def _zscore_columns(x):
    """
    Population z-scores of each column; NaN samples and constant columns give 0.

    A column is constant when its standard deviation is at most 1e-10 of its
    largest centered value, or within the round-off of removing its mean (a
    few float spacings of its largest raw value), so a large constant offset
    neither zeroes a varying column nor turns a constant one into signal.
    """
    valid = ~np.isnan(x)
    magnitude = np.where(valid, np.abs(x), 0.0).max(axis=0)
    x0 = np.where(valid, _nan_center(x), 0.0)
    sd = np.sqrt(_safe_divide((x0**2).sum(axis=0), valid.sum(axis=0)))
    constant = (sd <= _ZERO_VARIANCE_TOL * np.abs(x0).max(axis=0)) | (
        sd <= _ROUND_OFF * magnitude
    )
    return _safe_divide(x0, np.where(constant, 0.0, sd))


def _delta_zscores(a, b, weights):
    """
    Z-scored columns of ``a`` and ``b`` at each kernel row's non-zero position
    (a true delta kernel: one non-zero weight per row), shapes (T', Ka) and
    (T', Kb).
    """
    idx = np.argmax(weights != 0, axis=1)
    return _zscore_columns(a)[idx], _zscore_columns(b)[idx]


def _delta_wcorr(a, b, weights):
    """Co-fluctuation estimate for a true delta kernel (one non-zero weight per row)."""
    za, zb = _delta_zscores(a, b, weights)
    return np.einsum("ti,tj->ijt", za, zb)


def _wcorr_block_rows(T, Kb, n_products=1):
    """Rows of ``a`` per block in ``_wcorr_blocks`` (each row pairs with Kb columns)."""
    return max(1, _WCORR_BLOCK_ELEMENTS // (n_products * T * Kb))


def _wcorr_inputs(a, b, weights):
    """
    The estimator's inputs as float arrays: ``a`` and ``b`` as (T, K) matrices
    and a copy of ``weights`` with NaN weights set to 0.  Returns
    ``(a, b, weights, support)``, where ``support`` counts each kernel row's
    non-zero weights.

    Raises ``ValueError`` if the shapes do not match (T rows in ``a`` and
    ``b``, T columns in ``weights``) or if every row of the kernel is zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]
    weights = np.array(weights, dtype=np.float64, copy=True)
    weights[np.isnan(weights)] = 0
    if a.shape[0] != b.shape[0] or weights.ndim != 2 or weights.shape[1] != a.shape[0]:
        raise ValueError(
            "a and b must have the same number of rows (timepoints) T, and weights "
            f"must have T columns; got a {a.shape}, b {b.shape}, weights {weights.shape}."
        )

    support = np.count_nonzero(weights, axis=1)
    if np.all(support == 0):
        raise ValueError(
            "Every row of the kernel is zero; no correlation can be estimated."
        )
    return a, b, weights, support


def _wcorr_setup(a, b, weights, autocorrelation):
    """
    The terms shared by every block of ``_wcorr_block``: the kernel, the
    centered data and, without NaN samples, each feature's kernel-weighted
    moments (computed once; with ``autocorrelation`` those of ``a`` serve for
    ``b`` too).  Returns a dict.
    """
    W = weights
    A = np.abs(W)
    a = _nan_center(a)
    b = _nan_center(b)
    state = {"T": a.shape[0], "W": W, "A": A, "a": a, "b": b}
    state["active"] = A.sum(axis=1) > 0
    state["has_nan"] = bool(np.isnan(a).any() or np.isnan(b).any())

    if not state["has_nan"]:
        s = A.sum(axis=1)[:, np.newaxis]
        state["sw"] = W.sum(axis=1)[:, np.newaxis]

        def moments(x):
            mean = _safe_divide(A @ x, s)
            msq = A @ (x**2)
            var = msq - s * mean**2
            zero = var <= _ZERO_VARIANCE_TOL * msq
            return mean, W @ x, np.where(zero, 0.0, var), zero

        state["ma"], state["wa"], state["va"], state["za"] = moments(a)
        if autocorrelation:
            state["mb"], state["wb"], state["vb"], state["zb"] = (
                state["ma"],
                state["wa"],
                state["va"],
                state["za"],
            )
        else:
            state["mb"], state["wb"], state["vb"], state["zb"] = moments(b)
        state["n_products"] = 1
    else:
        state["ok_a"] = ~np.isnan(a)
        state["ok_b"] = ~np.isnan(b)
        state["a0"] = np.where(state["ok_a"], a, 0.0)
        state["b0"] = np.where(state["ok_b"], b, 0.0)
        # per pair and kernel row: the number of timepoints with non-zero weight
        # where both features are observed
        state["support"] = (A > 0).astype(np.float64)
        state["n_products"] = 10
    return state


def _wcorr_block(state, i0, i1, j0):
    """
    Kernel-weighted correlations between rows ``i0:i1`` of ``a`` and columns
    ``j0:`` of ``b`` (``state`` from ``_wcorr_setup``).

    Returns ``(r, undefined, zero_all, few_any)``: ``r`` of shape
    (T', i1 - i0, Kb - j0), with undefined entries set to 0 and the rest
    clipped to [-1, 1]; the mask of the undefined entries; per kernel row,
    whether every pair in the block has zero variance; and per kernel row,
    whether some pair has fewer than 3 observed samples with non-zero weight
    (always False without NaN samples).
    """
    T = state["T"]
    W = state["W"]
    A = state["A"]
    Kb = state["b"].shape[1]
    shape = (W.shape[0], i1 - i0, Kb - j0)

    if not state["has_nan"]:
        a, b = state["a"], state["b"]
        ma, mb, wa, wb = state["ma"], state["mb"], state["wa"], state["wb"]
        prod = (a[:, i0:i1, np.newaxis] * b[:, np.newaxis, j0:]).reshape(T, -1)
        cross = (W @ prod).reshape(shape)
        ma_i = ma[:, i0:i1, np.newaxis]
        mb_j = mb[:, np.newaxis, j0:]
        num = (
            cross
            - ma_i * wb[:, np.newaxis, j0:]
            - mb_j * wa[:, i0:i1, np.newaxis]
            + ma_i * mb_j * state["sw"][:, :, np.newaxis]
        )
        var_a = state["va"][:, i0:i1, np.newaxis]
        var_b = state["vb"][:, np.newaxis, j0:]
        zero = state["za"][:, i0:i1, np.newaxis] | state["zb"][:, np.newaxis, j0:]
        undefined = zero
        few_any = np.zeros(shape[0], dtype=bool)
    else:
        ok_a, ok_b = state["ok_a"], state["ok_b"]
        pair_ok = ok_a[:, i0:i1, np.newaxis] & ok_b[:, np.newaxis, j0:]
        x = state["a0"][:, i0:i1, np.newaxis] * pair_ok
        y = state["b0"][:, np.newaxis, j0:] * pair_ok
        n = x[0].size
        terms = np.concatenate(
            [t.reshape(T, -1) for t in (pair_ok.astype(float), x, y, x**2, y**2)],
            axis=1,
        )
        s0, sx, sy, sxx, syy = np.split(A @ terms, 5, axis=1)
        w0, wx, wy = np.split(W @ terms[:, : 3 * n], 3, axis=1)
        wxy = W @ (x * y).reshape(T, -1)
        mx = _safe_divide(sx, s0)
        my = _safe_divide(sy, s0)
        num = (wxy - mx * wy - my * wx + mx * my * w0).reshape(shape)
        var_a = sxx - s0 * mx**2
        var_b = syy - s0 * my**2
        zero = (var_a <= _ZERO_VARIANCE_TOL * sxx) | (var_b <= _ZERO_VARIANCE_TOL * syy)
        var_a, var_b, zero = (v.reshape(shape) for v in (var_a, var_b, zero))
        # two samples always give r = +-1: fewer than 3 is undefined
        few = (state["support"] @ terms[:, :n]).reshape(shape) < 3
        few_any = few.reshape(shape[0], -1).any(axis=1)
        undefined = zero | few

    zero_all = zero.reshape(shape[0], -1).all(axis=1)
    den = np.sqrt(np.where(undefined, 1.0, var_a * var_b))
    r = np.where(undefined, 0.0, np.clip(num / den, -1.0, 1.0))
    return r, undefined, zero_all, few_any


def _symmetrize_block(r, undefined, offset):
    """
    In place, for an autocorrelation block from ``_wcorr_block`` whose rows
    ``i0:i1`` start ``offset`` columns after its first column: set the
    diagonal to 1 (0 where undefined) and replace the square of columns
    ``offset:offset + (i1 - i0)`` by the mean of it and its transpose.
    """
    c = r.shape[1]
    diag = np.arange(c)
    r[:, diag, offset + diag] = np.where(undefined[:, diag, offset + diag], 0.0, 1.0)
    square = r[:, :, offset : offset + c]
    r[:, :, offset : offset + c] = (square + square.transpose(0, 2, 1)) / 2


def _wcorr_blocks(a, b, weights, autocorrelation):
    """
    Kernel-weighted correlations, computed with matrix products over blocks of
    feature pairs.  NaN samples are excluded pairwise (they carry no weight).

    Returns ``(out, cancelled, sparse)``: the (Ka, Kb, T) correlations; a mask
    of the kernel rows with any weight where every correlation was set to 0
    because the variances there are zero; and a mask of the kernel rows with
    any weight where some pair of features has fewer than 3 observed samples
    with non-zero weight (NaN samples; those correlations are set to 0, the
    diagonal of an autocorrelation included).  Without NaN samples ``sparse``
    is all False.
    """
    T, Ka = a.shape
    Kb = b.shape[1]
    state = _wcorr_setup(a, b, weights, autocorrelation)
    out = np.zeros([Ka, Kb, weights.shape[0]])
    cancelled = state["active"].copy()
    sparse = np.zeros_like(cancelled)

    rows_per_block = _wcorr_block_rows(T, Kb, state["n_products"])
    for i0 in range(0, Ka, rows_per_block):
        i1 = min(Ka, i0 + rows_per_block)
        j0 = i0 if autocorrelation else 0
        r, undefined, zero_all, few_any = _wcorr_block(state, i0, i1, j0)
        cancelled &= zero_all
        sparse |= few_any

        if autocorrelation:
            _symmetrize_block(r, undefined, 0)
            out[i0:i1, i0:, :] = r.transpose(1, 2, 0)
            out[i0:, i0:i1, :] = r.transpose(2, 1, 0)
        else:
            out[i0:i1, :, :] = r.transpose(1, 2, 0)

    # rows without weight (undefined kernel rows) are reported as such
    sparse &= state["active"]
    return out, cancelled, sparse


def _corr_row_blocks(a, b, weights, symmetric, rows_per_block=None):
    """
    The rows of ``_wcorr(a, b, weights)`` in blocks, without the full array.

    ``symmetric=True`` declares that ``a`` and ``b`` are the same data (an
    autocorrelation): each block's diagonal is set to 1 (0 where undefined)
    and the entries within the block's own rows and columns are averaged with
    their transpose.  Entries outside that square are computed once per
    orientation, so they equal ``_wcorr``'s mirrored values up to round-off.

    The kernel is prepared as in ``_wcorr``: NaN weights count as 0, a kernel
    with exactly one non-zero weight in every row gives the co-fluctuations
    of ``_delta_wcorr``, and otherwise rows with fewer than 3 non-zero
    weights are set to 0.  ``weights`` is not modified.  The inputs are
    checked when this function is called, before the first block.

    Returns a generator of ``(i0, i1, R, cancelled_all, sparse_any)``, with
    consecutive blocks covering rows ``0:Ka`` of ``a`` (``rows_per_block``
    rows each, the last possibly fewer; ``None`` chooses the block size used
    by ``_wcorr_blocks``).  ``R`` has shape (i1 - i0, Kb, T') and equals rows
    ``i0:i1`` of ``_wcorr``'s output.  ``cancelled_all`` marks the kernel rows
    with any weight where every correlation of the block was set to 0 because
    the variances are zero; ``sparse_any`` marks those where some pair of the
    block has fewer than 3 observed samples with non-zero weight.  AND-ing
    ``cancelled_all`` and OR-ing ``sparse_any`` over all blocks gives the
    masks ``_wcorr(..., return_cancelled=True)`` returns (all False for the
    delta case).

    Raises ``ValueError`` as ``_wcorr`` does, and if ``symmetric`` is True but
    ``a`` and ``b`` differ in shape.
    """
    a, b, weights, support = _wcorr_inputs(a, b, weights)
    if symmetric and a.shape != b.shape:
        raise ValueError(
            "symmetric=True requires a and b of the same shape; "
            f"got a {a.shape}, b {b.shape}."
        )
    if rows_per_block is not None and int(rows_per_block) < 1:
        raise ValueError(f"rows_per_block must be at least 1; got {rows_per_block}.")
    T, Ka = a.shape
    Kb = b.shape[1]

    if np.all(support == 1):
        za, zb = _delta_zscores(a, b, weights)
        step = int(rows_per_block or _wcorr_block_rows(T, Kb))

        def delta_blocks():
            for i0 in range(0, Ka, step):
                i1 = min(Ka, i0 + step)
                none = np.zeros(weights.shape[0], dtype=bool)
                R = np.einsum("ti,tj->ijt", za[:, i0:i1], zb)
                yield i0, i1, R, none, none.copy()

        return delta_blocks()

    weights[support < 3] = 0
    state = _wcorr_setup(a, b, weights, symmetric)
    step = int(rows_per_block or _wcorr_block_rows(T, Kb, state["n_products"]))
    active = state["active"]

    def blocks():
        for i0 in range(0, Ka, step):
            i1 = min(Ka, i0 + step)
            r, undefined, zero_all, few_any = _wcorr_block(state, i0, i1, 0)
            if symmetric:
                _symmetrize_block(r, undefined, i0)
            yield i0, i1, r.transpose(1, 2, 0), active & zero_all, active & few_any

    return blocks()


def wcorr(a, b, weights):
    """
    Compute kernel-weighted correlations between two sets of observations.

    Row ``t`` of ``weights`` (``weights[t, :]``) is the kernel used to estimate
    the correlations at timepoint ``t``.  For each ``t`` and each pair of
    features,

    ``r = sum_tau w d_a d_b / sqrt(sum_tau |w| d_a**2 * sum_tau |w| d_b**2)``,

    where ``w = weights[t, tau]`` and ``d = x - sum_tau(|w| x) / sum_tau |w|``
    are deviations from the kernel-weighted mean.  For non-negative kernels
    this is the weighted Pearson correlation; for signed kernels (for example
    the Mexican hat) it stays within [-1, 1].  The diagonal of an
    autocorrelation (``a`` and ``b`` equal) is set to exactly 1 (0 for a
    feature that is constant there).  For non-negative kernels the formula
    gives 1 there too, but for signed kernels the diagonal is an exception:
    the formula's value can be below 1, even negative (for the Mexican hat,
    values near -1 occur), and it is what a cross-correlation (``a`` and
    ``b`` different) returns for a pair of identical columns.

    Special cases:

    - A true delta kernel (every row has exactly one non-zero weight) gives the
      co-fluctuation ``z_a(tau_t) z_b(tau_t)^T``, where each column is z-scored
      over all timepoints and ``tau_t`` is the non-zero position in row ``t``.
      Averaged over timepoints this equals the static correlation.
    - Otherwise, a row with fewer than 3 non-zero weights is undefined (two
      samples always give r = +-1, which carries no information): its
      correlations are set to 0.  If every row is all zeros, ``ValueError``
      is raised.
    - NaN samples carry no weight: each pair of features uses only the
      timepoints where both are observed.  A pair with fewer than 3 observed
      samples with non-zero weight in a kernel row is undefined there, like a
      row with fewer than 3 weights: its correlation is set to 0 (the diagonal
      of an autocorrelation included).  NaN kernel weights count as 0.
    - A feature is treated as constant at timepoint ``t`` when its
      kernel-weighted variance there is at most 1e-10 times its
      kernel-weighted mean square (after the feature's overall mean is
      removed).  A constant feature has correlation 0 with every feature,
      itself included, at that timepoint.  This is a numerical limit:
      features whose local variance is below about 1e-10 of their mean square
      are treated as constant.  A constant offset of the whole series does
      not matter (the overall mean is removed first; an offset of 1e9
      standard deviations is tested), but a step much larger than the local
      fluctuations within a kernel window does: a unit-variance series that
      jumps by 1e6 or more halfway through is treated as constant on both
      sides of the jump.

    One ``UserWarning`` per call reports, when they occur, how many
    timepoints have fewer than 3 non-zero kernel weights; how many other
    timepoints have an effective sample size ``(sum |w|)**2 / sum w**2``
    below 2 (a kernel too narrow for a reliable estimate: for example a
    Gaussian with ``var=0.05``, whose rows are one sample plus weights below
    1e-4); how many timepoints have, because of NaN samples, pairs of
    features with fewer than 3 observed samples with non-zero weight; and how
    many timepoints have zero kernel-weighted variance in every feature, so
    that all their correlations are 0.  A true delta kernel never warns.

    This estimator differs from Eq. 4 of Owen, Chang & Manning (2021) and
    from timecorr 0.2.0, where the kernel set only the means and the
    deviations were summed over all timepoints with equal weight; here the
    kernel also weights the covariance and variance sums.  Numerical results
    therefore differ from 0.2.0.

    Parameters
    ----------
    a : numpy.ndarray
        Timepoints-by-features observations matrix (T, Ka).
    b : numpy.ndarray
        Timepoints-by-features observations matrix (T, Kb).
    weights : numpy.ndarray
        Kernel matrix with T columns; row ``t`` holds the weights used to
        estimate at timepoint ``t``.  Usually T x T.

    Returns
    -------
    numpy.ndarray
        Array of shape (Ka, Kb, weights.shape[0]); entry ``[i, j, t]`` is the
        correlation between ``a[:, i]`` and ``b[:, j]`` at timepoint ``t``.

    Warns
    -----
    UserWarning
        One per call, naming each that applies: timepoints whose kernel row has
        fewer than 3 non-zero kernel weights (their correlations are undefined
        and set to 0); other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there); timepoints where, because of NaN
        samples, some pairs of features have fewer than 3 observed samples with
        non-zero kernel weight (those correlations are undefined and set to 0);
        and timepoints with zero kernel-weighted variance in every feature
        (their correlations are set to 0).  Never for a delta kernel.

    Raises
    ------
    ValueError
        If the inputs do not have T rows / T kernel columns, or if every row of
        the kernel is zero.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> x = np.random.RandomState(0).randn(100, 3)
    >>> corrs = tc.wcorr(x, x, tc.gaussian_weights(100, {"var": 50}))
    >>> corrs.shape
    (3, 3, 100)
    >>> print(np.diag(corrs[:, :, 0]))
    [1. 1. 1.]
    >>> uniform = tc.wcorr(x, x, tc.uniform_weights(100))
    >>> bool(np.allclose(uniform[:, :, 10], np.corrcoef(x.T)))
    True
    """
    return _wcorr(a, b, weights)


def _wcorr(a, b, weights, warn=True, return_cancelled=False):
    """
    ``wcorr``.  With ``warn=False`` the warning is left to the caller; with
    ``return_cancelled=True`` the masks ``(cancelled, sparse)`` of the
    timepoints whose correlations all cancelled and of those where some pair
    has fewer than 3 observed samples (see ``_wcorr_blocks``) are returned too:
    ``(out, (cancelled, sparse))``.
    """
    a, b, weights, support = _wcorr_inputs(a, b, weights)
    if np.all(support == 1):
        out = _delta_wcorr(a, b, weights)
        none = np.zeros(weights.shape[0], dtype=bool)
        return (out, (none, none.copy())) if return_cancelled else out

    kernel = weights.copy() if warn else None
    weights[support < 3] = 0

    autocorrelation = a is b or (
        a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    )
    out, cancelled, sparse = _wcorr_blocks(a, b, weights, autocorrelation)
    if warn:
        message = _estimator_message(kernel, cancelled, sparse)
        if message is not None:
            _warn_user(message)
    return (out, (cancelled, sparse)) if return_cancelled else out


def _nan_weighted_mean(x, weights, tol=1e-5):
    """
    Weighted mean over the last axis of x, ignoring NaN entries.

    Matches ``weighted_mean`` when x has no NaNs: weights with magnitude at most
    ``tol`` are dropped and the rest are normalized to sum to 1 (over the
    entries that are observed).  If the weights sum to 0, the plain mean is
    used.  Entries with no observed, weighted values are NaN.
    """
    weights = np.array(weights, dtype=np.float64, copy=True)
    weights[np.isnan(weights)] = 0
    if np.sum(weights) == 0:
        weights = np.ones_like(weights)
    weights[np.abs(weights) <= tol] = 0
    valid = ~np.isnan(x)
    num = np.where(valid, x, 0.0) @ weights
    den = valid @ weights
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _fisher_symmetrize(c):
    """
    Symmetrize a (K, K, T) cross-correlation array as in Eq. 5 of the timecorr
    paper: ``z2r((r2z(C) + r2z(C.T)) / 2)``.

    Where one of ``C[i, j]`` and ``C[j, i]`` is exactly +-1 (infinite z), the
    result is that sign; where they are +1 and -1, it is 0.  Entries outside
    [-1, 1] (co-fluctuations from a delta kernel, which are not correlations)
    are averaged without the transform (see ``_fisher_mean``).

    ``c`` is overwritten and returned.  The work is done a block of timepoints
    at a time, so the temporaries stay at block size rather than the size of
    ``c``.
    """
    K = c.shape[0]
    # _fisher_mean makes about ten temporaries of its input's size (which is
    # twice the block): keep the block small enough that they stay below the
    # workspace of wcorr's own blocks
    step = max(1, _WCORR_BLOCK_ELEMENTS // (16 * max(1, K * K)))
    for t0 in range(0, c.shape[2], step):
        block = c[:, :, t0 : t0 + step]
        block[...] = _fisher_mean(
            np.stack([block, block.transpose(1, 0, 2)]), axis=0
        )
    return c


def _fisher_mean(r, axis):
    """
    Average correlations along ``axis`` in Fisher z space: ``z2r(mean(r2z(r)))``.

    Fisher z averaging applies to correlations only.  Where any value along
    ``axis`` lies outside [-1, 1] by more than 1e-12 (for example a
    co-fluctuation from a delta kernel), the values are averaged on the raw
    scale (plain mean) instead; values within 1e-12 of [-1, 1] are
    correlations with round-off and are clipped to +-1.
    Values of exactly +-1 have infinite z.  Where only one sign of infinity
    occurs along ``axis`` the result is that sign (+-1).  Where both +inf and
    -inf occur they are excluded and the finite z values are averaged (0 if
    none remain).  NaN anywhere along ``axis`` gives NaN.  No numpy warnings
    are raised.
    """
    r = np.asarray(r, dtype=np.float64)
    outside = np.abs(r) > 1 + _R_TOLERANCE
    raw = np.any(outside, axis=axis)
    z = r2z(np.where(outside, 0.0, np.clip(r, -1.0, 1.0)))
    pos = np.any(np.isposinf(z), axis=axis)
    neg = np.any(np.isneginf(z), axis=axis)
    nan = np.any(np.isnan(z), axis=axis)
    finite = np.isfinite(z)
    total = np.where(finite, z, 0.0).sum(axis=axis)
    mean = z2r(_safe_divide(total, finite.sum(axis=axis).astype(np.float64)))
    out = np.where(pos & ~neg, 1.0, np.where(neg & ~pos, -1.0, mean))
    out = np.where(raw, r.mean(axis=axis), out)
    return np.where(nan, np.nan, out)


def wisfc(data, timepoint_weights, subject_weights=None):
    """
    Compute weighted inter-subject functional connectivity (ISFC).

    For each subject, the kernel-weighted correlations (see ``wcorr``) are
    computed between that subject's features and a weighted average of the
    subjects' data.  When the average is not the subject's own data, the
    cross-correlation matrix ``C`` is symmetrized by averaging the Fisher
    z-transforms of ``C`` and ``C.T`` and inverting (``z2r``) before it is
    vectorized; an entry of exactly +-1 dominates its pair.  NaN samples carry no weight.

    Parameters
    ----------
    data : numpy.ndarray or list of numpy.ndarray
        One timepoints-by-features (T, K) matrix per subject.
    timepoint_weights : numpy.ndarray or list of numpy.ndarray
        Kernel matrix; row ``t`` is the kernel used to estimate at timepoint ``t``.
        Or a list (or tuple) of S such matrices, one per subject: subject
        ``s``'s correlations use item ``s``, applied both to subject ``s``'s
        data and to the reference it is correlated with (see
        ``timecorr(..., events=...)``).  Equal kernels in the list are
        grouped.
    subject_weights : numpy.ndarray, scalar, or None, optional
        Number-of-subjects by number-of-subjects matrix; row ``s`` weights the
        subjects averaged for subject ``s``.  A scalar gives every other
        subject that weight.  Default (None): each pair of subjects is weighted
        by the correlation between their static correlation matrices, with
        zero weight on the subject itself.  Where that correlation is not
        defined (``K <= 2`` features give fewer than 3 matrix entries) or is
        at most 1e-5 in magnitude for every other subject, the other subjects
        are weighted equally.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        One (T, K * (K + 1) / 2) array of vectorized correlation matrices per
        subject (see ``mat2vec``); a single array if ``data`` is an array.
        Row ``t`` is estimated with the kernel ``timepoint_weights[t, :]``.

    Warns
    -----
    UserWarning
        One per call, naming each that applies: timepoints whose kernel row has
        fewer than 3 non-zero kernel weights (their correlations are undefined
        and set to 0); other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there); timepoints where, because of NaN
        samples, some pairs of features have fewer than 3 observed samples with
        non-zero kernel weight (those correlations are undefined and set to 0);
        and timepoints with zero kernel-weighted variance in every feature
        (their correlations are set to 0).  Never for a delta kernel.  With
        several different kernels, the one warning names, for each kernel with
        a problem, the subjects that use it.  In the symmetrized output, such
        an undefined entry ``C[i, j]`` is averaged (in Fisher z space) with
        ``C[j, i]``, which can be defined, so the returned value need not be 0;
        a feature with zero kernel-weighted variance on one side only (subject
        or reference) is treated the same way.

    Raises
    ------
    ValueError
        If the subjects differ in shape, the kernel does not have T columns,
        every row of the kernel is zero, or a list of kernels does not have
        one kernel per subject.

    Notes
    -----
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
    >>> rng = np.random.RandomState(0)
    >>> data = [rng.randn(50, 4) for _ in range(3)]
    >>> out = tc.wisfc(data, tc.gaussian_weights(50, {"var": 20}))
    >>> len(out), out[0].shape
    (3, (50, 10))

    One kernel per subject, each restricted to that subject's events:

    >>> from timecorr.events import mask_kernel
    >>> w = tc.gaussian_weights(50, {"var": 20})
    >>> kernels = [mask_kernel(w, b) for b in ([20], [25], [30])]
    >>> out = tc.wisfc(data, kernels, subject_weights=1 - np.eye(3))
    >>> len(out), out[0].shape
    (3, (50, 10))
    """
    if type(data) != list:
        return wisfc([data], timepoint_weights, subject_weights=subject_weights)[0]

    S = len(data)
    if subject_weights is None:
        K = data[0].shape[1]
        iu = np.triu_indices(K, 1)
        others = 1 - np.eye(S)
        if len(iu[0]) < 3:
            # K <= 2: fewer than 3 connectome entries, so no correlation
            # between connectomes is defined; weight the other subjects equally
            subject_weights = others
        else:
            static = np.ones([1, data[0].shape[0]])
            connectomes = np.stack(
                [wcorr(d, d, static)[:, :, 0][iu] for d in data], axis=1
            )
            subject_weights = wcorr(connectomes, connectomes, np.ones([1, len(iu[0])]))[
                :, :, 0
            ]
            np.fill_diagonal(subject_weights, 0)
            # a subject whose connectome is uncorrelated with every other one
            # (all weights at most 1e-5, which _nan_weighted_mean drops) would
            # get no reference: weight the other subjects equally instead
            empty = ~np.any(np.abs(subject_weights) > 1e-5, axis=1)
            subject_weights[empty] = others[empty]
    elif np.isscalar(subject_weights):
        subject_weights = subject_weights * np.ones([S, S])
        np.fill_diagonal(subject_weights, 0)

    kernels, group_of = _subject_kernels(timepoint_weights, S)
    stacked = np.stack(data, axis=2)
    corrs = []
    cancelled = [False] * len(kernels)
    sparse = [False] * len(kernels)
    for s, a in enumerate(data):
        g = group_of[s]
        b = _nan_weighted_mean(stacked, subject_weights[s, :])
        wc, (c, few) = _wcorr(a, b, kernels[g], warn=False, return_cancelled=True)
        cancelled[g] = cancelled[g] | c
        sparse[g] = sparse[g] | few
        if not np.array_equal(a, b, equal_nan=True):
            wc = _fisher_symmetrize(wc)
        corrs.append(mat2vec(wc))
        # free the (K, K, T) array before the next subject's is allocated
        del wc

    # one warning per call, covering every kernel
    message = _kernels_message(kernels, group_of, cancelled, sparse)
    if message is not None:
        _warn_user(message)
    return corrs


def _subject_kernels(timepoint_weights, S):
    """
    ``(kernels, group_of)``: the distinct kernels and, for each of the S subjects,
    the index of its kernel.  ``timepoint_weights`` is one kernel shared by every
    subject, or a list or tuple of S kernels (subject s uses item s); equal kernels
    in a list are grouped (compared by identity, then by value).  A list or tuple
    is per-subject kernels when its items are 2-D, and one kernel (for example a
    T x T kernel given as a list of lists) when none is; a mix raises ValueError.
    """
    if not isinstance(timepoint_weights, (list, tuple)):
        return [timepoint_weights], [0] * S
    two_d = [np.ndim(w) == 2 for w in timepoint_weights]
    if any(two_d) and not all(two_d):
        raise ValueError(
            "timepoint_weights is a list mixing 2-D kernels with items that are not "
            "2-D; pass one kernel (an array or a list of rows), or one 2-D kernel per "
            "subject"
        )
    if not all(two_d):
        return [np.asarray(timepoint_weights, dtype=np.float64)], [0] * S
    if len(timepoint_weights) != S:
        raise ValueError(
            f"timepoint_weights is a list of {len(timepoint_weights)} kernels for "
            f"{S} subjects; pass one kernel, or one kernel per subject"
        )
    kernels, group_of = [], []
    for w in timepoint_weights:
        for g, k in enumerate(kernels):
            if w is k or (
                np.shape(w) == np.shape(k) and np.array_equal(w, k, equal_nan=True)
            ):
                break
        else:
            g = len(kernels)
            kernels.append(w)
        group_of.append(g)
    return kernels, group_of


def _kernels_message(kernels, group_of, cancelled, sparse=None):
    """
    The one warning of a correlation call with one or several kernels, or None.
    With one kernel it is ``_estimator_message``; with several, each kernel's
    message is prefixed by the subjects that use it.  ``cancelled`` and ``sparse``
    hold one mask (see ``_wcorr_blocks``) per kernel.
    """
    if sparse is None:
        sparse = [None] * len(kernels)
    if len(kernels) == 1:
        return _estimator_message(kernels[0], cancelled[0], sparse[0])
    parts = []
    for g, kernel in enumerate(kernels):
        message = _estimator_message(kernel, cancelled[g], sparse[g])
        if message is not None:
            subjects = [s for s, k in enumerate(group_of) if k == g]
            label = "Subject" if len(subjects) == 1 else "Subjects"
            parts.append(f"{label} {', '.join(map(str, subjects))}: {message}")
    return " ".join(parts) if parts else None


def isfc(data, timepoint_weights):
    """
    Compute Inter-Subject Functional Connectivity (ISFC).

    ISFC computes correlations between each subject's data and the average of all other subjects,
    providing a measure of shared neural patterns across participants while avoiding
    self-correlation artifacts.  The subject-versus-others cross-correlation
    matrix ``C`` is symmetrized by averaging the Fisher z-transforms of ``C``
    and ``C.T``.  Averaging the subjects' results with
    ``combine=corrmean_combine`` (also in Fisher z space) then gives Eq. 5 of
    Owen, Chang & Manning (2021).  With a single subject, the within-subject
    correlations are returned.

    Parameters
    ----------
    data : list of numpy.ndarray or numpy.ndarray
        List of timeseries data matrices, each of shape (timepoints, features).
        If a single array is provided, it will be converted to a list.
    timepoint_weights : numpy.ndarray or list of numpy.ndarray
        T x T kernel matrix; row ``t`` is the kernel used to estimate at
        timepoint ``t``.
        Or a list (or tuple) of S such matrices, one per subject: subject
        ``s``'s correlations use item ``s``, applied both to subject ``s``'s
        data and to the reference it is correlated with (see
        ``timecorr(..., events=...)``).  Equal kernels in the list are
        grouped.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        List of vectorized correlation matrices (one (T, K * (K + 1) / 2)
        array per subject, see ``mat2vec``), each showing correlations between
        that subject and the average of all others; a single array if
        ``data`` is an array.  Row ``t`` is estimated with the kernel
        ``timepoint_weights[t, :]``.

    Warns
    -----
    UserWarning
        One per call, naming each that applies: timepoints whose kernel row has
        fewer than 3 non-zero kernel weights (their correlations are undefined
        and set to 0); other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there); timepoints where, because of NaN
        samples, some pairs of features have fewer than 3 observed samples with
        non-zero kernel weight (those correlations are undefined and set to 0);
        and timepoints with zero kernel-weighted variance in every feature
        (their correlations are set to 0).  Never for a delta kernel.  With
        several different kernels, the one warning names, for each kernel with
        a problem, the subjects that use it.  In the symmetrized output, such
        an undefined entry ``C[i, j]`` is averaged (in Fisher z space) with
        ``C[j, i]``, which can be defined, so the returned value need not be 0;
        a feature with zero kernel-weighted variance on one side only (subject
        or reference) is treated the same way.

    Raises
    ------
    ValueError
        If the subjects differ in shape, the kernel does not have T columns,
        every row of the kernel is zero, or a list of kernels does not have
        one kernel per subject.

    Notes
    -----
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
    >>> import timecorr as tc
    >>> import numpy as np
    >>> data = [np.random.randn(100, 10) for _ in range(5)]  # 5 subjects
    >>> weights = tc.gaussian_weights(100, {'var': 10})
    >>> isfc_results = tc.isfc(data, weights)
    >>> len(isfc_results), isfc_results[0].shape
    (5, (100, 55))
    """
    if type(data) != list:
        return isfc([data], timepoint_weights)[0]

    return wisfc(data, timepoint_weights, subject_weights=1 - np.eye(len(data)))


def autofc(data, timepoint_weights):
    """
    Compute auto-correlations within each subject's data.

    This function computes correlations within each subject's own data,
    equivalent to standard within-subject dynamic correlations.

    Parameters
    ----------
    data : list of numpy.ndarray or numpy.ndarray
        List of timeseries data matrices, each of shape (timepoints, features).
        If a single array is provided, it will be converted to a list.
    timepoint_weights : numpy.ndarray or list of numpy.ndarray
        T x T kernel matrix; row ``t`` is the kernel used to estimate at
        timepoint ``t``.
        Or a list (or tuple) of S such matrices, one per subject: subject
        ``s``'s correlations use item ``s``, applied both to subject ``s``'s
        data and to the reference it is correlated with (see
        ``timecorr(..., events=...)``).  Equal kernels in the list are
        grouped.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        List of vectorized correlation matrices (one (T, K * (K + 1) / 2)
        array per subject, see ``mat2vec``), each showing within-subject
        correlations; a single array if ``data`` is an array.  Row ``t`` is
        estimated with the kernel ``timepoint_weights[t, :]``.

    Warns
    -----
    UserWarning
        One per call, naming each that applies: timepoints whose kernel row has
        fewer than 3 non-zero kernel weights (their correlations are undefined
        and set to 0); other timepoints whose effective sample size
        ``(sum |w|)**2 / sum w**2`` is below 2 (the kernel is too narrow to
        estimate correlations reliably there); timepoints where, because of NaN
        samples, some pairs of features have fewer than 3 observed samples with
        non-zero kernel weight (those correlations are undefined and set to 0);
        and timepoints with zero kernel-weighted variance in every feature
        (their correlations are set to 0).  Never for a delta kernel.  With
        several different kernels, the one warning names, for each kernel with
        a problem, the subjects that use it.

    Raises
    ------
    ValueError
        If the subjects differ in shape, the kernel does not have T columns,
        every row of the kernel is zero, or a list of kernels does not have
        one kernel per subject.

    Examples
    --------
    >>> import timecorr as tc
    >>> import numpy as np
    >>> data = [np.random.randn(100, 10) for _ in range(3)]  # 3 subjects
    >>> weights = tc.gaussian_weights(100, {'var': 10})
    >>> auto_results = tc.autofc(data, weights)
    >>> len(auto_results), auto_results[0].shape
    (3, (100, 55))
    """
    if type(data) != list:
        return autofc([data], timepoint_weights)[0]

    return wisfc(data, timepoint_weights, subject_weights=np.eye(len(data)))


def apply_by_row(corrs, f):
    """
    Apply a function to the correlation matrix in each row of ``corrs``.

    Internal helper of ``reduce`` (graph measures).  Each row is converted to
    a K x K matrix with ``vec2mat`` before ``f`` is applied.

    Parameters
    ----------
    corrs : numpy.ndarray or list of numpy.ndarray
        A (T, K * (K + 1) / 2) matrix of vectorized correlation matrices
        (output of ``mat2vec``), or a list of such matrices.
    f : callable
        Function applied to each K x K matrix.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The outputs of ``f`` stacked along a new first axis (one row per
        timepoint), or a list of such arrays for list input.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import apply_by_row, mat2vec
    >>> corrs = np.stack([mat2vec(np.eye(3)), mat2vec(np.ones((3, 3)))])
    >>> print(apply_by_row(corrs, lambda m: m.sum(axis=1)))
    [[1. 1. 1.]
     [3. 3. 3.]]
    """

    if type(corrs) is list:
        return list(map(lambda x: apply_by_row(x, f), corrs))

    corrs = vec2mat(corrs)
    return np.stack(
        list(map(lambda x: f(np.squeeze(x)), np.split(corrs, corrs.shape[2], axis=2))),
        axis=0,
    )


def corrmean_combine(corrs):
    """
    Compute the mean element-wise correlation across each matrix in a list.

    Correlations are averaged in Fisher z space (``z2r(mean(r2z(r)))``).  An
    entry of exactly +-1 has infinite z: where only one sign occurs across
    matrices the mean is that sign; where both +1 and -1 occur they are
    excluded and the remaining values are averaged (0 if none remain).
    Fisher z applies to correlations only: an entry where any matrix has a
    value outside [-1, 1] (for example the co-fluctuations of a delta kernel,
    ``eye_weights``) is averaged on the raw scale (plain mean), with no NaN
    and no warning.

    Parameters
    ----------
    corrs : numpy.ndarray or list of numpy.ndarray
        A matrix of vectorized correlation matrices (output of ``mat2vec``), or
        a list of such matrices (one per subject).

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The element-wise mean, the shape of one input matrix.  An array or a
        one-element list is returned unchanged (the list stays a list).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import corrmean_combine
    >>> corrs = [np.array([[1.0, 0.5]]), np.array([[-1.0, 0.5]]), np.array([[0.5, 1.0]])]
    >>> print(np.round(corrmean_combine(corrs), 6))
    [[0.5 1. ]]
    """
    if not (type(corrs) == list):
        return corrs

    elif np.shape(corrs)[0] == 1:
        return corrs

    else:
        return _fisher_mean(np.stack(corrs, axis=2), axis=2)


def mean_combine(vals):
    """
    Compute the element-wise mean across the matrices in a list.

    The values are averaged as they are (no Fisher z-transform; see
    ``corrmean_combine`` for averaging correlations in z space).

    Parameters
    ----------
    vals : numpy.ndarray or list of numpy.ndarray
        A matrix, or a list of matrices of the same shape (one per subject).

    Returns
    -------
    numpy.ndarray
        The element-wise mean; an array is returned unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> print(tc.mean_combine([np.ones((2, 3)), 3 * np.ones((2, 3))]))
    [[2. 2. 2.]
     [2. 2. 2.]]
    """
    if not (type(vals) == list):
        return vals
    else:
        return np.mean(np.stack(vals, axis=2), axis=2)


def tstat_combine(corrs, return_pvals=False):
    """
    Element-wise one-sample t-tests (mean 0) across the matrices in a list.

    Each entry is Fisher z-transformed (``r2z``) and tested across the list
    with ``scipy.stats.ttest_1samp``.  Fisher z applies to correlations only:
    an entry where any matrix has a value outside [-1, 1] (for example the
    co-fluctuations of a delta kernel, ``eye_weights``) is tested on the raw
    values, with no NaN and no warning; values within 1e-12 of [-1, 1] are
    correlations with round-off (clipped to +-1).  An entry with the same
    value in every matrix (zero variance across subjects, for example the
    diagonal of ``autofc`` results, where every value is 1) has t = +inf or
    -inf with the sign of that value (p = 0), or t = 0 (p = 1) when the value
    is 0, with no warning.

    Parameters
    ----------
    corrs : numpy.ndarray or list of numpy.ndarray
        A matrix of vectorized correlation matrices (output of ``mat2vec``), or
        a list of such matrices (one per subject).
    return_pvals : bool
        If True, also return the p-values.  Default: False.

    Returns
    -------
    ts : numpy.ndarray
        The t statistics, the shape of one input matrix.  An array input is
        returned unchanged.
    ps : numpy.ndarray
        Only with ``return_pvals=True``: the two-sided p-values (all NaN for an
        array input).

    Raises
    ------
    ValueError
        If ``corrs`` is a list of fewer than two matrices (a t statistic across
        subjects needs at least two).

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> rng = np.random.RandomState(0)
    >>> data = [rng.randn(30, 3) for _ in range(4)]
    >>> ts = tc.timecorr(data, cfun=tc.isfc, combine=tc.tstat_combine,
    ...                  weights_params={"var": 10})
    >>> ts.shape
    (30, 6)
    """
    if not (type(corrs) == list):
        ts = corrs
        ps = np.nan * np.zeros_like(corrs)
    elif len(corrs) < 2:
        raise ValueError(
            "tstat_combine needs at least two subjects (a t statistic across "
            f"subjects is undefined for fewer); got a list of {len(corrs)}."
        )
    else:
        stacked = np.asarray(np.stack(corrs, axis=2), dtype=np.float64)
        outside = np.abs(stacked) > 1 + _R_TOLERANCE
        raw = np.any(outside, axis=2, keepdims=True)
        z = r2z(np.where(outside, 0.0, np.clip(stacked, -1.0, 1.0)))
        vals = np.where(raw, stacked, z)
        # zero variance across subjects (every value equal, infinities
        # included): t is +-inf with the sign of the mean, 0 for a mean of 0
        constant = np.all(vals == vals[:, :, :1], axis=2)
        filler = np.arange(vals.shape[2], dtype=np.float64)  # any non-constant row
        ts, ps = ttest(np.where(constant[:, :, None], filler, vals), popmean=0, axis=2)
        first = vals[:, :, 0]
        signed_inf = np.where(first > 0, np.inf, np.where(first < 0, -np.inf, 0.0))
        ts = np.where(constant, signed_inf, ts)
        ps = np.where(constant, np.where(first == 0, 1.0, 0.0), ps)

    if return_pvals:
        return ts, ps
    else:
        return ts


def null_combine(corrs):
    """
    Return the input unchanged (the default ``combine`` of ``timecorr``).

    With list input, ``timecorr`` then returns one result per subject.

    Parameters
    ----------
    corrs : numpy.ndarray or list of numpy.ndarray
        A matrix of vectorized correlation matrices (output of ``mat2vec``), or
        a list of such matrices.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        ``corrs``.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> corrs = [np.zeros((5, 3)), np.ones((5, 3))]
    >>> tc.null_combine(corrs) is corrs
    True
    """
    return corrs


# Graph measures accepted (as strings) by ``reduce``.
_GRAPH_MEASURES = ("eigenvector_centrality", "pagerank_centrality", "strength")
# strings that create a fresh summary instance (timecorr.summaries)
_SUMMARY_NAMES = ("weighted_strength", "weighted_strength_split", "communities")

# Model names documented for ``reduce``; hypertools' own registry, when it can be
# read, adds the others it supports.
_DOCUMENTED_MODELS = (
    "PCA", "IncrementalPCA", "SparsePCA", "MiniBatchSparsePCA", "KernelPCA",
    "FastICA", "FactorAnalysis", "TruncatedSVD", "DictionaryLearning",
    "MiniBatchDictionaryLearning", "TSNE", "Isomap", "SpectralEmbedding",
    "LocallyLinearEmbedding", "MDS", "UMAP",
)


def _model_names():
    """Model names ``reduce`` accepts: the documented ones and hypertools' registry."""
    return set(_DOCUMENTED_MODELS) | set(_HYP_REDUCERS) | set(_HYP_AUTOENCODERS)


def _check_rfun(rfun, name="rfun"):
    """
    Raise for a reducer ``reduce`` cannot use: None, a callable (including a
    summary object), a graph measure, summary name or model name (string), or a
    dict whose ``'model'`` is a model name or a model class or instance.
    """
    if rfun is None or callable(rfun):
        return
    if isinstance(rfun, str):
        if rfun in _GRAPH_MEASURES or rfun in _SUMMARY_NAMES or rfun in _model_names():
            return
        raise ValueError(
            f"{name}={rfun!r} is neither a graph measure {_GRAPH_MEASURES}, a "
            f"summary {_SUMMARY_NAMES}, nor a model name known to hypertools "
            f"({', '.join(sorted(_model_names()))})."
        )
    if isinstance(rfun, dict):
        if "model" not in rfun:
            raise ValueError(
                f"A dict {name} needs a 'model' key (for example "
                "{'model': 'PCA', 'kwargs': {...}}); got keys "
                f"{sorted(rfun)}."
            )
        if isinstance(rfun["model"], str):
            _check_rfun(rfun["model"], name=f"{name}['model']")
            if rfun["model"] in _GRAPH_MEASURES:
                raise ValueError(
                    f"{name}['model'] must be a model; graph measures are given "
                    "as plain strings."
                )
        return
    raise TypeError(
        f"{name} must be None, a string, a dict or a callable; got "
        f"{type(rfun).__name__}."
    )


def _is_model(rfun):
    """
    True for a model-based reducer (a model name or dict), whose width is
    ``ndims``; False for graph measures, summaries (names or objects),
    callables and None.
    """
    if isinstance(rfun, dict):
        return True
    return (
        isinstance(rfun, str)
        and rfun not in _GRAPH_MEASURES
        and rfun not in _SUMMARY_NAMES
    )


def _check_ndims(ndims):
    if ndims is None:
        return None
    if (
        not isinstance(ndims, numbers.Integral)
        or isinstance(ndims, bool)
        or ndims < 1
    ):
        raise ValueError(f"ndims must be a positive integer or None; got {ndims!r}.")
    return int(ndims)


def _umap_defaults(rfun):
    """
    For UMAP, the model arguments timecorr sets unless the caller gives them:
    ``compatibility_layout=True`` (keeps UMAP's current layout and silences its
    FutureWarning, where the installed UMAP has the option) and ``n_jobs=1``
    (what UMAP uses anyway with a ``random_state``; silences its warning).
    """
    model = rfun["model"] if isinstance(rfun, dict) else rfun
    if model != "UMAP":
        return rfun
    with warnings.catch_warnings():
        # umap-learn warns at import about its optional tensorflow extra
        # (ParametricUMAP); hypertools filters the same message
        warnings.filterwarnings("ignore", message="Tensorflow not installed")
        from umap import UMAP

    defaults = {"n_jobs": 1}
    if "compatibility_layout" in inspect.signature(UMAP.__init__).parameters:
        defaults["compatibility_layout"] = True
    if isinstance(rfun, str):
        return {"model": "UMAP", "kwargs": defaults}
    return {**rfun, "kwargs": {**defaults, **(rfun.get("kwargs") or {})}}


def reduce(corrs, rfun=None, ndims=None):
    """
    Reduce vectorized correlation matrices back to a (timepoints by features) matrix.

    Parameters
    ----------
    corrs : numpy.ndarray or list of numpy.ndarray
        A matrix of vectorized correlation matrices (output of ``mat2vec``), or a
        list of such matrices (one per subject).
    rfun : None, str, dict, or callable
        How to reduce the dimensionality.

        - ``None``: no reduction; ``corrs`` is returned unchanged.
        - A hypertools / scikit-learn model name (PCA, IncrementalPCA, SparsePCA,
          MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
          DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
          SpectralEmbedding, LocallyLinearEmbedding, MDS, or UMAP), or a dict
          such as ``{'model': 'PCA', 'params': {'whiten': True}}`` (``'params'``
          and ``'kwargs'`` are both passed to the model; a key given in both
          with different values raises ``ValueError``).  When
          ``corrs`` is a list, one model is fit jointly to all subjects.
          ``random_state=0`` is passed to the model.  For UMAP, timecorr also
          passes ``n_jobs=1`` (what UMAP uses with a ``random_state``) and,
          where the installed UMAP has it, ``compatibility_layout=True`` (its
          current layout pipeline), unless the dict's ``'kwargs'`` give them.
        - One of the graph measures ``'eigenvector_centrality'``,
          ``'pagerank_centrality'``, or ``'strength'`` (strings only), computed
          per node for each timepoint.  A delta-kernel co-fluctuation matrix
          ``z z^T`` has rank one, so on the delta path of ``features``
          eigenvector centrality equals ``|z| / ||z||`` and the graph measures
          carry no network structure.
        - A squared-correlation summary (see ``timecorr.summaries``): a
          ``WeightedStrength`` or ``Communities`` instance, or the strings
          ``'weighted_strength'`` (``WeightedStrength()``),
          ``'weighted_strength_split'`` (``WeightedStrength(split=True)``) and
          ``'communities'`` (``Communities()``), which create a fresh instance
          per call.  The instance is fit to ``corrs`` and transforms
          ``corrs``; its ``fits_`` record the fit.  The two strings that need
          communities raise ``ValueError`` here, since direct calls need the
          number of communities ``n``.
        - A callable ``f(list of numpy.ndarray) -> list of numpy.ndarray``.  A
          single array is passed as a one-element list and the first element of
          the result is returned.

        Default: None (no reduction).
    ndims : int, optional
        Number of output columns for model-based reduction (a positive
        integer).  Default: None, which returns K columns for inputs with
        K(K+1)/2 features.  The model is never asked for more components than
        the data can supply (the number of features and of samples); any
        shortfall is filled with zero columns so the output has exactly the
        requested width, and a ``UserWarning`` says how many columns were
        padded.  Graph measures (which return K columns), summaries and
        callables define their own width: giving ``ndims`` with them raises
        ``ValueError``.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The reduced (or original) matrices, in the same container type as
        ``corrs``: (T, ndims) for model-based reduction, (T, K) for a graph
        measure, (T, c) for a summary, whatever the callable returns otherwise.

    Raises
    ------
    ValueError
        If a dict ``rfun`` gives a key in both ``'params'`` and ``'kwargs'`` with
        different values, if ``ndims`` is not a positive integer, if
        ``ndims`` is given with a graph measure, a summary or a callable, or
        from a summary (see ``WeightedStrength`` and ``Communities``).

    Warns
    -----
    UserWarning
        When the output is padded with zero columns (``ndims`` larger than the
        number of features or samples).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import reduce
    >>> corrs = np.random.RandomState(0).randn(20, 21)
    >>> reduce(corrs, rfun='PCA', ndims=3).shape
    (20, 3)
    >>> reduce(corrs, rfun='PCA').shape
    (20, 6)
    >>> import timecorr as tc
    >>> x = np.random.default_rng(0).standard_normal((50, 4))
    >>> reduce(tc.timecorr(x, cfun=tc.autofc), rfun='weighted_strength').shape
    (50, 4)
    """

    # Define graph measures directly instead of using brainconn
    def eigenvector_centrality_und(W):
        """
        Compute eigenvector centrality for undirected graph.

        Parameters
        ----------
        W : numpy.ndarray
            Undirected weighted/binary connection matrix

        Returns
        -------
        numpy.ndarray
            Eigenvector centrality
        """
        # Ensure matrix is symmetric
        W = (W + W.T) / 2

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eigh(W)

        # Get the eigenvector corresponding to the largest eigenvalue
        max_idx = np.argmax(eigenvalues)
        centrality = np.abs(eigenvectors[:, max_idx])

        return centrality

    def pagerank_centrality(W, d=0.85, falff=None):
        """
        Compute PageRank centrality using the same algorithm as brainconn.

        Parameters
        ----------
        W : numpy.ndarray
            Weighted/binary connection matrix (adjacency matrix)
        d : float
            Damping factor (default: 0.85)
        falff : numpy.ndarray or None
            Initial page rank probability, non-negative values. Default value is
            None. If not specified, a naive bayesian prior is used.

        Returns
        -------
        numpy.ndarray
            PageRank centrality values
        """
        N = len(W)
        if falff is None:
            norm_falff = np.ones((N,)) / N
        else:
            norm_falff = falff / np.sum(falff)

        deg = np.sum(W, axis=0)
        deg[deg == 0] = 1
        D1 = np.diag(1 / deg)
        B = np.eye(N) - d * np.dot(W, D1)
        b = (1 - d) * norm_falff
        r = linalg.solve(B, b)
        r /= np.sum(r)
        return r

    def strengths_und(W):
        """
        Compute node strength for undirected graph.

        Parameters
        ----------
        W : numpy.ndarray
            Undirected weighted connection matrix

        Returns
        -------
        numpy.ndarray
            Node strengths
        """
        # For undirected graphs, strength is the sum of weights
        return np.sum(W, axis=1)

    # Set up graph measures
    graph_measures = {
        "eigenvector_centrality": eigenvector_centrality_und,
        "pagerank_centrality": lambda x: pagerank_centrality(x, d=0.85),
        "strength": strengths_und,
    }

    ndims = _check_ndims(ndims)
    if rfun is None:
        return corrs

    if ndims is not None and not _is_model(rfun):
        raise ValueError(
            f"ndims={ndims} applies to model-based reduction only; the graph "
            "measure or callable rfun defines its own output width.  Leave ndims "
            "as None."
        )

    from .summaries import Communities, WeightedStrength, _Summary

    if isinstance(rfun, str) and rfun in _SUMMARY_NAMES:
        rfun = {
            "weighted_strength": lambda: WeightedStrength(),
            "weighted_strength_split": lambda: WeightedStrength(split=True),
            "communities": lambda: Communities(),
        }[rfun]()

    if isinstance(rfun, _Summary):
        # a summary takes a list or a single array and returns the same kind
        return rfun(corrs)

    if callable(rfun):
        if isinstance(corrs, list):
            return rfun(corrs)
        return rfun([corrs])[0]

    if isinstance(rfun, str) and rfun in graph_measures:
        return apply_by_row(corrs, graph_measures[rfun])

    arrays = corrs if isinstance(corrs, list) else [corrs]
    n_features = arrays[0].shape[1]
    if ndims is None:
        # K(K+1)/2 features back to K columns
        ndims = int(np.divide(np.sqrt(8 * n_features + 1) - 1, 2))

    # a model cannot return more components than there are features or samples
    n_samples = sum(a.shape[0] for a in arrays)
    n_fit = min(ndims, n_features, n_samples)

    if isinstance(rfun, dict) and "params" in rfun:
        # hypertools deprecated 'params' in favor of 'kwargs': merge both
        params = dict(rfun["params"] or {})
        kwargs = dict(rfun.get("kwargs") or {})
        clash = sorted(
            k for k in set(params) & set(kwargs) if params[k] is not kwargs[k]
            and params[k] != kwargs[k]
        )
        if clash:
            raise ValueError(
                f"rfun gives different values for {clash} in 'params' and "
                "'kwargs'; give each model argument once."
            )
        rfun = {
            **{k: v for k, v in rfun.items() if k not in ("params", "kwargs")},
            "kwargs": {**kwargs, **params},
        }

    rfun = _umap_defaults(rfun)
    red_corrs = hyp.reduce(corrs, reduce=rfun, ndims=n_fit, random_state=0)
    if isinstance(corrs, list) and not isinstance(red_corrs, list):
        # hypertools returns a bare array for a one-element list
        red_corrs = [red_corrs]

    if n_fit < ndims:
        # Pad with zeros to match the requested output width
        pad = ndims - n_fit
        _warn_user(
            f"reduce: the data supply at most {n_fit} components ({n_features} "
            f"features, {n_samples} samples) but {ndims} were requested; the "
            f"output is padded with {pad} zero column{'s' if pad > 1 else ''}."
        )
        if isinstance(red_corrs, list):
            red_corrs = [
                np.hstack((arr, np.zeros((arr.shape[0], pad)))) for arr in red_corrs
            ]
        else:
            red_corrs = np.hstack((red_corrs, np.zeros((red_corrs.shape[0], pad))))

    return red_corrs


def smooth(
    w,
    windowsize=10,
    weights_function=None,
    weights_params=None,
    kernel_fun=None,
    kernel_params=None,
):
    """
    Smooth each column of a timeseries with a kernel truncated to a window.

    The kernel is the middle row of
    ``weights_function(windowsize, params=weights_params)``
    (the average of the two middle rows for an even ``windowsize``), normalized
    to sum to one.  Row ``t`` of a kernel matrix weights the samples used at
    ``t``, so a one-sided (e.g. backward) kernel smooths with past samples only.

    Parameters
    ----------
    w : numpy.ndarray or list of numpy.ndarray
        A timepoints-by-features matrix, or a list of them.
    windowsize : int
        Width of the kernel window in samples.  Default: 10.
    weights_function : function, optional
        Kernel function, called as
        ``weights_function(windowsize, params=weights_params)``.  Default
        (None): ``laplace_weights``.
    weights_params : dict, optional
        Parameters for ``weights_function``.  Default: None (the kernel
        function's own defaults).
    kernel_fun, kernel_params : optional
        Aliases of ``weights_function`` and ``weights_params`` (their names in
        timecorr 0.2.0).  Passing a name and its alias raises ``TypeError``.
        Default: None.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The smoothed data, the same shape (and container) as ``w``.  Floating
        point input keeps its dtype; integer or boolean input gives float64
        output (the same values as smoothing the input converted to floats).

    Raises
    ------
    AssertionError
        If ``windowsize`` is not an ``int``.
    TypeError
        If a name and its alias are both given.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import smooth, laplace_weights
    >>> w = np.zeros((9, 1))
    >>> w[4, 0] = 1.0
    >>> out = smooth(w, windowsize=3, weights_function=laplace_weights,
    ...              weights_params={"scale": 1})
    >>> out.shape, bool(np.isclose(out.sum(), 1.0))
    ((9, 1), True)
    """
    weights_function = _weights_alias(
        "weights_function", weights_function, "kernel_fun", kernel_fun
    )
    weights_params = _weights_alias(
        "weights_params", weights_params, "kernel_params", kernel_params
    )
    if weights_function is None:
        weights_function = laplace_weights
    if type(w) is list:
        return [
            smooth(
                x,
                windowsize=windowsize,
                weights_function=weights_function,
                weights_params=weights_params,
            )
            for x in w
        ]

    assert type(windowsize) == int, "smoothing kernel must have integer width"
    k = weights_function(windowsize, params=weights_params)
    if iseven(windowsize):
        kernel = np.divide(
            k[int(np.floor(windowsize / 2) - 1), :]
            + k[int(np.ceil(windowsize / 2) - 1), :],
            2,
        )
    else:
        kernel = k[int(np.floor(windowsize / 2)), :]

    kernel /= kernel.sum()
    # floating-point output: an integer input must not truncate the result
    x = np.zeros(
        w.shape, dtype=w.dtype if np.issubdtype(w.dtype, np.inexact) else np.float64
    )
    # The kernel is row ``center`` of the window's kernel matrix, so entry j weights
    # the sample j - center steps away: x[t] = sum_j kernel[j] * w[t + j - center].
    # That is a convolution with the reversed row (this matters for one-sided
    # kernels, and for even windows, whose row is not symmetric).  The "full"
    # output is sliced so the result always has the length of the series
    # (np.convolve mode="same" fails when the series is shorter than the window).
    center = (len(kernel) - 1) // 2
    start = len(kernel) - 1 - center
    for i in range(0, w.shape[1]):
        x[:, i] = np.convolve(kernel[::-1], w[:, i], mode="full")[
            start : start + w.shape[0]
        ]
    return x


def weighted_mean(x, axis=None, weights=None, tol=1e-5):
    """
    Weighted mean of ``x`` along one axis (internal helper).

    NaN weights count as 0.  Weights with magnitude at most ``tol`` are
    dropped and the rest are normalized to sum to 1.  If the weights sum to 0,
    the plain mean is returned.  The caller's weights are not modified.

    Parameters
    ----------
    x : numpy.ndarray
        Values to average.
    axis : int, optional
        Axis to average over.  Default: the last axis.
    weights : array-like, optional
        One weight per entry along ``axis``.  Default: equal weights.
    tol : float, optional
        Weights with ``|w| <= tol`` are ignored.  Default: 1e-5.

    Returns
    -------
    numpy.ndarray
        ``x`` averaged over ``axis``.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import weighted_mean
    >>> print(weighted_mean(np.array([[1.0, 3.0]]), weights=[1, 3]))
    [2.5]
    """
    if axis is None:
        axis = len(x.shape) - 1
    if weights is None:
        weights = np.ones(x.shape[axis])
    else:
        # work on a copy: the caller's weights are not modified
        weights = np.array(weights, dtype=np.float64, copy=True)

    # remove nans and force weights to sum to 1
    weights[np.isnan(weights)] = 0
    if np.sum(weights) == 0:
        return np.mean(x, axis=axis)

    # get rid of 0 weights to avoid unnecessary computations
    good_inds = np.abs(weights) > tol
    weights[good_inds] /= np.sum(weights[good_inds])

    weighted_sum = np.zeros(np.take(x, 0, axis=axis).shape)
    for i in np.where(good_inds)[0]:
        weighted_sum += weights[i] * np.take(x, i, axis=axis)

    return weighted_sum


def rmdiag(m):
    """
    Return a copy of the square matrix ``m`` with its diagonal set to 0 (internal helper).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import rmdiag
    >>> print(rmdiag(np.ones((2, 2))))
    [[0. 1.]
     [1. 0.]]
    """
    return m - np.diag(np.diag(m))


def r2z(r):
    """
    Fisher z-transformation (``arctanh``) of correlation values.

    Parameters
    ----------
    r : float or numpy.ndarray
        Correlation value(s).

    Returns
    -------
    float or numpy.ndarray
        ``arctanh(r)``: ``+inf`` / ``-inf`` where ``r`` is exactly 1 / -1, and
        NaN where ``|r| > 1`` or ``r`` is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import r2z
    >>> print(r2z(np.array([0.0, 1.0, -1.0])))
    [  0.  inf -inf]
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.full(r.shape, np.nan)
    inside = np.abs(r) < 1
    z[inside] = np.arctanh(r[inside])
    z[r == 1] = np.inf
    z[r == -1] = -np.inf
    return z[()]


def z2r(z):
    """
    Inverse Fisher z-transformation (``tanh``).

    Parameters
    ----------
    z : float or numpy.ndarray
        Fisher z-transformed correlation value(s).

    Returns
    -------
    float or numpy.ndarray
        ``tanh(z)``: 1 / -1 for ``+inf`` / ``-inf``, and 0 where ``z`` is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import z2r
    >>> print(z2r(np.array([0.0, np.inf, -np.inf, np.nan])))
    [ 0.  1. -1.  0.]
    """
    r = np.tanh(np.asarray(z, dtype=np.float64))
    r = np.where(np.isnan(r), 0.0, r)
    return r[()]


def isodd(x):
    """
    Test whether numbers are odd.

    Parameters
    ----------
    x : int, float or array-like
        Number(s) to test.

    Returns
    -------
    bool or numpy.ndarray of bool
        ``numpy.remainder(x, 2) == 1``, elementwise.  Negative odd integers
        give True; non-integers give False.

    Examples
    --------
    >>> import timecorr as tc
    >>> bool(tc.isodd(3)), bool(tc.isodd(4))
    (True, False)
    >>> tc.isodd([-3, 0, 1.5]).tolist()
    [True, False, False]
    """
    return np.remainder(x, 2) == 1


def iseven(x):
    """
    Test whether numbers are even.

    Parameters
    ----------
    x : int, float or array-like
        Number(s) to test.

    Returns
    -------
    bool or numpy.ndarray of bool
        ``numpy.remainder(x, 2) == 0``, elementwise.  Non-integers give False.

    Examples
    --------
    >>> import timecorr as tc
    >>> bool(tc.iseven(4)), bool(tc.iseven(3))
    (True, False)
    >>> tc.iseven([-2, 1, 2.5]).tolist()
    [True, False, False]
    """
    return np.remainder(x, 2) == 0


def mat2vec(m):
    """
    Convert correlation matrices to vectors.

    Each K x K matrix becomes a vector of length K * (K + 1) / 2: its diagonal
    first, then its upper triangle in row-major order (the order of
    ``scipy.spatial.distance.squareform``).  Only the upper triangle is read,
    and NaNs there are replaced with 0.

    Parameters
    ----------
    m : numpy.ndarray
        A K x K matrix, or a K x K x T stack of matrices.

    Returns
    -------
    numpy.ndarray
        A vector of length K * (K + 1) / 2, or a T x (K * (K + 1) / 2) matrix
        with one row per timepoint.

    Raises
    ------
    ValueError
        If ``m`` is not 2- or 3-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import mat2vec
    >>> print(mat2vec(np.array([[1.0, 0.5], [0.5, 1.0]])))
    [1.  1.  0.5]
    """
    m = np.asarray(m)
    if m.ndim == 2:
        return mat2vec(m[:, :, np.newaxis])[0]
    if m.ndim != 3:
        raise ValueError("Input must be a 2 or 3 dimensional Numpy array")

    K = m.shape[0]
    y = np.empty([m.shape[2], K * (K + 1) // 2])
    y[:, :K] = np.diagonal(m, axis1=0, axis2=1)
    # copy the upper triangle row by row, so no full-size temporary is made
    col = K
    for i in range(K - 1):
        y[:, col : col + K - i - 1] = m[i, i + 1 :, :].T
        col += K - i - 1
    upper = y[:, K:]
    np.copyto(upper, 0.0, where=np.isnan(upper))
    return y


def vec2mat(v):
    """
    Convert vectors produced by ``mat2vec`` back to symmetric matrices.

    Parameters
    ----------
    v : numpy.ndarray
        A vector of length K * (K + 1) / 2, or a T x (K * (K + 1) / 2) matrix
        with one vector per row.

    Returns
    -------
    numpy.ndarray
        A K x K matrix for a vector input, or a K x K x T array for a matrix
        input (including a 1 x (K * (K + 1) / 2) matrix).

    Raises
    ------
    ValueError
        If ``v`` is not 1- or 2-dimensional, or its length is not
        K * (K + 1) / 2 for an integer K.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import vec2mat
    >>> print(vec2mat(np.array([1.0, 1.0, 0.5])))
    [[1.  0.5]
     [0.5 1. ]]
    >>> vec2mat(np.zeros([4, 6])).shape
    (3, 3, 4)
    """
    v = np.asarray(v)
    if v.ndim == 1:
        return vec2mat(v[np.newaxis, :])[:, :, 0]
    if v.ndim != 2:
        raise ValueError("Input must be a 1 or 2 dimensional Numpy array")

    V = v.shape[1]
    K = int(round(0.5 * (np.sqrt(8 * V + 1) - 1)))
    if K * (K + 1) // 2 != V:
        raise ValueError(
            f"Vectors of length {V} do not correspond to a square matrix "
            "(expected length K * (K + 1) / 2)."
        )
    iu = np.triu_indices(K, 1)
    m = np.zeros([K, K, v.shape[0]], dtype=np.result_type(v.dtype, np.float64))
    diag = np.arange(K)
    m[diag, diag, :] = v[:, :K].T
    m[iu[0], iu[1], :] = v[:, K:].T
    m[iu[1], iu[0], :] = v[:, K:].T
    return m


def symmetric(m):
    """
    Whether the square matrix ``m`` equals its transpose (internal helper).

    Entries are compared with ``numpy.isclose`` (default tolerances).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import symmetric
    >>> bool(symmetric(np.array([[1.0, 2.0], [2.0, 1.0]]))), bool(symmetric(np.triu(np.ones((2, 2)))))
    (True, False)
    """
    return np.isclose(m, m.T).all()


def plot_weights(
    weights,
    t=None,
    color="k",
    ax=None,
    xlab="Time (samples)",
    ylab="Weights",
    title=None,
    outfile=None,
):
    """
    Plot the kernel used at one timepoint (one row of a weights matrix).

    The figure is not shown; call ``matplotlib.pyplot.show()`` to display it.

    Parameters
    ----------
    weights : numpy.ndarray
        T x T kernel matrix; row ``t`` is the kernel used at timepoint ``t``.
    t : int, optional
        Row to plot.  Default: ``round(T / 2)``.
    color : str
        Line color.  Default: "k".
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Default: the current axes.
    xlab : str or None
        X-axis label (None leaves it unset).  Default: "Time (samples)".
    ylab : str or None
        Y-axis label (None leaves it unset).  Default: "Weights".
    title : str or None
        Axes title (None leaves it unset).  Default: None.
    outfile : str, optional
        If given, the figure is saved to this path.  Default: None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from timecorr.helpers import gaussian_weights, plot_weights
    >>> fig, ax = plt.subplots()
    >>> ax = plot_weights(gaussian_weights(20, {"var": 4}), t=5, ax=ax)
    >>> len(ax.lines), tuple(float(x) for x in ax.get_xlim())
    (1, (1.0, 20.0))
    >>> plt.close(fig)
    """
    T = weights.shape[0]
    if t is None:
        t = np.round(T / 2)
    ts = np.arange(1, T + 1)

    if ax is None:
        ax = plt.gca()

    ax.plot(ts, weights[int(t), :], color=color)
    ax.set_xlim([1, T])

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)

    ax.figure.tight_layout()
    if outfile:
        ax.figure.savefig(outfile)
    return ax


# Decoders live in timecorr.decoding; they are re-exported here so existing
# imports from timecorr.helpers keep working.  Imported last: decoding imports
# from this module.
from . import decoding as _decoding  # noqa: E402

calculate_error = _decoding.calculate_error
decoder = _decoding.decoder
folding_levels = _decoding.folding_levels
folding_levels_ec = _decoding.folding_levels_ec
get_xval_assignments = _decoding.get_xval_assignments
optimize_weights = _decoding.optimize_weights
pca_decoder = _decoding.pca_decoder
reduce_wrapper = _decoding.reduce_wrapper
sum_to_x = _decoding.sum_to_x
timepoint_decoder = _decoding.timepoint_decoder
weight_corrs = _decoding.weight_corrs
weighted_timepoint_decoder = _decoding.weighted_timepoint_decoder
weighted_timepoint_decoder_ec = _decoding.weighted_timepoint_decoder_ec
