# coding: utf-8
import functools
import inspect
import numbers
import os
import pickle
import sys
import warnings

import datawrangler as dw
import hypertools as hyp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.stats import ttest_1samp as ttest

from ._kernels import (
    CHUNK,
    LagKernel,
    as_kernel,
    blocks,
    boxcar_profile,
    eye_profile,
    gaussian_profile,
    laplace_profile,
    mexican_hat_profile,
    t_profile,
    uniform_profile,
)
from ._kernels import lags as _lags

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

    return _finish_kernel(gaussian_profile(_lags(T), params), direction, include_zero)


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

    return _finish_kernel(laplace_profile(_lags(T), params), direction, include_zero)


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

    # built directly (equal to eye_profile at every lag) without the T x T
    # matrix of lags
    return _finish_kernel(np.eye(int(T)), direction, include_zero)


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
    # built directly (equal to uniform_profile at every lag) without the
    # T x T matrix of lags
    return _finish_kernel(np.ones((int(T), int(T))), direction, include_zero)


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

    return _finish_kernel(t_profile(_lags(T), params), direction, include_zero)


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

    return _finish_kernel(
        mexican_hat_profile(_lags(T), params), direction, include_zero
    )


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

    return _finish_kernel(boxcar_profile(_lags(T), params), direction, include_zero)


# The lag profile and default parameters of each built-in kernel function.
# For these, ``timecorr``, ``features`` and the decoders compute the kernel a
# chunk of rows at a time (``_kernels.LagKernel``) instead of building the
# T x T matrix.
_LAG_PROFILES = {
    gaussian_weights: (gaussian_profile, gaussian_params),
    laplace_weights: (laplace_profile, laplace_params),
    eye_weights: (eye_profile, eye_params),
    uniform_weights: (uniform_profile, uniform_params),
    t_weights: (t_profile, t_params),
    mexican_hat_weights: (mexican_hat_profile, mexican_hat_params),
    boxcar_weights: (boxcar_profile, boxcar_params),
}


def _lag_kernel(T, weights_function, weights_params, include="all", include_zero=True,
                exclude=None):
    """
    The kernel ``weights_function(T, weights_params)`` restricted by
    ``include`` ('all', 'backward' or 'forward'), ``include_zero`` and
    ``exclude`` (``exclude_timepoints``) as a ``LagKernel`` (no T x T
    array), or None when ``weights_function`` is not a built-in kernel.
    """
    try:
        profile, defaults = _LAG_PROFILES[weights_function]
    except (KeyError, TypeError):
        return None
    params = defaults if weights_params is None else weights_params
    return LagKernel(profile(np.arange(int(T), dtype=np.int64), params), include,
                     include_zero, exclude,
                     lag_profile=functools.partial(profile, params=params))


def _as_kernel_array(weights):
    """A T x T array for ``weights`` (a ``LagKernel`` is built in full)."""
    if isinstance(weights, LagKernel):
        return weights.dense()
    return weights


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
    T_out = _kernel_shape(weights)[0]
    support = np.zeros(T_out, dtype=np.int64)
    ess = np.zeros(T_out)
    for t0, t1, w in _abs_row_chunks(weights):
        support[t0:t1] = np.count_nonzero(w, axis=1)
        # scaled by each row's largest weight, so tiny weights do not underflow
        # (rows whose largest weight is 0 are all 0 and stay so)
        peak = w.max(axis=1, keepdims=True, initial=0.0)
        w = np.divide(w, peak, out=np.zeros_like(w), where=peak != 0)
        total = w.sum(axis=1)
        np.multiply(w, w, out=w)
        ess[t0:t1] = _safe_divide(total**2, w.sum(axis=1))
    if np.all(support == 0) or np.all(support == 1):
        return None
    undefined = support < 3
    return undefined, ~undefined & (ess < 2)


def _estimator_message(weights, cancelled=None, sparse=None, warmup=None,
                       min_samples=None):
    """
    The one warning of a correlation call, or None.

    It names the kernel rows with fewer than 3 non-zero weights (undefined, set
    to 0), the other rows whose effective sample size is below 2, the
    timepoints (``sparse``, a boolean mask) where some pairs of features have
    fewer than 3 observed samples with non-zero weight because of NaN samples
    (undefined, set to 0), and the timepoints (``cancelled``, a boolean mask)
    where every correlation was set to 0 because the kernel-weighted variances
    there are zero.

    ``warmup`` (the causal estimator: a boolean mask of its undefined rows,
    with ``min_samples``) replaces the first of these: the message names how
    many rows are undefined and returned as NaN.
    """
    T = _kernel_shape(weights)[0]
    parts = []
    problems = _kernel_row_problems(weights)
    if warmup is not None:
        if np.any(warmup):
            rule = "" if min_samples is None else (
                " or an effective sample size ((sum |w|)**2 / sum w**2 over the "
                f"kernel row) below min_samples={min_samples:g}"
            )
            parts.append(
                f"{int(np.sum(warmup))} of {T} timepoints have fewer than 3 "
                f"weighted, observed samples{rule}; their correlations are "
                "undefined and returned as NaN (causal=True)."
            )
        low = None if problems is None else problems[1] & ~warmup
        if low is not None and low.any():
            parts.append(
                f"{int(low.sum())} of {T} timepoints have an effective sample size "
                "((sum |w|)**2 / sum w**2 over the kernel row) below 2; the kernel "
                "is too narrow to estimate correlations reliably there."
            )
    elif problems is not None:
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
# block-sized products per row in the estimator's NaN path (one without NaN)
_WCORR_NAN_PRODUCTS = 10

# Time slicing of the estimator's kernel products (see ``_time_slices``): the
# output timepoints are processed in chunks of ``_kernels.CHUNK`` kernel rows,
# each against the span of kernel columns where the chunk has non-zero weight.
# Slicing is used only when the spans cover at most this fraction of the
# kernel's entries; otherwise the full products are computed.
_TIME_SLICE_MAX_FRACTION = 0.5
# Private switch: False always computes the full products.
_TIME_SLICING = True
# Kernel blocks the estimator keeps between products (``_KernelRows``), in
# float64 elements (32 MB); blocks beyond it are computed again when needed.
_KERNEL_CACHE_ELEMENTS = 2**22
# Besides those blocks, the estimator holds up to this many blocks of
# CHUNK rows by up to T columns while it computes a LagKernel's rows
# (measured with tracemalloc: about 3 at T 3,000 and 20,000).
_KERNEL_WORK_BLOCKS = 4
# Private hook for tests: when a list, ``_time_slices`` appends its result.
_time_slice_log = None

# The PageRank system ``B r = b`` of ``reduce(..., rfun='pagerank_centrality')``
# is ill-posed for a timepoint when the solve fails, the solution is not
# finite, its sum is below _PAGERANK_SUM_TOL * max(1, max|r|), its relative
# residual ||B r - b|| / ||b|| exceeds _PAGERANK_RESIDUAL_TOL, or LAPACK's
# estimate of the reciprocal 1-norm condition number of B is below
# _PAGERANK_RCOND (singular to within about 50 units of round-off; the
# solution is then not unique to working precision).  _PAGERANK_RCOND is
# also the relative singular-value cutoff of the least-squares fallback.
_PAGERANK_SUM_TOL = 1e-12
_PAGERANK_RESIDUAL_TOL = 1e-8
_PAGERANK_RCOND = 1e-14

# The estimator's denominator is sqrt(var_a * var_b) where that product is
# finite and at least _VARIANCE_PRODUCT_MIN, and sqrt(var_a) * sqrt(var_b)
# elsewhere (products that overflow, or are subnormal or nearly so and have
# lost precision).
_VARIANCE_PRODUCT_MIN = 1e-290
# Kernel rows whose largest absolute weight lies outside
# [_KERNEL_ROW_MIN, _KERNEL_ROW_MAX] are divided by that weight before the
# moment sums; other rows are used as given.
_KERNEL_ROW_MIN = 1e-100
_KERNEL_ROW_MAX = 1e100

# Relative round-off of a column's samples: a column whose samples agree to
# within this fraction of their magnitude is constant (see
# _round_off_variance).  The bound is the round-off this library's own
# arithmetic produces, with a stated margin: on output columns that are +-1
# in exact arithmetic (diagonals and pairs of proportional columns, from
# timecorr with autofc, isfc and wisfc and from corrmean_combine and
# mean_combine; T 60 to 2000, K 3 and 6, nine built-in kernels as the next
# order's kernel), the statistic sd / rms of the rule reached 37.95 eps, the
# same with Accelerate and OpenBLAS.  48 eps is that times 1.26.
_ROUND_OFF = 48 * np.finfo(np.float64).eps


def _round_off_variance(var, s, mean):
    """
    True where a (kernel-)weighted variance is at the round-off level of its
    column: ``var <= _ROUND_OFF**2 * (var + s * mean**2)``.

    ``var`` is ``sum |w| (x - mean)**2``, ``s`` is ``sum |w|`` and ``mean``
    the weighted mean of the raw samples (overall mean included), so
    ``var + s * mean**2`` is the column's weighted mean square
    ``sum |w| x**2``.  Bound: samples ``x = c (1 + d)`` that equal ``c`` up
    to round-off, ``|d| <= k * eps``, have
    ``var <= sum |w| (c d)**2 <= (k eps)**2 * s c**2``, which is at most
    ``(k eps)**2 / (1 - k eps)**2`` times the mean square: below the bound
    for ``k`` below 48 (``_ROUND_OFF / eps``).  The largest value of
    ``sd / rms`` measured on the library's own outputs that are constant in
    exact arithmetic is 37.95 eps (see ``_ROUND_OFF``).  The variance is
    compared as a standard deviation with ``hypot``, so neither side
    overflows or underflows before the comparison; the rule does not depend
    on the units of the column.  A genuine spread whose ratio of standard
    deviation to root mean square is at most ``_ROUND_OFF`` (1.07e-14)
    counts as constant: a column on an offset of more than about 9.4e13
    standard deviations (``1 / _ROUND_OFF``).  The samples 0, ..., 4 on an
    offset of 1e14 (ratio 1.41e-14) keep their spread.
    """
    sd = np.sqrt(np.maximum(var, 0.0))
    return sd <= _ROUND_OFF * np.hypot(sd, np.sqrt(s) * np.abs(mean))


def _zero_variance(var, msq, s, mean):
    """
    The estimator's zero-variance rule: True where a kernel-weighted variance
    ``var`` counts as zero.  ``msq`` is the kernel-weighted mean square of the
    globally centered column, ``s`` the kernel's ``sum |w|`` and ``mean``
    the kernel-weighted mean of the raw column.  The variance is zero when it
    is at most ``_ZERO_VARIANCE_TOL`` times ``msq`` (the numerical limit of
    computing it from centered sums) or at the round-off level of the raw
    column (``_round_off_variance``).
    """
    return (var <= _ZERO_VARIANCE_TOL * msq) | _round_off_variance(var, s, mean)

# Values at most this far outside [-1, 1] are correlations with round-off error
# (clipped to +-1), not co-fluctuations.
_R_TOLERANCE = 1e-12
# Correlations at least 1 - _UNIT_TOLERANCE in magnitude are +-1 up to
# round-off in Fisher z averaging (see _correlation_z).  A correlation
# r = sxy / sqrt(sxx * syy) of exactly proportional columns carries the
# relative round-off of its three kernel-weighted sums and of the division
# and square root, a small multiple of eps that grows with the terms
# summed; measured on the outputs of timecorr (autofc, isfc, wisfc) and of
# corrmean_combine and mean_combine that are +-1 in exact arithmetic (T 60
# to 2000, K 3 and 6, nine built-in kernels), 1 - |r| reached 79 eps, the
# same with Accelerate and OpenBLAS.  128 eps (2.8e-14) is that times 1.6;
# 1 - 1e-13 (450 eps) is a correlation and keeps its finite z.
_UNIT_TOLERANCE = 128 * np.finfo(np.float64).eps


def _safe_divide(num, den):
    """Elementwise num / den, with 0 wherever den is 0."""
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _nan_mean(x):
    """Each column's mean over its non-NaN samples (0 for all-NaN columns)."""
    valid = ~np.isnan(x)
    total = np.where(valid, x, 0.0).sum(axis=0)
    return _safe_divide(total, valid.sum(axis=0))


def _nan_center(x):
    """Subtract each column's mean over its non-NaN samples (0 for all-NaN columns)."""
    return x - _nan_mean(x)


def _zscore_columns(x):
    """
    Population z-scores of each column; NaN samples and constant columns give 0.

    A column is constant when its standard deviation is at most 1e-10 of its
    largest centered value, or at the round-off level of its root mean
    square (``_round_off_variance``, the estimator's rule with equal
    weights): a varying column is zeroed only on an offset of more than
    about 9.4e13 of its standard deviations, and a constant one with
    round-off is not turned into signal.
    """
    valid = ~np.isnan(x)
    n = valid.sum(axis=0)
    mean = _nan_mean(x)
    x0 = np.where(valid, x - mean, 0.0)
    ss = (x0**2).sum(axis=0)
    sd = np.sqrt(_safe_divide(ss, n))
    constant = (sd <= _ZERO_VARIANCE_TOL * np.abs(x0).max(axis=0)) | (
        _round_off_variance(ss, n, mean + _safe_divide(x0.sum(axis=0), n))
    )
    return _safe_divide(x0, np.where(constant, 0.0, sd))


def _wcorr_block_rows(T, Kb, n_products=1):
    """Rows of ``a`` per block in ``_wcorr_blocks`` (each row pairs with Kb columns)."""
    return max(1, _WCORR_BLOCK_ELEMENTS // (n_products * T * Kb))


def _kernel_work_elements(T, dense_elements):
    """
    Float64 elements the estimator holds for one kernel of T timepoints that
    it reads: for a ``LagKernel``, at most its kept blocks
    (``_KERNEL_CACHE_ELEMENTS``) plus ``_KERNEL_WORK_BLOCKS`` blocks of
    ``CHUNK`` x T, and at most ``dense_elements`` (what the same
    kernel costs as T x T arrays), whichever is smaller.
    """
    T = int(T)
    return min(
        int(dense_elements),
        _KERNEL_CACHE_ELEMENTS + _KERNEL_WORK_BLOCKS * CHUNK * T,
    )


def _kernel_shape(weights):
    """``(T', T)`` of a kernel (an array, or a ``LagKernel``)."""
    if isinstance(weights, LagKernel):
        return weights.shape
    return np.shape(weights)


def _abs_row_chunks(weights):
    """
    ``(t0, t1, w)`` for consecutive chunks of ``CHUNK`` kernel rows: ``w``
    holds the absolute weights of rows ``t0:t1`` (NaN as 0) over a span of
    columns that contains every non-zero weight of the chunk.  Chunks with
    equal blocks share one array.
    """
    def absolute(w):
        # as np.abs(np.nan_to_num(w, nan=0.0)): NaN 0, +-inf the largest float
        w = np.abs(w)
        w[np.isnan(w)] = 0
        w[np.isinf(w)] = np.finfo(np.float64).max
        return w

    for t0, t1, _, _, _, w in blocks(as_kernel(weights), CHUNK, absolute):
        yield t0, t1, w


def _without_nan(w):
    """
    A kernel block with NaN weights set to 0 (inf kept): ``w`` itself when it
    has none, otherwise a copy.
    """
    nan = np.isnan(w)
    if not nan.any():
        return w
    w = w.copy()
    w[nan] = 0
    return w


class _KernelRows:
    """
    The estimator's working copy of a (T', T) kernel, read in chunks of
    ``CHUNK`` rows.

    ``weights`` is an array or a ``LagKernel``; both are read through the
    interface of ``_kernels`` (``as_kernel``): chunks of rows, each with the
    columns where it can have weight (all columns of an array) and a key
    shared by chunks with equal blocks.  A ``LagKernel``'s rows are computed
    chunk by chunk, so no T' x T array is allocated.  Each chunk's products
    and row sums are computed from a contiguous block of the chunk's rows
    over the same columns, in the same order, so an array and the equal
    ``LagKernel`` give the same values.  NaN weights count as 0.

    Use: ``support`` (non-zero weights per row) and ``first_nonzero()`` are
    available at once; ``zero_rows(mask)`` sets rows to 0; ``prepare(finite)``
    divides each row whose largest absolute weight is below
    ``_KERNEL_ROW_MIN`` or above ``_KERNEL_ROW_MAX`` by that weight (a
    correlation does not depend on the scale of its kernel row, and the
    moment sums of such rows would underflow or overflow) and chooses the
    columns of each chunk (``_time_slices``; every column when slicing is
    off, when ``finite`` is False, or when the spans cover more than
    ``_TIME_SLICE_MAX_FRACTION`` of the kernel); then ``product(x, kind)``
    and ``row_sum(kind)`` give ``M @ x`` and ``M.sum(axis=1)`` for ``M`` the
    kernel (``kind`` "w"), its absolute value ("a") or its support as 0/1
    ("s").  Blocks are kept between products up to
    ``_KERNEL_CACHE_ELEMENTS`` elements and computed again beyond that.
    """

    def __init__(self, weights):
        self.kernel = as_kernel(weights)
        self.shape = self.kernel.shape
        self._chunks = list(self.kernel.chunks(CHUNK))
        T_out = self.shape[0] if len(self.shape) == 2 else 0
        self.support = np.zeros(T_out, dtype=np.int64)
        self._peak = np.zeros(T_out)
        self._first = np.zeros(T_out, dtype=np.int64)
        for t0, t1, c0, c1, _, w in blocks(self.kernel, CHUNK, _without_nan):
            self.support[t0:t1] = np.count_nonzero(w, axis=1)
            if w.shape[1]:
                self._peak[t0:t1] = np.maximum(w.max(axis=1), -w.min(axis=1))
                self._first[t0:t1] = c0 + np.argmax(w != 0, axis=1)
        self.zero = None
        self.zero_cols = None
        self._scale = None
        self.spans = None
        self._cache = {}
        self._cached = 0
        self._kept = set()

    def first_nonzero(self):
        """The column of each row's first non-zero weight (0 for an empty row)."""
        return np.where(self.support > 0, self._first, 0)

    def zero_rows(self, mask):
        """Set the rows in ``mask`` to 0 (before ``prepare``)."""
        self.zero = np.asarray(mask, dtype=bool)

    def zero_columns(self, mask):
        """
        Set the columns in ``mask`` to 0 (before ``prepare``): timepoints
        that no row weights (the causal estimator's timepoints without a
        sample).
        """
        mask = np.asarray(mask, dtype=bool)
        self.zero_cols = mask if mask.any() else None

    def prepare(self, finite, always=False):
        """
        Rescale the extreme rows and choose each chunk's columns
        (``always``: the columns where the chunk has weight before
        ``zero_rows``, however much of the kernel they cover, as the causal
        estimator needs).
        """
        T = self.shape[1] if len(self.shape) == 2 else 0
        peak = self._peak if self.zero is None else np.where(self.zero, 0.0, self._peak)
        extreme = (
            (peak > 0) & np.isfinite(peak)
            & ((peak < _KERNEL_ROW_MIN) | (peak > _KERNEL_ROW_MAX))
        )
        self._scale = np.where(extreme, peak, 1.0) if extreme.any() else None
        # blocks of the rows before zeroing and rescaling are stale; the final
        # blocks _time_slices reads are kept for the products
        self._cache = {}
        self._cached = 0
        self._kept = set()
        slices = None
        if len(self.shape) == 2:
            if always:
                # from the kernel before any row or column is set to 0:
                # which rows the causal estimator leaves undefined and which
                # timepoints have no sample depends on the data, and the
                # columns of a chunk (the shapes of its sums) must not
                zero, cols = self.zero, self.zero_cols
                self.zero = self.zero_cols = None
                slices = _time_slices(self, finite, True)
                self.zero, self.zero_cols = zero, cols
            else:
                slices = _time_slices(self, finite)
        if slices is None:
            slices = [(t0, t1, 0, T) for t0, t1, _, _, _ in self._chunks]
        self.spans = slices

    # the kernel interface of ``_kernels`` over the prepared rows (NaN
    # weights 0, rows zeroed and rescaled), read by ``_time_slices``
    def chunks(self, size):
        assert size == CHUNK
        for t0, t1, c0, c1, _ in self._chunks:
            yield t0, t1, c0, c1, None

    def block(self, t0, t1, c0, c1):
        assert (t0, t1, c0, c1) == self._chunks[t0 // CHUNK][:4]
        return self._final(t0 // CHUNK)

    def _keep(self, key, block, shared=False):
        """Cache ``block`` under ``key`` within ``_KERNEL_CACHE_ELEMENTS``;
        ``shared``: it is an array already cached, not counted again."""
        if key is None:
            return block
        if shared:
            self._cache[key] = block
        elif self._cached + block.size <= _KERNEL_CACHE_ELEMENTS:
            self._cache[key] = block
            self._cached += block.size
            self._kept.add(id(block))  # cached arrays are never dropped
        return block

    def _row_state(self, t0, t1, c0=0, c1=0):
        """
        The zeroed rows and scales of rows t0:t1 and the zeroed columns
        among c0:c1, as part of a cache key.
        """
        zero = None if self.zero is None else self.zero[t0:t1].tobytes()
        scale = None if self._scale is None else self._scale[t0:t1].tobytes()
        cols = None
        if self.zero_cols is not None and self.zero_cols[c0:c1].any():
            cols = self.zero_cols[c0:c1].tobytes()
        return zero, scale, cols

    def _final(self, i):
        """
        Chunk i over its columns: NaN weights 0, the zeroed rows and
        columns 0, the extreme rows rescaled.  Chunks with equal blocks,
        equal zeroed rows and columns and equal scales share one array (do
        not modify it).
        """
        t0, t1, c0, c1, key = self._chunks[i]
        zero = None if self.zero is None else self.zero[t0:t1]
        scale = None if self._scale is None else self._scale[t0:t1]
        # a chunk without a shared key (at an edge or an event boundary) is
        # kept under its own index
        key = (("final", ("chunk", i) if key is None else key)
               + self._row_state(t0, t1, c0, c1))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        w = self.kernel.block(t0, t1, c0, c1)
        if not w.flags.writeable:
            w = w.copy()  # a view of an array
        w[np.isnan(w)] = 0
        if zero is not None:
            w[zero] = 0
        if key[-1] is not None:
            w[:, self.zero_cols[c0:c1]] = 0
        if scale is not None:
            rows = scale != 1.0
            if rows.any():
                w[rows] /= scale[rows, np.newaxis]
        return self._keep(key, w)

    def _block(self, i, kind):
        t0, t1, c0, c1 = self.spans[i]
        key = self._chunks[i][4]
        # chunks with equal blocks, equal rows zeroed and rescaled (as in
        # _final) and the same span relative to t0 have equal blocks
        key = (("chunk", i) if key is None
               else (key, c0 - t0, c1 - t0)
               + self._row_state(t0, t1, *self._chunks[i][2:4]))
        hit = self._cache.get((key, kind))
        if hit is not None:
            return hit
        if kind == "w":
            k0, k1 = self._chunks[i][2:4]
            w = self._final(i)
            if (c0, c1) == (k0, k1):
                # the final block itself: counted once, where _final kept it
                return self._keep((key, kind), w, shared=id(w) in self._kept)
            elif c0 >= k0 and c1 <= k1:
                block = np.ascontiguousarray(w[:, c0 - k0 : c1 - k0])
            else:
                # every column (no slicing): zeros outside the chunk's columns
                block = np.zeros((t1 - t0, c1 - c0))
                lo, hi = max(k0, c0), min(k1, c1)
                if hi > lo:
                    block[:, lo - c0 : hi - c0] = w[:, lo - k0 : hi - k0]
        elif kind == "a":
            w = self._block(i, "w")
            if not np.signbit(w).any():
                # no negative weights (nor -0.0): its own absolute value
                if (key, "w") in self._cache:
                    self._cache[(key, "a")] = w
                return w
            block = np.abs(w)
        else:
            block = (self._block(i, "w") != 0).astype(np.float64)
        return self._keep((key, kind), block)

    def product(self, x, kind="w"):
        """``M @ x`` for a (T, n) matrix ``x`` (``M`` as in the class docstring)."""
        out = np.zeros((self.shape[0], x.shape[1]), dtype=np.result_type(x, np.float64))
        for i, (t0, t1, c0, c1) in enumerate(self.spans):
            if c1 > c0:
                np.matmul(self._block(i, kind), x[c0:c1], out=out[t0:t1])
        return out

    def row_sum(self, kind="w"):
        """``M.sum(axis=1)`` (``M`` as in the class docstring)."""
        out = np.zeros(self.shape[0])
        for i, (t0, t1, c0, c1) in enumerate(self.spans):
            if c1 > c0:
                out[t0:t1] = self._block(i, kind).sum(axis=1)
        return out


def _delta_zscores(a, b, rows):
    """
    Z-scored columns of ``a`` and ``b`` at each kernel row's non-zero position
    (a true delta kernel: one non-zero weight per row; ``rows`` a
    ``_KernelRows``), shapes (T', Ka) and (T', Kb).
    """
    idx = rows.first_nonzero()
    return _zscore_columns(a)[idx], _zscore_columns(b)[idx]


def _delta_wcorr(a, b, rows):
    """Co-fluctuation estimate for a true delta kernel (one non-zero weight per row)."""
    za, zb = _delta_zscores(a, b, rows)
    return np.einsum("ti,tj->ijt", za, zb)


def _wcorr_inputs(a, b, weights, causal=False):
    """
    The estimator's inputs: ``a`` and ``b`` as float (T, K) matrices and the
    kernel as a ``_KernelRows`` (NaN weights count as 0).  Returns
    ``(a, b, rows)``.

    Raises ``ValueError`` if the shapes do not match (T rows in ``a`` and
    ``b``, T columns in ``weights``) or, unless ``causal``, if every row of the kernel is zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]
    rows = _KernelRows(weights)
    if a.shape[0] != b.shape[0] or len(rows.shape) != 2 or rows.shape[1] != a.shape[0]:
        raise ValueError(
            "a and b must have the same number of rows (timepoints) T, and weights "
            f"must have T columns; got a {a.shape}, b {b.shape}, weights {rows.shape}."
        )

    # (the causal estimator returns rows without weight as NaN: a series of
    # one timepoint has such a kernel when the diagonal is left out)
    if np.all(rows.support == 0) and not causal:
        raise ValueError(
            "Every row of the kernel is zero; no correlation can be estimated."
        )
    return a, b, rows


def _time_slices(weights, finite=True, always=False):
    """
    Chunks of a kernel's rows and the span of columns where each has weight,
    or None when slicing would not save work.

    ``weights`` is a (T', T) kernel without NaN weights: an array, a
    ``LagKernel``, or the prepared rows of a ``_KernelRows`` (which reads
    its kernel through the same interface).  The rows are split into
    consecutive chunks of ``CHUNK`` rows; for each chunk ``(t0, t1, c0,
    c1)``, every non-zero weight of rows ``t0:t1`` lies in columns ``c0:c1``
    (``c0 == c1`` when the chunk has no weight).  The bounds are found one
    chunk at a time, so no T' x T boolean matrix is held.  Returns None when
    ``_TIME_SLICING`` is False, when ``finite`` is False (the caller's
    products have non-finite operands, where the full products give
    ``0 * inf = NaN`` that slicing would skip), when T' or T is 0, or when
    the spans cover more than ``_TIME_SLICE_MAX_FRACTION`` of the kernel's
    entries.  ``always=True`` (the causal estimator) returns the spans
    whatever ``_TIME_SLICING``, ``finite`` and the fraction covered are.  With
    it, the first column of a ``LagKernel``'s chunk is the one the kernel
    gives (``LagKernel.span``), which no later row changes.
    """
    kernel = as_kernel(weights)
    T_out, T = kernel.shape
    slices = None
    if (always or (_TIME_SLICING and finite)) and T_out and T:
        # the causal estimator: the first column of a chunk must not depend
        # on the rows after the chunk's first (a row may exist or not, as
        # the series is longer or shorter).  A LagKernel's own first column
        # is that of the chunk's first row (the largest lag with weight,
        # the start of its event), whereas a weight that is exactly 0
        # inside the kernel (a Mexican hat at lag sigma) can keep the
        # first row from the column a later row weights.
        structural = always and isinstance(getattr(kernel, "kernel", kernel), LagKernel)
        slices, covered = [], 0
        for t0, t1, c0, _, _, w in blocks(kernel, CHUNK):
            cols = (w != 0).any(axis=0)
            if cols.any():
                s0 = c0 if structural else c0 + int(cols.argmax())
                s1 = c0 + w.shape[1] - int(cols[::-1].argmax())
            else:
                s0 = s1 = 0
            slices.append((t0, t1, s0, s1))
            covered += (t1 - t0) * (s1 - s0)
        if not always and covered > _TIME_SLICE_MAX_FRACTION * T_out * T:
            slices = None
    if _time_slice_log is not None:
        _time_slice_log.append(slices)
    return slices


def _products_finite(a, b):
    """
    True when every operand of the estimator's products is finite: the
    samples of ``a`` and ``b`` (NaN samples excluded), their squares and
    their cross products.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        peaks = [
            np.max(np.abs(x), initial=0.0, where=~np.isnan(x)) if x.size else 0.0
            for x in (a, b)
        ]
        bounds = (peaks[0], peaks[1], peaks[0] * peaks[0], peaks[1] * peaks[1],
                  peaks[0] * peaks[1])
    return bool(np.all(np.isfinite(bounds)))


def _wcorr_setup(a, b, rows, autocorrelation):
    """
    The terms shared by every block of ``_wcorr_block``: the kernel (a
    ``_KernelRows``, prepared here: rescaled rows and time slices), the
    centered data and, without NaN samples, each feature's kernel-weighted
    moments (computed once; with ``autocorrelation`` those of ``a`` serve for
    ``b`` too).  Returns a dict.
    """
    finite = _products_finite(a, b)  # an infinite sample makes its column NaN
    # the overall means are kept: with the kernel-weighted means of the
    # centered columns they give the raw ones, which the round-off rule needs
    ca = _nan_mean(a)
    cb = _nan_mean(b)
    a = a - ca
    b = b - cb
    rows.prepare(finite and _products_finite(a, b))
    state = {"T": a.shape[0], "rows": rows, "a": a, "b": b, "ca": ca, "cb": cb}
    s = rows.row_sum("a")
    state["active"] = s > 0
    state["has_nan"] = bool(np.isnan(a).any() or np.isnan(b).any())

    if not state["has_nan"]:
        s = s[:, np.newaxis]
        state["sw"] = rows.row_sum("w")[:, np.newaxis]

        def moments(x, center):
            mean = _safe_divide(rows.product(x, "a"), s)
            msq = rows.product(x**2, "a")
            var = msq - s * mean**2
            zero = _zero_variance(var, msq, s, mean + center)
            return mean, rows.product(x, "w"), np.where(zero, 0.0, var), zero

        state["ma"], state["wa"], state["va"], state["za"] = moments(a, ca)
        if autocorrelation:
            state["mb"], state["wb"], state["vb"], state["zb"] = (
                state["ma"],
                state["wa"],
                state["va"],
                state["za"],
            )
        else:
            state["mb"], state["wb"], state["vb"], state["zb"] = moments(b, cb)
        state["n_products"] = 1
    else:
        state["ok_a"] = ~np.isnan(a)
        state["ok_b"] = ~np.isnan(b)
        state["a0"] = np.where(state["ok_a"], a, 0.0)
        state["b0"] = np.where(state["ok_b"], b, 0.0)
        state["n_products"] = _WCORR_NAN_PRODUCTS
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
    if state.get("causal"):
        return _causal_block(state, i0, i1, j0)
    T = state["T"]
    rows = state["rows"]
    Kb = state["b"].shape[1]
    shape = (rows.shape[0], i1 - i0, Kb - j0)

    if not state["has_nan"]:
        a, b = state["a"], state["b"]
        ma, mb, wa, wb = state["ma"], state["mb"], state["wa"], state["wb"]
        prod = (a[:, i0:i1, np.newaxis] * b[:, np.newaxis, j0:]).reshape(T, -1)
        cross = rows.product(prod, "w").reshape(shape)
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
        s0, sx, sy, sxx, syy = np.split(rows.product(terms, "a"), 5, axis=1)
        w0, wx, wy = np.split(rows.product(terms[:, : 3 * n], "w"), 3, axis=1)
        wxy = rows.product((x * y).reshape(T, -1), "w")
        mx = _safe_divide(sx, s0)
        my = _safe_divide(sy, s0)
        num = (wxy - mx * wy - my * wx + mx * my * w0).reshape(shape)
        var_a = sxx - s0 * mx**2
        var_b = syy - s0 * my**2
        pairs = (i1 - i0, Kb - j0)
        ca = np.broadcast_to(state["ca"][i0:i1, np.newaxis], pairs).reshape(-1)
        cb = np.broadcast_to(state["cb"][np.newaxis, j0:], pairs).reshape(-1)
        zero = _zero_variance(var_a, sxx, s0, mx + ca) | _zero_variance(
            var_b, syy, s0, my + cb
        )
        var_a, var_b, zero = (v.reshape(shape) for v in (var_a, var_b, zero))
        # two samples always give r = +-1: fewer than 3 is undefined
        few = rows.product(terms[:, :n], "s").reshape(shape) < 3
        few_any = few.reshape(shape[0], -1).any(axis=1)
        undefined = zero | few

    zero_all = zero.reshape(shape[0], -1).all(axis=1)
    r = _correlation_ratio(num, var_a, var_b, undefined, shape)
    return r, undefined, zero_all, few_any


def _correlation_ratio(num, var_a, var_b, undefined, shape):
    """
    ``num / sqrt(var_a * var_b)`` clipped to [-1, 1], with 0 where
    ``undefined``; arrays of ``shape`` (the variances may broadcast to it).
    """
    with np.errstate(over="ignore", under="ignore"):
        product = np.where(undefined, 1.0, var_a * var_b)
    den = np.sqrt(product)
    extreme = ~(np.isfinite(product) & (product >= _VARIANCE_PRODUCT_MIN)) & ~undefined
    if extreme.any():
        # the product of the variances overflowed or is subnormal (or
        # nearly so): take the square roots first
        den[extreme] = (
            np.sqrt(np.broadcast_to(var_a, shape)[extreme])
            * np.sqrt(np.broadcast_to(var_b, shape)[extreme])
        )
    return np.where(undefined, 0.0, np.clip(num / den, -1.0, 1.0))


# ----------------------------------------------------------------------------
# the causal estimator (``causal=True``)
# ----------------------------------------------------------------------------

# Samples larger than this in magnitude (their squares overflow), and
# infinite ones, are missing data for the causal estimator.
_CAUSAL_MAX = 1e150
# Relative tolerance of the comparison of a row's effective sample size with
# ``min_samples`` (three equal weights give 3 up to round-off).
_MIN_SAMPLES_RTOL = 1e-12


def _check_min_samples(min_samples, causal):
    """
    ``min_samples`` as None or a float; ``ValueError`` unless it is None or
    a finite number >= 1, and when it is given without ``causal=True``.
    """
    if min_samples is None:
        return None
    if (
        isinstance(min_samples, (bool, np.bool_))
        or not isinstance(min_samples, numbers.Real)
        or not 1 <= min_samples < np.inf
    ):
        raise ValueError(
            "min_samples must be None or a finite number >= 1 (the effective "
            f"number of weighted samples a row needs); got {min_samples!r}."
        )
    if not causal:
        raise ValueError(
            f"min_samples={min_samples!r} applies with causal=True only; pass "
            "causal=True, or leave min_samples out."
        )
    return float(min_samples)


# ``_kernel_reach`` evaluates a lag profile at up to this many lags.
_REACH_LAGS = 2 ** 22


def _profile_tails(profile, n):
    """
    ``(sum of |w|, sum of w**2)`` over the lags ``n, n + 1, ...`` of the lag
    profile of a built-in kernel function, from the profile's formula (the
    sums over a heavy or very wide tail converge too slowly to be added
    up), or None when there is no formula (or the tail has no end:
    ``uniform_weights``).

    - ``laplace_weights``: the two geometric series, exactly.
    - ``boxcar_weights``: the number of lags below ``width / 2``, exactly.
    - ``t_weights`` and ``gaussian_weights`` (densities): the integral of
      the tail from the survival function, with the first two terms of the
      Euler-Maclaurin formula (``f(n) / 2 - f'(n) / 12``); the square of
      either density is a multiple of a density of the same family.
    - ``mexican_hat_weights``, for ``n >= 2 sigma`` (where ``|w|``
      decreases): an upper bound of the sum of ``|w|``, and 0 for the sum
      of squares, so that the reach computed from them is an upper bound.
    """
    from scipy.stats import norm, t as student

    family = getattr(profile, "func", None)
    params = (getattr(profile, "keywords", None) or {}).get("params")
    if params is None:
        return None
    n = int(n)

    def value(x):
        return float(profile(np.array([float(x)]))[0])

    with np.errstate(all="ignore"):
        if family is boxcar_profile:
            last = float(params["width"]) / 2.0  # lags d < last have weight 1
            if np.isnan(last):
                return None
            count = np.inf if np.isinf(last) and last > 0 else max(
                0.0, float(np.ceil(last)) - n)
            return count, count
        if family is laplace_profile:
            scale = float(params["scale"])
            if not (np.isfinite(scale) and scale > 0):
                return None
            f = abs(value(n))
            return f / -np.expm1(-1.0 / scale), f * f / -np.expm1(-2.0 / scale)
        if family is gaussian_profile:
            var = float(params["var"])
            if not (np.isfinite(var) and var > 0):
                return None
            f = value(n)
            slope = -n / var * f
            first = norm.sf(n / np.sqrt(var)) + f / 2 - slope / 12
            second = (norm.sf(n / np.sqrt(var / 2)) / (2 * np.sqrt(np.pi * var))
                      + f * f / 2 - 2 * f * slope / 12)
            return float(first), float(second)
        if family is t_profile:
            df = float(params["df"])
            f, peak = value(n), value(0)
            if not (np.isfinite(df) and df > 0 and np.isfinite(peak) and peak > 0):
                return None
            slope = f * (-(df + 1) * n / df) / (1 + n * n / df)
            first = student.sf(n, df) + f / 2 - slope / 12
            # the squared density is a t density of 2 df + 1 degrees of
            # freedom on a rescaled axis
            df2 = 2 * df + 1
            second = (peak * peak * np.sqrt(df / df2)
                      * student.sf(n * np.sqrt(df2 / df), df2) / student.pdf(0, df2)
                      + f * f / 2 - 2 * f * slope / 12)
            return float(first), float(second)
        if family is mexican_hat_profile:
            sigma = abs(float(params["sigma"]))
            if not (np.isfinite(sigma) and sigma > 0) or n < 2 * sigma:
                return None
            # the integral of |w| from n on is n |w(n)| / (n**2 / sigma**2 - 1)
            f = abs(value(n))
            return f + n * f / (n * n / (sigma * sigma) - 1), 0.0
    return None


# ``_strided_reach`` evaluates a lag profile at this many lags, a stride apart
_STRIDED_LAGS = 2 ** 16
# and returns its estimate times this, so that it is an upper bound
_STRIDED_MARGIN = 1.01


def _flat_profile(profile):
    """Whether a lag profile is the same weight at every lag
    (``uniform_weights``; ``boxcar_weights`` of infinite width)."""
    family = getattr(profile, "func", None)
    if family is uniform_profile:
        return True
    if family is boxcar_profile:
        params = (getattr(profile, "keywords", None) or {}).get("params") or {}
        try:
            return float(params.get("width")) == np.inf
        except (TypeError, ValueError):
            return False
    return False


def _strided_reach(profile, lo):
    """
    An upper bound of the largest effective sample size over the lags
    ``lo..n``, for every n, of a profile too wide for its lags to be added
    up one by one: the profile is read at every ``s``-th lag, for the
    smallest stride ``s`` (a power of 16) at which the sums converge within
    ``_STRIDED_LAGS`` lags, and the result is ``s`` times the effective
    sample size of those lags (the profile varies little within a stride;
    there are at least ``_STRIDED_LAGS / 16`` lags in its extent), times
    ``_STRIDED_MARGIN``.  None when the sums converge at no stride.
    """
    k = np.arange(_STRIDED_LAGS, dtype=np.float64)
    half = _STRIDED_LAGS // 2
    stride = 16.0
    while np.isfinite(lo + stride * _STRIDED_LAGS):
        with np.errstate(all="ignore"):
            a = np.abs(np.asarray(profile(lo + stride * k), dtype=np.float64))
        a[~np.isfinite(a)] = 0
        peak = a.max()
        if peak > 0:
            a /= peak
            total, squares = np.cumsum(a), np.cumsum(a * a)
            if total[-1] - total[half - 1] <= 1e-12 * total[-1]:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ess = total**2 / squares
                return float(_STRIDED_MARGIN * stride * np.nanmax(ess))
        stride *= 16.0
    return None


def _profile_reach(profile, include_zero, exclude):
    """``_kernel_reach`` from the lag profile function and the restrictions
    of a backward kernel (``exclude``: ``exclude_timepoints`` or None)."""
    lo, hi = (0 if include_zero else 1), np.inf  # the lags lo..hi are kept
    if exclude is not None:
        x = int(abs(exclude))
        if exclude >= 0:
            hi = x
        else:
            lo = max(lo, x + 1)
    if lo > hi:
        return 0.0
    if _flat_profile(profile):
        # equal weights: the number of lags
        return None if np.isinf(hi) else float(hi - lo + 1)
    n = 1024
    while n <= _REACH_LAGS:
        with np.errstate(over="ignore", invalid="ignore"):
            a = np.abs(np.asarray(profile(np.arange(n, dtype=np.int64)), dtype=np.float64))
        a[~np.isfinite(a)] = 0
        a[:lo] = 0
        if hi < n:
            a[hi + 1:] = 0
        peak = a.max()
        if peak > 0:
            a /= peak  # so that tiny weights do not underflow when squared
            total = np.cumsum(a)
            squares = np.cumsum(a * a)
            with np.errstate(invalid="ignore", divide="ignore"):
                ess = total**2 / squares
            # converged: the later half of the lags adds nothing to the sums
            if total[-1] - total[n // 2 - 1] <= 1e-15 * total[-1]:
                return float(np.nanmax(ess))
            # not converged: the rest of the sums from the profile's formula
            # (the lags from n on: all kept when no lag ends them)
            tails = _profile_tails(profile, n) if np.isinf(hi) else None
            if tails is not None and np.isfinite(tails[0]):
                first, second = tails[0] / peak, tails[1] / peak**2
                with np.errstate(invalid="ignore", divide="ignore"):
                    limit = (total[-1] + first) ** 2 / (squares[-1] + second)
                return float(np.nanmax([np.nanmax(ess), limit]))
        elif hi < n:
            return 0.0  # every lag kept was read, and none has weight
        n *= 2
    n //= 2  # the lags 0..n-1 were read
    if np.isfinite(hi):
        # kept up to a lag beyond those read
        if peak == 0:
            return 0.0  # no weight at any lag read
        near, far = _profile_tails(profile, n), _profile_tails(profile, hi + 1)
        if near is not None and far is not None and near[1] > 0 and np.isfinite(near[0]):
            # weights that fall with the lag: the sums over the lags n..hi
            # from the formula, and the largest value at the last lag
            first = (near[0] - far[0]) / peak
            second = (near[1] - far[1]) / peak**2
            with np.errstate(invalid="ignore", divide="ignore"):
                limit = (total[-1] + first) ** 2 / (squares[-1] + second)
            return float(np.nanmax([np.nanmax(ess), limit]))
        # no formula: at most what the kernel reaches without that end
        return _profile_reach(profile, lo == 0, None)
    if lo >= n:
        # only lags beyond those read.  The four families whose weights
        # fall with the lag: the effective sample size grows with every
        # lag added, to the limit of the formula's sums
        tails = _profile_tails(profile, lo)
        if tails is not None and tails[1] > 0:
            return float(tails[0] ** 2 / tails[1]) if np.isfinite(tails[0]) else None
        if tails is not None and tails[0] == 0:
            return 0.0
    elif peak == 0:
        return 0.0  # no weight at any lag read (parameters that are not finite)
    return _strided_reach(profile, lo)


def _kernel_reach(kernel):
    """
    The largest effective sample size ``(sum |w|)**2 / sum w**2`` a row of a
    backward kernel can have with any number of earlier timepoints: the
    largest over the lags kept up to lag n, for every n.  The lags kept are
    those the kernel's restrictions leave: from 1 without the diagonal
    (``include_zero=False``); up to ``x`` with ``exclude_timepoints=x >=
    0``, from ``x + 1`` with ``exclude_timepoints=-x``.  It does not depend
    on the length of the series (nor on events: the last block of a series
    grows with it).

    For a kernel with the same weight at every lag (``uniform_weights``, a
    ``boxcar_weights`` of infinite width) it is the number of lags kept.
    Otherwise the sums are added up over the first lags; when they have not
    converged there (a heavy tail, as of ``t_weights`` with few degrees of
    freedom, or a very wide kernel), the rest comes from the profile's
    formula (``_profile_tails``).  So the result is known for every
    built-in kernel function and parameter value: exactly for
    ``boxcar_weights``, ``laplace_weights`` and ``uniform_weights``, to
    round-off of the tail formula for ``gaussian_weights`` and
    ``t_weights``, and as an upper bound (so that a ``min_samples`` is
    never refused wrongly) for a ``mexican_hat_weights`` wider than about
    1000 samples (beyond 2 million samples from ``_strided_reach``; one
    that ``exclude_timepoints`` ends beyond the 4 million lags added up
    has what it reaches without that end).  0 for a kernel without weight
    at the lags kept.

    None when it has no limit (equal weights at lags without end: the
    effective sample size grows with the number of earlier timepoints) and
    for a kernel without a lag profile function (an array, a kernel
    function of the user's).  The result is kept on the kernel.
    """
    profile = getattr(kernel, "lag_profile", None)
    if profile is None or kernel.include != "backward":
        return None
    cached = getattr(kernel, "_reach", None)
    if cached is not None:
        return cached[0]
    reach = _profile_reach(profile, kernel.include_zero, kernel.exclude)
    if reach is not None and not np.isfinite(reach):
        reach = None
    kernel._reach = (reach,)
    return reach


def _check_reachable(kernel, min_samples):
    """
    ``ValueError`` when no row of ``kernel``, however many timepoints come
    before it, reaches ``min_samples`` (``_kernel_reach``; never for a kernel
    without a lag profile function).  The decision does not depend on the
    length of the series: rows of a short series that do not reach
    ``min_samples`` are undefined, not an error.
    """
    most = _kernel_reach(as_kernel(kernel))
    if most is None or most >= min_samples * (1 - _MIN_SAMPLES_RTOL):
        return
    raise ValueError(
        f"min_samples={min_samples:g} is more than a row of the kernel can "
        "have: with any number of earlier timepoints its largest effective "
        f"sample size ((sum |w|)**2 / sum w**2 over a row) is {most:.4g}.  "
        "Pass a min_samples of at most that, a wider kernel, or leave "
        "min_samples out."
    )


def _causal_clean(x):
    """
    ``x`` as the causal estimator reads it: infinite samples and samples
    above ``_CAUSAL_MAX`` in magnitude are NaN (missing).  ``x`` itself when
    it has none, otherwise a copy.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bad = np.abs(x) > _CAUSAL_MAX
    if bad.any():
        x = x.copy()
        x[bad] = np.nan
    return x


def _abs_weights(w):
    """The absolute values of a kernel block, NaN weights as 0 (a new array)."""
    w = np.abs(w)
    w[np.isnan(w)] = 0
    return w


def _check_causal_kernel(weights, what="the kernel"):
    """
    ``ValueError`` if a (T', T) kernel (an array or a ``LagKernel``) has
    weight above its diagonal: row t would weight timepoints after t.
    """
    kernel = as_kernel(weights)
    if isinstance(kernel, LagKernel) and kernel.include == "backward":
        return
    if len(kernel.shape) != 2:
        return
    for t0, t1, c0, c1, _, w in blocks(kernel, CHUNK):
        later = np.triu(w, t0 - c0 + 1)
        if np.any((later != 0) & ~np.isnan(later)):
            raise ValueError(
                f"causal=True: {what} gives weight to later timepoints (it has "
                "non-zero entries above its diagonal).  Pass a kernel whose row "
                "t weights only timepoints at or before t, for example "
                "tc.one_sided(weights, 'backward') or a kernel function called "
                "with direction='backward'."
            )


def _check_causal_rows(rows):
    """
    The checks of a causal call on its kernel (a ``_KernelRows``).

    A delta kernel is one with a single non-zero weight in every row at any
    length of the series.  The one row of a series of one timepoint has a
    single weight whatever the kernel: it is a delta kernel only when the
    kernel's lag profile says so (``_kernel_reach`` of at most 1), so a
    prefix of a series is never refused where the series is not; the same
    holds for a built-in kernel whose rows are cut by events.
    """
    delta = bool(rows.support.size) and bool(np.all(rows.support == 1))
    if delta and (rows.shape[0] == 1 or getattr(rows.kernel, "labels", None) is not None):
        # one row, or rows cut by events (blocks of one timepoint): the
        # kernel's own lag profile decides
        reach = _kernel_reach(rows.kernel)
        delta = reach is not None and reach <= 1 + _MIN_SAMPLES_RTOL
    if delta:
        raise ValueError(
            "causal=True: a delta kernel (one non-zero weight in every row, as "
            "eye_weights gives) yields co-fluctuations of series z-scored over "
            "the whole recording, which use later timepoints.  Pass a kernel "
            "with at least 3 weights per row, for example "
            "tc.boxcar_weights(T, {'width': 5}, direction='backward')."
        )
    _check_causal_kernel(rows.kernel)


def _kernel_samples(kernel, observed=None):
    """
    ``(count, ess)`` per row of a kernel: the number of observed timepoints
    with non-zero weight, and the effective sample size
    ``(sum |w|)**2 / sum w**2`` over them (0 for a row without any).

    ``observed`` is a boolean vector over the T timepoints (None: all).
    Each row is scaled by its largest weight first, so tiny weights do not
    underflow when squared.
    """
    T_out = kernel.shape[0]
    count = np.zeros(T_out, dtype=np.int64)
    ess = np.zeros(T_out)
    everything = observed is None or bool(np.all(observed))
    if not everything:
        observed = np.asarray(observed, dtype=np.float64)
    seen = {}
    for t0, t1, c0, c1, key, w in blocks(kernel, CHUNK, _abs_weights):
        stats = seen.get(key) if (key is not None and everything) else None
        if stats is None:
            if not everything:
                w = w * observed[c0:c1]
            n = np.count_nonzero(w, axis=1)
            peak = w.max(axis=1, keepdims=True, initial=0.0)
            with np.errstate(invalid="ignore"):
                w = np.divide(w, peak, out=np.zeros_like(w), where=peak != 0)
                total = w.sum(axis=1)
                np.multiply(w, w, out=w)
                stats = n, _safe_divide(total**2, w.sum(axis=1))
            if key is not None and everything:
                seen[key] = stats
        count[t0:t1], ess[t0:t1] = stats
    return count, ess


def _causal_defined(kernel, observed, min_samples, min_count=3):
    """
    The rows of a causal kernel that are defined: at least ``min_count``
    observed timepoints with non-zero weight and, with ``min_samples`` (not
    None), an effective sample size over the observed timepoints of at
    least ``min_samples`` (relative tolerance ``_MIN_SAMPLES_RTOL``).

    ``observed`` is a boolean vector over the T timepoints (a timepoint with
    no sample at all is not observed).  ``ValueError`` when the kernel
    cannot reach ``min_samples`` at any length (``_check_reachable``).
    """
    if min_samples is not None:
        _check_reachable(kernel, min_samples)
    count, ess = _kernel_samples(kernel, observed)
    defined = count >= min_count
    if min_samples is not None:
        with np.errstate(invalid="ignore"):
            defined &= ess >= min_samples * (1 - _MIN_SAMPLES_RTOL)
    return defined


# The causal estimator also counts a kernel-weighted standard deviation below
# this fraction of the magnitude of the kernel-weighted mean as zero: a column
# that is constant up to the round-off of the computation that produced it
# (an output of an earlier order; such columns are not set to their mean in
# causal mode, which would read later rows).
_CAUSAL_CONSTANT = 1e-9


def _causal_zero_variance(var, msq, s, mean):
    """
    The zero-variance rule of the causal estimator: ``_zero_variance`` (with
    ``msq`` the kernel-weighted mean square about the chunk's reference), or
    a kernel-weighted standard deviation ``sqrt(var / s)`` below
    ``_CAUSAL_CONSTANT`` times ``|mean|``, the kernel-weighted mean of the
    raw column.  Every quantity is a kernel-weighted sum of the row, so the
    rule reads no later sample.
    """
    sd = np.sqrt(np.maximum(var, 0.0))
    return _zero_variance(var, msq, s, mean) | (
        sd < _CAUSAL_CONSTANT * np.sqrt(s) * np.abs(mean)
    )


# The causal estimator centers the data anew for every group of this many
# kernel rows (see ``_group_reference``): a few standard deviations of a
# kernel as narrow as a Gaussian of variance 25.  Each group multiplies the
# centered columns over all the columns its chunk weights, so a chunk that
# weights more than ``_CAUSAL_SPAN`` columns (a kernel with weight at every
# lag, such as ``laplace_weights``) has proportionally larger groups, up to
# the whole chunk.
_CAUSAL_GROUP = 16
_CAUSAL_SPAN = 512


def _causal_reference(x, c0, t0, c1):
    """
    A centering reference of the columns of ``x`` for kernel rows from
    ``t0`` on that weight rows ``c0:c1``: the mean of each column's observed
    samples in rows ``c0:t0`` (before every such row); for a column without
    any, its first observed sample in rows ``c0:c1`` (0 when it has none).
    A kernel row that has samples of the column has that first sample in
    its past, so no row's reference holds a sample from its future.
    """
    past = x[c0:max(c0, t0)]
    ok = ~np.isnan(past)
    n = ok.sum(axis=0)
    ref = _safe_divide(np.where(ok, past, 0.0).sum(axis=0), n)
    none = n == 0
    if none.any():
        span = x[c0:c1][:, none]
        ok = ~np.isnan(span)
        first = span[ok.argmax(axis=0), np.arange(span.shape[1])]
        ref[none] = np.where(ok.any(axis=0), first, 0.0)
    return ref


def _group_reference(x, weights, c0, g0, c1):
    """
    The centering reference of the columns of ``x`` for a group of kernel
    rows that starts at row ``g0`` and weights rows ``c0:c1``: the
    kernel-weighted mean of each column's observed samples at row ``g0``
    (``weights``: the absolute weights of that row over ``c0:c1``), which
    every row of the group may read and which lies within the samples the
    group weights.  For a column without a weighted sample at row ``g0``
    (the first rows of a series or of an event, a row left undefined):
    ``_causal_reference``.

    A correlation does not depend on the reference; a reference close to
    the samples a row weights keeps its sums well conditioned.
    """
    span = _group_span(x, c0, c1)
    ok = ~np.isnan(span)
    num = weights @ np.where(ok, span, 0.0)
    den = weights @ ok.astype(np.float64)
    ref = _safe_divide(num, den)
    none = den == 0
    if none.any():
        ref[none] = _causal_reference(x[:, none], c0, g0, min(c1, x.shape[0]))
    return ref


def _causal_groups(rows):
    """
    ``(i, g0, g1, c0, c1, n)`` for the groups of consecutive rows ``g0:g1``
    of each chunk ``i`` of a prepared ``_KernelRows`` whose kernel has no
    weight above its diagonal.

    A group is computed with the shapes of a group of a series without an
    end, so that the operations that give a row do not depend on the number
    of timepoints after it (removing later rows changes no bit): ``n`` is
    the number of rows the group has in a complete chunk (``g1 - g0`` rows
    exist; the others are rows of zero weight), and its columns are
    ``c0:c1`` with ``c1 = g0 + n`` (the columns at or beyond the end of the
    data are columns of zeros).  ``n`` is ``_CAUSAL_GROUP``, times the
    number of ``_CAUSAL_SPAN`` columns a complete chunk weights (rounded
    up), at most what is left of the chunk.
    """
    groups = []
    for i, (t0, t1, c0, c1) in enumerate(rows.spans):
        if c1 <= c0:
            continue  # a chunk without weight
        end = t0 + CHUNK
        size = _CAUSAL_GROUP * max(1, -(-(end - c0) // _CAUSAL_SPAN))
        for g0 in range(t0, t1, size):
            n = min(end, g0 + size) - g0
            groups.append((i, g0, min(t1, g0 + n), c0, g0 + n, n))
    return groups


def _group_block(rows, group, kind):
    """The ``n`` rows and the columns ``c0:c1`` of a group of
    ``_causal_groups`` (a new array; zeros where the kernel has no row or
    column)."""
    i, g0, g1, c0, c1, n = group
    t0, _, _, k1 = rows.spans[i]
    block = rows._block(i, kind)
    out = np.zeros((n, c1 - c0))
    m = min(k1, c1) - c0
    out[: g1 - g0, :m] = block[g0 - t0 : g1 - t0, :m]
    return out


def _group_span(x, c0, c1):
    """Rows ``c0:c1`` of ``x`` (an array of timepoints, or timepoints by
    columns); rows at or beyond its end are zeros.  A view when every row
    exists."""
    part = x[c0:c1]
    if part.shape[0] == c1 - c0:
        return part
    out = np.zeros((c1 - c0,) + x.shape[1:], dtype=x.dtype)
    out[: part.shape[0]] = part
    return out


# The triangle of an autocorrelation is computed this many rows of features
# at a time, each with the features from the first of them on (the blocks of
# ``_wcorr_blocks`` are whole numbers of them in causal mode).
_CAUSAL_FEATURES = 8


def _causal_setup(a, b, rows, autocorrelation, min_samples):
    """
    ``_wcorr_setup`` for the causal estimator (``a`` and ``b`` from
    ``_causal_clean``, the kernel checked by ``_check_causal_rows``).

    Nothing computed for a kernel row uses a sample after the row, and the
    floating-point operations that give a row do not depend on the samples
    it does not weight:

    - the undefined rows (``_causal_defined``, from the timepoints where
      both ``a`` and ``b`` have a sample) are set to 0 here and to NaN in the
      output;
    - every chunk of ``CHUNK`` rows is multiplied with the columns where it
      has weight, whatever fraction of the kernel they cover, a group of
      ``_CAUSAL_GROUP`` rows at a time (``_causal_groups``);
    - each group centers the data on its own reference
      (``_group_reference``), and the zero-variance rule compares a
      variance with the kernel-weighted mean square about that reference
      and with the kernel-weighted mean (``_causal_zero_variance``);
    - a timepoint where ``a`` or ``b`` has no sample at all (a row of NaN,
      as the undefined rows of an earlier order are) has weight 0 in every
      row;
    - a row is computed with the sums for complete data exactly when no
      timepoint it weights has a NaN sample (``"plain"``), and with the
      pairwise sums otherwise.  NaN samples are set to 0 after centering, so
      the sums for complete data of such a row hold only exact zeros from
      the timepoints it does not weight: a NaN (or any other value) after a
      row changes no bit of the row.

    Returns a dict with ``"causal": True`` for ``_wcorr_block``.
    """
    T_out = rows.shape[0]
    Ka, Kb = a.shape[1], b.shape[1]
    nan_a = np.isnan(a)
    nan_b = nan_a if autocorrelation else np.isnan(b)
    # timepoints without a sample of ``a`` or of ``b``: no pair of features
    # has one there, so no row weights them (their kernel columns are 0)
    missing = nan_a.all(axis=1) | nan_b.all(axis=1)
    defined = _causal_defined(rows.kernel, ~missing, min_samples)
    rows.zero_rows(~defined)
    rows.zero_columns(missing)
    rows.prepare(True, always=True)
    # the timepoints with a NaN sample, their number before each row, and
    # those among them that rows can weight
    with_nan = nan_a.any(axis=1) | nan_b.any(axis=1)
    nan_before = np.concatenate([[0], np.cumsum(with_nan)])
    with_nan = (with_nan & ~missing).astype(np.float64)
    del nan_a, nan_b
    groups = _causal_groups(rows)
    n = len(groups)
    state = {
        "causal": True, "T": a.shape[0], "rows": rows, "a": a, "b": b,
        # an autocorrelation computed by _wcorr_blocks: its blocks hold the
        # features from their first row on (_corr_row_blocks: every feature)
        "triangular": autocorrelation,
        "defined": defined, "groups": groups,
        # groups whose columns hold a NaN sample; groups with a row that
        # weights one; the rows that weight none
        "dirty": np.zeros(n, dtype=bool),
        "nan": np.zeros(n, dtype=bool),
        "plain": np.ones(T_out, dtype=bool),
        "ref_a": np.zeros((n, Ka)),
        "sw": np.zeros((T_out, 1)),
        "ma": np.zeros((T_out, Ka)), "wa": np.zeros((T_out, Ka)),
        "va": np.zeros((T_out, Ka)), "za": np.ones((T_out, Ka), dtype=bool),
    }
    if autocorrelation:
        state.update(ref_b=state["ref_a"], mb=state["ma"], wb=state["wa"],
                     vb=state["va"], zb=state["za"])
    else:
        state.update({
            "ref_b": np.zeros((n, Kb)),
            "mb": np.zeros((T_out, Kb)), "wb": np.zeros((T_out, Kb)),
            "vb": np.zeros((T_out, Kb)), "zb": np.ones((T_out, Kb), dtype=bool),
        })
    active = np.zeros(T_out, dtype=bool)

    def moments(x, ref, dirty, A, W, s, g0, g1, side):
        k = g1 - g0  # the rows of the group that exist
        x = x - ref
        if dirty:
            x[np.isnan(x)] = 0
        mean = _safe_divide(A @ x, s)
        msq = A @ x**2
        var = msq - s * mean**2
        zero = _causal_zero_variance(var, msq, s, mean + ref)
        state["m" + side][g0:g1] = mean[:k]
        state["w" + side][g0:g1] = (W @ x)[:k]
        state["v" + side][g0:g1] = np.where(zero, 0.0, var)[:k]
        state["z" + side][g0:g1] = zero[:k]

    T = a.shape[0]
    for g, group in enumerate(groups):
        _, g0, g1, c0, c1, _ = group
        k = g1 - g0
        A = _group_block(rows, group, "a")
        s = A.sum(axis=1)[:, np.newaxis]
        active[g0:g1] = s[:k, 0] > 0
        dirty = state["dirty"][g] = nan_before[min(c1, T)] > nan_before[c0]
        if dirty:
            plain = (_group_block(rows, group, "s")
                     @ _group_span(with_nan, c0, c1))[:k] == 0
            state["plain"][g0:g1] = plain
            state["nan"][g] = not plain.all()
        state["ref_a"][g] = _group_reference(a, A[0], c0, g0, c1)
        if not autocorrelation:
            state["ref_b"][g] = _group_reference(b, A[0], c0, g0, c1)
        W = _group_block(rows, group, "w")
        state["sw"][g0:g1, 0] = W.sum(axis=1)[:k]
        moments(_group_span(a, c0, c1), state["ref_a"][g], dirty, A, W, s, g0, g1, "a")
        if not autocorrelation:
            moments(_group_span(b, c0, c1), state["ref_b"][g], dirty, A, W, s, g0, g1,
                    "b")
    state["active"] = active
    # the sums for complete data set the block size; the pairwise sums of
    # the rows that need them are computed in blocks of their own
    # (``_causal_block``), so the blocks of the first do not depend on
    # whether the data hold NaN samples
    state["n_products"] = 1
    return state


def _causal_pairwise(A, W, S, xa, ok_a, xb, ok_b, ref_a, ref_b):
    """
    The pairwise sums of a group of rows of the causal estimator (``A``,
    ``W``, ``S``: the group's absolute weights, weights and support) for
    the centered, zero-filled columns ``xa`` and ``xb`` (``ok_a``,
    ``ok_b``: their observed samples): ``(r, undefined, zero, few)``, each
    of shape (rows of the group, columns of ``xa``, columns of ``xb``).

    The sums of each column of ``xa`` are matrix products of their own
    (the items of stacked ``np.matmul`` calls), so they do not depend on
    the columns of ``xa`` given with it.
    """
    m = xb.shape[1]
    # (columns of xa, timepoints, columns of xb)
    pair_ok = ok_a.T[:, :, np.newaxis] & ok_b[np.newaxis, :, :]
    x = xa.T[:, :, np.newaxis] * pair_ok
    y = xb[np.newaxis, :, :] * pair_ok
    terms = np.concatenate([pair_ok.astype(float), x, y, x**2, y**2], axis=2)
    s0, sx, sy, sxx, syy = np.split(np.matmul(A, terms), 5, axis=2)
    w0, wx, wy = np.split(np.matmul(W, terms[:, :, : 3 * m]), 3, axis=2)
    wxy = np.matmul(W, x * y)
    few = np.matmul(S, terms[:, :, :m]) < 3
    del terms, x, y, pair_ok
    mx = _safe_divide(sx, s0)
    my = _safe_divide(sy, s0)
    num = wxy - mx * wy - my * wx + mx * my * w0
    var_a = sxx - s0 * mx**2
    var_b = syy - s0 * my**2
    ca = ref_a[:, np.newaxis, np.newaxis]
    cb = ref_b[np.newaxis, np.newaxis, :]
    zero = _causal_zero_variance(var_a, sxx, s0, mx + ca) | (
        _causal_zero_variance(var_b, syy, s0, my + cb)
    )
    # two samples always give r = +-1: fewer than 3 is undefined
    und = zero | few
    r = _correlation_ratio(num, var_a, var_b, und, num.shape)
    # (rows of the group, columns of xa, columns of xb)
    return tuple(v.transpose(1, 0, 2) for v in (r, und, zero, few))


def _causal_block(state, i0, i1, j0):
    """
    ``_wcorr_block`` for the causal estimator (``state`` from
    ``_causal_setup``): the same sums, a group of kernel rows at a time,
    each over its own columns and about its own centering reference.  Every
    group computes the sums for complete data, on the data with NaN samples
    set to 0; a group with rows that weight a NaN sample also computes the
    pairwise sums, which those rows take.

    The operations that give the correlations of a row of features with
    other features at a timepoint do not depend on the rows ``i0:i1`` asked
    for, nor on the length of the series: every group has the shapes of
    ``_causal_groups``, the cross products of each row of features are one
    matrix product of their own (the items of a stacked ``np.matmul``), and
    the features a row is multiplied with are fixed.  For the triangle of
    an autocorrelation (``state["triangular"]`` and ``j0 == i0``) these are
    the features from the first row of its ``_CAUSAL_FEATURES`` rows on
    (``i0`` is a multiple of that number; the entries before it are those
    of the transposed pairs, computed with the rows before); otherwise the
    features ``j0:``.
    """
    rows = state["rows"]
    a, b = state["a"], state["b"]
    Kb = b.shape[1]
    shape = (rows.shape[0], i1 - i0, Kb - j0)
    r = np.zeros(shape)
    undefined = np.ones(shape, dtype=bool)
    zero_all = np.ones(shape[0], dtype=bool)
    few_any = np.zeros(shape[0], dtype=bool)
    triangular = state["triangular"] and j0 == i0
    if triangular:
        assert i0 % _CAUSAL_FEATURES == 0
        parts = [(s0, min(i1, s0 + _CAUSAL_FEATURES), s0)
                 for s0 in range(i0, i1, _CAUSAL_FEATURES)]
    else:
        parts = [(i0, i1, j0)]
    for g, group in enumerate(state["groups"]):
        _, g0, g1, c0, c1, _ = group
        k = g1 - g0  # the rows of the group that exist
        ref_a = state["ref_a"][g]
        ref_b = state["ref_b"][g]
        xa = _group_span(a, c0, c1)[:, i0:i1] - ref_a[i0:i1]
        xb = _group_span(b, c0, c1)[:, j0:] - ref_b[j0:]
        if state["dirty"][g]:
            ok_a = ~np.isnan(xa)
            ok_b = ~np.isnan(xb)
            xa[~ok_a] = 0
            xb[~ok_b] = 0
        W = _group_block(rows, group, "w")
        pairwise = state["nan"][g]
        if pairwise:
            need = ~state["plain"][g0:g1]
            A = _group_block(rows, group, "a")
            S = _group_block(rows, group, "s")
        zero_group = np.ones(k, dtype=bool)
        few_group = np.zeros(k, dtype=bool)
        for s0, s1, jb in parts:
            rows_ = slice(s0 - i0, s1 - i0)
            cols = slice(jb - j0, None)
            sub = (k, s1 - s0, Kb - jb)
            # one product per row of features: (rows of the group) x (columns)
            stack = xa.T[rows_, :, np.newaxis] * xb[np.newaxis, :, cols]
            cross = np.matmul(W, stack)[:, :k].transpose(1, 0, 2)
            del stack
            ma_i = state["ma"][g0:g1, s0:s1, np.newaxis]
            mb_j = state["mb"][g0:g1, np.newaxis, jb:]
            num = (
                cross
                - ma_i * state["wb"][g0:g1, np.newaxis, jb:]
                - mb_j * state["wa"][g0:g1, s0:s1, np.newaxis]
                + ma_i * mb_j * state["sw"][g0:g1, :, np.newaxis]
            )
            var_a = state["va"][g0:g1, s0:s1, np.newaxis]
            var_b = state["vb"][g0:g1, np.newaxis, jb:]
            zero = (state["za"][g0:g1, s0:s1, np.newaxis]
                    | state["zb"][g0:g1, np.newaxis, jb:])
            r_sub = _correlation_ratio(num, var_a, var_b, zero, sub)
            und = zero
            if pairwise:
                # the rows that weight a NaN sample take the pairwise sums
                und = zero.copy()
                zero = zero.copy()
                few = np.zeros(sub, dtype=bool)
                # (a few rows of features at a time: 10 products of the
                # group's timepoints by the pairs; each row of features is
                # a product of its own, so the split changes no bit)
                step = _wcorr_block_rows(c1 - c0, Kb - jb, _WCORR_NAN_PRODUCTS)
                for f0 in range(s0, s1, step):
                    f1 = min(s1, f0 + step)
                    at = slice(f0 - i0, f1 - i0)
                    pr, pu, pz, pf = _causal_pairwise(
                        A, W, S, xa[:, at], ok_a[:, at], xb[:, cols], ok_b[:, cols],
                        ref_a[f0:f1], ref_b[jb:],
                    )
                    to = slice(f0 - s0, f1 - s0)
                    r_sub[:, to][need] = pr[:k][need]
                    und[:, to][need] = pu[:k][need]
                    zero[:, to][need] = pz[:k][need]
                    few[:, to][need] = pf[:k][need]
                few_group |= few.reshape(k, -1).any(axis=1)
            zero_group &= zero.reshape(k, -1).all(axis=1)
            r[g0:g1, rows_, cols] = r_sub
            undefined[g0:g1, rows_, cols] = und
            if jb > j0:
                # the pairs with the rows of features before: transposed
                before = slice(0, jb - j0)
                r[g0:g1, rows_, before] = r[g0:g1, before, rows_].transpose(0, 2, 1)
                undefined[g0:g1, rows_, before] = (
                    undefined[g0:g1, before, rows_].transpose(0, 2, 1))
        zero_all[g0:g1] = zero_group
        few_any[g0:g1] = few_group
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


def _wcorr_blocks(a, b, rows, autocorrelation, causal=False, min_samples=None):
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

    ``causal=True`` selects the causal estimator (``_causal_setup``, with
    ``min_samples``): the rows it leaves undefined are NaN in ``out``, and
    their mask is returned as a fourth item (None otherwise).
    """
    T, Ka = a.shape
    Kb = b.shape[1]
    if causal:
        state = _causal_setup(a, b, rows, autocorrelation, min_samples)
    else:
        state = _wcorr_setup(a, b, rows, autocorrelation)
    out = np.zeros([Ka, Kb, rows.shape[0]])
    cancelled = state["active"].copy()
    sparse = np.zeros_like(cancelled)

    rows_per_block = _wcorr_block_rows(T, Kb, state["n_products"])
    if causal:
        # whole numbers of the rows of features _causal_block computes together
        rows_per_block = max(1, rows_per_block // _CAUSAL_FEATURES) * _CAUSAL_FEATURES
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
    if causal:
        warmup = ~state["defined"]
        out[:, :, warmup] = np.nan
        return out, cancelled, sparse, warmup
    return out, cancelled, sparse, None


def _corr_row_blocks(a, b, weights, symmetric, rows_per_block=None, *,
                     causal=False, min_samples=None):
    """
    The rows of ``_wcorr(a, b, weights)`` in blocks, without the full array.

    ``symmetric=True`` declares that ``a`` and ``b`` are the same data (an
    autocorrelation): each block's diagonal is set to 1 (0 where undefined)
    and the entries within the block's own rows and columns are averaged with
    their transpose.  Entries outside that square are computed in the
    transposed orientation, so they agree with ``_wcorr``'s mirrored
    autocorrelation output to round-off, not bit for bit.

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

    With ``causal=True`` the rows are those of ``_wcorr(..., causal=True,
    min_samples=min_samples, autocorrelation=symmetric)``: the undefined
    (warm-up) rows are NaN in every block.

    Raises ``ValueError`` as ``_wcorr`` does, and if ``symmetric`` is True but
    ``a`` and ``b`` differ in shape.
    """
    same = a is b
    a, b, rows = _wcorr_inputs(a, b, weights, causal)
    if causal:
        min_samples = _check_min_samples(min_samples, True)
        _check_causal_rows(rows)
        a = _causal_clean(a)
        b = a if same else _causal_clean(b)
    if symmetric and a.shape != b.shape:
        raise ValueError(
            "symmetric=True requires a and b of the same shape; "
            f"got a {a.shape}, b {b.shape}."
        )
    if rows_per_block is not None and int(rows_per_block) < 1:
        raise ValueError(f"rows_per_block must be at least 1; got {rows_per_block}.")
    T, Ka = a.shape
    Kb = b.shape[1]

    if np.all(rows.support == 1) and not causal:
        za, zb = _delta_zscores(a, b, rows)
        step = int(rows_per_block or _wcorr_block_rows(T, Kb))

        def delta_blocks():
            for i0 in range(0, Ka, step):
                i1 = min(Ka, i0 + step)
                none = np.zeros(rows.shape[0], dtype=bool)
                R = np.einsum("ti,tj->ijt", za[:, i0:i1], zb)
                yield i0, i1, R, none, none.copy()

        return delta_blocks()

    if causal:
        state = _causal_setup(a, b, rows, symmetric, min_samples)
        state["triangular"] = False  # every block holds every feature
        warmup = ~state["defined"]
    else:
        rows.zero_rows(rows.support < 3)
        state = _wcorr_setup(a, b, rows, symmetric)
        warmup = None
    step = int(rows_per_block or _wcorr_block_rows(T, Kb, state["n_products"]))
    active = state["active"]

    def blocks():
        for i0 in range(0, Ka, step):
            i1 = min(Ka, i0 + step)
            r, undefined, zero_all, few_any = _wcorr_block(state, i0, i1, 0)
            if symmetric:
                _symmetrize_block(r, undefined, i0)
            if warmup is not None:
                r[warmup] = np.nan
            yield i0, i1, r.transpose(1, 2, 0), active & zero_all, active & few_any

    return blocks()


def wcorr(a, b, weights, *, causal=False, min_samples=None):
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
      removed), or at most ``(48 eps)**2`` (about 1.1e-28; ``eps`` is the
      float64 machine epsilon) times the kernel-weighted mean square of its
      raw samples: samples ``c (1 + d)`` with ``|d|`` below ``48 eps``
      (about 1.07e-14; for example values equal to 1 up to round-off, at any
      scale ``c``) are constant.  48 eps covers the round-off of the
      library's own outputs that are constant in exact arithmetic (at most
      38 eps measured).  A constant feature has correlation 0 with every
      feature, itself included, at that timepoint, and counts in the
      zero-variance warning as an exactly constant one does.  Both rules
      are relative, so multiplying a feature by a constant does not change
      which entries are set to 0.  This is a numerical limit:
      features whose local variance is below about 1e-10 of their mean square
      are treated as constant.  An offset of the whole series matters only
      when it is extreme: within a kernel window, a feature whose ratio of
      kernel-weighted standard deviation to root mean square is at most
      ``48 eps`` (1.07e-14), that is an offset of more than about 9.4e13
      local standard deviations, is treated as constant (an offset of 1e9
      standard deviations is tested, and 0, ..., 4 on an offset of 1e14,
      a ratio of 1.41e-14, keep correlation 1); a step much larger than the local
      fluctuations within a kernel window does: a unit-variance series that
      jumps by 1e6 or more halfway through is treated as constant on both
      sides of the jump.  Since the 1e-10 rule's mean square is about the
      overall mean, its decision at ``t`` depends on samples that row ``t``
      of the kernel does not weight: with a one-sided kernel, an entry near
      the threshold can differ between a series and a prefix of it (0 in
      one, up to +-1 in the other).  Entries whose variance is at least
      1e-3 of that mean square do not depend on those samples (tested to
      1e-9).

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

    With ``causal=True`` nothing computed for timepoint ``t`` reads a sample
    after ``t``: the correlations at ``t`` are unchanged (to 1e-9) when the
    samples after ``t`` are replaced by any values, NaN and inf included, or
    removed (and, when the kernel gives ``t`` itself no weight, when the
    samples at ``t`` change).  Replacing later samples changes no bit of
    an earlier timepoint.  Removing them changes no bit either when the
    first timepoint that a chunk of 128 kernel rows weights is one its
    first row weights (the kernels ``timecorr`` builds from the built-in
    kernel functions, and arrays of the same weights, a Mexican hat whose
    weight is exactly 0 at a lag inside it excepted): groups of kernel
    rows have the shapes of a series without an end, with zeros beyond
    the data, and each feature's products are a matrix product of their
    own.  For other kernel arrays removal is documented to 1e-9.  The kernel must give no weight to later timepoints.  The estimate is the same formula, with these differences:

    - Data are centered, a group of 16 timepoints at a time (more for a
      kernel that weights more than 512 timepoints), on the kernel-weighted
      mean of each feature at the group's first timepoint (where that
      timepoint has no weighted sample of the feature: the mean of its
      samples before the group among those the group weights, or its first
      sample), not on the mean over all timepoints; the 1e-10 rule above
      compares the variance with the kernel-weighted mean square about that
      reference.  A correlation does not depend on the reference; it only
      conditions the sums.
    - A feature whose kernel-weighted standard deviation at a timepoint is
      below 1e-9 times the magnitude of its kernel-weighted mean also
      counts as having zero variance there: a feature that is constant up
      to the round-off of whatever computed it.
    - A timepoint is computed with the sums for complete data when none of the
      timepoints it weights has a NaN sample, and with pairwise sums
      otherwise, whatever the later samples are.  A timepoint where ``a``
      or ``b`` has no sample at all has no weight in any row.
    - Infinite samples and samples above 1e150 in magnitude are missing
      data, like NaN.
    - A timepoint with fewer than 3 observed samples with non-zero weight
      is undefined: all its correlations are NaN (the diagonal included),
      not 0; so is, with ``min_samples``, a timepoint whose effective
      sample size ``(sum |w|)**2 / sum w**2`` over the observed timepoints
      of its kernel row is below it.  A timepoint counts as observed when
      ``a`` and ``b`` each have at least one sample there.  Other undefined
      entries are 0, as without ``causal``: constant features, and pairs of
      features with fewer than 3 common samples.
    - ``a`` and ``b`` are an autocorrelation (diagonal set to 1) only when
      they are the same object; equal values in two arrays are not compared.
    - A delta kernel raises ``ValueError`` (its z-scores use every
      timepoint).  A series of one timepoint is not a delta kernel, and a
      kernel row without weight is an undefined timepoint (NaN), not an
      error: every prefix of a series can be computed.

    Parameters
    ----------
    a : numpy.ndarray
        Timepoints-by-features observations matrix (T, Ka).
    b : numpy.ndarray
        Timepoints-by-features observations matrix (T, Kb).
    weights : numpy.ndarray
        Kernel matrix with T columns; row ``t`` holds the weights used to
        estimate at timepoint ``t``.  Usually T x T.
    causal : bool, optional
        True: the causal estimator described above.  ``weights`` must have
        no non-zero entry above its diagonal (for example a kernel function
        called with ``direction='backward'``).  Keyword only.  Default:
        False.
    min_samples : float or None, optional
        With ``causal=True``, the effective sample size
        ``(sum |w|)**2 / sum w**2`` a timepoint needs (a number >= 1,
        compared with a relative tolerance of 1e-12, so three equal weights
        are 3 samples); timepoints below it are NaN too (every timepoint,
        when no row of the array reaches the number).  Keyword only.
        Default: None (only the rule of 3 samples).

    Returns
    -------
    numpy.ndarray
        Array of shape (Ka, Kb, weights.shape[0]); entry ``[i, j, t]`` is the
        correlation between ``a[:, i]`` and ``b[:, j]`` at timepoint ``t``.
        With ``causal=True`` the undefined timepoints are NaN.

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
        ``causal=True`` the first of these names the undefined timepoints
        (returned as NaN).

    Raises
    ------
    ValueError
        If the inputs do not have T rows / T kernel columns, or (without
        ``causal``) if every row of the kernel is zero.  With ``causal=True``: if the kernel has weight
        above its diagonal or is a delta kernel, or if ``min_samples`` is
        not None or a finite number >= 1.  If ``min_samples`` is given
        without ``causal=True``.

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

    The causal estimator: timepoint 60 does not depend on later samples.

    The first 2 timepoints have fewer than 3 samples and are NaN (a
    ``UserWarning`` says so).

    >>> past = tc.gaussian_weights(100, {"var": 50}, direction="backward")
    >>> r = tc.wcorr(x, x, past, causal=True)
    >>> bool(np.isnan(r[:, :, :2]).all()), bool(np.isnan(r[:, :, 2:]).any())
    (True, False)
    >>> y = x.copy()
    >>> y[61:] = 1000.0
    >>> bool(np.array_equal(tc.wcorr(y, y, past, causal=True)[:, :, 60], r[:, :, 60]))
    True
    """
    min_samples = _check_min_samples(min_samples, causal)
    return _wcorr(a, b, weights, causal=causal, min_samples=min_samples)


def _wcorr(a, b, weights, warn=True, return_cancelled=False, causal=False,
           min_samples=None, autocorrelation=None):
    """
    ``wcorr``.  With ``warn=False`` the warning is left to the caller; with
    ``return_cancelled=True`` the masks ``(cancelled, sparse)`` of the
    timepoints whose correlations all cancelled and of those where some pair
    has fewer than 3 observed samples (see ``_wcorr_blocks``) are returned too:
    ``(out, (cancelled, sparse))``.

    ``causal=True`` is the causal estimator (see ``wcorr``): the masks are
    then ``(cancelled, sparse, warmup)``, ``warmup`` marking the undefined
    rows (NaN in ``out``).  ``autocorrelation`` says whether ``b`` is ``a``
    (None: whether they are the same object; comparing their values would
    read every timepoint).
    """
    same = a is b
    a, b, rows = _wcorr_inputs(a, b, weights, causal)
    if causal:
        min_samples = _check_min_samples(min_samples, True)
        _check_causal_rows(rows)
        a = _causal_clean(a)
        b = a if same else _causal_clean(b)
        if autocorrelation is None:
            autocorrelation = same
        out, cancelled, sparse, warmup = _wcorr_blocks(
            a, b, rows, autocorrelation, True, min_samples
        )
        del rows
        if warn:
            message = _estimator_message(weights, cancelled, sparse, warmup,
                                         min_samples)
            if message is not None:
                _warn_user(message)
        return (out, (cancelled, sparse, warmup)) if return_cancelled else out
    if np.all(rows.support == 1):
        out = _delta_wcorr(a, b, rows)
        none = np.zeros(rows.shape[0], dtype=bool)
        return (out, (none, none.copy())) if return_cancelled else out

    rows.zero_rows(rows.support < 3)
    autocorrelation = a is b or (
        a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    )
    out, cancelled, sparse, _ = _wcorr_blocks(a, b, rows, autocorrelation)
    del rows
    if warn:
        message = _estimator_message(weights, cancelled, sparse)
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


def _fisher_step(K):
    """Timepoints per block of ``_fisher_symmetrize`` for K features."""
    # _fisher_mean makes about ten temporaries of its input's size (which is
    # twice the block): keep the block small enough that they stay below the
    # workspace of wcorr's own blocks
    return max(1, _WCORR_BLOCK_ELEMENTS // (16 * max(1, K * K)))


def _fisher_workspace(K, T):
    """
    Float64 elements of the temporaries of ``_fisher_symmetrize`` on a
    (K, K, T) array at their peak: 11 times one block of K x K x step
    timepoints (measured with tracemalloc: 10.6 times for K 20 to 300).
    """
    return 11 * K * K * min(int(T), _fisher_step(K))


def _fisher_symmetrize(c):
    """
    Symmetrize a (K, K, T) cross-correlation array as in Eq. 5 of the timecorr
    paper: ``z2r((r2z(C) + r2z(C.T)) / 2)``.

    Where one of ``C[i, j]`` and ``C[j, i]`` is +-1 (infinite z; values
    within round-off of +-1 count as +-1, see ``_correlation_z``), the
    result is that sign; where they
    are +1 and -1, it is 0.  Entries outside
    [-1, 1] (co-fluctuations from a delta kernel, which are not correlations)
    are averaged without the transform (see ``_fisher_mean``).

    ``c`` is overwritten and returned.  The work is done a block of timepoints
    at a time, so the temporaries stay at block size rather than the size of
    ``c``.
    """
    step = _fisher_step(c.shape[0])
    for t0 in range(0, c.shape[2], step):
        block = c[:, :, t0 : t0 + step]
        block[...] = _fisher_mean(
            np.stack([block, block.transpose(1, 0, 2)]), axis=0
        )
    return c


def _correlation_z(r):
    """
    ``(z, outside)``: the Fisher z of correlations ``r`` for averaging, and
    the mask of values that are not correlations.

    Values more than ``_R_TOLERANCE`` (1e-12) outside [-1, 1] are not
    correlations (for example co-fluctuations from a delta kernel): they are
    marked in ``outside`` and their z is 0.  Values above 1 in magnitude by
    at most ``_R_TOLERANCE``, and values at most ``_UNIT_TOLERANCE`` (128
    eps, 2.8e-14) below it, are +-1 up to round-off and have infinite z, as
    exact +-1 does; so one ulp below 1 does not turn an infinite z into a
    finite one (which, averaged with -inf, gives 1 instead of 0).  A value
    further below, such as 1 - 1e-13, keeps its finite z.  NaN stays NaN.
    """
    r = np.asarray(r, dtype=np.float64)
    # no more temporaries of r's size than the previous clip-and-where (the
    # memory estimates of corrmean_combine and tstat_combine count them)
    a = np.abs(r)
    outside = a > 1 + _R_TOLERANCE
    unit = a >= 1 - _UNIT_TOLERANCE
    del a
    unit &= ~outside
    c = np.where(outside, 0.0, r)
    np.copysign(1.0, c, out=c, where=unit)
    return r2z(c), outside


def _fisher_mean(r, axis):
    """
    Average correlations along ``axis`` in Fisher z space: ``z2r(mean(r2z(r)))``.

    Fisher z averaging applies to correlations only.  Where any value along
    ``axis`` lies outside [-1, 1] by more than 1e-12 (for example a
    co-fluctuation from a delta kernel), the values are averaged on the raw
    scale (plain mean) instead.  Values within round-off of +-1 (at most
    1e-12 above, 2.8e-14 below) are correlations of +-1 and have infinite
    z, as exact +-1 does (``_correlation_z``).  Where only one sign of infinity
    occurs along ``axis`` the result is that sign (+-1).  Where both +inf and
    -inf occur they are excluded and the finite z values are averaged (0 if
    none remain).  NaN anywhere along ``axis`` gives NaN.  No numpy warnings
    are raised.
    """
    r = np.asarray(r, dtype=np.float64)
    z, outside = _correlation_z(r)
    raw = np.any(outside, axis=axis)
    pos = np.any(np.isposinf(z), axis=axis)
    neg = np.any(np.isneginf(z), axis=axis)
    nan = np.any(np.isnan(z), axis=axis)
    finite = np.isfinite(z)
    total = np.where(finite, z, 0.0).sum(axis=axis)
    mean = z2r(_safe_divide(total, finite.sum(axis=axis).astype(np.float64)))
    out = np.where(pos & ~neg, 1.0, np.where(neg & ~pos, -1.0, mean))
    out = np.where(raw, r.mean(axis=axis), out)
    return np.where(nan, np.nan, out)


def wisfc(data, timepoint_weights, subject_weights=None, *, causal=False,
          min_samples=None):
    """
    Compute weighted inter-subject functional connectivity (ISFC).

    For each subject, the kernel-weighted correlations (see ``wcorr``) are
    computed between that subject's features and a weighted average of the
    subjects' data.  When the average is not the subject's own data, the
    cross-correlation matrix ``C`` is symmetrized by averaging the Fisher
    z-transforms of ``C`` and ``C.T`` and inverting (``z2r``) before it is
    vectorized; an entry of +-1 (up to round-off, see ``_correlation_z``)
    dominates its pair.  NaN samples carry no weight.

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
    causal : bool, optional
        True: the causal estimator (see ``wcorr``): row ``t`` of the output
        is unchanged (to 1e-9) when the samples after ``t`` are replaced by
        any values, NaN and inf included, or removed.  Every kernel must
        give no weight to later timepoints (for example a kernel function
        called with ``direction='backward'``); infinite samples and samples
        above 1e150 in magnitude are missing data; undefined (warm-up)
        timepoints are rows of NaN, not 0.  ``subject_weights``
        must be given: the default weights are computed from every
        timepoint.  Keyword only.  Default: False.
    min_samples : float or None, optional
        With ``causal=True``, a timepoint with fewer than 3 weighted
        timepoints where the subject and its reference each have a sample
        is a row of NaN.  A number (>= 1) is also the effective sample size
        ``(sum |w|)**2 / sum w**2`` a timepoint needs over those timepoints
        (rows below it are NaN; every row, when no row of the array reaches
        the number).  Keyword only.  Default: None.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        One (T, K * (K + 1) / 2) array of vectorized correlation matrices per
        subject (see ``mat2vec``); a single array if ``data`` is an array.
        Row ``t`` is estimated with the kernel ``timepoint_weights[t, :]``.
        With ``causal=True`` the undefined timepoints are rows of NaN.

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
        or reference) is treated the same way.  With ``causal=True`` the
        first of these names the undefined timepoints (returned as NaN).

    Raises
    ------
    ValueError
        If the subjects differ in shape, the kernel does not have T columns,
        every row of the kernel is zero, or a list of kernels does not have
        one kernel per subject.  With ``causal=True``: if a
        kernel has weight above its diagonal or is a delta kernel, if ``subject_weights`` is None, or if
        ``min_samples`` is not None or a finite number >= 1.  If
        ``min_samples`` is given without ``causal=True``.

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
        return wisfc([data], timepoint_weights, subject_weights=subject_weights,
                     causal=causal, min_samples=min_samples)[0]

    min_samples = _check_min_samples(min_samples, causal)
    if causal:
        if subject_weights is None:
            raise ValueError(
                "causal=True: the default subject_weights of wisfc are computed "
                "from every timepoint of the recording.  Pass subject_weights (a "
                "number, or one row of weights per subject), or use isfc."
            )
        data = [_causal_clean(d) for d in data]
    S = len(data)
    subject_weights = _subject_weight_matrix(data, subject_weights)

    kernels, group_of = _subject_kernels(timepoint_weights, S)
    stacked = np.stack(data, axis=2)
    corrs = []
    cancelled = [False] * len(kernels)
    sparse = [False] * len(kernels)
    warmup = [False] * len(kernels) if causal else None
    for s, a in enumerate(data):
        g = group_of[s]
        b = _nan_weighted_mean(stacked, subject_weights[s, :])
        if causal:
            # whether the reference is the subject's own data is read from
            # the weights; comparing the samples would read every timepoint
            own = _own_reference(subject_weights[s, :], s)
            wc, (c, few, undefined) = _wcorr(
                a, a if own else b, kernels[g], warn=False, return_cancelled=True,
                causal=True, min_samples=min_samples, autocorrelation=own,
            )
            warmup[g] = warmup[g] | undefined
            symmetrize = not own
        else:
            wc, (c, few) = _wcorr(a, b, kernels[g], warn=False, return_cancelled=True)
            symmetrize = not np.array_equal(a, b, equal_nan=True)
        cancelled[g] = cancelled[g] | c
        sparse[g] = sparse[g] | few
        if symmetrize:
            wc = _fisher_symmetrize(wc)
        corrs.append(mat2vec(wc))
        if causal:
            # mat2vec sets NaN above the diagonal to 0: undefined rows are NaN
            corrs[-1][undefined] = np.nan
        # free the (K, K, T) array before the next subject's is allocated
        del wc

    # one warning per call, covering every kernel
    message = _kernels_message(kernels, group_of, cancelled, sparse, warmup,
                               min_samples)
    if message is not None:
        _warn_user(message)
    return corrs


def _own_reference(weights, s, tol=1e-5):
    """
    True when ``_nan_weighted_mean`` with these subject weights returns
    subject ``s``'s own data: after its rules (NaN weights 0, weights that
    sum to 0 replaced by ones, magnitudes at most ``tol`` dropped) only
    subject ``s`` has weight.
    """
    w = np.array(weights, dtype=np.float64, copy=True)
    w[np.isnan(w)] = 0
    if np.sum(w) == 0:
        w = np.ones_like(w)
    keep = np.abs(w) > tol
    return bool(keep[s] and keep.sum() == 1)


def _static_connectome(d, iu, rows_per_block):
    """
    The upper triangle (``iu``) of the static correlation matrix of ``d``
    (T, K): ``wcorr`` with one uniform kernel row, or, with
    ``rows_per_block``, computed that many rows at a time (the entries agree
    with ``wcorr``'s to round-off; the temporaries stay at block size).
    """
    static = np.ones([1, d.shape[0]])
    if rows_per_block is None:
        return wcorr(d, d, static)[:, :, 0][iu]
    K = d.shape[1]
    c = np.empty((K, K))
    cancelled, sparse = None, None
    for i0, i1, R, zero_all, few_any in _corr_row_blocks(d, d, static, True,
                                                         rows_per_block):
        c[i0:i1] = R[:, :, 0]
        # the masks of the whole call, and its warning (as wcorr gives)
        cancelled = zero_all if cancelled is None else cancelled & zero_all
        sparse = few_any if sparse is None else sparse | few_any
    if cancelled is not None:
        message = _estimator_message(static, cancelled, sparse)
        if message is not None:
            _warn_user(message)
    return c[iu]


def _subject_weight_matrix(data, subject_weights, rows_per_block=None):
    """
    The S x S subject weights ``wisfc`` uses for a list of S subjects: the
    given matrix, a scalar for every other subject, or (None) the default
    weights from the correlations between the subjects' static correlation
    matrices (see ``wisfc``).  ``rows_per_block``: compute the static
    correlation matrices that many rows at a time (the streamed summaries
    pass their block size, so the weights fit their memory budget).
    """
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
            connectomes = np.stack(
                [_static_connectome(d, iu, rows_per_block) for d in data], axis=1
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

    return subject_weights


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
    two_d = [isinstance(w, LagKernel) or np.ndim(w) == 2 for w in timepoint_weights]
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
            lazy = isinstance(w, LagKernel) or isinstance(k, LagKernel)
            if w is k or (w.same_as(k) if isinstance(w, LagKernel) else (
                not lazy and np.shape(w) == np.shape(k)
                and np.array_equal(w, k, equal_nan=True)
            )):
                break
        else:
            g = len(kernels)
            kernels.append(w)
        group_of.append(g)
    return kernels, group_of


def _kernels_message(kernels, group_of, cancelled, sparse=None, warmup=None,
                     min_samples=None):
    """
    The one warning of a correlation call with one or several kernels, or None.
    With one kernel it is ``_estimator_message``; with several, each kernel's
    message is prefixed by the subjects that use it.  ``cancelled`` and ``sparse``
    hold one mask (see ``_wcorr_blocks``) per kernel, as does ``warmup`` (the
    causal estimator's undefined rows, see ``_estimator_message``).
    """
    if sparse is None:
        sparse = [None] * len(kernels)
    if warmup is None:
        warmup = [None] * len(kernels)
    if len(kernels) == 1:
        return _estimator_message(kernels[0], cancelled[0], sparse[0], warmup[0],
                                  min_samples)
    parts = []
    for g, kernel in enumerate(kernels):
        message = _estimator_message(kernel, cancelled[g], sparse[g], warmup[g],
                                     min_samples)
        if message is not None:
            subjects = [s for s, k in enumerate(group_of) if k == g]
            label = "Subject" if len(subjects) == 1 else "Subjects"
            parts.append(f"{label} {', '.join(map(str, subjects))}: {message}")
    return " ".join(parts) if parts else None


def isfc(data, timepoint_weights, *, causal=False, min_samples=None):
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
    causal : bool, optional
        True: the causal estimator (see ``wcorr``): row ``t`` of the output
        is unchanged (to 1e-9) when the samples after ``t`` are replaced by
        any values, NaN and inf included, or removed.  Every kernel must
        give no weight to later timepoints (for example a kernel function
        called with ``direction='backward'``); infinite samples and samples
        above 1e150 in magnitude are missing data; undefined (warm-up)
        timepoints are rows of NaN, not 0.  Keyword only.  Default: False.
    min_samples : float or None, optional
        With ``causal=True``, a timepoint with fewer than 3 weighted
        timepoints where the subject and its reference each have a sample
        is a row of NaN.  A number (>= 1) is also the effective sample size
        ``(sum |w|)**2 / sum w**2`` a timepoint needs over those timepoints
        (rows below it are NaN; every row, when no row of the array reaches
        the number).  Keyword only.  Default: None.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        List of vectorized correlation matrices (one (T, K * (K + 1) / 2)
        array per subject, see ``mat2vec``), each showing correlations between
        that subject and the average of all others; a single array if
        ``data`` is an array.  Row ``t`` is estimated with the kernel
        ``timepoint_weights[t, :]``.  With ``causal=True`` the undefined
        timepoints are rows of NaN.

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
        one kernel per subject.  With ``causal=True``: if a
        kernel has weight above its diagonal or is a delta kernel, or if
        ``min_samples`` is not None or a finite number >= 1.  If
        ``min_samples`` is given without ``causal=True``.

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
        return isfc([data], timepoint_weights, causal=causal,
                    min_samples=min_samples)[0]

    return wisfc(data, timepoint_weights, subject_weights=1 - np.eye(len(data)),
                 causal=causal, min_samples=min_samples)


def autofc(data, timepoint_weights, *, causal=False, min_samples=None):
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
    causal : bool, optional
        True: the causal estimator (see ``wcorr``): row ``t`` of the output
        is unchanged (to 1e-9) when the samples after ``t`` are replaced by
        any values, NaN and inf included, or removed.  Every kernel must
        give no weight to later timepoints (for example a kernel function
        called with ``direction='backward'``); infinite samples and samples
        above 1e150 in magnitude are missing data; undefined (warm-up)
        timepoints are rows of NaN, not 0.  Keyword only.  Default: False.
    min_samples : float or None, optional
        With ``causal=True``, a timepoint with fewer than 3 weighted
        timepoints where the subject and its reference each have a sample
        is a row of NaN.  A number (>= 1) is also the effective sample size
        ``(sum |w|)**2 / sum w**2`` a timepoint needs over those timepoints
        (rows below it are NaN; every row, when no row of the array reaches
        the number).  Keyword only.  Default: None.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        List of vectorized correlation matrices (one (T, K * (K + 1) / 2)
        array per subject, see ``mat2vec``), each showing within-subject
        correlations; a single array if ``data`` is an array.  Row ``t`` is
        estimated with the kernel ``timepoint_weights[t, :]``.  With
        ``causal=True`` the undefined timepoints are rows of NaN.

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
        one kernel per subject.  With ``causal=True``: if a
        kernel has weight above its diagonal or is a delta kernel, or if
        ``min_samples`` is not None or a finite number >= 1.  If
        ``min_samples`` is given without ``causal=True``.

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
        return autofc([data], timepoint_weights, causal=causal,
                      min_samples=min_samples)[0]

    return wisfc(data, timepoint_weights, subject_weights=np.eye(len(data)),
                 causal=causal, min_samples=min_samples)


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
    # index the timepoint axis (np.squeeze would also drop K = 1 axes)
    return np.stack([f(corrs[:, :, t]) for t in range(corrs.shape[2])], axis=0)


def corrmean_combine(corrs):
    """
    Compute the mean element-wise correlation across each matrix in a list.

    Correlations are averaged in Fisher z space (``z2r(mean(r2z(r)))``).  An
    entry of +-1 has infinite z (values within round-off of +-1, at most
    1e-12 above or 2.8e-14 below, count as +-1, so round-off does not make
    the z finite; 1 - 1e-13 keeps its finite z): where only one sign occurs across
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


def _round_off_spread(stacked):
    """
    Entries of a stack of matrices (along the last axis) whose values are
    one value up to round-off: their range is at most ``_UNIT_TOLERANCE``
    (128 eps, 2.8e-14), relative to the largest magnitude when that is
    above 1 (correlations are computed to an absolute accuracy, raw
    co-fluctuations to a relative one).

    The same correlation computed in two orders (``isfc`` of two subjects:
    subject 0 with subject 1, and subject 1 with subject 0) differs by at
    most 61 eps in the cases measured (33.8 million entries: five kinds of
    data, T 100 to 700, 10 to 40 features, five kernels, with and without
    ``causal=True``; 2.46 million of them differ at all, 99.7% by less
    than 2 eps), except in the first defined row of the causal estimator,
    which has 3 samples (13 entries above 128 eps, up to 3108 eps).  Correlations of subjects
    that differ are far apart on that scale: 3000 random stacks of 3 to 6
    subjects have no range below 1e-9.  NaN and infinite entries are not
    selected.
    """
    with np.errstate(invalid="ignore"):
        spread = stacked.max(axis=-1) - stacked.min(axis=-1)
        scale = np.maximum(1.0, np.abs(stacked).max(axis=-1))
        return spread <= _UNIT_TOLERANCE * scale


def tstat_combine(corrs, return_pvals=False):
    """
    Element-wise one-sample t-tests (mean 0) across the matrices in a list.

    Each entry is Fisher z-transformed (``r2z``) and tested across the list
    with ``scipy.stats.ttest_1samp``.  Fisher z applies to correlations only:
    an entry where any matrix has a value outside [-1, 1] (for example the
    co-fluctuations of a delta kernel, ``eye_weights``) is tested on the raw
    values, with no NaN and no warning; values within round-off of +-1 (at
    most 1e-12 above, 2.8e-14 below) are correlations of +-1 (infinite z).  An entry with the same
    value in every matrix (zero variance across subjects, for example the
    diagonal of ``autofc`` results, where every value is 1) has t = +inf or
    -inf with the sign of that value (p = 0), or t = 0 (p = 1) when the value
    is 0, with no warning.  The same holds where the values differ by
    round-off only: a range across the matrices of at most 128 eps (2.8e-14;
    times the largest magnitude for values above 1) counts as none, and the
    entry has t = +inf, -inf or 0 with the sign of the mean of its values.
    That is the case of ``isfc`` with two subjects, whose correlations are
    the same numbers computed in two orders (they differed by up to 61 eps
    in the cases measured; in the first defined row of ``causal=True``,
    which has 3 samples, 13 of 33.8 million entries differed by more than
    128 eps, up to 3108 eps, and keep their finite statistic).  Where the values of the matrices
    nearly coincide beyond that (1e-13 is a difference)
    the statistic is ill-conditioned: round-off of the correlations is
    divided by a spread close to zero (with ``causal=True`` and a kernel
    function of your own such entries agree to about 1e-9 of their
    magnitude, not to 1e-9, when later rows are removed; with the built-in
    kernel functions removal changes no bit; see ``timecorr``).

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
        z, outside = _correlation_z(stacked)
        raw = np.any(outside, axis=2, keepdims=True)
        vals = np.where(raw, stacked, z)
        # zero variance across subjects (every value equal, infinities
        # included, or equal up to round-off): t is +-inf with the sign of
        # the mean, 0 for a mean of 0
        constant = np.all(vals == vals[:, :, :1], axis=2) | _round_off_spread(stacked)
        filler = np.arange(vals.shape[2], dtype=np.float64)  # any non-constant row
        ts, ps = ttest(np.where(constant[:, :, None], filler, vals), popmean=0, axis=2)
        with np.errstate(invalid="ignore"):  # +inf and -inf: NaN, as before
            mean = np.where(constant[:, :, None], vals, 0.0).mean(axis=2)
        signed_inf = np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0))
        ts = np.where(constant, signed_inf, ts)
        ps = np.where(constant, np.where(mean == 0, 1.0, 0.0), ps)

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
_GRAPH_MEASURES = (
    "eigenvector_centrality", "pagerank_centrality", "pagerank_centrality_abs",
    "strength",
)
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


def _pagerank_solution_ok(B, b, r):
    """True when ``r`` is finite, has a usable sum and solves ``B r = b``."""
    if not np.all(np.isfinite(r)):
        return False
    if abs(np.sum(r)) < _PAGERANK_SUM_TOL * max(1.0, np.max(np.abs(r))):
        return False
    residual = np.linalg.norm(B @ r - b) / np.linalg.norm(b)
    return bool(residual <= _PAGERANK_RESIDUAL_TOL)


def _pagerank_solve(B, b):
    """
    Solve the PageRank system with ``scipy.linalg.solve``, or return None.

    Returns the unnormalized solution when the system is well-posed (see
    ``_PAGERANK_RCOND``), otherwise None.  The solution is computed exactly as
    at commit 863a412, so well-posed outputs are unchanged bit for bit; the
    condition estimate comes from a separate LU factorization (``getrf``,
    ``gecon``), with no singular value decomposition.
    """
    getrf, gecon = linalg.get_lapack_funcs(("getrf", "gecon"), (B,))
    lu, _, info = getrf(B)
    if info > 0:
        return None
    rcond, _ = gecon(lu, np.max(np.sum(np.abs(B), axis=0)), norm="1")
    if not rcond >= _PAGERANK_RCOND:
        return None
    try:
        with warnings.catch_warnings():
            # an ill-conditioned warning is superseded by the checks here
            warnings.simplefilter("ignore", linalg.LinAlgWarning)
            r = linalg.solve(B, b)
    except np.linalg.LinAlgError:
        return None
    if not _pagerank_solution_ok(B, b, r):
        return None
    return r


def _model_reduce(corrs, rfun, ndims, return_model=False):
    """
    The model-based reduction of ``reduce`` (``rfun`` a model name or dict,
    ``ndims`` checked).  ``return_model=True`` returns ``(reduced, fitted)``,
    ``fitted`` a ``_FittedModel`` that reduces other rows with the model
    fitted here.
    """
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
    model = None
    if return_model:
        red_corrs, model = hyp.reduce(corrs, reduce=rfun, ndims=n_fit,
                                      random_state=0, return_model=True)
    else:
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

    if return_model:
        return red_corrs, _FittedModel(model, n_features, max(0, ndims - n_fit))
    return red_corrs


class _FitUnusable(ValueError):
    """
    A reducer could not be fitted to the training rows of a causal fit, or
    its fit cannot be used on other rows (no defined row, a clustering
    without a community of 2 members, a model that does not reproduce its
    rows).  A ``ValueError`` for the caller; a refit of ``features`` that
    raises it keeps the previous fit (``refit_every``).
    """


# the hypertools model names whose fitted estimator has ``components_`` (one
# vector per output column): the models a refit can match to the fit before
# (``features(..., refit_every=...)``)
_COMPONENT_MODELS = (
    "PCA", "IncrementalPCA", "SparsePCA", "MiniBatchSparsePCA", "FastICA",
    "FactorAnalysis", "TruncatedSVD", "DictionaryLearning",
    "MiniBatchDictionaryLearning",
)


# an output column of a fitted model is constant on its training rows when
# its range there is at most this fraction of its largest magnitude, or
# _FROZEN_ZERO_RTOL of the largest magnitude of the whole output (the
# tolerances of summaries._Summary._warn_constant)
_FROZEN_RTOL = 1e-9
_FROZEN_ZERO_RTOL = 1e-12


class _FittedModel:
    """
    A hypertools model fitted by ``_model_reduce``: ``model`` (the fitted
    ``hypertools`` reducer; None when hypertools fitted none: it returns
    data with no more columns than the components asked for as they are,
    and ``apply`` is then the identity), the number of input columns
    ``n_features`` and the number
    of zero columns ``pad`` that ``reduce`` appends when the data supplied
    fewer components than requested.  ``apply`` reduces other rows with it.

    ``freeze_constants(outputs)`` (the model's outputs for its training
    rows) records the output columns that are constant there up to
    round-off: a range of at most ``_FROZEN_RTOL`` of the column's largest
    magnitude, or ``_FROZEN_ZERO_RTOL`` of the largest magnitude of the
    whole output (a component along which the training rows do not vary,
    for example: its values are round-off of the others).  ``apply`` then
    returns a value of such a column that lies within that tolerance of
    the column's training mean as exactly that mean; other values are
    returned as computed.  The rule reads the training rows and the row it
    is applied to, nothing else, so it is causal; without it the next
    order would correlate that round-off.
    """

    def __init__(self, model, n_features, pad=0):
        # the model as a pickle restores it: a fitted model's arrays can be
        # laid out differently in memory after pickling, which changes the
        # round-off of its products, so a model and its unpickled copy
        # would differ in the last bit
        try:
            model = pickle.loads(pickle.dumps(model))
        except Exception:  # a model that cannot be pickled is used as it is
            pass
        self.model = model
        self.n_features = int(n_features)
        self.pad = int(pad)
        self.constants = None  # (columns, their training means, tolerances)

    @property
    def estimator(self):
        """The fitted scikit-learn model (None for the identity)."""
        return getattr(self.model, "model_", self.model)

    @property
    def components(self):
        """``components_`` of the fitted model, one vector per output
        column, as a 2-D array; None when it has none."""
        components = getattr(self.estimator, "components_", None)
        if components is None:
            return None
        components = np.asarray(components, dtype=np.float64)
        if components.ndim != 2 or components.shape[1] != self.n_features:
            return None
        return components

    def freeze_constants(self, outputs):
        stacked = np.vstack([np.asarray(o, dtype=np.float64) for o in outputs])
        if not stacked.shape[0]:
            return
        largest = np.max(np.abs(stacked), axis=0)
        tol = np.maximum(_FROZEN_RTOL * largest,
                         _FROZEN_ZERO_RTOL * np.max(largest, initial=0.0))
        columns = np.flatnonzero(np.ptp(stacked, axis=0) <= tol)
        if columns.size:
            self.constants = (columns, np.mean(stacked[:, columns], axis=0),
                              tol[columns])

    def apply(self, arrays):
        """The rows of a list of (rows, n_features) arrays, reduced with the
        fitted model (a list of arrays)."""
        arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
        for a in arrays:
            if a.ndim != 2 or a.shape[1] != self.n_features:
                raise ValueError(
                    f"the fitted model takes {self.n_features} columns; got an "
                    f"array of shape {a.shape}."
                )
        if self.model is None:
            out = [a.copy() for a in arrays]  # the identity
        else:
            out = hyp.reduce(arrays, reduce=self.model)
            out = out if isinstance(out, list) else [out]
            out = [np.asarray(o, dtype=np.float64) for o in out]
        if self.pad:
            out = [np.hstack((o, np.zeros((o.shape[0], self.pad)))) for o in out]
        if self.constants is not None:
            columns, means, tol = self.constants
            for o in out:
                values = o[:, columns]
                o[:, columns] = np.where(np.abs(values - means) <= tol, means, values)
        return out


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
          ``'pagerank_centrality'``, ``'pagerank_centrality_abs'``, or
          ``'strength'`` (strings only), computed
          per node for each timepoint from the K x K matrix (diagonal
          included).  All but ``'pagerank_centrality_abs'`` use signed
          weights as they are, without taking absolute values: ``'strength'``
          is the row sum, so negative
          correlations subtract; ``'eigenvector_centrality'`` is the absolute
          value of the unit eigenvector of the largest (most positive)
          eigenvalue of the symmetrized matrix; ``'pagerank_centrality'``
          (damping 0.85, uniform prior) divides each column by its column
          sum (a zero sum is replaced by 1; negative weights can make a
          column sum small or zero) and normalizes the result to sum to 1.
          PageRank on signed correlations is not always defined: for some
          valid correlation matrices the system
          ``(I - 0.85 W diag(1 / column sums)) r = b`` is singular.  A
          timepoint is ill-posed when the solve fails, the solution is not
          finite, its sum is below 1e-12 times ``max(1, max|r|)``, its
          relative residual ``||B r - b|| / ||b||`` exceeds 1e-8, or the
          LAPACK estimate of the reciprocal condition number of the system
          (1-norm) is below 1e-14.  Such a timepoint gets the minimum-norm
          least-squares solution (``numpy.linalg.lstsq`` with relative
          singular-value cutoff 1e-14), normalized, when it passes the same
          residual and sum tests, and otherwise the uniform vector 1/K; one
          ``UserWarning`` per call gives the counts.  Other timepoints are
          computed as before (``scipy.linalg.solve``).
          ``'pagerank_centrality_abs'`` is PageRank on the non-negative
          weights ``|W|`` (damping 0.85, uniform prior, diagonal included;
          a node whose weights are all zero spreads its rank uniformly).  It
          is well-posed for every finite matrix (the system is strictly
          diagonally dominant by columns, with 1-norm condition number at
          most ``(1 + 0.85) / (1 - 0.85)``, about 12.3), so it never falls back or
          warns; a timepoint with a non-finite entry gives NaNs.  It equals
          ``'pagerank_centrality'`` up to round-off when every weight is
          non-negative, and
          ignores the sign of the correlations otherwise.  A delta-kernel co-fluctuation matrix
          ``z z^T`` has rank one, so on the delta path of ``features`` with
          ``delta_weights_function=eye_weights`` eigenvector centrality equals ``|z| / ||z||`` and the graph measures
          carry no network structure.  With one node (K = 1, the 1 x 1
          matrix ``[[w]]``), ``'strength'`` is ``w`` and, for a finite ``w``,
          the other three measures are 1 (for a non-finite ``w``,
          ``'pagerank_centrality_abs'`` is NaN and
          ``'eigenvector_centrality'`` and ``'pagerank_centrality'`` raise
          an error, as they do for any matrix with a non-finite entry).
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
        ``corrs`` (a summary returns a list for a tuple of arrays): (T, ndims)
        for model-based reduction, (T, K) for a graph measure, (T, c) for a
        summary, whatever the callable returns otherwise.

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
        number of features or samples), and once per call when the PageRank
        system of ``'pagerank_centrality'`` is ill-posed for any timepoint
        (with the numbers of ill-posed timepoints, of least-squares
        solutions and of uniform vectors).

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

    pagerank_counts = {"total": 0, "lstsq": 0, "uniform": 0}

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
            PageRank centrality values.  For an ill-posed system (see
            ``_PAGERANK_RCOND``) the normalized minimum-norm least-squares
            solution when it solves the system, otherwise the uniform vector;
            ``pagerank_counts`` records which.
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
        pagerank_counts["total"] += 1
        r = _pagerank_solve(B, b)
        if r is None:
            r = np.linalg.lstsq(B, b, rcond=_PAGERANK_RCOND)[0]
            if _pagerank_solution_ok(B, b, r):
                pagerank_counts["lstsq"] += 1
            else:
                pagerank_counts["uniform"] += 1
                return np.full(N, 1.0 / N)
        r /= np.sum(r)
        return r

    def pagerank_centrality_abs(W, d=0.85):
        """
        PageRank centrality on the non-negative weights ``|W|``.

        Uniform prior; a node whose weights are all zero spreads its rank
        uniformly (equivalently, the solution of the linear system below is
        normalized to sum to 1).  Each column of ``|W|`` is divided by its
        largest entry first (PageRank depends only on the relative weights
        within each column), so the column sums, between 1 and K, neither
        overflow nor underflow, and a column of small weights is not lost
        next to a column of large ones.  With column sums
        ``s_j`` of the scaled weights (zero sums replaced by 1), the matrix
        ``B = I - d |W| diag(1 / s)`` is strictly diagonally dominant by
        columns for ``d < 1``, so the system ``B r = (1 - d) / K`` always has
        a unique, positive solution and its 1-norm condition number is at
        most ``(1 + d) / (1 - d)``.  Returns NaNs when ``W`` has a non-finite
        entry, and an empty array for K = 0.
        """
        N = len(W)
        if N == 0:
            return np.zeros(0)
        A = np.abs(W)
        if not np.all(np.isfinite(A)):
            return np.full(N, np.nan)
        top = np.max(A, axis=0)
        A = A / np.where(top > 0, top, 1.0)[None, :]
        deg = np.sum(A, axis=0)
        deg[deg == 0] = 1
        B = np.eye(N) - d * (A / deg[None, :])
        r = linalg.solve(B, np.full(N, (1 - d) / N))
        return r / np.sum(r)

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
        "pagerank_centrality_abs": lambda x: pagerank_centrality_abs(x, d=0.85),
        "strength": strengths_und,
    }

    ndims = _check_ndims(ndims)
    if rfun is None:
        return corrs

    if ndims is not None and not _is_model(rfun):
        raise ValueError(
            f"ndims={ndims} applies to model-based reduction only; the graph "
            "measure, summary or callable rfun defines its own output width.  "
            "Leave ndims as None."
        )

    # imported here: timecorr.summaries imports this module
    from .summaries import _Summary, _from_name

    if isinstance(rfun, str) and rfun in _SUMMARY_NAMES:
        rfun = _from_name(rfun)

    if isinstance(rfun, _Summary):
        # a summary takes a list or a single array and returns the same kind
        return rfun(corrs)

    if callable(rfun):
        if isinstance(corrs, list):
            return rfun(corrs)
        return rfun([corrs])[0]

    if isinstance(rfun, str) and rfun in graph_measures:
        out = apply_by_row(corrs, graph_measures[rfun])
        n_lstsq, n_uniform = pagerank_counts["lstsq"], pagerank_counts["uniform"]
        if n_lstsq + n_uniform:
            warnings.warn(
                "rfun='pagerank_centrality': the PageRank system "
                "(I - 0.85 W diag(1 / column sums)) r = b was ill-posed "
                f"(singular, inconsistent or not solved accurately) for "
                f"{n_lstsq + n_uniform} of {pagerank_counts['total']} timepoints; "
                f"{n_lstsq} used the minimum-norm least-squares solution and "
                f"{n_uniform} the uniform vector 1/K.  PageRank on signed "
                "correlations is not always defined (negative weights can "
                "cancel the column sums); rfun='pagerank_centrality_abs' "
                "(PageRank on |W|), rfun='strength' and "
                "rfun='eigenvector_centrality' are defined for every matrix.",
                UserWarning,
                stacklevel=2,
            )
        return out

    return _model_reduce(corrs, rfun, ndims)


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
