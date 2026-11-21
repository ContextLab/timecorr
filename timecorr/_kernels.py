"""
Kernels whose rows are computed on demand, a chunk of rows at a time.

The built-in kernel functions (``gaussian_weights``, ``laplace_weights``,
``t_weights``, ``mexican_hat_weights``, ``uniform_weights``,
``boxcar_weights``, ``eye_weights``) give each pair of timepoints a weight
that depends only on their lag ``t - tau``.  Each is defined here by its lag
profile: the weight at every absolute lag ``0, 1, ..., T - 1``, computed with
the same elementwise operations as the T x T matrix the kernel function
returns, so a weight gathered from the profile equals the matrix entry
exactly.  :class:`LagKernel` holds the profile of length T, the restrictions
``timecorr`` applies (``include_timepoints``, ``include_zero``,
``exclude_timepoints``) and the event blocks (``events.mask_kernel``), and
computes any rectangle of the T x T kernel without building the matrix.  The
estimator (``helpers._KernelRows``) reads it in chunks of rows, each over the
span of columns where the chunk can have non-zero weight, so its memory grows
linearly in T.
"""

import math

import numpy as np
from scipy.special import gamma

# Rows per chunk: the estimator, the smoothing of ``timecorr`` and the
# row statistics read every kernel in chunks of this many rows.
CHUNK = 128

# ----------------------------------------------------------------------------
# lag profiles: weights at the absolute lags ``d`` (an integer array), with the
# operations of the kernel functions in ``helpers``
# ----------------------------------------------------------------------------


def _as_lags(d):
    """Integer lags as int64 (``d**2`` of int32 lags overflows from 46,341)."""
    d = np.asarray(d)
    return d.astype(np.int64) if d.dtype.kind in "iu" else d


def gaussian_profile(d, params):
    d = _as_lags(d)
    c1 = np.divide(1, np.sqrt(2 * math.pi * params["var"]))
    c2 = np.divide(-1, 2 * params["var"])
    return c1 * np.exp(c2 * d**2)


def laplace_profile(d, params):
    d = _as_lags(d)
    return np.multiply(
        np.divide(1, 2 * params["scale"]), np.exp(-np.divide(d, params["scale"]))
    )


def eye_profile(d, params):
    d = _as_lags(d)
    return np.where(d == 0, 1.0, 0.0)


def uniform_profile(d, params):
    return np.ones(np.shape(d))


def t_profile(d, params):
    d = _as_lags(d)
    c1 = np.divide(
        gamma((params["df"] + 1) / 2),
        np.sqrt(params["df"] * math.pi) * gamma(params["df"] / 2),
    )
    c2 = np.divide(-(params["df"] + 1), 2)
    return np.multiply(c1, np.power(1 + np.divide(d**2, params["df"]), c2))


def mexican_hat_profile(d, params):
    d = _as_lags(d)
    a = np.divide(2, np.sqrt(3 * params["sigma"]) * np.power(math.pi, 0.25))
    b = 1 - np.power(np.divide(d, params["sigma"]), 2)
    c = np.exp(-np.divide(d**2, 2 * np.power(params["sigma"], 2)))
    return np.multiply(a, np.multiply(b, c))


def boxcar_profile(d, params):
    return np.multiply(d < params["width"] / 2.0, 1.0)


def lags(T):
    """The T x T matrix of absolute lags ``|t - tau|`` (int64)."""
    idx = np.arange(int(T), dtype=np.int64)
    return np.abs(np.subtract.outer(idx, idx))


def check_exclude(exclude):
    """
    ``exclude_timepoints`` as a float rounded to an integer, or None; a
    non-finite value (inf or NaN) is rejected, for every kernel function.
    """
    if exclude is None:
        return None
    x = float(exclude)
    if not math.isfinite(x):
        raise ValueError(
            f"exclude_timepoints must be a finite number or None, not {exclude!r}"
        )
    return float(np.round(x))


def _canonical_labels(labels, T):
    """
    Block labels renumbered 0, 1, ... along time: the blocks are the runs of
    consecutive equal labels (a label that returns after another starts a
    new block, as the boundaries of ``events.block_labels`` give).  None for
    a single block (no restriction).
    """
    labels = np.asarray(labels)
    if labels.shape != (T,):
        raise ValueError(f"labels must have length T={T}, not shape {labels.shape}")
    change = labels[1:] != labels[:-1]
    if not change.any():
        return None
    return np.concatenate([[0], np.cumsum(change)]).astype(np.int64)


# ----------------------------------------------------------------------------
# the kernel
# ----------------------------------------------------------------------------


class LagKernel:
    """
    A T x T kernel ``w[t, tau] = profile[|t - tau|]``, restricted as
    ``timecorr`` restricts a kernel, computed a rectangle at a time.

    Parameters
    ----------
    profile : numpy.ndarray
        The weights at the absolute lags ``0, ..., T - 1`` (length T).
    include : {'all', 'backward', 'forward'}
        ``include_timepoints``: 'backward' keeps ``tau <= t``, 'forward'
        ``tau >= t`` (as ``helpers.one_sided``).
    include_zero : bool
        False gives the diagonal weight 0.
    exclude : int, float or None
        ``exclude_timepoints``: ``x >= 0`` keeps the lags ``|t - tau| <= x``,
        ``-x`` the lags ``> x`` (rounded to an integer).
    labels : numpy.ndarray or None
        Event block of each timepoint (``events.block_labels``): weights
        between timepoints in different blocks are 0.  The blocks are the
        runs of consecutive equal labels (a label that returns after another
        starts a new block); labels with a single block are stored as None.

    The values equal those of the T x T matrix that ``timecorr`` builds from
    the kernel function and the same restrictions, entry for entry.
    """

    def __init__(self, profile, include="all", include_zero=True, exclude=None,
                 labels=None):
        self.profile = np.asarray(profile, dtype=np.float64)
        self.T = int(self.profile.shape[0])
        self.shape = (self.T, self.T)
        if include not in ("all", "backward", "forward"):
            raise ValueError(f"invalid include {include!r}")
        self.include = include
        self.include_zero = bool(include_zero)
        self.exclude = check_exclude(exclude)
        self.labels = None if labels is None else _canonical_labels(labels, self.T)
        self._lag_range()
        if self.labels is not None:
            # first and one-past-last timepoint of each timepoint's block
            change = np.flatnonzero(np.diff(self.labels)) + 1
            bounds = np.concatenate([[0], change, [self.T]])
            block = np.searchsorted(bounds, np.arange(self.T), side="right") - 1
            self._first = bounds[block]
            self._end = bounds[block + 1]

    # -- the signed lags t - tau that can have non-zero weight -------------
    def _lag_range(self):
        p = self.profile
        finite = bool(np.all(np.isfinite(p)))
        self._finite = finite
        self._symmetric = None
        nonzero = np.flatnonzero(p != 0)  # NaN counts as non-zero
        L = int(nonzero[-1]) if nonzero.size else -1
        if not finite:
            # 0 * inf and 0 * NaN are NaN in the matrix: every lag can hold one
            L = self.T - 1
        lo, hi = -L, L
        if self.include == "backward":
            lo = 0 if self.include_zero else 1
        elif self.include == "forward":
            hi = 0 if self.include_zero else -1
        if self.exclude is not None and self.exclude >= 0 and finite:
            k = int(min(abs(self.exclude), self.T - 1))
            lo, hi = max(lo, -k), min(hi, k)
        if L < 0 or lo > hi:
            lo, hi = 1, 0  # empty
        self.lo, self.hi = lo, hi

    @property
    def empty(self):
        return self.lo > self.hi

    def with_events(self, labels):
        """
        A copy confined to the blocks of ``labels`` (None: no further
        restriction) and of the events it already has: two timepoints keep
        their weight only when they share a block of both, as
        ``events.mask_kernel`` applied twice to the T x T matrix.
        """
        if labels is None:
            labels = self.labels
        elif self.labels is not None:
            new = _canonical_labels(labels, self.T)
            if new is not None:
                change = (np.diff(new) != 0) | (np.diff(self.labels) != 0)
                labels = np.concatenate([[0], np.cumsum(change)])
            else:
                labels = self.labels
        return LagKernel(self.profile, self.include, self.include_zero, self.exclude,
                         labels)

    def same_as(self, other):
        """True when ``other`` is a LagKernel with the same weights."""
        if other is self:
            return True
        if not isinstance(other, LagKernel):
            return False
        if (self.T, self.include, self.include_zero, self.exclude) != (
            other.T, other.include, other.include_zero, other.exclude
        ):
            return False
        if (self.labels is None) != (other.labels is None):
            return False
        if self.labels is not None and not np.array_equal(self.labels, other.labels):
            return False
        return bool(np.array_equal(self.profile, other.profile, equal_nan=True))

    # -- rows ---------------------------------------------------------------
    def span(self, t0, t1):
        """
        ``(c0, c1)``: the columns where rows ``t0:t1`` can have non-zero
        weight lie in ``c0:c1`` (``c0 == c1`` when they have none).
        """
        if self.empty or t1 <= t0:
            return t0, t0
        c0 = max(0, t0 - self.hi)
        c1 = min(self.T, t1 - self.lo)
        if self.labels is not None:
            c0 = max(c0, int(self._first[t0]))
            c1 = min(c1, int(self._end[t1 - 1]))
        if c1 <= c0:
            return c0, c0
        return c0, c1

    def key(self, t0, t1):
        """
        A key shared by the chunks of rows whose ``block(t0, t1, *span)``
        are equal (the chunks that reach neither an edge of the kernel nor an
        event boundary), or None.
        """
        if self.empty:
            return None
        lo_col, hi_col = t0 - self.hi, t1 - self.lo  # columns lo_col:hi_col
        if lo_col < 0 or hi_col > self.T:
            return None
        if self.labels is not None:
            first = min(t0, lo_col)
            last = max(t1, hi_col) - 1
            if self.labels[first] != self.labels[last]:
                return None
        return ("interior", t1 - t0)

    def block(self, t0, t1, c0, c1):
        """
        Rows ``t0:t1``, columns ``c0:c1`` of the kernel (a new float array).

        Row t is a slice of the weights at the signed lags, copied from a
        strided view; then the direction and the diagonal are set to 0, the
        ``exclude`` band multiplies the weights (as the 0/1 matrix of
        ``timecorr``'s ``exclude_timepoints`` does: ``0 * inf`` is NaN), and
        weights between event blocks are set to 0.
        """
        n, m = t1 - t0, c1 - c0
        if n <= 0 or m <= 0:
            return np.zeros((max(n, 0), max(m, 0)))
        T = self.T
        if self._symmetric is None:
            # weights at the signed lags T - 1, ..., -(T - 1) (symmetric)
            self._symmetric = self.profile[np.abs(np.arange(2 * T - 1) - (T - 1))]
        # row t holds entries T - 1 - t + c0 ... T - 1 - t + c1 - 1
        windows = np.lib.stride_tricks.sliding_window_view(self._symmetric, m)
        w = np.array(windows[T - t1 + c0 : T - t0 + c0][::-1])
        k = t0 - c0  # entry (r, j) is at the lag t - tau = r - j + k
        if self.include == "all":
            if not self.include_zero:
                r = np.arange(max(0, -k), min(n, m - k))
                w[r, r + k] = 0
        elif self.include == "backward":
            # keep t - tau >= 0 (> 0 without the diagonal): j <= r + k (- 1)
            w[~np.tri(n, m, k - (0 if self.include_zero else 1), dtype=bool)] = 0
        else:
            # keep t - tau <= 0 (< 0 without the diagonal): j >= r + k (+ 1)
            w[np.tri(n, m, k - (1 if self.include_zero else 0), dtype=bool)] = 0
        if self.exclude is not None:
            x = int(min(abs(self.exclude), T - 1))
            # |t - tau| <= x: r + k - x <= j <= r + k + x
            band = np.tri(n, m, k + x, dtype=bool)
            band &= ~np.tri(n, m, k - x - 1, dtype=bool)
            if self.exclude < 0:
                np.logical_not(band, out=band)
            np.multiply(w, band, out=w)
            del band
        if self.labels is not None and (
            self.labels[min(t0, c0)] != self.labels[max(t1, c1) - 1]
        ):
            other = self.labels[t0:t1, np.newaxis] != self.labels[np.newaxis, c0:c1]
            w[other] = 0
        return w

    def dense(self):
        """The T x T matrix."""
        return self.block(0, self.T, 0, self.T)

    def chunks(self, size):
        """``(t0, t1, c0, c1, key)`` for consecutive chunks of ``size`` rows."""
        for t0 in range(0, self.T, size):
            t1 = min(self.T, t0 + size)
            c0, c1 = self.span(t0, t1)
            yield t0, t1, c0, c1, self.key(t0, t1)

    def max_row_support(self, size):
        """The largest number of non-zero weights (NaN included) in a row."""
        best = 0
        for _, _, _, _, _, w in blocks(self, size):
            best = max(best, int(np.count_nonzero(w, axis=1).max(initial=0)))
        return best


class ArrayKernel:
    """
    A kernel given as an array (a kernel function of the user's, or a
    precomputed matrix), with the interface of :class:`LagKernel` that the
    estimator reads: ``shape``, ``chunks`` (every chunk spans all columns)
    and ``block`` (a read-only view; copy it before writing).
    """

    def __init__(self, weights):
        self.W = np.asarray(weights, dtype=np.float64)
        self.shape = self.W.shape

    def chunks(self, size):
        """``(t0, t1, 0, T, None)`` for consecutive chunks of ``size`` rows."""
        if self.W.ndim != 2:
            return
        T_out, T = self.shape
        for t0 in range(0, T_out, size):
            yield t0, min(T_out, t0 + size), 0, T, None

    def block(self, t0, t1, c0, c1):
        """Rows ``t0:t1``, columns ``c0:c1`` (a read-only view)."""
        view = self.W[t0:t1, c0:c1].view()
        view.flags.writeable = False
        return view


def as_kernel(weights):
    """
    ``weights`` with the kernel interface (``shape``, ``chunks``, ``block``):
    itself when it has it (a ``LagKernel``, an ``ArrayKernel``), otherwise
    the array in an ``ArrayKernel``.
    """
    if hasattr(weights, "chunks") and hasattr(weights, "block"):
        return weights
    return ArrayKernel(weights)


def blocks(kernel, size=CHUNK, transform=None, full=False):
    """
    ``(t0, t1, c0, c1, key, w)`` for consecutive chunks of ``size`` rows of
    ``kernel`` (a ``LagKernel`` or an ``ArrayKernel``): ``w`` is
    ``transform(kernel.block(t0, t1, c0, c1))`` (the block itself without
    ``transform``; do not modify it), computed once for the chunks that
    share a ``key``.  ``full=True`` spans every column of every chunk.
    """
    seen = {}
    T = kernel.shape[1] if len(kernel.shape) == 2 else 0
    for t0, t1, c0, c1, key in kernel.chunks(size):
        if full:
            c0, c1, key = 0, T, None
        w = seen.get(key) if key is not None else None
        if w is None:
            w = kernel.block(t0, t1, c0, c1)
            if transform is not None:
                w = transform(w)
            if key is not None:
                seen[key] = w
        yield t0, t1, c0, c1, key, w
