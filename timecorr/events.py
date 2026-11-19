# coding: utf-8
"""
Event boundaries and event-restricted kernels.

An *event specification* says which timepoints belong together. A *boundary set* is a 1-D
sequence of integer timepoint indices at which a new block starts; the blocks are
``[b_i, b_{i+1})`` and the last block ends at ``T``. The functions here parse specifications
into boundary sets (:func:`resolve_events`, :func:`resolve_events_by_order`), build block
labels (:func:`block_labels`), confine a kernel to blocks (:func:`mask_kernel`) and group
subjects that share the same boundaries (:func:`distinct_event_sets`).
"""

import numbers

import numpy as np
import pandas as pd

try:  # polars is optional
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

__all__ = [
    "check_boundaries",
    "block_labels",
    "resolve_events",
    "resolve_events_by_order",
    "mask_kernel",
    "distinct_event_sets",
]

_PARAM_KEYS = ("n", "kind", "max_events", "random_state")


def _is_int_scalar(x):
    """True for Python and numpy integers, False for bool and numpy bool."""
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _check_T(T):
    if not _is_int_scalar(T) or T < 1:
        raise ValueError(f"T must be a positive integer, got {T!r}")
    return int(T)


def _check_n_subjects(n_subjects):
    if not _is_int_scalar(n_subjects) or n_subjects < 1:
        raise ValueError(f"n_subjects must be a positive integer, got {n_subjects!r}")
    return int(n_subjects)


def check_boundaries(b, T):
    """
    Validate a boundary set and return the block starts.

    Parameters
    ----------
    b : array-like
        1-D sequence of integer timepoint indices at which a new block starts (a list, tuple,
        numpy array, or pandas or polars Series). Index 0 is implied, so it may be omitted or
        included. An empty sequence means a single block.
    T : int
        Number of timepoints.

    Returns
    -------
    boundaries : numpy.ndarray
        Sorted, unique ``int64`` block starts, always beginning with 0.

    Raises
    ------
    ValueError
        If ``b`` is not 1-D, does not have an integer dtype (boolean values and floats, even
        with integral values, are rejected), or has values outside ``[0, T)``.

    Examples
    --------
    >>> from timecorr.events import check_boundaries
    >>> check_boundaries([7, 3, 3, 5], 10)
    array([0, 3, 5, 7])
    """
    T = _check_T(T)
    if pl is not None and isinstance(b, pl.Series):
        arr = b.to_numpy()
    elif isinstance(b, (list, tuple)) and len(b) == 0:
        arr = np.zeros(0, dtype=np.int64)
    else:
        if isinstance(b, (list, tuple)) and any(isinstance(e, (bool, np.bool_)) for e in b):
            raise ValueError("boundaries must be integer indices, not booleans")
        try:
            arr = np.asarray(b)
        except Exception as exc:  # ragged input and the like
            raise ValueError(f"boundaries could not be converted to an array: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"boundaries must be 1-D, got an array with {arr.ndim} dimension(s)")
    if arr.dtype == np.bool_:
        raise ValueError("boundaries must be integer indices, not booleans")
    if arr.size == 0:
        return np.zeros(1, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"boundaries must have an integer dtype, got {arr.dtype}")
    if arr.min() < 0 or arr.max() >= T:
        raise ValueError(
            f"boundaries must lie in [0, {T}), got values from {arr.min()} to {arr.max()}"
        )
    return np.unique(np.concatenate([[0], arr.astype(np.int64)]))


def block_labels(boundaries, T):
    """
    Block index of each timepoint.

    Parameters
    ----------
    boundaries : array-like
        Boundary set (see :func:`check_boundaries`).
    T : int
        Number of timepoints.

    Returns
    -------
    labels : numpy.ndarray
        Integer array of length ``T``; entry ``t`` is the index of the block containing ``t``.

    Examples
    --------
    >>> from timecorr.events import block_labels
    >>> block_labels([3, 7], 10)
    array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    """
    b = check_boundaries(boundaries, T)
    return (np.searchsorted(b, np.arange(int(T)), side="right") - 1).astype(np.int64)


# ------------------------------------------------------------------ specification grammar
class _Features:
    """A features entry waiting to be segmented."""

    def __init__(self, entry):
        self.entry = entry


def _is_features_array(x):
    if isinstance(x, pd.DataFrame):
        return True
    if pl is not None and isinstance(x, pl.DataFrame):
        return True
    return isinstance(x, np.ndarray) and x.ndim == 2


def _is_single_entry(x):
    """Array-like types that count as one entry (type decides, never content)."""
    if isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
        return True
    return pl is not None and isinstance(x, (pl.DataFrame, pl.Series))


def _parse_entry(x, T):
    """Parse one entry: None, boundaries, or features. Returns array, None or _Features."""
    if x is None:
        return None
    if _is_features_array(x):
        if len(x) != T:
            raise ValueError(
                f"events features have {len(x)} rows but the data have {T} timepoints; "
                "pass boundaries (block-start indices) instead, or features with "
                f"exactly {T} rows"
            )
        return _Features(x)
    if _is_single_entry(x):
        return check_boundaries(x, T)  # 1-D (or an error for other shapes)
    if isinstance(x, (list, tuple)):
        if all(_is_int_scalar(e) for e in x):
            return check_boundaries(list(x), T)
        raise ValueError(
            "an events entry that is a list must hold only integer boundaries; "
            "pass features as an array or DataFrame"
        )
    raise ValueError(
        f"an events entry must be None, boundaries, or features (an array or DataFrame); "
        f"got {type(x).__name__}"
    )


def _parse_spec(events, T, n_subjects):
    """Parse a specification for one order into array, None, list, or _Features entries."""
    if events is None:
        return None
    if isinstance(events, dict):
        raise ValueError(
            "a dict of events keyed by order is only valid where several orders are resolved"
        )
    if _is_single_entry(events):
        return _parse_entry(events, T)
    if isinstance(events, (list, tuple)):
        if all(_is_int_scalar(e) for e in events):
            return check_boundaries(list(events), T)
        if len(events) != n_subjects:
            raise ValueError(
                f"per-subject events must have one entry per subject: got {len(events)} "
                f"entries for {n_subjects} subjects"
            )
        out = []
        for i, e in enumerate(events):
            if _is_int_scalar(e) or isinstance(e, (float, bool, np.bool_, np.floating, str)):
                raise ValueError(
                    f"events entry {i} is a scalar ({e!r}); a list mixing scalars with "
                    "per-subject entries is ambiguous. Boundaries must be integers; "
                    "per-subject entries must be None, a list/array of boundaries, "
                    "or an array or DataFrame of features"
                )
            out.append(_parse_entry(e, T))
        return out
    raise ValueError(
        f"events must be None, boundaries, features, or a per-subject list; "
        f"got {type(events).__name__}"
    )


def _check_params_keys(params, what="events_params"):
    if params is None:
        return None
    if not isinstance(params, dict):
        raise ValueError(f"{what} must be a dict or None, got {type(params).__name__}")
    bad = [k for k in params if k not in _PARAM_KEYS]
    if bad:
        raise ValueError(f"unknown {what} key(s) {bad}; allowed keys are {list(_PARAM_KEYS)}")
    return params


def _has_features(parsed):
    if isinstance(parsed, _Features):
        return True
    if isinstance(parsed, list):
        return any(isinstance(p, _Features) for p in parsed)
    return False


def _segment_features(entry, params):
    """Segment one features object (replaced when segmentation is added)."""
    raise NotImplementedError("segmentation is added in a later task")


def _finish(parsed, params, cache):
    """Segment the features entries (each distinct object once) and return the resolution."""

    def one(p):
        if isinstance(p, _Features):
            key = id(p.entry)
            if key not in cache:
                cache[key] = (p.entry, _segment_features(p.entry, dict(params or {})))
            return cache[key][1]
        return p

    if isinstance(parsed, list):
        return [one(p) for p in parsed]
    return one(parsed)


def resolve_events(events, T, n_subjects, events_params=None):
    """
    Parse the event specification for one order.

    Parameters
    ----------
    events : None, array-like, DataFrame, or list
        The specification. A numpy array, pandas Series or DataFrame, or polars Series or
        DataFrame is one entry: 1-D integer values are boundaries (shared by all subjects) and
        2-D values are features with ``T`` rows, to be segmented. A list or tuple holding only
        integers (an empty one included) is a boundary set shared by all subjects. Any other
        list or tuple is per-subject: its length must equal ``n_subjects`` and each element
        is ``None`` or one entry (an inner list of integers is boundaries; features must be an
        array or DataFrame). ``None`` means no restriction.
    T : int
        Number of timepoints.
    n_subjects : int
        Number of subjects.
    events_params : dict, optional
        Keyword arguments for the segmentation of features entries. Allowed keys are ``n``,
        ``kind``, ``max_events`` and ``random_state``. Unknown keys, or giving this when no
        entry holds features, raise ``ValueError``.

    Returns
    -------
    resolved : None, numpy.ndarray or list
        ``None`` (no restriction), one boundary array shared by all subjects, or a list of
        length ``n_subjects`` whose items are boundary arrays or ``None``. A features object
        that appears several times is segmented once.

    Raises
    ------
    ValueError
        For any malformed specification (see above).
    NotImplementedError
        If an entry holds features (segmentation is not available yet).

    Examples
    --------
    >>> from timecorr.events import resolve_events
    >>> resolve_events([4, 8], 12, 3)
    array([0, 4, 8])
    >>> resolve_events([[4], None], 12, 2)
    [array([0, 4]), None]
    """
    params = events_params
    _check_params_keys(params)
    parsed = _parse_spec(events, _check_T(T), _check_n_subjects(n_subjects))
    if params and not _has_features(parsed):
        raise ValueError("events_params was given but no events entry holds features")
    return _finish(parsed, params, {})


def _order_key(k, order):
    if not _is_int_scalar(k) or not 1 <= k <= order:
        raise ValueError(f"events dict keys must be integer orders in 1..{order}, got {k!r}")
    return int(k)


def resolve_events_by_order(events, T, n_subjects, order, events_params=None):
    """
    Parse the event specification for orders ``1..order``.

    Parameters
    ----------
    events : None, dict, or any specification accepted by :func:`resolve_events`
        A dict maps integer orders in ``1..order`` (booleans are rejected) to specifications;
        orders missing from the dict are unrestricted. Any other value applies to every
        order.
    T, n_subjects : int
        Number of timepoints and subjects.
    order : int
        Highest order.
    events_params : dict, optional
        As in :func:`resolve_events`. When ``events`` is a dict, ``events_params`` may instead
        be a dict keyed by order whose values are dicts of segmentation parameters; each such
        order must hold features.

    Returns
    -------
    resolved : dict
        Maps each order ``1..order`` to ``None``, a boundary array, or a list (see
        :func:`resolve_events`). A features object shared across orders or subjects is
        segmented once.

    Examples
    --------
    >>> from timecorr.events import resolve_events_by_order
    >>> out = resolve_events_by_order({2: [4]}, 12, 2, 2)
    >>> out[1] is None, out[2]
    (True, array([0, 4]))
    """
    T = _check_T(T)
    n_subjects = _check_n_subjects(n_subjects)
    if not _is_int_scalar(order) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")
    order = int(order)

    if isinstance(events, dict):
        specs = {}
        for k, v in events.items():
            specs[_order_key(k, order)] = v
    else:
        specs = {k: events for k in range(1, order + 1)}
    parsed = {k: _parse_spec(specs.get(k), T, n_subjects) for k in range(1, order + 1)}

    flat, by_order = None, None
    if events_params is not None:
        if not isinstance(events_params, dict):
            raise ValueError(
                f"events_params must be a dict or None, got {type(events_params).__name__}"
            )
        int_keys = [_is_int_scalar(k) for k in events_params]
        if not events_params:
            pass  # an empty dict gives no parameters
        elif all(int_keys):
            if not isinstance(events, dict):
                raise ValueError(
                    "events_params keyed by order requires events to be a dict keyed by order"
                )
            by_order = {}
            for k, v in events_params.items():
                k = _order_key(k, order)
                by_order[k] = _check_params_keys(v, f"events_params[{k}]")
        elif any(int_keys):
            raise ValueError("events_params mixes integer (order) keys with other keys")
        else:
            flat = _check_params_keys(events_params)

    if by_order is not None:
        for k, v in by_order.items():
            if v and not _has_features(parsed[k]):
                raise ValueError(f"events_params[{k}] was given but order {k} has no features")
    elif flat and not any(_has_features(p) for p in parsed.values()):
        raise ValueError("events_params was given but no events entry holds features")

    cache = {}
    out = {}
    for k in range(1, order + 1):
        params = by_order.get(k) if by_order is not None else flat
        out[k] = _finish(parsed[k], params, cache)
    return out


# ------------------------------------------------------------------ kernel mask
def mask_kernel(weights, boundaries):
    """
    Confine a kernel to blocks.

    Parameters
    ----------
    weights : numpy.ndarray
        ``T x T`` kernel; row ``t`` is the kernel used at timepoint ``t``.
    boundaries : None or array-like
        Boundary set (see :func:`check_boundaries`). ``None`` means no restriction.

    Returns
    -------
    masked : numpy.ndarray
        A new array equal to ``weights`` except that ``masked[t, u] = 0`` whenever ``t`` and
        ``u`` lie in different blocks. With ``boundaries=None``, ``weights`` itself is
        returned. The input is never modified.

    Raises
    ------
    ValueError
        If some row of ``weights`` has two or more non-zero weights but no row of the masked
        kernel has three or more, i.e. no block is long enough to estimate a correlation.
        (A kernel with at most one non-zero weight per row, such as a delta kernel, is
        passed through unchanged.)

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.events import mask_kernel
    >>> masked = mask_kernel(np.ones((6, 6)), [3])
    >>> masked[0].astype(int), masked[5].astype(int)
    (array([1, 1, 1, 0, 0, 0]), array([0, 0, 0, 1, 1, 1]))
    """
    if boundaries is None:
        return weights
    w = np.asarray(weights)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"kernel must be a square T x T array, got shape {w.shape}")
    T = w.shape[0]
    labels = block_labels(boundaries, T)
    same = labels[:, None] == labels[None, :]
    out = np.where(same, w, np.zeros((), dtype=w.dtype))
    if np.sum(w != 0, axis=1).max() >= 2 and np.sum(out != 0, axis=1).max() < 3:
        raise ValueError(
            "the events leave every row of the kernel with fewer than 3 non-zero weights: "
            "no block is long enough to estimate a correlation"
        )
    return out


def distinct_event_sets(resolved, n_subjects):
    """
    Group subjects that share identical boundaries.

    Parameters
    ----------
    resolved : None, numpy.ndarray or list
        Output of :func:`resolve_events`.
    n_subjects : int
        Number of subjects.

    Returns
    -------
    groups : list of tuple
        ``(boundaries, subject_indices)`` pairs in order of first appearance; ``boundaries`` is
        an array or ``None`` and ``subject_indices`` is a sorted list. Every subject appears
        in exactly one group. ``None`` and a single-block boundary set such as ``[0]`` form
        separate groups.

    Examples
    --------
    >>> from timecorr.events import distinct_event_sets
    >>> import numpy as np
    >>> [(None if b is None else b.tolist(), i) for b, i in
    ...  distinct_event_sets([np.array([0, 4]), None, np.array([0, 4])], 3)]
    [([0, 4], [0, 2]), (None, [1])]
    """
    n_subjects = _check_n_subjects(n_subjects)
    if resolved is None or isinstance(resolved, np.ndarray):
        return [(resolved, list(range(n_subjects)))]
    if len(resolved) != n_subjects:
        raise ValueError(
            f"per-subject events have {len(resolved)} entries for {n_subjects} subjects"
        )
    groups = {}
    for i, b in enumerate(resolved):
        key = None if b is None else np.asarray(b, dtype=np.int64).tobytes()
        if key not in groups:
            groups[key] = (b, [])
        groups[key][1].append(i)
    return list(groups.values())
