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
    "segment",
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
    """Segment one features object with :func:`segment` and return its boundaries."""
    return segment(entry, **params)


def _params_key(params):
    """Hashable key for segmentation parameters (unhashable values by identity)."""
    out = []
    for k in sorted(params):
        v = params[k]
        try:
            hash(v)
        except TypeError:
            v = ("id", id(v))
        out.append((k, v))
    return tuple(out)


def _finish(parsed, params, cache):
    """Segment the features entries and return the resolution.

    ``cache`` maps (features object, parameters) to the boundaries, so a features object
    that appears several times (across subjects, or across orders sharing ``cache``) with
    the same parameters is segmented once and every appearance gets the same array object.
    """
    params = dict(params or {})

    def one(p):
        if isinstance(p, _Features):
            key = (id(p.entry), _params_key(params))
            if key not in cache:
                # the entry is kept in the cache so that its id is not reused meanwhile
                cache[key] = (p.entry, _segment_features(p.entry, params))
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
        2-D values are features with ``T`` rows, segmented with :func:`segment`. A list or
        tuple holding only
        integers (an empty one included) is a boundary set shared by all subjects. Any other
        list or tuple is per-subject: its length must equal ``n_subjects`` and each element
        is ``None`` or one entry (an inner list of integers is boundaries; features must be an
        array or DataFrame). ``None`` means no restriction.
    T : int
        Number of timepoints.
    n_subjects : int
        Number of subjects.
    events_params : dict, optional
        Keyword arguments for :func:`segment` when an entry holds features. Allowed keys are
        ``n``, ``kind``, ``max_events`` and ``random_state``. Unknown keys, or giving this
        when no entry holds features, raise ``ValueError``.

    Returns
    -------
    resolved : None, numpy.ndarray or list
        ``None`` (no restriction), one boundary array shared by all subjects, or a list of
        length ``n_subjects`` whose items are boundary arrays or ``None``. A features object
        that appears several times is segmented once, and every appearance gets the same
        boundary array object.

    Raises
    ------
    ValueError
        For any malformed specification (see above), and as raised by :func:`segment` for
        features entries.
    RuntimeError, ImportError
        As raised by :func:`segment` for features entries.

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
        segmented once per distinct set of segmentation parameters.

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


# ------------------------------------------------------------------ segmentation
_KINDS = ("ordered", "recurring")

#: EM settings shared by both kinds
_N_ITER = 100
_TOL = 1e-3

#: restarts per fit (ordered: jittered equal-length segments; recurring: random seeds)
_N_RESTARTS = 3

#: variance floor (the features are z-scored before fitting, so this is relative)
_VAR_FLOOR = 1e-3

#: tiny prior pseudo-counts: a state that EM leaves without posterior weight keeps a
#: finite mean (the features' mean, 0) and transition rows that sum to 1, instead of 0/0
_PRIORS = {"transmat_prior": 1.0 + 1e-6, "means_prior": 0.0, "means_weight": 1e-6}


def _check_count(x, name, low, high=None):
    if not _is_int_scalar(x) or x < low or (high is not None and x > high):
        rng = f"{low}..{high}" if high is not None else f">= {low}"
        raise ValueError(f"{name} must be an integer in {rng}, got {x!r}")
    return int(x)


def _hmm():
    from ._lazy import lazy_import

    return lazy_import("hmmlearn.hmm", "hmmlearn", "segment")


class _quiet_hmmlearn:
    """Silence hmmlearn's logged convergence notes while fitting (fits are compared by score)."""

    def __enter__(self):
        import logging

        self.logger = logging.getLogger("hmmlearn")
        self.level = self.logger.level
        self.logger.setLevel(logging.ERROR)

    def __exit__(self, *exc):
        self.logger.setLevel(self.level)


def _zscore_columns(x):
    """Z-score each feature (constant features become 0)."""
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0)) / sd


def _finite_model(model):
    return all(
        np.all(np.isfinite(a)) for a in (model.means_, model.covars_, model.transmat_)
    )


def _diag_covars(model):
    """Per-state variances (n_states x n_features) from a diagonal-covariance model."""
    c = np.asarray(model.covars_)
    return np.diagonal(c, axis1=1, axis2=2) if c.ndim == 3 else c


def _restart_edges(T, n):
    """Segment edges for the ordered restarts: equal-length, then shifted by +-1/4 length."""
    seen, out = set(), []
    for shift in (0.0, 0.25, -0.25)[:_N_RESTARTS]:
        edges = np.round(np.linspace(0, T, n + 1) + shift * T / n).astype(int)
        edges[0], edges[-1] = 0, T
        edges = np.clip(edges, 0, T)
        if np.all(np.diff(edges) >= 1) and tuple(edges) not in seen:
            seen.add(tuple(edges))
            out.append(edges)
    return out


def _fit_ordered(z, n):
    """Fit the left-to-right Gaussian HMM; returns the model with the best log-likelihood."""
    hmm = _hmm()
    T, d = z.shape
    best, best_score = None, -np.inf
    for edges in _restart_edges(T, n):
        labels = np.repeat(np.arange(n), np.diff(edges))
        means = np.array([z[labels == k].mean(axis=0) for k in range(n)])
        var = np.mean((z - means[labels]) ** 2, axis=0) + _VAR_FLOOR
        stay = 1.0 - n / T  # expected event length T / n
        transmat = np.diag(np.full(n, stay)) + np.diag(np.full(n - 1, 1.0 - stay), 1)
        transmat[-1, -1] = 1.0
        startprob = np.zeros(n)
        startprob[0] = 1.0
        model = hmm.GaussianHMM(
            n_components=n,
            covariance_type="diag",
            min_covar=_VAR_FLOOR,
            **_PRIORS,
            init_params="",
            params="tmc",
            n_iter=_N_ITER,
            tol=_TOL,
        )
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = np.tile(var, (n, 1))
        try:
            with _quiet_hmmlearn(), np.errstate(all="ignore"):
                model.fit(z)
                score = model.score(z)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if _finite_model(model) and np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError(
            f"the ordered HMM fit with n={n} failed (non-finite parameters or likelihood) "
            "for every restart"
        )
    return best


def _log_emissions(z, model):
    """Diagonal-Gaussian log density of every timepoint under every state (T x n)."""
    means = np.asarray(model.means_)
    var = _diag_covars(model)
    sq = ((z[:, None, :] - means[None, :, :]) ** 2 / var[None, :, :]).sum(axis=2)
    return -0.5 * (z.shape[1] * np.log(2 * np.pi) + np.log(var).sum(axis=1)[None, :] + sq)


def _ordered_viterbi(log_e, transmat):
    """Most likely left-to-right path that starts in state 0 and ends in the last state.

    Each state is visited once, in order, for at least one timepoint. Transition
    probabilities are floored at the smallest positive float so that every such path is
    possible.
    """
    T, n = log_e.shape
    tiny = np.finfo(float).tiny
    log_stay = np.log(np.maximum(np.diag(transmat), tiny))
    log_move = np.log(np.maximum(np.diag(transmat, 1), tiny))
    delta = np.full(n, -np.inf)
    delta[0] = log_e[0, 0]
    moved = np.zeros((T, n), dtype=bool)
    for t in range(1, T):
        stay = delta + log_stay
        move = np.full(n, -np.inf)
        move[1:] = delta[:-1] + log_move
        moved[t] = move > stay
        delta = np.where(moved[t], move, stay) + log_e[t]
    states = np.empty(T, dtype=np.int64)
    k = n - 1
    for t in range(T - 1, -1, -1):
        states[t] = k
        if moved[t, k]:
            k -= 1
    return states


def _starts(states):
    """Block starts: 0 and every timepoint whose label differs from the previous one."""
    return np.concatenate([[0], np.flatnonzero(np.diff(states) != 0) + 1]).astype(np.int64)


def _segment_ordered(z, n):
    if n == 1:
        return np.zeros(z.shape[0], dtype=np.int64)
    if n == z.shape[0]:
        return np.arange(n, dtype=np.int64)
    model = _fit_ordered(z, n)
    return _ordered_viterbi(_log_emissions(z, model), np.asarray(model.transmat_))


def _segment_recurring(z, n, rng):
    if n == 1:
        return np.zeros(z.shape[0], dtype=np.int64)
    hmm = _hmm()
    best, best_score = None, -np.inf
    for _ in range(_N_RESTARTS):
        seed = None if rng is None else int(rng.randint(np.iinfo(np.int32).max))
        model = hmm.GaussianHMM(
            n_components=n,
            covariance_type="diag",
            min_covar=_VAR_FLOOR,
            **_PRIORS,
            n_iter=_N_ITER,
            tol=_TOL,
            random_state=seed,
        )
        try:
            with _quiet_hmmlearn(), np.errstate(all="ignore"):
                model.fit(z)
                score = model.score(z)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if _finite_model(model) and np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError(
            f"the recurring HMM fit with n={n} failed (non-finite parameters or "
            "likelihood) for every restart"
        )
    with _quiet_hmmlearn():
        return best.predict(z).astype(np.int64)


def _wasserstein_score(corr, states):
    """The paper's rule: distance between within- and across-event correlations.

    Pairs ``(t, u)`` with ``0 <= u - t < L`` are compared, where ``L`` is the length of the
    longest event (the smallest lag with no within-event pair). Returns None when either
    set is empty.
    """
    from scipy.stats import wasserstein_distance

    T = len(states)
    same = states[:, None] == states[None, :]
    lag = np.arange(T)[None, :] - np.arange(T)[:, None]
    L = int(np.max(np.diff(np.concatenate([_starts(states), [T]]))))
    local = (lag >= 0) & (lag < L)
    within = corr[same & local]
    across = corr[~same & local]
    within = within[np.isfinite(within)]
    across = across[np.isfinite(across)]
    if within.size == 0 or across.size == 0:
        return None
    return float(wasserstein_distance(within, across))


def _choose_ordered(x, z, max_events):
    """Fit n = 2..min(max_events, T // 2) and keep the n with the largest score."""
    T = x.shape[0]
    with np.errstate(all="ignore"):
        corr = np.corrcoef(x)
    best_states, best_score, scores = None, -np.inf, {}
    for n in range(2, min(max_events, T // 2) + 1):
        states = _segment_ordered(z, n)
        score = _wasserstein_score(corr, states)
        scores[n] = score
        if score is not None and score > best_score:  # ties keep the smaller n
            best_states, best_score = states, score
    if best_states is None:
        raise ValueError(
            "the number of events could not be chosen: no candidate gave both within- and "
            "across-event timepoint correlations (are the timepoint correlations undefined?); "
            "pass n"
        )
    return best_states, scores


def _segment_one(x, n, kind, max_events, rng):
    from .helpers import format_data

    x = format_data(x, fill_nan=False)[0]
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "features to segment must be finite: found NaN or infinite values "
            "(they are not filled in; remove or impute them first)"
        )
    T, d = x.shape
    if n is None:
        if T < 4:
            raise ValueError(f"choosing the number of events needs at least 4 timepoints, got {T}")
        if d < 2:
            raise ValueError(
                f"choosing the number of events needs at least 2 features, got {d}; pass n"
            )
    else:
        n = _check_count(n, "n", 1, T)
    z = _zscore_columns(x)
    if kind == "recurring":
        states = _segment_recurring(z, n, rng)
    elif n is None:
        states = _choose_ordered(x, z, max_events)[0]
    else:
        states = _segment_ordered(z, n)
    return _starts(states), states


def segment(data, n=None, kind="ordered", max_events=50, random_state=None, return_states=False):
    """
    Segment a timeseries into events with a hidden Markov model.

    The features are z-scored, then a Gaussian hidden Markov model (``hmmlearn``'s
    ``GaussianHMM``, diagonal covariance) is fitted to them and each timepoint is assigned
    a state. ``hmmlearn`` is installed on demand the first time it is needed (see Notes).

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or list of these
        Timepoints-by-features observations (for example a stimulus embedding). A list is
        segmented item by item, and the items may differ in length. Values must be finite.
    n : int or None
        Number of events (``kind="ordered"``) or states (``kind="recurring"``), in
        ``1..T``. With ``None`` (ordered kind only) the number is chosen automatically (see
        Notes). Default: None.
    kind : {"ordered", "recurring"}
        ``"ordered"``: every state is visited once, in order (left-to-right transitions,
        starting in the first state); exactly ``n`` blocks are returned. ``"recurring"``:
        an ordinary Gaussian hidden Markov model with ``n`` states; the blocks are the
        maximal runs of one state, so a revisited state starts a new block, and fewer or
        more than ``n`` blocks may result. Default: ``"ordered"``.
    max_events : int
        Largest number of events tried when ``n`` is None (at least 2). Default: 50.
    random_state : int, numpy.random.RandomState or None
        Seeds the random initializations of the recurring kind (the ordered kind is
        initialized deterministically). Default: None (fresh randomness).
    return_states : bool
        If True, also return the state of each timepoint. Default: False.

    Returns
    -------
    boundaries : numpy.ndarray
        Sorted ``int64`` block-start indices, beginning with 0 (a boundary set accepted by
        ``events`` in :func:`timecorr.timecorr`).
    states : numpy.ndarray
        Only with ``return_states=True``: the ``int64`` state of each timepoint (length
        ``T``). For the ordered kind it equals the block index.

    For a list ``data``, a list with one result per item.

    Raises
    ------
    ValueError
        For an unknown ``kind``; ``n`` not an integer in ``1..T``; ``n=None`` with
        ``kind="recurring"``, with fewer than 4 timepoints, or with fewer than 2 features;
        ``max_events`` not an integer of at least 2; features with NaN or infinite values.
    RuntimeError
        If every restart of a fit gives non-finite parameters or likelihood.
    ImportError
        If ``hmmlearn`` is missing and cannot be installed (see Notes).

    Notes
    -----
    Ordered kind. The model starts in the first state, may only stay or move to the next
    state, and its means start from ``n`` equal-length segments (variances from the pooled
    within-segment variance plus a floor). Three restarts with the segment edges shifted by
    a quarter segment are fitted and the one with the highest log-likelihood is kept. The
    fit itself does not force the sequence to end in the last state; the decoding does: a
    Viterbi pass restricted to paths that start in the first state, end in the last state
    and visit every state for at least one timepoint.

    Automatic number of events (``n=None``). For each ``n`` in
    ``2..min(max_events, T // 2)`` the ordered model is fitted, and the ``n`` with the
    largest Wasserstein distance (``scipy.stats.wasserstein_distance``) between the
    within-event and across-event timepoint-by-timepoint Pearson correlations of ``data``
    is kept (ties: the smallest ``n``). Only pairs ``(t, u)`` with ``0 <= u - t < L`` are
    compared, ``L`` being the length of the longest event (the diagonal included). This is
    the rule of Heusser, Fitzpatrick and Manning (Nature Human Behaviour), who segmented
    with ``brainiak``'s
    ``EventSegment``; this function is a reimplementation with ``hmmlearn``, and its
    results need not match ``brainiak``'s numerically. The rule is a heuristic for choosing
    among candidate counts: it does not test whether events exist.

    Recurring kind. ``hmmlearn``'s default initialization (k-means means) and three
    restarts drawn from ``random_state``; the highest log-likelihood is kept and decoded
    with hmmlearn's Viterbi algorithm.

    Both kinds use tiny prior pseudo-counts on the transitions and means, so that a state
    that receives no weight during fitting keeps finite parameters.

    Installation. ``hmmlearn`` is not a required dependency (it is in the ``segment``
    extra: ``pip install "timecorr[segment]"``). When it is missing, ``segment`` installs
    it into the running interpreter with ``python -m pip install hmmlearn`` and prints a
    one-line notice to standard error. Setting the environment variable
    ``TIMECORR_AUTO_INSTALL`` to ``0`` (or ``false`` or ``no``) turns this off.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> rng = np.random.RandomState(0)
    >>> means = rng.randn(3, 5)
    >>> x = np.repeat(means, [30, 40, 30], axis=0) + 0.3 * rng.randn(100, 5)
    >>> tc.segment(x, n=3)
    array([ 0, 30, 70])
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {list(_KINDS)}, got {kind!r}")
    if n is None and kind == "recurring":
        raise ValueError('kind="recurring" needs the number of states n (n=None is ordered only)')
    if n is not None and not _is_int_scalar(n):
        raise ValueError(f"n must be an integer or None, got {n!r}")
    max_events = _check_count(max_events, "max_events", 2)
    if random_state is None or isinstance(random_state, np.random.RandomState):
        rng = random_state
    elif _is_int_scalar(random_state):
        rng = np.random.RandomState(int(random_state))
    else:
        raise ValueError(
            f"random_state must be None, an integer or a numpy RandomState, got {random_state!r}"
        )

    def one(x):
        b, states = _segment_one(x, n, kind, max_events, rng)
        return (b, states) if return_states else b

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("data must contain at least one timeseries; got an empty list")
        return [one(x) for x in data]
    return one(data)
