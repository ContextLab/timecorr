"""
Squared-correlation summaries of time-varying correlation matrices.

``WeightedStrength`` and ``Communities`` turn each fitting unit's series of
correlation matrices (vectorized with ``mat2vec``: one (T, K(K+1)/2) array per
unit) into a few columns per node or per community.  Both are usable as
``rfun`` of ``reduce`` and ``timecorr`` (as objects, or through the strings
``"weighted_strength"``, ``"weighted_strength_split"`` and ``"communities"``).

Each summary works on blocks of complete rows of the (K, K, T) correlation
array: ``_accumulate`` adds a block's sums of ``r**2`` (and of ``r``) to a
fitting state, and ``_emit`` adds a block's contribution to the output.  The
materialized ``fit`` and ``transform`` use one block holding every row.
``_streamed_summary`` computes the correlations of ``autofc``, ``isfc`` or
``wisfc`` a block of rows at a time from the data (two passes: fit, then
output), so the (K, K, T) array is held only when one block covers all K
rows; ``_shift_null_threshold`` gives
the threshold of the automatic number of communities.
"""

import numbers
import warnings

import numpy as np
from sklearn.cluster import KMeans

from .helpers import (
    _WCORR_NAN_PRODUCTS,
    _corr_row_blocks,
    _fisher_mean,
    _kernels_message,
    _nan_weighted_mean,
    _subject_kernels,
    _subject_weight_matrix,
    _warn_user,
    _wcorr_block_rows,
    autofc,
    isfc,
    vec2mat,
    wisfc,
)

__all__ = ["WeightedStrength", "Communities"]


# ----------------------------------------------------------------------------
# row-block operations and weights
# ----------------------------------------------------------------------------


def _new_state(K, T, signed):
    """
    Empty fitting state of one unit with K nodes and T timepoints.

    ``"sum_r2"``: (K, K) sums over time of ``r_ij(t)**2``; ``"sum_r"``: (K, K)
    sums of ``r_ij(t)`` (None unless ``signed``); ``"T"``: the number of
    timepoints.
    """
    return {
        "sum_r2": np.zeros((K, K)),
        "sum_r": np.zeros((K, K)) if signed else None,
        "T": int(T),
    }


def _accumulate(state, i0, i1, R):
    """
    Add rows ``i0:i1`` to a fitting state.

    ``R`` has shape (i1 - i0, K, T): rows ``i0:i1`` of the (K, K, T)
    correlation array, diagonal included (the diagonal is ignored later by
    ``_pair_weights`` and ``_mean_r``).  Each row's sums depend on that row
    only, so the result does not depend on the block sizes.
    """
    state["sum_r2"][i0:i1] += np.einsum("ijt,ijt->ij", R, R)
    if state["sum_r"] is not None:
        state["sum_r"][i0:i1] += R.sum(axis=2)


def _pair_weights(sum_r2, T):
    """Pair weights ``w_ij = mean_t r_ij(t)**2`` from summed squares; zero diagonal."""
    w = np.asarray(sum_r2, dtype=float) / T
    np.fill_diagonal(w, 0)
    return w


def _mean_r(sum_r, T):
    """Mean correlations ``mean_t r_ij(t)`` from sums; zero diagonal."""
    m = np.asarray(sum_r, dtype=float) / T
    np.fill_diagonal(m, 0)
    return m


def _clip_weights(w):
    """
    Clipped weights ``a_ij = max(w_ij - m, 0)``, ``m`` = median of ``w_ij``
    over i < j; for K < 4 the weights are returned unclipped.  Zero diagonal.
    """
    K = w.shape[0]
    if K < 4:
        a = np.array(w, dtype=float)
    else:
        a = np.maximum(w - np.median(w[np.triu_indices(K, 1)]), 0)
    np.fill_diagonal(a, 0)
    return a


def _centered(wbar):
    """``C = Wbar - median(Wbar over i < j)`` with zero diagonal."""
    K = wbar.shape[0]
    C = wbar - np.median(wbar[np.triu_indices(K, 1)])
    np.fill_diagonal(C, 0)
    return C


def _count_above(wbar, threshold, K):
    """
    Number of communities from the eigenvalues of ``_centered(wbar)``: 0 when
    no eigenvalue exceeds ``threshold``; otherwise ``max(2, c)`` for ``c``
    eigenvalues above it, capped at ``K // 2``.
    """
    ev = np.linalg.eigvalsh(_centered(wbar))
    c = int(np.sum(ev > threshold))
    return 0 if c == 0 else int(min(max(2, c), K // 2))


def _renumber(labels):
    """Renumber labels in order of each label's smallest member index."""
    labels = np.asarray(labels)
    _, first = np.unique(labels, return_index=True)
    out = np.empty(len(labels), dtype=int)
    for new, idx in enumerate(np.sort(first)):
        out[labels == labels[idx]] = new
    return out


def _find_communities(wbar, n):
    """
    Community labels of the K nodes: k-means (``n_init=10``,
    ``random_state=0``) on the ``n`` eigenvectors of ``_centered(wbar)`` with
    the largest eigenvalues, each scaled by the square root of its absolute
    eigenvalue; renumbered by smallest member index.  ``n = 1`` puts every
    node in one community.
    """
    K = wbar.shape[0]
    if n == 1:
        return np.zeros(K, dtype=int)
    ev, U = np.linalg.eigh(_centered(wbar))
    emb = U[:, -n:] * np.sqrt(np.abs(ev[-n:]))
    with warnings.catch_warnings():
        # identical points (no structure) give fewer distinct clusters than n
        warnings.filterwarnings("ignore", message="Number of distinct clusters")
        labels = KMeans(n, n_init=10, random_state=0).fit_predict(emb)
    return _renumber(labels)


def _check_labels(labels, what):
    """Raise ``ValueError`` unless some community has at least 2 members."""
    if not np.any(np.bincount(labels) >= 2):
        raise ValueError(
            f"{what}: the clustering left no community with at least 2 members "
            f"({len(labels)} nodes in {len(np.unique(labels))} communities); "
            "use a smaller n."
        )


def _member_coefficients(labels):
    """
    (K, K) coefficients ``1 / (m (m - 1))`` for pairs i != j in the same
    community of ``m >= 2`` members (0 elsewhere), and the output column of
    each node (-1 for members of single-node communities).
    """
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    sizes = np.bincount(labels)
    coef = same / np.maximum(sizes[labels] * (sizes[labels] - 1), 1)[:, None]
    kept = np.flatnonzero(sizes >= 2)
    column = np.full(sizes.shape[0], -1)
    column[kept] = np.arange(len(kept))
    return coef, column[labels], len(kept)


# ----------------------------------------------------------------------------
# input checks
# ----------------------------------------------------------------------------


def _check_n(n):
    if n is None:
        return None
    if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer or None; got {n!r}.")
    return int(n)


def _as_units(corrs, name):
    """List of float (T, K(K+1)/2) arrays, K, and whether a single array was given."""
    single = not isinstance(corrs, (list, tuple))
    units = [corrs] if single else list(corrs)
    if not units:
        raise ValueError(f"{name}: no correlations were given.")
    out = []
    K = None
    for u in units:
        u = np.asarray(u, dtype=float)
        if u.ndim != 2:
            raise ValueError(
                f"{name}: each unit must be a 2-D (T, K(K+1)/2) array of "
                f"vectorized correlations; got shape {u.shape}."
            )
        if u.shape[0] < 1:
            raise ValueError(
                f"{name}: each unit needs at least one timepoint; got shape {u.shape}."
            )
        V = u.shape[1]
        if V < 1:
            raise ValueError(
                f"{name}: each unit needs at least one column (K(K+1)/2 for K >= 1 "
                f"nodes); got shape {u.shape}."
            )
        k = int(round(0.5 * (np.sqrt(8 * V + 1) - 1)))
        if k * (k + 1) // 2 != V:
            raise ValueError(
                f"{name}: rows of length {V} are not vectorized K x K matrices "
                "(expected length K(K+1)/2)."
            )
        if K is not None and k != K:
            raise ValueError(
                f"{name}: every unit must have the same number of nodes; got "
                f"{K} and {k}."
            )
        if not np.isfinite(u).all():
            raise ValueError(
                f"{name}: the correlations contain NaN or infinite values (the "
                "diagonal included); the summaries need finite correlations."
            )
        K = k
        out.append(u)
    return out, K, single


# ----------------------------------------------------------------------------
# the summaries
# ----------------------------------------------------------------------------


class _Summary:
    """Shared fit / transform logic of the squared-correlation summaries."""

    _needs_communities = False

    def __init__(self, n=None, signed=False):
        self.n = _check_n(n)
        self.signed = bool(signed)
        self.fits_ = []

    # -- fitting ------------------------------------------------------------

    def fit(self, corrs, *, order=None, path=None, null_threshold=None):
        """
        Fit the weights (and communities) to a list of units.

        Parameters
        ----------
        corrs : numpy.ndarray or list of numpy.ndarray
            One (T, K(K+1)/2) array of vectorized correlations (``mat2vec``)
            per fitting unit, or a single array (one unit).
        order : int or None
            Recorded in ``fits_``.  Default: None.
        path : {"analysis", "feeding"} or None
            Recorded in ``fits_``.  Default: None (a direct call).
        null_threshold : float or None
            With ``n=None``, the number of communities is ``max(2, c)``,
            capped at K // 2, where ``c`` is the number of eigenvalues of the
            centered mean weight matrix above this value; with ``c = 0``,
            one community holds every node.  Recorded in ``fits_``.
            Default: None.

        Returns
        -------
        self
            The fitted instance; ``fits_`` has one more entry.

        Raises
        ------
        ValueError
            If the input is not vectorized correlations, has no timepoints,
            contains NaN or infinite values, if ``null_threshold`` is given
            and not finite, if communities are needed with K < 4, ``n``
            larger than K // 2, ``n=None`` without ``null_threshold``, or if
            the clustering leaves no community with at least 2 members.
        """
        units, K, _ = _as_units(corrs, type(self).__name__)
        self._check_fit(K, null_threshold)
        states = []
        for u in units:
            R = vec2mat(u)
            state = _new_state(K, R.shape[2], self.signed)
            _accumulate(state, 0, K, R)
            states.append(state)
        self._fit_states(states, K, order=order, path=path, null_threshold=null_threshold)
        return self

    def _check_fit(self, K, null_threshold):
        """Raise ``ValueError`` for a fit to K nodes that cannot be done (see ``fit``)."""
        name = type(self).__name__
        needs = self._needs_communities
        if null_threshold is not None and not np.isfinite(float(null_threshold)):
            raise ValueError(
                f"{name}: null_threshold must be finite; got {null_threshold!r}."
            )
        if needs and K < 4:
            raise ValueError(
                f"{self._what()} needs K >= 4 nodes (got K = {K}); use "
                "WeightedStrength() (without split) for fewer nodes."
            )
        if needs and self.n is None and null_threshold is None:
            raise ValueError(
                f"{self._what()} needs the number of communities n: give n "
                f"(for example {name}(n=6"
                f"{', split=True' if name == 'WeightedStrength' else ''})).  "
                "The automatic count (n=None) is available in features and "
                "two_path_step on the feeding path, and on the analysis path "
                "with cfun=autofc and combine=null_combine; otherwise give n."
            )
        if needs and self.n is not None and self.n > K // 2:
            raise ValueError(
                f"{self._what()}: n={self.n} communities exceed K // 2 = {K // 2} "
                f"(K = {K} nodes); give n <= {K // 2}."
            )

    def _fit_states(self, states, K, *, order=None, path=None, null_threshold=None):
        """
        Finish a fit from one accumulated state per unit (``_new_state`` /
        ``_accumulate``): pair weights, mean correlations, communities; appends
        the record to ``fits_``.  ``_check_fit`` must have passed.
        """
        weights, means = [], []
        for state in states:
            weights.append(_pair_weights(state["sum_r2"], state["T"]))
            if self.signed:
                means.append(_mean_r(state["sum_r"], state["T"]))

        labels, n = None, None
        if self._needs_communities:
            wbar = np.mean(weights, axis=0)
            if self.n is not None:
                n = self.n
            else:
                n = _count_above(wbar, null_threshold, K)
                if n == 0:
                    _warn_user(
                        f"{type(self).__name__}: no community structure above the "
                        f"shift null (no eigenvalue above "
                        f"null_threshold={null_threshold!r}); one community holds "
                        "every node."
                    )
                    n = 1
            labels = _find_communities(wbar, n)
            _check_labels(labels, self._what())
            n = int(len(np.unique(labels)))

        self.fits_.append(
            {
                "order": order,
                "path": path,
                "weights": weights,
                "mean_r": means if self.signed else None,
                "labels": labels,
                "n": n,
                "null_threshold": null_threshold,
            }
        )

    # -- output -------------------------------------------------------------

    def transform(self, corrs):
        """
        Summarize correlations with the most recent fit.

        Parameters
        ----------
        corrs : numpy.ndarray or list of numpy.ndarray
            One (T, K(K+1)/2) array of vectorized correlations per unit, or a
            single array.  K must equal the fit's.

        Returns
        -------
        list of numpy.ndarray or numpy.ndarray
            One (T, c) array per unit (a single array for a single array).
            With as many units as the fit, unit u uses the fit's weights of
            unit u; otherwise every unit uses the fit's mean weights.

        Raises
        ------
        ValueError
            If the instance was never fitted, K differs from the fit's, or
            the input has no timepoints or contains NaN or infinite values.

        Warns
        -----
        UserWarning
            Once per call when some output column is constant over time (for
            example when all weights are 0).
        """
        name = type(self).__name__
        if not self.fits_:
            raise ValueError(f"{name}: call fit (or the instance itself) before transform.")
        fit = self.fits_[-1]
        units, K, single = _as_units(corrs, name)
        K_fit = fit["weights"][0].shape[0]
        if K != K_fit:
            raise ValueError(
                f"{name}: the fit has {K_fit} nodes but the correlations have {K} nodes."
            )
        same = len(units) == len(fit["weights"])
        c = self._n_columns(fit, K)
        outs = []
        for i, u in enumerate(units):
            R = vec2mat(u)
            unit = self._unit_params(fit, i if same else None)
            out = np.zeros((R.shape[2], c))
            self._emit(fit, unit, 0, K, R, out)
            outs.append(out)
        self._warn_constant(outs)
        return outs[0] if single else outs

    def _warn_constant(self, outs):
        """One ``UserWarning`` when some column of the outputs is constant over time."""
        constant = sum(
            int(np.sum(np.ptp(o, axis=0) == 0)) for o in outs if o.shape[0] > 0
        )
        if constant:
            _warn_user(
                f"{type(self).__name__}: {constant} output "
                f"column{'s are' if constant > 1 else ' is'} constant over time "
                "(zero variance), for example because every weight is 0; returned "
                "as computed."
            )

    def __call__(self, corrs):
        """
        Fit to ``corrs``, then transform ``corrs`` (see ``fit`` and
        ``transform``); ``fits_`` then holds only this fit.
        """
        out = self.fit(corrs).transform(corrs)
        del self.fits_[:-1]
        return out

    def _unit_params(self, fit, unit):
        """
        Per-unit arrays used by ``_emit``: ``"w"`` (pair weights) and, when
        signed, ``"mean_r"``, of fitted unit ``unit``, or the means over the
        fitted units when ``unit`` is None.
        """
        if unit is None:
            w = np.mean(fit["weights"], axis=0)
            m = np.mean(fit["mean_r"], axis=0) if self.signed else None
        else:
            w = fit["weights"][unit]
            m = fit["mean_r"][unit] if self.signed else None
        return {"w": w, "mean_r": m}


class WeightedStrength(_Summary):
    """
    Weighted strength of each node in squared correlations.

    For a unit with off-diagonal correlations ``r_ij(t)``: pair weights
    ``w_ij = mean_t r_ij(t)**2``; ``m`` = median of ``w_ij`` over i < j;
    clipped weights ``a_ij = max(w_ij - m, 0)`` (for K = 2 or 3, ``a_ij =
    w_ij``).  Column i is ``c_i(t) = sum_{j != i} a_ij r_ij(t)**2``.

    Parameters
    ----------
    split : bool
        If True, each column is split into partners j in node i's own
        community and partners in other communities (own block first); the
        two blocks add up to the unsplit columns.  The communities are found
        as by ``Communities`` and shared by all units; needs K >= 4 and ``n``.
        Default: False.
    signed : bool
        If True, K more columns ``s_i(t) = sum_{j != i} mean_t(r_ij)
        r_ij(t)``.  Default: False.
    n : int or None
        Number of communities for ``split=True`` (unused otherwise), at most
        K // 2.  ``n=None`` with ``split=True`` needs ``null_threshold`` in ``fit``.
        Default: None.

    Attributes
    ----------
    fits_ : list of dict
        One dict per fit, with keys ``"order"``, ``"path"`` (as given to
        ``fit``), ``"weights"`` (list of (K, K) pair weights ``w``, one per
        unit, zero diagonal), ``"mean_r"`` (list of (K, K) ``mean_t r_ij``
        when signed, otherwise None), ``"labels"`` ((K,) community of each
        node, or None), ``"n"`` (number of communities found, or None) and
        ``"null_threshold"``.  ``fit`` appends one entry; calling the
        instance (as ``reduce``, ``timecorr`` and the decoders do) leaves
        only the entry of that call.  ``features`` clears ``fits_`` at the
        start of a call and, like ``two_path_step``, appends one entry per
        step the instance reduces.

    Notes
    -----
    Output columns: K magnitude columns (2K with ``split``: own, then other),
    then, when signed, K signed columns (2K with ``split``: own, then other).
    Calling the instance fits to the given units and transforms the same
    units: the weights (and communities) come from the data being summarized.
    ``fit`` and ``transform`` separate the two.  The summary of a unit
    returned by a combine function other than ``null_combine`` describes the
    combined correlations.  The correlations must be finite (NaN or infinite
    values, on the diagonal too, raise ``ValueError``).

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_nested(S=2, T=300, n_supergroups=2,
    ...                                  modules_per_supergroup=2,
    ...                                  nodes_per_module=3, set_random_seed=0)
    >>> corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None,
    ...                     weights_params={"var": 25})
    >>> [c.shape for c in tc.WeightedStrength()(corrs)]
    [(300, 12), (300, 12)]
    >>> tc.WeightedStrength(signed=True)(corrs[0]).shape
    (300, 24)
    >>> tc.WeightedStrength(split=True, n=4)(corrs[0]).shape
    (300, 24)
    """

    def __init__(self, split=False, signed=False, n=None):
        super().__init__(n=n, signed=signed)
        self.split = bool(split)

    @property
    def _needs_communities(self):
        return bool(self.split)

    def __repr__(self):
        return f"WeightedStrength(split={self.split}, signed={self.signed}, n={self.n})"

    def _what(self):
        return "WeightedStrength(split=True)"

    def _n_columns(self, fit, K):
        return K * (2 if self.split else 1) * (2 if self.signed else 1)

    def _unit_params(self, fit, unit):
        p = super()._unit_params(fit, unit)
        p["a"] = _clip_weights(p["w"])
        if fit["labels"] is not None and self.split:
            labels = fit["labels"]
            p["own"] = labels[:, None] == labels[None, :]
        return p

    def _emit(self, fit, unit, i0, i1, R, out):
        """
        Add rows ``i0:i1`` (``R``: (i1 - i0, K, T)) to ``out`` (T, c).

        Row i gives the whole of node i's columns, so ``out`` columns of
        nodes ``i0:i1`` are filled by this block.
        """
        K = R.shape[1]
        a = unit["a"][i0:i1]
        R2 = R * R
        if self.split:
            own = unit["own"][i0:i1]
            out[:, i0:i1] += np.einsum("ij,ijt->ti", a * own, R2)
            out[:, K + i0 : K + i1] += np.einsum("ij,ijt->ti", a * ~own, R2)
        else:
            out[:, i0:i1] += np.einsum("ij,ijt->ti", a, R2)
        if self.signed:
            m = unit["mean_r"][i0:i1].copy()
            # the diagonal of the block (node i with itself) has no weight
            rows = np.arange(i1 - i0)
            m[rows, i0 + rows] = 0
            base = (2 if self.split else 1) * K
            if self.split:
                out[:, base + i0 : base + i1] += np.einsum("ij,ijt->ti", m * own, R)
                out[:, base + K + i0 : base + K + i1] += np.einsum(
                    "ij,ijt->ti", m * ~own, R
                )
            else:
                out[:, base + i0 : base + i1] += np.einsum("ij,ijt->ti", m, R)


class Communities(_Summary):
    """
    Mean squared correlation within each community of nodes.

    One column per community with at least 2 members, in order of the
    community's smallest member index: ``mean over i != j in g of
    r_ij(t)**2``.  The communities are shared by all units: with pair weights
    ``w_ij = mean_t r_ij(t)**2`` per unit and ``Wbar`` their mean over units,
    k-means (``n_init=10``, ``random_state=0``) groups the nodes on the ``n``
    eigenvectors with the largest eigenvalues, each scaled by the square root
    of its absolute eigenvalue, of ``C = Wbar - median(Wbar over i < j)``
    with zero diagonal.  Labels are renumbered by smallest member index.

    Parameters
    ----------
    n : int or None
        Number of communities, at most K // 2.  ``n=None`` needs
        ``null_threshold`` in ``fit``: with ``c`` eigenvalues of ``C`` above
        it, ``max(2, c)`` communities, capped at K // 2 (``c = 0``: one
        community); direct calls with ``n=None`` raise ``ValueError``.
        Default: None.
    signed : bool
        If True, as many more columns: ``mean over i != j in g of
        sign(mean_t r_ij) r_ij(t)``.  Default: False.

    Attributes
    ----------
    fits_ : list of dict
        One dict per fit, with keys ``"order"``, ``"path"``, ``"weights"``
        (list of (K, K) pair weights, one per unit), ``"mean_r"`` (list of
        (K, K) ``mean_t r_ij`` when signed, otherwise None), ``"labels"``
        ((K,)), ``"n"`` (number of communities found) and
        ``"null_threshold"``.  ``fit`` appends one entry; calling the
        instance (as ``reduce``, ``timecorr`` and the decoders do) leaves
        only the entry of that call.  ``features`` clears ``fits_`` at the
        start of a call and, like ``two_path_step``, appends one entry per
        step the instance reduces.

    Notes
    -----
    Needs K >= 4 nodes (``ValueError`` naming ``WeightedStrength``
    otherwise).  When no eigenvalue exceeds ``null_threshold``, a
    ``UserWarning`` says there is no community structure and one community
    holds every node (one column: the mean squared correlation).  A signal
    shared by every node raises the number of eigenvalues above a
    threshold, so the count from the shift null can exceed the number of
    modules.  On data without community structure, usually (not always) no
    eigenvalue exceeds the shift null; the null's false-positive rate is
    not controlled.  ``fits_[-1]["n"]`` is the number of communities the
    clustering found.  Calling the instance fits to the given units and
    transforms the same units; the communities come from the data being
    summarized.  ``fit`` and ``transform`` separate the two.  The
    correlations must be finite (NaN or infinite values, on the diagonal
    too, raise ``ValueError``).

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_nested(S=2, T=300, n_supergroups=2,
    ...                                  modules_per_supergroup=2,
    ...                                  nodes_per_module=3, set_random_seed=0)
    >>> corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None,
    ...                     weights_params={"var": 25})
    >>> comm = tc.Communities(n=4)
    >>> [c.shape for c in comm(corrs)]
    [(300, 4), (300, 4)]
    >>> comm.fits_[0]["labels"]
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    """

    _needs_communities = True

    def __init__(self, n=None, signed=False):
        super().__init__(n=n, signed=signed)

    def __repr__(self):
        return f"Communities(n={self.n}, signed={self.signed})"

    def _what(self):
        return "Communities"

    def _n_columns(self, fit, K):
        n_kept = int(np.sum(np.bincount(fit["labels"]) >= 2))
        return n_kept * (2 if self.signed else 1)

    def _unit_params(self, fit, unit):
        p = super()._unit_params(fit, unit)
        coef, column, n_kept = _member_coefficients(fit["labels"])
        p["coef"], p["column"], p["n_kept"] = coef, column, n_kept
        if self.signed:
            p["signed_coef"] = coef * np.sign(p["mean_r"])
        return p

    def _emit(self, fit, unit, i0, i1, R, out):
        """
        Add rows ``i0:i1`` (``R``: (i1 - i0, K, T)) to ``out`` (T, c).

        Each row adds its pairs to its community's column.
        """
        column = unit["column"][i0:i1]
        mag = np.einsum("ij,ijt->it", unit["coef"][i0:i1], R * R)
        sgn = (
            np.einsum("ij,ijt->it", unit["signed_coef"][i0:i1], R)
            if self.signed
            else None
        )
        for b, col in enumerate(column):
            if col < 0:
                continue
            out[:, col] += mag[b]
            if self.signed:
                out[:, unit["n_kept"] + col] += sgn[b]


# the summary each name in helpers._SUMMARY_NAMES stands for
_NAMED = {
    "weighted_strength": lambda: WeightedStrength(),
    "weighted_strength_split": lambda: WeightedStrength(split=True),
    "communities": lambda: Communities(),
}


def _from_name(name):
    """A new summary instance for a name in ``helpers._SUMMARY_NAMES``."""
    return _NAMED[name]()


# ----------------------------------------------------------------------------
# streaming: correlations in row blocks
# ----------------------------------------------------------------------------

# float64 elements held at the peak of a streamed step (see
# _streamed_peak_elements).  Measured with tracemalloc (autofc and isfc, with
# and without NaN samples, T 150-600, K 30-120, S 2-6): per row of a block up
# to 28 K T; fixed terms up to about 10 K T + 5.4 S K T + 6 T**2.
_PEAK_PER_BLOCK = 32  # times b * K * T: block arrays and estimator temporaries
_PEAK_PER_DATA = 16  # times T * K: the estimator's per-feature terms
_PEAK_PER_SUBJECT = 6  # times T * K per subject: stacked data, references, outputs
_PEAK_PER_KERNEL = 7  # times T * T: kernel copies of both orientations
_PEAK_PER_UNIT_PAIRS = 6  # times K * K per subject: accumulators and fitted weights
_FISHER_CHUNKS = 16  # the Fisher average of a block runs on 1/16 of it at a time


def _streamed_peak_elements(T, K, S, b):
    """
    Number of float64 elements a streamed step holds at its peak, for S
    subjects of T timepoints and K features and row blocks of b rows:
    ``32 b K T`` (the block, its other orientation and the estimator's
    temporaries) ``+ 16 T K`` (the estimator's per-feature terms) ``+ 6 S T
    K`` (stacked data, references, outputs of up to 4K columns) ``+ 7 T**2``
    (kernel copies of both orientations) ``+ 6 S K**2`` (accumulators and
    fitted weights).  The coefficients are measured with ``tracemalloc``
    (autofc and isfc, with and without NaN samples), with margin; the
    caller's inputs (data and kernels) are not included.
    """
    T, K, S, b = int(T), int(K), int(S), int(b)
    return (
        _PEAK_PER_BLOCK * b * K * T
        + _PEAK_PER_DATA * T * K
        + _PEAK_PER_SUBJECT * S * T * K
        + _PEAK_PER_KERNEL * T * T
        + _PEAK_PER_UNIT_PAIRS * S * K * K
    )


def _block_rows_for(T, K, S, max_bytes):
    """
    The largest number of rows per block b (at most K) whose peak
    (``8 * _streamed_peak_elements(T, K, S, b)`` bytes) fits in
    ``max_bytes``; 0 if one row does not fit.
    """
    fixed = 8 * _streamed_peak_elements(T, K, S, 0)
    per_row = 8 * _PEAK_PER_BLOCK * int(K) * int(T)
    if max_bytes < fixed + per_row:
        return 0
    return int(min(int(K), (max_bytes - fixed) // per_row))


def _correlation_kind(cfun):
    """``"auto"``, ``"isfc"`` or ``"wisfc"`` for the supported correlation functions."""
    for kind, f in (("auto", autofc), ("isfc", isfc), ("wisfc", wisfc)):
        if cfun is f:
            return kind
    raise ValueError(
        "the streamed summaries support cfun=autofc, isfc or wisfc; got "
        f"{getattr(cfun, '__name__', cfun)!r}."
    )


def _as_subjects(subjects):
    """List of float (T, K) arrays of equal shape."""
    if not isinstance(subjects, (list, tuple)):
        subjects = [subjects]
    out = [np.asarray(x, dtype=np.float64) for x in subjects]
    if not out or any(x.ndim != 2 for x in out):
        raise ValueError("subjects must be a non-empty list of 2-D (T, K) arrays.")
    if any(x.shape != out[0].shape for x in out):
        raise ValueError(
            "every subject must have the same shape; got "
            f"{sorted({x.shape for x in out})}."
        )
    return out


def _rows_per_block(max_block_elements, T_out, K, T=None, has_nan=False):
    """
    Rows per block for blocks of at most ``max_block_elements`` elements; for
    None, the estimator's own block size (``helpers._wcorr_block_rows`` for
    T input timepoints, ``T`` defaulting to ``T_out``, and K columns, with
    its smaller blocks for data with NaN samples), at most K.
    """
    if max_block_elements is None:
        T = T_out if T is None else T
        products = _WCORR_NAN_PRODUCTS if has_nan else 1
        return min(int(K), _wcorr_block_rows(int(T), int(K), products))
    b = int(max_block_elements) // (int(T_out) * int(K))
    if b < 1:
        raise MemoryError(
            f"the streamed step needs blocks of at least one row of {T_out} x {K} "
            f"correlations ({T_out * K} elements); max_block_elements="
            f"{max_block_elements} is smaller."
        )
    return min(b, K)


def _fisher_rows(R, Rt):
    """
    In place, ``R = _fisher_mean([R, Rt])`` elementwise (the symmetrization of
    ``wisfc``), a part of the block at a time.
    """
    flat, flat_t = R.reshape(-1), Rt.reshape(-1)
    step = max(1, flat.size // _FISHER_CHUNKS)
    for k0 in range(0, flat.size, step):
        k1 = min(flat.size, k0 + step)
        flat[k0:k1] = _fisher_mean(np.stack([flat[k0:k1], flat_t[k0:k1]]), axis=0)


def _unit_row_blocks(a, ref, kernel, symmetric, b):
    """
    Rows of one subject's (K, K, T') correlations as ``wisfc`` returns them
    (before ``mat2vec``), in blocks of b rows: ``(i0, i1, R, cancelled_all,
    sparse_any)`` as from ``helpers._corr_row_blocks``.  With ``symmetric``
    (the reference is the subject's own data) the rows are those of the
    autocorrelation; otherwise each block holds rows of ``C`` (subject versus
    reference) and of ``C.T`` (reference versus subject), averaged in Fisher z
    space as ``_fisher_symmetrize`` does.
    """
    if symmetric:
        for i0, i1, R, cancelled, sparse in _corr_row_blocks(a, ref, kernel, True, b):
            yield i0, i1, R, cancelled, sparse
        return
    forward = _corr_row_blocks(a, ref, kernel, False, b)
    backward = _corr_row_blocks(ref, a, kernel, False, b)
    for (i0, i1, R, cancelled, sparse), (_, _, Rt, _, _) in zip(forward, backward):
        R = np.ascontiguousarray(R)
        Rt = np.ascontiguousarray(Rt)
        _fisher_rows(R, Rt)
        del Rt
        yield i0, i1, R, cancelled, sparse


class _Units:
    """
    The fitting units of a streamed step: each subject, its kernel and its
    reference (the subject's own data for ``autofc``; otherwise the weighted
    mean of the subjects that ``wisfc`` correlates it with).
    """

    def __init__(self, subjects, kernels, cfun, grouping=None):
        self.kind = _correlation_kind(cfun)
        self.subjects = _as_subjects(subjects)
        self.has_nan = any(bool(np.isnan(x).any()) for x in self.subjects)
        S = len(self.subjects)
        if grouping is None:
            self.kernels, self.group_of = _subject_kernels(kernels, S)
            self.kernels = [np.asarray(k, dtype=np.float64) for k in self.kernels]
        else:
            # (kernels, group_of) of another _Units with the same subjects' kernels
            self.kernels, self.group_of = grouping
        if self.kind == "auto":
            self.weights = None
            self.stacked = None
        else:
            given = 1 - np.eye(S) if self.kind == "isfc" else None
            self.weights = _subject_weight_matrix(self.subjects, given)
            self.stacked = np.stack(self.subjects, axis=2)
        self._mean = _nan_weighted_mean

    def __len__(self):
        return len(self.subjects)

    def kernel(self, s):
        return self.kernels[self.group_of[s]]

    def blocks(self, s, b):
        a = self.subjects[s]
        if self.kind == "auto":
            return _unit_row_blocks(a, a, self.kernel(s), True, b)
        ref = self._mean(self.stacked, self.weights[s, :])
        symmetric = bool(np.array_equal(a, ref, equal_nan=True))
        return _unit_row_blocks(a, ref, self.kernel(s), symmetric, b)


def _accumulate_units(units, K, T_out, signed, b, masks=None):
    """
    Pass 1: one fitting state per unit; with ``masks`` (a pair of lists, one
    entry per distinct kernel), the estimator's cancelled and sparse masks
    are merged into it as ``wisfc`` merges them.
    """
    states = []
    for s in range(len(units)):
        state = _new_state(K, T_out, signed)
        cancelled = np.ones(T_out, dtype=bool)
        sparse = np.zeros(T_out, dtype=bool)
        for i0, i1, R, c, f in units.blocks(s, b):
            _accumulate(state, i0, i1, R)
            cancelled &= c
            sparse |= f
        if masks is not None:
            g = units.group_of[s]
            masks[0][g] = masks[0][g] | cancelled
            masks[1][g] = masks[1][g] | sparse
        states.append(state)
    return states


def _shift_null_threshold(
    subjects, kernels, n_null=5, bandwidth=5.0, max_block_elements=None
):
    """
    Threshold of the automatic number of communities (``autofc`` steps).

    Each repetition circularly shifts every column of every subject's data by
    an independent lag, drawn with ``numpy.random.default_rng(0)`` uniformly
    from the integers in ``[L, T - L]``, ``L = min(T // 4, max(10, 10 *
    bandwidth))``; computes the subjects' pair weights ``w`` with the given
    kernels and ``autofc`` (streamed, in blocks of at most
    ``max_block_elements`` correlations; None: the estimator's default block
    size, see ``_rows_per_block``); and records the
    largest eigenvalue of ``C = Wbar - median(Wbar over i < j)`` (zero
    diagonal).  Returns the maximum over ``n_null`` repetitions.  Lags are
    drawn per repetition, subject by subject, K at a time.  NumPy's global
    random state is not used.

    Raises
    ------
    ValueError
        If T < 40, or for inputs ``autofc`` rejects.
    """
    subjects = _as_subjects(subjects)
    T, K = subjects[0].shape
    if T < 40:
        raise ValueError(
            f"the shift null needs T >= 40 timepoints (got T = {T}); give the "
            "number of communities n."
        )
    L = int(min(T // 4, max(10, 10 * float(bandwidth))))
    rng = np.random.default_rng(0)
    probe = _Units(subjects, kernels, autofc)
    T_out = probe.kernels[0].shape[0]
    b = _rows_per_block(max_block_elements, T_out, K, T, probe.has_nan)
    rows = np.arange(T)[:, None]
    cols = np.arange(K)
    top = []
    for _ in range(int(n_null)):
        shifted = []
        for x in subjects:
            lags = rng.integers(L, T - L, size=K, endpoint=True)
            shifted.append(x[(rows - lags[None, :]) % T, cols])
        units = _Units(
            shifted, kernels, autofc, grouping=(probe.kernels, probe.group_of)
        )
        states = _accumulate_units(units, K, T_out, False, b)
        del units, shifted
        wbar = np.mean([_pair_weights(s["sum_r2"], s["T"]) for s in states], axis=0)
        top.append(float(np.linalg.eigvalsh(_centered(wbar))[-1]))
    return float(max(top))


def _streamed_summary(
    subjects,
    kernels,
    summary,
    cfun,
    *,
    max_block_elements,
    order=None,
    path=None,
    auto_n=False,
    bandwidth=5.0,
):
    """
    A summary of the subjects' correlations, computed in row blocks.

    Equals ``summary(cfun(subjects, kernels))`` (one fitting unit per
    subject, as with ``null_combine``) without the (K, K, T) correlation
    arrays: pass 1 accumulates every unit's pair weights block by block, the
    summary is fit (median clip, communities, the automatic count), and pass 2
    computes the correlations again and adds each block's contribution to the
    outputs.  Rows of ``isfc`` / ``wisfc`` correlations are built from both
    orientations (subject versus reference and reference versus subject) and
    averaged in Fisher z space as ``wisfc`` does.

    Parameters
    ----------
    subjects : list of numpy.ndarray
        One (T, K) array per subject.
    kernels : numpy.ndarray or list of numpy.ndarray
        One T' x T kernel, or one per subject (already restricted and
        event-masked).
    summary : WeightedStrength or Communities
        The summary; its ``fits_`` gains one entry.
    cfun : callable
        ``autofc``, ``isfc`` or ``wisfc`` (default subject weights).
    max_block_elements : int or None
        Largest block of correlations computed at once (b rows hold
        ``b * K * T'`` elements); None: the estimator's default block size
        (``_rows_per_block``).  Only the size
        of one block is checked here; the step's total memory (see
        ``_streamed_peak_elements``) is checked by the caller, which
        converts a byte budget to a block size with ``_block_rows_for``.
    order, path : optional
        Recorded in ``fits_``.
    auto_n : bool
        With ``summary.n`` None and a summary that needs communities, set the
        count with ``_shift_null_threshold`` (``autofc`` only; the callers
        in ``higher_order`` pass True for ``autofc`` only).  Default: False.
    bandwidth : float
        Kernel standard deviation for the shift null's lags.  Default: 5.0.

    Returns
    -------
    (list of numpy.ndarray, str or None)
        One (T', c) array per subject, and the estimator's warning message
        (as ``wisfc`` would issue it, built with ``_kernels_message``) or
        None.  The message is returned, not issued; the summary's own
        warnings (no community structure, constant output columns) are
        issued.

    Raises
    ------
    ValueError
        For an unsupported ``cfun``, the summary's fitting errors, or
        ``auto_n`` with a ``cfun`` other than ``autofc``.
    MemoryError
        If one row of correlations (``K * T'`` elements) exceeds
        ``max_block_elements``.
    """
    units = _Units(subjects, kernels, cfun)
    T, K = units.subjects[0].shape
    T_out = units.kernels[0].shape[0]
    b = _rows_per_block(max_block_elements, T_out, K, T, units.has_nan)

    null_threshold = None
    if auto_n and summary._needs_communities and summary.n is None:
        if units.kind != "auto":
            # not reached from features or two_path_step, which pass
            # auto_n=True only with autofc
            raise ValueError(
                "the automatic number of communities (n=None) is available for "
                "cfun=autofc only; give n."
            )
        summary._check_fit(K, 0.0)
        null_threshold = _shift_null_threshold(
            units.subjects,
            kernels,
            bandwidth=bandwidth,
            max_block_elements=max_block_elements,
        )
    summary._check_fit(K, null_threshold)

    n_kernels = len(units.kernels)
    masks = ([False] * n_kernels, [False] * n_kernels)
    states = _accumulate_units(units, K, T_out, summary.signed, b, masks)
    message = _kernels_message(units.kernels, units.group_of, masks[0], masks[1])
    summary._fit_states(
        states, K, order=order, path=path, null_threshold=null_threshold
    )
    del states

    fit = summary.fits_[-1]
    c = summary._n_columns(fit, K)
    outs = []
    for s in range(len(units)):
        unit = summary._unit_params(fit, s)
        out = np.zeros((T_out, c))
        for i0, i1, R, _, _ in units.blocks(s, b):
            summary._emit(fit, unit, i0, i1, R, out)
        outs.append(out)
    summary._warn_constant(outs)
    return outs, message
