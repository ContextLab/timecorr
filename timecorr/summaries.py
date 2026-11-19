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
"""

import numbers
import warnings

import numpy as np
from sklearn.cluster import KMeans

from .helpers import vec2mat

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
    K = len(labels)
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
        V = u.shape[1]
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
        if np.isnan(u).any():
            raise ValueError(
                f"{name}: the correlations contain NaN; the summaries need "
                "finite correlations."
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
            If the input is not vectorized correlations, contains NaN, if
            communities are needed with K < 4, ``n`` larger than K // 2,
            ``n=None`` without ``null_threshold``, or if the clustering leaves
            no community with at least 2 members.
        """
        name = type(self).__name__
        units, K, _ = _as_units(corrs, name)
        needs = self._needs_communities
        if needs and K < 4:
            raise ValueError(
                f"{self._what()} needs K >= 4 nodes (got K = {K}); use "
                "WeightedStrength() (without split) for fewer nodes."
            )
        if needs and self.n is None and null_threshold is None:
            raise ValueError(
                f"{self._what()} needs the number of communities n: give n "
                f"(for example {name}(n=6"
                f"{', split=True' if name == 'WeightedStrength' else ''})); the "
                "data-driven count (n=None) is available only inside features "
                "and two_path_step."
            )
        if needs and self.n is not None and self.n > K // 2:
            raise ValueError(
                f"{self._what()}: n={self.n} communities exceed K // 2 = {K // 2} "
                f"(K = {K} nodes); give n <= {K // 2}."
            )

        weights, means = [], []
        for u in units:
            R = vec2mat(u)
            state = _new_state(K, R.shape[2], self.signed)
            _accumulate(state, 0, K, R)
            weights.append(_pair_weights(state["sum_r2"], state["T"]))
            if self.signed:
                means.append(_mean_r(state["sum_r"], state["T"]))

        labels, n = None, None
        if needs:
            wbar = np.mean(weights, axis=0)
            if self.n is not None:
                n = self.n
            else:
                n = _count_above(wbar, null_threshold, K)
                if n == 0:
                    warnings.warn(
                        f"{name}: no community structure above the shift null "
                        f"(no eigenvalue above null_threshold={null_threshold!r}); "
                        "one community holds every node.",
                        UserWarning,
                        stacklevel=2,
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
        return self

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
            the input contains NaN.

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
        constant = sum(
            int(np.sum(np.ptp(o, axis=0) == 0)) for o in outs if o.shape[0] > 0
        )
        if constant:
            warnings.warn(
                f"{name}: {constant} output column{'s are' if constant > 1 else ' is'} "
                "constant over time (zero variance), for example because every "
                "weight is 0; returned as computed.",
                UserWarning,
                stacklevel=2,
            )
        return outs[0] if single else outs

    def __call__(self, corrs):
        """Fit to ``corrs``, then transform ``corrs`` (see ``fit`` and ``transform``)."""
        return self.fit(corrs).transform(corrs)

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
        ``"null_threshold"``.

    Notes
    -----
    Output columns: K magnitude columns (2K with ``split``: own, then other),
    then, when signed, K signed columns (2K with ``split``: own, then other).
    Calling the instance fits to the given units and transforms the same
    units: the weights (and communities) come from the data being summarized.
    ``fit`` and ``transform`` separate the two.  The summary of a unit
    returned by a combine function other than ``null_combine`` describes the
    combined correlations.  The correlations must be finite (NaN raises
    ``ValueError``).

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
        ``"null_threshold"``.

    Notes
    -----
    Needs K >= 4 nodes (``ValueError`` naming ``WeightedStrength``
    otherwise).  When no eigenvalue exceeds ``null_threshold``, a
    ``UserWarning`` says there is no community structure and one community
    holds every node (one column: the mean squared correlation).  A signal
    shared by every node raises the number of eigenvalues above a threshold,
    so the count from ``null_threshold`` can exceed the number of
    communities: in simulations with 18 modules and a global signal, the
    count from the shift null used by ``features`` was 20 to 25.  The shift
    null's false-positive rate is measured, not guaranteed: on independent
    autocorrelated noise, about 10% of recordings gave one eigenvalue above
    it.  ``fits_[-1]["n"]`` is the number of communities the clustering
    found.  Calling the instance fits to the given units and transforms the same units; the
    communities come from the data being summarized.  ``fit`` and
    ``transform`` separate the two.  The correlations must be finite (NaN
    raises ``ValueError``).

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
