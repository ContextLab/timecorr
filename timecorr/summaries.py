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
from fractions import Fraction

import numpy as np
from sklearn.cluster import KMeans

from ._kernels import LagKernel
from .helpers import (
    _WCORR_NAN_PRODUCTS,
    _corr_row_blocks,
    _fisher_mean,
    _kernel_work_elements,
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


# relative tolerance below which an output column's range over time is
# round-off (see ``_Summary._warn_constant``)
_CONSTANT_RTOL = 1e-9

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


def _check_center(center):
    """``center`` of ``WeightedStrength``: True, False or ``"median"``."""
    if isinstance(center, (bool, np.bool_)):
        return bool(center)
    if isinstance(center, str) and center == "median":
        return center
    raise ValueError(
        f"center must be True (the lower cluster of the pair weights), "
        f"'median' or False; got {center!r}."
    )


def _center_level(w, rule):
    """
    Level ``m`` the pair weights ``w_ij`` (i < j) are centered on.

    ``rule="median"``: their median.  ``rule="lower_cluster"``: the mean of
    the lower group of the exact two-group split of their sorted values that
    minimizes the total within-group sum of squares (one-dimensional
    two-means; splits fall between distinct values).  When all values are
    equal, that value.  The mean is kept within the lower group's range, so
    a group of equal values gives exactly that value despite round-off.
    Raises ``ValueError`` for non-finite weights.
    """
    x = np.sort(w[np.triu_indices(w.shape[0], 1)].astype(float))
    if not np.all(np.isfinite(x)):
        raise ValueError("The pair weights must be finite to be centered.")
    if rule == "median":
        return float(np.median(x))
    if x[-1] == x[0]:
        return float(x[0])
    n = len(x)
    k = np.arange(1, n)
    c = np.cumsum(x)
    low = c[:-1] / k
    high = (c[-1] - c[:-1]) / (n - k)
    # the between-group sum of squares; its maximum is the least within-group one
    between = k * (n - k) / n * (high - low) ** 2
    between[x[1:] == x[:-1]] = -np.inf
    cut = int(np.argmax(between)) + 1
    return float(np.clip(np.mean(x[:cut]), x[0], x[cut - 1]))


def _clip_weights(w, center=True):
    """
    Clipped weights ``a_ij = max(w_ij - m, 0)``: ``m`` = the mean of the
    lower cluster of the pair weights (``center=True``) or their median
    (``center="median"``; see ``_center_level``); for K < 4, or with
    ``center=False``, the weights are returned unclipped.  Zero diagonal.
    """
    K = w.shape[0]
    if K < 4 or center is False:
        a = np.array(w, dtype=float)
    else:
        rule = "median" if center == "median" else "lower_cluster"
        a = np.maximum(w - _center_level(w, rule), 0)
    np.fill_diagonal(a, 0)
    return a


def _centered(wbar):
    """``C = Wbar - median(Wbar over i < j)`` with zero diagonal."""
    K = wbar.shape[0]
    C = wbar - np.median(wbar[np.triu_indices(K, 1)])
    np.fill_diagonal(C, 0)
    return C


def _exceedances(wbar, threshold):
    """Number of eigenvalues of ``_centered(wbar)`` above ``threshold``."""
    ev = np.linalg.eigvalsh(_centered(wbar))
    return int(np.sum(ev > threshold))


def _count_from_exceedances(c, K):
    """``_count_above`` from the number ``c`` of eigenvalues above the threshold."""
    return 0 if c == 0 else int(min(max(2, c), K // 2))


def _count_above(wbar, threshold, K):
    """
    Number of communities from the eigenvalues of ``_centered(wbar)``: 0 when
    no eigenvalue exceeds ``threshold``; otherwise ``max(2, c)`` for ``c``
    eigenvalues above it, capped at ``K // 2``.
    """
    return _count_from_exceedances(_exceedances(wbar, threshold), K)


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


def _check_n_null(n_null):
    if (
        not isinstance(n_null, numbers.Integral)
        or isinstance(n_null, bool)
        or n_null < 1
    ):
        raise ValueError(f"n_null must be a positive integer; got {n_null!r}.")
    return int(n_null)


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

    def __init__(self, n=None, signed=False, n_null=5):
        self.n = _check_n(n)
        self.signed = bool(signed)
        self.n_null = _check_n_null(n_null)
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

    def _fit_states(self, states, K, *, order=None, path=None, null_threshold=None,
                    n_null=None):
        """
        Finish a fit from one accumulated state per unit (``_new_state`` /
        ``_accumulate``): pair weights, mean correlations, communities; appends
        the record to ``fits_``.  ``_check_fit`` must have passed.  ``n_null``:
        the number of surrogates of a shift null that gave ``null_threshold``
        (recorded as ``"n_null"``; None records nothing).
        """
        weights, means = [], []
        for state in states:
            weights.append(_pair_weights(state["sum_r2"], state["T"]))
            if self.signed:
                means.append(_mean_r(state["sum_r"], state["T"]))

        auto = getattr(self, "_from_auto", False)
        labels, n, n_exceed = None, None, None
        if self._needs_communities:
            wbar = np.mean(weights, axis=0)
            if self.n is not None:
                n = self.n
            else:
                n_exceed = _exceedances(wbar, null_threshold)
                n = _count_from_exceedances(n_exceed, K)
                if n == 0:
                    if not auto:
                        # the automatic feeding reducer warns in _auto_emitter
                        _warn_user(
                            f"{type(self).__name__}: no community structure above "
                            f"the shift null (no eigenvalue above "
                            f"null_threshold={null_threshold!r}); one community "
                            "holds every node."
                        )
                    n = 1
            labels = _find_communities(wbar, n)
            _check_labels(labels, self._what())
            n = int(len(np.unique(labels)))

        record = {
            "order": order,
            "path": path,
            "weights": weights,
            "mean_r": means if self.signed else None,
            "labels": labels,
            "n": n,
            "null_threshold": null_threshold,
        }
        if n_null is not None:
            record["n_null"] = int(n_null)
        if auto and n_exceed is not None:
            # the automatic feeding reducer: raw number of eigenvalues above
            # the threshold (_auto_emitter adds the emitted columns)
            record["n_exceed"] = n_exceed
        self.fits_.append(record)

    # -- output -------------------------------------------------------------

    def transform(self, corrs, *, fit_index=None):
        """
        Summarize correlations with the most recent fit.

        Parameters
        ----------
        corrs : numpy.ndarray or list of numpy.ndarray
            One (T, K(K+1)/2) array of vectorized correlations per unit, or a
            single array.  K must equal the fit's.
        fit_index : None, int, "mean", or list, tuple, range or 1-D array of int
            Which fitted unit's pair weights (and, when signed, mean
            correlations) each unit uses.  None: with as many units as the
            fit, unit u uses fitted unit u (matching by position); otherwise
            every unit uses the mean over the fitted units.  An int ``i``
            (a Python or NumPy integer, or a 0-d integer array): every unit
            uses fitted unit ``i``.  ``"mean"``: every unit uses the mean
            over the fitted units.  A list, tuple, range or 1-D array: unit
            u uses fitted unit ``fit_index[u]``; it needs one entry per unit
            (one for a single array), and entries may repeat.  Indices count
            from 0 and must be below the number of fitted units; bools and
            floats are not indices.  Default: None.

        Returns
        -------
        list of numpy.ndarray or numpy.ndarray
            One (T, c) array per unit (a single array for a single array).

        Notes
        -----
        With ``fit_index=None``, units are matched to the fitted units by
        position only: when the new batch has as many units as the fit, the
        unit at position u receives the pair weights (and, when signed, the
        mean correlations) fitted to the unit at position u, whether or not
        it is the same subject, and no warning is given.  Pass the units in
        the order of the fit, or name the fits with ``fit_index`` (for
        example ``[1, 0]`` for two subjects in the reverse order, or ``0``
        for a held-out subject summarized with the first subject's fit).
        The communities (``labels``) are shared by all units either way.

        Raises
        ------
        ValueError
            If the instance was never fitted, K differs from the fit's, the
            input has no timepoints or contains NaN or infinite values, or
            ``fit_index`` is not None, ``"mean"``, an int or a list, tuple,
            range or 1-D array of ints (bools and floats included), names a
            fitted unit that does not exist, or (a list, tuple, range or
            array) does not have one entry per unit.

        Warns
        -----
        UserWarning
            Once per call when some output column is constant over time (for
            example when all weights are 0).  A column whose range over time
            is nonzero but at most ``1e-9 * max|column|`` is
            round-off: it is set to its mean exactly and counted in the same
            warning.
        """
        name = type(self).__name__
        if not self.fits_:
            raise ValueError(f"{name}: call fit (or the instance itself) before transform.")
        return self._transform_fit(self.fits_[-1], corrs, fit_index=fit_index)

    def _fitted_units(self, fit_index, n_units, n_fit):
        """
        The fitted unit (an index, or None for the mean) each of ``n_units``
        units uses, from ``fit_index`` (see ``transform``) and the number of
        fitted units ``n_fit``.
        """
        name = type(self).__name__
        kinds = (
            "fit_index must be None, 'mean', an int, or a list, tuple, range "
            "or 1-D array of ints"
        )

        def check(i, label):
            if (
                not isinstance(i, numbers.Integral)
                or isinstance(i, (bool, np.bool_))
            ):
                raise ValueError(f"{name}.transform: {kinds}; got {label}={i!r}.")
            if not 0 <= i < n_fit:
                raise ValueError(
                    f"{name}.transform: {label}={i!r} names no fitted unit; the "
                    f"fit has {n_fit} fitted unit{'s' if n_fit != 1 else ''} "
                    f"(indices 0 to {n_fit - 1})."
                )
            return int(i)

        if fit_index is None:
            return list(range(n_units)) if n_units == n_fit else [None] * n_units
        if isinstance(fit_index, str):
            if fit_index != "mean":
                raise ValueError(
                    f"{name}.transform: {kinds}; got fit_index={fit_index!r}."
                )
            return [None] * n_units
        if isinstance(fit_index, np.ndarray):
            if fit_index.ndim == 0:  # a 0-d array is a scalar
                return [check(fit_index[()], "fit_index")] * n_units
            if fit_index.ndim != 1:
                raise ValueError(
                    f"{name}.transform: {kinds}; got an array with "
                    f"{fit_index.ndim} dimensions."
                )
        elif not isinstance(fit_index, (list, tuple, range)):
            # a scalar; check rejects bools, floats, bytes, sets and the like
            return [check(fit_index, "fit_index")] * n_units
        # a list, tuple, range or 1-D array: one entry per unit
        indices = list(fit_index)
        if len(indices) != n_units:
            raise ValueError(
                f"{name}.transform: fit_index has {len(indices)} entries but "
                f"the batch has {n_units} unit{'s' if n_units != 1 else ''}; "
                "give one fitted unit per unit, a single int for all of "
                "them, or None (by position when the batch has as many "
                f"units as the fit, {n_fit}; the mean otherwise)."
            )
        return [check(i, f"fit_index[{u}]") for u, i in enumerate(indices)]

    def _transform_fit(self, fit, corrs, fit_index=None):
        """``transform`` with the fit record ``fit`` (see ``_auto_emitter``)."""
        name = type(self).__name__
        units, K, single = _as_units(corrs, name)
        K_fit = fit["weights"][0].shape[0]
        if K != K_fit:
            raise ValueError(
                f"{name}: the fit has {K_fit} nodes but the correlations have {K} nodes."
            )
        chosen = self._fitted_units(fit_index, len(units), len(fit["weights"]))
        c = self._n_columns(fit, K)
        outs = []
        for i, u in enumerate(units):
            R = vec2mat(u)
            unit = self._unit_params(fit, chosen[i])
            out = np.zeros((R.shape[2], c))
            self._emit(fit, unit, 0, K, R, out)
            outs.append(out)
        self._warn_constant(outs)
        return outs[0] if single else outs

    def _warn_constant(self, outs):
        """
        Set columns that are constant up to round-off to their means (in
        place), then issue one ``UserWarning`` when some column of the
        outputs is constant over time.

        A column of one unit's output counts as constant when its range over
        time is at most ``_CONSTANT_RTOL * max|column|`` with
        ``_CONSTANT_RTOL = 1e-9`` (relative to the column, so the decision
        does not depend on its scale).  A nonzero range within that tolerance is
        round-off (for example 1e-13 on values of 3 when nearly identical
        input features give identical summaries), and the column is set to
        its mean exactly, so a later order meets exact zero variance rather
        than amplified round-off; exact constants are returned unchanged.
        """
        constant = 0
        for o in outs:
            if o.shape[0] == 0:
                continue
            rng = np.ptp(o, axis=0)
            tol = _CONSTANT_RTOL * np.max(np.abs(o), axis=0)
            snap = (rng > 0) & (rng <= tol)
            if snap.any():
                o[:, snap] = np.mean(o[:, snap], axis=0)
            constant += int(np.sum(rng <= tol))
        if constant:
            _warn_user(
                f"{type(self).__name__}: {constant} output "
                f"column{'s are' if constant > 1 else ' is'} constant over time "
                "(zero variance, or a range within round-off of 0, set to the "
                "column's mean), for example because every weight is 0; "
                "returned as computed otherwise."
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
    ``w_ij = mean_t r_ij(t)**2``; clipped weights ``a_ij = max(w_ij - m,
    0)``; column i is ``c_i(t) = sum_{j != i} a_ij r_ij(t)**2``.

    Centering (``center=True``, the default) subtracts the typical weight of
    the weakly coupled pairs, so that only pairs coupled more strongly than
    that background count, however many pairs any one community holds: the
    pair weights (i < j) are sorted and divided at the point that best
    separates a low group from a high group (the division with the smallest
    total squared distance of the weights from their group's mean), and
    ``m`` is the mean of the low group.  With ``center="median"``, ``m`` is
    the median of the pair weights; with ``center=False``, or for K = 2 or 3,
    ``a_ij = w_ij``.

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
    n_null : int
        Number of shift-null surrogates when ``features`` or
        ``two_path_step`` count the communities of ``split=True`` with
        ``n=None`` (see ``Communities``); a positive integer.  Default: 5.
    center : True, "median" or False
        True: for K >= 4 the pair weights are centered on the mean ``m`` of
        their low group (the "lower cluster", described above) and clipped
        at 0 (``a_ij = max(w_ij - m, 0)``).  ``"median"``: the same with
        ``m`` the median of the pair weights.  False: the weights are used
        as they are (``a_ij = w_ij``) for any K.  Any other value raises
        ``ValueError``.  Default: True.

    Attributes
    ----------
    fits_ : list of dict
        One dict per fit, with keys ``"order"``, ``"path"`` (as given to
        ``fit``), ``"weights"`` (list of (K, K) pair weights ``w``, one per
        unit, zero diagonal), ``"mean_r"`` (list of (K, K) ``mean_t r_ij``
        when signed, otherwise None), ``"labels"`` ((K,) community of each
        node, or None), ``"n"`` (number of communities found, or None) and
        ``"null_threshold"``; a fit whose number of communities came from
        the shift null also has ``"n_null"`` (the number of surrogates).
        ``fit`` appends one entry; calling the
        instance (as ``reduce``, ``timecorr`` and the decoders do) leaves
        only the entry of that call.  ``features`` clears ``fits_`` at the
        start of a call and, like ``two_path_step``, appends one entry per
        step the instance reduces.

    Notes
    -----
    Output columns: K magnitude columns (2K with ``split``: own, then other),
    then, when signed, K signed columns (2K with ``split``: own, then other).
    With ``center=True`` or ``"median"``, coupling that is the same for
    every pair (all pair weights equal) gives magnitude columns that are 0
    at every timepoint for K >= 4, while K < 4 is never centered;
    ``center=False`` keeps such coupling.

    Centering removes the background level of the pair weights (noise and
    coupling shared by every pair).  The median is taken over all pairs, so
    when more than half the pairs lie within one community the median is a
    within-community weight of that community, and median centering
    removes most of that community's weights and every weight of a more
    weakly coupled smaller community: with planted communities of 18 and 6
    nodes (153 of the 276 pairs within the larger one; 2 subjects, T 1000,
    static coupling, ``autofc`` with the default kernel, seeds 0-2) and
    loadings 0.4 (larger) and 0.2 (smaller), every median-clipped weight
    within the smaller community is 0.  In the time-varying setting of the
    first row of the table below, the lower cluster lies within the range
    of the between-community weights (34% of them stay above it), so both
    communities keep their weights.  Benchmark (2
    subjects, T 1000,
    ``autofc`` with variance 25; each community's coupling ``rho_g(t)``
    varies slowly over time; score = correlation between the mean of a
    community's magnitude columns and ``rho_g(t)**2``, mean over subjects;
    16 held-out seeds, mean +- standard error):

    ======================================  ==============  ===============
    planted communities (coupling)          median          lower cluster
    ======================================  ==============  ===============
    18 + 6 (0.5, 0.25): small community     .000 +- .000    .425 +- .021
    18 + 6 (0.5, 0.25): large community     .834 +- .007    .833 +- .007
    18 + 6 (0.45, 0.25), global 0.2: small  .000 +- .000    .508 +- .017
    18 + 6 (0.4, 0.4): small community      .639 +- .015    .664 +- .012
    18 + 6 (0.25, 0.5): large community     .652 +- .015    .655 +- .015
    16 + 4 + 4 (0.5, 0.25, 0.25): 3rd       .265 +- .030    .279 +- .027
    ======================================  ==============  ===============

    On the order-2 recovery cases of ``tests/test_nested_recovery.py``
    (T 1000; sustained, sign-flip and global 0.5, each with communities
    and weighted strength on the feeding path; same 16 held-out seeds) the
    lower cluster changed the decode score by -.002 to -.006 (paired
    standard errors .001 to .002), below the standard error of each score
    (.022 to .038).  Other rules measured on the development seeds: the
    lower quartile of the pair weights (also keeps the small community;
    order-2 scores lower by up to .026 +- .006), the shift-null mean pair
    weight (a lower small-community score with a global signal: .398 +-
    .018 versus .508 +- .017 on the held-out seeds), and no centering (order-2 scores lower by up to .346 +- .062
    with a global signal).
    With K = 2 the two magnitude columns are identical (both are
    ``a_12 r_12(t)**2``), so a following correlation step sees two copies of
    one series; with K = 1 there are no pairs and the column is 0.
    Calling the instance fits to the given units and transforms the same
    units: the weights (and communities) come from the data being summarized.
    ``fit`` and ``transform`` separate the two; ``transform`` matches units
    to the fitted units by position when their numbers are equal (unit u
    gets the weights fitted to unit u), and ``transform(...,
    fit_index=...)`` names the fitted unit of each unit instead (see
    ``transform``).  The summary of a unit
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

    def __init__(self, split=False, signed=False, n=None, n_null=5, center=True):
        super().__init__(n=n, signed=signed, n_null=n_null)
        self.split = bool(split)
        self.center = _check_center(center)

    @property
    def _needs_communities(self):
        return bool(self.split)

    def __repr__(self):
        return (
            f"WeightedStrength(split={self.split}, signed={self.signed}, n={self.n}, "
            f"n_null={self.n_null}, center={self.center!r})"
        )

    def _what(self):
        return "WeightedStrength(split=True)"

    def _n_columns(self, fit, K):
        return K * (2 if self.split else 1) * (2 if self.signed else 1)

    def _unit_params(self, fit, unit):
        p = super()._unit_params(fit, unit)
        p["a"] = _clip_weights(p["w"], self.center)
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
        sign(mean_t r_ij) r_ij(t)``.  Both columns of a community are
        computed from the same correlations ``r_ij(t)`` within it; on the
        feeding path of ``features``, the next order's correlations
        therefore include the correlation between each community's
        magnitude column and its own signed column.  Default: False.
    n_null : int
        Number of surrogates of the shift null that sets ``null_threshold``
        when ``features`` or ``two_path_step`` count the communities
        (``n=None``); the threshold is the largest of their top
        eigenvalues.  A positive integer.  Default: 5.

    Attributes
    ----------
    fits_ : list of dict
        One dict per fit, with keys ``"order"``, ``"path"``, ``"weights"``
        (list of (K, K) pair weights, one per unit), ``"mean_r"`` (list of
        (K, K) ``mean_t r_ij`` when signed, otherwise None), ``"labels"``
        ((K,)), ``"n"`` (number of communities found) and
        ``"null_threshold"``; a fit whose number of communities came from
        the shift null also has ``"n_null"`` (the number of surrogates).
        ``fit`` appends one entry; calling the
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
    not controlled.  If the top eigenvalue of the data and those of the
    ``n_null`` surrogates are exchangeable, the chance that it exceeds
    the largest of them is at most ``1 / (n_null + 1)`` (1/6 at the
    default 5; ties make it smaller),
    and any exceedance gives at least two communities; the automatic
    count is a heuristic, and ``n`` can be given instead.  The shift null
    circularly shifts each column of each subject by a lag from the n
    integers in ``[L, T - L]``: when K <= n, the K lags of a subject are
    distinct (drawn without replacement); when K > n, each lag is used
    ``K // n`` or ``ceil(K / n)`` times within a subject, and columns that
    share a lag (a fraction [n q (q - 1) + 2 r q] / [K (K - 1)] of the
    column pairs, K = q n + r: 1/231 at K = 22, n = 21) keep their
    dependence in that surrogate, and a ``UserWarning`` says so.  It assumes that dependence between
    columns does not survive a relative shift of the sizes drawn; strongly
    autocorrelated signals can violate this.  In ``features`` and
    ``two_path_step``, ``L = min(T // 4, max(10, 10 * sd))``, with sd =
    ``sqrt(params["var"])`` when the step's kernel parameters (or, for
    None, the kernel function's default parameters) have a ``"var"`` key,
    and sd = 5 samples otherwise: the width parameters of
    ``laplace_weights`` (``scale``), ``mexican_hat_weights`` (``sigma``),
    ``t_weights`` (``df``), ``boxcar_weights`` (``width``) and of custom
    kernels without ``"var"`` are not read, so ``L = min(T // 4, 50)`` for
    them, as for ``uniform_weights`` and for ``eye_weights`` on the
    feeding path.  With events, whole columns
    are shifted across event boundaries while the event masks stay
    fixed.  The surrogates' squared
    correlations are averaged over every ``s``-th timepoint, where ``s`` is
    the largest step at which two rows of the kernel (the row of the middle
    timepoint and its shifts) still overlap by ``exp(-1/4)`` (the standard
    deviation, rounded down, for ``gaussian_weights``; 1 for
    ``eye_weights``): correlations at closer timepoints are estimated from
    nearly the same samples (on 96 noise inputs of 4 sizes the threshold
    was 1.000 to 1.002 times the threshold from every timepoint, standard
    errors .001 to .003).  ``C`` is centered on the median over all pairs, as in
    ``WeightedStrength(center="median")`` (``WeightedStrength`` centers on
    the lower cluster by default): with communities of 18 and 6 nodes and loadings
    0.4 and 0.2 (see ``WeightedStrength``), the
    automatic count found 6 to 8 communities (adjusted Rand index 0.39 to
    0.90 against the two planted ones) at seeds 0-2, and 2 with equal
    loadings at seeds 0, 1 and 3 (3 at seed 2).  ``fits_[-1]["n"]`` is the
    number of communities the clustering found.  Calling the instance fits to the given units and
    transforms the same units; the communities come from the data being
    summarized.  ``fit`` and ``transform`` separate the two; with
    ``signed=True``, ``transform`` matches units to the fitted units by
    position when their numbers are equal (unit u gets the signs of the
    mean correlations fitted to unit u), and ``transform(...,
    fit_index=...)`` names the fitted unit of each unit instead (see
    ``transform``).  The
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

    def __init__(self, n=None, signed=False, n_null=5):
        super().__init__(n=n, signed=signed, n_null=n_null)

    def __repr__(self):
        return f"Communities(n={self.n}, signed={self.signed}, n_null={self.n_null})"

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


# the fewest community columns the automatic feeding reducer (features and
# two_path_step with delta_rfun="auto") emits; with fewer, the step emits
# _AUTO_FALLBACK of the same correlations instead
_AUTO_MIN_COLUMNS = 4


def _auto_fallback():
    """The summary an automatic feeding step emits when communities are too few."""
    return WeightedStrength(split=False, signed=False, center=False)


def _auto_emitter(summary, fit, K, order=None):
    """
    The summary that emits the output of a step fitted by ``summary``
    (``fit``: its new record, on K nodes).

    For any summary but the automatic feeding reducer (a ``Communities``
    marked ``_from_auto``), ``summary`` itself.  For the automatic feeding
    reducer: ``summary`` when its communities emit at least
    ``_AUTO_MIN_COLUMNS`` columns; otherwise ``_auto_fallback()``, which
    emits from the same fit (its pair weights, uncentered; no second null).  A
    ``UserWarning`` is issued only when no eigenvalue exceeded the shift
    null (one community); communities that emit fewer than 4 columns fall
    back without a warning.  The record gains ``"n_columns"`` (the columns the
    communities emit, singletons excluded) and ``"fallback"`` (None,
    ``"too_few_community_columns"`` or ``"no_null_exceedance"``).
    """
    if not getattr(summary, "_from_auto", False):
        return summary
    columns = summary._n_columns(fit, K)
    fit["n_columns"] = int(columns)
    fit["fallback"] = None
    if columns >= _AUTO_MIN_COLUMNS:
        return summary
    if fit.get("n_exceed") == 0:
        fit["fallback"] = "no_null_exceedance"
        at = "" if order is None else f" at order {order}"
        _warn_user(
            f"Automatic feeding reducer{at}: the shift null found no structure "
            f"(no eigenvalue above null_threshold={fit['null_threshold']:.4g}); "
            f"this order uses WeightedStrength on its {K} input features."
        )
    else:
        # communities were found but emit fewer than 4 columns: a documented
        # rule, recorded in the fit (and in features' return_info), no warning
        fit["fallback"] = "too_few_community_columns"
    return _auto_fallback()


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
_PEAK_PER_KERNEL = 7  # times T * T: kernel copies of both orientations (arrays)
_PEAK_PER_UNIT_PAIRS = 6  # times K * K per subject: accumulators and fitted weights
_FISHER_CHUNKS = 16  # the Fisher average of a block runs on 1/16 of it at a time


def _streamed_kernel_elements(T, dense=False):
    """
    Float64 elements the kernels of a streamed step hold while both
    orientations are read: ``7 T**2`` for arrays (``dense``), and for a
    ``LagKernel`` at most that and two kernels' worth of
    ``helpers._kernel_work_elements``.
    """
    T = int(T)
    dense_elements = _PEAK_PER_KERNEL * T * T
    if dense:
        return dense_elements
    return min(dense_elements, 2 * _kernel_work_elements(T, dense_elements))


def _streamed_peak_elements(T, K, S, b, dense=False):
    """
    Number of float64 elements a streamed step holds at its peak, for S
    subjects of T timepoints and K features and row blocks of b rows:
    ``32 b K T`` (the block, its other orientation and the estimator's
    temporaries) ``+ 16 T K`` (the estimator's per-feature terms) ``+ 6 S T
    K`` (stacked data, references, outputs of up to 4K columns) ``+`` the
    kernels (``_streamed_kernel_elements``: ``7 T**2`` for T x T arrays,
    ``dense``; linear in T for the built-in kernels at large T) ``+ 6 S
    K**2`` (accumulators and fitted weights).  The coefficients are measured
    with ``tracemalloc`` (autofc and isfc, with and without NaN samples),
    with margin; the caller's inputs (data and kernels) are not included.
    """
    T, K, S, b = int(T), int(K), int(S), int(b)
    return (
        _PEAK_PER_BLOCK * b * K * T
        + _PEAK_PER_DATA * T * K
        + _PEAK_PER_SUBJECT * S * T * K
        + _streamed_kernel_elements(T, dense)
        + _PEAK_PER_UNIT_PAIRS * S * K * K
    )


def _block_rows_for(T, K, S, max_bytes, dense=False):
    """
    The largest number of rows per block b (at most K) whose peak
    (``8 * _streamed_peak_elements(T, K, S, b, dense)`` bytes) fits in
    ``max_bytes``; 0 if one row does not fit.
    """
    fixed = 8 * _streamed_peak_elements(T, K, S, 0, dense)
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
            self.kernels = [
                k if isinstance(k, LagKernel) else np.asarray(k, dtype=np.float64)
                for k in self.kernels
            ]
        else:
            # (kernels, group_of) of another _Units with the same subjects' kernels
            self.kernels, self.group_of = grouping
        if self.kind == "auto":
            self.weights = None
            self.stacked = None
        else:
            # wisfc's default weights correlate the subjects' static
            # correlation matrices: computed on the first call of blocks(),
            # in that call's blocks of rows, so they fit the step's budget
            self.weights = (
                _subject_weight_matrix(self.subjects, 1 - np.eye(S))
                if self.kind == "isfc" else None
            )
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
        if self.weights is None:
            self.weights = _subject_weight_matrix(self.subjects, None, rows_per_block=b)
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


def _shift_lags(rng, T, K, L):
    """
    K circular lags from the n integers in ``[L, T - L]``, drawn with the
    generator ``rng``.

    When K <= n: K distinct lags drawn without replacement
    (``L + rng.choice(n, K, replace=False)``).  When K > n: a random
    permutation of the n lags is repeated, truncated to K and shuffled with
    the same generator, so each lag is used ``K // n`` or ``ceil(K / n)``
    times.  Columns that share a lag keep their dependence under that
    surrogate: with K = q n + r > n, a fraction
    [n q (q - 1) + 2 r q] / [K (K - 1)] of the column pairs share a lag
    (``_shared_lag_fraction``; below 1 / n, and near it only when K is much
    larger than n).  Raise ``ValueError`` when there is no lag (n < 1).
    """
    n_lags = int(T) - 2 * int(L) + 1
    K = int(K)
    if n_lags < 1:
        raise ValueError(
            f"the recording is too short for the shift null: T = {T} gives "
            f"no lag in [{L}, {T - L}]; give the number of communities n."
        )
    if K <= n_lags:
        return int(L) + rng.choice(n_lags, size=K, replace=False)
    lags = np.tile(rng.permutation(n_lags), -(-K // n_lags))[:K]
    rng.shuffle(lags)
    return int(L) + lags


def _shared_lag_fraction(K, n_lags):
    """
    The fraction of the K (K - 1) / 2 column pairs that share a lag in each
    draw of ``_shift_lags`` with ``n_lags`` lags: with K = q n + r (n the
    number of lags), r lags are used q + 1 times and n - r lags q times, so
    the fraction is [n q (q - 1) + 2 r q] / [K (K - 1)] (0 when K <= n).
    """
    K, n = int(K), int(n_lags)
    if K < 2:
        return Fraction(0)
    q, r = divmod(K, n)
    return Fraction(n * q * (q - 1) + 2 * r * q, K * (K - 1))


# The shift null averages the surrogates' squared correlations over every
# s-th timepoint: s is the largest step at which two rows of the kernel
# still have this normalized overlap (for a Gaussian kernel: its standard
# deviation, rounded down).  Correlations at timepoints closer than that are
# estimated from nearly the same samples, so the average over every s-th
# timepoint has the variance of the average over all of them (within about
# 1 % for a Gaussian kernel and white data) at 1 / s of the cost.  The level
# is exp(-1/4), the overlap of two Gaussian rows one standard deviation
# apart, less a margin for round-off.
_NULL_ROW_OVERLAP = float(np.exp(-0.25)) * (1 - 1e-6)


def _null_row_stride(kernel):
    """
    The step ``s`` between the timepoints at which the shift null evaluates
    a kernel: the largest ``d >= 1`` such that the kernel's row at the
    middle timepoint, shifted by every step up to ``d``, keeps a normalized
    overlap ``sum(w[i] w[i + d]) / sum(w**2)`` of at least
    ``_NULL_ROW_OVERLAP``, and at most ``T // 8``; 1 when no step does.
    The row is the kernel's as it is (restricted by ``include_timepoints``,
    ``include_zero``, ``exclude_timepoints`` and events), so an array and
    the equal ``LagKernel`` give the same step.
    """
    if isinstance(kernel, LagKernel):
        if kernel.empty:
            return 1
        middle = kernel.T // 2
        c0, c1 = kernel.span(middle, middle + 1)
        row = kernel.block(middle, middle + 1, c0, c1)[0]
        T = kernel.T
    else:
        weights = np.asarray(kernel, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] == 0:
            return 1
        row = weights[weights.shape[0] // 2]
        T = weights.shape[1]
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    row = row[np.flatnonzero(row)[0]: np.flatnonzero(row)[-1] + 1] if row.any() else row
    norm = float(row @ row)
    stride = 1
    for d in range(1, min(len(row), T // 8 + 1)):
        if norm <= 0 or float(row[d:] @ row[:-d]) / norm < _NULL_ROW_OVERLAP:
            break
        stride = d
    return stride


def _every_nth_row(kernel, stride):
    """Every ``stride``-th row of a kernel, starting at row ``stride // 2``:
    an ``_EveryNthRow`` for a ``LagKernel``, the rows of an array."""
    if isinstance(kernel, LagKernel):
        return _EveryNthRow(kernel, stride)
    return np.ascontiguousarray(kernel[stride // 2:: stride])


class _EveryNthRow:
    """
    Every ``stride``-th row of a ``LagKernel``, starting at row ``stride //
    2``, with the interface the estimator reads (``shape``, ``chunks``,
    ``block``): a (T', T) kernel whose rows are computed chunk by chunk.
    """

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = int(stride)
        self.rows = np.arange(self.stride // 2, kernel.shape[0], self.stride)
        self.shape = (len(self.rows), kernel.shape[1])

    def _rows(self, i0, i1):
        return int(self.rows[i0]), int(self.rows[i1 - 1]) + 1

    def chunks(self, size):
        """``(i0, i1, c0, c1, key)`` for consecutive chunks of ``size`` rows."""
        for i0 in range(0, len(self.rows), size):
            i1 = min(len(self.rows), i0 + size)
            t0, t1 = self._rows(i0, i1)
            c0, c1 = self.kernel.span(t0, t1)
            key = self.kernel.key(t0, t1)
            yield i0, i1, c0, c1, None if key is None else key + (self.stride,)

    def block(self, i0, i1, c0, c1):
        """Rows ``i0:i1`` (of the kept rows), columns ``c0:c1`` (a new array)."""
        t0, t1 = self._rows(i0, i1)
        return np.ascontiguousarray(self.kernel.block(t0, t1, c0, c1)[:: self.stride])


def _shift_null_threshold(
    subjects, kernels, n_null=5, bandwidth=5.0, max_block_elements=None,
    row_stride=None,
):
    """
    Threshold of the automatic number of communities (``autofc`` steps).

    Each repetition circularly shifts every column of every subject's data by
    a lag drawn with ``numpy.random.default_rng(0)`` from the n integers in
    ``[L, T - L]``, ``L = min(T // 4, max(10, 10 * bandwidth))``, without
    replacement within a subject when K <= n (``_shift_lags``: the K columns
    of a subject get K distinct lags); when K > n, each lag is used
    ``K // n`` or ``ceil(K / n)`` times within a subject, and columns that
    share a lag (a fraction [n q (q - 1) + 2 r q] / [K (K - 1)] of the
    column pairs, K = q n + r) keep their dependence in that surrogate, and
    one ``UserWarning`` per call says so; computes the subjects' pair weights ``w`` with the given
    kernels and ``autofc`` (streamed, in blocks of at most
    ``max_block_elements`` correlations; None: the estimator's default block
    size, see ``_rows_per_block``), averaging the squared correlations over
    every ``s``-th timepoint (rows ``s // 2, s // 2 + s, ...``), ``s`` being
    ``row_stride`` or, when that is None, the smallest ``_null_row_stride``
    of the kernels if every kernel is a ``LagKernel`` and 1 (every
    timepoint) for kernels given as arrays (``features`` and
    ``two_path_step`` pass the smallest ``_null_row_stride`` for every kind
    of kernel); and records the
    largest eigenvalue of ``C = Wbar - median(Wbar over i < j)`` (zero
    diagonal).  Returns the maximum over ``n_null`` repetitions.  Lags are
    drawn per repetition, subject by subject, K at a time.  NumPy's global
    random state is not used.

    The null assumes that dependence between columns does not survive a
    relative shift of the sizes drawn: distinct lags differ by at least 1
    sample, not by at least L, and strongly autocorrelated signals can stay
    dependent at such shifts, which violates the assumption.  With events
    (event-masked kernels), whole columns are shifted across event
    boundaries while the event masks stay fixed.

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
    n_null = _check_n_null(n_null)
    n_lags = T - 2 * L + 1
    if 1 <= n_lags < K:
        fraction = _shared_lag_fraction(K, n_lags)
        warnings.warn(
            f"The shift null has more columns (K = {K} columns) than "
            f"distinct lags ({n_lags} distinct lags in [{L}, {T - L}] at "
            f"T = {T}): each lag is used {K // n_lags} or "
            f"{-(-K // n_lags)} times within a subject, and the columns that "
            f"share a lag ({fraction} = {float(fraction):.3g} of the column "
            "pairs) keep their dependence in the surrogates.  To avoid this, "
            "give the number of communities n.",
            UserWarning,
            stacklevel=2,
        )
    rng = np.random.default_rng(0)
    probe = _Units(subjects, kernels, autofc)
    null_kernels = probe.kernels
    if row_stride is None:
        row_stride = min(
            _null_row_stride(k) if isinstance(k, LagKernel) else 1 for k in null_kernels
        )
    if row_stride > 1:
        null_kernels = [_every_nth_row(k, int(row_stride)) for k in null_kernels]
    T_out = null_kernels[0].shape[0]
    b = _rows_per_block(max_block_elements, T_out, K, T, probe.has_nan)
    rows = np.arange(T)[:, None]
    cols = np.arange(K)
    top = []
    for _ in range(n_null):
        shifted = []
        for x in subjects:
            lags = _shift_lags(rng, T, K, L)
            shifted.append(x[(rows - lags[None, :]) % T, cols])
        units = _Units(
            shifted, kernels, autofc, grouping=(null_kernels, probe.group_of)
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
    null_row_stride=None,
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
    null_row_stride : int or None
        ``row_stride`` of ``_shift_null_threshold``.  Default: None.

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
            n_null=summary.n_null,
            bandwidth=bandwidth,
            max_block_elements=max_block_elements,
            row_stride=null_row_stride,
        )
    summary._check_fit(K, null_threshold)

    n_kernels = len(units.kernels)
    masks = ([False] * n_kernels, [False] * n_kernels)
    states = _accumulate_units(units, K, T_out, summary.signed, b, masks)
    message = _kernels_message(units.kernels, units.group_of, masks[0], masks[1])
    summary._fit_states(
        states, K, order=order, path=path, null_threshold=null_threshold,
        n_null=None if null_threshold is None else summary.n_null,
    )
    del states

    fit = summary.fits_[-1]
    emitter = _auto_emitter(summary, fit, K, order)
    c = emitter._n_columns(fit, K)
    outs = []
    for s in range(len(units)):
        unit = emitter._unit_params(fit, s)
        out = np.zeros((T_out, c))
        for i0, i1, R, _, _ in units.blocks(s, b):
            emitter._emit(fit, unit, i0, i1, R, out)
        outs.append(out)
    emitter._warn_constant(outs)
    return outs, message
