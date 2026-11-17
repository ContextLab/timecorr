"""
Simulated timeseries with known dynamic correlations.

Order 1 data are Gaussian with a time-varying correlation matrix (the template).
For ``order > 1`` the template is embedded at exactly that order with the
sign-parity cascade: ``p`` latent signs ``A_j(t)`` are drawn with a correlation
matrix as close to the requested ``R(t)`` as signs of Gaussians allow (exactly
``R(t)`` whenever ``sin(pi R(t) / 2)`` is positive semidefinite), and each is
spread over a block of ``m = 2**(order-1)`` features whose product is ``A_j``.
Every proper subset of a block has a product with mean zero, so the structure is
absent below the embedded order.  ``amplitude_noise`` is the standard deviation
of the log-amplitude of each block's product (each of the ``m`` signs gets
log-amplitude noise with standard deviation ``amplitude_noise / sqrt(m)``), so it
sets the same difficulty at every order.  The order-``n`` correlation matrix
among the block products, which the returned truth reports, is the achieved
latent-sign correlation with its off-diagonal entries multiplied by
``exp(-amplitude_noise**2)``.
"""

import numbers
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from .helpers import mat2vec, vec2mat

_DATAGENS = ("block", "ramping", "constant", "random")


def _make_rng(set_random_seed):
    """A ``numpy.random.Generator`` from the ``set_random_seed`` convention."""
    if set_random_seed is None or set_random_seed is False:
        return np.random.default_rng()
    if set_random_seed is True:
        return np.random.default_rng(123)
    if isinstance(set_random_seed, numbers.Integral):
        return np.random.default_rng(int(set_random_seed))
    raise TypeError(
        "set_random_seed must be False, True, or an integer seed; "
        f"got {set_random_seed!r}."
    )


def _check_positive_int(name, value):
    if (
        not isinstance(value, numbers.Integral)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def random_corrmat(K, rng=None):
    """
    Draw a random K x K correlation matrix.

    The matrix is ``A @ A.T + I`` rescaled to unit diagonal, with ``A`` a
    ``K x max(2, K // 2)`` matrix of standard normal draws, so it is symmetric
    and positive definite.

    Parameters
    ----------
    K : int
        Number of features.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    numpy.ndarray
        A K x K correlation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.simulate import random_corrmat
    >>> C = random_corrmat(4, rng=np.random.default_rng(0))
    >>> C.shape, bool(np.allclose(np.diag(C), 1)), bool(np.linalg.eigvalsh(C).min() > 0)
    ((4, 4), True, True)
    """
    if rng is None:
        rng = np.random.default_rng()
    K = _check_positive_int("K", K)
    A = rng.standard_normal((K, max(2, K // 2)))
    C = A @ A.T + np.eye(K)
    s = np.sqrt(np.diag(C))
    C = C / np.outer(s, s)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    return C


def _template(datagen, K, T, B, rng):
    """(T, K, K) stack of correlation matrices for one of the template kinds."""
    if datagen == "constant":
        return np.repeat(random_corrmat(K, rng)[np.newaxis], T, axis=0)
    if datagen == "block":
        B = _check_positive_int("B", B)
        mats = np.stack([random_corrmat(K, rng) for _ in range(B)])
        # B events of near-equal integer length
        return mats[np.minimum(np.arange(T) * B // T, B - 1)]
    if datagen == "ramping":
        iu = np.triu_indices(K, 1)
        a = random_corrmat(K, rng)
        candidates = [random_corrmat(K, rng) for _ in range(101)]
        if K > 2:
            # end point: the candidate least correlated with the start
            dist = cdist(
                a[iu][np.newaxis], np.stack([c[iu] for c in candidates]), "correlation"
            )[0]
            b = candidates[int(np.argmax(dist))]
        else:
            # one entry: correlation distance is undefined; take the farthest value
            b = candidates[int(np.argmax([abs(c[0, 1] - a[0, 1]) for c in candidates]))]
        mu = np.linspace(1, 0, T)[:, np.newaxis, np.newaxis]
        return mu * a + (1 - mu) * b
    if datagen == "random":
        return np.stack([random_corrmat(K, rng) for _ in range(T)])
    raise ValueError(f"datagen must be one of {_DATAGENS}; got {datagen!r}.")


def _gaussian(R, mean, rng):
    """One (T, K) draw with x(t) ~ N(mean, R(t))."""
    T, K = R.shape[:2]
    L = np.linalg.cholesky(R)
    return mean + np.einsum("tij,tj->ti", L, rng.standard_normal((T, K)))


def _vectorize(R):
    """(T, K, K) stack -> (T, K(K+1)/2) rows in ``mat2vec`` layout."""
    return mat2vec(np.transpose(R, (1, 2, 0)))


def _dataset(datagen, K, T, B, rng):
    rng = np.random.default_rng() if rng is None else rng
    R = _template(datagen, K, T, B, rng)
    mean = 0.0 if datagen == "random" else 0.1
    return _gaussian(R, mean, rng), _vectorize(R)


def ramping_dataset(K, T, B=5, rng=None):
    """
    Gaussian data whose correlation matrix moves linearly between two templates.

    Parameters
    ----------
    K : int
        Number of features.
    T : int
        Number of timepoints.
    B : int
        Unused; accepted so that all dataset functions share one signature.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.1.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array: each timepoint's correlation matrix in ``mat2vec``
        layout.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.simulate import ramping_dataset
    >>> data, corrs = ramping_dataset(3, 20, rng=np.random.default_rng(0))
    >>> data.shape, corrs.shape
    ((20, 3), (20, 6))
    """
    return _dataset("ramping", K, T, B, rng)


def random_dataset(K, T, B=5, rng=None):
    """
    Gaussian data with an independent random correlation matrix at each timepoint.

    Parameters
    ----------
    K : int
        Number of features.
    T : int
        Number of timepoints.
    B : int
        Unused; accepted so that all dataset functions share one signature.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array in ``mat2vec`` layout.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.simulate import random_dataset
    >>> data, corrs = random_dataset(3, 20, rng=np.random.default_rng(0))
    >>> data.shape, corrs.shape
    ((20, 3), (20, 6))
    """
    return _dataset("random", K, T, B, rng)


def constant_dataset(K, T, B=5, rng=None):
    """
    Gaussian data with one correlation matrix at every timepoint.

    Parameters
    ----------
    K : int
        Number of features.
    T : int
        Number of timepoints.
    B : int
        Unused; accepted so that all dataset functions share one signature.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.1.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array in ``mat2vec`` layout (identical rows).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.simulate import constant_dataset
    >>> data, corrs = constant_dataset(3, 20, rng=np.random.default_rng(0))
    >>> data.shape, corrs.shape
    ((20, 3), (20, 6))
    """
    return _dataset("constant", K, T, B, rng)


def block_dataset(K, T, B=5, rng=None):
    """
    Gaussian data whose correlation matrix changes at B - 1 event boundaries.

    Parameters
    ----------
    K : int
        Number of features.
    T : int
        Number of timepoints.
    B : int
        Number of events (blocks of near-equal integer length).  Default: 5.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.1.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array in ``mat2vec`` layout.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.simulate import block_dataset
    >>> data, corrs = block_dataset(3, 20, B=4, rng=np.random.default_rng(0))
    >>> data.shape, corrs.shape
    ((20, 3), (20, 6))
    """
    return _dataset("block", K, T, B, rng)


def min_features(order, p=3):
    """
    Number of features needed to embed a p x p template at a given order.

    Parameters
    ----------
    order : int
        Order at which the template is embedded (positive integer).
    p : int
        Size of the template (at least 2).  Default: 3, the smallest size whose
        correlation matrix has more than one off-diagonal entry.

    Returns
    -------
    int
        ``p * 2**(order - 1)``.

    Examples
    --------
    >>> import timecorr as tc
    >>> tc.min_features(3), tc.min_features(4, p=4)
    (12, 32)
    """
    order = _check_positive_int("order", order)
    p = _check_positive_int("p", p)
    if p < 2:
        raise ValueError(f"p must be at least 2; got {p}.")
    return p * 2 ** (order - 1)


def _template_size(K, order):
    """p = K / 2**(order-1), validated."""
    m = 2 ** (order - 1)
    if K % m or K // m < 2:
        raise ValueError(
            f"order={order} needs K = p * 2**(order-1) features with a template of "
            f"p >= 2 features; got K={K}. min_features({order}) = "
            f"{min_features(order)} (p=3); min_features({order}, p) gives other sizes."
        )
    return K // m


_SINGLE_ENTRY = (
    "A template of p=2 features has a correlation matrix with a single entry; "
    "use p >= 3 (see min_features) for a pattern across entries."
)
_ORDER3 = (
    "The simulated structure exists only at order {order} (order >= 3). Variance-based "
    "reduction (for example PCA) may not retain it: in the higher-order tutorial's "
    "measurement at order 3 (20 seeds), PCA's event contrast was not distinguishable "
    "from its noise control (-0.00 versus 0.02, p = 0.20); orders above 3 were not "
    "measured. Analyze these data with delta_rfun=carrier_reducer(blocks) or without "
    "reduction (rfun=None)."
)


def _generate(datagen, p, order, S, T, B, amplitude_noise, rng):
    """
    S subjects sharing one template; returns (list of (T, p * m) arrays, truth).
    """
    if datagen not in _DATAGENS:
        raise ValueError(f"datagen must be one of {_DATAGENS}; got {datagen!r}.")
    R = _template(datagen, p, T, B, rng)
    m = 2 ** (order - 1)
    requested = _vectorize(R)
    truth = {
        "order": order,
        "corrs": requested.copy(),
        "requested": requested,
        "blocks": np.arange(p * m).reshape(p, m),
    }
    if order == 1:
        mean = 0.0 if datagen == "random" else 0.1
        return [_gaussian(R, mean, rng) for _ in range(S)], truth

    G, achieved = _latent_sign_correlation(R)
    corrs = _vectorize(achieved)
    # each block product is A_j * exp(amplitude_noise * z_j) with independent
    # z_j ~ N(0, 1), so its correlation with another product is the latent-sign
    # correlation times E[exp(a z)]**2 / E[exp(2 a z)] = exp(-amplitude_noise**2)
    corrs[:, p:] *= np.exp(-(amplitude_noise**2))
    truth["corrs"] = corrs
    A = np.sign(_gaussian(G, 0.0, rng))
    A[A == 0] = 1.0  # probability zero; keeps every value a sign

    data = []
    for _ in range(S):
        X = rng.choice([-1.0, 1.0], size=(T, p, m))  # free signs, per subject
        X[:, :, -1] = A * np.prod(X[:, :, :-1], axis=2)  # block product == A_j
        X = X.reshape(T, p * m)
        if amplitude_noise > 0:
            # per-sign SD amplitude_noise / sqrt(m): log|block product| has SD
            # amplitude_noise at every order
            X = X * np.exp(amplitude_noise / np.sqrt(m) * rng.standard_normal(X.shape))
        data.append(X)
    return data, truth


def _latent_sign_correlation(R):
    """
    Gaussian correlation G whose signs have correlation as close to R as possible.

    Signs of zero-mean Gaussians with correlation ``G`` have correlation
    ``(2 / pi) * arcsin(G)``, so ``G = sin(pi R / 2)`` reproduces ``R`` exactly
    when that matrix is positive semidefinite.  Otherwise (per timepoint) its
    eigenvalues are clipped at 1e-6 and it is rescaled to unit diagonal; the
    achieved sign correlation ``(2 / pi) * arcsin(G)`` then differs from ``R``.

    Returns ``(G, achieved)``, both (T, p, p); ``achieved`` is ``R`` itself at
    timepoints that need no clipping.
    """
    G = np.sin(np.pi * R / 2)
    w, v = np.linalg.eigh(G)
    clipped = w.min(axis=1) < 1e-6
    achieved = R.copy()
    if clipped.any():
        wc, vc = np.clip(w[clipped], 1e-6, None), v[clipped]
        Gc = (vc * wc[:, np.newaxis, :]) @ np.transpose(vc, (0, 2, 1))
        s = np.sqrt(np.einsum("tii->ti", Gc))
        Gc = Gc / (s[:, :, np.newaxis] * s[:, np.newaxis, :])
        Gc = (Gc + np.transpose(Gc, (0, 2, 1))) / 2
        idx = np.arange(Gc.shape[1])
        Gc[:, idx, idx] = 1.0
        G[clipped] = Gc
        Ac = (2 / np.pi) * np.arcsin(np.clip(Gc, -1.0, 1.0))
        Ac[:, idx, idx] = 1.0
        achieved[clipped] = Ac
    return G, achieved


def _check_common(S, T, amplitude_noise):
    S = _check_positive_int("S", S)
    T = _check_positive_int("T", T)
    if not np.isfinite(amplitude_noise) or amplitude_noise < 0:
        raise ValueError(
            f"amplitude_noise must be a non-negative number; got {amplitude_noise!r}."
        )
    return S, T, float(amplitude_noise)


def simulate_data(
    datagen="ramping",
    return_corrs=False,
    set_random_seed=False,
    S=1,
    T=100,
    K=10,
    B=5,
    order=1,
    amplitude_noise=0.2,
):
    """
    Simulate timeseries data with a known dynamic correlation matrix.

    With ``order=1`` the data are Gaussian, and the correlation matrix at
    timepoint ``t`` is the template ``R(t)``.  With ``order=n > 1`` the template
    is embedded at exactly order ``n`` by the sign-parity cascade: ``K`` must be
    ``p * 2**(n-1)``, the features form ``p`` blocks of ``2**(n-1)`` columns,
    and the order-``n`` correlation matrix among the ``p`` block products is the
    correlation of ``p`` latent signs.  That equals ``R(t)`` whenever
    ``sin(pi R(t) / 2)`` is positive semidefinite; otherwise the closest
    reachable matrix is used (see ``truth`` below).  This happens more often as
    ``p`` grows (for ``random_corrmat`` templates: rarely at p=3, mostly at p=6,
    almost always at p=10).  The products of proper subsets of a block have mean
    zero, so there is no structure at orders below ``n``.

    All subjects share the template (and, for ``order > 1``, the block layout and
    the latent signs ``A_j(t)``); everything else is drawn independently per
    subject.

    Parameters
    ----------
    datagen : str
        Template kind: "ramping" (moves linearly between two matrices),
        "block" (``B`` events of near-equal length), "constant", or "random"
        (an independent matrix at every timepoint).  Default: "ramping".
    return_corrs : bool
        If True, also return the ground truth.  Default: False.
    set_random_seed : bool or int
        False (default) draws unseeded; True uses seed 123; an integer is used as
        the seed.  A ``numpy.random.Generator`` is created from it; the global
        NumPy random state is not touched.
    S : int
        Number of subjects.  Default: 1.
    T : int
        Number of timepoints.  Default: 100.
    K : int
        Number of features.  For ``order > 1`` it must equal
        ``p * 2**(order-1)`` with ``p >= 2`` (see ``min_features``).  Default: 10.
    B : int
        Number of events for ``datagen="block"`` (positive integer; checked for
        every ``datagen``).  Default: 5.
    order : int
        Order at which the template is embedded (positive integer).  Default: 1.
    amplitude_noise : float
        For ``order > 1``: the standard deviation of the log-amplitude of each
        block's product, the same at every order.  Each of a block's
        ``m = 2**(order-1)`` signs is multiplied by
        ``exp(amplitude_noise / sqrt(m) * z)`` with independent ``z ~ N(0, 1)``,
        so the data are not binary and ``log|product|`` is
        ``N(0, amplitude_noise**2)``.  Ignored for ``order=1``.  Default: 0.2.

    Returns
    -------
    data : numpy.ndarray or list of numpy.ndarray
        A (T, K) array, or a list of S of them when ``S > 1``.
    corrs : numpy.ndarray or list of numpy.ndarray
        Only with ``return_corrs=True`` and ``order=1``: the template, a
        (T, K(K+1)/2) array in ``mat2vec`` layout (a list of S identical arrays
        when ``S > 1``).
    truth : dict or list of dict
        Only with ``return_corrs=True`` and ``order > 1`` (in place of
        ``corrs``): a dict (a list of S identical dicts when ``S > 1``) with

        - ``"order"``: the embedded order;
        - ``"corrs"``: (T, p(p+1)/2) correlation matrices, in ``mat2vec``
          layout, that the block products actually have.  The latent signs
          have correlation ``(2 / pi) * arcsin(G)`` for the Gaussian
          correlation ``G`` used (the requested template wherever no
          projection was needed); amplitude noise multiplies the
          off-diagonal entries by ``exp(-amplitude_noise**2)`` (0.96 at the
          default 0.2), and the diagonal is 1;
        - ``"requested"``: (T, p(p+1)/2) template that was asked for, same
          layout;
        - ``"blocks"``: (p, 2**(order-1)) int array of the column indices of
          each block.

    Raises
    ------
    ValueError
        If ``order``, ``S``, ``T`` or ``B`` is not a positive integer, ``datagen``
        is unknown, ``amplitude_noise`` is negative, or ``K`` does not fit the
        order (the message gives ``min_features(order)``).
    TypeError
        If ``set_random_seed`` is not a bool or an integer.

    Warns
    -----
    UserWarning
        When the template has ``p = 2`` features (its correlation matrix has a
        single entry), and when ``order >= 3``, because variance-based
        reduction such as PCA may not retain structure that exists only at
        order >= 3 (use ``carrier_reducer`` or ``rfun=None``).  Measured in
        the higher-order tutorial (lag-matched event contrast, 20 seeds, PCA
        on both paths): structure embedded at order 2 is partly kept (0.21,
        against 0.08 for noise and 0.47 with ``carrier_reducer``), so no
        warning is issued at order 2; for structure embedded at order 3,
        PCA's contrast was not distinguishable from its noise control (-0.00
        versus 0.02, Welch p = 0.20; 0.48 with ``carrier_reducer``).  Orders
        above 3 were not measured.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, corrs = tc.simulate_data(datagen="block", return_corrs=True,
    ...                                set_random_seed=1, T=50, K=4)
    >>> data.shape, corrs.shape
    ((50, 4), (50, 10))
    >>> data, truth = tc.simulate_data(return_corrs=True, set_random_seed=1,
    ...                                T=50, K=8, order=2)
    >>> data.shape, truth["corrs"].shape, truth["blocks"].tolist()
    ((50, 8), (50, 10), [[0, 1], [2, 3], [4, 5], [6, 7]])
    """
    if (
        not isinstance(order, numbers.Integral)
        or isinstance(order, bool)
        or order < 1
    ):
        raise ValueError(f"order must be a positive integer; got {order!r}.")
    order = int(order)
    if datagen not in _DATAGENS:
        raise ValueError(f"datagen must be one of {_DATAGENS}; got {datagen!r}.")
    S, T, amplitude_noise = _check_common(S, T, amplitude_noise)
    B = _check_positive_int("B", B)
    K = _check_positive_int("K", K)
    p = _template_size(K, order)
    rng = _make_rng(set_random_seed)

    if p == 2:
        warnings.warn(_SINGLE_ENTRY, UserWarning, stacklevel=2)
    if order >= 3:
        warnings.warn(_ORDER3.format(order=order), UserWarning, stacklevel=2)

    data, truth = _generate(datagen, p, order, S, T, B, amplitude_noise, rng)
    if order == 1:
        truth = truth["corrs"]
        truths = [truth] + [truth.copy() for _ in range(S - 1)]
    else:
        truths = [truth] + [
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in truth.items()}
            for _ in range(S - 1)
        ]

    if S == 1:
        data, truths = data[0], truths[0]
    if return_corrs:
        return data, truths
    return data


def simulate_multi_order(
    orders,
    p=4,
    datagen="ramping",
    set_random_seed=False,
    S=1,
    T=100,
    B=5,
    amplitude_noise=0.2,
):
    """
    Simulate data with templates embedded at several orders in disjoint features.

    Each entry of ``orders`` gets its own group of ``p * 2**(order-1)`` columns,
    generated as by ``simulate_data`` with that order (order 1: Gaussian),
    with its own template.  Groups are placed side by side in the order given.

    Parameters
    ----------
    orders : sequence of int
        The order of each group (positive integers).
    p : int
        Template size of every group (at least 2).  Default: 4.
    datagen, set_random_seed, S, T, B, amplitude_noise
        As in ``simulate_data``; one random generator is used for all groups.
        ``amplitude_noise`` is applied per group: in a group of order ``n`` each
        sign's log-amplitude has standard deviation
        ``amplitude_noise / sqrt(2**(n-1))``, so every group's block products
        have log-amplitude standard deviation ``amplitude_noise``.

    Returns
    -------
    data : numpy.ndarray or list of numpy.ndarray
        A (T, sum of the group widths) array, or a list of S of them when
        ``S > 1``.
    truths : list of dict
        One dict per group, in the order of ``orders``, with the keys of
        ``simulate_data``'s ``truth``: ``"order"``, ``"corrs"`` (the
        correlation the data carry, in mat2vec layout: for order > 1 the
        achieved latent-sign correlation with off-diagonal entries multiplied
        by ``exp(-amplitude_noise**2)``), ``"requested"`` (the template asked
        for; equal to ``"corrs"`` for order 1) and ``"blocks"`` (column
        indices in ``data``).  Shared by all subjects.

    Raises
    ------
    ValueError
        If ``orders`` is empty or holds a non-positive or non-integer order, or
        for the reasons listed in ``simulate_data``.

    Warns
    -----
    UserWarning
        As in ``simulate_data``, at most once each: for ``p = 2`` and when any
        order is 3 or more.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truths = tc.simulate_multi_order([1, 2], p=3, set_random_seed=0, T=40)
    >>> data.shape, [t["blocks"].tolist() for t in truths]
    ((40, 9), [[[0], [1], [2]], [[3, 4], [5, 6], [7, 8]]])
    """
    orders = list(orders)
    if not orders:
        raise ValueError("orders must hold at least one order.")
    for n in orders:
        if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
            raise ValueError(f"orders must be positive integers; got {n!r}.")
    if datagen not in _DATAGENS:
        raise ValueError(f"datagen must be one of {_DATAGENS}; got {datagen!r}.")
    S, T, amplitude_noise = _check_common(S, T, amplitude_noise)
    B = _check_positive_int("B", B)
    min_features(1, p)  # validates p
    p = int(p)
    rng = _make_rng(set_random_seed)

    if p == 2:
        warnings.warn(_SINGLE_ENTRY, UserWarning, stacklevel=2)
    if max(orders) >= 3:
        warnings.warn(_ORDER3.format(order=max(orders)), UserWarning, stacklevel=2)

    groups, truths, offset = [], [], 0
    for n in orders:
        group, truth = _generate(datagen, p, int(n), S, T, B, amplitude_noise, rng)
        truth["blocks"] = truth["blocks"] + offset
        offset += group[0].shape[1]
        groups.append(group)
        truths.append(truth)

    data = [np.hstack([g[s] for g in groups]) for s in range(S)]
    if S == 1:
        data = data[0]
    return data, truths


def nested_template(corrs):
    """
    Order-2 correlations implied by Gaussian order-1 correlations.

    For zero-mean Gaussian features with correlation matrix ``R``, the
    co-fluctuations ``z_i z_j`` (``i <= j``, in ``mat2vec`` order: squares first,
    then the upper triangle) have correlations

    ``(R_ik R_jl + R_il R_jk) / sqrt((1 + R_ij**2) (1 + R_kl**2))``

    (Isserlis' theorem).  This is the order-2 truth for order-1 simulated data
    whose means are removed exactly, as the delta path's z-scoring does for a
    constant template.

    Parameters
    ----------
    corrs : numpy.ndarray
        A K(K+1)/2 vector, or a (T, K(K+1)/2) array, of correlation matrices in
        ``mat2vec`` layout (for example the ``corrs`` from ``simulate_data``).

    Returns
    -------
    numpy.ndarray
        A V(V+1)/2 vector, or a (T, V(V+1)/2) array, with ``V = K(K+1)/2``: the
        correlation matrix among the co-fluctuations, in ``mat2vec`` layout.

    Raises
    ------
    ValueError
        If ``corrs`` is not 1- or 2-dimensional or its width is not K(K+1)/2.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> R = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> np.round(tc.vec2mat(tc.nested_template(tc.mat2vec(R))), 3)
    array([[1.   , 0.25 , 0.632],
           [0.25 , 1.   , 0.632],
           [0.632, 0.632, 1.   ]])
    """
    corrs = np.asarray(corrs, dtype=float)
    single = corrs.ndim == 1
    if corrs.ndim not in (1, 2):
        raise ValueError("corrs must be a vector or a (T, K(K+1)/2) array.")
    R = vec2mat(corrs[np.newaxis] if single else corrs)  # (K, K, T)
    R = np.transpose(R, (2, 0, 1))  # (T, K, K)
    K = R.shape[1]
    iu = np.triu_indices(K, 1)
    I = np.concatenate([np.arange(K), iu[0]])
    J = np.concatenate([np.arange(K), iu[1]])
    a, b = I[:, np.newaxis], I[np.newaxis, :]
    c, d = J[:, np.newaxis], J[np.newaxis, :]
    num = R[:, a, b] * R[:, c, d] + R[:, a, d] * R[:, c, b]
    var = R[:, I, I] * R[:, J, J] + R[:, I, J] ** 2
    out = num / np.sqrt(var[:, :, np.newaxis] * var[:, np.newaxis, :])
    out = _vectorize(out)
    return out[0] if single else out


def _check_blocks(blocks):
    blocks = np.asarray(blocks)
    if blocks.ndim != 2 or blocks.size == 0 or blocks.dtype.kind not in "iu":
        raise ValueError("blocks must be a 2-D integer array (p blocks x m columns).")
    p, m = blocks.shape
    if m & (m - 1):
        raise ValueError(f"blocks must have a power-of-2 number of columns; got {m}.")
    if blocks.min() < 0 or len(np.unique(blocks)) != blocks.size:
        raise ValueError("blocks must hold distinct non-negative column indices.")
    return blocks.astype(int)


def _pair_column(d, a, b):
    """Column of the product of inputs a and b in ``mat2vec`` layout of d inputs."""
    a, b = min(a, b), max(a, b)
    return d + d * a - a * (a + 1) // 2 + (b - a - 1)


def _pair_up(idx, d):
    """Adjacent-pair product columns, per block, of carriers ``idx`` among d inputs."""
    return np.array(
        [
            [_pair_column(d, row[2 * q], row[2 * q + 1]) for q in range(len(row) // 2)]
            for row in idx
        ],
        dtype=int,
    )


def _check_K(blocks, K):
    if K is None:
        return int(blocks.max()) + 1
    K = _check_positive_int("K", K)
    if K <= blocks.max():
        raise ValueError(
            f"K={K} is too small for blocks that reach column {int(blocks.max())}."
        )
    return K


def carrier_columns(blocks, level, K):
    """
    Columns of each block's carriers in unreduced delta features.

    Order-0 carriers are a block's own columns.  Order-``k`` carriers are the
    products of adjacent pairs of order-``(k-1)`` carriers.  For a run without
    reduction (``tc.features(..., rfun=None, return_delta=True)``), this gives
    the columns of ``D[level]`` that hold them.  At ``level = log2(m)`` each
    block has one carrier, whose series is the block's product (of z-scored
    lower carriers).

    Parameters
    ----------
    blocks : array-like of int
        (p, m) column indices of each block in the data, as in
        ``truth["blocks"]``; ``m`` is a power of 2.
    level : int
        Carrier order, from 0 to ``log2(m)``.
    K : int
        Number of columns in the data.

    Returns
    -------
    numpy.ndarray
        A (p, m / 2**level) int array of column indices in ``D[level]``.

    Raises
    ------
    ValueError
        If ``level`` is out of range, ``K`` is too small for ``blocks``, or
        ``blocks`` is malformed.

    Examples
    --------
    >>> import timecorr as tc
    >>> tc.carrier_columns([[0, 1], [2, 3]], 1, 4).tolist()
    [[4], [9]]
    """
    blocks = _check_blocks(blocks)
    K = _check_K(blocks, K)
    m = blocks.shape[1]
    top = m.bit_length() - 1
    if (
        not isinstance(level, numbers.Integral)
        or isinstance(level, bool)
        or not 0 <= level <= top
    ):
        raise ValueError(f"level must be an integer from 0 to {top}; got {level!r}.")
    idx, d = blocks, K
    for _ in range(int(level)):
        idx = _pair_up(idx, d)
        d = d * (d + 1) // 2
    return idx


def carrier_reducer(blocks, K=None):
    """
    Build a delta-path reducer that keeps only the block carriers.

    Pass the result as ``delta_rfun`` (with ``rfun=None``) to
    ``timecorr.features`` for data from ``simulate_data(order=n)``.  At each
    order the reducer keeps the products of adjacent pairs of the previous
    order's carriers, block by block, so order ``k`` sees ``p * m / 2**k``
    columns and the cost is linear in ``K``.  At order ``n`` the analysis path
    then gives the vectorized p x p correlation matrix among the block products.

    The reducer holds no state: it infers the order from its input width
    (``K(K+1)/2`` at order 1, then ``c(c+1)/2`` for the ``c`` carriers kept
    at the order before).  At order ``n``, where each block has a single carrier,
    it returns its input unchanged.  It is meant for runs up to order ``n``:
    beyond it the widths no longer identify the order (they usually match none
    and raise, but can coincide with an earlier order's width).

    Parameters
    ----------
    blocks : array-like of int
        (p, m) column indices of each block in the data, as in
        ``truth["blocks"]``; ``m`` is a power of 2.
    K : int, optional
        Number of columns in the data.  Default: ``blocks.max() + 1`` (the data
        are exactly the blocks, as ``simulate_data`` returns them); pass the full
        width for data from ``simulate_multi_order``.

    Returns
    -------
    callable
        ``f(list of (T, w) arrays) -> list of arrays``, the form accepted by
        ``rfun`` and ``delta_rfun``.  It raises ``ValueError`` naming the
        expected widths when ``w`` matches none of them.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_data(return_corrs=True, set_random_seed=0,
    ...                                T=60, K=8, order=2)
    >>> X, D = tc.features(data, order=2, cfun=tc.autofc, rfun=None,
    ...                    delta_rfun=tc.carrier_reducer(truth["blocks"]),
    ...                    return_delta=True, weights_params={"var": 25})
    >>> D[1].shape, X[2].shape
    ((60, 4), (60, 10))
    """
    blocks = _check_blocks(blocks)
    K = _check_K(blocks, K)
    p, m = blocks.shape
    n = m.bit_length() - 1 + 1  # embedded order

    # input width -> columns to keep (None: pass through)
    plan = {}
    idx, d = blocks, K
    for level in range(1, n):
        plan[d * (d + 1) // 2] = _pair_up(idx, d).ravel()
        c = p * m // 2**level
        idx, d = np.arange(c).reshape(p, c // p), c
    plan.setdefault(d * (d + 1) // 2, None)

    def reducer(arrays):
        single = not isinstance(arrays, (list, tuple))
        arrays = [arrays] if single else list(arrays)
        out = []
        for x in arrays:
            x = np.asarray(x)
            w = x.shape[1]
            if w not in plan:
                raise ValueError(
                    f"carrier_reducer: input has {w} columns, which matches no order "
                    f"of these blocks (expected one of {sorted(plan, reverse=True)} "
                    f"columns, for orders 1 to {n})."
                )
            keep = plan[w]
            out.append(x if keep is None else x[:, keep])
        return out[0] if single else out

    return reducer
