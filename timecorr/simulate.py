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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.stats import norm

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

    Raises
    ------
    ValueError
        If ``K`` is not a positive integer.

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
        Default: 5.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.1.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array: each timepoint's correlation matrix in ``mat2vec``
        layout.

    Raises
    ------
    ValueError
        If ``K`` is not a positive integer.

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
        Default: 5.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array in ``mat2vec`` layout.

    Raises
    ------
    ValueError
        If ``K`` is not a positive integer.

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
        Default: 5.
    rng : numpy.random.Generator, optional
        Source of random numbers.  Default: a new unseeded generator.

    Returns
    -------
    data : numpy.ndarray
        A (T, K) array with mean 0.1.
    corrs : numpy.ndarray
        A (T, K(K+1)/2) array in ``mat2vec`` layout (identical rows).

    Raises
    ------
    ValueError
        If ``K`` is not a positive integer.

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

    Raises
    ------
    ValueError
        If ``K`` or ``B`` is not a positive integer.

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

    Raises
    ------
    ValueError
        If ``order`` is not a positive integer or ``p`` is not an integer of at
        least 2.

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
    if order == 1 and K < 2:
        raise ValueError(
            f"order=1 needs K >= 2 features (a template of at least 2 features); "
            f"got K={K}. min_features(1) = 3 gives a template whose correlation "
            "matrix has more than one entry."
        )
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


def _generate(datagen, p, order, S, T, B, amplitude_noise, rng, subject_noise=None):
    """
    S subjects sharing one template; returns (list of (T, p * m) arrays, truth).

    With ``subject_noise`` (order 1 only) the subjects share one draw from the
    template and each adds independent Gaussian noise whose SD is
    ``subject_noise`` times each feature's SD in that draw.
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
        if subject_noise is None:
            return [_gaussian(R, mean, rng) for _ in range(S)], truth
        shared = _gaussian(R, mean, rng)
        sd = subject_noise * shared.std(axis=0)
        return [shared + sd * rng.standard_normal(shared.shape) for _ in range(S)], truth

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
    subject_noise=None,
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
    subject.  For ``order=1`` that means the subjects' samples are independent
    draws from the same template, so there is no signal shared across
    subjects for ISFC to find; ``subject_noise`` gives them one.

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
    subject_noise : float or None
        For ``order=1`` only.  None (default): each subject's samples are an
        independent draw from the template.  A non-negative number: all
        subjects share one draw from the template, and each adds independent
        Gaussian noise whose standard deviation is ``subject_noise`` times
        each feature's standard deviation in the shared draw.  A feature's
        correlation across subjects is then about
        ``1 / (1 + subject_noise**2)`` (0.8 for 0.5), and within a subject
        the template's off-diagonal correlations are divided by about
        ``1 + subject_noise**2`` (``corrs`` is the template, the correlation
        of the shared draw).

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
        is unknown, ``amplitude_noise`` is negative, ``subject_noise`` is not
        None or a non-negative number (or is given with ``order > 1``), or
        ``K`` does not fit the order (the message gives
        ``min_features(order)``).
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
    >>> import numpy as np
    >>> subjects = tc.simulate_data(S=4, T=500, K=3, set_random_seed=0,
    ...                             subject_noise=0.5)
    >>> r = np.corrcoef(subjects[0][:, 0], subjects[1][:, 0])[0, 1]
    >>> bool(0.7 < r < 0.9)  # about 1 / (1 + 0.5**2) = 0.8
    True
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
    if subject_noise is not None:
        if (
            not isinstance(subject_noise, numbers.Real)
            or isinstance(subject_noise, bool)
            or not np.isfinite(subject_noise)
            or subject_noise < 0
        ):
            raise ValueError(
                "subject_noise must be None or a non-negative number; got "
                f"{subject_noise!r}."
            )
        if order > 1:
            raise ValueError(
                "subject_noise applies to order=1 only (for order > 1 the "
                f"subjects already share the latent signs); got order={order}."
            )
        subject_noise = float(subject_noise)
    B = _check_positive_int("B", B)
    K = _check_positive_int("K", K)
    p = _template_size(K, order)
    rng = _make_rng(set_random_seed)

    if p == 2:
        warnings.warn(_SINGLE_ENTRY, UserWarning, stacklevel=2)
    if order >= 3:
        warnings.warn(_ORDER3.format(order=order), UserWarning, stacklevel=2)

    data, truth = _generate(
        datagen, p, order, S, T, B, amplitude_noise, rng, subject_noise
    )
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


def carrier_columns(blocks, level, K=None):
    """
    Columns of each block's carriers in unreduced delta features.

    Order-0 carriers are a block's own columns.  Order-``k`` carriers are the
    products of adjacent pairs of order-``(k-1)`` carriers.  For a run without
    reduction with co-fluctuations on the delta path (``tc.features(...,
    rfun=None, delta_rfun=None, delta_weights_function=tc.eye_weights,
    return_delta=True)``), this gives the columns of ``D[level]`` that hold
    them.  At ``level = log2(m)`` each
    block has one carrier, whose series is the block's product (of z-scored
    lower carriers).

    Parameters
    ----------
    blocks : array-like of int
        (p, m) column indices of each block in the data, as in
        ``truth["blocks"]``; ``m`` is a power of 2.
    level : int
        Carrier order, from 0 to ``log2(m)``.
    K : int, optional
        Number of columns in the data.  Default: ``blocks.max() + 1`` (the data
        are exactly the blocks, as ``simulate_data`` returns them), as in
        ``carrier_reducer``; pass the full width for data from
        ``simulate_multi_order``.

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

    Pass the result as ``delta_rfun`` (with ``rfun=None`` and
    ``delta_weights_function=tc.eye_weights``, the original pipeline's
    co-fluctuations, whose products the carriers are) to
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

    Raises
    ------
    ValueError
        If ``blocks`` is not a 2-D array of distinct non-negative integers with
        a power-of-2 number of columns, or ``K`` is not a positive integer
        larger than every index in ``blocks``.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_data(return_corrs=True, set_random_seed=0,
    ...                                T=60, K=8, order=2)
    >>> X, D = tc.features(data, order=2, cfun=tc.autofc, rfun=None,
    ...                    delta_rfun=tc.carrier_reducer(truth["blocks"]),
    ...                    delta_weights_function=tc.eye_weights,
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


def _smooth(rng, T, n, tau):
    """``n`` standardized Gaussian-smoothed white-noise series of length ``T``.

    White noise of length ``T + 8 * tau`` is filtered with a Gaussian of
    standard deviation ``tau``, the first and last ``4 * tau`` samples are
    dropped, and each column is standardized to mean 0 and variance 1.
    """
    tau = int(tau)
    x = gaussian_filter1d(rng.standard_normal((T + 8 * tau, n)), tau, axis=0)
    x = x[4 * tau : len(x) - 4 * tau]
    return (x - x.mean(0)) / x.std(0)


def simulate_nested(
    S=2,
    T=1000,
    n_supergroups=6,
    modules_per_supergroup=3,
    nodes_per_module=6,
    coupling="sustained",
    shared="order2",
    tau2=5,
    tau3=None,
    flip=40,
    duty=0.1,
    global_signal=0.0,
    set_random_seed=None,
):
    """
    Simulate nodes in modules in supergroups with nested time-varying coupling.

    Each supergroup has a latent strength ``h`` (order 3), each module a
    coupling ``g`` whose time course depends on its supergroup's ``h`` (order
    2), and each node's activity is driven by its module's coupling (order 1).
    With ``smooth(tau)`` a unit-variance Gaussian white noise series smoothed
    with a Gaussian of standard deviation ``tau`` and standardized, and
    ``Phi`` the standard normal CDF, for each subject:

    - ``h_s(t) = 0.05 + 0.9 Phi(1.5 smooth(tau3))``, one per supergroup ``s``.
    - ``a_m(t) = sqrt(h_s) u_s(t) + sqrt(1 - h_s) v_m(t)`` for module ``m`` in
      supergroup ``s``, where ``u`` and ``v`` are ``smooth(tau2)`` series.
    - ``g_m(t) = Phi(a_m(t))``; for ``coupling="transient"``,
      ``g_m(t) = Phi((a_m(t) - Phi^-1(1 - duty)) / 0.25)``.
    - ``x_i(t) = sqrt(g_m) e_i(t) f_m(t) + sqrt(1 - g_m) z_i(t) + global_signal * q(t)``
      for node ``i`` in module ``m``, where ``f``, ``z`` and ``q`` are white
      standard normal noise and ``e_i`` is a fixed random sign per node.  For
      ``coupling="signflip"``, ``e_i(t)`` is a random sign redrawn every
      ``flip`` samples, with a random module-specific phase.

    Parameters
    ----------
    S : int
        Number of subjects.  Default: 2.
    T : int
        Number of timepoints.  The smoothing needs ``T >= 8 * max(tau2,
        tau3)``.  Default: 1000.
    n_supergroups : int
        Number of supergroups.  Default: 6.
    modules_per_supergroup : int
        Number of modules in each supergroup.  Default: 3.
    nodes_per_module : int
        Number of nodes in each module.  Default: 6.
    coupling : {"sustained", "transient", "signflip"}
        ``"sustained"``: nodes in a module share ``f`` with fixed signs.
        ``"transient"``: ``g`` is near 0 except for roughly a fraction
        ``duty`` of the time.  ``"signflip"``: the signs ``e_i`` change over
        time, so the module's coupling averages to zero over the recording.
        Default: ``"sustained"``.
    shared : {"order2", "order1"}
        ``"order2"``: subjects share ``h`` only.  ``"order1"``: subjects
        share ``h``, ``u`` and ``v`` (so ``a`` and ``g``); ``f``, ``z``,
        ``e`` and ``q`` are always private to each subject.  Default:
        ``"order2"``.
    tau2 : int
        Standard deviation, in samples, of the Gaussian smoothing of ``u`` and
        ``v``.  Default: 5.
    tau3 : int or None
        Standard deviation, in samples, of the smoothing of ``h``.  ``None``
        means ``T // 8``.  Default: None.
    flip : int
        Number of samples between sign redraws for ``coupling="signflip"``.
        Default: 40.
    duty : float
        Fraction of the time a module is coupled for ``coupling="transient"``;
        in the open interval (0, 1).  Default: 0.1.
    global_signal : float
        Weight of ``q``, a series added to every node of a subject.
        Default: 0.0.
    set_random_seed : bool, int or None
        ``None`` or ``False``: a fresh random seed.  ``True``: seed 123.  An
        integer: that seed.  NumPy's global random state is not used.

    Returns
    -------
    data : list of numpy.ndarray
        ``S`` arrays of shape ``(T, K)``, with ``K = n_supergroups *
        modules_per_supergroup * nodes_per_module``.
    truth : dict
        ``"module"``: ``(K,)`` module index of each node.  ``"supergroup"``:
        ``(K,)`` supergroup index of each node.  ``"h"``: ``(T,
        n_supergroups)``.  ``"g"``: a list of ``S`` arrays of shape ``(T,
        n_supergroups * modules_per_supergroup)``.

    Raises
    ------
    ValueError
        For an unknown ``coupling`` or ``shared``, a size that is not a
        positive integer, ``duty`` outside (0, 1), a non-numeric
        ``global_signal``, or ``T < 8 * max(tau2, tau3)``.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_nested(S=2, T=200, set_random_seed=0)
    >>> len(data), data[0].shape
    (2, (200, 108))
    >>> truth["h"].shape, truth["g"][0].shape
    ((200, 6), (200, 18))
    """
    S = _check_positive_int("S", S)
    T = _check_positive_int("T", T)
    n_sg = _check_positive_int("n_supergroups", n_supergroups)
    per_sg = _check_positive_int("modules_per_supergroup", modules_per_supergroup)
    per_mod = _check_positive_int("nodes_per_module", nodes_per_module)
    tau2 = _check_positive_int("tau2", tau2)
    flip = _check_positive_int("flip", flip)
    if coupling not in ("sustained", "transient", "signflip"):
        raise ValueError(
            "coupling must be 'sustained', 'transient' or 'signflip'; "
            f"got {coupling!r}."
        )
    if shared not in ("order2", "order1"):
        raise ValueError(f"shared must be 'order2' or 'order1'; got {shared!r}.")
    if not isinstance(duty, numbers.Real) or not 0 < duty < 1:
        raise ValueError(f"duty must be in the open interval (0, 1); got {duty!r}.")
    if not isinstance(global_signal, numbers.Real) or isinstance(global_signal, bool):
        raise ValueError(f"global_signal must be a number; got {global_signal!r}.")
    tau3 = max(T // 8, 1) if tau3 is None else _check_positive_int("tau3", tau3)
    if T < 8 * max(tau2, tau3):
        raise ValueError(
            f"T must be at least 8 * max(tau2, tau3) = {8 * max(tau2, tau3)} for "
            f"the smoothing; got T={T}."
        )
    rng = _make_rng(set_random_seed)

    n_mod = n_sg * per_sg
    K = n_mod * per_mod
    module = np.repeat(np.arange(n_mod), per_mod)
    supergroup = module // per_sg
    sg_of_module = np.repeat(np.arange(n_sg), per_sg)

    h = norm.cdf(1.5 * _smooth(rng, T, n_sg, tau3)) * 0.9 + 0.05
    if shared == "order1":
        uv = (_smooth(rng, T, n_sg, tau2), _smooth(rng, T, n_mod, tau2))

    data, gs = [], []
    for _ in range(S):
        u, v = uv if shared == "order1" else (
            _smooth(rng, T, n_sg, tau2),
            _smooth(rng, T, n_mod, tau2),
        )
        hm = h[:, sg_of_module]
        a = np.sqrt(hm) * u[:, sg_of_module] + np.sqrt(1 - hm) * v
        if coupling == "transient":
            g = norm.cdf((a - norm.ppf(1 - duty)) / 0.25)
        else:
            g = norm.cdf(a)
        if coupling == "signflip":
            phase = rng.integers(0, flip, n_mod)
            block = (np.arange(T)[:, None] + phase[None, :]) // flip
            signs = rng.choice([-1.0, 1.0], (block.max() + 1, K))
            e = signs[block[:, module], np.arange(K)[None, :]]
        else:
            e = np.tile(rng.choice([-1.0, 1.0], K), (T, 1))
        f = rng.standard_normal((T, n_mod))
        z = rng.standard_normal((T, K))
        q = rng.standard_normal((T, 1))
        gk = g[:, module]
        data.append(
            np.sqrt(gk) * e * f[:, module] + np.sqrt(1 - gk) * z + global_signal * q
        )
        gs.append(g)
    return data, {"module": module, "supergroup": supergroup, "h": h, "g": gs}


def _is_int_sequence(value):
    """True for a list, a tuple or a 1-D numpy array (the accepted sequences)."""
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, (list, tuple))


def _check_int_sequence(name, value):
    """A tuple of positive integers from a list, tuple or 1-D numpy array
    (ValueError otherwise)."""
    if not _is_int_sequence(value):
        raise ValueError(
            f"{name} must be a sequence of positive integers (a list, tuple or "
            f"1-D numpy array); got {value!r}."
        )
    return tuple(_check_positive_int(f"{name}[{i}]", v) for i, v in enumerate(value))


def _hierarchy_signals(rng, T, counts, branching, taus, top=None):
    """
    ``[s_1, ..., s_n]`` of one subject (see ``simulate_hierarchy``):
    ``counts[l-1]`` level-l groups, ``top`` the level-n signals when given.
    """
    n = len(taus)
    s = [None] * n
    if top is None:
        top = 0.05 + 0.9 * norm.cdf(1.5 * _smooth(rng, T, counts[n - 1], taus[n - 1]))
    s[n - 1] = top
    for l in range(n - 1, 0, -1):  # s_l from s_{l+1}
        parent = np.repeat(np.arange(counts[l]), branching[l - 1])
        sp = s[l][:, parent]
        u = _smooth(rng, T, counts[l], taus[l - 1])[:, parent]
        v = _smooth(rng, T, counts[l - 1], taus[l - 1])
        a = np.sqrt(sp) * u + np.sqrt(1 - sp) * v
        s[l - 1] = norm.cdf(a) if l == 1 else 0.05 + 0.9 * norm.cdf(1.5 * a)
    return s


def simulate_hierarchy(
    S=2,
    T=6000,
    nodes_per_module=6,
    branching=(4, 4),
    n_top=4,
    taus=(5, 50, 500),
    shared="top",
    set_random_seed=None,
    signs="private",
    factors="private",
    volatility=None,
    volatility_tau=50,
):
    """
    Simulate nodes in nested groups of any depth with time-varying coupling.

    A level-1 group is a module of ``nodes_per_module`` nodes; a
    level-``(l+1)`` group holds ``branching[l-1]`` level-``l`` groups; there
    are ``n_top`` level-``n`` groups, with ``n = len(taus)``.  Each level-``l``
    group has a signal ``s_l(t)`` that sets how strongly its children are
    coupled.  With ``smooth(tau)`` a unit-variance Gaussian white noise series
    smoothed with a Gaussian of standard deviation ``tau`` and standardized,
    and ``Phi`` the standard normal CDF, for each subject:

    - ``s_n(t) = 0.05 + 0.9 Phi(1.5 smooth(taus[n-1]))``, one per top group.
    - For ``l < n``, group ``m`` at level ``l`` with parent ``p`` (at level
      ``l + 1``): ``a_m(t) = sqrt(s_{l+1,p}) u_p(t) + sqrt(1 - s_{l+1,p})
      v_m(t)``, where ``u`` (one per parent) and ``v`` (one per level-``l``
      group) are ``smooth(taus[l-1])`` series; ``s_1 = Phi(a)`` and, for
      ``1 < l < n``, ``s_l = 0.05 + 0.9 Phi(1.5 a)``.
    - ``x_i(t) = sqrt(g) e_i f_m(t) + sqrt(1 - g) z_i(t)`` for node ``i`` in
      module ``m``, with ``g = s_{1,m}(t)``, ``f`` and ``z`` white standard
      normal noise and ``e_i`` a fixed random sign per node (drawn per
      subject, or once for every subject with ``signs="shared"``).  ``f``
      (one series per module) is drawn per subject, or once for every
      subject with ``factors="shared"``; ``z`` is always private.
    - With ``volatility`` given, every ``x_i(t)`` of every subject is
      multiplied by one positive series ``v(t)``, with ``log v(t)`` a
      ``smooth(volatility_tau)`` series times ``volatility`` plus a constant
      that makes the mean of ``v**2`` equal to 1.

    Siblings at level ``l`` share ``u`` of their parent with weight
    ``sqrt(s_{l+1})``, so their latent series ``a`` are correlated by about
    ``s_{l+1}`` over windows short compared with ``s_{l+1}``'s timescale;
    each node has variance 1 whatever the signals (without ``volatility``;
    with it, variance ``v(t)**2`` at time ``t``, 1 on average, and the
    correlations between nodes at each time are unchanged).  With ``n = 2`` this is the
    sustained coupling of ``simulate_nested`` (``s_2`` is ``h``, ``s_1`` is
    ``g``).  The model assumes that higher levels vary more slowly
    (increasing ``taus``); this is not checked.

    Parameters
    ----------
    S : int
        Number of subjects.  Default: 2.
    T : int
        Number of timepoints; at least ``8 * max(taus)`` for the smoothing.
        Default: 6000.
    nodes_per_module : int
        Number of nodes in each module (level-1 group).  Default: 6.
    branching : sequence of int
        ``branching[l-1]`` is the number of level-``l`` groups in each
        level-``(l+1)`` group; ``len(branching) == len(taus) - 1`` (empty for
        one level).  A list, tuple or 1-D numpy array (other types, sets and
        strings included, raise ``ValueError``).  Default: (4, 4).
    n_top : int
        Number of level-``n`` groups.  Default: 4.
    taus : sequence of int
        ``taus[l-1]`` is the standard deviation, in samples, of the smoothing
        of the level-``l`` series (``u`` and ``v`` for ``l < n``, ``s_n`` for
        ``l = n``); its length is the depth ``n >= 1``.  A list, tuple or
        1-D numpy array, as for ``branching``.  Default: (5, 50, 500).
    shared : {"top", "all", "none"}
        ``"top"``: subjects share ``s_n`` only (every ``u`` and ``v`` is
        drawn per subject).  ``"all"``: subjects share every ``s_l`` (all
        ``u`` and ``v``).  ``"none"``: nothing is shared.  ``f``, ``z`` and
        ``e`` are always private to each subject (see ``signs`` for
        ``e``).  Default: ``"top"``.
    set_random_seed : bool, int or None
        ``None`` or ``False``: a fresh random seed.  ``True``: seed 123.  An
        integer: that seed.  NumPy's global random state is not used.
    signs : {"private", "shared"}
        ``"private"``: each subject draws its own node signs ``e`` (the
        output is the same as without this option).  ``"shared"``: one set
        of node signs, drawn after the shared signals and before the
        subjects' series, is used by every subject.  The shared signs are
        drawn from the same generator, so at the same seed every draw after
        them (each subject's private ``u``, ``v``, ``f`` and ``z``) differs
        from the ``"private"`` output: only the shared signals are the
        same, and the two settings are not paired draws of the subjects'
        series.  Default: ``"private"``.
    factors : {"private", "shared"}
        ``"private"``: each subject draws its own module series ``f``, so
        no time series is shared between subjects (the output is the same
        as without this option).  ``"shared"``: one set of module series,
        drawn after the shared signals and signs and before the subjects'
        series, is used by every subject, so nodes of one module are
        correlated across subjects: for node ``i`` of subject ``a`` and
        node ``j`` of subject ``b`` in module ``m``, the correlation at
        time ``t`` is ``e_i e_j sqrt(g_a(t) g_b(t))``, with ``g`` each
        subject's ``s_{1,m}`` and ``e`` each subject's signs (use
        ``signs="shared"`` for the same sign in every subject).  As for
        ``signs``, every draw after the shared series differs from the
        ``"private"`` output at the same seed.  Default: ``"private"``.
    volatility : float or None
        ``None``: no multiplier (the output is the same as without this
        option).  A positive number: the standard deviation of the
        logarithm of a multiplier ``v(t)`` common to every subject and
        node (volatility clustering shared by all series).  ``v`` is drawn
        after everything else, so at the same seed the data are exactly
        those without ``volatility`` times ``v`` and every other entry of
        ``truth`` is the same.  Default: ``None``.
    volatility_tau : int
        Standard deviation, in samples, of the Gaussian smoothing of
        ``log v``: its autocorrelation at lag ``k`` is
        ``exp(-k**2 / (4 * volatility_tau**2))``.  With ``volatility``
        given, ``T`` must be at least ``8 * volatility_tau``.  Default: 50.

    Returns
    -------
    data : list of numpy.ndarray
        ``S`` arrays of shape ``(T, K)``, with ``K = n_top * prod(branching) *
        nodes_per_module``.
    truth : dict
        ``"groups"``: ``(n, K)`` integer array; row ``l-1`` is the level-``l``
        group of each node.  ``"signals"``: a list over subjects of lists
        ``[s_1, ..., s_n]``, ``s_l`` of shape ``(T, number of level-l
        groups)``.  ``"top"``: ``s_n`` of subject 0 (shared by every subject
        unless ``shared="none"``).  ``"signs"``: ``(S, K)`` array; row
        ``s`` holds the node signs ``e`` of subject ``s`` (equal rows with
        ``signs="shared"``).  ``"factors"``: a list over subjects of
        ``(T, number of modules)`` arrays, the module series ``f`` of each
        subject (equal arrays with ``factors="shared"``).  ``"volatility"``:
        the multiplier ``v``, shape ``(T,)`` (ones without ``volatility``).
        Every array is a
        separate copy, so
        changing one subject's signals (or ``"top"``) leaves the others
        unchanged.  ``"taus"`` and ``"branching"``: tuples.

    Raises
    ------
    ValueError
        For a size that is not a positive integer, ``taus`` or
        ``branching`` that is not a sequence of positive integers, an empty
        ``taus``, ``len(branching) != len(taus) - 1``, ``T < 8 * max(taus)``,
        an unknown ``shared``, ``signs`` or ``factors``, a ``volatility``
        that is not None or a positive finite number, a ``volatility_tau``
        that is not a positive integer, or, with ``volatility`` given,
        ``T < 8 * volatility_tau``.
    TypeError
        For a ``set_random_seed`` that is not None, a bool or an integer.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, truth = tc.simulate_hierarchy(S=2, T=400, nodes_per_module=3,
    ...                                     branching=(2, 3), n_top=2,
    ...                                     taus=(2, 5, 20), set_random_seed=0)
    >>> len(data), data[0].shape, truth["groups"].shape
    (2, (400, 36), (3, 36))
    >>> [s.shape for s in truth["signals"][0]]
    [(400, 12), (400, 6), (400, 2)]
    """
    S = _check_positive_int("S", S)
    T = _check_positive_int("T", T)
    per_mod = _check_positive_int("nodes_per_module", nodes_per_module)
    n_top = _check_positive_int("n_top", n_top)
    if not _is_int_sequence(taus) or not len(taus):
        raise ValueError(
            "taus must be a non-empty sequence of positive integers (a list, "
            f"tuple or 1-D numpy array); got {taus!r}."
        )
    taus = _check_int_sequence("taus", taus)
    branching = _check_int_sequence("branching", branching)
    n = len(taus)
    if len(branching) != n - 1:
        raise ValueError(
            f"len(branching) must be len(taus) - 1 = {n - 1}; got {len(branching)}."
        )
    if T < 8 * max(taus):
        raise ValueError(
            f"T must be at least 8 * max(taus) = {8 * max(taus)} for the smoothing; "
            f"got T={T}."
        )
    if shared not in ("top", "all", "none"):
        raise ValueError(f"shared must be 'top', 'all' or 'none'; got {shared!r}.")
    if signs not in ("private", "shared"):
        raise ValueError(f"signs must be 'private' or 'shared'; got {signs!r}.")
    if factors not in ("private", "shared"):
        raise ValueError(
            f"factors must be 'private' or 'shared'; got {factors!r}."
        )
    volatility_tau = _check_positive_int("volatility_tau", volatility_tau)
    if volatility is not None:
        if (
            not isinstance(volatility, numbers.Real)
            or isinstance(volatility, bool)
            or not np.isfinite(volatility)
            or volatility <= 0
        ):
            raise ValueError(
                "volatility must be None or a positive finite number; "
                f"got {volatility!r}."
            )
        if T < 8 * volatility_tau:
            raise ValueError(
                f"T must be at least 8 * volatility_tau = {8 * volatility_tau} "
                f"for the smoothing; got T={T}."
            )
    rng = _make_rng(set_random_seed)

    counts = [n_top]  # counts[l-1]: number of level-l groups
    for b in reversed(branching):
        counts.insert(0, counts[0] * b)
    module = np.repeat(np.arange(counts[0]), per_mod)
    K = len(module)
    groups = [module]
    for b in branching:
        groups.append(groups[-1] // b)
    groups = np.vstack(groups).astype(int)

    if shared == "all":
        common = _hierarchy_signals(rng, T, counts, branching, taus)
    elif shared == "top":
        top = _hierarchy_signals(rng, T, counts[n - 1:], (), taus[n - 1:])[0]
    if signs == "shared":
        common_signs = rng.choice([-1.0, 1.0], K)
    if factors == "shared":
        common_f = rng.standard_normal((T, counts[0]))

    data, signals, node_signs, series = [], [], [], []
    for _ in range(S):
        if shared == "all":
            s = [x.copy() for x in common]
        else:
            s = _hierarchy_signals(
                rng, T, counts, branching, taus,
                top.copy() if shared == "top" else None,
            )
        g = s[0][:, module]
        e = rng.choice([-1.0, 1.0], K) if signs == "private" else common_signs
        if factors == "private":
            f = rng.standard_normal((T, counts[0]))
        else:
            f = common_f.copy()
        z = rng.standard_normal((T, K))
        data.append(np.sqrt(g) * e * f[:, module] + np.sqrt(1 - g) * z)
        signals.append(s)
        node_signs.append(e.copy())
        series.append(f)
    if volatility is None:
        v = np.ones(T)
    else:  # drawn last: every other draw is that of the call without it
        v = np.exp(float(volatility) * _smooth(rng, T, 1, volatility_tau)[:, 0])
        v /= np.sqrt(np.mean(v ** 2))
        data = [x * v[:, None] for x in data]
    truth = {
        "groups": groups,
        "signals": signals,
        "top": signals[0][n - 1].copy(),
        "taus": taus,
        "branching": branching,
        "signs": np.array(node_signs),
        "factors": series,
        "volatility": v,
    }
    return data, truth
