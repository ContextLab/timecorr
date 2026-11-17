# coding: utf-8
"""
Timepoint decoders: how well features at each order identify the timepoint a
pattern came from, across groups of subjects (Owen, Chang & Manning, 2021,
"Temporal decoding").
"""

import warnings

import hypertools as hyp
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
from scipy.optimize import minimize

from .helpers import (
    format_data,
    isfc,
    laplace_params,
    laplace_weights,
    mean_combine,
    r2z,
    reduce,
    z2r,
)
from .higher_order import _analysis_path, _delta_path


def _subjects(data):
    """Format decoder input as a list of float (T, K) arrays, one per subject."""
    if isinstance(data, np.ndarray) and data.ndim == 3:
        data = list(data)
    subjects = format_data(data, fill_nan=False)
    if len({x.shape for x in subjects}) != 1:
        raise ValueError(
            "all data matrices must have the same number of timepoints and features"
        )
    return subjects


def _parse_levels(level):
    """Return (sorted list of requested levels, highest level)."""
    levels = np.ravel(level)
    if (
        levels.size == 0
        or not np.issubdtype(levels.dtype, np.integer)
        or np.any(levels < 0)
    ):
        raise ValueError(
            f"level must be a non-negative integer or a list of them; got {level!r}."
        )
    levels = sorted({int(v) for v in levels})
    return levels, levels[-1]


def _per_level(setting, n, name):
    """One entry per level 0..n-1: broadcast a single setting, or check a list."""
    if isinstance(setting, (list, tuple, np.ndarray)):
        setting = list(setting)
        if len(setting) != n:
            raise ValueError(
                f"{name} must be a single value or have one entry per level "
                f"(0 to {n - 1}, {n} entries); got {len(setting)}."
            )
        return setting
    return [setting] * n


def _decoder_delta_chain(data, level, rfun=None):
    """
    Delta features ``[D_0, ..., D_{level-1}]`` for every subject.

    ``D_0`` is the data; ``D_k`` is the delta path of the shared pipeline
    (``higher_order._delta_path``: per-subject co-fluctuations, reduced jointly
    across all subjects with ``rfun[k]``).  The analysis features at order
    ``level`` need ``D_{level-1}``, so ``D_level`` is not computed.
    """
    subjects = _subjects(data)
    rfun = _per_level(rfun, level + 1, "rfun")
    D = [subjects]
    for k in range(1, level):
        D.append(_delta_path(D[-1], rfun=rfun[k]))
    return D


def _decoder_features(
    data,
    groups,
    level,
    cfun=isfc,
    weights_fun=laplace_weights,
    weights_params=laplace_params,
    combine=mean_combine,
    rfun=None,
    D=None,
):
    """
    Features of every order from 0 to ``level`` for each group of subjects.

    Returns ``feats`` with ``feats[k][g]`` the order-``k`` features of group
    ``g`` (a list of subject indices into ``data``):

    - ``k = 0``: ``combine[0]`` of the group's data.
    - ``k >= 1``: the analysis path of the shared pipeline applied to the
      group's delta features ``D_{k-1}`` (``cfun[k]`` and ``combine[k]`` see one
      group at a time), with all groups reduced jointly by ``rfun[k]`` so they
      share one space (``higher_order._analysis_path``).

    ``D`` (from ``_decoder_delta_chain``) may be passed to reuse the delta
    chain across folds.
    """
    subjects = _subjects(data)
    n = level + 1
    cfun = _per_level(cfun, n, "cfun")
    combine = _per_level(combine, n, "combine")
    rfun = _per_level(rfun, n, "rfun")
    if D is None:
        D = _decoder_delta_chain(subjects, level, rfun)
    groups = [[int(j) for j in g] for g in groups]

    feats = [[combine[0]([subjects[j] for j in g]) for g in groups]]
    for k in range(1, n):
        feats.append(
            _analysis_path(
                D[k - 1],
                weights_function=weights_fun,
                weights_params=weights_params,
                cfun=cfun[k],
                rfun=rfun[k],
                combine=combine[k],
                groups=groups,
            )
        )
    return feats


def _decoding_matrix(decode, template):
    """Correlations between rows: entry (t, u) compares decode[t] to template[u]."""
    return 1 - sd.cdist(decode, template, "correlation")


def _level_matrices(subjects, decode, template, level, D, **settings):
    feats = _decoder_features(subjects, [decode, template], level, D=D, **settings)
    return np.array([_decoding_matrix(f[0], f[1]) for f in feats])


def timepoint_decoder(
    data,
    mu=None,
    nfolds=2,
    level=0,
    cfun=isfc,
    weights_fun=laplace_weights,
    weights_params=laplace_params,
    combine=mean_combine,
    rfun=None,
):
    """
    Decode timepoints across groups of subjects from features at each order.

    Subjects are split into ``nfolds`` groups.  In fold ``i``, group ``i`` is
    decoded and the remaining subjects form the template.  For each order ``k``
    the features of both groups are computed with the shared two-path pipeline
    (``timecorr.higher_order``): order 0 is the data, combined within each group;
    order ``k >= 1`` is the analysis path (``cfun`` with the analysis kernel,
    then ``combine``) applied within each group to the delta features
    ``D_{k-1}`` (unblurred co-fluctuations), with both groups reduced jointly.
    Each decoded timepoint is assigned the template timepoint whose features
    are most correlated with it.

    Parameters
    ----------
    data : list of numpy.ndarray, pandas.DataFrame or polars.DataFrame, or a 3D numpy.ndarray
        One timepoints-by-features matrix per subject; all the same shape.
    mu : array-like, optional
        Non-negative weights summing to one, one per order from 0 to
        ``max(level)``.  If given, the per-order decoding matrices are averaged
        in Fisher z space with these weights and a single result per fold is
        returned (labeled with the highest level).  Default: None.
    nfolds : int
        Number of groups (at most the number of subjects).  ``nfolds=1`` warns
        that the analysis is circular and splits the subjects into two groups,
        as ``weighted_timepoint_decoder`` does.  Default: 2.
    level : int or list of int
        Order(s) to report.  Every order from 0 to the highest is computed.
        Default: 0.
    cfun : function or list of functions
        Analysis-path correlation function, or one per order (entry 0 unused).
        Default: ``isfc``.
    weights_fun, weights_params : analysis kernel and its parameters.
        Default: ``laplace_weights`` with ``laplace_params``.
    combine : function or list of functions
        Combines each group's per-subject features, or one per order.
        Default: ``mean_combine``.
    rfun : None, str, dict, callable, or a list of these
        Reducer for orders ``k >= 1`` on both paths (see ``helpers.reduce``), or
        one per order (entry 0 unused).  Default: None (no reduction).

    Returns
    -------
    pandas.DataFrame
        One row per fold and reported level, with columns ``rank`` (mean
        fraction of template timepoints whose correlation is at or below the
        true timepoint's), ``accuracy`` (fraction decoded exactly), ``error``
        (mean absolute decoding error as a fraction of the number of
        timepoints), ``level`` and ``folds``.

    Raises
    ------
    ValueError
        If ``nfolds`` is below 1 or above the number of subjects, if ``level``
        is not a non-negative integer (or list of them), or if ``mu`` has the
        wrong length or does not sum to one.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> rng = np.random.RandomState(0)
    >>> shared = rng.randn(30, 4)
    >>> data = [shared + 0.5 * rng.randn(30, 4) for _ in range(4)]
    >>> np.random.seed(0)
    >>> results = tc.timepoint_decoder(data, nfolds=2, level=[0, 1])
    >>> list(results.columns)
    ['rank', 'accuracy', 'error', 'level', 'folds']
    >>> results["level"].tolist(), results["folds"].tolist()
    ([0, 1, 0, 1], [0, 0, 1, 1])
    >>> bool(results.loc[results["level"] == 0, "accuracy"].mean() > 3 / 30)  # chance: 1/30
    True
    """
    subjects = _subjects(data)
    report, top = _parse_levels(level)
    n = top + 1

    if mu is not None:
        mu = np.asarray(mu, dtype=float).ravel()
        if mu.shape[0] != n:
            raise ValueError(
                f"mu needs one weight per level from 0 to {top} ({n}); got {mu.shape[0]}."
            )
        if np.any(mu < 0) or not np.isclose(mu.sum(), 1.0):
            raise ValueError("mu must be non-negative and sum to one.")
        report = [top]

    if nfolds == 1:
        warnings.warn("When nfolds is set to one, the analysis will be circular.")
        nfolds = 2
    if not 2 <= nfolds <= len(subjects):
        raise ValueError(
            f"nfolds must be between 1 and the number of subjects ({len(subjects)}); "
            f"got {nfolds}."
        )

    settings = dict(
        cfun=cfun,
        weights_fun=weights_fun,
        weights_params=weights_params,
        combine=combine,
        rfun=rfun,
    )
    D = _decoder_delta_chain(subjects, top, rfun)
    group_assignments = get_xval_assignments(len(subjects), nfolds)

    rows = []
    for i in range(nfolds):
        decode = np.where(group_assignments == i)[0]
        template = np.where(group_assignments != i)[0]
        corrs = _level_matrices(subjects, decode, template, top, D, **settings)
        if mu is not None:
            mixed = {top: weight_corrs(corrs, mu)}
        else:
            mixed = {v: corrs[v] for v in report}
        for v in report:
            result = decoder(mixed[v])
            result["level"] = v
            result["folds"] = i
            rows.append(result)
    return pd.concat(rows)


def weighted_timepoint_decoder(
    data,
    nfolds=2,
    level=0,
    optimize_levels=None,
    cfun=isfc,
    weights_fun=laplace_weights,
    weights_params=laplace_params,
    combine=mean_combine,
    rfun=None,
    opt_init=None,
):
    """
    Decode timepoints from a weighted mix of the orders' decoding matrices.

    Features are computed as in ``timepoint_decoder``.  In fold ``i``, group
    ``i`` is the test group and the remaining subjects are the training group.
    The training group is split in two halves; weights over the orders (non-
    negative, summing to one) are fit to the half-versus-half decoding
    matrices by ``optimize_weights``, and are then used to mix the
    test-versus-training decoding matrices in Fisher z space (Owen, Chang &
    Manning, 2021, "Feature weighting and testing").  ``optimize_weights``
    minimizes a smooth cross-entropy surrogate of decoding accuracy; the paper
    describes maximizing decoding accuracy itself, which is piecewise constant
    in the weights (timecorr 0.2.0's optimizer, started from equal weights,
    stayed there).  The supervised step, fitting the weights, uses
    the training group only.  The dimensionality reduction (``rfun``) is
    unsupervised and, as in the paper (Fig. 10b-c), fit jointly on all
    subjects, test group included, so that every group's features share one
    space; with a fitted ``rfun``, the test group's data therefore affect the
    training features through that shared space (never through timepoint
    labels).  With ``rfun=None`` nothing computed from the test group reaches
    the weights.  Weights are fit by ``optimize_weights``.

    Parameters
    ----------
    data : list of numpy.ndarray, pandas.DataFrame or polars.DataFrame, or a 3D numpy.ndarray
        One timepoints-by-features matrix per subject; all the same shape.
    nfolds : int
        Number of groups.  With ``nfolds=1`` a warning is issued, the subjects
        are split in two, and the weights are fit on the same test-versus-
        training matrices they are evaluated on (circular).  Default: 2.
    level : int or list of int
        Every order from 0 to the highest given is computed.  Default: 0.
    optimize_levels : list of int, optional
        Orders to mix, added one at a time: a result is reported after each
        addition (e.g. ``[0, 1, 2]`` reports orders {0}, {0, 1}, {0, 1, 2}).
        Default: every order from 0 to the highest level.
    cfun, weights_fun, weights_params, combine, rfun
        As in ``timepoint_decoder``.
    opt_init : None, "random", or "last"
        Starting weights for the optimizer: equal (None), random, or equal with
        the last set to one.  Default: None.

    Returns
    -------
    pandas.DataFrame
        One row per fold and step of ``optimize_levels``, with columns
        ``rank``, ``accuracy``, ``error`` (as in ``timepoint_decoder``),
        ``level`` (the order added at that step), ``folds``, and
        ``level_<k>``, the weight given to order ``k`` (NaN for orders not yet
        mixed in).

    Raises
    ------
    ValueError
        If the training group of a fold has fewer than two subjects (it cannot
        be split), if ``nfolds`` exceeds the number of subjects, or if
        ``optimize_levels`` names an order above ``level``.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> rng = np.random.RandomState(0)
    >>> shared = rng.randn(30, 4)
    >>> data = [shared + 0.5 * rng.randn(30, 4) for _ in range(6)]
    >>> np.random.seed(0)
    >>> results = tc.weighted_timepoint_decoder(data, nfolds=2, level=1)
    >>> results["level"].tolist(), results["folds"].tolist()
    ([0, 1, 0, 1], [0, 0, 1, 1])
    >>> mu = results.loc[results["level"] == 1, ["level_0", "level_1"]]
    >>> np.allclose(mu.sum(axis=1), 1)
    True
    """
    subjects = _subjects(data)
    _, top = _parse_levels(level)
    nfolds, circular = _check_folds(nfolds, len(subjects))
    optimize_levels = _check_optimize_levels(optimize_levels, top)

    settings = dict(
        cfun=cfun,
        weights_fun=weights_fun,
        weights_params=weights_params,
        combine=combine,
        rfun=rfun,
    )
    D = _decoder_delta_chain(subjects, top, rfun)
    return _weighted_decode(
        subjects, D, nfolds, circular, top, optimize_levels, opt_init, settings
    )


def _check_folds(nfolds, n_subjects):
    """Return (nfolds, circular); nfolds=1 warns and becomes 2."""
    circular = nfolds == 1
    if circular:
        warnings.warn("When nfolds is set to one, the analysis will be circular.")
        nfolds = 2
    if nfolds > n_subjects:
        raise ValueError(
            f"nfolds ({nfolds}) exceeds the number of subjects ({n_subjects})."
        )
    return nfolds, circular


def _check_optimize_levels(optimize_levels, top):
    if optimize_levels is None:
        return list(range(top + 1))
    optimize_levels = [int(v) for v in optimize_levels]
    if any(v < 0 or v > top for v in optimize_levels):
        raise ValueError(
            f"optimize_levels must lie between 0 and the highest level ({top}); "
            f"got {optimize_levels}."
        )
    return optimize_levels


def _weighted_decode(
    subjects, D, nfolds, circular, top, optimize_levels, opt_init, settings
):
    """
    Cross-validated decoding with level weights fit on the training group only.

    In fold ``i`` the test group is group ``i`` and the training group the rest.
    The training group is split in two halves; the level weights are fit to the
    half-versus-half decoding matrices and applied to the test-versus-training
    matrices.  ``D`` is the delta chain ``[D_0, ..., D_{top-1}]`` for every
    subject.  With ``circular`` the weights are fit to the test-versus-training
    matrices themselves.
    """
    group_assignments = get_xval_assignments(len(subjects), nfolds)

    rows = []
    for i in range(nfolds):
        test = np.where(group_assignments == i)[0]
        train = np.where(group_assignments != i)[0]
        corrs = _level_matrices(subjects, test, train, top, D, **settings)

        if circular:
            sub_corrs = corrs
        else:
            if len(train) < 2:
                raise ValueError(
                    f"The training group of fold {i} has {len(train)} subject(s); at "
                    "least 2 are needed to fit the level weights. Use more subjects "
                    "or fewer folds."
                )
            halves = get_xval_assignments(len(train), 2)
            sub_corrs = _level_matrices(
                subjects, train[halves == 0], train[halves == 1], top, D, **settings
            )

        opt_over = []
        for lev in optimize_levels:
            opt_over.append(lev)
            mu = optimize_weights(sub_corrs[opt_over], opt_init)
            result = decoder(weight_corrs(corrs[opt_over], mu))
            result["level"] = lev
            result["folds"] = i
            for c, m in zip(opt_over, mu):
                result["level_" + str(c)] = m
            rows.append(result)
    return pd.concat(rows)


def folding_levels(
    infold_data,
    outfold_data,
    level=0,
    cfun=None,
    weights_fun=None,
    weights_params=None,
    combine=None,
    rfun=None,
):
    """
    One order of decoder features for an in-fold and an out-of-fold group.

    Parameters
    ----------
    infold_data, outfold_data : list of numpy.ndarray
        Each group's previous delta features ``D_{level-1}`` (its data for
        ``level`` 0 or 1), one (T, d) array per subject.
    level : int
        Order computed.  Default: 0.
    cfun, combine, rfun : function (or None) or a list indexed by level
        Analysis-path correlation function (default ``isfc``), within-group
        combine (default ``mean_combine``) and reducer (default None).
    weights_fun, weights_params : analysis kernel and parameters.
        Default: ``laplace_weights`` with ``laplace_params``; with another
        ``weights_fun`` and no ``weights_params``, that kernel's defaults.

    Returns
    -------
    in_smooth, out_smooth : numpy.ndarray
        Level 0: each group's data smoothed with the kernel (``timecorr`` with
        ``cfun=None``) and combined.  This differs from ``timepoint_decoder``
        and ``weighted_timepoint_decoder``, whose order-0 features are each
        group's data combined without smoothing.  Level ``k >= 1``: the
        analysis path of the shared pipeline within each group, both groups
        reduced jointly.
    in_raw, out_raw : list of numpy.ndarray
        Level 0: the inputs.  Level ``k >= 1``: the delta path (per-subject
        co-fluctuations), all subjects reduced jointly.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.helpers import folding_levels
    >>> rng = np.random.RandomState(0)
    >>> a = [rng.randn(20, 3) for _ in range(2)]
    >>> b = [rng.randn(20, 3) for _ in range(2)]
    >>> in_s, out_s, in_r, out_r = folding_levels(a, b, level=1)
    >>> in_s.shape, out_s.shape, len(in_r), in_r[0].shape
    ((20, 6), (20, 6), 2, (20, 6))
    """
    from .timecorr import timecorr

    n = level + 1
    cfun = _per_level(isfc if cfun is None else cfun, n, "cfun")
    combine = _per_level(mean_combine if combine is None else combine, n, "combine")
    rfun = _per_level(rfun, n, "rfun")
    if weights_fun is None:
        weights_fun = laplace_weights
        if weights_params is None:
            weights_params = laplace_params

    if level == 0:
        smooth = [
            np.asarray(
                timecorr(
                    [x for x in group],
                    cfun=None,
                    rfun=rfun[0],
                    combine=combine[0],
                    weights_function=weights_fun,
                    weights_params=weights_params,
                )
            )
            for group in (infold_data, outfold_data)
        ]
        return smooth[0], smooth[1], infold_data, outfold_data

    subjects = format_data(list(infold_data) + list(outfold_data), fill_nan=False)
    n_in = len(infold_data)
    groups = [list(range(n_in)), list(range(n_in, len(subjects)))]
    in_smooth, out_smooth = _analysis_path(
        subjects,
        weights_function=weights_fun,
        weights_params=weights_params,
        cfun=cfun[level],
        rfun=rfun[level],
        combine=combine[level],
        groups=groups,
    )
    raw = _delta_path(subjects, rfun=rfun[level])
    return in_smooth, out_smooth, raw[:n_in], raw[n_in:]


def weighted_timepoint_decoder_ec(
    data,
    nfolds=2,
    level=0,
    optimize_levels=None,
    cfun=isfc,
    weights_fun=laplace_weights,
    weights_params=laplace_params,
    combine=mean_combine,
    rfun=None,
    opt_init=None,
):
    """
    ``weighted_timepoint_decoder`` with externally computed lower-order features.

    Instead of the delta path, ``data[k]`` supplies every subject's order-``k``
    input features ``D_k``: ``data[0]`` is the data (order 0), and for
    ``level >= 2`` ``data[k]`` (``1 <= k < level``) holds features computed
    elsewhere (for example a graph measure of each timepoint's correlations).
    Order ``k >= 1`` features are the analysis path (``cfun`` with the kernel,
    then ``combine``) applied within each group to ``data[k - 1]``, both groups
    reduced jointly by ``rfun``.  Folds, the training-only fit of the level
    weights, and the output are as in ``weighted_timepoint_decoder``.

    Parameters
    ----------
    data : sequence
        Indexed by order: ``data[k]`` is a list (or 3D array) with one (T, d_k)
        array per subject, for ``k = 0, ..., max(level - 1, 0)``.
    nfolds, level, optimize_levels, cfun, weights_fun, weights_params, combine, rfun, opt_init
        As in ``weighted_timepoint_decoder``.

    Returns
    -------
    pandas.DataFrame
        As in ``weighted_timepoint_decoder``.

    Raises
    ------
    ValueError
        If ``data`` is not indexed by order (``data[0][0]`` must be a (T, K)
        array), has fewer than ``level`` orders, or the subject counts differ
        across orders; and as in ``weighted_timepoint_decoder``.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> from timecorr.helpers import weighted_timepoint_decoder_ec
    >>> rng = np.random.RandomState(0)
    >>> shared = rng.randn(30, 4)
    >>> subjects = [shared + 0.5 * rng.randn(30, 4) for _ in range(6)]
    >>> np.random.seed(0)
    >>> results = weighted_timepoint_decoder_ec([subjects], nfolds=2, level=1)
    >>> results["level"].tolist()
    [0, 1, 0, 1]
    """
    if np.ndim(data[0][0]) != 2:
        raise ValueError(
            "data must be indexed by level: data[0] holds one (T, K) array per "
            "subject (pass [subjects] for raw data only)."
        )
    _, top = _parse_levels(level)
    if len(data) < max(top, 1):
        raise ValueError(
            f"level {top} needs data for orders 0 to {top - 1}; got {len(data)} order(s)."
        )
    D = [_subjects(data[k]) for k in range(max(top, 1))]
    if len({len(d) for d in D}) != 1:
        raise ValueError("every order of data must have one entry per subject.")
    subjects = D[0]
    nfolds, circular = _check_folds(nfolds, len(subjects))
    optimize_levels = _check_optimize_levels(optimize_levels, top)
    settings = dict(
        cfun=cfun,
        weights_fun=weights_fun,
        weights_params=weights_params,
        combine=combine,
        rfun=rfun,
    )
    return _weighted_decode(
        subjects, D, nfolds, circular, top, optimize_levels, opt_init, settings
    )


def _take(seq, mask):
    """The entries of a sequence (list or array) where a boolean mask is True."""
    return [seq[j] for j in np.where(mask)[0]]


def folding_levels_ec(
    infold_data,
    outfold_data,
    level=0,
    cfun=None,
    weights_fun=None,
    weights_params=None,
    combine=None,
    rfun=None,
):
    """
    One level of decoder features for two groups, from precomputed inputs.

    Unlike ``folding_levels`` there is no delta path: each group's input is
    used as given (for example features computed elsewhere for that level).

    Parameters
    ----------
    infold_data, outfold_data : list of numpy.ndarray
        Each group's input features for this level, one (T, d) array per
        subject.
    level : int
        Level computed; selects the entry of list-valued settings.  Default: 0.
    cfun, combine, rfun : function (or None), or a list indexed by level
        Correlation function (used for ``level >= 1``), within-group combine
        (default ``mean_combine``) and reducer (default None), each applied to
        each group separately.
    weights_fun, weights_params : kernel function and its parameters
        Passed to ``timecorr``.  ``weights_fun`` is required.

    Returns
    -------
    in_smooth, out_smooth : numpy.ndarray
        Level 0: each group's data smoothed with the kernel (``timecorr`` with
        ``cfun=None``) and combined.  Level ``k >= 1``: ``timecorr`` with
        ``cfun`` applied within each group, then combined and reduced.
    in_raw, out_raw : list of numpy.ndarray
        The inputs, unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> from timecorr.helpers import folding_levels_ec
    >>> rng = np.random.RandomState(0)
    >>> a = [rng.randn(20, 3) for _ in range(2)]
    >>> b = [rng.randn(20, 3) for _ in range(2)]
    >>> out = folding_levels_ec(a, b, level=1, cfun=tc.isfc,
    ...                         weights_fun=tc.laplace_weights,
    ...                         weights_params={"scale": 5})
    >>> out[0].shape, out[1].shape
    ((20, 6), (20, 6))
    """
    from .timecorr import timecorr

    # settings may be single values or lists indexed by level (callers pass
    # the lists for every level)
    def by_level(setting):
        if isinstance(setting, (list, tuple, np.ndarray)):
            return list(setting)
        return [setting] * (level + 1)

    cfun = by_level(cfun)
    combine = by_level(mean_combine if combine is None else combine)
    rfun = by_level(rfun)

    if level == 0:

        in_fold_smooth = np.asarray(
            timecorr(
                [x for x in infold_data],
                cfun=None,
                rfun=rfun[level],
                combine=combine[level],
                weights_function=weights_fun,
                weights_params=weights_params,
            )
        )
        out_fold_smooth = np.asarray(
            timecorr(
                [x for x in outfold_data],
                cfun=None,
                rfun=rfun[level],
                combine=combine[level],
                weights_function=weights_fun,
                weights_params=weights_params,
            )
        )

        in_fold_raw = infold_data
        out_fold_raw = outfold_data
    else:

        in_fold_smooth = np.asarray(
            timecorr(
                list(infold_data),
                cfun=cfun[level],
                rfun=rfun[level],
                combine=combine[level],
                weights_function=weights_fun,
                weights_params=weights_params,
            )
        )
        out_fold_smooth = np.asarray(
            timecorr(
                list(outfold_data),
                cfun=cfun[level],
                rfun=rfun[level],
                combine=combine[level],
                weights_function=weights_fun,
                weights_params=weights_params,
            )
        )
        in_fold_raw = infold_data
        out_fold_raw = outfold_data

    return in_fold_smooth, out_fold_smooth, in_fold_raw, out_fold_raw


def pca_decoder(
    data,
    nfolds=2,
    dims=10,
    cfun=isfc,
    weights_fun=laplace_weights,
    weights_params=laplace_params,
    combine=mean_combine,
    rfun=None,
):
    """
    Decode timepoints from the first ``d`` principal components, for each ``d``.

    All subjects are reduced jointly with PCA (``min(dims, features, samples)``
    components).  Subjects are split into ``nfolds`` groups; in fold ``i``
    group ``i`` is decoded against the remaining subjects.  For each ``d``
    from 1 to ``dims``, each group's first ``d`` components are smoothed with
    the kernel (``timecorr`` with ``cfun=None``, level 0 of
    ``folding_levels``) and averaged across the group's subjects (padded with
    zero columns to at least 3), and each decoded timepoint is assigned the
    template timepoint whose pattern is most correlated with it.  These
    features are smoothed, unlike the order-0 features of
    ``timepoint_decoder``, which are the unsmoothed data.

    Parameters
    ----------
    data : list of numpy.ndarray
        One timepoints-by-features matrix per subject, all the same shape.
    nfolds : int
        Number of groups.  Default: 2.
    dims : int
        Largest number of components.  Default: 10.
    cfun, combine, rfun
        Accepted for compatibility and not used: the features are always the
        smoothed components, combined with ``mean_combine``, without further
        reduction.
    weights_fun, weights_params : kernel function and its parameters
        The smoothing kernel.  Default: ``laplace_weights`` with
        ``laplace_params``.

    Returns
    -------
    pandas.DataFrame
        One row per fold and number of components, with columns ``rank``,
        ``accuracy`` and ``error`` (as in ``timepoint_decoder``), ``dims`` and
        ``folds``.

    Examples
    --------
    >>> import numpy as np
    >>> import timecorr as tc
    >>> from timecorr.helpers import pca_decoder
    >>> rng = np.random.RandomState(0)
    >>> shared = rng.randn(30, 6)
    >>> data = [shared + 0.5 * rng.randn(30, 6) for _ in range(4)]
    >>> np.random.seed(0)
    >>> results = pca_decoder(data, nfolds=2, dims=3,
    ...                       weights_params={"scale": 2})
    >>> list(results.columns)
    ['rank', 'accuracy', 'error', 'dims', 'folds']
    >>> results["dims"].tolist(), results["folds"].tolist()
    ([1, 2, 3, 1, 2, 3], [0, 0, 0, 1, 1, 1])
    """

    assert (
        len(np.unique(list(map(lambda x: x.shape[0], data)))) == 1
    ), "all data matrices must have the same number of timepoints"
    assert (
        len(np.unique(list(map(lambda x: x.shape[1], data)))) == 1
    ), "all data matrices must have the same number of features"

    # never request more components than the data has features or samples; the
    # loop below slices the first d columns, so d > n_components is harmless
    n_components = min(dims, data[0].shape[1], sum(x.shape[0] for x in data))
    pca_data = np.asarray(hyp.reduce(list(data), ndims=n_components, random_state=0))

    group_assignments = get_xval_assignments(len(pca_data), nfolds)
    results_pd = pd.DataFrame()

    for i in range(0, nfolds):
        for d in range(1, dims + 1):

            in_data = np.asarray([x for x in pca_data[group_assignments == i]])[
                :, :, :d
            ]
            out_data = np.asarray([x for x in pca_data[group_assignments != i]])[
                :, :, :d
            ]

            in_smooth, out_smooth, in_raw, out_raw = folding_levels(
                in_data,
                out_data,
                level=0,
                cfun=isfc,
                rfun=[None],
                combine=[mean_combine],
                weights_fun=weights_fun,
                weights_params=weights_params,
            )

            if d < 3:
                in_smooth = np.hstack(
                    (in_smooth, np.zeros((in_smooth.shape[0], 3 - in_smooth.shape[1])))
                )
                out_smooth = np.hstack(
                    (
                        out_smooth,
                        np.zeros((out_smooth.shape[0], 3 - out_smooth.shape[1])),
                    )
                )
            corrs = 1 - sd.cdist(in_smooth, out_smooth, "correlation")

            corrs = np.array(corrs)
            next_results_pd = decoder(corrs)
            next_results_pd["dims"] = d
            next_results_pd["folds"] = i

            results_pd = pd.concat([results_pd, next_results_pd])

    return results_pd


def reduce_wrapper(data, dims=10, level=0, rfun=None):

    if not level == 0:

        all_smooth = list(data[0][np.newaxis, :, :]) + list(data[1][np.newaxis, :, :])
        all_raw = list(data[2]) + list(data[3])
        all_smooth_reduced = reduce(all_smooth, rfun=rfun[level])
        all_raw_reduced = reduce(all_raw, rfun=rfun[level])

        return (
            all_smooth_reduced[0],
            all_smooth_reduced[1],
            all_raw_reduced[0],
            all_raw_reduced[1],
        )

    else:

        return data[0], data[1], data[2], data[3]


_MAX_ABS_R = 1 - 1e-7  # keeps Fisher z finite for |r| = 1


def _weights_loss(mu, Z):
    """
    Cross-entropy of the true timepoint under a softmax over the mixed matrix.

    ``Z`` is (levels, T, T) Fisher z decoding matrices; the mixed matrix is
    ``sum_i mu_i Z_i``.  For each decoded timepoint t, a softmax over its row
    gives a probability to every template timepoint; the loss is the mean
    negative log probability of the true one (t).  It is smooth and convex in
    ``mu``.  Returns (loss, gradient).
    """
    mixed = np.tensordot(mu, Z, axes=1)
    shifted = mixed - mixed.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=1, keepdims=True)
    idx = np.arange(Z.shape[1])
    log_p_true = shifted[idx, idx] - np.log(np.exp(shifted).sum(axis=1))
    loss = -np.mean(log_p_true)
    # d loss / d mu_i = mean_t (sum_u p[t, u] Z_i[t, u] - Z_i[t, t])
    grad = np.einsum("tu,itu->i", p, Z) / Z.shape[1] - Z[:, idx, idx].mean(axis=1)
    return loss, grad


def optimize_weights(corrs, opt_init=None):
    """
    Non-negative weights, summing to one, for mixing decoding matrices.

    Decoding accuracy is piecewise constant in the weights (it changes only when
    an argmax switches), so a gradient-based optimizer started anywhere sees zero
    gradient and stays put.  The weights are therefore fit to a smooth surrogate
    of decoding accuracy: the cross-entropy of the true timepoint under a softmax
    over each row of the mixed matrix ``sum_i mu_i r2z(corrs_i)`` (a multinomial
    logistic likelihood of correct decoding), minimized by SLSQP with its
    analytic gradient on the simplex.

    Parameters
    ----------
    corrs : array-like, shape (levels, T, T)
        Decoding matrices (correlations), one per level; row t compares decoded
        timepoint t with every template timepoint.
    opt_init : None, "random", or "last"
        Starting weights: equal (None), random (``sum_to_x``, global numpy
        random state), or all weight on the last level.  Default: None.

    Returns
    -------
    numpy.ndarray
        One weight per level, non-negative, summing to one.  Deterministic for a
        given input (and seed, with ``opt_init="random"``).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.decoding import optimize_weights
    >>> chance = np.zeros((5, 5))
    >>> informative = np.eye(5) * 0.9
    >>> mu = optimize_weights([chance, informative])
    >>> int(np.argmax(mu)), bool(np.isclose(mu.sum(), 1))
    (1, True)
    """
    corrs = np.asarray(corrs, dtype=float)
    n = corrs.shape[0]
    Z = r2z(np.clip(corrs, -_MAX_ABS_R, _MAX_ABS_R))

    if opt_init == "random":
        x0 = sum_to_x(n, 1)
    elif opt_init == "last":
        x0 = np.zeros(n)
        x0[-1] = 1.0
    else:
        x0 = np.repeat(1 / n, n)

    result = minimize(
        _weights_loss,
        x0,
        args=(Z,),
        jac=True,
        method="SLSQP",
        bounds=((0, 1),) * n,
        constraints={"type": "eq", "fun": lambda x: 1 - np.sum(x)},
        options={"disp": False},
    )
    # remove the solver's tiny bound and constraint violations
    mu = np.clip(result.x, 0, None)
    return mu / mu.sum()


def sum_to_x(n, x):
    values = [0.0, x] + list(np.random.uniform(low=0.0, high=x, size=n - 1))
    values.sort()
    return np.asarray([values[i + 1] - values[i] for i in range(n)])


def calculate_error(mu, corrs, metric="error", sign=1):

    results = decoder(weight_corrs(corrs, mu))
    return sign * results[metric].values


def weight_corrs(corrs, mu):
    """
    Mix decoding matrices with weights, in Fisher z space.

    Parameters
    ----------
    corrs : array-like, shape (levels, T, T)
        Correlation matrices, one per level.
    mu : array-like
        One weight per level.

    Returns
    -------
    numpy.ndarray
        ``z2r(sum_i mu_i * r2z(corrs_i))``, shape (T, T).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.decoding import weight_corrs
    >>> mixed = weight_corrs(np.array([np.eye(3) * 0.5, np.zeros((3, 3))]), [0.5, 0.5])
    >>> print(np.round(np.diag(mixed), 3))
    [0.268 0.268 0.268]
    """
    assert np.shape(mu)[0] == len(corrs)
    weighted_corrs = 0

    for i in np.arange(np.shape(corrs)[0]):
        weighted_corrs += mu[i] * r2z(corrs[i])

    return z2r(weighted_corrs)


def decoder(corrs):
    """
    Score a decoding matrix.

    Row ``t`` of ``corrs`` compares decoded timepoint ``t`` with every
    template timepoint; the decoded label is the column with the largest
    value.

    Parameters
    ----------
    corrs : numpy.ndarray, shape (T, T)
        Decoding matrix.

    Returns
    -------
    pandas.DataFrame
        One row with ``rank`` (mean fraction of template timepoints whose value
        is at or below the true timepoint's), ``accuracy`` (fraction decoded
        exactly) and ``error`` (mean absolute decoding error as a fraction of
        T).

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.decoding import decoder
    >>> decoder(np.eye(5))[["accuracy", "error"]].values.tolist()
    [[1.0, 0.0]]
    """
    next_results_pd = pd.DataFrame({"rank": [0], "accuracy": [0], "error": [0]})
    for t in np.arange(corrs.shape[0]):
        decoded_inds = np.argmax(corrs[t, :])
        next_results_pd["error"] += (
            np.mean(np.abs(decoded_inds - np.array(t))) / corrs.shape[0]
        )
        next_results_pd["accuracy"] += np.mean(decoded_inds == np.array(t))
        next_results_pd["rank"] += np.mean(
            list(map((lambda x: int(x)), (corrs[t, :] <= corrs[t, t])))
        )

    next_results_pd["error"] = next_results_pd["error"].values / corrs.shape[0]
    next_results_pd["accuracy"] = next_results_pd["accuracy"].values / corrs.shape[0]
    next_results_pd["rank"] = next_results_pd["rank"].values / corrs.shape[0]

    return next_results_pd



def get_xval_assignments(ndata, nfolds):
    """
    Random assignment of ``ndata`` items to ``nfolds`` groups of near-equal size.

    Uses the global NumPy random state (``numpy.random.shuffle``).

    Parameters
    ----------
    ndata : int
        Number of items (for example subjects).
    nfolds : int
        Number of groups.

    Returns
    -------
    numpy.ndarray
        Group index (as float) of each item; groups have ``ceil(ndata / nfolds)``
        items, the last one possibly fewer.

    Examples
    --------
    >>> import numpy as np
    >>> from timecorr.decoding import get_xval_assignments
    >>> np.random.seed(0)
    >>> sorted(get_xval_assignments(5, 2).tolist())
    [0.0, 0.0, 0.0, 1.0, 1.0]
    """
    group_assignments = np.zeros(ndata)
    groupsize = int(np.ceil(ndata / nfolds))

    for i in range(1, nfolds):
        inds = np.arange(i * groupsize, np.min([(i + 1) * groupsize, ndata]))
        group_assignments[inds] = i
    np.random.shuffle(group_assignments)
    return group_assignments


