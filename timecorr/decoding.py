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
        returned (labelled with the highest level).  Default: None.
    nfolds : int
        Number of groups (at least 2, at most the number of subjects).
        Default: 2.
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
        If ``nfolds`` is below 2 or above the number of subjects, if ``level``
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

    if not 2 <= nfolds <= len(subjects):
        raise ValueError(
            f"nfolds must be between 2 and the number of subjects ({len(subjects)}); "
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
    negative, summing to one) are chosen to minimize the decoding error between
    the halves, and are then used to mix the test-versus-training decoding
    matrices in Fisher z space (Owen, Chang & Manning, 2021, "Feature
    weighting and testing").  The test group never influences the weights.

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

    circular = nfolds == 1
    if circular:
        warnings.warn("When nfolds is set to one, the analysis will be circular.")
        nfolds = 2
    if nfolds > len(subjects):
        raise ValueError(
            f"nfolds ({nfolds}) exceeds the number of subjects ({len(subjects)})."
        )

    if optimize_levels is None:
        optimize_levels = list(range(top + 1))
    optimize_levels = [int(v) for v in optimize_levels]
    if any(v < 0 or v > top for v in optimize_levels):
        raise ValueError(
            f"optimize_levels must lie between 0 and the highest level ({top}); "
            f"got {optimize_levels}."
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
        ``cfun=None``) and combined.  Level ``k >= 1``: the analysis path of the
        shared pipeline within each group, both groups reduced jointly.
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
    :param data: a list of number-of-observations by number-of-features matrices
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :param level: integer or list of integers for levels to be evaluated (default:0)
    :param cfun: function for transforming the group data (default: isfc)
    :param weights_fun: used to compute per-timepoint weights for cfun; default: laplace_weights
    :param  weights_params: parameters passed to weights_fun; default: laplace_params
    :params combine: function for combining data within each group, or a list of such functions (default: mean_combine)
    :param rfun: function for reducing output (default: None)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
    """

    if nfolds == 1:
        sub_nfolds = 1
        nfolds = 2
        warnings.warn("When nfolds is set to one, the analysis will be circular.")
    else:
        sub_nfolds = 1

    # data[0] holds one array per subject; data[v - 1] (v >= 2) holds each
    # subject's precomputed level-(v - 1) features
    if np.ndim(data[0][0]) != 2:
        raise ValueError(
            "data must be indexed by level: data[0] holds one (T, K) array per "
            "subject (pass [subjects] for raw data only)."
        )
    if len(data[0]) < 2 * nfolds:
        raise ValueError(
            f"{len(data[0])} subjects cannot fill {nfolds} folds of at least 2 "
            "subjects each (each fold's group is split in two)."
        )
    group_assignments = get_xval_assignments(len(data[0]), nfolds)

    orig_level = level
    orig_level = np.ravel(orig_level)

    if type(level) is int:
        level = np.arange(level + 1)

    level = np.ravel(level)

    assert type(level) is np.ndarray, "level needs be an integer, list, or np.ndarray"
    assert not np.any(level < 0), "level cannot contain negative numbers"

    if not np.all(np.arange(level.max() + 1) == level):
        level = np.arange(level.max() + 1)

    if callable(combine):
        combine = [combine] * np.shape(level)[0]

    combine = np.ravel(combine)

    assert type(combine) is np.ndarray and type(combine[0]) is not np.str_, (
        "combine needs to be a function, list of "
        "functions, or np.ndarray of functions"
    )
    assert len(level) == len(
        combine
    ), "combine length need to be the same as level if input is type np.ndarray or list"

    if callable(cfun):
        cfun = [cfun] * np.shape(level)[0]

    cfun = np.ravel(cfun)

    assert type(cfun) is np.ndarray and type(cfun[0]) is not np.str_, (
        "combine needs be a function, list of functions, " "or np.ndarray of functions"
    )
    assert len(level) == len(
        cfun
    ), "cfun length need to be the same as level if input is type np.ndarray or list"

    if type(rfun) not in [list, np.ndarray]:
        rfun = [rfun] * np.shape(level)[0]

    p_rfun = [None] * np.shape(level)[0]

    assert len(level) == len(rfun), (
        "parameter lengths need to be the same as level if input is "
        "type np.ndarray or list"
    )

    results_pd = pd.DataFrame()

    for i in range(0, nfolds):

        sub_corrs = []
        corrs = []

        subgroup_assignments = get_xval_assignments(
            len(_take(data[0], group_assignments == i)), nfolds
        )

        in_data = _take(data[0], group_assignments == i)
        out_data = _take(data[0], group_assignments != i)

        for v in level:

            if v == 0:

                in_smooth, out_smooth, in_raw, out_raw = folding_levels_ec(
                    in_data,
                    out_data,
                    level=v,
                    cfun=None,
                    rfun=p_rfun,
                    combine=combine,
                    weights_fun=weights_fun,
                    weights_params=weights_params,
                )

                # next_corrs = (1 - sd.cdist(mean_combine(in_smooth), mean_combine(out_smooth),
                #                            'correlation'))
                next_corrs = 1 - sd.cdist(
                    mean_combine(in_raw), mean_combine(out_raw), "correlation"
                )
                corrs.append(next_corrs)

                for s in range(0, 1):

                    sub_in_data = [
                        x
                        for x in _take(
                            _take(data[0], group_assignments == i),
                            subgroup_assignments == s,
                        )
                    ]
                    sub_out_data = [
                        x
                        for x in _take(
                            _take(data[0], group_assignments == i),
                            subgroup_assignments != s,
                        )
                    ]

                    sub_in_smooth, sub_out_smooth, sub_in_raw, sub_out_raw = (
                        folding_levels_ec(
                            sub_in_data,
                            sub_out_data,
                            level=v,
                            cfun=None,
                            rfun=p_rfun,
                            combine=combine,
                            weights_fun=weights_fun,
                            weights_params=weights_params,
                        )
                    )
                    # next_subcorrs = (1 - sd.cdist(mean_combine(sub_in_smooth),
                    #                               mean_combine(sub_out_smooth), 'correlation'))
                    next_subcorrs = 1 - sd.cdist(
                        mean_combine(sub_in_raw),
                        mean_combine(sub_out_raw),
                        "correlation",
                    )
                    sub_corrs.append(next_subcorrs)

            elif v == 1:

                in_smooth, out_smooth, in_raw, out_raw = folding_levels_ec(
                    in_raw,
                    out_raw,
                    level=v,
                    cfun=cfun,
                    rfun=rfun,
                    combine=combine,
                    weights_fun=weights_fun,
                    weights_params=weights_params,
                )

                next_corrs = 1 - sd.cdist(
                    mean_combine(in_smooth), mean_combine(out_smooth), "correlation"
                )
                corrs.append(next_corrs)

                for s in range(0, 1):

                    sub_in_smooth, sub_out_smooth, sub_in_raw, sub_out_raw = (
                        folding_levels_ec(
                            sub_in_raw,
                            sub_out_raw,
                            level=v,
                            cfun=cfun,
                            rfun=rfun,
                            combine=combine,
                            weights_fun=weights_fun,
                            weights_params=weights_params,
                        )
                    )
                    next_subcorrs = 1 - sd.cdist(
                        mean_combine(sub_in_smooth),
                        mean_combine(sub_out_smooth),
                        "correlation",
                    )
                    sub_corrs.append(next_subcorrs)

            else:

                in_raw = _take(data[v - 1], group_assignments == i)
                out_raw = _take(data[v - 1], group_assignments != i)

                in_smooth, out_smooth, in_raw, out_raw = folding_levels_ec(
                    in_raw,
                    out_raw,
                    level=v,
                    cfun=cfun,
                    rfun=rfun,
                    combine=combine,
                    weights_fun=weights_fun,
                    weights_params=weights_params,
                )

                next_corrs = 1 - sd.cdist(in_smooth, out_smooth, "correlation")
                corrs.append(next_corrs)

                for s in range(0, 1):

                    sub_in_raw = [
                        x
                        for x in _take(
                            _take(data[v - 1], group_assignments == i),
                            subgroup_assignments == s,
                        )
                    ]
                    sub_out_raw = [
                        x
                        for x in _take(
                            _take(data[v - 1], group_assignments == i),
                            subgroup_assignments != s,
                        )
                    ]

                    sub_in_smooth, sub_out_smooth, sub_in_raw, sub_out_raw = (
                        folding_levels_ec(
                            sub_in_raw,
                            sub_out_raw,
                            level=v,
                            cfun=cfun,
                            rfun=rfun,
                            combine=combine,
                            weights_fun=weights_fun,
                            weights_params=weights_params,
                        )
                    )
                    next_subcorrs = 1 - sd.cdist(
                        sub_in_smooth, sub_out_smooth, "correlation"
                    )
                    sub_corrs.append(next_subcorrs)

        sub_corrs = np.array(sub_corrs)
        corrs = np.array(corrs)

        if sub_nfolds == 1:
            sub_corrs = corrs

        if not optimize_levels:
            optimize_levels = range(v + 1)

        opt_over = []

        for lev in optimize_levels:

            opt_over.append(lev)

            sub_out_corrs = sub_corrs[opt_over, :, :]
            out_corrs = corrs[opt_over, :, :]

            mu = optimize_weights(sub_out_corrs, opt_init)

            w_corrs = weight_corrs(out_corrs, mu)

            next_results_pd = decoder(w_corrs)
            next_results_pd["level"] = lev
            next_results_pd["folds"] = i
            for c, m in zip(opt_over, mu):
                next_results_pd["level_" + str(c)] = m

            results_pd = pd.concat([results_pd, next_results_pd])

    return results_pd


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
    :param data: a list of number-of-observations by number-of-features matrices
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :param cfun: function for transforming the group data (default: isfc)
    :param weights_fun: used to compute per-timepoint weights for cfun; default: laplace_weights
    :param  weights_params: parameters passed to weights_fun; default: laplace_params
    :params combine: function for combining data within each group, or a list of such functions (default: mean_combine)
    :param rfun: function for reducing output (default: None)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
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


def optimize_weights(corrs, opt_init=None):

    b = (0, 1)
    bns = (b,) * np.shape(corrs)[0]
    con1 = {"type": "eq", "fun": lambda x: 1 - np.sum(x)}
    if opt_init == "random":
        x0 = sum_to_x(np.shape(corrs)[0], 1)
    elif opt_init == "last":
        x0 = np.repeat(1 / np.shape(corrs)[0], np.shape(corrs)[0])
        x0[-1] = 1
    else:
        x0 = np.repeat(1 / np.shape(corrs)[0], np.shape(corrs)[0])

    min_mu = minimize(
        calculate_error,
        x0,
        args=corrs,
        bounds=bns,
        constraints=con1,
        options={"disp": False, "eps": 1e-1},
    )

    return min_mu.x


def sum_to_x(n, x):
    values = [0.0, x] + list(np.random.uniform(low=0.0, high=x, size=n - 1))
    values.sort()
    return np.asarray([values[i + 1] - values[i] for i in range(n)])


def calculate_error(mu, corrs, metric="error", sign=1):

    results = decoder(weight_corrs(corrs, mu))
    return sign * results[metric].values


def weight_corrs(corrs, mu):

    assert np.shape(mu)[0] == len(corrs)
    weighted_corrs = 0

    for i in np.arange(np.shape(corrs)[0]):
        weighted_corrs += mu[i] * r2z(corrs[i])

    return z2r(weighted_corrs)


def decoder(corrs):

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
    group_assignments = np.zeros(ndata)
    groupsize = int(np.ceil(ndata / nfolds))

    for i in range(1, nfolds):
        inds = np.arange(i * groupsize, np.min([(i + 1) * groupsize, ndata]))
        group_assignments[inds] = i
    np.random.shuffle(group_assignments)
    return group_assignments


