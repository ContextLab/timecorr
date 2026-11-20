"""
What one-sided kernels guarantee.

With ``include_timepoints="backward"`` the kernel row of timepoint t weights
only samples at or before t (``"forward"``: at or after t).  The estimator
centers each column on its mean over all timepoints before its moment sums,
and its zero-variance rule compares a kernel-weighted variance with the mean
square of that centered column.  So:

- an entry whose estimate is well conditioned (its kernel-weighted variance
  is not a small fraction of that mean square) does not depend on the
  samples the kernel row does not weight: removing or replacing them changes
  it by at most 1e-9;
- an entry near the zero-variance threshold, or whose variance nearly
  cancels, may differ, because the overall mean (and the order of
  floating-point summation) changes.

The tests assert the first part, and that every entry that differs is one
the conditioning measure flags (as in ``test_time_sliced_estimation.py``),
so a well-conditioned entry that leaks fails them.
"""

import functools
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import helpers

_WELL_CONDITIONED = 1e-3
_ROUND_OFF2 = helpers._ROUND_OFF**2
ATOL = 1e-9
VAR = 30


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _column_conditioning(x, w):
    """
    Per kernel row and column (T', K): ``cond``, the kernel-weighted variance
    (two passes, over the observed samples) divided by the kernel-weighted
    mean square of the column centered on its overall mean (as the estimator
    centers it); ``roundoff``, that variance divided by the mean square of
    the raw column; and ``undefined``, rows with fewer than 3 observed
    samples with non-zero weight.
    """
    A = np.abs(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        center = np.nan_to_num(np.nanmean(x, axis=0))
    xc = x - center
    T_out = A.shape[0]
    K = x.shape[1]
    cond = np.zeros((T_out, K))
    roundoff = np.zeros((T_out, K))
    undefined = np.zeros((T_out, K), dtype=bool)
    for k in range(K):
        ok = np.isfinite(x[:, k])
        Wk = A * ok
        s = Wk.sum(axis=1)
        v = np.where(ok, xc[:, k], 0.0)
        raw = np.where(ok, x[:, k], 0.0)
        mean = np.divide(Wk @ v, s, out=np.zeros(T_out), where=s > 0)
        var = (Wk * (v[None, :] - mean[:, None]) ** 2).sum(axis=1)
        msq = Wk @ v**2
        rmsq = Wk @ raw**2
        cond[:, k] = np.divide(var, msq, out=np.zeros(T_out), where=msq > 0)
        roundoff[:, k] = np.divide(var, rmsq, out=np.zeros(T_out), where=rmsq > 0)
        undefined[:, k] = (Wk > 0).sum(axis=1) < 3
    return cond, roundoff, undefined


def _entry_classes(columns):
    """
    The mask of well-conditioned entries, shape (T', V) (``mat2vec`` order),
    from per-column conditioning triples, one per input matrix whose columns
    an entry (i, j) reads (the worst of columns i and j over all of them):
    defined, variance at least 1e-3 of the centered mean square, and well
    above the round-off rule's bound.
    """
    conds, rounds, undefs = zip(*columns)
    cond = np.minimum.reduce(conds)
    rnd = np.minimum.reduce(rounds)
    und = np.logical_or.reduce(undefs)

    def pair(m, f):
        P = f(m[:, :, None], m[:, None, :])  # (T', K, K)
        return tc.mat2vec(np.moveaxis(P, 0, -1))

    c = pair(cond, np.minimum)
    r = pair(rnd, np.minimum)
    u = pair(und.astype(float), np.maximum) > 0
    return ~u & (c >= _WELL_CONDITIONED) & (r >= _ROUND_OFF2 * 10)


def _compare(full, part, rows, good_full, good_part):
    """
    ``part`` equals ``full[rows]`` to 1e-9, with the same zeros, on every entry
    that is well conditioned in both computations, so every entry that
    differs is one the conditioning measure flags in at least one of them.
    Returns the masks of the differing and of the well-conditioned entries.
    """
    f = full[rows]
    assert f.shape == part.shape
    good = good_full[rows] & good_part
    differs = (np.abs(f - part) > ATOL) | ((f == 0) != (part == 0))
    assert not np.any(differs & good)
    return differs, good


def _data(T=150, K=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, K))
    x[:, 0] += 3.0  # an offset
    x[:, 2] = 2.0 * (1 + 1e-16 * rng.standard_normal(T))  # constant up to round-off
    x[rng.random((T, K)) < 0.04] = np.nan
    x[0, 1] = np.nan
    return x


def _kernel_args(form):
    if form == "built-in":
        return dict(weights_function=tc.gaussian_weights, weights_params={"var": VAR})
    return dict(weights_function=lambda T, p: tc.gaussian_weights(T, {"var": VAR}))


def _cfun(name):
    if name == "wisfc":
        # wisfc's default subject weights come from static correlations over
        # all timepoints (not causal by design); fixed weights are
        return functools.partial(tc.wisfc, subject_weights=1.0)
    return {"autofc": tc.autofc, "isfc": tc.isfc}[name]


def _isfc_reference(subjects, s):
    others = [d for i, d in enumerate(subjects) if i != s]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.stack(others), axis=0)


def _good(subjects, s, cfun, side, params=None):
    """Well-conditioned entries of subject s's result (``_entry_classes``)."""
    w = helpers._as_kernel_array(helpers._lag_kernel(
        len(subjects[s]), tc.gaussian_weights, params or {"var": VAR}, side))
    cols = [_column_conditioning(subjects[s], w)]
    if cfun != "autofc":
        cols.append(_column_conditioning(_isfc_reference(subjects, s), w))
    return _entry_classes(cols)


def _cut(x, n, side, mode, rng):
    """Remove, or replace with other data, the samples that a one-sided
    kernel's first n rows (last n rows, for forward) do not weight."""
    T = len(x)
    if side == "backward":
        if mode == "remove":
            return x[:n]
        y = x.copy()
        y[n:] = rng.standard_normal(y[n:].shape) * 50 - 20
        return y
    if mode == "remove":
        return x[T - n:]
    y = x.copy()
    y[: T - n] = rng.standard_normal(y[: T - n].shape) * 50 - 20
    return y


def _rows(T, n, side, mode):
    """(rows of the full result, rows of the cut result) that must agree."""
    full = np.arange(n) if side == "backward" else np.arange(T - n, T)
    if mode == "remove":
        return full, np.arange(n)
    return full, full


@pytest.mark.parametrize("side", ["backward", "forward"])
@pytest.mark.parametrize("form", ["built-in", "array"])
@pytest.mark.parametrize("cfun", ["autofc", "isfc", "wisfc"])
@pytest.mark.parametrize("mode", ["remove", "replace"])
def test_order1_well_conditioned_estimates_ignore_the_other_side(side, form, cfun, mode):
    rng = np.random.default_rng(1)
    subjects = [_data(seed=s) for s in range(3)]
    T, n = 150, 90
    kw = dict(cfun=_cfun(cfun), combine=tc.null_combine, include_timepoints=side,
              **_kernel_args(form))
    full = _quiet(tc.timecorr, subjects, **kw)
    cut = [_cut(d, n, side, mode, rng) for d in subjects]
    part = _quiet(tc.timecorr, cut, **kw)
    rows_full, rows_part = _rows(T, n, side, mode)
    for s in range(3):
        good = _good(subjects, s, cfun, side)
        assert good.mean() > 0.5  # most entries are well conditioned
        _compare(full[s], part[s][rows_part], rows_full, good,
                 _good(cut, s, cfun, side)[rows_part])


@pytest.mark.parametrize("side", ["backward", "forward"])
@pytest.mark.parametrize("cfun", ["autofc", "isfc", "wisfc"])
@pytest.mark.parametrize("mode", ["remove", "replace"])
def test_order2_well_conditioned_features_ignore_the_other_side(side, cfun, mode):
    rng = np.random.default_rng(2)
    subjects = [_data(T=160, seed=10 + s) for s in range(3)]
    T, n = 160, 100
    kw = dict(order=2, cfun=_cfun(cfun), rfun=None, delta_rfun=None,
              include_timepoints=side, weights_params={"var": VAR},
              delta_weights_params={"var": VAR}, return_delta=True)
    X, D = _quiet(tc.features, subjects, **kw)
    cut = [_cut(d, n, side, mode, rng) for d in subjects]
    Xc, Dc = _quiet(tc.features, cut, **kw)
    rows_full, rows_part = _rows(T, n, side, mode)
    d1 = [np.asarray(d) for d in D[1]]
    d1c = [np.asarray(d) for d in Dc[1]]
    for s in range(3):
        # order 1 and the feeding path read the data; order 2 reads D[1]
        _compare(np.asarray(X[1][s]), np.asarray(Xc[1][s])[rows_part], rows_full,
                 _good(subjects, s, cfun, side), _good(cut, s, cfun, side)[rows_part])
        _compare(d1[s], d1c[s][rows_part], rows_full,
                 _good(subjects, s, "autofc", side),
                 _good(cut, s, "autofc", side)[rows_part])
        _compare(np.asarray(X[2][s]), np.asarray(Xc[2][s])[rows_part], rows_full,
                 _good(d1, s, cfun, side), _good(d1c, s, cfun, side)[rows_part])


def test_backward_kernel_reproduction():
    # the reported case: with rfun=None, D[1] holds the diagonal of the
    # order-1 correlations, 0 at the 2 undefined rows of the backward kernel
    # and exactly 1 after.  The kernel's tail on those rows gives these
    # columns a genuine, tiny variance, about 1e-10 of the mean square about
    # the column's overall mean, which differs between T 300 and T 200: rows
    # 69 to 79 of X[2] differ (row 79 by 1: 1 in one call, 0 in the other).
    # Those entries are flagged as ill conditioned; all others agree.
    x = np.random.default_rng(0).standard_normal((300, 5))
    kw = dict(order=2, cfun=tc.autofc, rfun=None, delta_rfun=None,
              include_timepoints="backward", return_delta=True)
    X, D = _quiet(tc.features, x, **kw)
    Xs, Ds = _quiet(tc.features, x[:200], **kw)
    rows = np.arange(200)
    assert np.allclose(D[1][:200], Ds[1], atol=ATOL, rtol=0)
    assert np.allclose(X[1][:200], Xs[1], atol=ATOL, rtol=0)
    # order 2: the analysis path (default kernel) and the feeding path
    # (Gaussian of variance 25) read D[1]
    for out, out_s, params in ((X[2], Xs[2], helpers.gaussian_params),
                               (D[2], Ds[2], {"var": 25})):
        good = [_good([np.asarray(d)], 0, "autofc", "backward", params)
                for d in (D[1], Ds[1])]
        _compare(np.asarray(out), np.asarray(out_s), rows, good[0], good[1])
    good = [_good([np.asarray(d)], 0, "autofc", "backward", helpers.gaussian_params)
            for d in (D[1], Ds[1])]
    differs, both_good = _compare(np.asarray(X[2]), np.asarray(Xs[2]), rows,
                                  good[0], good[1])
    # the case is real: entries differ (all of them flagged), most agree
    assert differs.any() and both_good.mean() > 0.5
