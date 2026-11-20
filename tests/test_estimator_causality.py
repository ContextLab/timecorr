"""
One-sided kernels are causal.

With ``include_timepoints="backward"`` the estimate at row t weights only
samples at or before t, so it must not depend on later samples; with
``"forward"``, mirrored, not on earlier ones.  That includes the decision to
treat a variance as zero: the estimator centers each column on a reference
that a one-sided kernel's rows do not look past (the first observed, finite
sample; the last for a forward kernel), not on the mean over all timepoints.

Entries whose kernel-weighted variance nearly cancels are sensitive to the
order of floating-point summation, which changes with the number of
timepoints.  As in ``test_time_sliced_estimation.py``, they are compared by a
conditioning measure computed from the data and the kernel: well-conditioned
entries agree to 1e-9, and the others by their pattern of zeros where the
estimator's zero-variance rule is unambiguous.
"""

import functools
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import helpers

EPS = np.finfo(np.float64).eps
_WELL_CONDITIONED = 1e-3
_SURELY_ZERO = helpers._ZERO_VARIANCE_TOL / 10
_SURELY_NONZERO = helpers._ZERO_VARIANCE_TOL * 10
_ROUND_OFF2 = helpers._ROUND_OFF**2
ATOL = 1e-9


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _reference(x, forward):
    """The estimator's center: each column's first (last) observed finite sample."""
    out = np.zeros(x.shape[1])
    for k in range(x.shape[1]):
        ok = np.flatnonzero(np.isfinite(x[:, k]))
        if ok.size:
            out[k] = x[ok[-1] if forward else ok[0], k]
    return out


def _column_conditioning(x, w, forward):
    """
    Per kernel row and column (T', K): ``cond``, the kernel-weighted variance
    (two passes, over the observed samples) divided by the kernel-weighted
    mean square of the column centered on the estimator's reference;
    ``roundoff``, that variance divided by the mean square of the raw
    column; and ``undefined``, rows with fewer than 3 observed samples with
    non-zero weight.
    """
    A = np.abs(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0))
    xc = x - _reference(x, forward)
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
    ``(good, ambiguous)`` masks of shape (T', V) (``mat2vec`` order) from
    a list of per-column conditioning triples, one per input matrix whose
    columns an entry (i, j) reads (entry conditioning: the worst of columns
    i and j over all of them).
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
    good = ~u & (c >= _WELL_CONDITIONED) & (r >= _ROUND_OFF2 * 10)
    # near a threshold of the zero-variance rule, round-off may decide
    ambiguous = ((c > _SURELY_ZERO) & (c < _SURELY_NONZERO)) | (
        (r > _ROUND_OFF2 / 10) & (r < _ROUND_OFF2 * 10)
    )
    return good, ambiguous


def _compare(full, part, rows, classes):
    """``part`` equals ``full[rows]`` on the well-conditioned entries, and has
    the same zeros wherever the zero-variance rule is not near a threshold."""
    good, ambiguous = (m[rows] for m in classes)
    f = full[rows]
    assert f.shape == part.shape
    assert np.allclose(f[good], part[good], atol=ATOL, rtol=0)
    clear = ~ambiguous
    assert np.array_equal(f[clear] == 0, part[clear] == 0)


def _data(T=150, K=4, seed=0, nan=True, near_constant=True):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, K))
    x[:, 0] += 3.0  # an offset, so the overall mean and the first sample differ
    if near_constant:
        x[:, 2] = 2.0 * (1 + 1e-16 * rng.standard_normal(T))
    if nan:
        x[rng.random((T, K)) < 0.04] = np.nan
        x[0, 1] = np.nan  # the first observed sample of column 1 is not row 0
    return x


def _kernel_args(form, var):
    if form == "built-in":
        return dict(weights_function=tc.gaussian_weights, weights_params={"var": var})
    return dict(weights_function=lambda T, p: tc.gaussian_weights(T, {"var": var}))


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


def _classes(subjects, s, cfun, w, forward):
    cols = [_column_conditioning(subjects[s], w, forward)]
    if cfun != "autofc":
        cols.append(_column_conditioning(_isfc_reference(subjects, s), w, forward))
    return _entry_classes(cols)


def _cut(x, n, side, mode, rng):
    """Remove (or replace) the samples a one-sided kernel's first n rows (last
    n rows, for forward) do not weight."""
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


VAR = 30


@pytest.mark.parametrize("side", ["backward", "forward"])
@pytest.mark.parametrize("form", ["built-in", "array"])
@pytest.mark.parametrize("cfun", ["autofc", "isfc", "wisfc"])
@pytest.mark.parametrize("mode", ["remove", "replace"])
def test_order1_one_sided_estimates_ignore_the_other_side(side, form, cfun, mode):
    rng = np.random.default_rng(1)
    subjects = [_data(seed=s) for s in range(3)]
    T, n = 150, 90
    kw = dict(cfun=_cfun(cfun), combine=tc.null_combine, include_timepoints=side,
              **_kernel_args(form, VAR))
    full = _quiet(tc.timecorr, subjects, **kw)
    cut = [_cut(d, n, side, mode, rng) for d in subjects]
    part = _quiet(tc.timecorr, cut, **kw)
    w = helpers._as_kernel_array(
        helpers._lag_kernel(T, tc.gaussian_weights, {"var": VAR}, side))
    rows_full, rows_part = _rows(T, n, side, mode)
    for s in range(3):
        classes = _classes(subjects, s, cfun, w, side == "forward")
        assert classes[0].mean() > 0.5  # most entries are well conditioned
        _compare(full[s], part[s][rows_part], rows_full, classes)


@pytest.mark.parametrize("side", ["backward", "forward"])
@pytest.mark.parametrize("cfun", ["autofc", "isfc", "wisfc"])
@pytest.mark.parametrize("mode", ["remove", "replace"])
def test_order2_one_sided_features_ignore_the_other_side(side, cfun, mode):
    rng = np.random.default_rng(2)
    subjects = [_data(T=160, K=4, seed=10 + s) for s in range(3)]
    T, n = 160, 100
    kw = dict(order=2, cfun=_cfun(cfun), rfun=None, delta_rfun=None,
              include_timepoints=side, weights_params={"var": VAR},
              return_delta=True)
    X, D = _quiet(tc.features, subjects, **kw)
    cut = [_cut(d, n, side, mode, rng) for d in subjects]
    Xc, Dc = _quiet(tc.features, cut, **kw)
    rows_full, rows_part = _rows(T, n, side, mode)
    w = helpers._as_kernel_array(
        helpers._lag_kernel(T, tc.gaussian_weights, {"var": VAR}, side))
    forward = side == "forward"
    for s in range(3):
        # order 1 reads the data; order 2 reads the feeding features D[1]
        c1 = _classes(subjects, s, cfun, w, forward)
        _compare(np.asarray(X[1][s]), np.asarray(Xc[1][s])[rows_part], rows_full, c1)
        d1 = [np.asarray(d) for d in D[1]]
        c2 = _classes(d1, s, cfun, w, forward)
        _compare(np.asarray(X[2][s]), np.asarray(Xc[2][s])[rows_part], rows_full, c2)


def test_backward_kernel_reproduction():
    # the reported case: with rfun=None, D[1] holds the diagonal of the
    # order-1 correlations, 0 at the 2 undefined rows of the backward kernel
    # and exactly 1 after; the kernel's tail on those rows gives a genuine,
    # tiny variance, and the decision to treat it as zero must not depend on
    # the samples after row 200
    x = np.random.default_rng(0).standard_normal((300, 5))
    kw = dict(order=2, cfun=tc.autofc, rfun=None, delta_rfun=None,
              include_timepoints="backward", return_delta=True)
    X, D = _quiet(tc.features, x, **kw)
    Xs, Ds = _quiet(tc.features, x[:200], **kw)
    rows = np.arange(200)
    for k in (1, 2):
        assert np.allclose(D[k][:200], Ds[k], atol=ATOL, rtol=0)
    assert np.allclose(X[1][:200], Xs[1], atol=ATOL, rtol=0)
    w = helpers._as_kernel_array(
        helpers._lag_kernel(300, tc.gaussian_weights, tc.gaussian_params, "backward"))
    classes = _entry_classes([_column_conditioning(np.asarray(D[1]), w, False)])
    # the case is real: some entries are ill conditioned
    assert not classes[0][:200].all()
    _compare(np.asarray(X[2]), np.asarray(Xs[2]), rows, classes)
    # row 79, where the previous rule kept the diagonal columns' correlations
    # (1) at T 300 and set them to 0 at T 200, now agrees
    assert np.array_equal(X[2][79] == 0, Xs[2][79] == 0)


def _exact_wcorr(x, w):
    """Kernel-weighted correlations in exact rational arithmetic (fractions)."""
    from fractions import Fraction

    T, K = x.shape
    out = np.zeros((K, K, w.shape[0]))
    X = [[Fraction(float(v)) for v in x[:, k]] for k in range(K)]
    for t in range(w.shape[0]):
        W = [Fraction(float(v)) for v in w[t]]
        A = [abs(v) for v in W]
        s = sum(A)
        if sum(1 for v in W if v != 0) < 3:
            continue
        means = [sum(a * v for a, v in zip(A, col)) / s for col in X]
        dev = [[v - m for v in col] for col, m in zip(X, means)]
        var = [sum(a * d * d for a, d in zip(A, dv)) for dv in dev]
        for i in range(K):
            for j in range(K):
                if var[i] == 0 or var[j] == 0:
                    continue
                num = sum(wv * di * dj for wv, di, dj in zip(W, dev[i], dev[j]))
                out[i, j, t] = float(num) / np.sqrt(float(var[i]) * float(var[j]))
    return out


@pytest.mark.parametrize("outlier", [1e6, -1e9])
@pytest.mark.parametrize("side", ["all", "backward"])
def test_first_sample_outlier_gives_exact_correlations(outlier, side):
    # the first observed sample is the reference of a backward kernel's
    # centering; an outlier there must not cost accuracy elsewhere
    rng = np.random.default_rng(3)
    T = 60
    x = rng.standard_normal((T, 3))
    x[:, 1] += 0.6 * x[:, 0]
    x[0, 0] = outlier
    w = helpers._as_kernel_array(
        helpers._lag_kernel(T, tc.gaussian_weights, {"var": 20}, side))
    out = _quiet(helpers.wcorr, x, x, w)
    ref = _exact_wcorr(x, w)
    # the diagonal of an autocorrelation is set to exactly 1 (0 where undefined)
    for t in range(T):
        defined = np.count_nonzero(w[t]) >= 3
        for i in range(3):
            ref[i, i, t] = 1.0 if defined else 0.0
    assert np.allclose(out, ref, atol=1e-9, rtol=0)
