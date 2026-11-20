"""
Squared-correlation summaries (``timecorr.summaries``): ``WeightedStrength`` and
``Communities``, alone and as ``rfun`` of ``reduce`` and ``timecorr``.

The summaries run on real correlations of ``simulate_nested`` data (computed by
``timecorr`` with ``autofc``) and on small hand-built correlation series; the
expected values are computed directly in the tests with ``vec2mat`` and
``einsum``.  Nothing is mocked.
"""

import warnings

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

import timecorr as tc
from timecorr import summaries as sm

N_SG, PER_SG, PER_MOD = 3, 2, 4
K_NESTED = N_SG * PER_SG * PER_MOD  # 24 nodes in 6 modules


@pytest.fixture(scope="module")
def nested():
    data, truth = tc.simulate_nested(
        S=2,
        T=600,
        n_supergroups=N_SG,
        modules_per_supergroup=PER_SG,
        nodes_per_module=PER_MOD,
        set_random_seed=0,
    )
    corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None, weights_params={"var": 25})
    return data, truth, corrs


@pytest.fixture(scope="module")
def nested_other():
    """A second recording with the same number of nodes (another seed)."""
    data, truth = tc.simulate_nested(
        S=3,
        T=400,
        n_supergroups=N_SG,
        modules_per_supergroup=PER_SG,
        nodes_per_module=PER_MOD,
        set_random_seed=7,
    )
    return tc.timecorr(data, cfun=tc.autofc, rfun=None, weights_params={"var": 25})


def _random_corr_series(K, T, seed):
    """A (T, K(K+1)/2) series of valid correlation matrices from smoothed noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T + 40, K))
    x = np.cumsum(x, axis=0)[40:] + rng.standard_normal((T, K)) * 5
    return tc.timecorr(x, cfun=tc.autofc, rfun=None, weights_params={"var": 9})


# ----------------------------------------------------------------------------
# direct formulas
# ----------------------------------------------------------------------------


def _offdiag(v):
    M = tc.vec2mat(v).copy()
    K = M.shape[0]
    M[np.arange(K), np.arange(K), :] = 0
    return M


def _pair_weights(M):
    return (M**2).mean(axis=2)


def _clipped(w):
    """Clipped weights of ``WeightedStrength(center="median")`` (the median
    rule; the tests that use this reference pass ``center="median"``)."""
    K = w.shape[0]
    if K < 4:
        a = w.copy()
    else:
        iu = np.triu_indices(K, 1)
        a = np.maximum(w - np.median(w[iu]), 0)
    np.fill_diagonal(a, 0)
    return a


def ref_weighted_strength(v, signed=False, labels=None, w=None, mean_r=None):
    M = _offdiag(v)
    if w is None:
        w = _pair_weights(M)
    if mean_r is None:
        mean_r = M.mean(axis=2)
    np.fill_diagonal(mean_r, 0)
    a = _clipped(w)
    if labels is None:
        cols = [np.einsum("ij,ijt->ti", a, M**2)]
        if signed:
            cols.append(np.einsum("ij,ijt->ti", mean_r, M))
    else:
        own = labels[:, None] == labels[None, :]
        cols = [
            np.einsum("ij,ijt->ti", a * own, M**2),
            np.einsum("ij,ijt->ti", a * ~own, M**2),
        ]
        if signed:
            cols += [
                np.einsum("ij,ijt->ti", mean_r * own, M),
                np.einsum("ij,ijt->ti", mean_r * ~own, M),
            ]
    return np.hstack(cols)


def ref_communities(v, labels, signed=False, mean_r=None):
    M = _offdiag(v)
    if mean_r is None:
        mean_r = M.mean(axis=2)
    mag, sgn = [], []
    for g in np.unique(labels):
        idx = np.flatnonzero(labels == g)
        m = len(idx)
        if m < 2:
            continue
        sub = M[np.ix_(idx, idx)]
        mag.append((sub**2).sum(axis=(0, 1)) / (m * (m - 1)))
        s = np.sign(mean_r[np.ix_(idx, idx)])
        sgn.append((s[:, :, None] * sub).sum(axis=(0, 1)) / (m * (m - 1)))
    cols = mag + (sgn if signed else [])
    return np.array(cols).T


def _renumbered(labels):
    """True if labels are numbered in order of each label's smallest member."""
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(first)
    return np.array_equal(np.unique(labels)[order], np.arange(len(first)))


# ----------------------------------------------------------------------------
# WeightedStrength
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("signed", [False, True])
def test_weighted_strength_formula(nested, signed):
    _, _, corrs = nested
    out = tc.WeightedStrength(signed=signed, center="median")(corrs)
    assert isinstance(out, list) and len(out) == len(corrs)
    for o, v in zip(out, corrs):
        assert o.shape == (v.shape[0], (2 if signed else 1) * K_NESTED)
        np.testing.assert_allclose(o, ref_weighted_strength(v, signed), atol=1e-10, rtol=0)


def test_weighted_strength_ignores_diagonal(nested):
    _, _, corrs = nested
    v = corrs[0].copy()
    v2 = v.copy()
    v2[:, :K_NESTED] = 5.0  # a different diagonal
    a = tc.WeightedStrength(signed=True)([v])[0]
    b = tc.WeightedStrength(signed=True)([v2])[0]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("K", [2, 3])
def test_weighted_strength_small_K_unclipped(K):
    v = _random_corr_series(K, 120, seed=K)
    out = tc.WeightedStrength()(v)
    assert isinstance(out, np.ndarray) and out.shape == (120, K)
    M = _offdiag(v)
    w = _pair_weights(M)
    np.testing.assert_allclose(out, np.einsum("ij,ijt->ti", w, M**2), atol=1e-12, rtol=0)
    assert np.any(out != 0)


@pytest.mark.parametrize("signed", [False, True])
def test_weighted_strength_split(nested, signed):
    _, _, corrs = nested
    split = tc.WeightedStrength(split=True, signed=signed, n=6, center="median")
    out = split(corrs)
    whole = tc.WeightedStrength(signed=signed, center="median")(corrs)
    K = K_NESTED
    labels = split.fits_[-1]["labels"]
    assert labels.shape == (K,)
    for o, u, v in zip(out, whole, corrs):
        assert o.shape == (v.shape[0], (4 if signed else 2) * K)
        # magnitude [own, other], then signed [own, other]
        np.testing.assert_allclose(o[:, :K] + o[:, K : 2 * K], u[:, :K], atol=1e-12, rtol=0)
        if signed:
            np.testing.assert_allclose(
                o[:, 2 * K : 3 * K] + o[:, 3 * K :], u[:, K:], atol=1e-12, rtol=0
            )
        np.testing.assert_allclose(
            o, ref_weighted_strength(v, signed, labels=labels), atol=1e-10, rtol=0
        )


# ----------------------------------------------------------------------------
# Communities
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("signed", [False, True])
def test_communities_recovers_modules(nested, signed):
    _, truth, corrs = nested
    comm = tc.Communities(n=6, signed=signed)
    out = comm(corrs)
    labels = comm.fits_[-1]["labels"]
    assert adjusted_rand_score(truth["module"], labels) == 1.0
    assert _renumbered(labels)
    for o, v in zip(out, corrs):
        assert o.shape == (v.shape[0], (12 if signed else 6))
        np.testing.assert_allclose(o, ref_communities(v, labels, signed), atol=1e-10, rtol=0)


def test_communities_singleton_has_no_column():
    # nodes 0-3 and 4-7 correlated in two groups; node 8 independent of both
    rng = np.random.default_rng(3)
    T = 300
    f1, f2 = rng.standard_normal((2, T))
    x = np.column_stack(
        [f1 + 0.5 * rng.standard_normal(T) for _ in range(4)]
        + [f2 + 0.5 * rng.standard_normal(T) for _ in range(4)]
        + [rng.standard_normal(T)]
    )
    v = tc.timecorr(x, cfun=tc.autofc, rfun=None, weights_params={"var": 25})
    comm = tc.Communities(n=3)
    out = comm(v)
    labels = comm.fits_[-1]["labels"]
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 1, 1, 1, 1, 2])
    assert out.shape == (T, 2)
    np.testing.assert_allclose(out, ref_communities(v, labels), atol=1e-12, rtol=0)
    signed = tc.Communities(n=3, signed=True)(v)
    assert signed.shape == (T, 4)


# ----------------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------------


def test_n_none_raises(nested):
    _, _, corrs = nested
    for obj in (tc.Communities(), tc.WeightedStrength(split=True)):
        with pytest.raises(ValueError, match=r"\bn\b"):
            obj(corrs)
    with pytest.raises(ValueError, match=r"\bn\b"):
        tc.reduce(corrs, "communities")
    with pytest.raises(ValueError, match=r"\bn\b"):
        tc.reduce(corrs, "weighted_strength_split")


@pytest.mark.parametrize("K", [2, 3])
def test_small_K_needs_weighted_strength(K):
    v = _random_corr_series(K, 100, seed=10 + K)
    with pytest.raises(ValueError, match="WeightedStrength"):
        tc.Communities(n=1)(v)
    with pytest.raises(ValueError, match="WeightedStrength"):
        tc.WeightedStrength(split=True, n=1)(v)


def test_nan_raises(nested):
    _, _, corrs = nested
    v = corrs[0].copy()
    v[10, 40] = np.nan
    for obj in (tc.WeightedStrength(), tc.Communities(n=6)):
        with pytest.raises(ValueError, match="NaN"):
            obj([v])
    fitted = tc.WeightedStrength().fit(corrs)
    with pytest.raises(ValueError, match="NaN"):
        fitted.transform([v])


def test_ndims_with_summary_raises(nested):
    _, _, corrs = nested
    for rfun in ("weighted_strength", tc.WeightedStrength()):
        with pytest.raises(ValueError, match="ndims=3 applies to model-based"):
            tc.reduce(corrs, rfun, ndims=3)


def test_bad_arguments(nested):
    _, _, corrs = nested
    with pytest.raises(ValueError, match="n must be a positive integer"):
        tc.Communities(n=0)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        tc.Communities(n=2.5)
    with pytest.raises(ValueError):
        tc.Communities(n=K_NESTED + 1)(corrs)
    with pytest.raises(ValueError, match=r"expected length K\(K\+1\)/2"):
        tc.WeightedStrength()([np.zeros((10, 5))])  # 5 is not K(K+1)/2
    with pytest.raises(ValueError, match="fit"):
        tc.WeightedStrength().transform(corrs)
    fitted = tc.WeightedStrength().fit(corrs)
    with pytest.raises(ValueError, match="nodes"):
        fitted.transform([np.zeros((10, 6))])


# ----------------------------------------------------------------------------
# fit / transform and fits_
# ----------------------------------------------------------------------------


def test_fits_records(nested):
    _, _, corrs = nested
    K = K_NESTED
    ws = tc.WeightedStrength(signed=True)
    assert ws.fits_ == []
    ws.fit(corrs)
    ws.fit(corrs, order=2, path="feeding")
    assert len(ws.fits_) == 2
    keys = {"order", "path", "weights", "mean_r", "labels", "n", "null_threshold"}
    for f in ws.fits_:
        assert set(f) == keys
        assert len(f["weights"]) == 2 and all(w.shape == (K, K) for w in f["weights"])
        assert len(f["mean_r"]) == 2 and all(m.shape == (K, K) for m in f["mean_r"])
        assert f["labels"] is None and f["n"] is None and f["null_threshold"] is None
    assert ws.fits_[0]["order"] is None and ws.fits_[0]["path"] is None
    assert ws.fits_[1]["order"] == 2 and ws.fits_[1]["path"] == "feeding"
    M = _offdiag(corrs[1])
    np.testing.assert_allclose(ws.fits_[0]["weights"][1], _pair_weights(M), atol=1e-12)
    mr = M.mean(axis=2)
    np.testing.assert_allclose(ws.fits_[0]["mean_r"][1], mr, atol=1e-12)

    comm = tc.Communities(n=6)
    comm.fit(corrs)
    f = comm.fits_[0]
    assert set(f) == keys
    assert f["mean_r"] is None and f["labels"].shape == (K,) and f["n"] == 6


def test_fit_on_one_list_transform_another(nested, nested_other):
    _, _, A = nested
    B = nested_other
    # same number of units: unit-specific weights of the fit
    ws = tc.WeightedStrength(signed=True, center="median").fit(A)
    B2 = B[:2]
    out = ws.transform(B2)
    for i, (o, v) in enumerate(zip(out, B2)):
        expected = ref_weighted_strength(
            v, True, w=ws.fits_[0]["weights"][i], mean_r=ws.fits_[0]["mean_r"][i].copy()
        )
        np.testing.assert_allclose(o, expected, atol=1e-10, rtol=0)
    own = tc.WeightedStrength(signed=True, center="median")(B2)
    assert not all(np.allclose(o, p) for o, p in zip(out, own))

    # different number of units: the mean weights
    out3 = ws.transform(B)
    wbar = np.mean(ws.fits_[0]["weights"], axis=0)
    mbar = np.mean(ws.fits_[0]["mean_r"], axis=0)
    for o, v in zip(out3, B):
        expected = ref_weighted_strength(v, True, w=wbar, mean_r=mbar.copy())
        np.testing.assert_allclose(o, expected, atol=1e-10, rtol=0)

    # Communities: the fit's communities
    comm = tc.Communities(n=6, signed=True).fit(A)
    labels = comm.fits_[0]["labels"]
    outc = comm.transform(B)
    mbar = np.mean(comm.fits_[0]["mean_r"], axis=0)
    for o, v in zip(outc, B):
        np.testing.assert_allclose(
            o, ref_communities(v, labels, True, mean_r=mbar), atol=1e-10, rtol=0
        )
    own_c = tc.Communities(n=6, signed=True)(B)
    assert not all(np.allclose(o, p) for o, p in zip(outc, own_c))


def test_transform_matches_units_by_position(nested):
    # Present behavior (pinned, not endorsed): with as many units as the fit,
    # unit i of the new batch gets the fitted weights of unit i, whatever its
    # identity.  Swapping the two subjects swaps the weights they receive.
    _, _, A = nested
    swapped = [A[1], A[0]]
    ws = tc.WeightedStrength(signed=True, center="median").fit(A)
    fit = ws.fits_[0]
    out = ws.transform(swapped)
    for i, v in enumerate(swapped):
        expected = ref_weighted_strength(
            v, True, w=fit["weights"][i], mean_r=fit["mean_r"][i].copy()
        )
        np.testing.assert_allclose(out[i], expected, atol=1e-10, rtol=0)
    # subject A[1] at position 0 is not summarized with its own weights
    own = ref_weighted_strength(
        A[1], True, w=fit["weights"][1], mean_r=fit["mean_r"][1].copy()
    )
    assert not np.allclose(out[0], own)

    comm = tc.Communities(n=6, signed=True).fit(A)
    cfit = comm.fits_[0]
    outc = comm.transform(swapped)
    for i, v in enumerate(swapped):
        np.testing.assert_allclose(
            outc[i],
            ref_communities(v, cfit["labels"], True, mean_r=cfit["mean_r"][i]),
            atol=1e-10,
            rtol=0,
        )


def _ws_ref(v, fit, unit):
    """Expected signed weighted strength of ``v`` with fitted unit ``unit``."""
    if unit == "mean":
        w = np.mean(fit["weights"], axis=0)
        m = np.mean(fit["mean_r"], axis=0)
    else:
        w, m = fit["weights"][unit], fit["mean_r"][unit]
    return ref_weighted_strength(v, True, w=w, mean_r=m.copy())


def test_fit_index_reorders_outputs_only_when_it_says_so(nested):
    _, _, A = nested
    swapped = [A[1], A[0]]
    # center="median": the rule of the reference ref_weighted_strength
    ws = tc.WeightedStrength(signed=True, center="median").fit(A)
    fit = ws.fits_[0]
    original = ws.transform(A)
    # default: by position, so the swapped batch is not the swapped output
    by_position = ws.transform(swapped)
    assert not np.allclose(by_position[0], original[1])
    # fit_index names each unit's fit: the outputs are the originals, swapped
    named = ws.transform(swapped, fit_index=[1, 0])
    np.testing.assert_allclose(named[0], original[1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(named[1], original[0], atol=1e-12, rtol=0)
    # fit_index=None is the default rule; [0, 1] is position matching spelled out
    for a, b in zip(ws.transform(swapped, fit_index=None), by_position):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(ws.transform(swapped, fit_index=[0, 1]), by_position):
        np.testing.assert_array_equal(a, b)

    comm = tc.Communities(n=6, signed=True).fit(A)
    corig = comm.transform(A)
    cnamed = comm.transform(swapped, fit_index=[1, 0])
    np.testing.assert_allclose(cnamed[0], corig[1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(cnamed[1], corig[0], atol=1e-12, rtol=0)
    assert not np.allclose(comm.transform(swapped)[0], corig[1])


def test_fit_index_transforms_a_held_out_subject_with_a_chosen_fit(
    nested, nested_other
):
    _, _, A = nested
    held_out = nested_other[2]
    # center="median": the rule of the reference ref_weighted_strength
    ws = tc.WeightedStrength(signed=True, center="median").fit(A)
    fit = ws.fits_[0]
    for unit in (0, 1):
        # a single array with a chosen fit
        got = ws.transform(held_out, fit_index=unit)
        assert isinstance(got, np.ndarray)
        np.testing.assert_allclose(got, _ws_ref(held_out, fit, unit), atol=1e-10, rtol=0)
        # an int applies one fit to every unit of a list of any length
        got3 = ws.transform(nested_other, fit_index=unit)
        for o, v in zip(got3, nested_other):
            np.testing.assert_allclose(o, _ws_ref(v, fit, unit), atol=1e-10, rtol=0)
    # without fit_index a single array (1 unit, fit of 2) gets the mean weights
    np.testing.assert_allclose(
        ws.transform(held_out), _ws_ref(held_out, fit, "mean"), atol=1e-10, rtol=0
    )
    # "mean" asks for the mean weights even when the counts are equal
    for o, v in zip(ws.transform(A, fit_index="mean"), A):
        np.testing.assert_allclose(o, _ws_ref(v, fit, "mean"), atol=1e-10, rtol=0)
    # a list of indices may repeat a fit
    got = ws.transform([held_out, held_out], fit_index=[1, 1])
    for o in got:
        np.testing.assert_allclose(o, _ws_ref(held_out, fit, 1), atol=1e-10, rtol=0)
    assert not np.allclose(got[0], _ws_ref(held_out, fit, 0))


@pytest.mark.parametrize(
    "fit_index, match",
    [
        (2, r"fit_index=2 .*2 fitted units"),
        (-1, r"fit_index=-1 .*2 fitted units"),
        (True, "fit_index must be"),
        (1.0, "fit_index must be"),
        ("first", "fit_index must be"),
        ([0], r"1 entries .* 2 units"),
        ([0, 1, 0], r"3 entries .* 2 units"),
        ([0, 2], r"fit_index\[1\]=2 .*2 fitted units"),
        ([0, "mean"], "fit_index must be"),
        ([], r"0 entries .* 2 units"),
    ],
)
def test_bad_fit_index_raises(nested, fit_index, match):
    _, _, A = nested
    for summary in (tc.WeightedStrength(signed=True), tc.Communities(n=6)):
        summary.fit(A)
        with pytest.raises(ValueError, match=match):
            summary.transform(A, fit_index=fit_index)


@pytest.mark.parametrize(
    "spelling, plain",
    [
        (np.int64(1), 1),
        (np.uint8(0), 0),
        (np.array(1), 1),
        (np.array(0, dtype=np.int32), 0),
        (range(2), [0, 1]),
        (range(1, -1, -1), [1, 0]),
        ((1, 0), [1, 0]),
        (np.array([1, 0]), [1, 0]),
        (np.array([1, 1], dtype=np.uint16), [1, 1]),
        ([np.int64(0), np.int32(1)], [0, 1]),
    ],
)
def test_fit_index_accepts_numpy_integers_ranges_tuples_and_arrays(
    nested, spelling, plain
):
    _, _, A = nested
    for summary in (tc.WeightedStrength(signed=True), tc.Communities(n=6, signed=True)):
        summary.fit(A)
        expected = summary.transform(A, fit_index=plain)
        got = summary.transform(A, fit_index=spelling)
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "fit_index, match",
    [
        (np.True_, "fit_index must be"),
        (np.array(True), "fit_index must be"),
        (np.float64(1.0), "fit_index must be"),
        (np.array(1.0), "fit_index must be"),
        (np.array([1.0, 0.0]), r"fit_index must be .*got fit_index\[0\]=.*1\.0"),
        (np.array([True, False]), r"fit_index must be .*got fit_index\[0\]=.*True"),
        ([0, False], r"fit_index must be .*got fit_index\[1\]=False"),
        ((0, 1.0), r"fit_index must be .*got fit_index\[1\]=1\.0"),
        (np.array([[0, 1]]), r"fit_index must be .*1-D.*2 dimensions"),
        (b"mean", "fit_index must be"),
        (bytearray(b"\x00\x01"), "fit_index must be"),
        ({0, 1}, "fit_index must be"),
        (1 + 0j, "fit_index must be"),
    ],
)
def test_fit_index_rejects_bools_floats_and_other_types_clearly(nested, fit_index, match):
    _, _, A = nested
    for summary in (tc.WeightedStrength(signed=True), tc.Communities(n=6)):
        summary.fit(A)
        with pytest.raises(ValueError, match=match):
            summary.transform(A, fit_index=fit_index)


def test_fit_index_with_a_single_array_needs_a_scalar(nested):
    _, _, A = nested
    ws = tc.WeightedStrength().fit(A)
    with pytest.raises(ValueError, match=r"2 entries .* 1 unit"):
        ws.transform(A[0], fit_index=[0, 1])
    np.testing.assert_array_equal(
        ws.transform(A[0], fit_index=[1]), ws.transform(A[0], fit_index=1)
    )


# ----------------------------------------------------------------------------
# null_threshold (the data-driven count)
# ----------------------------------------------------------------------------


def _centered_eigs(corrs):
    w = np.mean([_pair_weights(_offdiag(v)) for v in corrs], axis=0)
    K = w.shape[0]
    C = w - np.median(w[np.triu_indices(K, 1)])
    np.fill_diagonal(C, 0)
    return np.sort(np.linalg.eigvalsh(C))[::-1]


def test_null_threshold_count(nested):
    _, truth, corrs = nested
    ev = _centered_eigs(corrs)
    thr = 0.5 * (ev[5] + ev[6])
    comm = tc.Communities()
    comm.fit(corrs, null_threshold=thr)
    f = comm.fits_[-1]
    assert f["n"] == 6 and f["null_threshold"] == thr
    np.testing.assert_array_equal(f["labels"], tc.Communities(n=6).fit(corrs).fits_[0]["labels"])

    # capped at K // 2
    comm.fit(corrs, null_threshold=ev[-1] - 1.0)
    assert comm.fits_[-1]["n"] == K_NESTED // 2

    # nothing above the threshold: one community holding every node
    with pytest.warns(UserWarning, match="no community structure above the shift null"):
        out = tc.Communities().fit(corrs, null_threshold=ev[0] + 1.0).transform(corrs)
    assert all(o.shape == (v.shape[0], 1) for o, v in zip(out, corrs))
    for o, v in zip(out, corrs):
        M = _offdiag(v)
        np.testing.assert_allclose(
            o[:, 0], (M**2).sum(axis=(0, 1)) / (K_NESTED * (K_NESTED - 1)), atol=1e-12
        )

    # split through the threshold
    ws = tc.WeightedStrength(split=True).fit(corrs, null_threshold=thr)
    assert ws.fits_[-1]["n"] == 6

    # n given: the threshold is recorded and not used for the count
    c6 = tc.Communities(n=3).fit(corrs, null_threshold=thr)
    assert c6.fits_[-1]["n"] == 3


# ----------------------------------------------------------------------------
# row blocks (the streaming interface)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "summary",
    [
        tc.WeightedStrength(),
        tc.WeightedStrength(signed=True),
        tc.WeightedStrength(split=True, signed=True, n=6),
        tc.Communities(n=6),
        tc.Communities(n=6, signed=True),
    ],
    ids=["ws", "ws-signed", "ws-split-signed", "comm", "comm-signed"],
)
def test_block_sizes_agree(nested, summary):
    _, _, corrs = nested
    K = K_NESTED
    R_all = [tc.vec2mat(v) for v in corrs]
    summary.fits_ = []
    expected = summary.fit(corrs).transform(corrs)
    fit = summary.fits_[-1]
    for bs in (1, 2, K):
        states = []
        for R in R_all:
            state = sm._new_state(K, R.shape[2], summary.signed)
            for i0 in range(0, K, bs):
                i1 = min(i0 + bs, K)
                sm._accumulate(state, i0, i1, R[i0:i1])
            states.append(state)
        one = []
        for R in R_all:
            state = sm._new_state(K, R.shape[2], summary.signed)
            sm._accumulate(state, 0, K, R)
            one.append(state)
        for s, o in zip(states, one):
            np.testing.assert_array_equal(s["sum_r2"], o["sum_r2"])
            if summary.signed:
                np.testing.assert_array_equal(s["sum_r"], o["sum_r"])
        for u, R in enumerate(R_all):
            unit = summary._unit_params(fit, u)
            out = np.zeros((R.shape[2], summary._n_columns(fit, K)))
            for i0 in range(0, K, bs):
                i1 = min(i0 + bs, K)
                summary._emit(fit, unit, i0, i1, R[i0:i1], out)
            np.testing.assert_allclose(out, expected[u], atol=1e-12, rtol=0)


def test_one_eigenvalue_gives_two_communities():
    # 4 strongly correlated nodes plus 8 independent nodes: one eigenvalue of
    # C stands out; a threshold between the first and second eigenvalue gives
    # max(2, 1) = 2 communities, with the 4 nodes together
    rng = np.random.default_rng(5)
    T = 400
    f = rng.standard_normal(T)
    x = np.column_stack(
        [f + 0.4 * rng.standard_normal(T) for _ in range(4)]
        + [rng.standard_normal(T) for _ in range(8)]
    )
    v = tc.timecorr(x, cfun=tc.autofc, rfun=None, weights_params={"var": 25})
    ev = _centered_eigs([v])
    thr = 0.5 * (ev[0] + ev[1])
    comm = tc.Communities().fit(v, null_threshold=thr)
    f0 = comm.fits_[-1]
    labels = f0["labels"]
    assert f0["n"] == 2 and len(np.unique(labels)) == 2
    assert len(np.unique(labels[:4])) == 1
    assert not np.any(labels[4:] == labels[0])


def test_explicit_n_above_half_K_raises(nested):
    _, _, corrs = nested
    with pytest.raises(ValueError, match=r"exceed K // 2"):
        tc.Communities(n=K_NESTED // 2 + 1)(corrs)
    with pytest.raises(ValueError, match=r"exceed K // 2"):
        tc.WeightedStrength(split=True, n=K_NESTED)(corrs)
    tc.Communities(n=K_NESTED // 2)(corrs)  # the cap itself is allowed


def test_no_community_with_two_members_raises():
    # with n <= K // 2 some community always has 2 members; the check guards
    # the labels themselves
    with pytest.raises(ValueError, match="no community with at least 2 members"):
        sm._check_labels(np.arange(5), "Communities")
    sm._check_labels(np.array([0, 0, 1, 2]), "Communities")


def test_fits_n_is_number_found():
    v = np.zeros((50, 15))  # K = 5: identical embeddings, one cluster
    comm = tc.Communities(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        comm(v)
    labels = comm.fits_[-1]["labels"]
    assert comm.fits_[-1]["n"] == len(np.unique(labels))
    # n=2 was requested; identical embeddings give one community
    assert comm.fits_[-1]["n"] == 1


def test_split_flag_drives_communities():
    ws = tc.WeightedStrength()
    assert not ws._needs_communities
    ws.split = True
    assert ws._needs_communities
    ws.split = False
    assert not ws._needs_communities


def test_private_helpers():
    w = np.array(
        [[0.0, 0.5, 0.1, 0.2], [0.5, 0.0, 0.3, 0.05], [0.1, 0.3, 0.0, 0.4],
         [0.2, 0.05, 0.4, 0.0]]
    )
    sum_r2 = w * 10
    np.fill_diagonal(sum_r2, 10.0)
    np.testing.assert_allclose(sm._pair_weights(sum_r2, 10), w)
    a = sm._clip_weights(w, "median")
    # median of (0.5, 0.1, 0.2, 0.3, 0.05, 0.4) is 0.25
    np.testing.assert_allclose(a[0, 1], 0.25)
    assert a[0, 2] == 0 and np.all(np.diag(a) == 0)
    C = sm._centered(w)
    np.testing.assert_allclose(C[0, 1], 0.25)
    assert np.all(np.diag(C) == 0)
    ev = np.linalg.eigvalsh(C)
    assert sm._count_above(w, ev[-1] + 1, 4) == 0
    assert sm._count_above(w, ev[0] - 1, 4) == 2
    labels = sm._find_communities(w, 2)
    assert labels.shape == (4,) and labels[0] == 0 and _renumbered(labels)


# ----------------------------------------------------------------------------
# reduce and timecorr
# ----------------------------------------------------------------------------


def test_through_reduce_and_timecorr(nested):
    data, _, corrs = nested
    expected = tc.WeightedStrength()(corrs)
    via_obj = tc.reduce(corrs, tc.WeightedStrength())
    via_str = tc.reduce(corrs, "weighted_strength")
    for e, a, b in zip(expected, via_obj, via_str):
        np.testing.assert_array_equal(e, a)
        np.testing.assert_array_equal(e, b)
    single = tc.reduce(corrs[0], "weighted_strength")
    assert isinstance(single, np.ndarray)
    np.testing.assert_allclose(single, tc.WeightedStrength()(corrs[0]), atol=0)

    obj = tc.Communities(n=6)
    out = tc.reduce(corrs, obj)
    assert len(obj.fits_) == 1 and len(out) == 2

    split = tc.reduce(corrs, tc.WeightedStrength(split=True, n=6))
    assert split[0].shape == (600, 2 * K_NESTED)

    via_tc = tc.timecorr(
        data, cfun=tc.autofc, rfun="weighted_strength", weights_params={"var": 25}
    )
    for e, a in zip(expected, via_tc):
        np.testing.assert_allclose(a, e, atol=1e-12, rtol=0)


def test_rfun_classification():
    from timecorr.helpers import _check_rfun, _is_model

    for r in ("weighted_strength", "weighted_strength_split", "communities"):
        _check_rfun(r)
        assert not _is_model(r)
    for r in (tc.WeightedStrength(), tc.Communities(n=2)):
        _check_rfun(r)
        assert not _is_model(r)
    assert _is_model("PCA")


def test_strings_make_fresh_instances(nested):
    _, _, corrs = nested
    a = tc.reduce(corrs, "weighted_strength")
    b = tc.reduce(corrs, "weighted_strength")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# ----------------------------------------------------------------------------
# determinism and warnings
# ----------------------------------------------------------------------------


def test_deterministic_and_global_state(nested):
    _, _, corrs = nested
    np.random.seed(12345)
    before = np.random.get_state()
    a = tc.Communities(n=6, signed=True)(corrs)
    b = tc.Communities(n=6, signed=True)(corrs)
    c = tc.WeightedStrength(split=True, n=6)(corrs)
    d = tc.WeightedStrength(split=True, n=6)(corrs)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for x, y in zip(a + c, b + d):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "summary, name",
    [
        (tc.WeightedStrength(), "WeightedStrength"),
        (tc.WeightedStrength(signed=True), "WeightedStrength"),
        (tc.Communities(n=2), "Communities"),
    ],
)
def test_constant_output_warns(summary, name):
    v = np.zeros((50, 15))  # K = 5, every correlation 0
    with pytest.warns(UserWarning, match=name) as record:
        out = summary(v)
    assert len([r for r in record if name in str(r.message)]) == 1
    assert np.all(np.isfinite(out))


def test_no_warning_on_real_correlations(nested):
    _, _, corrs = nested
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tc.WeightedStrength(signed=True)(corrs)
        tc.Communities(n=6, signed=True)(corrs)


def test_named_summaries_match_the_recognised_names():
    from timecorr.helpers import _SUMMARY_NAMES

    assert set(sm._NAMED) == set(_SUMMARY_NAMES)


# ----------------------------------------------------------------------------
# input validation (review item 7)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "summary",
    [tc.WeightedStrength(), tc.WeightedStrength(signed=True), tc.Communities(n=2)],
    ids=["ws", "ws-signed", "comm"],
)
def test_non_finite_and_empty_inputs_raise(summary):
    K = 5
    v = _random_corr_series(K, 60, seed=3)
    inf = v.copy()
    inf[7, 9] = np.inf
    # the diagonal (columns 0..K-1 of mat2vec) is ignored by the formulas,
    # but 0 * inf there would still give NaN outputs
    diag = v.copy()
    diag[5, 2] = -np.inf
    for bad in (np.full((2, 15), np.inf), inf, diag):
        with pytest.raises(ValueError, match="finite"):
            summary.fit(bad)
    with pytest.raises(ValueError, match="timepoint"):
        summary.fit(np.zeros((0, 15)))
    with pytest.raises(ValueError, match="column"):
        summary.fit(np.zeros((10, 0)))
    fitted = summary.fit(v)
    for bad in (inf, diag):
        with pytest.raises(ValueError, match="finite"):
            fitted.transform(bad)
    with pytest.raises(ValueError, match="timepoint"):
        fitted.transform(np.zeros((0, 15)))


def test_non_finite_null_threshold_raises():
    v = _random_corr_series(8, 80, seed=4)
    for threshold in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="null_threshold"):
            tc.Communities().fit(v, null_threshold=threshold)
    tc.Communities().fit(v, null_threshold=0.0)


# ----------------------------------------------------------------------------
# median centering of the pair weights (review item 8)
# ----------------------------------------------------------------------------


def _equicorrelated(K, T=200):
    """Exact correlations with every off-diagonal entry equal to rho(t) (a
    time-varying common correlation), vectorized with mat2vec, and rho."""
    t = np.arange(T)
    rho = 0.45 + 0.35 * np.sin(2 * np.pi * t / 80)
    M = np.empty((K, K, T))
    M[:] = rho
    M[np.arange(K), np.arange(K), :] = 1
    return tc.mat2vec(M), rho


def test_center_true_erases_coupling_shared_by_every_pair():
    v, rho = _equicorrelated(6)
    ws = tc.WeightedStrength()
    assert ws.center is True
    with pytest.warns(UserWarning, match="constant over time"):
        out = ws(v)
    np.testing.assert_allclose(out, 0, atol=1e-12)


@pytest.mark.parametrize("K", [3, 4, 6])
def test_center_false_keeps_coupling_shared_by_every_pair(K):
    v, rho = _equicorrelated(K)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tc.WeightedStrength(center=False)(v)
        signed = tc.WeightedStrength(center=False, signed=True)(v)
    assert out.shape == (len(rho), K) and signed.shape == (len(rho), 2 * K)
    for c in range(K):
        assert np.corrcoef(out[:, c], rho)[0, 1] > 0.8
    # uncentered: a_ij = w_ij, so the column is sum_j w_ij r_ij(t)**2
    w = np.mean(rho**2)
    np.testing.assert_allclose(out, (K - 1) * w * rho[:, None] ** 2 * np.ones(K), rtol=1e-10)


def test_center_applies_from_four_nodes():
    # K = 3 is never centered, so center=True and center=False agree; from
    # K = 4 on the equal pair weights are centered to 0
    v3, _ = _equicorrelated(3)
    np.testing.assert_allclose(tc.WeightedStrength()(v3),
                               tc.WeightedStrength(center=False)(v3), rtol=1e-12)
    v4, _ = _equicorrelated(4)
    with pytest.warns(UserWarning, match="constant over time"):
        centered = tc.WeightedStrength()(v4)
    np.testing.assert_allclose(centered, 0, atol=1e-12)
    assert np.ptp(tc.WeightedStrength(center=False)(v4), axis=0).min() > 0


def test_center_false_matches_the_uncentered_formula(nested):
    _, _, corrs = nested
    out = tc.WeightedStrength(center=False, signed=True)(corrs)
    for o, v in zip(out, corrs):
        M = _offdiag(v)
        a = _pair_weights(M)
        np.fill_diagonal(a, 0)
        mean_r = M.mean(axis=2)
        expected = np.hstack([np.einsum("ij,ijt->ti", a, M**2),
                              np.einsum("ij,ijt->ti", mean_r, M)])
        np.testing.assert_allclose(o, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_center_on_simulated_equicorrelated_data(seed):
    # x_i(t) = sqrt(rho(t)) g(t) + sqrt(1 - rho(t)) e_i(t): every pair has
    # correlation rho(t).  Estimated pair weights are nearly tied, so centered
    # magnitudes are small (measured 2-3% of the uncentered ones on seeds 0-4)
    # and the uncentered columns track rho (measured min corr .893 on seeds 0-4)
    rng = np.random.default_rng(seed)
    T, K = 1000, 8
    rho = 0.45 + 0.35 * np.sin(2 * np.pi * np.arange(T) / 250)
    x = (np.sqrt(rho)[:, None] * rng.standard_normal(T)[:, None]
         + np.sqrt(1 - rho)[:, None] * rng.standard_normal((T, K)))
    c = tc.timecorr(x, cfun=tc.autofc, rfun=None, weights_params={"var": 100})
    centered = tc.WeightedStrength()(c)
    plain = tc.WeightedStrength(center=False)(c)
    for i in range(K):
        assert np.corrcoef(plain[:, i], rho)[0, 1] > 0.8
    assert np.mean(np.abs(centered)) < 0.1 * np.mean(np.abs(plain))


# columns constant up to round-off ------------------------------------------


def _duplicated_pair(seed=10, T=100):
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(T), rng.standard_normal(T)
    return np.column_stack([a, a, b, b])


def _order2_feeding_corrs():
    """The correlations of the order-2 feeding step of the default
    ``features`` call on the duplicated pair [a, a, b, b] (whose order-1
    feeding features are four copies of nearly one column)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, D = tc.features(_duplicated_pair(), order=1, rfun=None, return_delta=True)
        return tc.timecorr(D[1], cfun=tc.autofc, weights_params={"var": 25})


def test_round_off_constant_columns_are_set_to_their_mean():
    corrs = _order2_feeding_corrs()
    ws = tc.WeightedStrength(split=False, signed=False, center=False)
    # what the column range was before: round-off of about 1e-13 on values
    # of about 3
    with pytest.warns(UserWarning, match="constant over time"):
        out = ws(corrs)
    assert out.shape == (100, 4)
    assert np.all(np.ptp(out, axis=0) == 0)
    np.testing.assert_allclose(out, 3, rtol=1e-12)


def test_small_real_variation_is_not_snapped():
    # a column whose range is 1e-6 of its magnitude is kept as computed
    T = 200
    small = 1 + 1e-6 * np.sin(2 * np.pi * np.arange(T) / 50)
    M = np.empty((4, 4, T))
    M[:] = 0.5 * np.sqrt(small)
    M[np.arange(4), np.arange(4), :] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tc.WeightedStrength(center=False)(tc.mat2vec(M))
    rel = np.ptp(out, axis=0) / np.abs(out).max(axis=0)
    assert np.all(rel > 1e-7) and np.all(rel < 1e-5)


def test_exact_constants_are_returned_unchanged():
    # exact constants (range 0) are not replaced by a recomputed mean
    v, _ = _equicorrelated(4)
    M = tc.vec2mat(v)
    M[:] = 0.3
    M[np.arange(4), np.arange(4), :] = 1
    with pytest.warns(UserWarning, match="constant over time"):
        out = tc.WeightedStrength(center=False)(tc.mat2vec(M))
    direct = 3 * np.mean(np.full(M.shape[2], 0.3) ** 2) * 0.3**2
    assert np.all(np.ptp(out, axis=0) == 0)
    np.testing.assert_allclose(out, direct, rtol=1e-12)


def test_round_off_snapping_agrees_streamed_and_materialized():
    x = _duplicated_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, D = tc.features(x, order=2, rfun=None, return_delta=True)
    corrs = _order2_feeding_corrs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = tc.WeightedStrength(split=False, signed=False, center=False)(corrs)
    assert np.all(np.ptp(D[2], axis=0) == 0)
    np.testing.assert_allclose(D[2], mat, rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_constant_column_rule_does_not_depend_on_the_scale(scale):
    # a column is constant when its range is at most 1e-9 of its largest
    # absolute value, whatever its scale: round-off (1e-13 of the values) is
    # set to the mean and reported, a variation of 1e-6 of the values is kept
    T = 200
    rng = np.random.default_rng(0)
    round_off = scale * 3 * (1 + 1e-13 * rng.standard_normal(T))
    real = scale * 3 * (1 + 1e-6 * np.sin(2 * np.pi * np.arange(T) / 50))
    out = np.column_stack([round_off, real, np.zeros(T), np.full(T, scale)])
    kept = out.copy()
    with pytest.warns(UserWarning, match="3 output columns are constant"):
        tc.WeightedStrength()._warn_constant([out])
    assert np.ptp(out[:, 0]) == 0
    assert out[0, 0] == pytest.approx(3 * scale, rel=1e-12)
    np.testing.assert_array_equal(out[:, 1:], kept[:, 1:])


# two planted communities of unequal size: centering on the median ---------


def _unequal_communities(rhos, seed, sizes=(18, 6), T=1000):
    """Two subjects; community g (sizes[g] columns) shares a factor with
    loading sqrt(rhos[g]); static coupling."""
    labels = np.repeat(np.arange(len(sizes)), sizes)
    r = np.array(rhos)[labels]
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(2):
        f = rng.standard_normal((T, len(sizes)))
        noise = rng.standard_normal((T, len(labels)))
        data.append(np.sqrt(r) * f[:, labels] + np.sqrt(1 - r) * noise)
    return data, labels


def _within(labels, g):
    iu = np.triu_indices(len(labels), 1)
    return (labels[iu[0]] == g) & (labels[iu[1]] == g), iu


@pytest.mark.parametrize("seed", [10, 11])
def test_centering_with_a_large_community(seed):
    # 18 + 6 columns: 153 of the 276 pairs are within the large community,
    # so the median pair weight m is a within-community weight of the large
    # community.  Measured (seeds 0-2): with equal loadings (0.3) 15-18%
    # of the large community's clipped weights are 0 and their mean is
    # 10-12% of its mean pair weight; when the small community is weaker
    # (loadings 0.4 and 0.2) every clipped weight of the small community is
    # 0 and the automatic Communities count splits the large community
    # (6 to 8 communities); with 12 + 12 columns no within weight is 0.
    # The assertions use seeds 10 and 11, not used for those measurements
    # (there: 16% of the clipped weights 0, means 12-13%; 6 and 7
    # communities)
    for rhos in ((0.3, 0.3), (0.4, 0.2)):
        data, labels = _unequal_communities(rhos, seed)
        ws, comm = tc.WeightedStrength(center="median"), tc.Communities()
        quiet_features = dict(order=1, cfun=tc.autofc, rfun=ws, delta_rfun=comm,
                              return_delta=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tc.features(data, **quiet_features)
        wbar = np.mean(ws.fits_[-1]["weights"], axis=0)
        a = sm._clip_weights(wbar, "median")
        big, iu = _within(labels, 0)
        small, _ = _within(labels, 1)
        w, a = wbar[iu], a[iu]
        m = np.median(w)
        assert w[big].min() - 0.05 < m < w[big].max()
        assert np.all(a[~(big | small)] == 0)
        if rhos == (0.3, 0.3):
            assert 0 < np.mean(a[big] == 0) < 0.5
            assert a[big].mean() < 0.25 * w[big].mean()
        else:
            assert np.all(a[small] == 0)
            assert a[big].mean() < 0.25 * w[big].mean()
            assert comm.fits_[-1]["n"] > 2
    data, labels = _unequal_communities((0.3, 0.3), seed, sizes=(12, 12))
    ws = tc.WeightedStrength(center="median")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tc.features(data, order=1, cfun=tc.autofc, rfun=ws,
                    delta_rfun="weighted_strength")
    wbar = np.mean(ws.fits_[-1]["weights"], axis=0)
    a = sm._clip_weights(wbar, "median")
    for g in (0, 1):
        sel, iu = _within(labels, g)
        assert np.all(a[iu][sel] > 0)


# centering on the lower cluster of the pair weights -----------------------


def _two_means_low_reference(x):
    """Mean of the lower group of the best split of the sorted values into
    two groups (least total within-group sum of squares), by brute force."""
    x = np.sort(np.asarray(x, dtype=float))
    if x[0] == x[-1]:
        return x[0]
    best, cut = np.inf, None
    for k in range(1, len(x)):
        if x[k - 1] == x[k]:
            continue
        sse = ((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum()
        if sse < best:
            best, cut = sse, k
    return x[:cut].mean()


def test_lower_cluster_level_matches_brute_force_two_means():
    rng = np.random.default_rng(3)
    for trial in range(40):
        K = int(rng.integers(4, 30))
        S = rng.gamma(2.0, 0.01, (K, K))
        n_hi = int(rng.integers(2, K + 1))
        S[:n_hi, :n_hi] += rng.gamma(4.0, 0.05, (n_hi, n_hi))
        w = (S + S.T) / 2
        if trial % 4 == 0:
            w = np.round(w, 2)  # ties
        np.fill_diagonal(w, 0)
        level = _two_means_low_reference(w[np.triu_indices(K, 1)])
        np.testing.assert_allclose(sm._center_level(w, "lower_cluster"), level,
                                   rtol=1e-12, atol=0)
        a = sm._clip_weights(w)
        np.testing.assert_allclose(a, np.maximum(w - level, 0) * (1 - np.eye(K)),
                                   rtol=0, atol=1e-15)


def test_center_median_is_the_previous_rule():
    rng = np.random.default_rng(4)
    w = rng.gamma(2.0, 0.05, (9, 9))
    w = (w + w.T) / 2
    np.fill_diagonal(w, 0)
    m = np.median(w[np.triu_indices(9, 1)])
    expected = np.maximum(w - m, 0)
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(sm._clip_weights(w, "median"), expected)
    np.testing.assert_array_equal(sm._clip_weights(w, False), w)


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_default_centering_keeps_a_weaker_small_community(seed):
    # 18 + 6 columns, loadings 0.4 and 0.2: centering on the median removes
    # every weight of the small community; the default (lower cluster) keeps
    # them all, and keeps every weight of the large community too
    data, labels = _unequal_communities((0.4, 0.2), seed)
    corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None, weights_params={"var": 25})
    fits = {}
    for center in (True, "median"):
        ws = tc.WeightedStrength(center=center)
        ws(corrs)
        fits[center] = sm._clip_weights(np.mean(ws.fits_[-1]["weights"], axis=0),
                                        center)
    big, iu = _within(labels, 0)
    small, _ = _within(labels, 1)
    assert np.all(fits["median"][iu][small] == 0)
    assert np.all(fits[True][iu][small] > 0)
    assert np.all(fits[True][iu][big] > 0)


def test_default_centering_columns_of_the_small_community_vary():
    data, labels = _unequal_communities((0.4, 0.2), 23)
    corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None, weights_params={"var": 25})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        median = tc.WeightedStrength(center="median")(corrs)
        default = tc.WeightedStrength()(corrs)
    for m, d in zip(median, default):
        assert np.all(np.ptp(m[:, labels == 1], axis=0) == 0)
        assert np.all(np.ptp(d[:, labels == 1], axis=0) > 0)


def _clipped_lower_cluster(w):
    """Reference for ``center=True``, written without the package: subtract
    the mean of the lower group of the best split of the sorted pair weights
    into two groups (brute force over the split points), clip at 0."""
    K = w.shape[0]
    a = w.copy() if K < 4 else np.maximum(
        w - _two_means_low_reference(w[np.triu_indices(K, 1)]), 0)
    np.fill_diagonal(a, 0)
    return a


@pytest.mark.parametrize("signed", [False, True])
def test_weighted_strength_formula_default_centering(nested, signed):
    _, _, corrs = nested
    out = tc.WeightedStrength(signed=signed)(corrs)
    for o, v in zip(out, corrs):
        M = _offdiag(v)
        a = _clipped_lower_cluster(_pair_weights(M))
        cols = [np.einsum("ij,ijt->ti", a, M**2)]
        if signed:
            mean_r = M.mean(axis=2)
            np.fill_diagonal(mean_r, 0)
            cols.append(np.einsum("ij,ijt->ti", mean_r, M))
        np.testing.assert_allclose(o, np.hstack(cols), atol=1e-10, rtol=0)


def _sym(values, K):
    """K x K symmetric matrix with ``values`` on the pairs i < j, zero diagonal."""
    w = np.zeros((K, K))
    w[np.triu_indices(K, 1)] = values
    return w + w.T


def test_lower_cluster_degenerate_cases():
    # all weights equal: the level is that value and every clipped weight is 0
    w = _sym(np.full(6, 0.3), 4)
    assert sm._center_level(w, "lower_cluster") == 0.3
    np.testing.assert_array_equal(sm._clip_weights(w), 0)
    # two distinct values: the lower one is the level; the higher pairs keep
    # the difference
    vals = np.array([0.1, 0.1, 0.5, 0.1, 0.5, 0.1])
    w = _sym(vals, 4)
    assert sm._center_level(w, "lower_cluster") == pytest.approx(0.1, abs=1e-15)
    np.testing.assert_allclose(sm._clip_weights(w)[np.triu_indices(4, 1)],
                               np.where(vals == 0.5, 0.4, 0.0), atol=1e-15)
    # one pair above many equal ones: the upper group holds that pair only
    vals = np.r_[np.full(14, 0.02), 0.9]
    w = _sym(vals, 6)
    assert sm._center_level(w, "lower_cluster") == pytest.approx(0.02, abs=1e-15)
    a = sm._clip_weights(w)[np.triu_indices(6, 1)]
    assert np.count_nonzero(a) == 1 and a[-1] == pytest.approx(0.88)
    # ties at the split point stay together in one group
    vals = np.array([0.1, 0.1, 0.1, 0.4, 0.4, 0.4])
    assert sm._center_level(_sym(vals, 4), "lower_cluster") == pytest.approx(0.1)
    # zero weights: no structure at all
    np.testing.assert_array_equal(sm._clip_weights(np.zeros((5, 5))), 0)
    # K < 4 is never centered
    w3 = _sym([0.1, 0.2, 0.3], 3)
    np.testing.assert_array_equal(sm._clip_weights(w3), w3)


@pytest.mark.parametrize("rule", ["lower_cluster", "median"])
def test_center_level_needs_finite_weights(rule):
    w = _sym(np.array([0.1, 0.2, np.nan, 0.3, 0.1, 0.2]), 4)
    with pytest.raises(ValueError, match="finite"):
        sm._center_level(w, rule)
    w[0, 1] = w[1, 0] = np.inf
    w[0, 3] = w[3, 0] = 0.3
    with pytest.raises(ValueError, match="finite"):
        sm._clip_weights(w)


def _time_varying_communities(sizes, scale, gamma, seed, T=1000):
    """Two subjects; community g's columns share a factor whose squared
    loading rho_g(t) = scale[g] * s_g(t) varies slowly (s_g in (0.1, 1));
    ``gamma``: squared loading of a factor shared by every column.  Returns
    the data, the labels and rho (T, G).  The generator of the benchmark in
    the ``WeightedStrength`` Notes."""
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    rho = []
    for c in scale:
        z = gaussian_filter1d(rng.standard_normal(T + 240), 40)[120:-120]
        z = (z - z.mean()) / z.std()
        rho.append(c * (0.1 + 0.9 / (1 + np.exp(-2 * z))))
    rho = np.stack(rho, axis=1)
    data = []
    for _ in range(2):
        f = rng.standard_normal((T, len(sizes)))
        g = rng.standard_normal((T, 1))
        e = rng.standard_normal((T, len(labels)))
        r = rho[:, labels]
        data.append(np.sqrt(gamma) * g + np.sqrt(r) * f[:, labels]
                    + np.sqrt(1 - gamma - r) * e)
    return data, labels, rho


@pytest.mark.parametrize(
    "gamma, scale, min_score",
    [(0.0, (0.5, 0.25), 0.28), (0.2, (0.45, 0.25), 0.31)],
)
def test_default_centering_keeps_the_small_community(gamma, scale, min_score):
    # Benchmark of the WeightedStrength Notes on seeds 2000-2003, which were
    # used neither to choose the rule (development seeds 0-7) nor to
    # evaluate it (held-out seeds 1000-1015).  The thresholds were fixed
    # from the development seeds before these seeds were run: the
    # small-community score of the lower cluster had mean .403 (sd .076)
    # without and .465 (sd .097) with the global signal over seeds 0-7, and
    # min_score is that mean minus 3 standard errors of a 4-seed mean,
    # rounded down; every development seed kept all 15 small-community
    # pairs (the bound allows one to drop); median centering scored 0 on
    # every development seed.
    scores, kept, scores_median = [], [], []
    for seed in range(2000, 2004):
        data, labels, rho = _time_varying_communities((18, 6), scale, gamma, seed)
        corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None,
                            weights_params={"var": 25})
        for center, store in ((True, scores), ("median", scores_median)):
            ws = tc.WeightedStrength(center=center)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out = ws(corrs)
            col = [o[:, labels == 1].mean(axis=1) for o in out]
            store.append(np.mean([
                0.0 if np.ptp(c) == 0 else np.corrcoef(c, rho[:, 1] ** 2)[0, 1]
                for c in col
            ]))
            if center is True:
                a = sm._clip_weights(np.mean(ws.fits_[-1]["weights"], axis=0))
                small, iu = _within(labels, 1)
                kept.append(np.mean(a[iu][small] > 0))
    assert np.mean(scores) > min_score, scores
    assert min(kept) >= 0.9, kept
    assert np.all(np.array(scores_median) == 0), scores_median


@pytest.mark.parametrize(
    "center", ["mean", "lower", "Median", "lower_cluster", "true", 0.5, None, 1, 0, [True]]
)
def test_bad_center_raises(center):
    with pytest.raises(ValueError, match="center must be"):
        tc.WeightedStrength(center=center)


def test_center_values_are_kept_and_shown():
    assert tc.WeightedStrength().center is True
    assert tc.WeightedStrength(center="median").center == "median"
    assert tc.WeightedStrength(center=False).center is False
    assert "center='median'" in repr(tc.WeightedStrength(center="median"))
    assert tc.WeightedStrength(center=np.True_).center is True
