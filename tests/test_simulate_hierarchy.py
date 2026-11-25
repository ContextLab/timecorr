"""
``simulate_hierarchy``: nested structure of any depth.

Real draws from the generator, no mocks.
"""

import os

import numpy as np
import pytest
from scipy.stats import norm

import timecorr as tc


def small(**kwargs):
    params = dict(S=2, T=400, nodes_per_module=3, branching=(2, 3), n_top=2,
                  taus=(2, 5, 20), set_random_seed=0)
    params.update(kwargs)
    return tc.simulate_hierarchy(**params)


def test_shapes_and_group_labels():
    data, truth = small()
    # 2 top groups x 3 level-2 groups x 2 modules x 3 nodes
    K = 2 * 3 * 2 * 3
    assert len(data) == 2
    assert all(x.shape == (400, K) for x in data)
    groups = truth["groups"]
    assert groups.shape == (3, K) and groups.dtype.kind == "i"
    np.testing.assert_array_equal(groups[0], np.repeat(np.arange(12), 3))
    np.testing.assert_array_equal(groups[1], groups[0] // 2)
    np.testing.assert_array_equal(groups[2], groups[1] // 3)
    assert groups[2].max() == 1
    assert len(truth["signals"]) == 2
    for signals in truth["signals"]:
        assert [s.shape for s in signals] == [(400, 12), (400, 6), (400, 2)]
        assert all(np.all((s > 0) & (s < 1)) for s in signals)
        # levels above 1 lie in (0.05, 0.95)
        assert all(np.all((s > 0.05) & (s < 0.95)) for s in signals[1:])
    assert truth["top"].shape == (400, 2)
    assert truth["taus"] == (2, 5, 20) and truth["branching"] == (2, 3)


def test_defaults():
    data, truth = tc.simulate_hierarchy(S=1, set_random_seed=0)
    # 4 top groups x 4 x 4 modules x 6 nodes, T 6000
    assert data[0].shape == (6000, 384)
    assert [s.shape[1] for s in truth["signals"][0]] == [64, 16, 4]
    assert truth["taus"] == (5, 50, 500) and truth["branching"] == (4, 4)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_any_depth(depth):
    taus = (2, 5, 10, 20)[:depth]
    branching = (2, 2, 2)[: depth - 1]
    data, truth = tc.simulate_hierarchy(S=1, T=200, nodes_per_module=2,
                                        branching=branching, n_top=2, taus=taus,
                                        set_random_seed=1)
    K = 2 * 2 * 2 ** (depth - 1)
    assert data[0].shape == (200, K)
    assert truth["groups"].shape == (depth, K)
    assert len(truth["signals"][0]) == depth
    np.testing.assert_array_equal(truth["top"], truth["signals"][0][-1])


def test_two_levels_have_the_form_of_simulate_nested():
    # with n = 2: s_2 is h, s_1 is g; the same formulas as simulate_nested's
    # sustained coupling
    data, truth = small(S=1, branching=(3,), taus=(5, 40))
    h, g = truth["signals"][0][1], truth["signals"][0][0]
    assert h.shape == (400, 2) and g.shape == (400, 6)
    # h = 0.05 + 0.9 Phi(1.5 smooth): the standardized smooth series back
    latent = norm.ppf((h - 0.05) / 0.9) / 1.5
    np.testing.assert_allclose(latent.mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(latent.std(0), 1, atol=1e-9)
    # g = Phi(sqrt(h) u + sqrt(1 - h) v): unit variance pointwise
    a = norm.ppf(g)
    assert np.all(np.abs(a.std(0) - 1) < 0.3)
    # three modules per top group, three nodes per module
    np.testing.assert_array_equal(truth["groups"][1], np.repeat(np.arange(2), 9))


def test_determinism_and_global_state():
    state = np.random.get_state()
    a, ta = small(set_random_seed=5)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    b, tb = small(set_random_seed=5)
    c, _ = small(set_random_seed=6)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    for sa, sb in zip(ta["signals"][1], tb["signals"][1]):
        np.testing.assert_array_equal(sa, sb)
    assert not np.array_equal(a[0], c[0])
    # True is seed 123
    d, _ = small(set_random_seed=True)
    e, _ = small(set_random_seed=123)
    np.testing.assert_array_equal(d[0], e[0])


def test_sharing_modes():
    _, top = small(S=3, shared="top")
    _, every = small(S=3, shared="all")
    _, none = small(S=3, shared="none")
    for truth in (top, every):
        for s in truth["signals"][1:]:
            np.testing.assert_array_equal(s[-1], truth["signals"][0][-1])
        np.testing.assert_array_equal(truth["top"], truth["signals"][0][-1])
    for s in none["signals"][1:]:
        assert not np.array_equal(s[-1], none["signals"][0][-1])
    # lower signals equal only for "all"
    for level in range(2):
        for s in every["signals"][1:]:
            np.testing.assert_array_equal(s[level], every["signals"][0][level])
        for truth in (top, none):
            for s in truth["signals"][1:]:
                assert not np.array_equal(s[level], truth["signals"][0][level])
    # the data are private in every mode (f, z, e)
    data, _ = small(S=2, shared="all")
    assert not np.array_equal(data[0], data[1])


@pytest.mark.parametrize("shared", ["top", "all", "none"])
def test_returned_signals_are_not_shared_objects(shared):
    # mutating one subject's signals (or truth["top"]) leaves the others
    # unchanged
    _, truth = small(S=3, shared=shared)
    before = [[a.copy() for a in s] for s in truth["signals"]]
    top_before = truth["top"].copy()
    for a in truth["signals"][1]:
        a += 1.0
    for subject in (0, 2):
        for a, b in zip(truth["signals"][subject], before[subject]):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(truth["top"], top_before)
    truth["top"][:] = -1.0
    for subject in (0, 2):
        np.testing.assert_array_equal(truth["signals"][subject][-1],
                                      before[subject][-1])


# Reference values of small(S=3, shared=..., set_random_seed=seed) for these
# keys, written by tests/fixtures/make_private_signs_reference.py with the
# package at commit 012f5e0^ (before the signs option was added): for each
# subject's data and each of its signals, 8 fixed entries and the sum.  The
# default, signs="private", must reproduce them: the same random draws in the
# same order.  They are compared up to round-off (rtol 1e-9, atol 1e-12), not
# bit for bit: transcendental functions and sums round differently in their
# last bits across platforms and NumPy/SciPy builds.
PRIVATE_SIGNS_KEYS = [(shared, seed) for shared in ("top", "all", "none")
                      for seed in (0, 1, 2)]
PRIVATE_SIGNS_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                                     "private_signs_reference.npz")


def reference_values(data, truth):
    """(arrays, 9): 8 entries at fixed flat positions and the sum of each
    subject's data, then of each subject's signals (level by level)."""
    arrays = list(data) + [a for s in truth["signals"] for a in s]
    out = []
    for x in arrays:
        flat = np.ravel(x)
        idx = np.linspace(0, flat.size - 1, 8).round().astype(int)
        out.append(np.concatenate([flat[idx], [flat.sum()]]))
    return np.stack(out)


def private_signs_reference_name(key):
    return f"{key[0]}_{key[1]}"


@pytest.mark.parametrize("key", PRIVATE_SIGNS_KEYS)
def test_private_signs_are_the_previous_output(key):
    shared, seed = key
    with np.load(PRIVATE_SIGNS_FIXTURE) as f:
        expected = f[private_signs_reference_name(key)]
    data, truth = small(S=3, shared=shared, set_random_seed=seed)
    got = reference_values(data, truth)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)
    # signs="private" is the default call exactly (same process, same code)
    data_p, truth_p = small(S=3, shared=shared, set_random_seed=seed,
                            signs="private")
    for x, y in zip(data, data_p):
        assert np.array_equal(x, y)
    for s, sp in zip(truth["signals"], truth_p["signals"]):
        for x, y in zip(s, sp):
            assert np.array_equal(x, y)
    assert np.array_equal(truth["signs"], truth_p["signs"])
    assert truth["signs"].shape == (3, data[0].shape[1])
    assert set(np.unique(truth["signs"])) <= {-1.0, 1.0}


def _node_signs(data, truth):
    """Sign of each node relative to the first node of its module, from the
    data: the sign of the correlation with that node."""
    module = truth["groups"][0]
    out = []
    for x in data:
        r = np.corrcoef(x.T)
        first = np.array([np.flatnonzero(module == m)[0] for m in module])
        out.append(np.sign(r[np.arange(len(module)), first]))
    return np.array(out)


@pytest.mark.parametrize("shared", ["top", "all", "none"])
def test_shared_signs(shared):
    data, truth = small(S=3, T=4000, shared=shared, signs="shared")
    signs = truth["signs"]
    assert signs.shape == (3, data[0].shape[1])
    for row in signs[1:]:
        np.testing.assert_array_equal(row, signs[0])
    # the signs relative to each module's first node are visible in the data
    module = truth["groups"][0]
    first = np.array([np.flatnonzero(module == m)[0] for m in module])
    expected = signs[:, np.arange(len(module))] * signs[:, first]
    np.testing.assert_array_equal(_node_signs(data, truth), expected)
    # with private signs they differ between subjects (seed 0)
    _, private = small(S=3, shared=shared, signs="private")
    assert not np.array_equal(private["signs"][0], private["signs"][1])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(S=0), "S must be a positive integer"),
        (dict(T=2.5), "T must be a positive integer"),
        (dict(nodes_per_module=0), "nodes_per_module must be a positive integer"),
        (dict(n_top=True), "n_top must be a positive integer"),
        (dict(taus=()), "taus must be a non-empty sequence"),
        (dict(taus=5), "taus must be a non-empty sequence"),
        (dict(taus=(2, 0, 20)), r"taus\[1\] must be a positive integer"),
        (dict(branching=(2, 1.5)), r"branching\[1\] must be a positive integer"),
        (dict(branching=(2,)), "len\\(branching\\) must be len\\(taus\\) - 1 = 2"),
        (dict(branching=3), "branching must be a sequence"),
        # a list, tuple or 1-D numpy array only
        (dict(taus=np.array(5), branching=()), "taus must be a non-empty sequence"),
        (dict(taus=(5,), branching=np.array(2)), "branching must be a sequence"),
        (dict(taus={5, 50, 500}), "taus must be a non-empty sequence"),
        (dict(taus=frozenset({5, 50, 500})), "taus must be a non-empty sequence"),
        (dict(branching={2, 3}), "branching must be a sequence"),
        (dict(taus="5"), "taus must be a non-empty sequence"),
        (dict(branching="23"), "branching must be a sequence"),
        (dict(taus=np.array([[2, 6, 20]])), "taus must be a non-empty sequence"),
        (dict(taus=range(2, 5)), "taus must be a non-empty sequence"),
        (dict(T=100), "T must be at least 8 \\* max\\(taus\\) = 160"),
        (dict(shared="order1"), "shared must be 'top', 'all' or 'none'"),
        (dict(signs="all"), "signs must be 'private' or 'shared'"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        small(**kwargs)


def test_taus_and_branching_accept_lists_tuples_and_1d_arrays():
    base_data, base_truth = small()
    for convert in (list, tuple, np.array):
        data, truth = small(taus=convert(base_truth["taus"]),
                            branching=convert(base_truth["branching"]))
        assert truth["taus"] == base_truth["taus"]
        assert truth["branching"] == base_truth["branching"]
        for x, y in zip(data, base_data):
            np.testing.assert_array_equal(x, y)


def test_bad_seed():
    with pytest.raises(TypeError, match="set_random_seed"):
        small(set_random_seed="a")


# the defining property, measured on the true signals ------------------------


def _latent(s, level):
    """The latent series a of level-l signals (Phi^-1 of s_1; for l > 1,
    Phi^-1((s - 0.05) / 0.9) / 1.5)."""
    return norm.ppf(s) if level == 1 else norm.ppf((s - 0.05) / 0.9) / 1.5


def _window_statistic(a, parent_signal, pairs, W):
    """Mean over pairs (i, j, p) of the correlation, across non-overlapping
    windows of W samples, between the window correlation of a_i and a_j and
    the window mean of parent p's signal."""
    nw = a.shape[0] // W
    out = []
    for i, j, p in pairs:
        A = a[: nw * W, i].reshape(nw, W)
        B = a[: nw * W, j].reshape(nw, W)
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
        r = (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))
        sp = parent_signal[: nw * W, p].reshape(nw, W).mean(1)
        out.append(np.corrcoef(r, sp)[0, 1])
    return float(np.mean(out))


def _sibling_statistics(truth, windows):
    """{level: (siblings, non-siblings)} for every level below the top; as
    many non-sibling pairs (drawn with a fixed seed) as sibling pairs."""
    s = truth["signals"][0]
    out = {}
    for level in range(1, len(s)):
        a = _latent(s[level - 1], level)
        parent = np.arange(a.shape[1]) // truth["branching"][level - 1]
        m = a.shape[1]
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        sib = [(i, j, parent[i]) for i, j in pairs if parent[i] == parent[j]]
        non = [(i, j, parent[i]) for i, j in pairs if parent[i] != parent[j]]
        pick = np.random.default_rng(0).choice(len(non), len(sib), replace=False)
        non = [non[k] for k in pick]
        W = windows[level - 1]
        out[level] = (_window_statistic(a, s[level], sib, W),
                      _window_statistic(a, s[level], non, W))
    return out


# Development seeds 0-7, defaults (T 6000, taus (5, 50, 500), branching
# (4, 4), 4 top groups), windows of 50 samples at level 1 and 400 at level 2:
#   level 1 siblings .622 +- .008 (min .611), non-siblings -.006 +- .011
#     (max |.031|)
#   level 2 siblings .617 +- .033 (min .566), non-siblings -.044 +- .075
#     (max |.167|)
#   node variance (windows of 100) vs the top signal, mean over the 4 top
#     groups: .009 +- .072 (max |.126|); per group (32 values) mean .009,
#     sd .123 (60 windows: about 1 / sqrt(60) = .13 for independent
#     series), max |r| .306; mean |r| over the 4 groups at most .126
# The assertions below use other seeds (100, 101).
@pytest.mark.parametrize("seed", [100, 101])
def test_siblings_are_coupled_by_their_parents_signal(seed):
    _, truth = tc.simulate_hierarchy(S=1, set_random_seed=seed)
    stats = _sibling_statistics(truth, windows=(50, 400))
    sib1, non1 = stats[1]
    sib2, non2 = stats[2]
    assert sib1 > 0.5 and abs(non1) < 0.1, stats
    assert sib2 > 0.4 and abs(non2) < 0.3, stats


@pytest.mark.parametrize("seed", [100, 101])
def test_node_variance_does_not_follow_the_top_signal(seed):
    data, truth = tc.simulate_hierarchy(S=1, set_random_seed=seed)
    x = data[0]
    W = 100
    nw = x.shape[0] // W
    variance = x[: nw * W].reshape(nw, W, -1).var(1)
    top = truth["top"][: nw * W].reshape(nw, W, -1).mean(1)
    group = truth["groups"][-1]
    r = [np.corrcoef(variance[:, group == t].mean(1), top[:, t])[0, 1]
         for t in range(top.shape[1])]
    # each group, not only their mean (opposite dependencies would cancel)
    assert np.max(np.abs(r)) < 0.4, r
    assert np.mean(np.abs(r)) < 0.2, r
    # variance 1 by construction
    assert abs(x.var() - 1) < 0.02


@pytest.mark.parametrize("shared", ["top", "none"])
def test_shared_signs_change_the_private_draws_at_the_same_seed(shared):
    # documented: the shared signs are drawn from the same generator before
    # the subjects' series, so the private draws differ from signs="private"
    data_p, truth_p = small(shared=shared, signs="private")
    data_s, truth_s = small(shared=shared, signs="shared")
    if shared == "top":
        np.testing.assert_array_equal(truth_p["top"], truth_s["top"])
    assert not np.array_equal(truth_p["signals"][0][0], truth_s["signals"][0][0])
    assert not np.array_equal(data_p[0], data_s[0])


# --- factors: module series shared across subjects ---------------------------

@pytest.mark.parametrize("shared", ["top", "all", "none"])
def test_private_factors_are_the_default_call(shared):
    # that the default output is the one of earlier versions is pinned by
    # test_private_signs_are_the_previous_output (stored values)
    data, truth = small(S=3, shared=shared)
    data_p, truth_p = small(S=3, shared=shared, factors="private")
    for x, y in zip(data, data_p):
        assert np.array_equal(x, y)
    n_modules = truth["signals"][0][0].shape[1]
    assert len(truth["factors"]) == 3
    for f in truth["factors"]:
        assert f.shape == (400, n_modules)
    # private: the subjects' module series differ
    assert not np.array_equal(truth["factors"][0], truth["factors"][1])


@pytest.mark.parametrize("shared", ["top", "all", "none"])
@pytest.mark.parametrize("signs", ["private", "shared"])
def test_shared_factors_are_equal_copies(shared, signs):
    data, truth = small(S=3, shared=shared, signs=signs, factors="shared")
    f = truth["factors"]
    for other in f[1:]:
        np.testing.assert_array_equal(other, f[0])
        assert other is not f[0]
    # the subjects' data still differ (z is private)
    assert not np.array_equal(data[0], data[1])
    # a copy: changing one subject's series leaves the others unchanged
    before = f[1].copy()
    f[0][:] = 0.0
    np.testing.assert_array_equal(f[1], before)
    # the signals are those of the "private" call at the same seed when
    # they are drawn before the shared series
    if shared == "all":
        _, private = small(S=3, shared=shared, signs=signs)
        for x, y in zip(truth["signals"][0], private["signals"][0]):
            np.testing.assert_array_equal(x, y)


def _across_subject_correlations(factors):
    """Correlations between the nodes of subjects 0 and 1, times the sign
    product, and the module mean of g, for a long series with every signal
    and sign shared (so sqrt(g_a g_b) is g)."""
    data, truth = small(S=2, T=20000, shared="all", signs="shared",
                        factors=factors)
    module = truth["groups"][0]
    e = truth["signs"][0]
    K = len(module)
    r = np.corrcoef(data[0].T, data[1].T)[:K, K:] * np.outer(e, e)
    same = module[:, None] == module[None, :]
    expected = np.where(same, truth["signals"][0][0].mean(axis=0)[module][:, None], 0.0)
    return r, expected, same


def test_shared_factors_correlate_modules_across_subjects():
    # measured at seeds 0 to 3: largest deviation 0.017 within modules and
    # 0.032 between modules (sampling error at T = 20000)
    r, expected, same = _across_subject_correlations("shared")
    assert np.abs(r - expected).max() < 0.05
    assert r[same].mean() > 0.4


def test_private_factors_share_no_series():
    r, _, same = _across_subject_correlations("private")
    assert np.abs(r).max() < 0.05


def test_data_are_the_documented_mixture_of_the_returned_factors():
    # x = sqrt(g) e f + sqrt(1 - g) z: what is left after removing the
    # factor term is sqrt(1 - g) z, uncorrelated with f and of variance 1 - g
    data, truth = small(S=2, T=20000, factors="shared")
    module = truth["groups"][0]
    for x, signals, e, f in zip(data, truth["signals"], truth["signs"],
                                truth["factors"]):
        g = signals[0][:, module]
        rest = x - np.sqrt(g) * e * f[:, module]
        np.testing.assert_allclose(rest.var(axis=0), (1 - g).mean(axis=0),
                                   rtol=0.06)
        r = [np.corrcoef(rest[:, k], f[:, module[k]])[0, 1]
             for k in range(x.shape[1])]
        assert np.abs(r).max() < 0.03


def test_factors_validation():
    with pytest.raises(ValueError, match="factors must be 'private' or 'shared'"):
        small(factors="all")


# --- volatility: one multiplier common to every subject and node -------------

COMBOS = [(sh, sg, fa) for sh in ("top", "all", "none")
          for sg in ("private", "shared") for fa in ("private", "shared")]


def test_no_volatility_is_the_default_call():
    data, truth = small(S=3)
    data_n, truth_n = small(S=3, volatility=None)
    for a, b in zip(data, data_n):
        assert np.array_equal(a, b)
    assert np.array_equal(truth["volatility"], np.ones(len(data[0])))
    assert np.array_equal(truth_n["volatility"], np.ones(len(data[0])))


@pytest.mark.parametrize("shared,signs,factors", COMBOS)
def test_volatility_multiplies_the_output_without_it(shared, signs, factors):
    """Same seed: every other draw is unchanged, the data are multiplied."""
    kw = dict(S=3, shared=shared, signs=signs, factors=factors)
    data, truth = small(**kw)
    data_v, truth_v = small(volatility=0.5, volatility_tau=10, **kw)
    v = truth_v["volatility"]
    assert v.shape == (len(data[0]),)
    assert np.all(v > 0) and np.ptp(v) > 0
    for a, b in zip(data, data_v):
        assert np.array_equal(b, a * v[:, None])
    assert np.array_equal(truth["signs"], truth_v["signs"])
    for a, b in zip(truth["factors"], truth_v["factors"]):
        assert np.array_equal(a, b)
    for sa, sb in zip(truth["signals"], truth_v["signals"]):
        for a, b in zip(sa, sb):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("sd", [0.1, 0.5, 1.5])
def test_volatility_scale(sd):
    """log v has standard deviation `volatility`; mean(v**2) is 1."""
    _, truth = small(T=4000, volatility=sd, volatility_tau=25)
    v = truth["volatility"]
    assert abs(np.log(v).std() - sd) < 1e-12
    assert abs(np.mean(v ** 2) - 1) < 1e-12


def test_volatility_timescale():
    """Autocorrelation of log v at lag k is exp(-k**2 / (4 tau**2))."""
    tau = 20
    _, truth = tc.simulate_hierarchy(
        S=1, T=200000, nodes_per_module=2, branching=(), n_top=2, taus=(5,),
        set_random_seed=4, volatility=0.5, volatility_tau=tau)
    h = np.log(truth["volatility"])
    h = h - h.mean()
    for k in (tau // 2, tau, 2 * tau):
        r = np.mean(h[:-k] * h[k:]) / np.mean(h * h)
        assert abs(r - np.exp(-k ** 2 / (4 * tau ** 2))) < 0.05


def test_volatility_clusters_squared_values():
    """Squared values are autocorrelated with the multiplier and not without."""
    kw = dict(S=2, T=40000, nodes_per_module=3, branching=(2,), n_top=2,
              taus=(5, 50), set_random_seed=9)

    def lag1(data):
        out = []
        for x in data:
            q = x ** 2 - (x ** 2).mean(0)
            out.append(np.mean(np.sum(q[:-1] * q[1:], 0) / np.sum(q * q, 0)))
        return np.mean(out)

    plain = lag1(tc.simulate_hierarchy(**kw)[0])
    clustered = lag1(tc.simulate_hierarchy(volatility=0.5, volatility_tau=50, **kw)[0])
    assert abs(plain) < 0.01
    assert clustered > 0.2


def test_volatility_is_reproducible_and_a_copy():
    a, ta = small(volatility=0.3)
    b, tb = small(volatility=0.3)
    assert np.array_equal(ta["volatility"], tb["volatility"])
    before = a[0].copy()
    ta["volatility"][:] = 7
    assert np.array_equal(a[0], before)


@pytest.mark.parametrize("bad", [0, -0.5, np.nan, np.inf, True, "0.5", (0.5, 10)])
def test_volatility_validation(bad):
    with pytest.raises(ValueError, match="volatility must be None or a positive"):
        small(volatility=bad)


@pytest.mark.parametrize("bad", [0, -3, 2.5, "10", True])
def test_volatility_tau_validation(bad):
    with pytest.raises(ValueError, match="volatility_tau"):
        small(volatility=0.5, volatility_tau=bad)


def test_volatility_tau_needs_enough_timepoints():
    with pytest.raises(ValueError, match="8 \\* volatility_tau"):
        small(T=400, volatility=0.5, volatility_tau=51)
    small(T=400, volatility=0.5, volatility_tau=50)
    small(T=400, volatility_tau=51)  # not used without volatility
