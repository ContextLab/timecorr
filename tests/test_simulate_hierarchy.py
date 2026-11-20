"""
``simulate_hierarchy``: nested structure of any depth.

Real draws from the generator, no mocks.
"""

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
        (dict(T=100), "T must be at least 8 \\* max\\(taus\\) = 160"),
        (dict(shared="order1"), "shared must be 'top', 'all' or 'none'"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        small(**kwargs)


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
#     groups: .009 +- .072 (max |.126|)
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
    assert abs(np.mean(r)) < 0.25, r
    # variance 1 by construction
    assert abs(x.var() - 1) < 0.02
