"""
Recovery of block-wise correlation structure with event-restricted kernels.

Order 1: the data are drawn block by block (events of 40 to 120 samples), each
block from its own random correlation matrix, so the true correlation matrix at
timepoint ``t`` is the one of ``t``'s block.  A Gaussian kernel confined to the
true events (``events=`` the true block starts) recovers the true per-timepoint
correlations better than the same kernel unrestricted, overall and within 10
samples of a boundary.  The control confines the kernel to the same number of
events with WRONG (random) boundaries; it does not help.  Score: at each
timepoint, the Pearson correlation between the estimated and the true
off-diagonal correlations, averaged over timepoints and then over seeds.

Measured on 24 seeds (T=600, K=10, var 100) before this test was written:
unrestricted / true events / wrong events 0.888 / 0.919 / 0.876 overall and
0.766 / 0.890 / 0.752 within 10 samples of a boundary.  The margins below are
smaller than those gaps (0.031 and 0.124) and larger than the seed noise of the
mean over 8 seeds.

Order 2: on data whose order-2 structure (how strongly modules' coupling
strengths co-vary) changes at known event boundaries, the pipeline
``features(order=2, rfun=None, events=...)`` recovered little of the simulated order-2 structure with or without events (24
seeds; true boundaries, no events and wrong boundaries all gave recovery r
about 0.02 to 0.07), so the measurement does not show whether events help
there.  The order-2 test therefore checks behavior only: events at order 2
change ``X[2]``, leave ``X[1]`` unchanged, and give the same result as
``timecorr`` applied to the delta features with the same events and reducer.
"""

import numpy as np

import timecorr as tc

T, K = 600, 10
VAR = 100.0
SEEDS = range(8)
NEAR = 10  # samples on each side of a boundary


def _block_data(seed):
    """Block-wise correlated data, the true boundaries, wrong boundaries, truth."""
    rng = np.random.default_rng(seed)
    bounds = [0]
    while bounds[-1] < T:
        bounds.append(bounds[-1] + int(rng.integers(40, 121)))
    bounds = [b for b in bounds if b < T]
    x, truth = np.zeros((T, K)), np.zeros((T, K * (K - 1) // 2))
    iu = np.triu_indices(K, 1)
    for a, b in zip(bounds, bounds[1:] + [T]):
        A = rng.standard_normal((K, 3))
        C = A @ A.T + np.eye(K)
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
        x[a:b] = rng.multivariate_normal(np.zeros(K), C, size=b - a)
        truth[a:b] = C[iu]
    wrong = [0] + sorted(
        rng.choice(np.arange(20, T - 20), len(bounds) - 1, replace=False).tolist()
    )
    return x, bounds, wrong, truth


def _scores(x, truth, bounds, events):
    """Mean per-timepoint recovery r, overall and near the true boundaries."""
    est = tc.timecorr(
        x,
        weights_function=tc.gaussian_weights,
        weights_params={"var": VAR},
        cfun=tc.autofc,
        events=events,
    )
    iu = np.triu_indices(K, 1)
    est = np.moveaxis(tc.vec2mat(est), 2, 0)[:, iu[0], iu[1]]
    r = np.array([np.corrcoef(est[t], truth[t])[0, 1] for t in range(T)])
    near = np.zeros(T, bool)
    for b in bounds[1:]:
        near[max(0, b - NEAR) : b + NEAR] = True
    # a final block of 1 or 2 samples has undefined correlations (r is NaN there)
    return np.nanmean(r), np.nanmean(r[near])


def test_true_events_recover_order1_blocks_better_and_wrong_events_do_not():
    results = {"unrestricted": [], "true": [], "wrong": []}
    for seed in SEEDS:
        x, bounds, wrong, truth = _block_data(seed)
        results["unrestricted"].append(_scores(x, truth, bounds, None))
        results["true"].append(_scores(x, truth, bounds, bounds))
        results["wrong"].append(_scores(x, truth, bounds, wrong))
    mean = {k: np.mean(v, axis=0) for k, v in results.items()}
    overall = {k: v[0] for k, v in mean.items()}
    near = {k: v[1] for k, v in mean.items()}
    assert np.isfinite([*overall.values(), *near.values()]).all()
    assert overall["true"] > overall["unrestricted"] + 0.02, mean
    assert near["true"] > near["unrestricted"] + 0.08, mean
    assert overall["wrong"] < overall["unrestricted"] + 0.005, mean


def test_order2_events_restrict_only_the_order2_analysis_kernel():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 6))
    bounds = [0, 70, 140]
    for rfun in (None, "PCA"):
        kwargs = dict(weights_params={"var": 100}, cfun=tc.autofc, rfun=rfun)
        X, D = tc.features(x, order=2, events={2: bounds}, return_delta=True, **kwargs)
        X_none = tc.features(x, order=2, **kwargs)
        # order 1 and the delta path are not restricted
        np.testing.assert_array_equal(X[1], X_none[1])
        # the order-2 analysis kernel is: X[2] changes (measured maximum
        # differences: 0.81 with rfun=None, 1.55 with PCA) ...
        assert np.abs(X[2] - X_none[2]).max() > 0.1
        # ... and equals timecorr on the delta features with the same events
        expected = tc.timecorr(D[1], events=bounds, **kwargs)
        np.testing.assert_allclose(X[2], expected, rtol=0, atol=1e-10)
