"""Tests for the timepoint decoders (timecorr.decoding) on the shared pipeline."""

import inspect
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import timecorr as tc
from timecorr.helpers import (
    decoder,
    folding_levels,
    gaussian_weights,
    get_xval_assignments,
    isfc,
    laplace_params,
    laplace_weights,
    mean_combine,
    plot_weights,
    weight_corrs,
    weighted_timepoint_decoder,
    weighted_timepoint_decoder_ec,
)

S = 8
T = 60
K = 6


def _random_corrmat(rng, K):
    """A positive semidefinite correlation matrix."""
    return np.corrcoef(rng.randn(K, 3 * K))


def shared_ramping_data(S=S, T=T, K=K, seed=0):
    """
    S subjects sharing one draw from a ramping correlation template, plus noise.

    The template moves linearly from one random correlation matrix to another
    (as in ``simulate.ramping_dataset``).  One seeded (T, K) draw from it is the
    signal shared by every subject; each subject adds independent Gaussian
    noise with standard deviation 0.5 times the signal's standard deviation.
    """
    rng = np.random.RandomState(seed)
    a = _random_corrmat(rng, K)
    b = _random_corrmat(rng, K)
    mix = np.linspace(1, 0, T)
    signal = np.vstack(
        [
            rng.multivariate_normal(np.zeros(K) + 0.1, mix[t] * a + (1 - mix[t]) * b)
            for t in range(T)
        ]
    )
    noise_sd = 0.5 * signal.std()
    return [signal + noise_sd * rng.randn(T, K) for _ in range(S)]


def _mean_accuracy(results, level):
    return results.loc[results["level"] == level, "accuracy"].mean()


# ------------------------------------------------------- decoding accuracy


@pytest.mark.parametrize("seed", range(5))
def test_timepoint_decoder_level0_above_chance(seed):
    data = shared_ramping_data(seed=seed)
    np.random.seed(seed)
    results = tc.timepoint_decoder(data, nfolds=2, level=0)
    acc = _mean_accuracy(results, 0)
    assert acc >= 3 * (1 / T), f"accuracy {acc:.4f} at level 0, seed {seed}"


@pytest.mark.parametrize("seed", range(5))
def test_timepoint_decoder_level1_above_chance(seed):
    """Level 1 with an analysis kernel suited to T=60 (Laplace, scale 5)."""
    data = shared_ramping_data(seed=seed)
    np.random.seed(seed)
    results = tc.timepoint_decoder(
        data,
        nfolds=2,
        level=1,
        weights_fun=tc.laplace_weights,
        weights_params={"scale": 5},
    )
    acc = _mean_accuracy(results, 1)
    assert acc >= 3 * (1 / T), f"accuracy {acc:.4f} at level 1, seed {seed}"


def test_weighted_decoder_level2_weights_finite_and_sum_to_one():
    data = shared_ramping_data()
    np.random.seed(0)
    results = weighted_timepoint_decoder(data, nfolds=2, level=2)
    assert isinstance(results, pd.DataFrame)
    # one row per (fold, cumulative level set): levels 0, 0-1, 0-2 in each of 2 folds
    assert len(results) == 2 * 3
    full = results[results["level"] == 2]
    mu = full[["level_0", "level_1", "level_2"]].to_numpy()
    assert mu.shape == (2, 3)
    assert np.all(np.isfinite(mu))
    assert np.all(mu >= 0)
    assert np.allclose(mu.sum(axis=1), 1.0)
    for col in ("rank", "accuracy", "error"):
        assert np.all(np.isfinite(results[col]))
        assert np.all((results[col] >= 0) & (results[col] <= 1))


# --------------------------------------------- features equal the pipeline


@pytest.mark.parametrize("level", [1, 2])
def test_decoder_features_equal_pipeline_unreduced(level):
    from timecorr.decoding import _decoder_features

    data = shared_ramping_data(S=6, T=30, K=4)
    groups = [[0, 2, 4], [1, 3, 5]]
    params = {"var": 25}
    feats = _decoder_features(
        data,
        groups,
        level,
        cfun=isfc,
        weights_fun=gaussian_weights,
        weights_params=params,
        combine=mean_combine,
        rfun=None,
    )
    assert len(feats) == level + 1
    for g, members in enumerate(groups):
        X = tc.features(
            [data[j] for j in members],
            order=level,
            weights_function=gaussian_weights,
            weights_params=params,
            cfun=isfc,
            rfun=None,
            combine=mean_combine,
        )
        assert np.allclose(feats[0][g], mean_combine(X[0]), atol=1e-12, rtol=0)
        for k in range(1, level + 1):
            assert feats[k][g].shape == X[k].shape
            assert np.allclose(feats[k][g], X[k], atol=1e-12, rtol=0)


def test_decoder_features_delta_chain_equals_pipeline_pca():
    from timecorr.decoding import _decoder_delta_chain

    data = shared_ramping_data(S=6, T=30, K=4)
    params = {"var": 25}
    D = _decoder_delta_chain(data, level=3, rfun="PCA")
    _, D_ref = tc.features(
        data,
        order=2,
        weights_function=gaussian_weights,
        weights_params=params,
        rfun="PCA",
        return_delta=True,
    )
    assert len(D) == 3
    for k in range(3):
        for s in range(len(data)):
            assert np.allclose(D[k][s], D_ref[k][s], atol=1e-10, rtol=0)


def test_decoder_features_reduce_groups_jointly():
    from timecorr.decoding import _decoder_features
    from timecorr.helpers import reduce

    data = shared_ramping_data(S=6, T=30, K=4)
    groups = [[0, 2, 4], [1, 3, 5]]
    params = {"var": 25}
    feats = _decoder_features(
        data, groups, 1, weights_fun=gaussian_weights, weights_params=params, rfun="PCA"
    )
    unreduced = _decoder_features(
        data, groups, 1, weights_fun=gaussian_weights, weights_params=params, rfun=None
    )
    joint = reduce([unreduced[1][0], unreduced[1][1]], rfun="PCA")
    assert feats[1][0].shape == (30, 4)
    assert np.allclose(feats[1][0], joint[0], atol=1e-10, rtol=0)
    assert np.allclose(feats[1][1], joint[1], atol=1e-10, rtol=0)


# ------------------------------------------------------------- defects


def test_weighted_decoder_level1_runs():
    """Crashed at level 1 (.tolist() on a list) and on concat(join_axes=...)."""
    data = shared_ramping_data(S=6, T=20, K=4)
    np.random.seed(1)
    results = weighted_timepoint_decoder(
        data, nfolds=2, level=1, weights_params={"scale": 5}
    )
    assert list(results["level"]) == [0, 1, 0, 1]
    assert list(results["folds"]) == [0, 0, 1, 1]
    assert np.isnan(results["level_1"].iloc[0])  # level-0 row has one weight
    assert np.allclose(results.loc[results["level"] == 1, ["level_0", "level_1"]].sum(axis=1), 1)


def test_weighted_decoder_prints_nothing(capsys):
    data = shared_ramping_data(S=6, T=20, K=4)
    np.random.seed(1)
    weighted_timepoint_decoder(data, nfolds=2, level=1, weights_params={"scale": 5})
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def _run_weighted(which, data, **kwargs):
    if which == "weighted":
        return weighted_timepoint_decoder(data, **kwargs)
    return weighted_timepoint_decoder_ec([data], **kwargs)


@pytest.mark.parametrize("which", ["weighted", "weighted_ec"])
def test_weighted_decoder_weights_use_training_subjects_only(which):
    """
    The paper fits level weights on two halves of the training group only.

    With no fitted reduction (rfun=None) nothing computed from the test group
    reaches the weights, so replacing the test group's data leaves them unchanged.
    """
    data = shared_ramping_data(S=8, T=30, K=4)
    kwargs = dict(nfolds=2, level=1, weights_params={"scale": 5})
    np.random.seed(3)
    first = _run_weighted(which, data, **kwargs)
    # the decoder's first random draw is the fold assignment
    np.random.seed(3)
    assignments = get_xval_assignments(len(data), 2)
    test_group = np.where(assignments == 0)[0]
    rng = np.random.RandomState(99)
    changed = [rng.randn(30, 4) if j in test_group else x for j, x in enumerate(data)]
    np.random.seed(3)
    second = _run_weighted(which, changed, **kwargs)
    cols = ["level_0", "level_1"]
    fold0 = (first["folds"] == 0) & (first["level"] == 1)
    mu = first.loc[fold0, cols].to_numpy()
    # the optimizer moved away from its uniform start, so equality is not trivial
    assert not np.allclose(mu, 0.5, atol=0.05)
    assert np.allclose(mu, second.loc[fold0, cols].to_numpy())
    # the test group's accuracy does change: its data are now noise
    assert not np.allclose(
        first.loc[fold0, "rank"].to_numpy(), second.loc[fold0, "rank"].to_numpy()
    )


def test_weighted_decoder_nfolds_one_warns_circular():
    data = shared_ramping_data(S=4, T=20, K=4)
    np.random.seed(0)
    with pytest.warns(UserWarning, match="circular"):
        results = weighted_timepoint_decoder(
            data, nfolds=1, level=1, weights_params={"scale": 5}
        )
    assert len(results) == 2 * 2


def test_timepoint_decoder_mu_weights_in_fisher_z():
    """mu mixes the levels' decoding matrices as z2r(sum mu r2z(corrs)) in every fold."""
    from timecorr.decoding import _decoding_matrix, _decoder_features

    data = shared_ramping_data(S=6, T=20, K=4)
    params = {"scale": 5}
    mu = np.array([0.3, 0.7])
    np.random.seed(5)
    results = tc.timepoint_decoder(data, nfolds=2, level=1, mu=mu, weights_params=params)
    np.random.seed(5)
    assignments = get_xval_assignments(len(data), 2)
    expected = []
    for i in range(2):
        groups = [np.where(assignments == i)[0], np.where(assignments != i)[0]]
        feats = _decoder_features(data, groups, 1, weights_params=params)
        corrs = np.array([_decoding_matrix(f[0], f[1]) for f in feats])
        expected.append(decoder(weight_corrs(corrs, mu)))
    expected = pd.concat(expected)
    assert list(results["level"]) == [1, 1]
    for col in ("rank", "accuracy", "error"):
        assert np.allclose(results[col].to_numpy(), expected[col].to_numpy())


def test_timepoint_decoder_one_fold_warns_circular():
    """As in 0.2.0's documented behavior: nfolds=1 warns and splits the subjects in two."""
    data = shared_ramping_data(S=4, T=20, K=4)
    np.random.seed(0)
    with pytest.warns(UserWarning, match="circular"):
        results = tc.timepoint_decoder(data, nfolds=1, level=0)
    assert results["folds"].tolist() == [0, 1]


def test_timepoint_decoder_accepts_dataframes():
    data = shared_ramping_data(S=4, T=20, K=4)
    frames = [pd.DataFrame(x) for x in data]
    np.random.seed(2)
    a = tc.timepoint_decoder(data, nfolds=2, level=1, weights_params={"scale": 5})
    np.random.seed(2)
    b = tc.timepoint_decoder(frames, nfolds=2, level=1, weights_params={"scale": 5})
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_folding_levels_does_not_eval():
    assert "eval(" not in inspect.getsource(folding_levels)


def test_folding_levels_level1_reduces_groups_jointly():
    """folding_levels is one pipeline step for two groups (shared reduction space)."""
    from timecorr.decoding import _decoder_features

    data = shared_ramping_data(S=4, T=20, K=4)
    params = {"scale": 5}
    in_s, out_s, in_r, out_r = folding_levels(
        data[:2],
        data[2:],
        level=1,
        cfun=[isfc, isfc],
        weights_fun=laplace_weights,
        weights_params=params,
        combine=[mean_combine, mean_combine],
        rfun=[None, "PCA"],
    )
    feats = _decoder_features(
        data, [[0, 1], [2, 3]], 1, weights_params=params, rfun="PCA"
    )
    assert np.allclose(in_s, feats[1][0], atol=1e-10, rtol=0)
    assert np.allclose(out_s, feats[1][1], atol=1e-10, rtol=0)
    assert len(in_r) == 2 and len(out_r) == 2
    assert in_r[0].shape == (20, 4) and out_r[1].shape == (20, 4)


def test_weighted_decoder_ec_runs():
    """weighted_timepoint_decoder_ec crashed with a broadcast error."""
    data = np.array(shared_ramping_data(S=6, T=20, K=4))
    np.random.seed(4)
    results = weighted_timepoint_decoder_ec(
        [data], nfolds=2, level=1, weights_params={"scale": 5}
    )
    assert isinstance(results, pd.DataFrame)
    assert list(results["level"]) == [0, 1, 0, 1]
    full = results[results["level"] == 1]
    assert np.allclose(full[["level_0", "level_1"]].sum(axis=1), 1)


def test_pca_decoder_runs():
    from timecorr.helpers import pca_decoder

    data = np.array(shared_ramping_data(S=4, T=20, K=4))
    np.random.seed(6)
    results = pca_decoder(data, nfolds=2, dims=3, weights_params={"scale": 5})
    assert list(results["dims"]) == [1, 2, 3, 1, 2, 3]
    assert np.all((results["accuracy"] >= 0) & (results["accuracy"] <= 1))


# ------------------------------------------------------------- plotting


def test_plot_weights_draws_row_t_without_warning():
    import matplotlib.pyplot as plt

    w = laplace_weights(20, laplace_params)
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = plot_weights(w, t=4, ax=ax)
    assert out is ax
    y = ax.lines[0].get_ydata()
    assert np.allclose(y, w[4, :])
    assert ax.get_xlim() == (1.0, 20.0)
    plt.close(fig)


# ------------------------------------------------- kernel row convention


def test_smooth_one_sided_kernel_uses_only_past_samples():
    """Row t of the kernel weights the samples used at t; smooth must not flip it."""
    from timecorr.helpers import smooth

    def backward(T, params=None):
        return laplace_weights(T, {"scale": 2}, direction="backward")

    w = np.zeros((30, 1))
    w[20, 0] = 1.0  # an impulse at t = 20
    out = smooth(w, windowsize=9, kernel_fun=backward)
    # a backward kernel lets the impulse reach t >= 20 only
    assert np.allclose(out[:20, 0], 0)
    assert out[20:, 0].max() > 0


# ------------------------------------------------- reduction and weights


def test_weighted_decoder_pca_level2_joint_reduction():
    """
    With rfun="PCA" the delta path is reduced by one unsupervised model fit on
    all subjects, test group included (as in the paper, Fig. 10b-c), so the test
    group's data do reach the training features through that shared space.  The
    supervised step, the level weights, is still fit on the training halves only.
    """
    from timecorr.decoding import _decoder_delta_chain

    data = shared_ramping_data(S=8, T=30, K=4)
    np.random.seed(0)
    results = weighted_timepoint_decoder(
        data, nfolds=2, level=2, rfun="PCA", weights_params={"scale": 5}
    )
    assert np.all(np.isfinite(results["accuracy"]))
    full = results[results["level"] == 2][["level_0", "level_1", "level_2"]]
    assert np.all(np.isfinite(full.to_numpy()))
    assert np.allclose(full.sum(axis=1), 1)

    # replacing four subjects changes the other subjects' reduced delta features
    rng = np.random.RandomState(99)
    changed = [rng.randn(30, 4) if j < 4 else x for j, x in enumerate(data)]
    D = _decoder_delta_chain(data, level=2, rfun="PCA")
    D_changed = _decoder_delta_chain(changed, level=2, rfun="PCA")
    assert not np.allclose(D[1][7], D_changed[1][7])


@pytest.mark.parametrize("seed", range(5))
def test_optimize_weights_favors_informative_level(seed):
    """One informative level among two at chance gets the largest weight."""
    from timecorr.decoding import _decoding_matrix, optimize_weights

    rng = np.random.RandomState(seed)
    T_, K_ = 40, 10
    template = rng.randn(T_, K_)
    informative = _decoding_matrix(template + 0.7 * rng.randn(T_, K_), template)
    chance = [_decoding_matrix(rng.randn(T_, K_), rng.randn(T_, K_)) for _ in range(2)]
    corrs = np.array([chance[0], informative, chance[1]])
    mu = optimize_weights(corrs)
    assert np.all(np.isfinite(mu)) and np.all(mu >= -1e-12)
    assert np.isclose(mu.sum(), 1)
    assert np.argmax(mu) == 1
    assert mu[1] >= 0.6  # uniform start: 1/3
    # deterministic
    assert np.array_equal(mu, optimize_weights(corrs))


def test_weighted_decoder_ec_accepts_numpy_integer_level():
    data = np.array(shared_ramping_data(S=6, T=20, K=4))
    np.random.seed(4)
    results = weighted_timepoint_decoder_ec(
        [data], nfolds=2, level=np.int64(1), weights_params={"scale": 5}
    )
    assert list(results["level"]) == [0, 1, 0, 1]


def _shared_signal_data(S=6, T=50, K=5, seed=0):
    """Subjects who share one activity pattern plus independent noise."""
    rng = np.random.RandomState(seed)
    shared = rng.randn(T, K)
    return [shared + 0.5 * rng.randn(T, K) for _ in range(S)]


def test_timepoint_decoder_level0_ignores_corrmean_combine():
    # order-0 features are activations, not correlations: they are averaged
    # with a plain mean whatever combine is given for the correlation orders
    data = _shared_signal_data()
    np.random.seed(0)
    with_corrmean = tc.timepoint_decoder(data, level=0, combine=tc.corrmean_combine)
    np.random.seed(0)
    with_mean = tc.timepoint_decoder(data, level=0, combine=mean_combine)
    pd.testing.assert_frame_equal(with_corrmean, with_mean)
    assert np.all(np.isfinite(with_corrmean["accuracy"]))
    assert with_corrmean["accuracy"].mean() > 0.5  # chance: 1/50


def test_weighted_decoder_level0_ignores_corrmean_combine():
    data = _shared_signal_data(S=8)
    np.random.seed(1)
    a = weighted_timepoint_decoder(data, nfolds=2, level=0, combine=tc.corrmean_combine)
    np.random.seed(1)
    b = weighted_timepoint_decoder(data, nfolds=2, level=0, combine=mean_combine)
    pd.testing.assert_frame_equal(a, b)
    assert a["accuracy"].mean() > 0.5


@pytest.mark.parametrize("fun", [folding_levels, tc.helpers.folding_levels_ec])
def test_folding_levels_level0_ignores_corrmean_combine(fun):
    data = _shared_signal_data()
    kw = {"weights_fun": laplace_weights, "weights_params": {"scale": 5}}
    a = fun(data[:3], data[3:], level=0, combine=tc.corrmean_combine, **kw)
    b = fun(data[:3], data[3:], level=0, combine=mean_combine, **kw)
    assert np.all(np.isfinite(a[0])) and np.all(np.isfinite(a[1]))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ------------------------------------------------------------ folds


@pytest.mark.parametrize("ndata", range(1, 13))
def test_get_xval_assignments_never_makes_empty_folds(ndata):
    for nfolds in range(1, ndata + 1):
        np.random.seed(ndata * 100 + nfolds)
        a = get_xval_assignments(ndata, nfolds)
        assert len(a) == ndata
        sizes = np.bincount(a.astype(int), minlength=nfolds)
        assert len(sizes) == nfolds
        assert sizes.min() >= 1, (ndata, nfolds, sizes)
        assert sizes.max() - sizes.min() <= 1


@pytest.mark.parametrize("nfolds", [0, -1, 8])
def test_get_xval_assignments_rejects_impossible_fold_counts(nfolds):
    with pytest.raises(ValueError, match="nfolds"):
        get_xval_assignments(7, nfolds)


@pytest.mark.parametrize("S,nfolds", [(7, 5), (5, 4)])
def test_decoders_run_when_folds_do_not_divide_subjects(S, nfolds):
    data = shared_ramping_data(S=S, T=30, K=4)
    np.random.seed(0)
    res = tc.timepoint_decoder(data, nfolds=nfolds, level=0)
    assert sorted(res["folds"].unique().tolist()) == list(range(nfolds))
    assert np.all(np.isfinite(res["accuracy"]))
    np.random.seed(0)
    res = weighted_timepoint_decoder(data, nfolds=nfolds, level=1,
                                     weights_params={"scale": 5})
    assert sorted(res["folds"].unique().tolist()) == list(range(nfolds))
    assert np.all(np.isfinite(res["accuracy"]))


@pytest.mark.parametrize("which", ["weighted", "weighted_ec"])
def test_empty_optimize_levels_raises_clear_error(which):
    data = shared_ramping_data(S=6, T=30, K=4)
    with pytest.raises(ValueError, match="optimize_levels must name at least one level"):
        _run_weighted(which, data, nfolds=2, level=1, optimize_levels=[],
                      weights_params={"scale": 5})


# ------------------------------------- all-issues wave (R-C, R-D, R-F, R-G,
# ------------------------------------- R-J, R-L, R-P, R-R)

_DECODERS = {
    "timepoint_decoder": lambda data, **kw: tc.timepoint_decoder(data, **kw),
    "weighted_timepoint_decoder": lambda data, **kw: weighted_timepoint_decoder(
        data, **kw
    ),
    "weighted_timepoint_decoder_ec": lambda data, **kw: weighted_timepoint_decoder_ec(
        [data], **kw
    ),
}


def test_weight_corrs_clips_exact_correlations():
    # R-C (M5): an exact 1 with weight 0 gave 0 * inf = NaN (then 0) and a
    # RuntimeWarning; it is clipped like optimize_weights
    corrs = np.array([np.eye(3), 0.5 * np.eye(3)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed = weight_corrs(corrs, [0.0, 1.0])
    assert np.all(np.isfinite(mixed))
    np.testing.assert_allclose(np.diag(mixed), 0.5, rtol=0, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = weight_corrs(corrs, [1.0, 0.0])
    assert np.all(np.diag(full) > 0.9999) and np.all(np.diag(full) < 1)


@pytest.mark.parametrize("name", sorted(_DECODERS))
def test_decoders_refuse_unreduced_steps_that_exceed_max_memory_gb(name):
    # R-D (M6): K=30, T=20, unreduced: order 1 needs ~1e-3 GB, order 2 (465
    # input features) ~0.04 GB; with max_memory_gb=0.01 the order-2 step is
    # refused before it allocates, as features does
    data = shared_ramping_data(S=4, T=20, K=30)
    np.random.seed(0)
    if name == "weighted_timepoint_decoder_ec":
        # its order-1 input features are given: 465 columns, as unreduced D_1
        rng = np.random.RandomState(1)
        d1 = [rng.randn(20, 465) for _ in data]
        with pytest.raises(MemoryError, match="order 2"):
            weighted_timepoint_decoder_ec([data, d1], nfolds=2, level=2, rfun=None,
                                          max_memory_gb=0.01)
        return
    with pytest.raises(MemoryError, match="order 2"):
        _DECODERS[name](data, nfolds=2, level=2, rfun=None, max_memory_gb=0.01)


def test_folding_levels_refuses_steps_that_exceed_max_memory_gb():
    data = shared_ramping_data(S=4, T=20, K=30)
    with pytest.raises(MemoryError, match="order 1"):
        folding_levels(data[:2], data[2:], level=1, max_memory_gb=1e-5)


def test_decoders_default_memory_limit_refuses_huge_unreduced_steps():
    # order 3 of K=30 unreduced: 108,345 input features, about 4.7e3 GB for
    # the (d, d, T) array alone: refused by the default max_memory_gb=4
    data = shared_ramping_data(S=4, T=20, K=30)
    np.random.seed(0)
    with pytest.raises(MemoryError, match="order 3"):
        tc.timepoint_decoder(data, nfolds=2, level=3, rfun=None)


@pytest.mark.parametrize("name", sorted(_DECODERS))
def test_decoders_accept_weights_function_and_the_weights_fun_alias(name):
    # R-F (M9): weights_function is the documented name; weights_fun is an alias
    data = shared_ramping_data(S=4, T=20, K=4)
    kw = dict(nfolds=2, level=1, weights_params={"var": 10})
    np.random.seed(3)
    a = _DECODERS[name](data, weights_function=gaussian_weights, **kw)
    np.random.seed(3)
    b = _DECODERS[name](data, weights_fun=gaussian_weights, **kw)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    with pytest.raises(TypeError, match="weights_fun, not both"):
        _DECODERS[name](
            data, weights_function=gaussian_weights, weights_fun=gaussian_weights, **kw
        )


def test_decoder_positional_order_is_unchanged():
    # 0.2.0 positional order (the kernel is the sixth argument of
    # timepoint_decoder, the fifth of the folding functions)
    params = list(inspect.signature(tc.timepoint_decoder).parameters)
    assert params[:9] == ["data", "mu", "nfolds", "level", "cfun",
                          "weights_function", "weights_params", "combine", "rfun"]
    for fun in (folding_levels, tc.helpers.folding_levels_ec):
        params = list(inspect.signature(fun).parameters)
        assert params[:8] == ["infold_data", "outfold_data", "level", "cfun",
                              "weights_function", "weights_params", "combine", "rfun"]
    params = list(inspect.signature(tc.helpers.pca_decoder).parameters)
    assert params[:8] == ["data", "nfolds", "dims", "cfun", "weights_function",
                          "weights_params", "combine", "rfun"]


@pytest.mark.parametrize("fun", [folding_levels, tc.helpers.folding_levels_ec])
def test_folding_functions_accept_weights_function(fun):
    data = _shared_signal_data()
    kw = {"weights_params": {"scale": 5}}
    a = fun(data[:3], data[3:], 1, tc.isfc, laplace_weights, **kw)
    b = fun(data[:3], data[3:], level=1, cfun=tc.isfc, weights_fun=laplace_weights, **kw)
    np.testing.assert_array_equal(a[0], b[0])
    with pytest.raises(TypeError, match="weights_fun, not both"):
        fun(data[:3], data[3:], level=1, cfun=tc.isfc, weights_function=laplace_weights,
            weights_fun=laplace_weights, **kw)


def test_pca_decoder_accepts_weights_function_and_validates_with_value_error():
    data = [np.random.RandomState(s).randn(20, 5) for s in range(4)]
    np.random.seed(0)
    a = tc.helpers.pca_decoder(data, nfolds=2, dims=2, weights_function=laplace_weights,
                               weights_params={"scale": 2})
    np.random.seed(0)
    b = tc.helpers.pca_decoder(data, nfolds=2, dims=2, weights_fun=laplace_weights,
                               weights_params={"scale": 2})
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    # R-G (M10): shape mismatches are a ValueError, not an AssertionError
    with pytest.raises(ValueError, match="timepoints"):
        tc.helpers.pca_decoder([data[0], data[1][:10]], nfolds=2, dims=2)
    with pytest.raises(ValueError, match="features"):
        tc.helpers.pca_decoder([data[0], data[1][:, :3]], nfolds=2, dims=2)


def test_folding_levels_ec_without_a_weights_function_raises_value_error():
    # R-J (M13): it raised "TypeError: 'NoneType' object is not callable"
    data = _shared_signal_data()
    with pytest.raises(ValueError, match="weights_function"):
        tc.helpers.folding_levels_ec(data[:3], data[3:], level=1, cfun=tc.isfc)


def test_timepoint_decoder_one_fold_warning_says_what_happens():
    # R-L (M18): the split into two groups is not circular; say so
    data = shared_ramping_data(S=4, T=20, K=4)
    np.random.seed(0)
    with pytest.warns(UserWarning, match="would be circular; using nfolds=2 instead"):
        tc.timepoint_decoder(data, nfolds=1, level=0)


@pytest.mark.parametrize("nfolds", [2.5, "2", True])
def test_get_xval_assignments_rejects_non_integer_fold_counts(nfolds):
    # R-P: 2.5 was truncated to 2
    with pytest.raises(ValueError, match="nfolds"):
        get_xval_assignments(5, nfolds)


def test_get_xval_assignments_accepts_numpy_integers():
    np.random.seed(0)
    assert np.bincount(get_xval_assignments(5, np.int64(2)).astype(int)).tolist() == [3, 2]


def test_reduce_wrapper_reduces_both_groups_jointly_and_keeps_their_structure():
    # R-R: the raw entries of both groups (one array per subject) are reduced
    # into one space and returned as the groups they came in
    from timecorr.decoding import reduce_wrapper

    rng = np.random.RandomState(7)
    raw_in = [rng.randn(20, 10) for _ in range(2)]
    raw_out = [rng.randn(20, 10) for _ in range(3)]
    smooth_in, smooth_out = rng.randn(20, 10), rng.randn(20, 10)
    out = reduce_wrapper((smooth_in, smooth_out, raw_in, raw_out), level=1,
                         rfun=[None, "PCA"])
    joint = tc.reduce(raw_in + raw_out, rfun="PCA")
    assert isinstance(out[2], list) and len(out[2]) == 2
    assert isinstance(out[3], list) and len(out[3]) == 3
    for got, ref in zip(out[2] + out[3], joint):
        np.testing.assert_allclose(got, ref)
    joint_smooth = tc.reduce([smooth_in, smooth_out], rfun="PCA")
    np.testing.assert_allclose(out[0], joint_smooth[0])
    np.testing.assert_allclose(out[1], joint_smooth[1])
    # one array per group (the layout reduce_wrapper was written for)
    grouped = reduce_wrapper((smooth_in, smooth_out, raw_in[0], raw_out[0]), level=1,
                             rfun=[None, "PCA"])
    pair = tc.reduce([raw_in[0], raw_out[0]], rfun="PCA")
    np.testing.assert_allclose(grouped[2], pair[0])
    np.testing.assert_allclose(grouped[3], pair[1])


def test_smooth_accepts_weights_function_and_the_kernel_aliases():
    # R-F (M9): smooth's kernel arguments are named as everywhere else
    x = np.random.RandomState(8).randn(30, 2)
    a = tc.smooth(x, 5, gaussian_weights, {"var": 4})
    b = tc.smooth(x, windowsize=5, weights_function=gaussian_weights,
                  weights_params={"var": 4})
    c = tc.smooth(x, windowsize=5, kernel_fun=gaussian_weights, kernel_params={"var": 4})
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    with pytest.raises(TypeError, match="kernel_fun, not both"):
        tc.smooth(x, 5, weights_function=gaussian_weights, kernel_fun=gaussian_weights)
    with pytest.raises(TypeError, match="kernel_params, not both"):
        tc.smooth(x, 5, weights_params={"var": 4}, kernel_params={"var": 4})


# ------------------------------------------------- review minors (m2)


@pytest.mark.parametrize("nfolds", [2.0, np.float64(2.0)])
def test_get_xval_assignments_accepts_integer_valued_floats(nfolds):
    # m2: 0.2.0 accepted nfolds=2.0; an integer-valued number is that integer
    np.random.seed(0)
    out = get_xval_assignments(5, nfolds)
    assert np.bincount(out.astype(int)).tolist() == [3, 2]


def test_get_xval_assignments_rejects_two_and_a_half_clearly():
    with pytest.raises(ValueError, match=r"nfolds must be an integer; got 2\.5"):
        get_xval_assignments(5, 2.5)


def test_timepoint_decoder_accepts_float_fold_count():
    data = shared_ramping_data(S=4, T=20, K=4)
    np.random.seed(0)
    a = tc.timepoint_decoder(data, nfolds=2.0, level=0)
    np.random.seed(0)
    b = tc.timepoint_decoder(data, nfolds=2, level=0)
    pd.testing.assert_frame_equal(a, b)


def test_pca_decoder_accepts_float_fold_count():
    from timecorr.helpers import pca_decoder

    data = np.array(shared_ramping_data(S=4, T=20, K=4))
    np.random.seed(0)
    a = pca_decoder(data, nfolds=2.0, dims=3, weights_params={"scale": 5})
    np.random.seed(0)
    b = pca_decoder(data, nfolds=2, dims=3, weights_params={"scale": 5})
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------- event-restricted kernels

_EV_KW = dict(weights_function=gaussian_weights, weights_params={"var": 25})
# each decoder with the settings of a small run (S=6, T=24, K=3)
_EV_RUNS = {
    "timepoint_decoder": lambda data, **kw: tc.timepoint_decoder(
        data, nfolds=2, level=2, **_EV_KW, **kw
    ),
    "weighted_timepoint_decoder": lambda data, **kw: weighted_timepoint_decoder(
        data, nfolds=2, level=2, **_EV_KW, **kw
    ),
    "weighted_timepoint_decoder_ec": lambda data, **kw: weighted_timepoint_decoder_ec(
        [data], nfolds=2, level=1, **_EV_KW, **kw
    ),
    "pca_decoder": lambda data, **kw: tc.helpers.pca_decoder(
        data, nfolds=2, dims=2, **_EV_KW, **kw
    ),
}


def _seeded(run, seed=3):
    np.random.seed(seed)
    return run().reset_index(drop=True)


def _differing_boundaries():
    # one boundary set per subject (S=6, T=24), no two alike
    return [[8], [12], [6, 16], [10], [], [5, 14]]


def test_decoders_events_and_events_params_are_keyword_only():
    for f, positional in [
        (tc.timepoint_decoder, ["data", "mu", "nfolds", "level", "cfun",
                                "weights_function", "weights_params", "combine",
                                "rfun", "weights_fun", "max_memory_gb"]),
        (weighted_timepoint_decoder, ["data", "nfolds", "level", "optimize_levels",
                                      "cfun", "weights_function", "weights_params",
                                      "combine", "rfun", "opt_init", "weights_fun",
                                      "max_memory_gb"]),
        (weighted_timepoint_decoder_ec, ["data", "nfolds", "level", "optimize_levels",
                                         "cfun", "weights_function", "weights_params",
                                         "combine", "rfun", "opt_init", "weights_fun",
                                         "max_memory_gb"]),
        (tc.helpers.pca_decoder, ["data", "nfolds", "dims", "cfun", "weights_function",
                                  "weights_params", "combine", "rfun", "weights_fun"]),
    ]:
        params = inspect.signature(f).parameters
        names = [n for n, p in params.items()
                 if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert names == positional, f.__name__
        assert list(params) == positional + ["events", "events_params"], f.__name__
        assert all(params[n].kind == inspect.Parameter.KEYWORD_ONLY
                   for n in ("events", "events_params")), f.__name__
        assert params["events"].default is None
        assert params["events_params"].default is None


@pytest.mark.parametrize("name", sorted(_EV_RUNS))
def test_decoders_events_none_and_one_block_are_unchanged(name):
    # guards that events=None and events=[] (one block) give exactly the
    # default call's result, for the same seed
    data = shared_ramping_data(S=6, T=24, K=3)
    run = _EV_RUNS[name]
    today = _seeded(lambda: run(data))
    pd.testing.assert_frame_equal(_seeded(lambda: run(data, events=None)), today,
                                  check_exact=True)
    # an empty boundary set is one block: no restriction
    pd.testing.assert_frame_equal(_seeded(lambda: run(data, events=[])), today,
                                  check_exact=True)


def test_decoder_features_with_shared_events_equal_pipeline():
    from timecorr.decoding import _decoder_features
    from timecorr.events import resolve_events_by_order

    data = shared_ramping_data(S=6, T=24, K=3)
    groups = [[0, 2, 4], [1, 3, 5]]
    events = {1: [12], 2: [8, 16]}
    resolved = resolve_events_by_order(events, 24, 6, 2)
    feats = _decoder_features(data, groups, 2, rfun=None, combine=mean_combine,
                              events=resolved, **_EV_KW)
    plain = _decoder_features(data, groups, 2, rfun=None, combine=mean_combine,
                              **_EV_KW)
    for g, members in enumerate(groups):
        X = tc.features([data[j] for j in members], order=2, cfun=isfc, rfun=None,
                        combine=mean_combine, events=events, **_EV_KW)
        assert np.array_equal(feats[0][g], plain[0][g])  # level 0 unchanged
        for k in (1, 2):
            assert np.allclose(feats[k][g], X[k], atol=1e-12, rtol=0)
            assert not np.allclose(feats[k][g], plain[k][g], atol=1e-3)


def test_decoder_features_per_subject_events_follow_original_subject_index():
    from timecorr.decoding import _decoder_features
    from timecorr.events import resolve_events_by_order

    data = shared_ramping_data(S=6, T=24, K=3)
    events = _differing_boundaries()
    groups = [[4, 0, 3], [5, 2, 1]]  # shuffled membership
    resolved = resolve_events_by_order(events, 24, 6, 1)
    feats = _decoder_features(data, groups, 1, rfun=None, combine=mean_combine,
                              events=resolved, **_EV_KW)
    for g, members in enumerate(groups):
        own = tc.features([data[j] for j in members], order=1, rfun=None,
                          combine=mean_combine, events=[events[j] for j in members],
                          **_EV_KW)
        # the group's subjects given the events of positions 0..2 instead
        wrong = tc.features([data[j] for j in members], order=1, rfun=None,
                            combine=mean_combine, events=events[:3], **_EV_KW)
        assert np.allclose(feats[1][g], own[1], atol=1e-12, rtol=0)
        assert not np.allclose(feats[1][g], wrong[1], atol=1e-3)


def test_timepoint_decoder_per_subject_events_follow_subjects_through_folds():
    from timecorr.decoding import _level_matrices
    from timecorr.events import resolve_events_by_order

    data = shared_ramping_data(S=6, T=24, K=3)
    events = _differing_boundaries()
    np.random.seed(5)
    results = tc.timepoint_decoder(data, nfolds=3, level=[0, 1], events=events,
                                 **_EV_KW)
    resolved = resolve_events_by_order(events, 24, 6, 1)
    np.random.seed(5)
    assignments = get_xval_assignments(6, 3)
    D = [data]
    rows = []
    for i in range(3):
        decode = np.where(assignments == i)[0]
        template = np.where(assignments != i)[0]
        corrs = _level_matrices(data, decode, template, 1, D, rfun=None,
                                events=resolved, **_EV_KW)
        for v in (0, 1):
            rows.append(decoder(corrs[v]))
    expected = pd.concat(rows).reset_index(drop=True)
    got = results[["rank", "accuracy", "error"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    np.random.seed(5)
    plain = tc.timepoint_decoder(data, nfolds=3, level=[0, 1], **_EV_KW)
    assert not plain[["rank"]].reset_index(drop=True).equals(got[["rank"]])


def test_weighted_decoder_per_subject_events_follow_subjects_into_training_halves():
    from timecorr.decoding import _level_matrices, optimize_weights
    from timecorr.events import resolve_events_by_order

    data = shared_ramping_data(S=6, T=24, K=3)
    events = _differing_boundaries()
    np.random.seed(7)
    results = weighted_timepoint_decoder(data, nfolds=2, level=1, events=events,
                                         **_EV_KW)
    resolved = resolve_events_by_order(events, 24, 6, 1)
    np.random.seed(7)
    assignments = get_xval_assignments(6, 2)
    D = [data]
    for i in range(2):
        test = np.where(assignments == i)[0]
        train = np.where(assignments != i)[0]
        corrs = _level_matrices(data, test, train, 1, D, rfun=None, events=resolved,
                                **_EV_KW)
        halves = get_xval_assignments(len(train), 2)
        sub = _level_matrices(data, train[halves == 0], train[halves == 1], 1, D,
                              rfun=None, events=resolved, **_EV_KW)
        mu = optimize_weights(sub[[0, 1]])
        row = results[(results["folds"] == i) & (results["level"] == 1)]
        assert np.array_equal(row[["level_0", "level_1"]].values[0], mu)
        expected = decoder(weight_corrs(corrs[[0, 1]], mu))
        assert np.array_equal(row[["rank", "accuracy", "error"]].values,
                              expected.values)


@pytest.mark.parametrize("name", sorted(_EV_RUNS))
def test_decoders_run_with_events_and_return_the_documented_shape(name):
    data = shared_ramping_data(S=6, T=24, K=3)
    run = _EV_RUNS[name]
    today = _seeded(lambda: run(data))
    shared = _seeded(lambda: run(data, events=[12]))
    per_subject = _seeded(lambda: run(data, events=_differing_boundaries()))
    for res in (shared, per_subject):
        assert list(res.columns) == list(today.columns)
        assert len(res) == len(today)
        for col in ("rank", "accuracy", "error"):
            assert np.all((res[col] >= 0) & (res[col] <= 1))
    for res in (shared, per_subject):
        # every decoder's ranks change when its kernels are restricted to events
        assert not np.array_equal(res["rank"].values, today["rank"].values)
        if name in ("weighted_timepoint_decoder", "weighted_timepoint_decoder_ec"):
            # level 0 (the raw data) does not use the kernel; levels >= 1 do
            pd.testing.assert_frame_equal(res[res.level == 0],
                                          today[today.level == 0],
                                          check_exact=True)
            assert not np.array_equal(res[res.level >= 1]["rank"].values,
                                      today[today.level >= 1]["rank"].values)


@pytest.mark.parametrize("name", sorted(_EV_RUNS))
def test_decoders_check_events_before_computing(name):
    data = shared_ramping_data(S=6, T=24, K=3)
    run = _EV_RUNS[name]
    with pytest.raises(ValueError):
        run(data, events=[[4], [8]])  # two entries for six subjects
    with pytest.raises(ValueError):
        run(data, events=[30])  # outside [0, T)
    with pytest.raises(ValueError, match="events_params"):
        run(data, events=[12], events_params={"n": 2})  # no features to segment


def test_level_zero_decoders_and_pca_decoder_refuse_events_by_order():
    data = shared_ramping_data(S=6, T=24, K=3)
    np.random.seed(0)
    with pytest.raises(ValueError, match="order"):
        tc.timepoint_decoder(data, level=0, events={1: [12]})
    with pytest.raises(ValueError, match="order"):
        tc.helpers.pca_decoder(data, dims=2, events={1: [12]})
    # a level-0 decoder checks the events and is otherwise unchanged
    np.random.seed(0)
    a = tc.timepoint_decoder(data, level=0, events=[12])
    np.random.seed(0)
    b = tc.timepoint_decoder(data, level=0)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_decoders_segment_features_events_once_for_all_folds():
    # a features array with T rows is segmented once, on the full subject list;
    # the decoder equals one run with the resulting boundaries
    from timecorr.events import segment

    data = shared_ramping_data(S=6, T=24, K=3)
    rng = np.random.RandomState(2)
    stim = np.repeat([0.0, 3.0, 0.0], 8)[:, None] + 0.2 * rng.randn(24, 2)
    bounds = segment(stim, n=3)
    np.random.seed(1)
    a = tc.timepoint_decoder(data, level=1, events=stim, events_params={"n": 3},
                             **_EV_KW)
    np.random.seed(1)
    b = tc.timepoint_decoder(data, level=1, events=bounds, **_EV_KW)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def _pca_decoder_by_hand(data, nfolds, dims, events_of, seed):
    """pca_decoder recomputed with ``tc.timecorr`` smoothing, per fold."""
    import hypertools as hyp
    import scipy.spatial.distance as sd

    pca_data = np.asarray(hyp.reduce(list(data), ndims=min(dims, data[0].shape[1]),
                                     random_state=0))
    np.random.seed(seed)
    assignments = get_xval_assignments(len(data), nfolds)
    rows = []
    for i in range(nfolds):
        for d in range(1, dims + 1):
            smooth = []
            for members in (np.where(assignments == i)[0],
                            np.where(assignments != i)[0]):
                x = np.asarray(tc.timecorr([pca_data[j][:, :d] for j in members],
                                           cfun=None, rfun=None, combine=mean_combine,
                                           events=events_of(members), **_EV_KW))
                if d < 3:
                    x = np.hstack((x, np.zeros((x.shape[0], 3 - x.shape[1]))))
                smooth.append(x)
            rows.append(decoder(1 - sd.cdist(smooth[0], smooth[1], "correlation")))
    return pd.concat(rows).reset_index(drop=True)


def test_pca_decoder_smoothing_kernel_follows_each_subjects_events():
    data = shared_ramping_data(S=6, T=24, K=3)
    events = _differing_boundaries()
    np.random.seed(4)
    got = tc.helpers.pca_decoder(data, nfolds=2, dims=2, events=events, **_EV_KW)
    got = got[["rank", "accuracy", "error"]].reset_index(drop=True)
    own = _pca_decoder_by_hand(data, 2, 2, lambda m: [events[j] for j in m], seed=4)
    pd.testing.assert_frame_equal(got, own, check_exact=True)
    # the fold's members given the events of positions 0..n-1 instead
    wrong = _pca_decoder_by_hand(data, 2, 2, lambda m: events[: len(m)], seed=4)
    assert not np.array_equal(got.values, wrong.values)
    # shared boundaries are passed as they are
    np.random.seed(4)
    shared = tc.helpers.pca_decoder(data, nfolds=2, dims=2, events=[12], **_EV_KW)
    pd.testing.assert_frame_equal(
        shared[["rank", "accuracy", "error"]].reset_index(drop=True),
        _pca_decoder_by_hand(data, 2, 2, lambda m: [12], seed=4),
        check_exact=True,
    )


@pytest.mark.parametrize("name", sorted(_EV_RUNS))
def test_decoders_segmentation_leaves_the_global_random_state_unchanged(name):
    # recurring segmentation without random_state draws from NumPy's global
    # state; the decoders restore it, so folds match the no-events run
    data = shared_ramping_data(S=6, T=24, K=3)
    run = _EV_RUNS[name]
    rng = np.random.RandomState(2)
    stim = np.repeat([0.0, 3.0, 0.0], 8)[:, None] + 0.2 * rng.randn(24, 2)
    params = {"n": 2, "kind": "recurring"}
    np.random.seed(11)
    run(data)
    after_plain = np.random.random_sample(3)
    np.random.seed(11)
    with_events = run(data, events=stim, events_params=params)
    after_events = np.random.random_sample(3)
    assert np.array_equal(after_events, after_plain)
    # the fold assignment is the one the seed gives without events
    np.random.seed(11)
    folds = get_xval_assignments(6, 2)
    np.random.seed(11)
    resolved_state = np.random.get_state()
    from timecorr.events import resolve_events
    resolve_events(stim, 24, 6, params)  # advances the global state
    assert not np.array_equal(np.random.get_state()[1], resolved_state[1])
    np.random.seed(11)
    assert np.array_equal(get_xval_assignments(6, 2), folds)
    assert len(with_events) == len(_seeded(lambda: run(data)))


# ------------------------------------------------------ summaries as rfun


def test_timepoint_decoder_accepts_a_summary_rfun():
    data = shared_ramping_data(K=10, seed=3)
    ws = tc.WeightedStrength()
    np.random.seed(3)
    results = tc.timepoint_decoder(
        data, nfolds=2, level=[0, 1, 2], weights_fun=tc.laplace_weights,
        weights_params={"scale": 5}, rfun="weighted_strength",
    )
    assert list(results.columns) == ["rank", "accuracy", "error", "level", "folds"]
    assert results["level"].tolist() == [0, 1, 2, 0, 1, 2]
    assert results["folds"].tolist() == [0, 0, 0, 1, 1, 1]
    values = results[["rank", "accuracy", "error"]].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    # an instance: each call keeps only its latest fit
    np.random.seed(3)
    tc.timepoint_decoder(data, nfolds=2, level=1, weights_fun=tc.laplace_weights,
                         weights_params={"scale": 5}, rfun=ws)
    assert len(ws.fits_) == 1 and len(ws.fits_[0]["weights"]) > 0
