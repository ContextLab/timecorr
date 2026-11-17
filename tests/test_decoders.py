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


@pytest.mark.parametrize("level", [0, 1])
def test_timepoint_decoder_above_chance(level):
    data = shared_ramping_data()
    np.random.seed(0)
    results = tc.timepoint_decoder(data, nfolds=2, level=level)
    acc = _mean_accuracy(results, level)
    # "at least 3 times chance"; isclose absorbs the rounding of a mean of folds
    threshold = 3 * (1 / T)
    assert acc >= threshold or np.isclose(acc, threshold), f"accuracy {acc:.4f} at level {level}"


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


def test_weighted_decoder_weights_use_training_subjects_only():
    """The paper fits level weights on two halves of the training group only."""
    data = shared_ramping_data(S=8, T=30, K=4)
    np.random.seed(3)
    first = weighted_timepoint_decoder(data, nfolds=2, level=1, weights_params={"scale": 5})
    # the decoder's first random draw is the fold assignment
    np.random.seed(3)
    assignments = get_xval_assignments(len(data), 2)
    test_group = np.where(assignments == 0)[0]
    rng = np.random.RandomState(99)
    changed = [rng.randn(30, 4) if j in test_group else x for j, x in enumerate(data)]
    np.random.seed(3)
    second = weighted_timepoint_decoder(
        changed, nfolds=2, level=1, weights_params={"scale": 5}
    )
    cols = ["level_0", "level_1"]
    fold0 = (first["folds"] == 0) & (first["level"] == 1)
    assert np.allclose(first.loc[fold0, cols].to_numpy(), second.loc[fold0, cols].to_numpy())
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


def test_timepoint_decoder_rejects_one_fold():
    data = shared_ramping_data(S=4, T=20, K=4)
    with pytest.raises(ValueError, match="nfolds"):
        tc.timepoint_decoder(data, nfolds=1, level=0)


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
