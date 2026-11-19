"""Tests for timecorr.simulate: template generators and the sign-parity cascade."""

import inspect
import itertools
import re
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.helpers import mat2vec, vec2mat
from timecorr.simulate import random_corrmat

# The settings of the original two-path pipeline (co-fluctuations on the delta
# path, PCA on both paths; the former defaults of features), merged under the
# arguments of the calls of tests that pin its numbers, shapes, widths or
# messages.
LEGACY = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)


def legacy_features(*args, **kwargs):
    return tc.features(*args, **{**LEGACY, **kwargs})

DATAGENS = ["block", "ramping", "constant", "random"]


def _zscore(x):
    # the delta path z-scores each column over all timepoints (population sd)
    return (x - x.mean(0)) / x.std(0)


def _offdiag(p):
    """Columns of the off-diagonal entries in mat2vec layout for p x p matrices."""
    return np.arange(p, p * (p + 1) // 2)


# ---------------------------------------------------------------- templates


@pytest.mark.parametrize("K", [2, 3, 5, 10, 50])
def test_random_corrmat_is_a_correlation_matrix(K):
    for seed in range(5):
        C = random_corrmat(K, rng=np.random.default_rng(seed))
        assert C.shape == (K, K)
        assert np.allclose(C, C.T, atol=0, rtol=0)
        assert np.allclose(np.diag(C), 1.0, atol=1e-12, rtol=0)
        assert np.linalg.eigvalsh(C).min() >= -1e-10
        assert np.abs(C).max() <= 1.0 + 1e-12


def test_random_corrmat_without_rng_is_psd():
    for _ in range(20):
        assert np.linalg.eigvalsh(random_corrmat(8)).min() >= -1e-10


@pytest.mark.parametrize("datagen", DATAGENS)
def test_templates_are_correlation_matrices(datagen):
    _, corrs = tc.simulate_data(
        datagen=datagen, return_corrs=True, set_random_seed=3, T=30, K=6, B=4
    )
    R = vec2mat(corrs)
    assert R.shape == (6, 6, 30)
    for t in range(30):
        assert np.allclose(np.diag(R[:, :, t]), 1.0)
        assert np.linalg.eigvalsh(R[:, :, t]).min() >= -1e-10


def test_block_template_has_integer_event_boundaries():
    _, corrs = tc.simulate_data(
        datagen="block", return_corrs=True, set_random_seed=1, T=103, K=4, B=5
    )
    assert corrs.shape == (103, 10)
    changes = np.flatnonzero(np.any(np.diff(corrs, axis=0) != 0, axis=1)) + 1
    assert len(changes) == 4
    lengths = np.diff(np.concatenate([[0], changes, [103]]))
    assert lengths.min() >= 103 // 5 and lengths.max() <= 103 // 5 + 1


# ---------------------------------------------------------- warnings, seeds


@pytest.mark.parametrize("datagen", DATAGENS)
@pytest.mark.parametrize("order", [1, 2])
def test_no_warnings_are_silenced_or_raised(datagen, order):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        before = list(warnings.filters)
        data, truth = tc.simulate_data(
            datagen=datagen,
            return_corrs=True,
            set_random_seed=2,
            S=2,
            T=40,
            K=8,
            order=order,
        )
        assert list(warnings.filters) == before
    assert len(data) == 2 and data[0].shape == (40, 8)


def test_global_numpy_random_state_is_untouched():
    np.random.seed(5)
    expected = np.random.rand(3)
    np.random.seed(5)
    tc.simulate_data(set_random_seed=1, T=20, K=4)
    tc.simulate_data(set_random_seed=True, T=20, K=8, order=2)
    assert np.array_equal(np.random.rand(3), expected)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("datagen", DATAGENS)
def test_seeding(datagen, order):
    kw = dict(datagen=datagen, S=2, T=25, K=8, order=order, return_corrs=True)
    a, ta = tc.simulate_data(set_random_seed=7, **kw)
    b, tb = tc.simulate_data(set_random_seed=7, **kw)
    c, _ = tc.simulate_data(set_random_seed=8, **kw)
    for s in range(2):
        assert np.array_equal(a[s], b[s])
        assert not np.array_equal(a[s], c[s])
    t1, t2 = tc.simulate_data(set_random_seed=True, **kw), tc.simulate_data(
        set_random_seed=123, **kw
    )
    assert np.array_equal(t1[0][0], t2[0][0])
    u1, u2 = tc.simulate_data(set_random_seed=False, **kw), tc.simulate_data(**kw)
    assert not np.array_equal(u1[0][0], u2[0][0])


def test_invalid_seed_raises():
    with pytest.raises(TypeError, match="set_random_seed"):
        tc.simulate_data(set_random_seed="abc")


# ------------------------------------------------------------ multi-subject


@pytest.mark.parametrize("datagen", DATAGENS)
def test_order1_subjects_share_template(datagen):
    data, corrs = tc.simulate_data(
        datagen=datagen, return_corrs=True, set_random_seed=4, S=3, T=30, K=5
    )
    assert isinstance(data, list) and isinstance(corrs, list)
    assert len(data) == 3 and len(corrs) == 3
    for s in range(1, 3):
        assert np.array_equal(corrs[s], corrs[0])
        assert not np.array_equal(data[s], data[0])
        assert corrs[s].shape == (30, 15)


def test_order2_subjects_share_latent_signs_not_free_signs():
    data, truth = tc.simulate_data(
        datagen="block",
        return_corrs=True,
        set_random_seed=4,
        S=3,
        T=200,
        K=8,
        order=2,
    )
    assert len(truth) == 3
    blocks = truth[0]["blocks"]
    products = [np.prod(np.sign(d[:, blocks]), axis=2) for d in data]
    for s in range(1, 3):
        assert np.array_equal(truth[s]["corrs"], truth[0]["corrs"])
        assert np.array_equal(truth[s]["blocks"], blocks)
        assert np.array_equal(products[s], products[0])
        assert not np.array_equal(np.sign(data[s]), np.sign(data[0]))


# ---------------------------------------------------------- order handling


def test_min_features():
    assert tc.min_features(1) == 3
    assert tc.min_features(3) == 12
    assert tc.min_features(2, p=4) == 8
    assert tc.min_features(6, p=4) == 128
    with pytest.raises(ValueError):
        tc.min_features(0)
    with pytest.raises(ValueError):
        tc.min_features(2, p=1)


def test_wrong_k_for_order_raises_naming_min_features():
    with pytest.raises(ValueError, match="min_features"):
        tc.simulate_data(K=6, order=3)
    with pytest.raises(ValueError, match="min_features"):
        tc.simulate_data(K=4, order=3)  # p = 1
    with pytest.raises(ValueError, match="min_features"):
        tc.simulate_data(K=1, order=1)


@pytest.mark.parametrize("order", [0, -1, 1.5, True])
def test_invalid_order_raises(order):
    with pytest.raises(ValueError, match="order"):
        tc.simulate_data(K=8, order=order)


def test_negative_amplitude_noise_raises():
    with pytest.raises(ValueError, match="amplitude_noise"):
        tc.simulate_data(K=8, order=2, amplitude_noise=-0.1)


def test_unknown_datagen_raises():
    with pytest.raises(ValueError, match="datagen"):
        tc.simulate_data(datagen="sawtooth")


@pytest.mark.parametrize("K,order", [(2, 1), (4, 2)])
def test_two_template_features_warn_single_entry(K, order):
    with pytest.warns(UserWarning, match="single entry") as record:
        tc.simulate_data(K=K, order=order, T=20, set_random_seed=1)
    assert len(record) == 1


def test_order3_warns_once_about_variance_based_reduction():
    with pytest.warns(UserWarning) as record:
        tc.simulate_data(K=12, order=3, T=20, set_random_seed=1)
    assert len(record) == 1
    msg = str(record[0].message)
    assert "PCA" in msg and "carrier_reducer" in msg and "rfun=None" in msg
    assert "order >= 3" in msg
    # states what was measured, and no more
    assert "not distinguishable from its noise control" in msg
    assert "-0.00 versus 0.02, p = 0.20" in msg
    assert "orders above 3 were not measured" in msg
    assert "cannot" not in msg and "none" not in msg


def test_order2_truth_structure():
    data, truth = tc.simulate_data(
        datagen="ramping", return_corrs=True, set_random_seed=0, T=50, K=12, order=2
    )
    assert data.shape == (50, 12)
    assert set(truth) == {"order", "corrs", "requested", "blocks"}
    assert truth["requested"].shape == (50, 21)
    assert truth["order"] == 2
    assert truth["corrs"].shape == (50, 21)
    assert np.array_equal(truth["blocks"], np.arange(12).reshape(6, 2))
    assert truth["blocks"].dtype.kind == "i"
    assert np.allclose(truth["corrs"][:, :6], 1.0)
    # amplitude noise makes the data non-binary but keeps the signs
    assert len(np.unique(np.abs(data))) > 2


def test_amplitude_noise_zero_gives_signs():
    data = tc.simulate_data(set_random_seed=0, T=50, K=8, order=2, amplitude_noise=0)
    assert set(np.unique(data)) == {-1.0, 1.0}


@pytest.mark.parametrize("order", [2, 5])
def test_amplitude_noise_is_the_sd_of_each_block_products_log_amplitude(order):
    # Ruling R21: amplitude_noise is the SD of log|product| of each block, at
    # every order (per-sign SD = amplitude_noise / sqrt(2**(order-1))).
    s, T, p = 0.3, 4000, 4
    K = tc.min_features(order, p=p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the order >= 3 reduction warning
        data, truth = tc.simulate_data(
            return_corrs=True, set_random_seed=3, T=T, K=K, order=order,
            amplitude_noise=s,
        )
    for block in truth["blocks"]:
        log_amp = np.log(np.abs(data[:, block])).sum(axis=1)
        # log|product| is Gaussian; the sample SD has standard error s / sqrt(2T)
        assert abs(log_amp.std(ddof=1) - s) <= 4 * s / np.sqrt(2 * T)
        # each member carries an equal share of the variance
        per_sign = np.log(np.abs(data[:, block])).std(axis=0, ddof=1)
        expected = s / np.sqrt(2 ** (order - 1))
        assert np.all(np.abs(per_sign - expected) <= 4 * expected / np.sqrt(2 * T))


def test_sign_parity_cascade_order3():
    T = 4000
    with pytest.warns(UserWarning, match="carrier_reducer"):
        data, truth = tc.simulate_data(
            datagen="block",
            return_corrs=True,
            set_random_seed=11,
            T=T,
            K=12,
            order=3,
            B=4,
        )
    blocks = truth["blocks"]
    assert blocks.shape == (3, 4)
    signs = np.sign(data)
    A = np.prod(signs[:, blocks], axis=2)  # (T, p) block products
    # signs carry the latent-sign correlation (truth without amplitude noise)
    target = vec2mat(_latent(truth["corrs"], 3).mean(axis=0))
    assert np.abs(np.corrcoef(A.T) - target).max() <= 0.08
    for b in blocks:
        for i, j in itertools.combinations(b, 2):
            assert abs(np.mean(signs[:, i] * signs[:, j])) < 0.06
        assert abs(np.mean(np.prod(signs[:, b[:3]], axis=1))) < 0.06
        for i in b:
            assert abs(np.mean(signs[:, i])) < 0.06


def _carried(latent, p, a=0.2):
    """Correlation of the block products for latent-sign correlation ``latent``
    (rows in mat2vec layout): off-diagonals times exp(-a**2), a the amplitude
    noise (default 0.2), computed as the generator does."""
    c = np.array(latent, dtype=float, copy=True)
    c[..., p:] *= np.exp(-(a**2))
    return c


def _latent(carried, p, a=0.2):
    """Inverse of ``_carried``: the correlation of the signs of the products."""
    c = np.array(carried, dtype=float, copy=True)
    c[..., p:] /= np.exp(-(a**2))
    return c


def _needs_clipping(corrs):
    """Whether sin(pi R / 2) has an eigenvalue below the generator's 1e-6 floor."""
    G = np.sin(np.pi * vec2mat(corrs) / 2)
    return np.linalg.eigvalsh(np.transpose(G, (2, 0, 1))).min(axis=1) < 1e-6


def test_truth_is_the_achieved_correlation_at_p10():
    # p=10: sin(pi R / 2) is almost never PSD, so the requested template cannot
    # be reached exactly.  corr of two +/-1 signs has standard error at most
    # 1/sqrt(T) = 1/sqrt(40000) = 0.005; the 0.03 tolerance is 6 standard errors.
    data, truth = tc.simulate_data(
        datagen="constant",
        return_corrs=True,
        set_random_seed=21,
        T=40000,
        K=20,
        order=2,
    )
    assert _needs_clipping(truth["requested"][:1]).all()
    A = np.prod(np.sign(data[:, truth["blocks"]]), axis=2)
    empirical = np.corrcoef(A.T)
    # signs carry the latent-sign correlation (truth without amplitude noise)
    assert np.abs(empirical - vec2mat(_latent(truth["corrs"][0], 10))).max() <= 0.03
    # the requested template differs from what the data carry
    assert np.abs(vec2mat(truth["requested"][0]) - empirical).max() > 0.05
    assert np.allclose(truth["corrs"][:, :10], 1.0)
    assert np.linalg.eigvalsh(vec2mat(truth["corrs"][0])).min() >= -1e-10


def test_truth_is_the_achieved_correlation_block_p6():
    # two events of 20000 timepoints: standard error <= 1/sqrt(20000) = 0.0071,
    # tolerance 0.04 is more than 5 standard errors
    T = 40000
    data, truth = tc.simulate_data(
        datagen="block",
        return_corrs=True,
        set_random_seed=3,
        T=T,
        K=12,
        B=2,
        order=2,
    )
    clipped = _needs_clipping(truth["requested"])
    assert clipped.any()
    A = np.prod(np.sign(data[:, truth["blocks"]]), axis=2)
    for event in (slice(0, T // 2), slice(T // 2, T)):
        target = vec2mat(_latent(truth["corrs"][event][0], 6))
        assert np.abs(np.corrcoef(A[event].T) - target).max() <= 0.04
    # unclipped timepoints report the requested template exactly (as carried by
    # the block products: off-diagonals times exp(-amplitude_noise**2))
    assert np.array_equal(
        truth["corrs"][~clipped], _carried(truth["requested"][~clipped], 6)
    )
    assert not np.allclose(truth["corrs"][clipped], truth["requested"][clipped])


def test_truth_equals_request_when_no_clipping_is_needed():
    data, truth = tc.simulate_data(
        datagen="constant", return_corrs=True, set_random_seed=1, T=20, K=6, order=2
    )
    assert not _needs_clipping(truth["requested"]).any()
    # the block products carry the request with off-diagonals times exp(-0.2**2)
    assert np.array_equal(truth["corrs"], _carried(truth["requested"], 3))


def test_multi_order_truth_is_achieved_correlation():
    data, truths = tc.simulate_multi_order(
        [1, 2], p=10, set_random_seed=21, T=40000, datagen="constant"
    )
    assert np.array_equal(truths[0]["corrs"], truths[0]["requested"])
    t = truths[1]
    assert _needs_clipping(t["requested"][:1]).all()
    A = np.prod(np.sign(data[:, t["blocks"]]), axis=2)
    # same standard-error argument as the p=10 test above
    assert np.abs(np.corrcoef(A.T) - vec2mat(_latent(t["corrs"][0], 10))).max() <= 0.03


@pytest.mark.parametrize("datagen", DATAGENS)
@pytest.mark.parametrize("B", [0, -2, 2.5])
def test_bad_B_raises_for_every_datagen(datagen, B):
    with pytest.raises(ValueError, match="B must be a positive integer"):
        tc.simulate_data(datagen=datagen, B=B, K=4)
    with pytest.raises(ValueError, match="B must be a positive integer"):
        tc.simulate_multi_order([2], p=3, datagen=datagen, B=B)


# ------------------------------------------------------------ nested template


def test_nested_template_matches_empirical_cofluctuation_correlations():
    data, corrs = tc.simulate_data(
        datagen="constant", return_corrs=True, set_random_seed=5, T=20000, K=4
    )
    # T=20000: the three T x T kernel arrays of the analysis step alone take
    # about 9.6 GB, which the memory guard counts (default limit 4 GB)
    _, D = legacy_features(data, order=1, rfun=None, return_delta=True,
                           max_memory_gb=12.0)
    empirical = np.corrcoef(D[1].T)
    truth = tc.nested_template(corrs[0])
    assert truth.shape == (55,)
    assert np.abs(vec2mat(truth) - empirical).max() <= 0.05
    stacked = tc.nested_template(corrs[:3])
    assert stacked.shape == (3, 55)
    assert np.allclose(stacked[0], truth)


def test_nested_template_identity_and_bad_input():
    out = vec2mat(tc.nested_template(mat2vec(np.eye(3))))
    # independent unit Gaussians: squares have variance 2, everything uncorrelated
    assert np.allclose(out, np.eye(6))
    with pytest.raises(ValueError):
        tc.nested_template(np.ones(4))



# ----------------------------------------------------- carriers and reducer


def _direct_carriers(data, blocks, level):
    """Order-`level` carriers per block, computed directly from the data."""
    cur = [data[:, b] for b in blocks]
    for _ in range(level):
        cur = [
            np.stack(
                [
                    _zscore(c[:, [2 * q]])[:, 0] * _zscore(c[:, [2 * q + 1]])[:, 0]
                    for q in range(c.shape[1] // 2)
                ],
                axis=1,
            )
            for c in cur
        ]
    return cur


def _order3_data(T=200, amplitude_noise=0.2, seed=6):
    with pytest.warns(UserWarning, match="carrier_reducer"):
        return tc.simulate_data(
            datagen="block",
            return_corrs=True,
            set_random_seed=seed,
            T=T,
            K=12,
            order=3,
            amplitude_noise=amplitude_noise,
        )


def test_carrier_columns_select_block_products_in_unreduced_run():
    # T=2000 keeps the sample means of the +/-1 columns (which z-scoring
    # removes) small enough for the raw-product check at the end
    data, truth = _order3_data(T=2000, amplitude_noise=0)
    blocks = truth["blocks"]
    K = data.shape[1]
    _, D = legacy_features(data, order=2, rfun=None, return_delta=True)
    for level, m in [(0, 4), (1, 2), (2, 1)]:
        cols = tc.carrier_columns(blocks, level, K)
        assert cols.shape == (3, m)
        direct = _direct_carriers(data, blocks, level)
        for j in range(3):
            # exact: pipeline columns equal products of z-scored lower carriers
            assert np.allclose(D[level][:, cols[j]], direct[j], atol=1e-10, rtol=0)
    # the order-2 carrier of each block tracks the raw product of the block
    top = tc.carrier_columns(blocks, 2, K)[:, 0]
    for j, b in enumerate(blocks):
        raw = np.prod(data[:, b], axis=1)
        assert np.corrcoef(D[2][:, top[j]], raw)[0, 1] > 0.99


def test_carrier_columns_level0_is_blocks_and_bad_level_raises():
    blocks = np.arange(8).reshape(2, 4) + 3
    assert np.array_equal(tc.carrier_columns(blocks, 0, 11), blocks)
    # pair (3, 4) among 11 columns: 11 diagonal entries, then squareform index 0
    assert tc.carrier_columns(blocks, 1, 11)[0, 0] == 11 + 3 * 11 - 3 * 4 // 2
    with pytest.raises(ValueError, match="level"):
        tc.carrier_columns(blocks, 3, 11)
    with pytest.raises(ValueError, match="K"):
        tc.carrier_columns(blocks, 1, 10)


def test_carrier_reducer_matches_unreduced_carriers():
    data, truth = _order3_data()
    blocks = truth["blocks"]
    K = data.shape[1]
    _, D_full = legacy_features(data, order=2, rfun=None, return_delta=True)
    X, D = legacy_features(
        data,
        order=3,
        cfun=tc.autofc,
        rfun=None,
        delta_rfun=tc.carrier_reducer(blocks),
        return_delta=True,
        weights_params={"var": 100},
    )
    assert [d.shape[1] for d in D] == [12, 6, 3, 6]
    assert X[3].shape == (200, 6)  # vectorized 3 x 3 matrix among block products
    assert np.allclose(
        D[1], D_full[1][:, tc.carrier_columns(blocks, 1, K).ravel()], atol=1e-12
    )
    direct = _direct_carriers(data, blocks, 2)
    assert np.allclose(D[2], np.hstack(direct), atol=1e-10, rtol=0)


def test_carrier_reducer_is_stateless_and_rejects_unknown_widths():
    data, truth = _order3_data()
    f = tc.carrier_reducer(truth["blocks"])
    _, D = legacy_features(data, order=1, rfun=None, return_delta=True)
    first = f([D[1]])
    again = f([D[1], D[1]])
    assert len(first) == 1 and len(again) == 2
    assert np.array_equal(first[0], again[0]) and np.array_equal(again[0], again[1])
    with pytest.raises(ValueError, match="78") as err:
        f([np.zeros((10, 7))])
    assert "21" in str(err.value) and "6" in str(err.value)


def test_carrier_reducer_with_extra_columns():
    blocks = np.arange(4).reshape(2, 2) + 2  # data columns 2..5 of 7
    f = tc.carrier_reducer(blocks, K=7)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 7))
    _, D = legacy_features(x, order=1, rfun=None, return_delta=True)
    out = f([D[1]])[0]
    z = _zscore(x)
    assert np.allclose(out, np.stack([z[:, 2] * z[:, 3], z[:, 4] * z[:, 5]], 1))
    with pytest.raises(ValueError, match="K"):
        tc.carrier_reducer(blocks, K=5)


def test_order2_end_to_end_recovery():
    data, truth = tc.simulate_data(
        datagen="block",
        return_corrs=True,
        set_random_seed=0,
        T=1000,
        K=8,
        B=4,
        order=2,
    )
    blocks = truth["blocks"]
    X = legacy_features(
        data,
        order=2,
        cfun=tc.autofc,
        rfun=None,
        delta_rfun=tc.carrier_reducer(blocks),
        weights_function=tc.gaussian_weights,
        weights_params={"var": 625},
    )
    est = X[2]
    assert est.shape == (1000, 10)
    off = _offdiag(4)
    r = np.corrcoef(est[:, off].ravel(), truth["corrs"][:, off].ravel())[0, 1]
    assert r >= 0.8


# ------------------------------------------------------------- multi-order


def test_simulate_multi_order_layout():
    with pytest.warns(UserWarning, match="carrier_reducer") as record:
        data, truths = tc.simulate_multi_order(
            [1, 2, 3], p=3, set_random_seed=9, T=3000, datagen="constant", S=2
        )
    assert len(record) == 1
    assert len(data) == 2 and data[0].shape == (3000, 3 + 6 + 12)
    assert [t["order"] for t in truths] == [1, 2, 3]
    assert np.array_equal(truths[0]["blocks"], np.arange(3)[:, None])
    assert np.array_equal(truths[1]["blocks"], 3 + np.arange(6).reshape(3, 2))
    assert np.array_equal(truths[2]["blocks"], 9 + np.arange(12).reshape(3, 4))
    for t in truths:
        assert t["corrs"].shape == (3000, 6)
    # the order-1 group is Gaussian with the order-1 template
    emp = np.corrcoef(data[0][:, :3].T)
    assert np.abs(emp - vec2mat(truths[0]["corrs"][0])).max() < 0.1
    # each higher group's block products carry its template
    for t in truths[1:]:
        A = np.prod(np.sign(data[0][:, t["blocks"]]), axis=2)
        assert np.abs(np.corrcoef(A.T) - vec2mat(t["corrs"][0])).max() < 0.1
    # the reducer finds the order-2 group's carriers inside the wider data
    K = data[0].shape[1]
    f = tc.carrier_reducer(truths[1]["blocks"], K=K)
    _, D = legacy_features(data[0], order=1, rfun=None, return_delta=True)
    z = _zscore(data[0])
    b = truths[1]["blocks"]
    assert np.allclose(f([D[1]])[0], z[:, b[:, 0]] * z[:, b[:, 1]])


def test_simulate_multi_order_bad_input():
    with pytest.raises(ValueError, match="orders"):
        tc.simulate_multi_order([])
    with pytest.raises(TypeError):
        tc.simulate_multi_order([2], K=8)


# ----------------------------------------------------------------- docstring


def _returns_names(doc):
    # cleandoc gives the same text on every Python version (3.13 strips
    # docstring indentation at compile time; earlier versions keep it).
    doc = inspect.cleandoc(doc)
    section = doc.split("Returns\n-------\n")[1].split("\nRaises\n")[0]
    return re.findall(r"^(\w+) : ", section, flags=re.M)


def test_simulate_data_docstring_returns_matches_output():
    names = _returns_names(tc.simulate_data.__doc__)
    assert names == ["data", "corrs", "truth"]
    assert "sub_locs" not in tc.simulate_data.__doc__
    assert "electrode" not in tc.simulate_data.__doc__
    data = tc.simulate_data(T=10, K=4, set_random_seed=1)
    assert isinstance(data, np.ndarray)
    out = tc.simulate_data(T=10, K=4, set_random_seed=1, return_corrs=True)
    assert len(out) == 2


# ----------------------------------------------- truth with amplitude noise


def _pearson_with_se(x, y):
    """Pearson r and its large-sample standard error for non-normal data
    (delta method: SE = sd(u) / sqrt(T) with u = zx zy - r (zx**2 + zy**2) / 2)."""
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    r = np.mean(zx * zy)
    u = zx * zy - r * (zx**2 + zy**2) / 2
    return r, u.std() / np.sqrt(len(x))


def test_truth_scaled_by_amplitude_noise_matches_block_products():
    # Each block product is A_j * exp(a * z_j) with independent z_j ~ N(0, 1), so
    # corr(P_i, P_j) = corr(A_i, A_j) * E[e^{az}]**2 / E[e^{2az}] = corr(A) * exp(-a**2).
    a, T = 0.5, 200000
    data, truth = tc.simulate_data(
        datagen="constant", return_corrs=True, set_random_seed=4, T=T, K=8,
        order=2, amplitude_noise=a,
    )
    p = 4
    assert not _needs_clipping(truth["requested"][:1]).any()
    P = np.prod(data[:, truth["blocks"]], axis=2)  # (T, p) block products
    carried = vec2mat(truth["corrs"][0])
    unscaled = vec2mat(truth["requested"][0])  # the latent-sign correlation here
    assert np.allclose(np.diag(carried), 1.0)
    off = [(i, j) for i in range(p) for j in range(i + 1, p)]
    assert np.allclose([carried[i, j] for i, j in off],
                       [np.exp(-a**2) * unscaled[i, j] for i, j in off],
                       rtol=1e-15, atol=0)
    for i, j in off:
        r, se = _pearson_with_se(P[:, i], P[:, j])
        assert se < 0.01
        # within 4 standard errors of the reported truth ...
        assert abs(r - carried[i, j]) <= 4 * se, (i, j, r, carried[i, j], se)
        # ... and, where the template is not near 0, far from the unscaled value
        if abs(unscaled[i, j]) > 0.2:
            assert abs(r - unscaled[i, j]) > 8 * se, (i, j, r, unscaled[i, j], se)
    # the signs of the products still carry the unscaled correlation
    A = np.sign(P)
    assert np.abs(np.corrcoef(A.T) - unscaled).max() <= 4 / np.sqrt(T) * 4


def test_truth_unscaled_without_amplitude_noise():
    data, truth = tc.simulate_data(
        datagen="constant", return_corrs=True, set_random_seed=1, T=20, K=6,
        order=2, amplitude_noise=0.0,
    )
    assert np.array_equal(truth["corrs"], truth["requested"])


# ------------------------------------------ shared signal across subjects (R-K)


def test_subject_noise_gives_subjects_a_shared_signal():
    # One draw from the ramping template is shared by all subjects; each adds
    # Gaussian noise with SD 0.5 x each feature's SD, so the cross-subject
    # correlation of a feature is 1 / (1 + 0.5**2) = 0.8.  Each pair's
    # estimate has a (delta-method) standard error se; the mean over pairs
    # has a standard error of at most the largest se.
    S, T, K = 8, 2000, 6
    data = tc.simulate_data(datagen="ramping", S=S, T=T, K=K, set_random_seed=3,
                            subject_noise=0.5)
    for i in range(K):
        stats = [_pearson_with_se(data[0][:, i], data[s][:, i]) for s in range(1, S)]
        r = np.mean([x[0] for x in stats])
        se = max(x[1] for x in stats)
        assert abs(r - 0.8) < 4 * se, (i, r, se)


def test_subject_noise_none_keeps_independent_draws():
    S, T, K = 8, 2000, 6
    data = tc.simulate_data(datagen="ramping", S=S, T=T, K=K, set_random_seed=3)
    same = tc.simulate_data(datagen="ramping", S=S, T=T, K=K, set_random_seed=3,
                            subject_noise=None)
    for a, b in zip(data, same):
        np.testing.assert_array_equal(a, b)
    # independent draws: cross-subject correlations are N(0, 1/T)
    for i in range(K):
        r = np.mean([np.corrcoef(data[0][:, i], data[s][:, i])[0, 1] for s in range(1, S)])
        assert abs(r) < 4 / np.sqrt(T), (i, r)


def test_subject_noise_zero_gives_identical_subjects():
    data = tc.simulate_data(S=3, T=50, K=4, set_random_seed=1, subject_noise=0.0)
    np.testing.assert_array_equal(data[0], data[1])
    np.testing.assert_array_equal(data[0], data[2])


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf, "0.5"])
def test_subject_noise_must_be_a_non_negative_number(bad):
    with pytest.raises(ValueError, match="subject_noise"):
        tc.simulate_data(S=2, T=20, K=4, subject_noise=bad)


def test_subject_noise_is_for_order_one():
    with pytest.raises(ValueError, match="subject_noise"):
        tc.simulate_data(S=2, T=20, K=8, order=2, subject_noise=0.5)


def test_simulate_k1_message_names_the_order1_requirement():
    # R-J (M13): the order-1 message no longer quotes the order > 1 formula
    with pytest.raises(ValueError, match="min_features") as info:
        tc.simulate_data(K=1)
    assert "K >= 2" in str(info.value)
    assert "2**(order-1)" not in str(info.value)


def test_carrier_columns_k_defaults_like_carrier_reducer():
    blocks = [[0, 1], [2, 3]]
    np.testing.assert_array_equal(tc.carrier_columns(blocks, 1),
                                  tc.carrier_columns(blocks, 1, 4))


# ------------------------------------------------------------ simulate_nested


def _mean_abs_corr(x, module):
    c = np.abs(np.corrcoef(x.T))
    same = module[:, None] == module[None, :]
    off = ~np.eye(len(module), dtype=bool)
    return c[same & off].mean(), c[~same].mean()


def test_nested_shapes_and_truth_keys():
    data, truth = tc.simulate_nested(S=3, T=200, set_random_seed=1)
    assert len(data) == 3
    assert all(d.shape == (200, 108) and d.dtype == float for d in data)
    assert set(truth) == {"module", "supergroup", "h", "g"}
    assert truth["h"].shape == (200, 6)
    assert len(truth["g"]) == 3
    assert all(g.shape == (200, 18) for g in truth["g"])
    assert truth["module"].shape == (108,)
    np.testing.assert_array_equal(np.bincount(truth["module"]), np.full(18, 6))
    np.testing.assert_array_equal(truth["supergroup"], truth["module"] // 3)


def test_nested_custom_sizes():
    data, truth = tc.simulate_nested(
        S=1, T=100, n_supergroups=2, modules_per_supergroup=2, nodes_per_module=3,
        set_random_seed=0,
    )
    assert data[0].shape == (100, 12)
    assert truth["h"].shape == (100, 2)
    assert truth["g"][0].shape == (100, 4)


@pytest.mark.parametrize("coupling", ["sustained", "signflip", "transient"])
def test_nested_seed_reproducible_and_global_state_untouched(coupling):
    np.random.seed(7)
    before = np.random.get_state()
    a, ta = tc.simulate_nested(T=150, coupling=coupling, set_random_seed=5)
    b, tb = tc.simulate_nested(T=150, coupling=coupling, set_random_seed=5)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(ta["h"], tb["h"])
    c, _ = tc.simulate_nested(T=150, coupling=coupling, set_random_seed=6)
    assert not np.array_equal(a[0], c[0])


def test_nested_ranges():
    _, truth = tc.simulate_nested(T=400, set_random_seed=2)
    assert truth["h"].min() >= 0.05 and truth["h"].max() <= 0.95
    for g in truth["g"]:
        assert g.min() >= 0 and g.max() <= 1


def test_nested_shared_order2():
    data, truth = tc.simulate_nested(T=300, shared="order2", set_random_seed=3)
    assert not np.array_equal(truth["g"][0], truth["g"][1])
    assert not np.array_equal(data[0], data[1])


def test_nested_shared_order1():
    data, truth = tc.simulate_nested(T=300, shared="order1", set_random_seed=3)
    np.testing.assert_array_equal(truth["g"][0], truth["g"][1])
    assert not np.array_equal(data[0], data[1])


def test_nested_sustained_coupling_within_module():
    data, truth = tc.simulate_nested(S=1, T=4000, set_random_seed=4)
    within, across = _mean_abs_corr(data[0], truth["module"])
    assert within > 0.2
    assert across < 0.05


def test_nested_signflip_has_no_whole_recording_structure():
    data, truth = tc.simulate_nested(S=1, T=4000, coupling="signflip", set_random_seed=4)
    within, across = _mean_abs_corr(data[0], truth["module"])
    assert within < 0.08
    assert across < 0.05


def test_nested_transient_is_mostly_off():
    _, truth = tc.simulate_nested(T=1000, coupling="transient", set_random_seed=4)
    assert np.mean([g.mean() for g in truth["g"]]) < 0.25


def test_nested_global_signal_adds_across_module_correlation():
    data, truth = tc.simulate_nested(S=1, T=4000, global_signal=0.5, set_random_seed=4)
    _, across = _mean_abs_corr(data[0], truth["module"])
    assert across > 0.1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"coupling": "bogus"}, "coupling must be 'sustained', 'transient' or 'signflip'"),
        ({"shared": "bogus"}, "shared must be 'order2' or 'order1'"),
        ({"S": 0}, "S must be a positive integer"),
        ({"T": 0}, "T must be a positive integer"),
        ({"n_supergroups": 0}, "n_supergroups must be a positive integer"),
        ({"modules_per_supergroup": -1}, "modules_per_supergroup must be a positive integer"),
        ({"nodes_per_module": 0}, "nodes_per_module must be a positive integer"),
        ({"tau2": 0}, "tau2 must be a positive integer"),
        ({"flip": 0}, "flip must be a positive integer"),
        ({"duty": 0.0}, "duty must be in the open interval"),
        ({"duty": 1.0}, "duty must be in the open interval"),
        ({"T": 39, "tau2": 5}, r"T must be at least 8 \* max\(tau2, tau3\) = 40"),
        ({"T": 100, "tau3": 20}, r"T must be at least 8 \* max\(tau2, tau3\) = 160"),
        ({"global_signal": "x"}, "global_signal must be a number"),
    ],
)
def test_nested_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        tc.simulate_nested(**{"set_random_seed": 0, **kwargs})


def _pair_corr(g, supergroup_of_module, rows):
    """Mean over supergroups and within-supergroup module pairs of corr(g_m, g_m')
    over the timepoints in ``rows`` (a boolean (T, n_supergroups) mask)."""
    out = []
    for s in range(rows.shape[1]):
        mods = np.where(supergroup_of_module == s)[0]
        for a, b in itertools.combinations(mods, 2):
            out.append(np.corrcoef(g[rows[:, s], a], g[rows[:, s], b])[0, 1])
    return np.mean(out)


def test_nested_order2_coupling_between_modules_follows_h():
    # T=6000, seeds 0-2, measured: top-third mean pair corr 0.85-0.87, bottom
    # third 0.08-0.11; mean |corr| of g across supergroups 0.036-0.041.
    data, truth = tc.simulate_nested(S=1, T=6000, set_random_seed=0)
    g, h = truth["g"][0], truth["h"]
    sg = np.arange(g.shape[1]) // 3
    lo, hi = np.quantile(h, [1 / 3, 2 / 3], axis=0)
    top = _pair_corr(g, sg, h >= hi)
    bottom = _pair_corr(g, sg, h <= lo)
    assert top > 0.6
    assert bottom < 0.3
    assert top - bottom > 0.4
    c = np.corrcoef(g.T)
    assert np.abs(c[sg[:, None] != sg[None, :]]).mean() < 0.08


def test_nested_shared_order2_g_is_private_though_h_is_shared():
    # measured mean |corr| of g between subjects, per module: 0.029-0.037
    _, truth = tc.simulate_nested(S=2, T=6000, shared="order2", set_random_seed=0)
    c = [abs(np.corrcoef(truth["g"][0][:, m], truth["g"][1][:, m])[0, 1]) for m in range(18)]
    assert np.mean(c) < 0.08
    _, truth = tc.simulate_nested(S=2, T=300, shared="order1", set_random_seed=0)
    np.testing.assert_array_equal(truth["g"][0], truth["g"][1])


def _windowed_module_corr(x, module, width):
    same = module[:, None] == module[None, :]
    off = ~np.eye(len(module), dtype=bool)
    w = []
    for i in range(0, len(x) - width + 1, width):
        c = np.abs(np.corrcoef(x[i : i + width].T))
        w.append((c[same & off].mean(), c[~same].mean()))
    return np.mean(w, axis=0)


def test_nested_signflip_structure_is_local_in_time():
    # T=4000, 20-sample windows, measured (within, across): signflip
    # (0.46, 0.19), sustained (0.51, 0.19); the across value is the noise floor
    # of a 20-sample correlation.
    data, truth = tc.simulate_nested(S=1, T=4000, coupling="signflip", flip=40, set_random_seed=0)
    within, across = _windowed_module_corr(data[0], truth["module"], 20)
    assert within > across + 0.15
    assert within > 0.3


def test_nested_signflip_with_flip_beyond_T_is_sustained():
    # measured whole-recording within-module mean |corr|: 0.495-0.505
    data, truth = tc.simulate_nested(
        S=1, T=4000, coupling="signflip", flip=10**6, set_random_seed=0
    )
    within, across = _mean_abs_corr(data[0], truth["module"])
    assert within > 0.2
    assert across < 0.05


@pytest.mark.parametrize("duty", [0.1, 0.3])
def test_nested_transient_fraction_on_matches_duty(duty):
    # T=6000, 2 subjects, 3 seeds, measured fraction of g > 0.5: 0.098-0.100
    # (duty 0.1) and 0.299-0.303 (duty 0.3); tolerance 0.03.
    _, truth = tc.simulate_nested(
        S=2, T=6000, coupling="transient", duty=duty, set_random_seed=0
    )
    frac = np.mean([(g > 0.5).mean() for g in truth["g"]])
    assert abs(frac - duty) < 0.03
