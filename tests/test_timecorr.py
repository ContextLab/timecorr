import hypertools as hyp

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend

import timecorr as tc
from timecorr.helpers import gaussian_params, gaussian_weights, isfc
from timecorr.simulate import simulate_data
from timecorr.timecorr import timecorr

# TODO: need *real* tests-- e.g. generate a small dataset and verify that we actually get the correct answers

# gaussian_params = {'var': 1000}
data_list = np.random.randn(10, 3)
pandas_dataframe = pd.DataFrame(np.random.randint(low=0, high=10, size=(2, 2)))
numpy_array = np.array([[5, 9], [10, 7]])
numpy_array_list = np.array([[8, 2], [4, 6]]).tolist()
# if above is how to make a numpy list than TC isn't capible np.lists currently
random_numbers = (2, 3, 5, 10, 12, 4, 6)

sim_1 = simulate_data(S=1, T=30, K=50, set_random_seed=100)
sim_3 = simulate_data(S=3, T=30, K=50, set_random_seed=100)

width = 10
laplace = {"name": "Laplace", "weights": tc.laplace_weights, "params": {"scale": width}}


def test_reduce_shape():
    # one subject, T=30 samples, K=50 columns requested: 20 are zero padding,
    # which is reported (Ruling R-U)
    with pytest.warns(UserWarning, match="padded with 20 zero columns"):
        dyna_corrs_reduced_1 = timecorr(
            sim_1,
            rfun="PCA",
            weights_function=laplace["weights"],
            weights_params=laplace["params"],
        )

    dyna_corrs_reduced_3 = timecorr(
        sim_3,
        rfun="PCA",
        weights_function=laplace["weights"],
        weights_params=laplace["params"],
    )
    assert np.shape(dyna_corrs_reduced_1) == np.shape(sim_1)
    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_nans():
    sim_3[0][0] = np.nan
    dyna_corrs_reduced_3 = timecorr(
        sim_3,
        rfun="PCA",
        weights_function=laplace["weights"],
        weights_params=laplace["params"],
    )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_all():
    dyna_corrs_reduced_3 = timecorr(
        sim_3,
        rfun="PCA",
        weights_function=laplace["weights"],
        weights_params=laplace["params"],
        include_timepoints="all",
    )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_pre():
    with pytest.warns(UserWarning, match="fewer than 3 non-zero kernel weights"):
        dyna_corrs_reduced_3 = timecorr(
            sim_3,
            rfun="PCA",
            weights_function=laplace["weights"],
            weights_params=laplace["params"],
            include_timepoints="pre",
        )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_post():
    with pytest.warns(UserWarning, match="fewer than 3 non-zero kernel weights"):
        dyna_corrs_reduced_3 = timecorr(
            sim_3,
            rfun="PCA",
            weights_function=laplace["weights"],
            weights_params=laplace["params"],
            include_timepoints="post",
        )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_exclude_timepoints_pos():
    dyna_corrs_reduced_3 = timecorr(
        sim_3,
        rfun="PCA",
        weights_function=laplace["weights"],
        weights_params=laplace["params"],
        exclude_timepoints=3,
    )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_exclude_timepoints_neg():
    dyna_corrs_reduced_3 = timecorr(
        sim_3,
        rfun="PCA",
        weights_function=laplace["weights"],
        weights_params=laplace["params"],
        exclude_timepoints=-3,
    )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_timecorr():

    data_dl = hyp.tools.format_data(data_list)
    data_pdf = hyp.tools.format_data(pandas_dataframe)
    data_npa = hyp.tools.format_data(numpy_array)
    #   data_npl = hyp.tools.format_data(numpy_array_list)
    #   data_rand = hyp.tools.format_data(random_numbers)
    #   these are now lists
    assert isinstance(data_dl, list)

    Test_dl = data_dl[0].shape[0]
    Test_pdf = data_pdf[0].shape[0]
    Test_npa = data_npa[0].shape[0]
    # Test returns the shape of the weights_function
    #   Test_npl=  data_npl[0].shape[0]
    #   Test_rand=  data_rn[0].shape[0]

    assert isinstance(Test_pdf, int)

    dl_tester = gaussian_weights(Test_dl, params=gaussian_params)
    pdf_tester = gaussian_weights(Test_pdf, params=gaussian_params)
    npa_tester = gaussian_weights(Test_npa, params=gaussian_params)
    #   thrid_tester = gaussian_weights(T3, params=gaussian_params)
    #   fourth_tester = gaussian_weights(T4, params=gaussian_params)

    assert isinstance(npa_tester, np.ndarray)


# assert npa_tester.shape == data_npa.shape
# assert dl_tester.shape > data_dl.shape


# unsure how to test 'across' mode

def test_two_timepoint_input_warns_and_returns_zeros():
    # Every timepoint has fewer than 3 non-zero kernel weights, so the
    # correlations are undefined: one warning, and zeros in their place.
    with pytest.warns(UserWarning, match="fewer than 3 non-zero kernel weights"):
        corrs = timecorr(
            numpy_array,
            weights_function=gaussian_weights,
            weights_params=gaussian_params,
            cfun=isfc,
        )
    assert corrs.shape == (2, 3)
    assert np.array_equal(corrs, np.zeros((2, 3)))


# assert len(corrs.get_time_data()[0]) == len(numpy_array)


# ------------------------------------------- smoothing path (cfun=None)


def _impulse(T=40, K=2, at=20):
    x = np.zeros((T, K))
    x[at, :] = 1.0
    return x


def test_smoothing_path_is_kernel_weighted_mean_of_rows():
    """cfun=None gives sum_j |w[t, j]| x[j] / sum_j |w[t, j]| with the full kernel."""
    rng = np.random.RandomState(0)
    x = rng.randn(40, 3)
    w = tc.gaussian_weights(40, {"var": 9})
    out = tc.timecorr(x, cfun=None, weights_function=tc.gaussian_weights, weights_params={"var": 9})
    expected = (w @ x) / w.sum(axis=1, keepdims=True)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected, atol=1e-12, rtol=0)


def test_smoothing_path_respects_include_timepoints():
    x = _impulse()
    back = tc.timecorr(x, cfun=None, weights_params={"var": 9}, include_timepoints="backward")
    fwd = tc.timecorr(x, cfun=None, weights_params={"var": 9}, include_timepoints="forward")
    assert np.allclose(back[:20], 0) and np.all(back[20:] > 0)
    assert np.allclose(fwd[21:], 0) and np.all(fwd[:21] > 0)


def test_smoothing_path_respects_include_zero_and_exclude_timepoints():
    x = _impulse()
    no_zero = tc.timecorr(x, cfun=None, weights_params={"var": 9}, include_zero=False)
    assert np.allclose(no_zero[20], 0) and np.all(no_zero[19] > 0)
    near = tc.timecorr(x, cfun=None, weights_params={"var": 9}, exclude_timepoints=3)
    assert np.allclose(near[:17], 0) and np.allclose(near[24:], 0) and np.all(near[17:24] > 0)


def test_smoothing_path_list_input_returns_arrays():
    """List input with null_combine crashed on list.tolist()."""
    rng = np.random.RandomState(1)
    data = [rng.randn(30, 3) for _ in range(2)]
    out = tc.timecorr(data, cfun=None, weights_params={"var": 9})
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(o, np.ndarray) and o.shape == (30, 3) for o in out)
    single = tc.timecorr(data[0], cfun=None, weights_params={"var": 9})
    assert np.allclose(out[0], single)
