import hypertools as hyp

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
import numpy as np
import pandas as pd

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
    dyna_corrs_reduced_3 = timecorr(
        sim_3,
        rfun="PCA",
        weights_function=laplace["weights"],
        weights_params=laplace["params"],
        include_timepoints="pre",
    )

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_post():
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

corrs = timecorr(
    numpy_array,
    weights_function=gaussian_weights,
    weights_params=gaussian_params,
    cfun=isfc,
)
# assert()
# assert len(corrs.get_time_data()[0]) == len(numpy_array)
