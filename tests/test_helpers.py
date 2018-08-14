from past.utils import old_div
import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

#using method from supereeg
from timecorr.helpers import gaussian_weights, gaussian_params, wcorr, wisfc, isfc, smooth, timepoint_decoder, predict

T = 10
D = 4
S = 5

n_elecs = 20

n_samples = 1000

R = toeplitz(np.linspace(0, 1, n_elecs)[::-1])

data_sim = np.random.multivariate_normal(np.zeros(n_elecs), R, size=n_samples)

gps = {'var': 10}

T_sim = data_sim.shape[0]

weights_sim = gaussian_weights(T_sim, gps)

template_data = np.cumsum(np.random.randn(T, D), axis=0)
data = []

for s in np.arange(S):
    data.append(template_data + np.multiply(0.1, np.random.randn(template_data.shape[0], template_data.shape[1])))

def test_gaussian_weights():
    test_gw = gaussian_weights(T, params=gaussian_params)

    assert isinstance(test_gw, np.ndarray)

def test_wcorr():
    weights = gaussian_weights(T, params=gaussian_params)
    corrs = wcorr(template_data[:, 0][:, np.newaxis], template_data[:, 1][:, np.newaxis], weights)

    col_1 = np.atleast_2d(data_sim[:, 0]).T

    col_2 = np.atleast_2d(data_sim[:, 1]).T

    corrs_col_arrays = np.squeeze(wcorr(data_sim, data_sim, weights_sim))

    corrs_multidim = wcorr(col_1, data_sim, weights_sim)

    corrs_col = np.squeeze(wcorr(col_1, col_1, weights_sim))

    corrs_col_neg = np.squeeze(wcorr(-col_1, col_1, weights_sim))

    corrs_col_12 = np.squeeze(wcorr(col_2, col_1, weights_sim))

    assert (np.allclose(corrs_col, corrs_multidim[0][0]))
    assert (np.allclose(corrs_col_12, corrs_multidim[0][1]))
    assert (np.allclose(-corrs_col, corrs_col_neg))
    assert (np.isclose(corrs_col_arrays[1,1,50],1))
    assert isinstance(corrs, np.ndarray)

def test_wisfc():
    weights = gaussian_weights(T, params=gaussian_params)
    w_list = wisfc(data, weights)
    assert isinstance(w_list, np.ndarray)

    w_array = wisfc(template_data, weights)
    assert isinstance(w_array, np.ndarray)

def test_isfc():
    test_wisfc()

#commenting out: smoothing not implemented #TODO: implement smooth function
#def test_smooth():
#   smooth_tester= smooth()
#   assert isinstance(smooth_tester, np.array)
