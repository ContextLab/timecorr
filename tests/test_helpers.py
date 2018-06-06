from past.utils import old_div
import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

#using method from supereeg
from timecorr.helpers import gaussian_weights, gaussian_params, wcorr, wisfc, isfc, smooth, timepoint_decoder, predict

T = 10
D = 4
S = 5

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
