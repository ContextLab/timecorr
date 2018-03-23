from past.utils import old_div
import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

#using method from supereeg
from timecorr.helpers import gaussian_weights, wcorr, wisfc, isfc, smooth, timepoint_decoder


gaussian_params = {'var': 1000}
data_numpy= np.random.randn(10,3)
param_a= np.array([[2, 6], [8, 12]])
param_b= np.array([[5, 9], [10, 7]])
param_weights= np.random.randn(2,2)
zero_weights= np.array([[0, 0], [0, 0]])

def test_gaussian_weights():
    test_gw = gaussian_weights(T, params=gaussian_params)
    assert isinstance(test_gw, np.array)


def test_wcorr():
    # do I need to test weighted_mean_var_diffs as well? or is placing it here okay?
    def weighted_mean_var_diffs(x, weights):
        weights[np.isnan(zero_weights)] = 0
        if np.sum(weights) == 0:
            weights = np.ones(x.shape)
        assert isinstance(weights> zero_weights)
    second_tester = test_wcorr()
    assert isinstance(test_autocorrelation == autocorrelation)

def test_wisfc():
    data= data_numpy
    wisfc_test_1= wisfc(data)
    wisfc_test_2 = 2*data
    assert isinstance(wisfc_test_2>wisfc_test_1)

def test_isfc():
    testing_np.ones= np.ones[1, len(data_numpy)]
    assert isinstance(test_isfc = testing_np.ones)

def test_smooth():
    smooth_tester= smooth()
    assert isinstance(smooth_tester, np.array)
