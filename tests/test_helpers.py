from past.utils import old_div
import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

#using method from supereeg
from timecorr.helpers import gaussian_weights, wcorr, wisfc, isfc, smooth, timepoint_decoder


gaussian_params = {'var': 1000}
data_numpy= np.random.randn(10,3)
""" :param a: a number-of-timepoints by number-of-features observations matrix
    :param b: a number-of-timepoints by number-of-features observations matrix
    :param weights: a number-of-timepoints by number-of-timepoints weights matrix
    specifying the per-timepoint weights to be considered (for each timepoint) """
param_a= np.array([[2, 6], [8, 12]])
param_b= np.array([[5, 9], [10, 7]])
param_weights= np.random.randn(2,2)
zero_weights= np.array([[0, 0], [0, 0]])
#window size = 0
#mu = 0
#nfolds = 2
#connectivity_fun= test_isfc

### function:
def gaussian_weights(T, params=gaussian_params):
    """
    gaussian_weights takes the input, T, a non-negative integer
    specifying the number of timepoints to consider.

    Function should return a T by T array
    """
    c1 = np.divide(1, np.sqrt(2 * np.math.pi * params['var']))
    c2 = np.divide(-1, 2 * params['var'])
    sqdiffs = toeplitz(np.arange(T)) ** 2
    return c1 * np.exp(c2 * sqdiffs)

### test:

def test_gaussian_weights():
    first_tester = gaussian_weights(T, params=gaussian_params)
    assert isinstance(first_tester, np.array)


def test_wcorr():
    # do I need to test weighted_mean_var_diffs as well? or is placing it here okay?
    def weighted_mean_var_diffs(x, weights):
        # goal to check if weighted_zeros is the same as the if statement
        weights[np.isnan(zero_weights)] = 0
        if np.sum(weights) == 0:
            weights = np.ones(x.shape)
        assert isinstance(weights> zero_weights)
        
    second_tester = test_wcorr()
    assert isinstance(test_autocorrelation == autocorrelation)
    assert isinstance()
    assert isinstance()


def test_wisfc_type_not_list():
    data= data_numpy
    wisfc_test_1= wisfc(data)
    wisfc_test_2 = 2*data
    assert isinstance(wisfc_test_2>wisfc_test_1)

def test_isfc():
    testing_np.ones= np.ones[1, len(data_numpy)]
    assert isinstance(test_isfc = testing_np.ones)

#working from here up

def test_smooth():
    smooth_tester= smooth()
    assert isinstance(smooth_tester, np.array)

def test_timepoint_decoder():
    test5= timepoint_decoder(data, windowsize=0, mu=0, nfolds=2, connectivity_fun=isfc)

    assert isinstance()
