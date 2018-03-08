from past.utils import old_div
import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

#using method from supereeg
from timecorr.helpers import gaussian_weights, wcorr, weighted_mean_var_diffs, \
     wisfc, isfc, smooth, timepoint_decoder


gaussian_params = {'var': 1000}

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
        return mx, varx, diffs
    second_tester = test_wcorr()
    assert isinstance()

def test_wisfc():
    
    assert isinstance()

def test_isfc():
    assert isinstance()

def test_smooth():
    assert isinstance()

def test_timepoint_decoder():
    assert isinstance()
