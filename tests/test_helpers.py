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

    """
    c1 = np.divide(1, np.sqrt(2 * np.math.pi * params['var']))
    c2 = np.divide(-1, 2 * params['var'])
    sqdiffs = toeplitz(np.arange(T)) ** 2
    return c1 * np.exp(c2 * sqdiffs)




### test:
def test_gaussian_weights():
    assert
