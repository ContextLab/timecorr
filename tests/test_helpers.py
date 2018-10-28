import numpy as np
from scipy.linalg import toeplitz

#using method from supereeg
from timecorr.helpers import gaussian_weights, gaussian_params, wcorr, wisfc, mat2vec, vec2mat

T = 10
D = 4
S = 5

n_elecs = 20

n_samples = 1000

R = toeplitz(np.linspace(0, 1, n_elecs)[::-1])

data_sim = np.random.multivariate_normal(np.zeros(n_elecs), R, size=n_samples)

gps = {'var': 100}

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

    # correlate one column with itself and the same column in larger array is the same
    assert (np.allclose(corrs_col, corrs_multidim[0][0]))
    # correlating a timeseries with -1 times itself produces -1's
    assert corrs_col.mean() == 1
    # correlate one column with the second column and the same column in larger array is the same
    assert (np.allclose(corrs_col_12, corrs_multidim[0][1]))
    # correlating a timeseries with -1 times itself produces negative correlations
    assert (np.allclose(-corrs_col, corrs_col_neg))
    # correlating a timeseries with -1 times itself produces -1's
    assert corrs_col_neg.mean()==-1
    # check if corresponding columns in 3d array produces 1
    assert (np.isclose(corrs_col_arrays[4,4,500],1))
    # check if toeplitz matrix is produced
    assert (np.allclose(corrs_col_arrays[:, :, 500], R, atol=.1))
    # check if corrs is a numpy array
    assert isinstance(corrs, np.ndarray)

def test_wisfc():
    weights = gaussian_weights(T, params=gaussian_params)
    w_list = wisfc(data, weights)
    assert isinstance(w_list, list)

    w_array = wisfc(template_data, weights)
    assert isinstance(w_array, np.ndarray)

def test_isfc():
    test_wisfc()


def test_mat2vec_vec2mat():

    corrs = np.squeeze(wcorr(data_sim, data_sim, weights_sim))

    ## check for 3d
    V = mat2vec(corrs)

    M = vec2mat(V)

    ## check for 2d
    v = mat2vec(corrs[:, :, 50])

    m = vec2mat(v)

    assert (np.allclose(M, corrs))
    assert (np.allclose(m, corrs[:, :, 50]))

#commenting out: smoothing not implemented #TODO: implement smooth function
#def test_smooth():
#   smooth_tester= smooth()
#   assert isinstance(smooth_tester, np.array)
