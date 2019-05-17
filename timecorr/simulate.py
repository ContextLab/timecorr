import warnings

import numpy as np
import timecorr as tc

from scipy.spatial.distance import cdist


def random_corrmat(K):
    x = np.random.randn(K, K)
    x = x * x.T
    x /= np.max(np.abs(x))
    np.fill_diagonal(x, 1.)
    return x


def ramping_dataset(K, T, *args):
    warnings.simplefilter('ignore')

    def dist(a, b):
        return cdist(np.atleast_2d(a), np.atleast_2d(b), 'correlation')

    a = tc.mat2vec(random_corrmat(K))
    b = tc.mat2vec(random_corrmat(K))
    max_dist = dist(a, b)
    max_iter = 100
    for i in np.arange(max_iter):
        next_b = tc.mat2vec(random_corrmat(K))
        next_dist = dist(a, next_b)
        if next_dist > max_dist:
            b = next_b
            max_dist = next_dist

    mu = np.linspace(1, 0, T)

    corrs = np.zeros([T, int((K ** 2 - K) / 2 + K)])
    Y = np.zeros([T, K])

    for t in np.arange(T):
        corrs[t, :] = mu[t] * a + (1 - mu[t]) * b
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]) + .1, cov=tc.vec2mat(corrs[t, :]))

    return Y, corrs


def random_dataset(K, T, *args):
    warnings.simplefilter('ignore')

    corrs = np.zeros([T, int((K ** 2 - K) / 2 + K)])
    Y = np.zeros([T, K])

    for t in np.arange(T):
        corrs[t, :] = tc.mat2vec(random_corrmat(K))
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]), cov=tc.vec2mat(corrs[t, :]))

    return Y, corrs


def constant_dataset(K, T, *args):
    warnings.simplefilter('ignore')

    C = random_corrmat(K)
    corrs = np.tile(tc.mat2vec(C), [T, 1])

    Y = np.random.multivariate_normal(mean=np.zeros([K]) + .1, cov=C, size=T)

    return Y, corrs


def block_dataset(K, T, B=5):
    warnings.simplefilter('ignore')
    block_len = np.ceil(T / B)

    corrs = np.zeros([B, int((K ** 2 - K) / 2 + K)])
    Y = np.zeros([T, K])

    for b in np.arange(B):
        corrs[b, :] = tc.mat2vec(random_corrmat(K))
    corrs = np.repeat(corrs, block_len, axis=0)
    corrs = corrs[:T, :]

    for t in np.arange(T):
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]) + .1, cov=tc.vec2mat(corrs[t, :]))

    return Y, corrs


def simulate_data(datagen='ramping', return_corrs=False, set_random_seed=False, S=1, T=100, K=10, B=5):
    """
    Simulate timeseries data

    Parameters
    ----------
    datagen : str
        Data generation function.  Options:
            - ramping
            - block
            - constant
            - random

    return_corrs : bool
        If true, returns the correlations used to create the data

    set_random_seed : bool or int
        Default False (choose a random seed).  If True, set random seed to 123.  If int, set random seed to the specified value.

    S : int
        Number of subjects.

    T : int
        Total time

    K : int
        Number of features

    B : int
        Number of blocks

    Returns
    ----------
    data : np.ndarray
        A samples by number of electrodes array of simulated iEEG data

    sub_locs : pd.DataFrame
        A location by coordinate (x,y,z) matrix of simulated electrode locations

    """

    datagen_funcs = {'block': block_dataset, 'ramping': ramping_dataset, 'constant': constant_dataset, 'random':ramping_dataset}

    if set_random_seed:
        if isinstance(set_random_seed, bool):
            np.random.seed(123)
        else:
            np.random.seed(set_random_seed)

    if S > 1:

        s_data = []
        s_corrs = []
        for _ in np.arange(S):

            if set_random_seed:
                np.random.seed(np.random.get_state()[1][0] + 1)

            s_temp_data, s_temp_corrs = datagen_funcs[datagen](K, T, B)
            s_data.append(s_temp_data)
            s_corrs.append(s_temp_corrs)

        data = s_data
        corrs = s_corrs
    else:
        data, corrs = datagen_funcs[datagen](K, T, B)

    if return_corrs:
        return data, corrs

    else:
        return data
