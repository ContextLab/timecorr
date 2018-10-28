import timecorr as tc
import hypertools as hyp
import seaborn as sns
import numpy as np
from scipy.io import loadmat as load
from timecorr.helpers import isfc, wisfc
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import wishart
from scipy.spatial.distance import cdist
import warnings

# Synthetic data tests
def random_corrmat(K):
    x = np.random.randn(K, K)
    x = x * x.T
    x /= np.max(np.abs(x))
    np.fill_diagonal(x, 1.)
    return x


def ramping_dataset(K, T):
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
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]), cov=tc.vec2mat(corrs[t, :]))

    return Y, corrs


def random_dataset(K, T):
    warnings.simplefilter('ignore')

    corrs = np.zeros([T, int((K ** 2 - K) / 2 + K)])
    Y = np.zeros([T, K])

    for t in np.arange(T):
        corrs[t, :] = tc.mat2vec(random_corrmat(K))
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]), cov=tc.vec2mat(corrs[t, :]))

    return Y, corrs


def constant_dataset(K, T):
    warnings.simplefilter('ignore')

    C = random_corrmat(K)
    corrs = np.tile(tc.mat2vec(C), [T, 1])

    Y = np.random.multivariate_normal(mean=np.zeros([K]), cov=C, size=T)

    return Y, corrs


B = 5  # number of blocks


def block_dataset(K, T):
    warnings.simplefilter('ignore')
    block_len = np.ceil(T / B)

    corrs = np.zeros([B, int((K ** 2 - K) / 2 + K)])
    Y = np.zeros([T, K])

    for b in np.arange(B):
        corrs[b, :] = tc.mat2vec(random_corrmat(K))
    corrs = np.repeat(corrs, block_len, axis=0)
    corrs = corrs[:T, :]

    for t in np.arange(T):
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]), cov=tc.vec2mat(corrs[t, :]))

    return Y, corrs


def identity_compare(obs_corrs=None):
    if obs_corrs is None:
        return 1

    return [obs_corrs]


def first_compare(obs_corrs=None):
    if obs_corrs is None:
        return 1

    T = obs_corrs.shape[0]
    return [np.tile(obs_corrs[0, :], [T, 1])]


def last_compare(obs_corrs=None):
    if obs_corrs is None:
        return 1

    T = obs_corrs.shape[0]
    return [np.tile(obs_corrs[-1, :], [T, 1])]


def ramping_compare(obs_corrs=None):
    if obs_corrs is None:
        return 2

    T = obs_corrs.shape[0]
    return [np.tile(obs_corrs[0, :], [T, 1]), np.tile(obs_corrs[-1, :], [T, 1])]


def block_compare(obs_corrs=None):
    if obs_corrs is None:
        return B

    T = obs_corrs.shape[0]
    block_len = np.ceil(T / B)

    bs = np.repeat(np.atleast_2d(np.arange(B)).T, block_len, axis=0)
    bs = bs[:T, :]

    obs_parsed = []
    for b in np.arange(B):
        i = np.where(bs == b)[0][0]
        obs_parsed.append(np.tile(obs_corrs[i, :], [T, 1]))
    return obs_parsed

width = 10
delta = {'name': '$\delta$', 'weights': tc.eye_weights, 'params': tc.eye_params}
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
mexican_hat = {'name': 'Mexican hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}
kernels = [delta, gaussian, laplace, mexican_hat]

# K = 25
# T = 1000
# N = 1000
#
# synth_data, synth_corrs = ramping_dataset(K, T)
# recovered_corrs = []
# for k in kernels:
#     recovered_corrs.append(tc.timecorr(synth_data, weights_function=k['weights'], weights_params=k['params']))



pieman_data = load('/Users/jeremy/timecorr-paper/data/pieman_ica100.mat')
pieman_conds = ['intact', 'paragraph', 'word', 'rest']

x = []
conds = []
for c in pieman_conds:
    next_data = list(map(lambda i: pieman_data[c][:, i][0], np.arange(pieman_data[c].shape[1])))
    x.extend(next_data)
    conds.extend([c]*len(next_data))
del pieman_data

x = [x[i] for i in np.where(np.array(conds) == 'intact')[0]]

isfc_within = tc.timecorr(x, cfun=isfc)
isfc_across = tc.helpers.corrmean_combine(isfc_within)
isfc_eig = tc.helpers.reduce(isfc_across, rfun='eigenvector_centrality')
isfc_pagerank = tc.helpers.reduce(isfc_across, rfun='pagerank_centrality')
isfc_strength = tc.helpers.reduce(isfc_across, rfun='strength')
isfc_PCA = tc.helpers.reduce(isfc_across, rfun='IncrementalPCA')

hyp.plot([isfc_eig, isfc_pagerank, isfc_strength, isfc_PCA], legend=['eig', 'pagerank', 'strength', 'PCA'], align='hyper')

wisfc_within = tc.timecorr(x, cfun=wisfc)
wisfc_across = tc.helpers.corrmean_combine(isfc_within)

hyp.plot([isfc_across, wisfc_across])
hyp.plot(x)

sns.heatmap(isfc_across)
sns.heatmap(wisfc_across)


