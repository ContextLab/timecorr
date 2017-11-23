# coding: utf-8

import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

gaussian_params = {'var': 1000}

def gaussian_weights(T, params=gaussian_params):
    c1 = np.divide(1, np.sqrt(2 * np.math.pi * params['var']))
    c2 = np.divide(-1, 2 * params['var'])
    sqdiffs = toeplitz(np.arange(T)) ** 2
    return c1 * np.exp(c2 * sqdiffs)

def wcorr(a, b, weights, tol=1e-5):
    '''
    Compute moment-by-moment correlations between sets of observations

    :param a: a number-of-timepoints by number-of-features observations matrix
    :param b: a number-of-timepoints by number-of-features observations matrix
    :param weights: a number-of-timepoints by number-of-timepoints weights matrix specifying the per-timepoint weights
              to be considered (for each timepoint)
    :param tol: ignore all weights less than or equal (in absolute value) to tol
    :return: a a.shape[1] by b.shape[1] by weights.shape[0] array of per-timepoint correlation matrices.
    '''
    def weighted_mean_var_diffs(x, weights):
        weights[np.isnan(weights)] = 0
        if np.sum(weights) == 0:
            weights = np.ones(x.shape)

        # get rid of 0 weights to avoid unnecessary computations
        good_inds = np.abs(weights) > tol
        weights[good_inds] /= np.sum(weights[good_inds])

        weights = np.tile(weights[good_inds, np.newaxis], [1, x.shape[1]])
        x = x[good_inds, :]

        mx = np.sum(x * weights, axis=0)
        diffs = x - np.tile(mx, [x.shape[0], 1])
        varx = np.sum(np.abs(diffs) * weights, axis=0)
        return mx, varx, diffs

    autocorrelation = np.isclose(a, b).all()

    corrs = np.zeros([a.shape[1], b.shape[1], weights.shape[1]])
    for t in np.arange(weights.shape[1]):
        ma, vara, diffs_a = weighted_mean_var_diffs(a, weights[:, t])

        if autocorrelation:
            mb = ma
            varb = vara
            diffs_b = diffs_a
        else:
            mb, varb, diffs_b = weighted_mean_var_diffs(b, weights[:, t])

        alpha = np.dot(diffs_a.T, diffs_b)
        beta = np.dot(vara[:, np.newaxis], varb[np.newaxis, :])

        corrs[:, :, t] = np.divide(alpha, beta)
    return corrs

def wisfc(data, timepoint_weights, subject_weights=None):
    if type(data) != list:
        sum = 2 * wcorr(data, data, timepoint_weights)
        sum[np.isinf(sum) | np.isnan(sum)] = 0
        S = 1
        K = data.shape[1]
        T = data.shape[0]
    elif len(data) == 1:
        sum = 2 * wcorr(data[0], data[0], timepoint_weights)
        sum[np.isinf(sum) | np.isnan(sum)] = 0
        S = 1
        K = data[0].shape[1]
        T = data[0].shape[0]
    else:
        subjects = np.arange(len(data))
        S = len(subjects)
        K = data[0].shape[1]
        T = data[0].shape[0]

        if subject_weights == None:
            connectomes = np.zeros([S, (K**2 - K) / 2])
            for s in subjects:
                connectomes[s, :] = 1 - sd.pdist(data[s].T, metric='correlation')
            subject_weights = 1 - sd.squareform(sd.pdist(connectomes.T, metric='correlation'))

        sum = np.zeros([K, K, T])
        for s in subjects:
            a = data[s]
            other_inds = list([subjects[subjects != s]][0])
            b = weighted_mean(np.stack([data[x] for x in other_inds], axis=2), axis=2, weights=subject_weights[s, other_inds])

            next = wcorr(a, b, timepoint_weights)
            for t in np.arange(T):
                x = next[:, :, t]
                x[np.isinf(x) | np.isnan(x)] = 0
                z = r2z(x)
                sum[:, :, t] = np.nansum(np.stack([sum[:, :, t], z + z.T], axis=2), axis=2)

    corrs = np.zeros([T, ((K ** 2 - K)/2) + K])
    for t in np.arange(T):
        corrs[t, :] = mat2vec(np.squeeze(z2r(np.divide(sum[:, :, t], 2*S))))

    return corrs




def isfc(data, timepoint_weights):
    if type(data) == list:
        subject_weights = np.ones([1, len(data)])
    else:
        subject_weights = None

    return wisfc(data, timepoint_weights, subject_weights=subject_weights)




def weighted_mean(x, axis=None, weights=None, tol=1e-5):
    if axis is None:
        axis=len(x.shape)-1
    if weights is None:
        weights = np.ones([1, x.shape[axis]])

    #remove nans and force weights to sum to 1
    weights[np.isnan(weights)] = 0
    if np.sum(weights) == 0:
        return np.mean(x, axis=axis)

    # get rid of 0 weights to avoid unnecessary computations
    good_inds = np.abs(weights) > tol
    weights[good_inds] /= np.sum(weights[good_inds])

    weighted_sum = np.zeros(np.take(x, 0, axis=axis).shape)
    for i in np.where(good_inds)[0]:
        weighted_sum += weights[i] * np.take(x, i, axis=axis)

    return weighted_sum


def rmdiag(m):
    return m - np.diag(np.diag(m))


def r2z(r):
    return 0.5*(np.log(1+r) - np.log(1-r))


def z2r(z):
    r = np.divide((np.exp(2*z) - 1), (np.exp(2*z) + 1))
    r[np.isnan(r)] = 0
    r[np.isinf(r)] = np.sign(r)[np.isinf(r)]
    return r


def mat2vec(m):
    x = m.shape[0]
    v = np.zeros((x*x - x)/2 + x)
    v[0:x] = np.diag(m)

    #force m to be symmetric (sometimes rounding errors get introduced)
    m = np.triu(rmdiag(m))
    m += m.T

    v[x:] = sd.squareform(rmdiag(m))
    return v


def vec2mat(v):
    x = 0.5*(np.sqrt(8*len(v) + 1) - 1)
    return sd.squareform(v[(x+1):]) + np.diag(v[0:x])


def symmetric(m):
    return np.isclose(m, m.T).all()
