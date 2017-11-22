# coding: utf-8

import numpy as np
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

gaussian_params = {'var': 10000}

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
    def weighted_mean_var(x, weights): #TODO: DEBUG THIS...the variances don't seem right
        weights[np.isnan(weights)] = 0
        if np.sum(weights) == 0:
            weights = np.ones(x.shape)

        # get rid of 0 weights to avoid unnecessary computations
        good_inds = np.abs(weights) > tol
        weights[good_inds] /= np.sum(weights[good_inds])

        weights = np.tile(weights[good_inds, np.newaxis], [1, x.shape[1]])
        x = x[good_inds, :]

        mx = np.sum(x * weights, axis=0)
        varx = np.sum((x - np.tile(mx, [x.shape[0], 1])) * weights, axis=0)
        return mx, varx

    autocorrelation = np.isclose(a, b).all()

    corrs = np.zeros([a.shape[1], b.shape[1], weights.shape[1]])
    for t in np.arange(weights.shape[1]):
        ma, vara = weighted_mean_var(a, weights[:, t])
        diffs_a = a - np.tile(ma, [a.shape[0], 1])

        if autocorrelation:
            mb = ma
            varb = vara
            diffs_b = diffs_a
        else:
            mb, varb = weighted_mean_var(b, weights[:, t])
            diffs_b = b - np.tile(mb, [b.shape[0], 1])

        alpha = np.dot(diffs_a.T, diffs_b)
        beta = np.dot(vara[:, np.newaxis], varb[np.newaxis, :])

        corrs[:, :, t] = np.divide(alpha, beta)
    return corrs

def wisfc(data, timepoint_weights, subject_weights=None):
    if type(data) != list:
        return wcorr(data, data, timepoint_weights)
    elif len(data) == 1:
        return wcorr(data[0], data[0], timepoint_weights)

    subjects = np.arange(len(data))
    S = len(subjects)
    K = data[0].shape[1]
    T = timepoint_weights.shape[1]

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

    corrs = np.zeros([T, (K ** 2 - K)/2])
    for t in np.arange(T):
        corrs[t, :] = squareform(z2r(np.divide(sum[:, :, t], (2*S))))

    return corrs




def isfc(data, timepoint_weights):
    if type(data) == list:
        subject_weights = np.ones([1, len(data)])
    else:
        subject_weights = None

    return wisfc(data, timepoint_weights, subject_weights=subject_weights)

    #f (type(data) != list):
    #    return wcorr(data, data, weights)
    #elif len(data) == 1:
    #    return wcorr(data[0], data[0], weights)
    #else: #need to do the multi-subject thing
    #    subjects = np.arange(len(data))
    #    T = data[0].shape[0]
    #    V = data[0].shape[1]
    #    sum = np.zeros([T, ((V ** 2) - V) / 2])
    #    for s in subjects:
    #        other_inds = subjects[subjects != s]
    #        other_mean = np.mean(np.stack([data[x] for x in other_inds], axis=2), axis=2)
    #        sum += r2z(wcorr(data[s], other_mean, weights))
    #    return z2r(np.divide(sum, len(data)))



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


def squareform(m):
    if len(m.shape) == 3:
        v = m.shape[0]
        x = np.zeros([m.shape[2], (v*v - v)/2 + v])
        for i in range(0, m.shape[2]):
            x[i, :] = mat2vec(np.squeeze(m[:, :, i]))
    elif len(m.shape) == 2:
        v = 0.5*(np.sqrt(8*m.shape[1] + 1) + 1)
        x = np.zeros([v, v, m.shape[0]])
        for i in range(0, m.shape[0]):
            x[:, :, i] = vec2mat(m[i, :])
    else: #do nothing
        x = m
    return x


def rmdiag(m):
    return m - np.diag(np.diag(m))


def r2z(r):
    return 0.5*(np.log(1+r) - np.log(1-r))


def z2r(z):
    return (np.exp(2*z) - 1)/(np.exp(2*z) + 1)


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
