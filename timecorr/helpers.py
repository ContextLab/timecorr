# coding: utf-8

import numpy as np
import scipy.spatial.distance as sd
from multiprocessing import Pool

gaussian_params = {'var': 1000}

def gaussian_weights(timepoints, t, params=gaussian_params):
    def exp(x):
        return np.array(list(map(np.math.exp, x)))

    return np.divide(exp(-np.divide(np.power(np.subtract(timepoints, t), 2), (2 * params['var']))), np.math.sqrt(2 * np.math.pi * params['var']))

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


def wisfc(data, timepoint_weights, subject_weights=None):
    if (type(data) != list):
        return wcorr(data, data, weights)
    elif len(data) == 1:
        return wcorr(data[0], data[0], weights)

    T = data[0].shape[0]
    V = data[0].shape[1]

    connectomes = np.zeros([len(data), ((V ** 2) - V) / 2])
    subjects = np.arange(len(data))

    def get_weighted_connectome(s):
        print('computing connectome for subject ' + str(s))
        return wcorr(data[s], data[s], timepoint_weights)

    #for s in subjects:
    #    connectomes[s, :] = wcorr(data[s], data[s], timepoint_weights)

    #weight subjects by how similar they are to each other
    if subject_weights is None:
        p = Pool(len(subjects))
        connectomes = list(p.map(get_weighted_connectome, subjects))
        connectomes = np.vstack(connectomes)

        subject_weights = 1 - sd.squareform(sd.cdist(connectomes.T, metric='correlation'))

    def subfun(s):
        print('working on subject ' + str(s))
        other_inds = subjects[subjects != s]
        other_mean = weighted_mean(np.stack(data[other_inds], axis=2), axis=2, weights=subject_weights[s, :])
        return r2z(wcorr(data[s], other_mean, timepoint_weights))

    p = Pool(len(subjects))
    zcorrs = list(p.map(subfun, subjects))

    sum = np.zeros([T, ((V ** 2) - V) / 2])
    for s in subjects:
        #other_inds = subjects[subjects != s]
        #other_mean = weighted_mean(np.stack(data[other_inds], axis=2), axis=2, weights=subject_weights[s, :])
        sum += zcorrs[s] #r2z(wcorr(data[s], other_mean, timepoint_weights))
    return z2r(np.divide(sum, len(subjects)))


def weighted_mean(x, axis=None, weights=None):
    if axis is None:
        axis=len(x.shape)-1
    if weights is None:
        weights = np.ones([1, x.shape[axis]])

    #remove nans and force weights to sum to 1
    weights[np.isnan(weights)] = 0
    if np.sum(weights) == 0:
        return np.mean(x, axis=axis)

    weights = np.divide(weights, np.sum(weights))

    #multiply each slice of x by its weight and then sum along the specified dimension
    return np.sum(np.stack(list(map(lambda w, x: np.multiply(w, x), weights, np.split(x, x.shape[axis], axis=axis))), axis=axis), axis=axis)

def weighted_std(x, axis=None, weights=None):
    if axis is None:
        axis=len(x.shape)-1

    mean_x = weighted_mean(x, axis=axis, weights=weights)
    dims = np.ones([1, len(x.shape)], dtype=np.int)[0]
    dims[axis] = x.shape[axis]

    diffs = x - np.tile(mean_x, dims)
    return np.sqrt(weighted_mean(np.power(diffs, 2), axis=axis, weights=weights))

def wcorr(x, y, weights):
    def wcorr_helper(a, b, weights):
        a = a[:, np.newaxis]
        b = b[:, np.newaxis]
        weights = weights[:, np.newaxis]
        ma = weighted_mean(a, axis=0, weights=weights)
        mb = weighted_mean(b, axis=0, weights=weights)
        sa = weighted_std(a, axis=0, weights=weights)
        sb = weighted_std(b, axis=0, weights=weights)

        return np.divide(weighted_mean((a - ma) * (b - mb), axis=0, weights=weights), sa * sb)[0][0]

    wcorrfun = lambda a, b: wcorr_helper(a, b, weights)
    if np.isclose(x, y).all(): #autocorrelation
        return sd.squareform(sd.pdist(x.T, metric=wcorrfun))
    else:
        return sd.cdist(x.T, y.T, metric=wcorrfun)

    #for i in np.arange(x.shape[1]):
    #    if autocorr:
    #        stop = i
    #        r[i, i] = 1
    #    else:
    #        stop = y.shape[1]
    #    a = x[:, i]
    #    for j in np.arange(stop):
    #        b = y[:, j]
    #        r[i, j] = wcorr_helper(a[:, np.newaxis], b[:, np.newaxis], weights[:, np.newaxis])
    #
    #        if autocorr:
    #            r[j, i] = r[i, j]
    #return r


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


def get_xval_assignments(ndata, nfolds):
    group_assignments = np.zeros(ndata)
    groupsize = int(np.ceil(ndata / nfolds))

    # group assignments
    for i in range(1, nfolds):
        inds = np.arange(i * groupsize, np.min([(i + 1) * groupsize, ndata]))
        group_assignments[inds] = i
    np.random.shuffle(group_assignments)
    return group_assignments


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
