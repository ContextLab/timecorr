# coding: utf-8
from __future__ import division
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
    :param weights: a number-of-timepoints by number-of-timepoints weights matrix
        specifying the per-timepoint weights to be considered (for each timepoint)
    :param tol: ignore all weights less than or equal (in absolute value) to tol
    :return: a a.shape[1] by b.shape[1] by weights.shape[0] array of per-timepoint
        correlation matrices.
    '''
    def weighted_mean_var_diffs(x, weights):
        weights[np.isnan(weights)] = 0
        if np.sum(weights) == 0:
            weights = np.ones(x.shape)

        # get rid of 0 weights to avoid unnecessary computations
        good_inds = (np.abs(weights) > tol)
        weights[good_inds] /= np.sum(weights[good_inds])

        weights = np.tile(weights[good_inds, np.newaxis], [1, x.shape[1]])
        x = x[good_inds, :]

        mx = np.sum(x * weights, axis=0)
        diffs = x - np.tile(mx, [x.shape[0], 1])
        varx = np.sqrt(np.sum((diffs ** 2) * weights, axis=0))

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

        corrs[:, :, t] = np.divide(alpha, weights.shape[1] * beta)
    return corrs


def wisfc(data, timepoint_weights, subject_weights=None):
    '''
    Compute moment-by-moment correlations between sets of observations

    :timepoint weights: a number-of-timepoints by number-of-timepoints weights matrix
        specifying the per-timepoint weights to be considered (for each timepoint)
    :subject weights: ignore all weights less than or equal (in absolute value) to tol
    :return: a a.shape[1] by b.shape[1] by weights.shape[0] array of per-timepoint
        correlation matrices.
    '''
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

        if subject_weights is None:
            connectomes = np.zeros([S, int((K**2 - K) / 2)])
            for s in subjects:
                connectomes[s, :] = 1 - sd.pdist(data[s].T, metric='correlation')
            subject_weights = 1 - sd.squareform(sd.pdist(connectomes.T,
                                                metric='correlation'))
        else:
            subject_weights = np.tile(subject_weights, [S, 1])

        sum = np.zeros([K, K, T])
        for s in subjects:
            a = data[s]
            other_inds = list([subjects[subjects != s]][0])
            b = weighted_mean(np.stack([data[x] for x in other_inds], axis=2),
                              axis=2, weights=subject_weights[s, other_inds])

            next = wcorr(a, b, timepoint_weights)
            for t in np.arange(T):
                x = next[:, :, t]
                x[np.isinf(x) | np.isnan(x)] = 0
                z = r2z(x)
                sum[:, :, t] = np.nansum(np.stack([sum[:, :, t], z + z.T],
                                                  axis=2), axis=2)

    corrs = np.zeros([T, int(((K**2 - K) / 2) + K)])
    for t in np.arange(T):
        corrs[t, :] = mat2vec(np.squeeze(z2r(np.divide(sum[:, :, t], 2*S))))

    corrs[np.isinf(corrs)] = np.sign(corrs[np.isinf(corrs)])
    corrs[np.isnan(corrs)] = 0
    return corrs


def isfc(data, timepoint_weights):

    if type(data) == list:
        subject_weights = np.ones([1, len(data)])
    else:
        subject_weights = None
    return wisfc(data, timepoint_weights, subject_weights=subject_weights)


# TODO: UPDATE THIS FUNCTION FOR USE WITH TIMECORR
def smooth(w, windowsize):
    kernel = np.ones(windowsize)
    w /= kernel.sum()
    x = np.zeros([w.shape[0] - windowsize + 1, w.shape[1]])
    for i in range(0, w.shape[1]):
        x[:, i] = np.convolve(kernel, w[:, i], mode='valid')
    return x


# TODO: UPDATE THIS FUNCTION FOR USE WITH TIMECORR
# WISHLIST:
#   - support passing in a list of connectivity functions and a mixing
#     proportions vector; compute stats for all non-zero
#     mixing proportions and use those stats (weighted appropriately) to do the
#     decoding
def timepoint_decoder(data, windowsize=0, mu=0, nfolds=2, connectivity_fun=isfc):
    """
    :param data: a number-of-observations by number-of-features matrix
    :param windowsize: number of observations to include in each sliding window
                      (set to 0 or don't specify if all timepoints should be used)
    :param mu: mixing parameter-- mu = 0 means decode using raw features;
               mu = 1 means decode using ISFC; 0 < mu < 1 means decode using a
               weighted mixture of the two estimates
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :param connectivity_fun: function for transforming the group data (default: isfc)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
    """
    assert ((mu >= 0) and (mu <= 1))

    group_assignments = get_xval_assignments(len(data), nfolds)

    # fill in results
    results_template = {'rank': 0, 'accuracy': 0, 'error': 0}
    results = copy(results_template)
    for i in range(0, nfolds):
        if mu > 0:
            in_fold_isfc = squareform(connectivity_fun(data
                                      [group_assignments == i], windowsize))
            out_fold_isfc = squareform(connectivity_fun(data
                                       [group_assignments != i], windowsize))
            isfc_corrs = 1 - sd.cdist(in_fold_isfc, out_fold_isfc, 'correlation')
            if mu == 1:
                corrs = isfc_corrs

        if mu < 1:
            in_fold_raw = smooth(np.mean(data[np.where(group_assignments == i)[0]], axis=0), windowsize)
            out_fold_raw = smooth(np.mean(data[np.where(group_assignments != i)[0]], axis=0), windowsize)
            raw_corrs = 1 - sd.cdist(in_fold_raw, out_fold_raw, 'correlation')
            if mu == 0:
                corrs = raw_corrs

        if 0 < mu < 1:
            corrs = z2r(mu*r2z(raw_corrs) + (1 - mu)*r2z(isfc_corrs))

        next_results = copy(results_template)

        timepoint_dists = la.toeplitz(np.arange(corrs.shape[0]))
        for t in range(0, corrs.shape[0]):
            include_inds = np.unique(np.append(np.where(timepoint_dists[t, :] > 0), np.array(t)))
            # include_inds = np.unique(np.append(np.where(timepoint_dists[t, :] >
            # windowsize), np.array(t))) # more liberal test

            decoded_inds = include_inds[np.where(corrs[t, include_inds] ==
                                        np.max(corrs[t, include_inds]))]
            next_results['error'] += np.mean(np.abs(decoded_inds - np.array(t)))/(corrs.shape[0] - 1)
            next_results['accuracy'] += np.mean(decoded_inds == np.array(t))
            next_results['rank'] += np.mean(map((lambda x: int(x)), (corrs[t, :] <= corrs[t, t])))
        next_results['error'] /= corrs.shape[0]
        next_results['accuracy'] /= corrs.shape[0]
        next_results['rank'] /= corrs.shape[0]

        results['error'] += next_results['error']
        results['accuracy'] += next_results['accuracy']
        results['rank'] += next_results['rank']

    results['error'] /= nfolds
    results['accuracy'] /= nfolds
    results['rank'] /= nfolds
    return results


def weighted_mean(x, axis=None, weights=None, tol=1e-5):
    if axis is None:
        axis = len(x.shape) - 1
    if weights is None:
        weights = np.ones([1, x.shape[axis]])

    # remove nans and force weights to sum to 1
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
    v = np.zeros(((x*x - x)// 2) + x)
    v[0:x] = np.diag(m)

    # force m to be symmetric (sometimes rounding errors get introduced)
    m = np.triu(rmdiag(m))
    m += m.T

    v[x:] = sd.squareform(rmdiag(m))
    # before returning v, make every element of v an int?

    return v


def vec2mat(v):
    x = int(0.5*(np.sqrt(8*len(v) + 1) - 1))
    return sd.squareform(v[(x+1):]) + np.diag(v[0:x])


def symmetric(m):
    return np.isclose(m, m.T).all()


def get_xval_assignments(ndata, nfolds):
    group_assignments = np.zeros(ndata)
    groupsize = int(np.ceil(ndata / nfolds))

    # group assignments
    for i in range(1, nfolds):
        inds = np.arange(i * groupsize, np.min([(i + 1) * groupsize, ndata]))
        group_assignments[inds] = i
    np.random.shuffle(group_assignments)
    return group_assignments
