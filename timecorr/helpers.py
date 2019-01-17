# coding: utf-8
from __future__ import division
import numpy as np
import scipy.spatial.distance as sd
from scipy.special import gamma
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.stats import ttest_1samp as ttest
import hypertools as hyp
import brainconn as bc
import pandas as pd
import warnings
from matplotlib import pyplot as plt

from copy import copy, deepcopy

graph_measures = {'eigenvector_centrality': bc.centrality.eigenvector_centrality_und,
                  'pagerank_centrality': lambda x: bc.centrality.pagerank_centrality(x, d=0.85),
                  'strength': bc.degree.strengths_und}

gaussian_params = {'var': 100}
laplace_params = {'scale': 100}
eye_params = {}
t_params = {'df': 100}
mexican_hat_params = {'sigma': 10}
uniform_params = {}


def gaussian_weights(T, params=gaussian_params):
    if params is None:
        params = gaussian_params

    c1 = np.divide(1, np.sqrt(2 * np.math.pi * params['var']))
    c2 = np.divide(-1, 2 * params['var'])
    sqdiffs = toeplitz(np.arange(T) ** 2)
    return c1 * np.exp(c2 * sqdiffs)

def laplace_weights(T, params=laplace_params):
    if params is None:
        params = laplace_params

    absdiffs = toeplitz(np.arange(T))
    return np.multiply(np.divide(1, 2 * params['scale']), np.exp(-np.divide(absdiffs, params['scale']))) #scale by a factor of 2.5 to prevent near-zero rounding issues


def eye_weights(T, params=eye_params):
    #if params is None:
    #    params = eye_params

    return np.eye(T)

def uniform_weights(T, params=uniform_params):
    return np.ones([T, T])

def t_weights(T, params=t_params):
    if params is None:
        params = t_params

    c1 = np.divide(gamma((params['df'] + 1) / 2), np.sqrt(params['df'] * np.math.pi) * gamma(params['df'] / 2))
    c2 = np.divide(-params['df'] + 1, 2)

    sqdiffs = toeplitz(np.arange(T) ** 2)
    return np.multiply(c1, np.power(1 + np.divide(sqdiffs, params['df']), c2))

def mexican_hat_weights(T, params=mexican_hat_params):
    if params is None:
        params = mexican_hat_params

    absdiffs = toeplitz(np.arange(T))
    sqdiffs = toeplitz(np.arange(T) ** 2)

    a = np.divide(2, np.sqrt(3 * params['sigma']) * np.power(np.math.pi, 0.25))
    b = 1 - np.power(np.divide(absdiffs, params['sigma']), 2)
    c = np.exp(-np.divide(sqdiffs, 2 * np.power(params['sigma'], 2)))

    return np.multiply(a, np.multiply(b, c))

def format_data(data):
    return hyp.tools.format_data(data)

def _is_empty(dict):
    if not bool(dict):
        return True
    return False



def wcorr(a, b, weights):
    '''
    Compute moment-by-moment correlations between sets of observations

    :param a: a number-of-timepoints by number-of-features observations matrix
    :param b: a number-of-timepoints by number-of-features observations matrix
    :param weights: a number-of-timepoints by number-of-timepoints weights matrix
        specifying the per-timepoint weights to be considered (for each timepoint)
    :return: a a.shape[1] by b.shape[1] by weights.shape[0] array of per-timepoint
        correlation matrices.
    '''
    def weighted_var_diffs(x, w):
        w[np.isnan(w)] = 0
        if np.sum(np.abs(w)) == 0:
            weights_tiled = np.ones(x.shape)
        else:
            weights_tiled = np.tile(w[:, np.newaxis], [1, x.shape[1]])

        mx = np.sum(np.multiply(weights_tiled, x), axis=0)[:, np.newaxis].T
        diffs = x - np.tile(mx, [x.shape[0], 1])
        varx = np.sum(diffs ** 2, axis=0)[:, np.newaxis].T

        return varx, diffs

    autocorrelation = np.isclose(a, b).all()
    corrs = np.zeros([a.shape[1], b.shape[1], weights.shape[1]])

    for t in np.arange(weights.shape[1]):
        vara, diffs_a = weighted_var_diffs(a, weights[:, t])

        if autocorrelation:
            varb = vara
            diffs_b = diffs_a
        else:
            varb, diffs_b = weighted_var_diffs(b, weights[:, t])

        alpha = np.dot(diffs_a.T, diffs_b)
        beta = np.sqrt(np.dot(vara.T, varb))
        corrs[:, :, t] = np.divide(alpha, beta)

    return corrs

def wisfc(data, timepoint_weights, subject_weights=None):
    '''
    Compute moment-by-moment correlations between sets of observations

    :data: a list of number-of-timepoints by V matrices
    :timepoint weights: a number-of-timepoints by number-of-timepoints weights matrix
        specifying the per-timepoint weights to be considered (for each timepoint)
    :subject weights: number-of-subjects by number-of-subjects weights matrix
    :return: a list of number-of-timepoints by (V^2 - V)/2 + V correlation matrices
    '''
    if type(data) != list:
        return wisfc([data], timepoint_weights, subject_weights=subject_weights)[0]

    if subject_weights is None: #similarity-based weights
        K = data[0].shape[1]
        connectomes = np.zeros([len(data), int((K ** 2 - K) / 2)])
        for s in np.arange(len(data)):
            connectomes[s, :] = 1 - sd.pdist(data[s].T, metric='correlation')
        subject_weights = 1 - sd.squareform(sd.pdist(connectomes, metric='correlation'))
        np.fill_diagonal(subject_weights, 0)
    elif np.isscalar(subject_weights):
        subject_weights = subject_weights * np.ones([len(data), len(data)])
        np.fill_diagonal(subject_weights, 0)

    corrs = []
    for s, a in enumerate(data):
        b = weighted_mean(np.stack(data, axis=2), axis=2, weights=subject_weights[s, :])
        corrs.append(mat2vec(wcorr(a, b, timepoint_weights)))

    return corrs


def isfc(data, timepoint_weights):
    if type(data) != list:
        return isfc([data], timepoint_weights)[0]

    return wisfc(data, timepoint_weights, subject_weights=1 - np.eye(len(data)))

def autofc(data, timepoint_weights):
    if type(data) != list:
        return autofc([data], timepoint_weights)[0]

    return wisfc(data, timepoint_weights, subject_weights=np.eye(len(data)))

def apply_by_row(corrs, f):
    '''
    apply the function f to the correlation matrix specified in each row, and return a
    matrix of the concatenated results

    :param corrs: a matrix of vectorized correlation matrices (output of mat2vec), or a list
                  of such matrices
    :param f: a function to apply to each vectorized correlation matrix
    :return: a matrix of function outputs (for each row of the given matrices), or a list of
            such matrices
    '''

    if type(corrs) is list:
        return list(map(lambda x: apply_by_row(x, f), corrs))

    corrs = vec2mat(corrs) #V by V by T
    return np.stack(list(map(lambda x: f(np.squeeze(x)), np.split(corrs, corrs.shape[2], axis=2))), axis=0)

def corrmean_combine(corrs):
    '''
    Compute the mean element-wise correlation across each matrix in a list.

    :param corrs: a matrix of vectorized correlation matrices (output of mat2vec), or a list
                  of such matrices
    :return: a mean vectorized correlation matrix
    '''
    if not (type(corrs) == list):
        return corrs
    else:
        return z2r(np.mean(r2z(np.stack(corrs, axis=2)), axis=2))

def mean_combine(vals):
    '''
    Compute the element-wise mean across each matrix in a list.

    :param vals: a matrix, or a list of matrices
    :return: a mean matrix
    '''
    if not (type(vals) == list):
        return vals
    else:
        return np.mean(np.stack(vals, axis=2), axis=2)

def tstat_combine(corrs, return_pvals=False):
    '''
    Compute element-wise t-tests (comparing distribution means to 0) across each
    correlation matrix in a list.

    :param corrs: a matrix of vectorized correlation matrices (output of mat2vec), or a list
                  of such matrices

    :param return_pvals: Boolean (default: False).  If True, return a second matrix (or list)
                         of the corresponding t-tests' p-values

    :return: a matrix of t-statistics of the same shape as a matrix of vectorized correlation
             matrices
    '''
    if not (type(corrs) == list):
        ts = corrs
        ps = np.nan * np.zeros_like(corrs)
    else:
        ts, ps = ttest(r2z(np.stack(corrs, axis=2)), popmean=0, axis=2)

    if return_pvals:
        return ts, ps
    else:
        return ts

def null_combine(corrs):
    '''
    Placeholder function that returns the input

    :param corrs: a matrix of vectorized correlation matrices (output of mat2vec), or a list
                  of such matrices

    :return: the input
    '''
    return corrs

def reduce(corrs, rfun=None):
    '''
    :param corrs: a matrix of vectorized correlation matrices (output of mat2vec), or a list
                  of such matrices

    :param rfun: function to use for dimensionality reduction.  All hypertools and
        scikit-learn functions are supported: PCA, IncrementalPCA, SparsePCA,
        MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
        DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
        SpectralEmbedding, LocallyLinearEmbedding, MDS, and UMAP.

        Can be passed as a string, but for finer control of the model
        parameters, pass as a dictionary, e.g.
        reduction={‘model’ : ‘PCA’, ‘params’ : {‘whiten’ : True}}.

        See scikit-learn specific model docs for details on parameters supported
        for each model.

        Another option is to use graph theoretic measures computed for each node.
        The following measures are supported (via the brainconn toolbox):
        eigenvector_centrality, pagerank_centrality, and strength.  (Each
        of these must be specified as a string; dictionaries not supported.)

        Default: None (no dimensionality reduction)

    :return: dimensionality-reduced (or original) correlation matrices
    '''

    if rfun is None:
        return corrs

    get_V = lambda x: int(np.divide(np.sqrt(8 * x + 1) - 1, 2))

    if type(corrs) is list:
        V = get_V(corrs[0].shape[1])
    else:
        V = get_V(corrs.shape[1])

    if rfun in graph_measures.keys():
        return apply_by_row(corrs, graph_measures[rfun])
    else:  # use hypertools
        return hyp.reduce(corrs, reduce=rfun, ndims=V)


def smooth(w, windowsize=10, kernel_fun=laplace_weights, kernel_params=laplace_params):
    if type(w) is list:
        return list(map(lambda x: smooth(x, windowsize=windowsize, kernel_fun=kernel_fun, kernel_params=kernel_params), w))

    assert type(windowsize) == int, 'smoothing kernel must have integer width'
    k = kernel_fun(windowsize, params=kernel_params)
    if iseven(windowsize):
        kernel = np.divide(k[int(np.floor(windowsize/2) - 1), :] + k[int(np.ceil(windowsize/2) - 1), :], 2)
    else:
        kernel = k[int(np.floor(windowsize/2)), :]

    kernel /= kernel.sum()
    x = np.zeros_like(w)
    for i in range(0, w.shape[1]):
        x[:, i] = np.convolve(kernel, w[:, i], mode='same')
    return x


def timepoint_decoder(data, mu=None, nfolds=2, level=0, cfun=isfc, weights_fun=laplace_weights, weights_params=laplace_params,
                      combine=mean_combine, rfun=None):
    """
    :param data: a list of number-of-observations by number-of-features matrices
    :param mu: list of floats sum to one for mixing proportions vector
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :param level: integer or list of integers for levels to be evaluated (default:0)
    :param cfun: function for transforming the group data (default: isfc)
    :param weights_fun: used to compute per-timepoint weights for cfun; default: laplace_weights
    :param  weights_params: parameters passed to weights_fun; default: laplace_params
    :params combine: function for combining data within each group, or a list of such functions (default: mean_combine)
    :param rfun: function for reducing output (default: None)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
    """

    assert len(np.unique(
        list(map(lambda x: x.shape[0], data)))) == 1, 'all data matrices must have the same number of timepoints'
    assert len(np.unique(
        list(map(lambda x: x.shape[1], data)))) == 1, 'all data matrices must have the same number of features'

    group_assignments = get_xval_assignments(len(data), nfolds)


    orig_level = level
    orig_level = np.ravel(orig_level)

    if type(level) is int:
        level = np.arange(level + 1)

    level = np.ravel(level)

    assert type(level) is np.ndarray, 'level needs be an integer, list, or np.ndarray'
    assert not np.any(level < 0), 'level cannot contain negative numbers'

    if mu:
        orig_level = level.max()
        orig_level = np.ravel(orig_level)
        assert np.sum(mu)==1, 'weights must sum to one'
        assert np.shape(mu)[0]== level.max()+1, 'weights lengths need to be the same as number of levels'

    if not np.all(np.arange(level.max()+1)==level):
        level = np.arange(level.max()+1)

    if callable(combine):
        combine = [combine] * np.shape(level)[0]

    combine = np.ravel(combine)

    assert type(combine) is np.ndarray and type(combine[0]) is not np.str_, 'combine needs to be a function, list of functions, or np.ndarray of functions'
    assert len(level)==len(combine), 'combine length need to be the same as level if input is type np.ndarray or list'

    if callable(cfun):
        cfun = [cfun] * np.shape(level)[0]

    cfun = np.ravel(cfun)

    assert type(cfun) is np.ndarray and type(cfun[0]) is not np.str_, 'combine needs be a function, list of functions, or np.ndarray of functions'
    assert len(level)==len(cfun), 'cfun length need to be the same as level if input is type np.ndarray or list'


    if type(rfun) not in [list, np.ndarray]:
        rfun = [rfun] * np.shape(level)[0]

    assert len(level)==len(rfun), 'parameter lengths need to be the same as level if input is ' \
                                                           'type np.ndarray or list'

    results_pd = pd.DataFrame()

    corrs = 0
    for i in range(0, nfolds):

        in_raw = []
        out_raw = []

        for v in level:

            if v==0:
                in_data = [x for x in data[group_assignments == i]]
                out_data = [x for x in data[group_assignments != i]]

                in_smooth, out_smooth, in_raw, out_raw = folding_levels(in_data, out_data, level=v, cfun=None,rfun=rfun,
                                                                        combine=combine, weights_fun=weights_fun,
                                                                        weights_params=weights_params)

            else:

                in_smooth, out_smooth, in_raw, out_raw = folding_levels(in_raw, out_raw, level=v, cfun=cfun,
                                                                        rfun=rfun, combine=combine,
                                                                        weights_fun=weights_fun,
                                                                        weights_params=weights_params)

            if mu:
                next_corrs = (1 - sd.cdist(in_smooth, out_smooth, 'correlation'))
                corrs += mu[v] * z2r(next_corrs)

            else:
                corrs = (1 - sd.cdist(in_smooth, out_smooth, 'correlation'))

            if v in orig_level:

                if mu:
                    corrs = r2z(corrs)

                next_results_pd = decoder(corrs)
                next_results_pd['level'] = v
                next_results_pd['folds'] = i
                results_pd = pd.concat([results_pd, next_results_pd])


    return results_pd

def optimize_weighted_timepoint_decoder(data, nfolds=2, level=0, cfun=isfc, weights_fun=laplace_weights,
                                        weights_params=laplace_params, combine=mean_combine, rfun=None):
    """
    :param data: a list of number-of-observations by number-of-features matrices
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :param level: integer or list of integers for levels to be evaluated (default:0)
    :param cfun: function for transforming the group data (default: isfc)
    :param weights_fun: used to compute per-timepoint weights for cfun; default: laplace_weights
    :param  weights_params: parameters passed to weights_fun; default: laplace_params
    :params combine: function for combining data within each group, or a list of such functions (default: mean_combine)
    :param rfun: function for reducing output (default: None)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
    """

    assert len(np.unique(
        list(map(lambda x: x.shape[0], data)))) == 1, 'all data matrices must have the same number of timepoints'
    assert len(np.unique(
        list(map(lambda x: x.shape[1], data)))) == 1, 'all data matrices must have the same number of features'

    if nfolds == 1:
        sub_nfolds = 1
        nfolds = 2
        warnings.warn('When nfolds is set to one, the analysis will be circular.')
    else:
        sub_nfolds = nfolds

    group_assignments = get_xval_assignments(len(data), nfolds)

    # if nfolds == 1:
    #     sub_nfolds = 1
    #     nfolds = 2
    #     warnings.warn('When nfolds is set to one, the analysis will be circular.')


    subgroup_assignments = get_xval_assignments(len(data[group_assignments == 0]), nfolds)

    orig_level = level
    orig_level = np.ravel(orig_level)

    if type(level) is int:
        level = np.arange(level + 1)

    level = np.ravel(level)

    assert type(level) is np.ndarray, 'level needs be an integer, list, or np.ndarray'
    assert not np.any(level < 0), 'level cannot contain negative numbers'

    if not np.all(np.arange(level.max()+1)==level):
        level = np.arange(level.max()+1)

    if callable(combine):
        combine = [combine] * np.shape(level)[0]

    combine = np.ravel(combine)

    assert type(combine) is np.ndarray and type(combine[0]) is not np.str_, 'combine needs to be a function, list of ' \
                                                                            'functions, or np.ndarray of functions'
    assert len(level)==len(combine), 'combine length need to be the same as level if input is type np.ndarray or list'

    if callable(cfun):
        cfun = [cfun] * np.shape(level)[0]

    cfun = np.ravel(cfun)

    assert type(cfun) is np.ndarray and type(cfun[0]) is not np.str_, 'combine needs be a function, list of functions, ' \
                                                                      'or np.ndarray of functions'
    assert len(level)==len(cfun), 'cfun length need to be the same as level if input is type np.ndarray or list'


    if type(rfun) not in [list, np.ndarray]:
        rfun = [rfun] * np.shape(level)[0]

    assert len(level)==len(rfun), 'parameter lengths need to be the same as level if input is ' \
                                                           'type np.ndarray or list'

    results_pd = pd.DataFrame()

    for i in range(0, nfolds):

        in_raw = []
        out_raw = []
        sub_in_raw = []
        sub_out_raw = []
        sub_corrs = []
        corrs = []
        for v in level:

            if v==0:
                in_data = [x for x in data[group_assignments == i]]
                out_data = [x for x in data[group_assignments != i]]

                # if nfolds == 1:
                #     in_data = out_data

                in_smooth, out_smooth, in_raw, out_raw = folding_levels(in_data, out_data, level=v, cfun=None,rfun=rfun,
                                                                        combine=combine, weights_fun=weights_fun,
                                                                        weights_params=weights_params)
                #
                for s in range(0, nfolds):

                    sub_in_data = [x for x in data[group_assignments == i][subgroup_assignments==s]]
                    sub_out_data = [x for x in data[group_assignments == i][subgroup_assignments!=s]]

                    sub_in_smooth, sub_out_smooth, sub_in_raw, sub_out_raw = folding_levels(sub_in_data, sub_out_data,
                                                                                            level=v, cfun=None, rfun=rfun,
                                                                                            combine=combine,
                                                                                            weights_fun=weights_fun,
                                                                                            weights_params=weights_params)

            else:

                in_smooth, out_smooth, in_raw, out_raw = folding_levels(in_raw, out_raw, level=v, cfun=cfun,
                                                                        rfun=rfun, combine=combine,
                                                                        weights_fun=weights_fun,
                                                                        weights_params=weights_params)

                for s in range(0, nfolds):

                    sub_in_smooth, sub_out_smooth, sub_in_raw, sub_out_raw = folding_levels(sub_in_raw, sub_out_raw,
                                                                                            level=v, cfun=cfun,
                                                                                            rfun=rfun, combine=combine,
                                                                                            weights_fun=weights_fun,
                                                                                            weights_params=weights_params)


            next_corrs = (1 - sd.cdist(in_smooth, out_smooth, 'correlation'))
            next_subcorrs = (1 - sd.cdist(sub_in_smooth, sub_out_smooth, 'correlation'))

            corrs.append(next_corrs)
            sub_corrs.append(next_subcorrs)

        sub_corrs = np.array(sub_corrs)
        corrs = np.array(corrs)

        if sub_nfolds == 1:
            sub_corrs = corrs

        for lev in range(v+1):

            mu = optimize_weights(sub_corrs[0:lev+1,:,:])
            w_corrs = weight_corrs(corrs[0:lev+1,:,:], mu)

            next_results_pd = decoder(w_corrs)
            next_results_pd['level'] = v
            next_results_pd['folds'] = i

            mu_pd = pd.DataFrame()
            for c in np.arange(lev + 1):
                mu_pd['level_' + str(c)] = [0]

            #mu_pd['level_' + str(lev)] = [0]
            mu_pd += mu

            next_results_pd = pd.concat([next_results_pd, mu_pd], axis=1, join_axes=[next_results_pd.index])

            results_pd = pd.concat([results_pd, next_results_pd])


    return results_pd

def folding_levels(infold_data, outfold_data, level=0, cfun=None, weights_fun=None, weights_params=None, combine=None,
                   rfun=None):

    from .timecorr import timecorr

    if level == 0:

        in_fold_smooth = np.asarray(timecorr([x for x in infold_data], cfun=None,
                                             rfun=rfun[level], combine=combine[level], weights_function=weights_fun,
                                             weights_params=weights_params))
        out_fold_smooth = np.asarray(timecorr([x for x in outfold_data], cfun=None,
                                                  rfun=rfun[level], combine=combine[level], weights_function=weights_fun,
                     weights_params=weights_params))
        in_fold_raw = mean_combine([x for x in infold_data])
        out_fold_raw = mean_combine([x for x in outfold_data])

    else:
        in_fold_smooth = np.asarray(timecorr(infold_data, cfun=cfun[level], rfun=rfun[level], combine=combine[level],
                                                 weights_function=weights_fun, weights_params=weights_params))
        out_fold_smooth = np.asarray(timecorr(outfold_data, cfun=cfun[level], rfun=rfun[level], combine=combine[level],
                                                  weights_function=weights_fun, weights_params=weights_params))
        in_fold_raw = np.asarray(timecorr(infold_data, cfun=cfun[level], rfun=rfun[level], combine=null_combine,
                                              weights_function=eye_weights, weights_params=eye_params))
        out_fold_raw = np.asarray(timecorr(outfold_data, cfun=cfun[level], rfun=rfun[level], combine=null_combine,
                                               weights_function=eye_weights, weights_params=eye_params))

    return in_fold_smooth, out_fold_smooth, in_fold_raw, out_fold_raw

def optimize_weights(corrs):

    b = (0, 1)
    bns = (b,) * np.shape(corrs)[0]
    con1 = {'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}
    x0 = np.repeat(1/np.shape(corrs)[0], np.shape(corrs)[0])

    min_mu = minimize(calculate_error, x0, args=corrs, bounds=bns, constraints=con1, options={'disp': True, 'eps': 1e-1})

    return min_mu.x

def calculate_error(mu, corrs, metric='error', sign=1):

    results = decoder(weight_corrs(corrs, mu))
    return sign * results[metric].values


def weight_corrs(corrs, mu):

    assert np.shape(mu)[0] == len(corrs)
    weighted_corrs = 0

    for i in np.arange(np.shape(corrs)[0]):
        weighted_corrs += mu[i] * r2z(corrs[i])

    return z2r(weighted_corrs)

def decoder(corrs):

    next_results_pd = pd.DataFrame({'rank': [0], 'accuracy': [0], 'error': [0]})
    for t in np.arange(corrs.shape[0]):
        decoded_inds = np.argmax(corrs[t, :])
        next_results_pd['error'] += np.mean(np.abs(decoded_inds - np.array(t))) / corrs.shape[0]
        next_results_pd['accuracy'] += np.mean(decoded_inds == np.array(t))
        next_results_pd['rank'] += np.mean(list(map((lambda x: int(x)), (corrs[t, :] <= corrs[t, t]))))

    next_results_pd['error'] =  next_results_pd['error'].values / corrs.shape[0]
    next_results_pd['accuracy'] = next_results_pd['accuracy'].values / corrs.shape[0]
    next_results_pd['rank']= next_results_pd['rank'].values / corrs.shape[0]

    return next_results_pd


# def predict(x, n=1):
#     '''
#     Use a Kalman filter (with automatically inferred parameters) to estimate
#     future states of a signal, n timepoints into the future.
#
#     x: timepoints by features signal (numpy array)
#     n: number of timepoints into the future (must be an integer)
#
#     Returns a new numpy array with x.shape[0] + n rows and x.shape[1] columns,
#     where the last n rows contain the predicted future states.  The other
#     entries contain "smoothed" estimates of the observed signals.
#     '''
#
#     if n == 0:
#         return kf.em(x).smooth(x)[0]
#
#     x_masked = np.ma.MaskedArray(np.vstack((x, np.tile(np.nan, (1, x.shape[1])))))
#     x_masked[-1, :] = np.ma.masked
#
#     kf = pykalman.KalmanFilter(initial_state_mean=np.mean(x, axis=0), n_dim_obs=x.shape[1], n_dim_state=x.shape[1])
#     x_predicted = kf.em(x_masked, em_vars='all').smooth(x_masked)
#
#     if n == 1:
#         return x_predicted[0] #x_predicted[1] contains timepoint-by-timepoint covariance estimates
#     elif n > 1:
#         next_x_predicted = predict(x_predicted[0], n-1)
#         diff = next_x_predicted.shape[0] - x_predicted[0].shape[0]
#         next_x_predicted[:-diff, :] = x_predicted[0]
#         return next_x_predicted


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


def generate_random_covariance_matrix(N):
    template = np.random.randn(N, N)
    cov = np.multiply(template, template.T)
    return mat2vec(cov)

def generate_data(sq_cov, T):
    import warnings
    warnings.simplefilter('ignore')
    covmat = vec2mat(sq_cov)
    return np.random.multivariate_normal(np.zeros(covmat.shape[0]), covmat, (T))

def generate_template_data(covs, T):
    data = []
    for i in np.arange(covs.shape[0]):
        data.extend(generate_data(covs[i, :], T))
    return np.vstack(data)

def generate_subject_data(T, E, template):
    noise = np.random.randn(T*E, template.shape[1])
    return np.add(template, noise)




def r2z(r):
    """
    Function that calculates the Fisher z-transformation

    Parameters
    ----------
    r : int or ndarray
        Correlation value

    Returns
    ----------
    result : int or ndarray
        Fishers z transformed correlation value

    """
    return 0.5*(np.log(1+r) - np.log(1-r))


def z2r(z):
    """
    Function that calculates the inverse Fisher z-transformation

    Parameters
    ----------
    z : int or ndarray
        Fishers z transformed correlation value

    Returns
    ----------
    result : int or ndarray
        Correlation value

    """
    r = np.divide((np.exp(2*z) - 1), (np.exp(2*z) + 1))
    r[np.isnan(r)] = 0
    r[np.isinf(r)] = np.sign(r)[np.isinf(r)]
    return r

def isodd(x):
    return np.remainder(x, 2) == 1

def iseven(x):
    return np.remainder(x, 2) == 0

def mat2vec(m):
    """
    Function that converts correlation matrix to a vector

    Parameters
    ----------
    m : ndarray
        Correlation matix

    Returns
    ----------
    result : ndarray
        Vector

    """
    K = m.shape[0]
    V = int((((K ** 2) - K) / 2) + K)

    if m.ndim == 2:
        y = np.zeros(V)
        y[0:K] = np.diag(m)

        #force m to by symmetric
        m = np.triu(rmdiag(m))
        m += m.T

        y[K:] = sd.squareform(rmdiag(m))
    elif m.ndim == 3:
        T = m.shape[2]
        y = np.zeros([T, V])
        for t in np.arange(T):
            y[t, :] = mat2vec(np.squeeze(m[:, :, t]))
    else:
        raise ValueError('Input must be a 2 or 3 dimensional Numpy array')

    return y


def vec2mat(v):
    """
    Function that converts vector back to correlation matrix

    Parameters
    ----------
    result : ndarray
        Vector

    Returns
    ----------
    m : ndarray
        Correlation matix

    """
    if (v.ndim == 1) or (v.shape[0] == 1):
        x = int(0.5*(np.sqrt(8*len(v) + 1) - 1))
        return sd.squareform(v[x:]) + np.diag(v[0:x])
    elif v.ndim == 2:
        a = vec2mat(v[0, :])
        y = np.zeros([a.shape[0], a.shape[1], v.shape[0]])
        y[:, :, 0] = a
        for t in np.arange(1, v.shape[0]):
            y[:, :, t] = vec2mat(v[t, :])
    else:
        raise ValueError('Input must be a 1 or 2 dimensional Numpy array')

    return y

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

# set some defaults for plots
SMALL_SIZE = 18
MEDIUM_SIZE = 21
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_weights(weights, t=None, color='k', ax=None, xlab='Time (samples)', ylab='Weights', title=None, outfile=None):
    """
    Plot weights


    Parameters
    ----------
    weights : int or float or None
        Weights to plot

    t : int or float
        Time

    Returns
    ----------
    results : png
         Plot of weights

    """
    T = weights.shape[0]
    if t is None:
        t = np.round(T / 2)
    ts = np.arange(1, T + 1)

    if ax is None:
        ax = plt.gca()

    ax.plot(ts, weights[int(t), :], color=color)
    plt.xlim([1, T])

    if not (xlab is None):
        plt.xlabel(xlab)

    if not (ylab is None):
        plt.ylabel(ylab)

    if not (title is None):
        plt.title(title)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)

    else:
        plt.show()