import numpy as np
import hypertools as hyp
from copy import copy
from random import shuffle
import scipy.spatial.distance as sd
from _shared.helpers import isfc, wcorr


def timecorr(x, var=1000, mode="within", cfun=isfc):
    if (not type(x) == list) and (len(x.shape)==2):
        return wcorr(x.T, var)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        x = np.array(x)
        x = np.swapaxes(x, 1, 2)
        # Calculate correlation for activations within each subject
        if mode=="within":
            S, V, T = x.shape
            result = np.zeros([S, T, (V * (V - 1) / 2)])
            for i in range(S):
                result[i] = wcorr(x[i], var)
            return result
        elif mode=="across":
            return cfun(x, var)
        else:
            raise NameError('Mode unknown or not supported: ' + mode)


def levelup(x0, var=1000):
    if type(x0) == list:
        V = np.max(np.array(map(lambda x: x.shape[1], x0)))
    else:
        V = x0.shape[1]
    c = timecorr(x0, var=var, mode="within")
    return hyp.tools.reduce(c, ndims=V)


def timepoint_decoder(data, var=1000, nfolds=2, cfun=isfc):
    """
    :param data: a number-of-observations by number-of-features matrix
    :param var: Gaussian variance of kernel for computing timepoint correlations
    :param nfolds: number of cross-validation folds (train using out-of-fold data; test using in-fold data)
    :param cfun: function for transforming the group data (default: isfc)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between the decoded and actual window numbers,
                expressed as a percentage of the total number of windows
    """
    subj_num=len(data)
    subj_indices = range(subj_num)
    results_template = {'rank': 0, 'accuracy': 0, 'error': 0}
    results = copy(results_template)
    for i in range(0, nfolds):
        shuffle(subj_indices)
        in_fold_corrs = timecorr([data[z] for z in subj_indices[:(subj_num/2)]], var=var, cfun=cfun, mode="across")
        out_fold_corrs = timecorr([data[z] for z in subj_indices[(subj_num/2):]], var=var, cfun=cfun, mode="across")
        corrs = 1 - sd.cdist(in_fold_corrs, out_fold_corrs, 'correlation')

        next_results = copy(results_template)

        #timepoint_dists = la.toeplitz(np.arange(corrs.shape[0]))
        for t in range(0, corrs.shape[0]):

            include_inds = np.arange(corrs.shape[0])
            # include_inds = np.unique(np.append(np.where(timepoint_dists[t, :] > 0), np.array(t)))
            #include_inds = np.unique(np.append(np.where(timepoint_dists[t, :] > windowsize), np.array(t))) # more liberal test

            decoded_inds = include_inds[np.where(corrs[t, include_inds] == np.max(corrs[t, include_inds]))]
            next_results['error'] += np.mean(np.abs(decoded_inds - np.array(t)))/(corrs.shape[0] - 1)
            next_results['accuracy'] += np.mean(decoded_inds == np.array(t))
            next_results['rank'] += np.mean(map((lambda x: int(x)), (corrs[t, :] <= corrs[t, t])))

        next_results['error'] /= corrs.shape[0]
        next_results['accuracy'] /= corrs.shape[0]
        next_results['rank'] /= corrs.shape[0]
        print(next_results)

        results['error'] += next_results['error']
        results['accuracy'] += next_results['accuracy']
        results['rank'] += next_results['rank']

    results['error'] /= nfolds
    results['accuracy'] /= nfolds
    results['rank'] /= nfolds
    return results
