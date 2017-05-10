<<<<<<< HEAD
##Packages##
import sys
import numpy as np
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import squareform
from corr_helper import ISFC, correlation_calculation_single

# the data file is expected to be of dimensions [subject number, time length, activations length]
# and converted to dimensions [subject number, activations length, time length]
def timecorr(activations, gaussian_variance, estimation_range=3, mode = "within", coefficients = None):
    # if activations for only one subject is input, then calculate activations correlation for single subject
    if len(activations)==1:
        return correlation_calculation_single(activations[0].T, gaussian_variance, estimation_range)

    # if activations for multiple subjects is input, then two options are available
    else:
        activations = np.swapaxes(np.array(activations),1,2)
        subject_num, activations_len, time_len = activations.shape
=======
import numpy as np
import hypertools as hyp
from copy import copy
import scipy.spatial.distance as sd
#import scipy.linalg as la
from _shared.helpers import isfc, wcorr


def timecorr(x, var=100, wlen=3, mode="within", cfun=isfc):
    if (not type(x) == list) and (len(x.shape)==2):
        return wcorr(x.T, var, wlen)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        x = np.array(x)
        x = np.swapaxes(x, 1, 2)
>>>>>>> ce8dd56dc805598031aadeb926bdd672e217a06d

        # Calculate correlation for activations within each subject
        if mode=="within":
<<<<<<< HEAD
            subject_num = len(activations)
            activations_len, time_len= activations[0].shape
            result = np.zeros([subject_num, time_len,(activations_len * (activations_len-1) / 2)])
            for subject in range(subject_num):
                result[subject] = correlation_calculation_single(activations[subject], gaussian_variance, estimation_range,coefficients)
            return result

        # Calculate ISFC, average correlation between activations across subjects
        else:
            return ISFC(activations, gaussian_variance, estimation_range,coefficients)


if __name__ == "__main__":
    filename, gaussian_variance, estimation_range, mode = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
    data = np.load(filename)
    activations = data['arr_2']
    true_covariances = data['arr_0']
    true_covariances1 = data['arr_1']
    activations_len, time_len = activations.shape
    covariances_vector = timecorr(np.tile(activations.T,[10,1,1]), int(gaussian_variance), int(estimation_range), mode)
    true_covariances_vector = squareform(true_covariances,checks=False)
    true_covariances_vector1 = squareform(true_covariances1,checks=False)
    Y = np.array([pearsonr(covariances_vector[i,],true_covariances_vector)[0] for i in range(time_len)])
    Y1 = np.array([pearsonr(covariances_vector[i,],true_covariances_vector1)[0] for i in range(time_len)])
    plt.plot(range(time_len),Y,'b-')
    plt.plot(range(time_len),Y1,'r-')
    plt.show()
=======
            S = len(x)
            V, T = x[0].shape
            result = np.zeros([S, T, (V * (V - 1) / 2)])
            for i in range(S):
                result[i] = wcorr(x[i], var, wlen)
            return result
        elif mode=="across":
            return cfun(x, var, wlen)
        else:
            raise NameError('Mode unknown or not supported: ' + mode)


def levelup(x0, var=100, wlen=3):
    if type(x) == list:
        V = np.max(np.array(map(lambda x: x.shape[1], x0)))
    else:
        V = x0.shape[1]
    c = timecorr(x0, var=var, wlen=wlen, mode="within")
    return hyp.tools.reduce(c, ndims=V)


def timepoint_decoder(data, var=100, wlen=3, nfolds=2, cfun=isfc):
    """
    :param data: a number-of-observations by number-of-features matrix
    :param var: Gaussian variance of kernel for computing timepoint correlations
    :param wlen: length of envelope for estimating local covariance structure
    :param nfolds: number of cross-validation folds (train using out-of-fold data; test using in-fold data)
    :param cfun: function for transforming the group data (default: isfc)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between the decoded and actual window numbers,
                expressed as a percentage of the total number of windows
    """
    def get_xval_assignments(ndata, nfolds):
        group_assignments = np.zeros(ndata)
        groupsize = int(np.ceil(ndata / nfolds))
        
        # group assignments
        for i in range(1, nfolds):
            inds = np.arange(i * groupsize, np.min([(i + 1) * groupsize, ndata]))
            group_assignments[inds] = i
        np.random.shuffle(group_assignments)
        return group_assignments

    group_assignments = get_xval_assignments(len(data), nfolds)

    # fill in results
    results_template = {'rank': 0, 'accuracy': 0, 'error': 0}
    results = copy(results_template)
    for i in range(0, nfolds):
        in_fold_corrs = timecorr(data[group_assignments == i], var=var, wlen=wlen, cfun=cfun)
        out_fold_corrs = timecorr(data[group_assignments != i], var=var, wlen=wlen, cfun=cfun)
        corrs = 1 - sd.cdist(in_fold_corrs, out_fold_corrs, 'correlation')

        next_results = copy(results_template)

        #timepoint_dists = la.toeplitz(np.arange(corrs.shape[0]))
        for t in range(0, corrs.shape[0]):
            include_inds = np.unique(np.append(np.where(timepoint_dists[t, :] > 0), np.array(t)))
            #include_inds = np.unique(np.append(np.where(timepoint_dists[t, :] > windowsize), np.array(t))) # more liberal test

            decoded_inds = include_inds[np.where(corrs[t, include_inds] == np.max(corrs[t, include_inds]))]
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
>>>>>>> ce8dd56dc805598031aadeb926bdd672e217a06d
