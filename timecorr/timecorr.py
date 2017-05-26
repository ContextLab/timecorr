import numpy as np
import hypertools as hyp
from copy import copy
from random import shuffle
import scipy.spatial.distance as sd
from _shared.helpers import isfc, wcorr, sliding_window, sliding_window_isfc


def timecorr(x, mode="within", cfun=isfc):
    """
    Performs the timecorr operation on a brain dynamics dataset

    This function has three modes of operation that calculates the temporal
    "within" or "across" correlation within a dataset containing a time series
    of brain activations. See parameter "mode" for detailed explanation

    Parameters
    ----------
    x: Numpy matrix or a list of Numpy matrices
        When calculating temporal brain activation correlation for a single
        subject, x should be a single Numpy matrix of dimensions T x V,
        where T represents the number of timepoints and V represents the
        number of voxels in the dataset. When calculating temporal brain
        activation correlation for multiple subjects, x should be a list of
        Numpy matrices, each containing the brain activations for a single
        subject. The Numpy matrix for each subject should be of dimensions
        T x V, where T represents the number of timepoints and V represents
        the number of voxels in the dataset

    mode: "across" or "within", default to "within"
        When x is a single Numpy matrix, this function assumes x
        only contains information for a single subject and will default to
        "within" mode, which calculates the correlation between subject
        voxel activations at each timepoints
        When x is a list of Numpy matrices, the user may choose "across" or
        "within" mode of operation:
        In the "within" mode of operation, this function calculates the
        temporal voxel activation correlation for each subject independently.
        In the "across" mode of operation, this function calculates temporal
        voxel activation correlation between the activations of each subject
        and the mean activation of all other subjects, and then returning
        the mean correlation across all subjects. The specific process is
        called inter-subject functional connectivity (ISFC) and the details are
        described in this paper: https://www.nature.com/articles/ncomms12141

    cfunc: isfc, defaults to isfc
        This parameter specifies the type of operation for the "across" mode
        of this function. Right now only ISFC is available.

    Returns
    ----------
    timecorr_correlations: Numpy matrix or a list of Numpy matrices
        If x is a single Numpy matrix, this function will return a T x (V^2-V)/2
        dimensional matrix containing the square form of the voxel activation
        correlation at each timepoint.
        If x is a list of Numpy matrices and mode="within", this function will
        return a list of T x (V^2-V)/2 dimensional Numpy matrices containing
        the square form of the voxel activation correlation at each timepoint
        for each subject.
        If x is a list of Numpy matrices and mode="across", this function will
        return a T x (V^2-V)/2 dimensional matrix containing the square form
        of the ISFC voxel activation correlation at each timepoint.


    """
    if (not type(x) == list) and (len(x.shape)==2):
        var = min(1000, x.shape[0])
        return wcorr(x.T, var)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        x = np.array(x)
        x = np.swapaxes(x, 1, 2)
        # Calculate correlation for activations within each subject
        S, V, T = x.shape
        var = min(T, 1000)
        if mode=="within":
            result = []
            for i in range(S):
                result.append(wcorr(x[i], var))
            return result
        elif mode=="across":
            return cfun(x, var)
        else:
            raise NameError('Mode unknown or not supported: ' + mode)


def levelup(x0, mode = "within"):
    """
    This function applies timecorr function on a brain activation dataset and
    uses PCA to reduce the output to the original dimensions as representation
    of the subject's brain activity at a higher level.

    Parameters
    ----------
    x0: Numpy matrix or a list of Numpy matrices
        When performing the level up function for a single subject, x should be
        a single Numpy matrix of dimensions T x V, where T represents the number
        of timepoints and V represents the number of voxels in the dataset.
        When performing the level up function for multiple subjects, x should
        be a list of Numpy matrices, each containing the brain activations for
        a single subject. The Numpy matrix for each subject should be of
        dimensions T x V, where T represents the number of timepoints and V
        represents the number of voxels in the dataset

    mode: "across" or "within", default to "within"
        When x is a single Numpy matrix, this function assumes x
        only contains information for a single subject and will default to
        "within" mode, which performs the levelup operation for a single
        subject.
        When x is a list of Numpy matrices, the user may choose "across" or
        "within" mode of operation:
        In the "within" mode of operation, this function performs the levelup
        operation for each subject independently.
        In the "across" mode of operation, this function applies the "across"
        mode of timecorr to the subject data to obtain inter-subject functional
        connectivity(ISFC). Then, the function reduces the ISFC to the a matrix
        of dimensions T x V, where T represents the number of timepoints and
        V represents the number of voxels for each subject in the original
        dataset. The specific process of ISFC and the details are
        described in this paper: https://www.nature.com/articles/ncomms12141

    Returns
    ----------
    Results: Numpy matrix or a list of Numpy matrices
        If x is a single Numpy matrix, this function will return a T x V
        dimensional matrix containing the representation of the subject's brain
        patterns at a higher level.
        If x is a list of Numpy matrices and mode="within", this function will
        return a list of T x V dimensional Numpy matrices containing
        the representations of each subject's brain patterns at a higher level.
        If x is a list of Numpy matrices and mode="across", this function will
        return a T x V dimensional matrix containing the representation of the
        ISFC patterns across the subjects at a higher level.

    """
    if type(x0) == list or len(x0.shape)>2:
        V = np.max(np.array(map(lambda x: x.shape[1], x0)))
        T = np.min(np.array(map(lambda x: x.shape[0], x0)))
    else:
        T, V = x0.shape
    var = min(T, 1000)
    c = timecorr(x0, mode=mode)
    return hyp.tools.reduce(c, ndims=V)


def timepoint_decoder(data, nfolds=2, cfun=isfc):
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
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs = timecorr([data[z] for z in subj_indices[:(subj_num/2)]], cfun=cfun, mode="across")
        out_fold_corrs = timecorr([data[z] for z in subj_indices[(subj_num/2):]], cfun=cfun, mode="across")
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
        results['error'] += next_results['error']
        results['accuracy'] += next_results['accuracy']
        results['rank'] += next_results['rank']

    results['error'] /= nfolds
    results['accuracy'] /= nfolds
    results['rank'] /= nfolds
    print(results)
    return results
