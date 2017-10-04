import numpy as np
from copy import copy
from random import shuffle
import scipy.spatial.distance as sd
from _shared.helpers import isfc, wcorr, sliding_window, sliding_window_isfc, timecorr_smoothing, sliding_window_smoothing
from sklearn import decomposition
np.seterr(all='ignore')

def smoothing(data, varr=5, mode = "timecorr"):
    if mode == "timecorr":
        cfunc = timecorr_smoothing
    else:
        cfunc = sliding_window_smoothing
    if (not type(data) == list) and (len(data.shape)==2):
        return cfunc(data.T, varr)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        data = np.array(data)
        data = np.swapaxes(data, 1, 2)
        # Calculate correlation for activations within each subject
        S, V, T = data.shape
        result = []
        for i in range(S):
            result.append(cfunc(data[i], varr))
        return result


def timecorr(data, var=None, mode="within", cfun=isfc):
    """
    Performs the timecorr operation on a brain dynamics dataset

    This function has three modes of operation that calculates the temporal "within" or "across" correlation within a dataset containing a time series of brain activations. See parameter "mode" for detailed explanation

    Parameters
    ----------
    data: Numpy matrix or a list of Numpy matrices
        When calculating temporal brain activation correlation for a single subject, x should be a single Numpy matrix of dimensions T x V, where T represents the number of timepoints and V represents the number of voxels in the dataset. When calculating temporal brain
        activation correlation for multiple subjects, x should be a list of Numpy matrices, each containing the brain activations for a single subject. The Numpy matrix for each subject should be of dimensions T x V, where T represents the number of timepoints and V represents
        the number of voxels in the dataset

    var: int, defaults to the minimum between time length and 1000

        The variance of the Gaussian distribution used to represent the influence of neighboring timepoints on calculation of correlation at the current timepoint in timecorr

    mode: "across" or "within", default to "within"

        When x is a single Numpy matrix, this function assumes x only contains information for a single subject and will default to "within" mode, which calculates the correlation between subject
        voxel activations at each timepoints When x is a list of Numpy matrices, the user may choose "across" or "within" mode of operation:

        In the "within" mode of operation, this function calculates the temporal voxel activation correlation for each subject independently.

        In the "across" mode of operation, this function calculates temporal voxel activation correlation between the activations of each subject and the mean activation of all other subjects, and then returning the mean correlation across all subjects. The specific process is
        called inter-subject functional connectivity (ISFC) and the details are described in this paper: https://www.nature.com/articles/ncomms12141

    cfunc: isfc, defaults to isfc

        This parameter specifies the type of operation for the "across" mode of this function. Right now only ISFC is available.

    Returns
    ----------
    timecorr_correlations: Numpy matrix or a list of Numpy matrices

        If x is a single Numpy matrix, this function will return a T x (V^2-V)/2 dimensional matrix containing the square form of the voxel activation correlation at each timepoint.
        If x is a list of Numpy matrices and mode="within", this function will return a list of T x (V^2-V)/2 dimensional Numpy matrices containing the square form of the voxel activation correlation at each timepoint for each subject.

        If x is a list of Numpy matrices and mode="across", this function will return a T x (V^2-V)/2 dimensional matrix containing the square form of the ISFC voxel activation correlation at each timepoint.


    """
    if (not type(data) == list) and (len(data.shape)==2):
        return wcorr(data.T, var=var)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        data = np.array(data)
        data = np.swapaxes(data, 1, 2)
        # Calculate correlation for activations within each subject
        S, V, T = data.shape
        if mode=="within":
            result = []
            for i in range(S):
                result.append(wcorr(data[i], var=var))
            return result
        elif mode=="across":
            return cfun(data, var=var)
        else:
            raise NameError('Mode unknown or not supported: ' + mode)


def levelup(data, var=None, mode = "within"):
    """
    This function applies timecorr function on a brain activation dataset and uses PCA to reduce the output to the original dimensions as representation of the subject's brain activity at a higher level.

    Parameters
    ----------
    data: Numpy matrix or a list of Numpy matrices

        When performing the level up function for a single subject, x should be a single Numpy matrix of dimensions T x V, where T represents the number of timepoints and V represents the number of voxels in the dataset.

        When performing the level up function for multiple subjects, x should be a list of Numpy matrices, each containing the brain activations for a single subject. The Numpy matrix for each subject should be of dimensions T x V, where T represents the number of timepoints and V
        represents the number of voxels in the dataset

    var: int, defaults to the minimum between time length and 1000

        The variance of the Gaussian distribution used to represent the influence of neighboring timepoints on calculation of correlation at the current timepoint in timecorr

    mode: "across" or "within", default to "within"

        When x is a single Numpy matrix, this function assumes x only contains information for a single subject and will default to "within" mode, which performs the levelup operation for a single
        subject.

        When x is a list of Numpy matrices, the user may choose "across" or "within" mode of operation:
        In the "within" mode of operation, this function performs the levelup operation for each subject independently.

        In the "across" mode of operation, this function applies the "across" mode of timecorr to the subject data to obtain inter-subject functional connectivity(ISFC). Then, the function reduces the ISFC to the a matrix of dimensions T x V, where T represents the number of timepoints and
        V represents the number of voxels for each subject in the original dataset. The specific process of ISFC and the details are described in this paper: https://www.nature.com/articles/ncomms12141

    Returns
    ----------
    Results: Numpy matrix or a list of Numpy matrices

        If x is a single Numpy matrix, this function will return a T x V dimensional matrix containing the representation of the subject's brain patterns at a higher level.

        If x is a list of Numpy matrices and mode="within", this function will return a list of T x V dimensional Numpy matrices containing the representations of each subject's brain patterns at a higher level.

        If x is a list of Numpy matrices and mode="across", this function will return a T x V dimensional matrix containing the representation of the ISFC patterns across the subjects at a higher level.

    """
    if type(data) == list or len(data.shape)>2:
        V = np.max(np.array(map(lambda x: x.shape[1], data)))
        T = np.min(np.array(map(lambda x: x.shape[0], data)))
    else:
        T, V = data.shape
    ipca = decomposition.IncrementalPCA(n_components=V, batch_size=V)
    c = timecorr(data, var = var, mode=mode)
    if mode == "within" and (type(data) == list or len(data.shape)>2):
        stacked_data = np.concatenate(c,0)
        pca_reduced = ipca.fit_transform(stacked_data)
        result = []
        for subject in range(len(c)):
            result.append(pca_reduced[subject*T:(subject+1)*T])
    else:
        result = ipca.fit_transform(c)
    return result


def decode(data, var=None, nfolds=2, cfun=isfc):
    """
    This function applies decoding analysis across multi-subject fMRI dataset to find the decoding accuracy of the dynamic correlations, a parameter describing timepoint similarity between subjects. The process is described in more detail in the following paper: http://biorxiv.org/content/early/2017/02/07/106690

    Parameters
    ----------
    data: a list of Numpy matrices

        When calculating temporal brain activation correlation for multiple subjects, x should be a list of Numpy matrices, each containing the brain activations for a single subject. The Numpy matrix for each subject should be of dimensions T x V, where T represents the number of timepoints and V represents the number of voxels in the dataset

    var: int, defaults to the minimum between time length and 1000

        The variance of the Gaussian distribution used to represent the influence of neighboring timepoints on calculation of correlation at the current timepoint in timecorr

    nfolds: int, defaults to 2

        The number of decoding analysis repetitions to perform to obtin stability

    cfunc: isfc, defaults to isfc

        This parameter specifies the type of operation for to calculate the correlation matrix. Currently, only ISFC is available.

    Returns
    ----------
    Results: float

        The decoding accurcy of the dynamic correlations of the input fRMI matrix
    """
    subj_num=len(data)
    subj_indices = range(subj_num)
    accuracy = 0
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs = timecorr([data[z] for z in subj_indices[:(subj_num/2)]], var=var, cfun=cfun, mode="across")
        out_fold_corrs = timecorr([data[z] for z in subj_indices[(subj_num/2):]], var=var, cfun=cfun, mode="across")
        corrs = 1 - sd.cdist(in_fold_corrs, out_fold_corrs, 'correlation')
        accuracy_temp = 0

        #timepoint_dists = la.toeplitz(np.arange(corrs.shape[0]))
        for t in range(0, corrs.shape[0]):
            include_inds = np.arange(corrs.shape[0])
            decoded_inds = include_inds[np.where(corrs[t, include_inds] == np.max(corrs[t, include_inds]))]
            accuracy_temp += np.mean(decoded_inds == np.array(t))

        accuracy_temp /= corrs.shape[0]
        accuracy += accuracy_temp

    accuracy /= nfolds
    return accuracy

def decode_comp(data, var=None, nfolds=2, cfun=isfc):
    """
    This function applies decoding analysis across multi-subject fMRI dataset to find the decoding accuracy of the dynamic correlations, a parameter describing timepoint similarity between subjects. The process is described in more detail in the following paper: http://biorxiv.org/content/early/2017/02/07/106690

    Parameters
    ----------
    data: a list of Numpy matrices

        When calculating temporal brain activation correlation for multiple subjects, x should be a list of Numpy matrices, each containing the brain activations for a single subject. The Numpy matrix for each subject should be of dimensions T x V, where T represents the number of timepoints and V represents the number of voxels in the dataset

    var: int, defaults to the minimum between time length and 1000

        The variance of the Gaussian distribution used to represent the influence of neighboring timepoints on calculation of correlation at the current timepoint in timecorr

    nfolds: int, defaults to 2

        The number of decoding analysis repetitions to perform to obtin stability

    cfunc: isfc, defaults to isfc

        This parameter specifies the type of operation for to calculate the correlation matrix. Currently, only ISFC is available.

    Returns
    ----------
    Results: float

        The decoding accurcy of the dynamic correlations of the input fRMI matrix
    """
    subj_num=len(data)
    time_len = len(data[0])
    subj_indices = range(subj_num)
    accuracy_tc, accuracy_sw = 0,0
    tc_diag, sw_diag = np.zeros((time_len-10)*2),np.zeros((time_len-10)*2)
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs_tc = timecorr([data[z] for z in subj_indices[:(subj_num/2)]], var=5, cfun=isfc, mode="across")[5:-5]
        out_fold_corrs_tc = timecorr([data[z] for z in subj_indices[(subj_num/2):]], var=5, cfun=isfc, mode="across")[5:-5]
        in_fold_corrs_sw = timecorr([data[z] for z in subj_indices[:(subj_num/2)]], var=11, cfun=sliding_window_isfc, mode="across")
        out_fold_corrs_sw = timecorr([data[z] for z in subj_indices[(subj_num/2):]], var=11, cfun=sliding_window_isfc, mode="across")

        corrs_tc = 1 - sd.cdist(in_fold_corrs_tc, out_fold_corrs_tc, 'correlation')
        corrs_sw = 1 - sd.cdist(in_fold_corrs_sw, out_fold_corrs_sw, 'correlation')
        time_len =  corrs_tc.shape[0]
        accuracy_temp_tc = 0
        accuracy_temp_sw = 0
        tc_diag[i*time_len:((i+1)*time_len)] = np.diagonal(corrs_tc)
        sw_diag[i*time_len:((i+1)*time_len)] = np.diagonal(corrs_sw)

        #timepoint_dists = la.toeplitz(np.arange(corrs.shape[0]))
        for t in range(0, time_len):
            include_inds = np.arange(time_len)
            decoded_inds_tc = include_inds[np.where(corrs_tc[t, include_inds] == np.max(corrs_tc[t, include_inds]))]
            accuracy_temp_tc += np.mean(decoded_inds_tc == np.array(t))
            decoded_inds_sw = include_inds[np.where(corrs_sw[t, include_inds] == np.max(corrs_sw[t, include_inds]))]
            accuracy_temp_sw += np.mean(decoded_inds_sw == np.array(t))

        accuracy_temp_tc /= time_len
        accuracy_temp_sw /= time_len
        accuracy_tc += accuracy_temp_tc
        accuracy_sw += accuracy_temp_sw

    accuracy_tc /= nfolds
    accuracy_sw /= nfolds
    trace_tc,trace_tc_std = np.mean(tc_diag), np.std(tc_diag)/nfolds/time_len
    trace_sw,trace_sw_std = np.mean(sw_diag), np.std(sw_diag)/nfolds/time_len
    return np.array([accuracy_tc, accuracy_sw, trace_tc, trace_tc_std, trace_sw, trace_sw_std])


def decode_raw_data(data, nfolds=2, cfun=isfc):
    """
    This function applies decoding analysis across multi-subject fMRI dataset to find the decoding accuracy of the dataset, a parameter describing timepoint similarity between subjects. The process is described in more detail in the following paper: http://biorxiv.org/content/early/2017/02/07/106690

    Parameters
    ----------
    data: a list of Numpy matrices

        When calculating temporal brain activation correlation for multiple subjects, x should be a list of Numpy matrices, each containing the brain activations for a single subject. The Numpy matrix for each subject should be of dimensions T x V, where T represents the number of timepoints and V represents the number of voxels in the dataset

    nfolds: int, defaults to 2

        The number of decoding analysis repetitions to perform to obtin stability

    cfunc: isfc, defaults to isfc

        This parameter specifies the type of operation for to calculate the correlation matrix. Currently, only ISFC is available.

    Returns
    ----------
    Results: float

        The decoding accurcy of the input fRMI matrix
    """
    # data = smoothing(data)
    subj_num=len(data)
    subj_indices = range(subj_num)
    accuracy = 0
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs = np.mean(data[subj_indices[:(subj_num/2)]],0)
        out_fold_corrs = np.mean(data[subj_indices[(subj_num/2):]],0)
        corrs = 1 - sd.cdist(in_fold_corrs, out_fold_corrs, 'correlation')
        accuracy_temp = 0

        #timepoint_dists = la.toeplitz(np.arange(corrs.shape[0]))
        for t in range(0, corrs.shape[0]):
            include_inds = np.arange(corrs.shape[0])
            decoded_inds = include_inds[np.where(corrs[t, include_inds] == np.max(corrs[t, include_inds]))]
            accuracy_temp += np.mean(decoded_inds == np.array(t))
        accuracy_temp /= corrs.shape[0]
        accuracy += accuracy_temp
    accuracy /= nfolds
    return accuracy

def decode_pair(arr1, arr2, cfun=isfc):
    """
    This function finds the decoding accuracy between two given matrices

    Parameters
    ----------
    arr1: a Numpy matrices

        The array to decode

    arr2: a Numpy matrices

        The array to decode with, must have same dimension as the first numpy matrix

    cfunc: isfc, defaults to isfc

        This parameter specifies the type of operation for to calculate the correlation matrix. Currently, only ISFC is available.

    Returns
    ----------
    Results: float

        The decoding accurcy of the input fRMI matrix
    """
    accuracy = 0
    corrs = 1 - sd.cdist(arr1, arr2, 'correlation')

    for t in range(0, corrs.shape[0]):
        include_inds = np.arange(corrs.shape[0])
        decoded_inds = include_inds[np.where(corrs[t, include_inds] == np.max(corrs[t, include_inds]))]
        accuracy += np.mean(decoded_inds == np.array(t))

    accuracy /= corrs.shape[0]
    return accuracy
