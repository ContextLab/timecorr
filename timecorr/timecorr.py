from ._shared.helpers import isfc, gaussian_weights, gaussian_params

#TO DO:
# - modify isfc to accept a list OR numpy array (single-subject)
# - create gaussian_weights and gaussian_params
# - expose format_data function in hypertools
# - create a synthetic dataset (ideally write a function to do this)
# - write a smooth function that uses per-timepoint weights
# - create a sliding window function that can be used for cfun (that pads the result with nans)
# - debug everything and write unit tests

import numpy as np
import hypertools as hyp
import scipy.spatial.distance as sd

def timecorr(data, weight_function=gaussian_weights, weights_params=gaussian_params, mode="within", cfun=isfc):
    """
    Computes dynamics correlations in single-subject or multi-subject data.

    Inputs
    ----------
    data: numpy array, pandas dataframe, or a list of numpy arrays/dataframes
        Each numpy array (or dataframe) should have size timepoints by features.
        If a list of arrays are passed, there should be one array per subject.

    weights_function: a function of the form func(timepoints, t, params) where
        timepoints is a numpy array of times to evaluate the function at
        t is a specific timepoint (must be a member of timepoints)
        params is weights_params (described next)

        The function should return an array of the same size as timepoints,
        containing the per-timepoint weights.

        Default: gaussian_weights

    weights_params: used to pass parameters to the weights_params function. This
        can be specified in any format (e.g. a scalar, list, object, dictionary,
        etc.).

        Default: gaussian_variance

    mode: 'within' (default) or 'across'
        When mode is 'within' (default), the cfun operation (defined below) is
        applied independently to each data array.  The result is a list (of the
        same length as data) containing the outputs of the cfun operation for
        each array.

        When mode is 'across', the cfun operation is applied to the full data
        list simultaneously.  This is useful for across-subject analyses or
        other scenarios where the relevant function needs to account for all
        data simultaneously.

        If data is a numpy array (rather than a list), mode is ignored (both
        'within' and 'across' mode return the output of cfun applied to the
        single data array.

    cfunc: function to apply to the data array(s)
        This function should be of the form
        func(data, weights)

        The function should support data as a numpy array or list of numpy
        arrays.  When a list of numpy arrays is passed, the function should
        apply the "across subjects" version of the analysis.  When a single
        numpy array is passed, the function should apply the "within subjects"
        version of the analysis.

        weights is a numpy array with per-timepoint weights

        The function should return a single numpy array with 1 row and an
        arbitrary number of columns (the number of columns may be determined by
        the function).

        Default: A continuous verison of Inter-Subject Functional Connectivity
        (Simony et al. 2017).  If only one data array is passed (rather than a
        list), the default cfun returns the moment-by-moment correlations for
        that array.  (Reference: http://www.nature.com/articles/ncomms12141)

    Outputs
    ----------
    corrmats: moment-by-moment correlations

        If mode is 'within', corrmats is a list of the same length as data,
        containing the outputs of cfun for each data array.

        If mode is 'across', corrmats is an array with number-of-timepoints rows
        and an arbitrary number of columns (determined by cfun).
    """

    def get_corrs(data, weights):
        corrs = list(map(lambda w: cfun(data, w), weights))
        return np.vstack(corrs)

    data = hyp.tools.format_data(data)

    if type(data) == list:
        T = data[0].shape[0]
    else:
        T = data.shape[0]

    timepoints = np.arange(T)
    weights = list(map(lambda t: weights_function(timepoints, t, weights_params), timepoints))

    if (mode == 'across') or (type(data) != list) or (len(data) == 1):
        return get_corrs(data, weights)
    elif mode == 'within': #data must also be a list
        return list(map(lambda d: get_corrs(d, weights), data))


def levelup(data, mode='within', weight_function=gaussian_weights, weights_params=gaussian_params, cfun=isfc, reduce='IncrementalPCA'):
    """
    Convenience function that performs two steps:
    1.) Uses timecorr to compute within-subject moment-by-moment correlations
    2.) Uses dimensinality reduction to project the output onto the same number
    of dimensions as the original data.

    Inputs
    ----------
    data: numpy array, pandas dataframe, or a list of numpy arrays/dataframes
        Each numpy array (or dataframe) should have size timepoints by features.
        If a list of arrays are passed, there should be one array per subject.

    mode: 'within' (default) or 'across'
        When mode is 'within' (default), the cfun operation (defined below) is
        applied independently to each data array.  The result is a list (of the
        same length as data) containing the outputs of the cfun operation for
        each array.

        When mode is 'across', the cfun operation is applied to the full data
        list simultaneously.  This is useful for across-subject analyses or
        other scenarios where the relevant function needs to account for all
        data simultaneously.

        If data is a numpy array (rather than a list), mode is ignored (both
        'within' and 'across' mode return the output of cfun applied to the
        single data array.

    weights_function: a function of the form func(timepoints, t, params) where
        timepoints is a numpy array of times to evaluate the function at
        t is a specific timepoint (must be a member of timepoints)
        params is weights_params (described next)

        The function should return an array of the same size as timepoints,
        containing the per-timepoint weights.

        Default: gaussian_weights

    weights_params: used to pass parameters to the weights_params function. This
        can be specified in any format (e.g. a scalar, list, object, dictionary,
        etc.).

        Default: gaussian_variance

    cfunc: function to apply to the data array(s)
        This function should be of the form
        func(data, weights)

        The function should support data as a numpy array or list of numpy
        arrays.  When a list of numpy arrays is passed, the function should
        apply the "across subjects" version of the analysis.  When a single
        numpy array is passed, the function should apply the "within subjects"
        version of the analysis.

        weights is a numpy array with per-timepoint weights

        The function should return a single numpy array with 1 row and an
        arbitrary number of columns (the number of columns may be determined by
        the function).

        Default: A continuous verison of Inter-Subject Functional Connectivity
        (Simony et al. 2017).  If only one data array is passed (rather than a
        list), the default cfun returns the moment-by-moment correlations for
        that array.

    reduce: function to use for dimensionality reduction.  All hypertools and
        scikit-learn functions are supported: PCA, IncrementalPCA, SparsePCA,
        MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
        DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
        SpectralEmbedding, LocallyLinearEmbedding, and MDS.

        Can be passed as a string, but for finer control of the model
        parameters, pass as a dictionary, e.g.
        reduce={‘model’ : ‘PCA’, ‘params’ : {‘whiten’ : True}}.

        See scikit-learn specific model docs for details on parameters supported
        for each model.

    Outputs
    ----------
    A single data array or list of arrays, of the same size(s) as the original
    dataset(s)
    """

    data = hyp.tools.format_data(data)
    if type(data) == list:
        T = data[0].shape[0]
    else:
        T = data.shape[0]

    corrs = timecorr(data, weight_function=weight_function, weights_params=weights_params, mode="within", cfun=isfc)
    return hyp.reduce(corrs, reduce=reduce, ndims=T)

#JEREMY NOTE: need to rewrite everything below this point (potentially copy from brain-dynamics repo)
def decode(data, var=3, nfolds=2, cfun=isfc):
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
    subj_indices = list(range(subj_num))
    accuracy = 0
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs = timecorr([data[z] for z in subj_indices[:(np.divide(subj_num,2))]], var=var, cfun=cfun, mode="across")
        out_fold_corrs = timecorr([data[z] for z in subj_indices[(np.divide(subj_num,2)):]], var=var, cfun=cfun, mode="across")
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
    subj_indices = list(range(subj_num))
    accuracy_tc, accuracy_sw = 0,0
    tc_diag, sw_diag = np.zeros((time_len-10)*2),np.zeros((time_len-10)*2)
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs_tc = timecorr([data[z] for z in subj_indices[:(np.divide(subj_num,2))]], var=8, cfun=isfc, mode="across")[5:-5]
        out_fold_corrs_tc = timecorr([data[z] for z in subj_indices[(np.divide(subj_num,2)):]], var=8, cfun=isfc, mode="across")[5:-5]
        in_fold_corrs_sw = timecorr([data[z] for z in subj_indices[:(np.divide(subj_num,2))]], var=11, cfun=sliding_window_isfc, mode="across")
        out_fold_corrs_sw = timecorr([data[z] for z in subj_indices[(np.divide(subj_num,2)):]], var=11, cfun=sliding_window_isfc, mode="across")

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

    subj_num=len(data)
    subj_indices = list(range(subj_num))
    accuracy = 0
    for i in range(nfolds):
        shuffle(subj_indices)
        in_fold_corrs = np.mean(data[subj_indices[:(np.divide(subj_num,2))]],0)
        out_fold_corrs = np.mean(data[subj_indices[(np.divide(subj_num,2)):]],0)
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
