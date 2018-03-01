# coding: utf-8

from .helpers import isfc, gaussian_weights, gaussian_params
import hypertools as hyp

# TO DO (JEREMY):
# - create a synthetic dataset (ideally write a function to do this)
# - write a smooth function that uses per-timepoint weights
# - create a sliding window function(s) for ISFC, WISFC, and SMOOTH that can be
#   used for cfun (that pads the result with nans)
# - debug everything and write unit tests
#
# TO DO (EMILY):
# - update documentation...a lot of stuff is now out of date
# - figure out (with Andy's help?) how to make a Sphinx website for the TimeCorr
#   API (some of this might have been done by Tom, but it'll now need to be updated)


def timecorr(data, weights_function=gaussian_weights,
             weights_params=gaussian_params, mode="within", cfun=isfc):
    """
    Computes dynamics correlations in single-subject or multi-subject data.

    Inputs
    ----------
    data: numpy array, pandas dataframe, or a list of numpy arrays/dataframes
        Each numpy array (or dataframe) should have size timepoints by features.
        If a list of arrays are passed, there should be one array per subject.

    weights_function: a function of the form func(T, params) where
        T is a non-negative integer specifying the number of timepoints to consider.

        The function should return a T by T array containing the timepoint-specific
        weights for each consecutive time point from 0 to T (not including T).

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
    data = hyp.tools.format_data(data)

    if type(data) == list:
        T = data[0].shape[0]
    else:
        T = data.shape[0]

    weights = weights_function(T, weights_params)

    if (mode == 'across') or (type(data) != list) or (len(data) == 1):
        return cfun(data, weights)
    elif mode == 'within':
        return list(map(lambda x: cfun(x, weights), data))
    else:
        print('Unknown mode: ' + mode)
        raise


def levelup(data, mode='within', weight_function=gaussian_weights,
            weights_params=gaussian_params, cfun=isfc, reduce='IncrementalPCA'):
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

    weights_function: see description from timecorr

        Default: gaussian_weights

    weights_params: see description from timecorr

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

        Default: A continuous version of Inter-Subject Functional Connectivity
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
        V = data[0].shape[1]
    else:
        V = data.shape[1]

    corrs = timecorr(data, weights_function=weight_function,
                     weights_params=weights_params, mode="within", cfun=isfc)
    return hyp.reduce(corrs, reduce=reduce, ndims=V)
