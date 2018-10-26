# coding: utf-8

from .helpers import isfc, autofc, laplace_weights, format_data, r2z, z2r
import hypertools as hyp
import numpy as np



def timecorr(data, weights_function=laplace_weights,
             weights_params=None, combine=False, cfun=isfc):
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

        Default: laplace_weights; options: laplace_weights, gaussian_weights,
        t_weights, eye_weights, mexican_hat_weights

    weights_params: used to pass parameters to the weights_params function. This
        can be specified in any format (e.g. a scalar, list, object, dictionary,
        etc.).

        Default: None (use default parameters for the given weights function).
        Options: gaussian_params, laplace_params, t_params, eye_params,
        mexican_hat_params.

    combine: Boolean (default: False)
        When combine = False, timecorr returns a matrix or list in the same
        format as the input.  When combine = True, timecorr returns a single
        numpy array that reflects the mean taken across input arrays.  When only
        a single input array is passed, this argument is ignored.

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
    """

    if type(data) == list:
        T = data[0].shape[0]
        return_list = True
    else:
        T = data.shape[0]
        data = [data]
        return_list = False

    data = format_data(data)

    weights = weights_function(T, weights_params)
    corrs = cfun(data, weights)

    if combine:
        corrs = z2r(np.mean(r2z(np.stack(corrs, axis=2)), axis=2))
        return_list = False

    if return_list and (not (type(corrs) == list)):
        return [corrs]
    else:
        return corrs


def levelup(data, combine=False, weight_function=laplace_weights,
            weights_params=None, cfun=isfc, reduce='IncrementalPCA'):
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

    combine: Boolean (default: False)
        See `timecorr`

    weights_function: see description from timecorr

        Default: laplace_weights

    weights_params: see description from timecorr

        Default: None

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
        SpectralEmbedding, LocallyLinearEmbedding, MDS, and UMAP.

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
    get_V = lambda x: int(np.divide(-np.sqrt(8*x + 1) - 1, 2))

    corrs = timecorr(data, weights_function=weight_function, weights_params=weights_params, combine=combine, cfun=cfun)
    if type(corrs) is list:
        V = get_V(corrs[0].shape[1])
    else:
        V = get_V(corrs.shape[1])

    #TODO: add support for graph theory reduce operations
    return hyp.reduce(corrs, reduce=reduce, ndims=V)
