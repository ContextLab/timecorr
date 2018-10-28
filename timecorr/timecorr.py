# coding: utf-8

from .helpers import isfc, laplace_weights, format_data, null_combine, reduce

def timecorr(data, weights_function=laplace_weights,
             weights_params=None, combine=null_combine,
             cfun=isfc, rfun=None):
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

    combine: a function applied to either a single matrix of vectorized correlation
        matrices, or a list of such matrices.  The function should return either
        a numpy array or a list of numpy arrays.

        Default: helpers.null_combine (a function that returns its input).  Other
        useful functions:

        helpers.corrmean_combine: take the element-wise average correlations across matrices
        helpers.tstat_combine: return element-wise t-statistics across matrices

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

    rfun: function to use for dimensionality reduction.  All hypertools and
        scikit-learn functions are supported: PCA, IncrementalPCA, SparsePCA,
        MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
        DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
        SpectralEmbedding, LocallyLinearEmbedding, MDS, and UMAP.

        Can be passed as a string, but for finer control of the model
        parameters, pass as a dictionary, e.g.
        reduce={‘model’ : ‘PCA’, ‘params’ : {‘whiten’ : True}}.

        See scikit-learn specific model docs for details on parameters supported
        for each model.

        Another option is to use graph theoretic measures computed for each node.
        The following measures are supported (via the brainconn toolbox):
        eigenvector_centrality, pagerank_centrality, and strength.  (Each
        of these must be specified as a string; dictionaries not supported.)

        Default: None (no dimensionality reduction)

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
    corrs = reduce(combine(cfun(data, weights)), rfun=rfun)

    if return_list and (not (type(corrs) == list)):
        return [corrs]
    elif (not return_list) and (type(corrs) == list) and (len(corrs) == 1):
        return corrs[0]
    else:
        return corrs


