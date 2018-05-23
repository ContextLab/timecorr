# coding: utf-8

import numpy as np
from .timecorr import timecorr
from .helpers import mat2vec, vec2mat


def simulate_timecorr_data(subs=5, events=10, event_time=100, features=100, noise=0):
    """
    Simulates data for several subjects using shared timeseries of covariance matrices

    Parameters
    ----------
    subs : int
        Number of simulated subjects
    events : int
        Number of events
    event_time : int
        Number of timepoionts per event
    features : int
        Number of features
    noise : int or float
        Noise added to simulation

    Returns
    ----------
     data : np.ndarray
        Returns simulated data
    """
    covs = generate_timeseries_covs(events, features)

    data = []
    for s in np.arange(subs):
        data.append(generate_subject_data(covs, event_time, noise))

    return data


def generate_random_covariance_matrix(N):
    """
    Simulate random covariance matrix

    Parameters
    ----------
    N : int
        Desired dimension of matrix (NxN)

    Returns
    ----------
     covariance matrix: np.ndarray
        Returns the 'squareformed' covariance matrix (includeing the diagonal)
        Shape will be: ((N^2-N)/2) + N
    """
    template = np.random.randn(N, N)
    cov = np.multiply(template, template.T)
    return mat2vec(cov)

def generate_timeseries_covs(E, N):
    """
    Simulate a timeseries of covariance matrices

    Parameters
    ----------
    N : int
        Desired dimension of covariance matrix (NxN)
    E : int
        Desired number of events
    Returns
    ----------
     covs: np.ndarray
        Returns 'squareformed' covariance matrices (includeing the diagonal)
        for E events.
        Shape will be: Ex((N^2-N)/2) + N
    """

    covs = np.zeros((E, int((N**2 - N)/2 + N)))
    for event in np.arange(E):
        covs[event, :] = generate_random_covariance_matrix(N)
    return covs

def generate_data(sq_cov, T, noise=0):
    """
    Simulates data using one covariance matrix

    Parameters
    ----------
    sq_cov :  np.ndarray
        Returns the 'squareformed' covariance matrix (includeing the diagonal)
        Shape will be: ((N^2-N)/2) + N
    T : int
        Number of timepoionts per event

    Returns
    ----------
     data : np.ndarray
        Returns simulated data (TxN)
    """
    import warnings
    warnings.simplefilter('ignore')
    covmat = vec2mat(sq_cov)
    n = np.random.normal(0, noise, np.shape(covmat)[1])
    covmat =covmat+n*n.T
    return np.random.multivariate_normal(np.zeros(covmat.shape[0]), covmat, (T))

def generate_subject_data(covs, T, noise=0):
    """
    Simulates data using several covariance matrices

    Parameters
    ----------
    covs :  np.ndarray
        The 'squareformed' covariance matrices (includeing the diagonal)
        for E events.
        Shape will be: Ex((N^2-N)/2) + N
    T : int
        Number of timepoionts per event

    Returns
    ----------
     data : np.ndarray
        Returns simulated data ((T*E)xN)
    """
    data = []
    for i in np.arange(covs.shape[0]):
        data.extend(generate_data(covs[i, :], T, noise))
    return np.vstack(data)
