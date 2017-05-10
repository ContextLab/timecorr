# cython: profile=True
import numpy as np
cimport numpy as np
import h5py
from scipy.stats.stats import pearsonr
from scipy.io import loadmat
import sys
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

# correlation calculation for single subject
def correlation_calculation_single(activations, gaussian_variance, estimation_range):
    #cython variable declaration for faster implementation
    cdef np.ndarray gaussian_array, correlations, correlations_vector
    cdef np.ndarray correlation_fragments
    cdef int timepoint, i, half_range

    #initialize parameters
    half_range = int((estimation_range-1)/2)
    activations_len, time_len = activations.shape[0], activations.shape[1]
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    correlations = correlation_fragments = np.zeros([time_len, activations_len, activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])

    #generate correlation fragments
    for timepoint in range(half_range,time_len-half_range):
        correlation_fragments[timepoint,:,:] = np.corrcoef(activations[:,(timepoint-1):(timepoint+2)])
    correlation_fragments[range(half_range),:,:], correlation_fragments[range(time_len-half_range, time_len),:,:] = correlation_fragments[half_range,:,:], correlation_fragments[time_len-half_range-1,:,:]

    #apply gaussian coefficients
    for timepoint in range(time_len):
        coefficients = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficients = coefficients/np.sum(coefficients)
        coefficients = np.swapaxes(np.tile(np.tile(coefficients,[activations_len,1]),[activations_len,1,1]),2,0)
        correlations[timepoint,:,:] = np.sum(np.multiply(coefficients, correlation_fragments),0)

    #converting matrix to square form
    for i in range(time_len):
        correlations_vector[i] = squareform(correlations[i,:,:],checks=False)

    return correlations_vector

#correlation calculation for multiple subjects
def ISFC(activation, gaussian_variance, estimation_range, coeffcients=None):
    # cython variable declaration
    cdef int time_len, activations_len, subj_num, timepoint, subject, half_range
    cdef np.ndarray correlations_vector
    cdef np.ndarray activations_sum, correlations_mean
    cdef np.ndarray correlations, correlation_fragments, coefficients
    cdef np.ndarray gaussian_array
    cdef np.ndarray activations = np.array(activation)

    # assign initial parameters
    subj_num = activations.shape[0]
    half_range = int((estimation_range-1)/2)
    activations_len, time_len= activations[0].shape
    correlations = correlation_fragments = np.zeros([subj_num,time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    coefficients = np.zeros([time_len,time_len,activations_len,activations_len])

    # if no coefficients were passed in, generate new coefficients
    if coefficients == None:
        gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
        for timepoint in range(time_len):
            coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
            coefficient = coefficient/np.sum(coefficient)
            coefficients[timepoint] = np.swapaxes(np.tile(np.tile(coefficient,[activations_len,1]),[activations_len,1,1]),2,0)

    # if coefficient file was specified, then read in coefficients
    else:
        h5f = h5py.File(coeffcients,'r')
        coefficients = h5f['dataset_1'][:]
        h5f.close()

    #create a matrix that, for each subject, contains the sum of the data for all others
    activations_sum = (np.tile(np.sum(activations,0),[subj_num,1,1])- activations)/(subj_num-1.0)

    # calculate the correlations for each timepoint for each subject
    for subject in range(subj_num):

        # calculate correlation fragments for each timepoint using 3 consecutive timepoints
        for timepoint in range(half_range,time_len-half_range):
            correlation_fragments[subject,timepoint,:,:] = np.corrcoef(activations[subject,:,(timepoint-1):(timepoint+2)],\
                                                            activations_sum[subject,:,(timepoint-1):(timepoint+2)])[:activations_len,activations_len:]

        # the first and last timepoint of the correlation fragments are equal to the second and second to last fragments
        correlation_fragments[subject,range(half_range),:,:], correlation_fragments[subject,range(time_len-half_range, time_len),:,:] = correlation_fragments[subject,half_range,:,:], correlation_fragments[subject,time_len-half_range-1,:,:]
        # correlation_fragments[subj,0,:,:], correlation_fragments[subj,time_len-1,:,:] = correlation_fragments[subj,1,:,:], correlation_fragments[subj,time_len-2,:,:]

        # multiply the correlation fragments with the gaussian coefficients
        for timepoint in range(time_len):
            correlations[subject, timepoint,:,:] = np.sum(np.multiply(coefficients[timepoint], correlation_fragments[subject]),0)

    #normalize and average the correlation matrix
    correlations_mean = np.mean(0.5*(np.log(1+correlations) - np.log(1-correlations)),0)/2
    correlations_mean =  (np.exp(2*correlations_mean) - 1)/(np.exp(2*correlations_mean) + 1)

    #convert the square correlation matrices into vector form
    for i in range(time_len):
        correlations_vector[i] = squareform(correlations_mean[i,:,:],checks=False)

    return correlations_vector
