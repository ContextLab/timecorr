import numpy as np
# cimport numpy as np
from scipy.stats.stats import pearsonr
from scipy.io import loadmat
import sys
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

def correlation_calculation_single(activations, gaussian_variance, estimation_range):
    # cdef np.ndarray gaussian_array, covariances, covariances_vector
    # cdef np.ndarray activations_shifted, covariance_fragments
    # cdef int timepoint, i, half_range

    half_range = int((estimation_range-1)/2)
    activations_len, time_len = activations.shape[0], activations.shape[1]
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    covariances = covariance_fragments = np.zeros([time_len, activations_len, activations_len])
    covariances_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    activations_shifted = activations
    # - np.concatenate((np.zeros([activations_len,1]),activations[:,:-1]),1)

    for timepoint in range(half_range,time_len-half_range):
        covariance_fragments[timepoint,:,:] = np.cov(activations_shifted[:,(timepoint-1):(timepoint+2)])
    covariance_fragments[range(half_range),:,:], covariance_fragments[range(time_len-half_range, time_len),:,:] = covariance_fragments[1,:,:], covariance_fragments[time_len-2,:,:]

    for timepoint in range(time_len):
        coefficients = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficients = coefficients/np.sum(coefficients)
        coefficients = np.array([np.tile(val,[activations_len,activations_len]) for val in coefficients])
        covariances[timepoint,:,:] = np.sum(np.multiply(coefficients, covariance_fragments),0)

    for i in range(time_len):
        covariances_vector[i] = squareform(covariances[i,:,:],checks=False)

    return covariances_vector


def ISFC(activations, gaussian_variance, estimation_range):
    #cython variable declaration
    # cdef int time_len, activations_len, subj_num, timepoint, subject
    # cdef double val
    # cdef np.ndarray[double, ndim=2] covariances_vector
    # cdef np.ndarray[double, ndim=3] activations_shifted, activations_sum, covariances_mean
    # cdef np.ndarray[double, ndim=4] covariances, covariance_fragments, coefficients
    # cdef np.ndarray gaussian_array,

    #assign initial parameters
    subj_num = activations.shape[0]
    activations_len, time_len= activations[0].shape
    covariances = covariance_fragments = np.zeros([subj_num,time_len,activations_len,activations_len])
    covariances_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    coefficients = np.zeros([time_len,time_len,activations_len,activations_len])
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])

    for timepoint in range(time_len):
        coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficient = coefficient/np.sum(coefficient)
        coefficients[timepoint] = np.swapaxes(np.tile(np.tile(coefficient,[activations_len,1]),[activations_len,1,1]),2,0)

    #generate the activation matrix by finding difference between consecutive datapoints
    activations_shifted = activations
    #  - np.concatenate((np.zeros([subj_num,activations_len,1]),activations[:,:,:-1]),2)

    #create a matrix that, for each subject, contains the sum of the data for all other subjects
    activations_sum = (np.tile(np.sum(activations_shifted,0),[subj_num,1,1]) - activations_shifted)/(subj_num-1)

    #calculate the covariances for each timepoint for each subject
    for subj in range(subj_num):
        #calculate covariance fragments for each timepoint using 3 consecutive timepoints
        for timepoint in range(1,time_len-1):
            covariance_fragments[subj,timepoint,:,:] = np.cov(activations_shifted[subj,:,(timepoint-1):(timepoint+2)],\
                                                            activations_sum[subj,:,(timepoint-1):(timepoint+2)])[:activations_len,activations_len:]
        #the first and last timepoint of the covariance fragments are equal to the second and second to last fragments
        covariance_fragments[subj,0,:,:], covariance_fragments[subj,time_len-1,:,:] = covariance_fragments[subj,1,:,:], covariance_fragments[subj,time_len-2,:,:]

        #multiply the covariance fragments with the gaussian coefficients
        for timepoint in range(time_len):
            covariances[subj, timepoint,:,:] = np.sum(np.multiply(coefficients[timepoint], covariance_fragments[subj]),0)

    #normalize and average the covariance matrix
    covariances_mean = np.mean(0.5*(np.log(1+covariances) - np.log(1-covariances)),0)/2
    covariances_mean =  (np.exp(2*covariances_mean) - 1)/(np.exp(2*covariances_mean) + 1)

    #convert the square covariance matrices into vector form
    for i in range(time_len):
        covariances_vector[i] = squareform(covariances_mean[i,:,:],checks=False)

    return covariances_vector
