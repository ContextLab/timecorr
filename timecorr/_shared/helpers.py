import numpy as np
cimport numpy as np
from math import exp, sqrt, pi
from scipy.spatial.distance import squareform,cdist

def sliding_window(activations, estimation_range):
    cdef np.ndarray correlations_vector, correlations
    cdef int timepoint, i, time_len, activations_len

    activations_len, time_len = activations.shape
    time_len -= estimation_range+1
    correlations = np.zeros([time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])

    for timepoint in range(time_len):
        correlations[timepoint] = np.corrcoef(activations[:,timepoint:(timepoint+estimation_range)])
        correlations_vector[i] = squareform(correlations[i,:,:],checks=False)
        
    return correlations_vector



def wcorr(activations, gaussian_variance, estimation_range):
    cdef np.ndarray gaussian_array, correlations, correlations_vector
    cdef np.ndarray c_activations, correlation_fragments
    cdef int timepoint, i, half_range, time_len

    half_range = int((estimation_range-1)/2)
    activations_len, time_len = activations.shape[0], activations.shape[1]
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    correlations = correlation_fragments = np.zeros([time_len, activations_len, activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    c_activations = activations

    for timepoint in range(half_range,time_len-half_range):
        correlation_fragments[timepoint,:,:] = np.corrcoef(c_activations[:,(timepoint-1):(timepoint+2)])
    correlation_fragments[range(half_range),:,:], correlation_fragments[range(time_len-half_range, time_len),:,:] = correlation_fragments[1,:,:], correlation_fragments[time_len-2,:,:]

    for timepoint in range(time_len):
        coefficients = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficients = coefficients/np.sum(coefficients)
        coefficients = np.swapaxes(np.tile(np.tile(coefficients,[activations_len,1]),[activations_len,1,1]),2,0)
        correlations[timepoint,:,:] = np.sum(np.multiply(coefficients, correlation_fragments),0)

    for i in range(time_len):
        correlations_vector[i] = squareform(correlations[i,:,:],checks=False)

    return correlations_vector


def isfc(activations, gaussian_variance, estimation_range):
    #cython variable declaration
    cdef int time_len, activations_len, subj_num, timepoint, subj, half_range
    cdef np.ndarray[double, ndim=2] correlations_vector
    cdef np.ndarray[double, ndim=3] c_activations, activations_sum, correlations_mean
    cdef np.ndarray[double, ndim=4] correlations, correlation_fragments, coefficients
    cdef np.ndarray gaussian_array

    #assign initial parameters
    subj_num = activations.shape[0]
    half_range = int((estimation_range-1)/2)
    activations_len, time_len= activations[0].shape
    correlations = correlation_fragments = np.zeros([subj_num,time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    coefficients = np.zeros([time_len,time_len,activations_len,activations_len])
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    c_activations = activations

    for timepoint in range(time_len):
        coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficient = coefficient/np.sum(coefficient)
        coefficients[timepoint] = np.swapaxes(np.tile(np.tile(coefficient,[activations_len,1]),[activations_len,1,1]),2,0)

    #create a matrix that, for each subject, contains the sum of the data for all other subjects
    activations_sum = (np.tile(np.sum(c_activations,0),[subj_num,1,1]) - c_activations)/(subj_num-1.0)

    #calculate the correlations for each timepoint for each subject
    for subj in range(subj_num):
        #calculate correlation fragments for each timepoint using 3 consecutive timepoints
        for timepoint in range(half_range,time_len-half_range):
            correlation_fragments[subj,timepoint,:,:] = np.cov(c_activations[subj,:,(timepoint-1):(timepoint+2)],\
                                                            activations_sum[subj,:,(timepoint-1):(timepoint+2)])[:activations_len,activations_len:]
            # correlation_fragments[subj,timepoint,:,:] = 1-cdist(c_activations[subj,:,(timepoint-1):(timepoint+2)],\
            #                                                 activations_sum[subj,:,(timepoint-1):(timepoint+2)],'correlation')
        #the first and last timepoint of the correlation fragments are equal to the second and second to last fragments
        correlation_fragments[subj,range(half_range),:,:], correlation_fragments[subj,range(time_len-half_range, time_len),:,:] = correlation_fragments[subj,half_range,:,:], correlation_fragments[subj,time_len-half_range-1,:,:]

        #multiply the correlation fragments with the gaussian coefficients
        for timepoint in range(time_len):
            correlations[subj, timepoint,:,:] = np.sum(np.multiply(coefficients[timepoint], correlation_fragments[subj]),0)

    #normalize and average the correlation matrix
    correlations_mean = np.mean(0.5*(np.log(1+correlations) - np.log(1-correlations)),0)/2
    correlations_mean =  (np.exp(2*correlations_mean) - 1)/(np.exp(2*correlations_mean) + 1)

    #convert the square correlation matrices into vector form
    for i in range(time_len):
        correlations_vector[i] = squareform(correlations_mean[i,:,:],checks=False)

    return correlations_vector
