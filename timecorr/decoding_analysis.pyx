import sys
import h5py
import numpy as np
cimport numpy as np
from scipy.io import loadmat
from random import shuffle
import matplotlib.pyplot as plt
from math import exp, sqrt, pi
from scipy.spatial.distance import squareform

def covariance_calculation(np.ndarray[double,ndim=4] coefficients, np.ndarray[double, ndim=3] activations):
    #cython variable declaration
    cdef int time_len, activations_len, subj_num, timepoint, gaussian_variance, subject
    cdef double val
    cdef np.ndarray[double, ndim=2] covariances_vector
    cdef np.ndarray[double, ndim=3] activations_shifted, activations_sum, covariances_mean
    cdef np.ndarray[double, ndim=4] covariances, covariance_fragments
    cdef np.ndarray gaussian_array,

    #assign initial parameters
    gaussian_variance = 100
    subj_num = activations.shape[0]
    activations_len, time_len= activations[0].shape
    covariances = covariance_fragments = np.zeros([subj_num,time_len,activations_len,activations_len])
    covariances_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])

    #generate the activation matrix by finding difference between consecutive datapoints
    print(subj_num, activations_len, activations.shape[0],activations.shape[1])
    activations_shifted = activations - np.concatenate((np.zeros([subj_num,activations_len,1]),activations[:,:,:-1]),2)

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

def decoding_analysis():
    cdef int time_len, activations_len, subj_num, timepoint, gaussian_variance, subject, tests
    cdef double accuracy
    cdef np.ndarray[double,ndim=2] covariance1, covariance2, correlation, correlation1, correlation2
    cdef np.ndarray[double,ndim=3] activations
    cdef np.ndarray[double,ndim=4] coefficients
    accuracy_list = np.zeros([1,100])

    #load in data
    filename = "../../data/weights.mat"
    data_dict = loadmat(filename, squeeze_me=True)
    subject_weights  = data_dict['weights']

    subj_num = subject_weights.size
    time_len, activations_len= subject_weights[0].shape

    subj_indices = np.arange(subj_num)

    #generate activations from subject weights
    activations = np.zeros([subj_num, activations_len, time_len])
    for subject, subject_data in enumerate(subject_weights):
        activations[subject] = subject_data.T

    #load in gaussian coefficients
    f = h5py.File("coefficients.hdf5", "r")
    coefficients = f['coefficients'][:]

    for tests in range(1):
        shuffle(subj_indices)
        covariance1 = covariance_calculation(coefficients, activations[subj_indices[:(subj_num/2)],:,:]).T
        covariance2 = covariance_calculation(coefficients, activations[subj_indices[(subj_num/2):],:,:]).T
        correlation = np.corrcoef(covariance1, covariance1)
        correlation1 = correlation[:time_len,time_len:]
        correlation2 = correlation[time_len:,:time_len]
        accuracy = 0.0
        for timepoint in range(time_len):
            if np.argmax(correlation1[timepoint])==timepoint:
                accuracy+=1
            if np.argmax(correlation2[timepoint])==timepoint:
                accuracy+=1
        accuracy_list[0,tests]=accuracy/(time_len*2)

    print accuracy_list
    print(np.mean(accuracy_list))

    f.close()
