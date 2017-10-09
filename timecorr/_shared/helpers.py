from __future__ import division
from builtins import map
from builtins import range
from past.utils import old_div
import numpy as np
from math import exp, sqrt, pi
from scipy.spatial.distance import squareform, cdist
from multiprocessing import Pool,cpu_count

def coefficient_generation(timepoint):
    '''
    Given the guassian array generated in the main function, this function calculates the Gaussian coefficients for each timepoint

    Input:
        timepoint: the timepoint for which the coeficients are generated

    Return:
        The sum and a tiled version of the coefficient array
    '''
    coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
    return np.tile(coefficient,[activations_len,1]), np.sum(coefficient)

def isfc_helper(subj):
    '''
    Helper function to calculate the dynamic correlation matrix for each subject

    Input:
        subj: the index of the subjec to calculate the dynamic correlation matrix for

    Return:
        A time_len x activations_len x activations_len matrix
    '''
    # helper method to calculate correlation matrices at each timepoint
    def isfc_timepoint_helper(timepoint):
        # normalize activations, summing  over other subjects and calcualte standard deviations
        normalized_activations = activations[subj] - old_div(np.tile(np.reshape(np.sum(np.multiply(coefficients[timepoint],activations[subj]),1),[activations_len,1]),[1,time_len]),coefficients_sum[timepoint])
        normalized_sum_activations = activations_sum[subj] - old_div(np.tile(np.reshape(np.sum(np.multiply(coefficients[timepoint],activations_sum[subj]),1),[activations_len,1]),[1,time_len]),coefficients_sum[timepoint])
        sigma_activations  = np.sqrt(old_div(np.sum(np.multiply(coefficients[timepoint], np.square(normalized_activations)),1),coefficients_sum[timepoint]))
        sigma_activations_sum = np.sqrt(old_div(np.sum(np.multiply(coefficients[timepoint], np.square(normalized_sum_activations)),1),coefficients_sum[timepoint]))
        normalized_activations = np.divide(normalized_activations,np.tile(np.reshape(sigma_activations,[activations_len,1]),[1,time_len]))
        normalized_sum_activations = np.divide(normalized_sum_activations,np.tile(np.reshape(sigma_activations_sum,[activations_len,1]),[1,time_len]))

        return old_div(np.dot(np.multiply(np.tile(coefficients[timepoint,0],[activations_len,1]),normalized_activations),normalized_sum_activations.T),coefficients_sum[timepoint])

    return np.array(list(map(isfc_timepoint_helper, list(range(time_len)))))

def isfc(multi_activations, var=None):
    '''
    Function to calculate the ISFC for a multi-subject fRMI dataset

    Input:
        multi_activations: a subject_num x voxel_num x time_len numpy matrix containing the fRMI data   of multiple subjects

        var: The variance of the Gaussian distribution used to represent the influence of neighboring timepoints on calculation of correlation at the current timepoint in timecorr

    Return:
        A time_len x (voxel_num^2-voxel_num)/2 dimension matrix containing the ISFC of the input fRMI dataset
    '''
    # reference global variables to be used in multiprocessing helper functions
    global coefficients, activations_sum, coefficients_sum, activations
    global gaussian_array, time_len, subj_num, activations_len
    # assign initial parameters
    subj_num, activations_len, time_len = multi_activations.shape[0],multi_activations.shape[1],multi_activations.shape[2]
    if var is None:
        gaussian_variance = min(time_len, 1000)
    else:
        gaussian_variance = var

    coefficients_sum = np.zeros(time_len)
    correlations= np.zeros([subj_num, time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    coefficients = np.zeros([time_len, activations_len,time_len])
    gaussian_array = np.array([old_div(exp(-timepoint**2/2/gaussian_variance),sqrt(2*pi*gaussian_variance)) for timepoint in range(-time_len+1,time_len)])
    activations = np.array(multi_activations)

    # generate the gaussian coefficients
    for timepoint in range(time_len):
        coefficients[timepoint], coefficients_sum[timepoint] = coefficient_generation(timepoint)

    # create a matrix that, for each subject, contains the sum of the data for all other subjects
    activations_sum = old_div((np.tile(np.sum(activations,0),[subj_num,1,1]) - activations),(subj_num-1.0))

    # calculate the correlations for each timepoint for each subject
    p = Pool(min(cpu_count()-1,subj_num))
    correlations = np.array(p.map(isfc_helper,list(range(subj_num))))
    p.terminate()

    # normalize and average the correlation matrix
    correlations_mean = np.mean(0.5*(np.log(1e-5+1+correlations) - np.log(1e-5+1-correlations)),0)
    correlations_mean = correlations_mean+np.swapaxes(correlations_mean,1,2)
    correlations_mean =  old_div((np.exp(correlations_mean) - 1),(np.exp(correlations_mean) + 1))

    # transform into reverse squareform
    for i in range(time_len):
        correlations_vector[i] = squareform(correlations_mean[i,:,:],checks=False)

    return correlations_vector
 

def wcorr_helper(timepoint):
    '''
    Helper function to calculate the dynamic correlation at each timepoint

    Input:
        timepoint: the timepoint at which to calculate the correlationi matrix

    Ouput:
        A activations_len x activations_len matrix containing the correlation at the input timepoint
    '''
    # generate coefficients
    coefficient_tiled, coefficient_sum = coefficient_generation(timepoint)

    # normalize activations and calculate standard deviations
    normalized_activations = activations - old_div(np.tile(np.reshape(np.sum(np.multiply(coefficient_tiled,activations),1),[activations_len,1]),[1,time_len]),coefficient_sum)
    sigma  = np.sqrt(old_div(np.sum(np.multiply(coefficient_tiled, np.square(normalized_activations)),1),coefficient_sum))
    normalized_activations = np.divide(normalized_activations,np.tile(np.reshape(sigma,[activations_len,1]),[1,time_len]))

    return squareform(old_div(np.dot(np.multiply(coefficient_tiled,normalized_activations),normalized_activations.T),coefficient_sum), checks=False)

def wcorr(single_activations, var=None):
    '''
    Function to calculate the ISFC for a single-subject fRMI dataset

    Input:
        single_activations: a voxel_num x time_len numpy matrix containing the fRMI data of a single subject

        var: The variance of the Gaussian distribution used to represent the influence of neighboring timepoints on calculation of correlation at the current timepoint in timecorr

    Return:
        A time_len x (voxel_num^2-voxel_num)/2 dimension matrix containing the ISFC of the input fRMI dataset
    '''
    # reference global paramters for multiprocessing
    global gaussian_array, activations, time_len, activations_len
    # assign initial parameters
    activations = single_activations
    activations_len, time_len= activations.shape
    if var is None:
        gaussian_variance = min(time_len, 1000)
    else:
        gaussian_variance = var

    # generate gaussian coefficients
    gaussian_array = np.array([old_div(exp(-timepoint**2/2/gaussian_variance),sqrt(2*pi*gaussian_variance)) for timepoint in range(-time_len+1,time_len)])

    # using multiprocessing to calculate correlations at each timepoint
    p = Pool(cpu_count()-1)
    correlations_vectors = np.array(p.map(wcorr_helper, list(range(time_len))))
    p.terminate()

    return correlations_vectors

def timecorr_smoothing(single_activations, var=None):
    global gaussian_array, activations, time_len, activations_len
    activations = single_activations
    activations_len, time_len= activations.shape
    smoothed_activations = np.zeros([time_len,activations_len])
    if var is None:
        gaussian_variance = min(time_len, 1000)
    else:
        gaussian_variance = var
    # generate gaussian coefficients
    gaussian_array = np.array([old_div(exp(-timepoint**2/2/gaussian_variance),sqrt(2*pi*gaussian_variance)) for timepoint in range(-time_len+1,time_len)])
    for timepoint in range(time_len):
        coefficient_tiled, coefficient_sum = coefficient_generation(timepoint)
        smoothed_activations[timepoint,:] = old_div(np.sum(np.multiply(coefficient_tiled,activations),1),coefficient_sum)

    return smoothed_activations

def sliding_window(activations, window_length):
    '''
    Sliding window approach to calculate dynamic correlations for single subject

    Input:
        activations: activations_len x time_len matrix containing fRMI for a single subject

        window_length: length of the window to use to calculate correlation at a timepoint

    Return:
        A time_len x (voxel_num^2-voxel_num)/2 dimension matrix containing the dynamic correlations of the input fRMI dataset
    '''
    activations_len, time_len = activations.shape
    time_len -= window_length-1
    correlations = np.zeros([time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])

    for timepoint in range(time_len):
        correlations[timepoint] = np.corrcoef(activations[:,timepoint:(timepoint+window_length)])
        correlations_vector[timepoint] = squareform(correlations[timepoint,:,:],checks=False)

    return correlations_vector

def sliding_window_smoothing(activations, window_length):
    activations_len, time_len = activations.shape
    time_len -= window_length-1
    smoothed_activations = np.zeros([time_len,activations_len])
    for timepoint in range(time_len):
        smoothed_activations[timepoint,:] = np.mean(activations[:,timepoint:(timepoint+window_length)],1)

    return smoothed_activations

def sliding_window_isfc(activations, window_length):
    '''
    Sliding window approach to calculate dynamic correlations for multiple subjects

    Input:
        activations: subject_num x activations_len x time_len matrix containing fRMI for a multiple subjects

        window_length: length of the window to use to calculate correlation at a timepoint

    Return:
        A time_len x (voxel_num^2-voxel_num)/2 dimension matrix containing the ISFC of the input fRMI dataset
    '''
    subj_num, activations_len, time_len= activations.shape[0],activations.shape[1],activations.shape[2]-window_length+1
    correlations= np.zeros([subj_num, time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    activations = np.array(activations)
    activations_sum = old_div((np.tile(np.sum(activations,0),[subj_num,1,1]) - activations),(subj_num-1.0))
    for subj in range(subj_num):
        for timepoint in range(time_len):
            correlations[subj, timepoint] = 1-cdist(activations[subj,:,timepoint:(timepoint+window_length)],activations_sum[subj,:,timepoint:(timepoint+window_length)],"correlation")
    #normalize and average the correlation matrix
    correlations_mean = np.mean(0.5*(np.log(1e-5+1+correlations) - np.log(1e-5+1-correlations)),0)
    correlations_mean = correlations_mean+np.swapaxes(correlations_mean,1,2)
    correlations_mean =  old_div((np.exp(correlations_mean) - 1),(np.exp(correlations_mean) + 1))

    for i in range(time_len):
        correlations_vector[i] = squareform(correlations_mean[i,:,:],checks=False)

    return correlations_vector
