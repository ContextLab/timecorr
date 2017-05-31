import numpy as np
from math import exp, sqrt, pi
from scipy.spatial.distance import squareform, cdist
import dill
from multiprocessing import Pool,cpu_count
coefficients, activations_sum, coefficients_sum, c_activations=0,0,0,0
time_len, subj_num, activations_len = 0,0,0
def isfc_helper(subj):
    correlation = np.zeros([time_len,activations_len,activations_len])
    for timepoint in range(time_len):
        normalized_activations = c_activations[subj] - np.tile(np.reshape(np.sum(np.multiply(coefficients[timepoint],c_activations[subj]),1),[activations_len,1]),[1,time_len])/coefficients_sum[timepoint]
        normalized_sum_activations = activations_sum[subj] - np.tile(np.reshape(np.sum(np.multiply(coefficients[timepoint],activations_sum[subj]),1),[activations_len,1]),[1,time_len])/coefficients_sum[timepoint]
        sigma_activations  = np.sqrt(np.sum(np.multiply(coefficients[timepoint], np.square(normalized_activations)),1)/coefficients_sum[timepoint])
        sigma_activations_sum = np.sqrt(np.sum(np.multiply(coefficients[timepoint], np.square(normalized_sum_activations)),1)/coefficients_sum[timepoint])

        for i in range(activations_len):
            for j in range(activations_len):
                correlation[timepoint, i,j] = np.sum(np.multiply(np.multiply(coefficients[timepoint,0], normalized_activations[i]), normalized_sum_activations[j]))/(sigma_activations[i]*sigma_activations_sum[j]*coefficients_sum[timepoint])
    return correlation

def isfc(multi_activations, gaussian_variance):
    #reference global variables to be used in multiprocessing helper functions
    global coefficients, activations_sum, coefficients_sum, c_activations, time_len, subj_num, activations_len

    #assign initial parameters
    subj_num, activations_len, time_len= multi_activations.shape[0],multi_activations.shape[1],multi_activations.shape[2]
    coefficients_sum = np.zeros(time_len)
    correlations= np.zeros([subj_num, time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    coefficients = np.zeros([time_len, activations_len,time_len])
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    c_activations = np.array(multi_activations)

    for timepoint in range(time_len):
        coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficients_sum[timepoint] = np.sum(coefficient)
        coefficients[timepoint] = np.tile(coefficient,[activations_len,1])

    #create a matrix that, for each subject, contains the sum of the data for all other subjects
    activations_sum = (np.tile(np.sum(c_activations,0),[subj_num,1,1]) - c_activations)/(subj_num-1.0)

    #calculate the correlations for each timepoint for each subject
    p = Pool(cpu_count()-1)
    correlations = np.array(map(isfc_helper,range(subj_num)))
    p.terminate()

    #normalize and average the correlation matrix
    correlations_mean = np.mean(0.5*(np.log(1+correlations) - np.log(1-correlations)),0)
    correlations_mean = correlations_mean+np.swapaxes(correlations_mean,1,2)
    correlations_mean =  (np.exp(correlations_mean) - 1)/(np.exp(correlations_mean) + 1)


    for i in range(time_len):
        correlations_vector[i] = squareform(correlations_mean[i,:,:],checks=False)

    return correlations_vector



def wcorr(activations, gaussian_variance):
    #assign initial parameters
    activations_len, time_len= activations.shape
    correlations_vectors = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])


    def wcorr_helper(timepoint):
        coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        correlations_vector = np.zeros([(activations_len * (activations_len-1) / 2)])
        coefficient_tiled = np.tile(coefficient,[activations_len,1])
        coefficient_sum = np.sum(coefficient)
        normalized_activations = activations - np.tile(np.reshape(np.sum(np.multiply(coefficient_tiled,activations),1),[activations_len,1]),[1,time_len])/coefficient_sum
        sigma  = np.sqrt(np.sum(np.multiply(coefficient_tiled, np.square(normalized_activations)),1)/coefficient_sum)
        index = 0
        for i in range(activations_len-1):
            for j in range(i+1,activations_len):
                correlations_vector[index] = np.sum(np.multiply(np.multiply(coefficient, normalized_activations[i]), normalized_activations[j]))/(sigma[i]*sigma[j]*coefficient_sum)
                index+=1

        return correlations_vector

    correlations_vectors = np.array(map(wcorr_helper, range(time_len)))
    return correlations_vectors

def sliding_window(activations, estimation_range):
    activations_len, time_len = activations.shape
    time_len -= estimation_range-1
    correlations = np.zeros([time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])

    for timepoint in range(time_len):
        correlations[timepoint] = np.corrcoef(activations[:,timepoint:(timepoint+estimation_range)])
        correlations_vector[timepoint] = squareform(correlations[timepoint,:,:],checks=False)

    return correlations_vector


def sliding_window_isfc(activations, estimation_range):
    subj_num, activations_len, time_len= activations.shape[0],activations.shape[1],activations.shape[2]
    correlations= np.zeros([subj_num, time_len,activations_len,activations_len])
    correlations_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    c_activations = np.array(activations)
    activations_sum = (np.tile(np.sum(c_activations,0),[subj_num,1,1]) - c_activations)/(subj_num-1.0)
    for subj in range(subj_num):
        for timepoint in range(time_len):
            correlations[subj, timepoint] = 1-cdist(c_activations[subj,:,timepoint:(timepoint+estimation_range)],activations_sum[subj,:,timepoint:(timepoint+estimation_range)],"correlation")

    #normalize and average the correlation matrix
    correlations_mean = np.mean(0.5*(np.log(1+correlations) - np.log(1-correlations)),0)
    correlations_mean = correlations_mean+np.swapaxes(correlations_mean,1,2)
    correlations_mean =  (np.exp(correlations_mean) - 1)/(np.exp(correlations_mean) + 1)

    for i in range(time_len):
        correlations_vector[i] = squareform(correlations_mean[i,:,:],checks=False)

    return correlations_vector
