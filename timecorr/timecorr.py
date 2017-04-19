import numpy as np
from scipy.stats.stats import pearsonr
from scipy.io import loadmat
import sys
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

#parameters
time_len = 1000
activations_len = 5
gaussian_variance = 100

#datasets
activations = None
gaussian_array = None
covariances = None
# @profile
def correlation_calculation_single(filename):

    data = np.load(filename)
    true_covariances = data['arr_0']
    true_covariances1 = data['arr_1']
    activations = data['arr_2']
    activations_len, time_len = activations.shape
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    covariances = covariance_fragments = np.zeros([time_len, activations_len, activations_len])
    covariances_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    activations_shifted = activations - np.concatenate((np.zeros([activations_len,1]),activations[:,:-1]),1)

    for timepoint in range(1,time_len-1):
        covariance_fragments[timepoint,:,:] = np.cov(activations_shifted[:,(timepoint-1):(timepoint+2)])
    covariance_fragments[0,:,:], covariance_fragments[time_len-1,:,:] = covariance_fragments[1,:,:], covariance_fragments[time_len-2,:,:]


    for timepoint in range(time_len):
        coefficients = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficients = coefficients/np.sum(coefficients)
        coefficients = np.array([np.tile(val,[activations_len,activations_len]) for val in coefficients])
        covariances[timepoint,:,:] = np.sum(np.multiply(coefficients, covariance_fragments),0)

    # covariances = np.mean(0.5*(np.log(1+covariances) - np.log(1-covariances)),0)/2
    # covariances =  (np.exp(2*covariances) - 1)/(np.exp(2*covariances) + 1)

    for i in range(time_len):
        covariances_vector[i] = squareform(covariances[i,:,:],checks=False)



    true_covariances_vector = squareform(true_covariances,checks=False)
    true_covariances_vector1 = squareform(true_covariances1,checks=False)
    Y = np.array([pearsonr(covariances_vector[i,],true_covariances_vector)[0] for i in range(time_len)])
    Y1 = np.array([pearsonr(covariances_vector[i,],true_covariances_vector1)[0] for i in range(time_len)])
    plt.plot(range(time_len),Y,'b-')
    plt.plot(range(time_len),Y1,'r-')
    plt.show()

# @profile
def correlation_calculation_multiple(data):
    #parameter initialization
    subj_num = data.size
    time_len, activations_len = data[0].shape
    covariances = covariance_fragments = np.zeros([subj_num,time_len,activations_len,activations_len])
    gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
    covariances_vector = np.zeros([time_len,(activations_len * (activations_len-1) / 2)])
    coefficients = np.zeros([time_len,time_len,activations_len,activations_len])

    for timepoint in range(time_len):
        coefficient = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficient = coefficient/np.sum(coefficient)
        coefficients[timepoint] = np.array([np.tile(val,[activations_len,activations_len]) for val in coefficient])

    for subject,subject_data in enumerate(data):
        activations[subject] = subject_data.T
    activations_shifted = activations - np.concatenate((np.zeros([subj_num,activations_len,1]),activations[:,:,:-1]),2)
    activations_sum = (np.tile(np.sum(activations_shifted,0),[subj_num,1,1]) - activations_shifted)/(subj_num-1)

    for subj in range(subj_num):
        for timepoint in range(1,time_len-1):
            covariance_fragments[subj,timepoint,:,:] = np.cov(activations_shifted[subj,:,(timepoint-1):(timepoint+2)],\
                                                            activations_sum[subj,:,(timepoint-1):(timepoint+2)])[:activations_len,activations_len:]
        covariance_fragments[subj,0,:,:], covariance_fragments[subj,time_len-1,:,:] = covariance_fragments[subj,1,:,:], covariance_fragments[subj,time_len-2,:,:]

        for timepoint in range(time_len):
            covariances[subj, timepoint,:,:] = np.sum(np.multiply(coefficients[timepoint], covariance_fragments[subj]),0)

    covariances = np.mean(0.5*(np.log(1+covariances) - np.log(1-covariances)),0)/2
    covariances =  (np.exp(2*covariances) - 1)/(np.exp(2*covariances) + 1)

    for i in range(time_len):
        covariances_vector[i] = squareform(covariances[i,:,:],checks=False)

    Y = np.array([pearsonr(covariances_vector[i,],true_covariances_vector)[0] for i in range(time_len)])
    Y1 = np.array([pearsonr(covariances_vector[i,],true_covariances_vector1)[0] for i in range(time_len)])
    plt.plot(range(time_len),Y,'b-')
    plt.plot(range(time_len),Y1,'r-')
    plt.show()
    # return result

if __name__ == "__main__":
    filename = sys.argv[1]
    if len(sys.argv)>2 and sys.argv[2]=="single":
        print("single:")
        correlation_calculation_single(filename)
    else:
        correlation_calculation_multiple(filename)
#temp
# print("True covariance:")
# print(true_covariances)
#
# print("True covariance1:")
# print(true_covariances1)

# print("Overall covariances:")
# print(np.cov(activations_shifted))
# print("Mean covariance:")
# print(np.mean(covariances,0))
#
# print("Calculated Covariance:")
# print(covariances)

# Y = np.array([pearsonr(covariances_vector[i,],true_covariances_vector)[0] for i in range(time_len)])
# Y1 = np.array([pearsonr(covariances_vector[i,],true_covariances_vector1)[0] for i in range(time_len)])
# plt.plot(range(time_len),Y,'b-')
# plt.plot(range(time_len),Y1,'r-')
# plt.show()
