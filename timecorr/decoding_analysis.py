from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import scipy.spatial.distance as sd
import numpy as np
import sys
from random import shuffle
from os import listdir,getcwd
from os.path import isfile, join
#from timecorr import levelup, decode, decode_raw_data, timecorr, decode_pair 
from sklearn import decomposition
from scipy.io import loadmat
from scipy import optimize
from time import time
np.seterr(all='ignore')

def load_and_levelup(directory, nvoxels, nlevels):
    '''
    Read in fRMI data from nii files in directory, reduce voxel dimensions and then level up fRMI activations to the specified number of levels using levelup function from timecorr and store all level activations in a temporary file

    Input:
        directory: the path of the directory in which the nii files are stored

        nvoxels: the number of voxels to reduce the fRMI data to

    Return:
        A list of numpy matrices containing the reduced voxels activations
    '''
    directory = directory+"/"
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    data = []
    ipca = decomposition.IncrementalPCA(n_components=nvoxels, batch_size=nvoxels)
    for fname in onlyfiles:
        if '.npy' in fname:
            temp = np.load(join(directory, fname)).T
        elif '.mat' in fname:
            temp = loadmat(join(directory,fname))['Y']
        else:
            continue
        data.append(temp)

    subject_num, time_len = len(data), data[0].shape[0]
    stacked_data = np.concatenate(data,0)
    pca_reduced = ipca.fit_transform(stacked_data[:,~np.isnan(stacked_data).any(axis=0)])
    all_activations = np.zeros([(nlevels+1),subject_num,time_len,nvoxels])
    for i in range(subject_num):
        all_activations[0,i] = pca_reduced[i*time_len:(i+1)*time_len]
    for l in range(nlevels):
        all_activations[(l+1)] = np.array(levelup(all_activations[(l)],mode="within"))
    np.save(directory+"/results/all_level_activations", all_activations)
    print("saved to "+directory+"/results/all_level_activations")

def decoding_analysis(directory, nlevels, repetition_index, nfolds=1, noise=0):
    '''
    Performs decoding analysis

    Input:
        directory: string
            path to the directory with the nii files

        nlevels: int
            the number of levels to levelup and perform decoding analysis

        repetition_index: int
            the index of the repetition to help record data

        nfolds: int, defaults to 3
            The number of repetitions to reshuffle in decoding analysis

    Return:
        A numpy array with the decoding accuracy at each level
    '''
    activations_file=directory+"/results/all_level_activations.npy"
    accuracy_file=directory+"/results/decoding_accuracy_"+str(repetition_index)
    all_activations = np.load(activations_file)
    decoding_accuracy = np.zeros(nlevels+1)
    decoding_accuracy[0] = decode_raw_data(all_activations[0],nfolds=nfolds)
    for l in range(int(nlevels)):
        decoding_accuracy[l+1]=decode(all_activations[l],nfolds=nfolds)
    accuracy = np.save(accuracy_file, decoding_accuracy)
    print("saved to "+accuracy_file)

def divide_and_timecorr(directory, repetition_index):
    '''
    Randomly divides the subjects into 4 groups and calculate the isfc at each level for each group

    Input:
        director: string
            path to the directory containing activations at all levels

        repetition_index: string
            the index of the repetition to help record data in the

    '''
    activations = np.load(directory+"/results/all_level_activations.npy")
    nlevels, nsubjects, ntimepoints, nvoxels = activations.shape
    group_size, subjects = int(old_div(nsubjects,4)), list(range(nsubjects)) # do we really want floor division here?
    shuffle(subjects)
    groups = [subjects[0:group_size],subjects[group_size:2*(group_size)],subjects[2*group_size:3*(group_size)], subjects[3*(group_size):]]
    np.save(directory+"/results/group_assignment_"+str(repetition_index), subjects)
    isfc = np.zeros([4,nlevels, ntimepoints,old_div((nvoxels**2-nvoxels),2)]) # do we really want floor division here?
    for level in range(nlevels):
        for group in range(4):
            isfc[group, level] = timecorr(activations[level, groups[group]],mode = "across")
    np.save(directory+"/results/isfc_"+str(repetition_index), isfc)

def optimal_level_weights(correlations):
    '''
    Find the weights array for each level that returns the highest decoding accuracy

    Input:
        corr1: numpy array
            containing the "across" timecorr correlation for group 1

        corr2: numpy array
            containing the "across" timecorr correlation for group 2

    Returns optimal weights array
    '''
    nlevels, ntimepoints = correlations.shape[0], correlations.shape[1]
    def weighted_decoding_analysis(w):
        '''
        Find the decoding accuracy of the weighted sum of correlations at each level

        Input:
            w: numpy array
                contains the weights for each level

        Returns decoding accuracy between the two groups after summing each level by the corresponding weights
        '''
        w = np.absolute(w)
        w = old_div(w,np.sum(w)) # # do we really want floor division here?
        accuracy=0
        weighted = np.sum([correlations[x]*w[x] for x in range(nlevels)],axis=0)
        weighted =  old_div((np.exp(2*weighted) - 1),(np.exp(2*weighted) + 1)) # do we really want floor division here?
        include_inds = np.arange(ntimepoints)
        for t in range(0, ntimepoints):
            decoded_inds = include_inds[np.where(weighted[t, include_inds] == np.max(weighted[t, include_inds]))]
            accuracy += np.mean(decoded_inds == np.array(t))
        accuracy/=float(ntimepoints)
        print(accuracy,w)
        return -1*accuracy

    def constraint1(x):
        return 1-np.max(x)
    def constraint2(x):
        return np.min(x)

    w = np.absolute(np.random.normal(0,1,nlevels))
    w = old_div(w,np.sum(w)) # do we really want floor division here?
  #  w = np.array([1,0])
    weights = optimize.minimize(weighted_decoding_analysis, w, method="COBYLA", constraints = ({'type': 'ineq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2}),tol=1e-4)["x"]
    return old_div(np.absolute(weights),np.sum(np.absolute(weights))) # do we really want floor division here?

def optimal_decoding_accuracy(directory, repetition_index):
    '''
    Given random division of subjects into groups A1, A2, B1 and B2. Calculate optimal level weights using A1 and A2, then find the new decoding accuracy using optimal weights on B1 and B2

    Input:
        directory: string
            Path to the directory containing the activations dataset

        repetition_index: str
            specific repetition to access

    Returns None
    '''
    isfc = np.load(directory+"/results/isfc_"+str(repetition_index)+".npy")
    group_assignments = np.load(directory+"/results/group_assignment_"+str(repetition_index)+".npy")
    group_size = int(old_div(len(group_assignments),4)) # do we really want floor division here?
    group_assignments = [[group_assignments[0:group_size]],[group_assignments[group_size:2*(group_size)]],[group_assignments[2*group_size:3*(group_size)]], [group_assignments[3*(group_size):]]]
    raw_activation = np.load(directory+"/results/all_level_activations.npy")[0]
    print("Load data complete")
    nlevels, ntimepoints = isfc.shape[1]+1, isfc.shape[2]
    A_correlations, B_correlations = np.zeros([nlevels, ntimepoints, ntimepoints]),  np.zeros([nlevels, ntimepoints, ntimepoints])
    A_correlations[0] = 1 - sd.cdist(np.mean(raw_activation[group_assignments[0]],axis=0), np.mean(raw_activation[group_assignments[1]],axis=0), 'correlation')
    B_correlations[0] = 1 - sd.cdist(np.mean(raw_activation[group_assignments[2]],axis=0), np.mean(raw_activation[group_assignments[3]],axis=0), 'correlation')
    for i in range(0,nlevels-1):
        A_correlations[i+1] = 1 - sd.cdist(isfc[0,i], isfc[1,i], 'correlation')
        B_correlations[i+1] = 1 - sd.cdist(isfc[2,i], isfc[3,i], 'correlation')
    print("Correlation calculation complete")
    A_correlations = 0.5*(np.log(1e-5+1+A_correlations) - np.log(1e-5+1-A_correlations))
    B_correlations = 0.5*(np.log(1e-5+1+B_correlations) - np.log(1e-5+1-B_correlations))

    weights = optimal_level_weights(A_correlations)
    print("Optimization Complete")

    weighted = np.sum([B_correlations[x]*weights[x] for x in range(nlevels)],axis=0)
    weighted =  old_div((np.exp(2*weighted) - 1),(np.exp(2*weighted) + 1)) # do we really want floor division here?
    accuracy = 0
    include_inds = np.arange(ntimepoints)
    for t in range(0, ntimepoints):
        decoded_inds = include_inds[np.where(weighted[t, include_inds] == np.max(weighted[t, include_inds]))]
        accuracy += np.mean(decoded_inds == np.array(t))
    accuracy/=ntimepoints
    print(weights,accuracy)

    out_file=directory+"/results/optimal_weights_and_accuracy_"+str(repetition_index)
    np.savez(out_file,weights,accuracy)
    print(weights,accuracy)
    print("saved to "+out_file)

if __name__== '__main__':
#    load_and_levelup(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
 #   decoding_analysis(sys.argv[1],int(sys.argv[2]),sys.argv[3])
#    divide_and_timecorr(sys.argv[1],sys.argv[2])
    optimal_decoding_accuracy(sys.argv[1],sys.argv[2])
