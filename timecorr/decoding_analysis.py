import hypertools as hyp
import numpy as np
from os import listdir
from os.path import isfile, join
from loadnii import loadnii as ln

def load_fmri_data(directory, nvoxels):
    '''
    Read in fRMI data from nii files in directory, reduce voxel dimensions and then store as a list of numpy arrays

    Input:
        directory: the path of the directory in which the nii files are stored

        nvoxels: the number of voxels to reduce the fRMI data to

    Return:
        A list of numpy matrices containing the reduced voxels activations
    '''
    directory = directory+"/"
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    data = []
    for fname in onlyfiles:
        if '.nii' in fname:
            temp = ln(fname)
            temp = hyp.tools.reduce(temp.Y,nvoxels)
            data.append(temp)
    return data

def
