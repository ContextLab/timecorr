import numpy as np
from os import listdir
from os.path import join, isfile
import sys

def decoding_accuracy(directory,nlevels=11):
    accuracy = np.zeros(nlevels)
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "decoding_accuracy" in f)]
    for fname in onlyfiles:
        accuracy+=np.load(join(directory,fname))
    accuracy/=len(onlyfiles)
    print(accuracy)

def mixture_level_accuracy(directory, nlevels = 11):
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "optimal_weights_and_accuracy_" in f)]
    weights = np.zeros([nlevels, len(onlyfiles)])
    for index,fname in enumerate(onlyfiles):
        weights[:,index] = np.load(join(directory,fname))
    np.save(directory+"/mixture_weights_for_plot",weights)
    print("saved to "+directory+"/mixture_weights_for_plot")


if __name__== '__main__':
    directory, nlevels = sys.argv[1], sys.argv[2]
    decoding_accuracy(directory, nlevels)
