import numpy as np
from os import listdir
from os.path import join, isfile
import sys

def decoding_accuracy(directory,nlevels=11):
    directory = directory+"/results/"
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "decoding_accuracy" in f)]
    print(len(onlyfiles))
    accuracy = np.zeros([len(onlyfiles),nlevels])
    for index,fname in enumerate(onlyfiles):
        accuracy[index]=np.load(join(directory,fname))
    mean = np.mean(accuracy,axis=0)
    std = np.std(accuracy,axis=0)
    print(mean, std)
    np.savez(directory+"/intra_level_decoding_analysis",mean, std)

def mixture_level_accuracy(directory, nlevels = 11):
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "optimal_weights_and_accuracy_" in f)]
    weights = np.zeros([nlevels, len(onlyfiles)])
    for index,fname in enumerate(onlyfiles):
        weights[:,index] = np.load(join(directory,fname))
    np.save(directory+"/mixture_weights_for_plot",weights)
    df = pd.DataFrame(weights,columns = range(nlevels))
    newplot = sns.violinplot(data = df,orient = 'v')
    plt.show()
    print("saved to "+directory+"/mixture_weights_for_plot")


if __name__== '__main__':
    directory, nlevels = sys.argv[1], sys.argv[2]
    decoding_accuracy(directory, int(nlevels))
