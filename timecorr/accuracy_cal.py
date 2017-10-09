import numpy as np
from os import listdir
from os.path import join, isfile
import sys
if __name__== '__main__':
    a = np.zeros(11)
    directory = sys.argv[1]
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    print(len(onlyfiles))
    for fname in onlyfiles:
        if "decoding_accuracy" in fname:
            a+=np.load(join(directory,fname))
    a/=100
    print(a)
