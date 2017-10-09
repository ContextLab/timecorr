from __future__ import print_function
from builtins import range
from os import listdir,getcwd
from os.path import isfile, join
import sys

if __name__== '__main__':
    directory, prefix, start, end = sys.argv[1:5]
    completed = set([int(f[len(prefix):f.index(".")]) for f in listdir(directory+"/") if prefix in f])
    print(len(completed))
    print(set(range(int(start),int(end)+1)).difference(completed))
