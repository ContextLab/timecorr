# TimeCorr #

## Overview ##
The TimeCorr toolbox provides tools for computing and exploring the correlational structure of timeseries data.  There are two main functions:
* `timecorr` is used to compute dynamic correlations from a timeseries of observations, organized as a number-of-timepoints by number-of-features matrix.  The correlations may be computed _within_ a single matrix, or _across_ a list of such matrices.
* `levelup` is used to find higher order structure in the data.  Calling `levelup` converts a timeseries of observations into a timeseries of correlations _with the same number of features_.  This function may be called recursively to compute dynamic correlations ("level 1"), dynamic correlations _between_ correlations ("level 2"), dynamic correlations between correlations between correlations ("level 3"), etc.  The new features may be computed _within_ a single matrix, or a list of matrices (this returns a new single matrix or list of matrices of the same shape, but containing the higher order features), or _across_ a list of such matrices (given a list of matrices, the _across_ matrix higher-order structure comprises a single matrix).

## Installing ##
To install the TimeCorr toolbox directly from GitHub, type `pip install git+https://github.com/ContextLab/timecorr`.  You can also install by calling `pip install .` from within the TimeCorr folder (after cloning this repository).

## Requirements ##

* Python 2.7, 3.4+
* Numpy >= 1.10.4
* Scipy >= 1.0.0
* Seaborn >= 0.8.1
* Matplotlib >=2.0.1
* Pandas >= 0.22.0
* Hypertools >= 0.4.2


## Basic usage ##
### `timecorr` "within" mode ###

The timecorr within mode can be used for one or multiple subjects. There are two variations of this mode, _isfc_ and _wisfc_. In the isfc mode the functional connectivity is calculated by using a timepoint by number of voxel matrix and finding the reverse squareform of the correlation matrix. The wisfc mode the inter-subject functional connectivity does the same thing yet is found by using a weighted correlational average of correlational matrices.   


### `timecorr` "across" mode ###
The timecorr across mode may only be used with multiple subject inputs. There are two variations of this mode, _isfc_ and _wisfc_. In the isfc mode, the inter-subject functional connectivity is calculated by using a timepoint by number of voxel matrix and finding the reverse squareform of the correlation matrix. the wisfc mode calculates the inter-subject functional connectivity the same way, yet is found by using a weighted correlational average of correlational matrices.


### `levelup` "within" mode ###

The levelup within mode can be used for both multiple and single subjects. In this mode the higher-order brain dynamics of a subject are calculated by transforming a timepoint by number of voxel matrix into a PCA (are we still using PCA?) reduced correlation matrix at each timepoint.  


### `levelup` "across" mode ###

The levelup across mode may only be used with multiple subject inputs. By computing the average correlation matrix of the correlation matrices at each level. When using the wisfc function the weight of the average correlation matrices can be adjusted so to best express an individual subject.


## Citing this toolbox ##
(add citation information)

## Further reading ##
For additional information and the full API, please see our [readthedocs](timecorr.readthedocs.com) site. (Need to make this website.)
