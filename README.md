# TimeCorr #

## Overview ##
The TimeCorr toolbox provides tools for computing and exploring the correlational
structure of timeseries data.  There are two main functions:
* `timecorr` is used to compute dynamic correlations from a timeseries of observations, organized as a number-of-timepoints by number-of-features matrix.  The correlations may be computed _within_ a single matrix, or _across_ a list of such matrices.
* `levelup` is used to find higher order structure in the data.  Calling `levelup` converts a timeseries of observations into a timeseries of correlations _with the same number of features_.  This function may be called recursively to compute dynamic correlations ("level 1"), dynamic correlations _between_ correlations ("level 2"), dynamic correlations between correlations between correlations ("level 3"), etc.  The new features may be computed _within_ a single matrix, or a list of matrices (this returns a new single matrix or list of matrices of the same shape, but containing the higher order features), or _across_ a list of such matrices (given a list of matrices, the _across_ matrix higher-order structure comprises a single matrix).

## Installing ##
While running Python 2.7:
To install the TimeCorr toolbox, type `pip install .` from within the TimeCorr folder. 

## Basic usage ##
### `timecorr` "within" mode ###
Example command: `c0 = timecorr(x0, mode="within")`

The "within" mode of `timecorr` is available for single subject and multiple subject inputs:
Single subject:
  * Input x0:
      * A T x V dimensional Numpy matrix, where T and V represent the number of timepoints and the number of voxels in the dataset, respectively.
  * Return:
      * A T x (V^2 - V)/2 dimensional Numpy matrix, where each row is the reverse square form of the correlation matrix at each timepoint.
Multiple subjects:
  * Input x0:
      * A list of T x V dimensional Numpy matrices, where each matrix represents the activations of a single subject. T and V represent the number of timepoints and the number of voxels in the dataset, respectively.
  * Return:
      * A list of T x (V^2 - V)/2 dimensional Numpy matrices, where each matrix is the represents the correlation patterns for a single subject. Within each matrix, each row is the reverse square form of the correlation matrix at each timepoint.

### `timecorr` "across" mode ###
Example command: `c = timecorr(x0, mode="across")`

The "across" mode of `timecorr` is only available for multiple subject inputs:
  * Input x0:
      * A list of T x V dimensional Numpy matrices, where each matrix represents the activations of a single subject. T and V represent the number of timepoints and the number of voxels in the dataset, respectively.
  * Return:
      * A T x (V^2 - V)/2 dimensional Numpy matrix, where each row is the reverse square form of the inter-subject Functional Connectivity matrix at each timepoint.

### `levelup` "within" mode ###
Example command: `x1 = levelup(x0, mode="within")`

The "within" mode of `levelup` is available for single subject and multiple subject inputs:
Single subject:
  * Input x0:
      * A T x V dimensional Numpy matrix, where T and V represent the number of timepoints and the number of voxels in the dataset, respectively.
  * Return:
      A T x V dimensional Numpy matrix, where each row is the PCA reduced reverse square form of the correlation matrix at each timepoint.
* Multiple subjects:
  * Input x0:
      * A list of T x V dimensional Numpy matrices, where each matrix represents the activations of a single subject. T and V represent the number of timepoints and the number of voxels in the dataset, respectively.
  * Return:
      * A list of T x V dimensional Numpy matrices, where each matrix is the represents the correlation patterns for a single subject. Within each matrix, each row is the PCA reduced reverse square form of the correlation matrix at each timepoint.

### `levelup` "across" mode ###
Example command: `x1 = levelup(x0, mode="across")`

The "across" mode of `levelup` is only available for multiple subject inputs:
  * Input x0:
      * A list of T x V dimensional Numpy matrices, where each matrix represents the activations of a single subject. T and V represent the number of timepoints and the number of voxels in the dataset, respectively.
  * Return:
      * A T x V dimensional Numpy matrix, where each row is the PCA reduced reverse square form of the inter-subject Functional Connectivity matrix at each timepoint.

## Citing this toolbox ##
(add citation information)

## Further reading ##
For additional information and the full API, please see our [readthedocs](timecorr.readthedocs.com) site. (Need to make this website.)
