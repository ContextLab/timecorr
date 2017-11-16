# TimeCorr #

## Overview ##
The TimeCorr toolbox provides tools for computing and exploring the correlational
structure of timeseries data.  There are two main functions:
* `timecorr` is used to compute dynamic correlations from a timeseries of observations, organized as a number-of-timepoints by number-of-features matrix.  The correlations may be computed _within_ a single matrix, or _across_ a list of such matrices.
* `levelup` is used to find higher order structure in the data.  Calling `levelup` converts a timeseries of observations into a timeseries of correlations _with the same number of features_.  This function may be called recursively to compute dynamic correlations ("level 1"), dynamic correlations _between_ correlations ("level 2"), dynamic correlations between correlations between correlations ("level 3"), etc.  The new features may be computed _within_ a single matrix, or a list of matrices (this returns a new single matrix or list of matrices of the same shape, but containing the higher order features), or _across_ a list of such matrices (given a list of matrices, the _across_ matrix higher-order structure comprises a single matrix).

## Installing ##
To install the TimeCorr toolbox, type `pip install .` from within the TimeCorr folder.

## Basic usage ##
### `timecorr` "within" mode ###
(add instructions for timecorr within mode)

### `timecorr` "across" mode ###
(add instructions for timecorr across mode)

### `levelup` "within" mode ###
(add instructions for levelup within mode)

### `levelup` "across" mode ###
(add instructions for levelup across mode)

## Citing this toolbox ##
(add citation information)

## Further reading ##
For additional information and the full API, please see our [readthedocs](timecorr.readthedocs.com) site. (Need to make this website.)
