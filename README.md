<h2>Overview</h2>

The TimeCorr toolbox provides tools for computing and exploring the correlational structure of timeseries data.  There is one main function:
+ `timecorr` is used to compute dynamic correlations from a timeseries of observations and to find higher order structure in the data, organized as a number-of-timepoints by number-of-features matrix.


<h3>Basic usage</h3>


## `timecorr`
The `timecorr` function takes your data and returns moment-by-moment correlations during the same timepoints. `timecorr` also lets you explore higher order structure in the data in a computationally tractable way by specifiying a dimensionality reduction technique.
+ `timecorr` computes dynamic correlations and return a result in the same format, but where each data matrix has number-of-timepoints rows and $\frac{n^2 - n}{2}$ features (i.e. a vectorized version of the upper triangle of each timepoint's correlation matrix).
+ `timecorr` also lets you explore higher order structure in the data by projecting the correlations back onto the original number-of-timepoints by number-of-featuers space.

### Format your data
You should format your data as a Numpy array or Pandas dataframe with one row per observation and one column per feature (i.e. things you're tracking over time-- e.g. a voxel, electrode, channel, etc.).  You can then pass timecorr a single dataframe or a list of dataframes (each with the same numbers of timepoints and features).

### Pick a `weights_function`
How much the observed data at every timepoint contributes to the correlations at each timepoint.

### Specifiy the `weights_params`
Parameters for `weights_function`

### Choose `cfun` for computing dynamic correlations
The correlations may be computed _within_ a single matrix, or _across_ a list of such matrices. If a list of data matrices are passed, each data matrix is compared to the average of the other data matrices (`isfc` mode) or a similarity-weighted average of the other data matrices (`wisfc` mode).  If only a single data matrix is passed, the correlations are computed with respect to the same data matrix.
Computing correlations _across_ a list is for finding shared correlation across sets of observations (e.g. from different experimental participants).  If only a single data matrix is passed, `across` mode will behave the same as `within` mode.  If a list of data matrices is passed, `isfc` mode computes each matrix's correlations with respect to the average of the others, and then averages across all of those correlations.  `wisfc` mode behaves similarly, but computes weighted averages (e.g. based on inter-matrix similarities).

### Choose `rfun` for reducing the data and exploring higher order structure
By specifiying a reduction technique, `rfun`, `timecorr` takes a timeseries of observations and returns a timeseries of correlations _with the same number of features_. This is useful in that it prevents "dimension blowup" whereby running timecorr its own output squares the number of features-- thereby preventing the efficient exploration of higher-order correlations. This function may be called recursively to compute dynamic correlations ("level 1"), dynamic correlations _between_ correlations ("level 2"), dynamic correlations between correlations between correlations ("level 3"), etc. If `rfun` is not specified, the returned data matrix will have number-of-timepoints rows and $\frac{n^2 - n}{2}$ features.


<h2>Installation</h2>

<h3>Recommended way of installing the toolbox</h3>
You may install the latest stable version of our toolbox using [pip](https://pypi.python.org/pypi/pip):

`pip install timecorr`

or if you have a previous version already installed:

`pip install --upgrade timecorr`


<h3>Dangerous/hacker/developer way of installing the toolbox (use caution!)</h3>
To install the latest (bleeding edge) version directly from this repository use:

`pip install --upgrade git+https://github.com/ContextLab/timecorr.git`


<h2>Requirements</h2>

The toolbox is currently supported on Mac and Linux.  It has not been tested on Windows (and we expect key functionality not to work properly on Windows systems).
Dependencies:
+ Python 3.4+
+ Numpy >= 1.10.4
+ Scipy >= 1.0.0
+ Seaborn >= 0.8.1
+ Matplotlib >=2.0.1
+ Pandas >= 0.22.0
+ Hypertools >= 0.4.2



<h2>Contributing</h2>

Thanks for considering adding to our toolbox!  Some text below hoas been borrowed from the [Matplotlib contributing guide](http://matplotlib.org/devdocs/devel/contributing.html).

<h3>Submitting a bug report</h3>

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2 sentences.
2. A short, self-contained code snippet to reproduce the bug, ideally allowing a simple copy and paste to reproduce. Please do your best to reduce the code snippet to the minimum required.
3. The actual outcome of the code snippet
4. The expected outcome of the code snippet

<h3>Contributing code</h3>

The preferred way to contribute to supereeg is to fork the main repository on GitHub, then submit a pull request.

+ If your pull request addresses an issue, please use the title to describe the issue and mention the issue number in the pull request description to ensure a link is created to the original issue.

+ All public methods should be documented in the README.

+ Each high-level plotting function should have a simple example in the examples folder. This should be as simple as possible to demonstrate the method.

+ Changes (both new features and bugfixes) should be tested using `pytest`.  Add tests for your new feature to the `tests/` repo folder.

+ Please note that the code is currently in beta thus the API may change at any time. BE WARNED.

<h2>Testing</h2>

<!-- [![Build Status](https://travis-ci.com/ContextLab/quail.svg?token=hxjzzuVkr2GZrDkPGN5n&branch=master) -->

To test timecorr, install pytest (`pip install pytest`) and run `pytest` in the timecorr folder
