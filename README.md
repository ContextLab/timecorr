# Overview
The TimeCorr toolbox provides tools for computing and exploring the correlational structure of timeseries data.  There are two main functions:
* `timecorr` is used to compute dynamic correlations from a timeseries of observations, organized as a number-of-timepoints by number-of-features matrix.  The correlations may be computed _within_ a single matrix, or _across_ a list of such matrices.
* `levelup` is used to find higher order structure in the data.  Calling `levelup` converts a timeseries of observations into a timeseries of correlations _with the same number of features_.  This function may be called recursively to compute dynamic correlations ("level 1"), dynamic correlations _between_ correlations ("level 2"), dynamic correlations between correlations between correlations ("level 3"), etc.  The new features may be computed _within_ a single matrix, or a list of matrices (this returns a new single matrix or list of matrices of the same shape, but containing the higher order features), or _across_ a list of such matrices (given a list of matrices, the _across_ matrix higher-order structure comprises a single matrix).

# Installing
To install the TimeCorr toolbox directly from GitHub, type `pip install git+https://github.com/Summer-MIND/timecorr`.  You can also install by calling `pip install .` from within the TimeCorr folder (after cloning this repository).

# Requirements

* Python 2.7, 3.4+
* Numpy >= 1.10.4
* Scipy >= 1.0.0
* Seaborn >= 0.8.1
* Matplotlib >=2.0.1
* Pandas >= 0.22.0
* Hypertools >= 0.4.2

# Contributing
Start by downloading and installing the package in your Docker instance so that you can use start to play around with the toolbox.  Then check out the **Basic usage** instructions to see how to use the toolbox.  Remember that this is beta (non production-ready) code, so you are likely to encounter bugs, errors, etc.

There are many ways you can help out!  After trying out the code yourself, you could:
- Clean up documentation (e.g. if you notice typos, bad or inefficient grammar, stuff that's confusing or could be better described, etc.) you can change it (place to start: to do column of the [project board](https://github.com/Summer-MIND/timecorr/projects/1))
- Clean up or fix existing code (place to start: to do column of [project board](https://github.com/Summer-MIND/timecorr/projects/1))
- Add new code
- Add new ideas to the [issue tracker](https://github.com/Summer-MIND/timecorr/issues) (see **Tracking issues**)
- Add comments or requests for clarification to existing issues
- Pair up with another student or faculty member and help them work through the logic
- Ask good questions to help stimulate discussion and new ideas
- Do something else helpful-- there are nearly limitless ways of helping out; be creative and try to cultivate a "hacker mindset" by figuring out innovative and off-the-beaten path ways of helping out!

## Adding and tracking issues/feature requests
If you encounter a bug, or if you have an idea for a new feature (or how to modify an existing feature to make it more useful!), or if you notice some other issue or reqeust, please add an issue using the [issue tracker](https://github.com/Summer-MIND/timecorr/issues).  If you associate it with the MIND 2018 Hackathon project, it will automatically populate the [project board](https://github.com/Summer-MIND/timecorr/projects/1)'s "to do" column with a new item.

## Changing code/documentation
Start with the [project board](https://github.com/Summer-MIND/timecorr/projects/1).  If you see an existing issue in the "to do" column that you want to help out with, open the associated issue, assign it to yourself (and whoever else you're working on that issue with) and move it to the "in progress" column.

If there's no "to do" item for what you want to contribue, please add a [new issue](https://github.com/Summer-MIND/timecorr/issues) describing what you want to do.  (If you associate it with the MIND 2018 Hackathon project, his will add it automatically to the project board; assign yourself to the issue and move it from the "to do" to the "in progress" column.

Once you've defined what you want to do and moved your issue to the "in progress" column, *Fork* this repository (click the button in the upper right).  That's the version of the project that you'll be editing directly.  Clone that (forked) repository into the mount point of your Docker image (probably `~/Desktop`).  Then:
- Navigate to `/mnt/timecorr` (from within the Docker terminal)
- Set up the Summer-MIND fork of timecorr as your "upstream" repository: `git remote add upstream https://github.com/Summer-MIND/timecorr.git`
- Make sure your fork is up to date with the Summer-MIND fork: `git pull upstream master`
- Now load up your favorite coding envornment, and code away!  Remember to commit frequently!
- When you're ready to back up your code, do another `git pull upstream master` to pull in any new changes.  Then `git add` any files you changed (`git status` will tell you what's changed) and `git commit -a -m "your description of what changed"` to commit a snapshot of your code locally.  Finally, type `git pull` to put your local changes on your forked copy of timecorr.
- Once you want to add your code back to the main repository, start a new pull request (see "Pull requests" above), describe what you added, add some reviewers, and submit it!

# Basic usage
## Formatting your data
You should format your data as a Numpy array or Pandas dataframe with one row per observation and one column per feature (i.e. things you're tracking over time-- e.g. a voxel, electrode, channel, etc.).  You can then pass timecorr a single dataframe or a list of dataframes (each with the same numbers of timepoints and features).

## `timecorr`
The `timecorr` function takes your data (formatted as described above) and returns moment-by-moment correlations during the same timepoints.

### `within` mode

Timecorr's `within` mode can be used for a single data matrix or a list of data matrices.  Timecorr will return a result in the same format, but where each data matrix has number-of-timepoints rows and $\frac{n^2 - n}{2}$ features (i.e. a vectorized version of the upper triangle of each timepoint's correlation matrix).  If a list of data matrices are passed, each data matrix is compared to the average of the other data matrices (`isfc` mode) or a similarity-weighted average of the other data matrices (`wisfc` mode).  If only a single data matrix is passed, the correlations are computed with respect to the same data matrix.

### `across` "across" mode ###
Timecorr's `across` mode is for finding shared correlation across sets of observations (e.g. from different experimental participants).  If only a single data matrix is passed, `across` mode will behave the same as `within` mode.  If a list of data matrices is passed, `isfc` mode computes each matrix's correlations with respect to the average of the others, and then averages across all of those correlations.  `wisfc` mode behaves similarly, but computes weighted averages (e.g. based on inter-matrix similarities).


## `levelup`
The levelup function runs `timecorr` and then uses dimensionality reduction to project the correlations back onto the original number-of-timepoints by number-of-featuers space.  This is useful in that it prevents "dimension blowup" whereby running timecorr its own output squares the number of features-- thereby preventing the efficient exploration of higher-order correlations.  In `within` mode, the data matrix (or list of matrices) are returned in the same format (and size) as the original; in `across` mode a single number-of-timepoints by number-of-features matrix is returned.
