.. _api_ref:

.. currentmodule:: timecorr

API Reference
=============

Every function listed here has a page with its parameters, return values and a
runnable example; every dictionary has a page with its value. Unless a section names another module, the functions are
available as ``timecorr.<name>`` (for example ``tc.timecorr`` after
``import timecorr as tc``).

Dynamic correlations
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timecorr
   wcorr
   isfc
   wisfc
   autofc

``timecorr`` builds the kernel, restricts it (``include_timepoints``,
``include_zero``, ``exclude_timepoints``, and ``events``: see Events below), applies the correlation function
(``cfun``) to every subject, combines the subjects (``combine``) and reduces the
result (``rfun``). ``wcorr`` is the estimator: a kernel-weighted Pearson
correlation in which row ``t`` of the kernel weights the means, covariances and
variances used at timepoint ``t``. Version 0.3.0 changed this estimator, so its
results differ from version 0.2.0 and from Eq. 4 of Owen, Chang & Manning
(2021).

.. code-block:: python

   import numpy as np
   import timecorr as tc

   rng = np.random.default_rng(0)
   data = rng.standard_normal((100, 5))
   corrs = tc.timecorr(data, weights_params={"var": 10})   # (100, 15)

   subjects = [rng.standard_normal((100, 5)) for _ in range(4)]
   isfc = tc.timecorr(subjects, cfun=tc.isfc)               # list of 4 (100, 15) arrays
   group = tc.timecorr(subjects, cfun=tc.isfc, combine=tc.corrmean_combine)  # (100, 15)

Higher-order features
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   features
   two_path_step

``features`` computes features of every order through the two paths of Fig. 10
of Owen, Chang & Manning (2021): at each order the analysis path applies the
analysis kernel to the previous order's feeding features, and the feeding path
(named ``delta_*`` in the arguments for historical reasons) gives the features
passed to the next order. With the defaults both kernels are Gaussians whose
width is set from each order's input: the feeding kernel's standard deviation
is 5 autocorrelation widths of the input (at most a hundredth of its rows,
but at least 5 rows, which wins below 500 rows) and the analysis kernel's is that or 10 rows, whichever is
larger. The feeding path summarizes each subject's correlations with
``Communities()``; the analysis path is reduced with PCA at order 1 and with
``WeightedStrength(signed=True)`` at orders 2 and above. In the original
pipeline of the paper a delta kernel
(``eye_weights``) gives the co-fluctuations passed to the next order and PCA
reduces both paths.

.. currentmodule:: timecorr.higher_order

``two_path_step`` computes one order of the pipeline from the previous
order's feeding features. The memory estimate is in ``timecorr.higher_order``:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   estimate_memory_gb

.. code-block:: python

   data, truth = tc.simulate_nested(S=2, T=1000, set_random_seed=0)
   X = tc.features(data, order=2, cfun=tc.autofc, weights_params={"var": 33 ** 2})
   # X[1]: one (1000, 108) array per subject (PCA-reduced);
   # X[2]: one (1000, 2 * c) array per subject, c = number of communities (18 here)

   # the original two-path pipeline
   X = tc.features(subjects, order=2, weights_params={"var": 10},
                   rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
   # X[0]: the data; X[1], X[2]: one (100, 5) array per subject (PCA-reduced)

The keyword-only arguments ``delta_weights_function`` and
``delta_weights_params`` set the kernel of the feeding path (default:
``gaussian_weights``); ``eye_weights`` gives
co-fluctuations, and any other kernel gives kernel-weighted correlations
(``autofc``), shaped by ``include_timepoints``, ``include_zero`` and the
events of each order like the analysis kernel. ``rfun`` and ``delta_rfun``
accept ``"auto"`` (the defaults; analysis path: PCA at order 1 and
``WeightedStrength(signed=True)`` above; feeding path: ``Communities()``,
with the number of communities set by a shift null) and dicts keyed by order.
In ``features``, ``weights_params`` and ``delta_weights_params`` also accept
dicts keyed by order (for example ``{1: {"var": 100}, 2: {"var": 400}}``), which
set each order's analysis or feeding kernel; orders missing from such a dict use
the default. Kernel parameters of None (the default) with Gaussian kernels on
both paths are set from each order's input: the feeding kernel's standard
deviation is ``min(5 w, max(5, T' / 100))`` rows and the analysis kernel's is
that or 10 rows, whichever is larger, where ``w`` is the autocorrelation width
of the order's input (the lag at which the mean autocorrelation of its columns
falls below ``exp(-1/2)``, at least 1) and ``T'`` its number of rows. For white
data these are the fixed kernels of earlier versions at order 1 (variances 100
and 25, which can also be given: ``weights_params={"var": 100},
delta_weights_params={"var": 25}``), and recordings of up to 1000 timepoints
use kernels of at most 10 samples at every order. When no kernel parameters
and no events are given and ``include_timepoints`` is ``"all"``, each order's
feeding features are passed on at every
``floor(sd / 2)``-th row while at least 1000 rows remain, so later orders of
long recordings are computed at fewer timepoints, and every output is returned
at all timepoints by linear interpolation; ``return_info=True`` gives each
automatic kernel's ``"kernel_sd"`` (in rows of its order's input) and
``"sampling"`` (timepoints per row), and for every step ``"constant_columns"``
(the output columns that are constant over time in the returned arrays) and,
for feeding steps, ``"labels"`` (the community of each input column when the
output columns are the communities of a ``Communities`` object, otherwise
None). Giving kernel parameters for any order or
path, events, or a one-sided kernel makes every order be computed at every
timepoint. With another kernel function, or with
``eye_weights`` on the feeding path, None means that function's own default
parameters. A dict whose keys are all integers is read as such
an order map, so the parameters of a custom kernel that are keyed by integers
must be wrapped in one (for example ``{1: params, 2: params}``); only the outer
dict is examined, and each value is passed to the kernel unchanged. A dict
mixing integer and other keys, or with boolean keys (Python or NumPy), raises
``ValueError``.
``two_path_step`` takes the parameters of one order.
The default feeding reducer needs at least 4 features and 40 timepoints. Its
shift null circularly shifts each column by a lag from the n integers in
``[L, T - L]`` (``L = min(T // 4, max(10, 10 * sd))``, sd the kernel's standard
deviation ``sqrt(var)`` when its parameters have a ``"var"``, and 5 otherwise:
the width parameters of other kernels, such as ``scale`` of
``laplace_weights``, are not read): with at most n columns, the lags of a subject are distinct; with
K > n columns, each lag is used ``K // n`` or ``ceil(K / n)`` times within a
subject, and columns that share a lag keep their dependence in that surrogate:
with K = q n + r, a fraction [n q (q - 1) + 2 r q] / [K (K - 1)] of the column
pairs (1/231 at K = 22, n = 21), and one ``UserWarning`` per fit says so (giving
the number of communities ``n`` avoids the shift null). For a built-in kernel
without events, one-sided restriction or ``exclude_timepoints``, the surrogates'
squared correlations are averaged over every ``s``-th timepoint, ``s`` being the
largest step at which two rows of the kernel still overlap by ``exp(-1/4)`` (the
standard deviation, rounded down, for ``gaussian_weights``); for any other
kernel (an array, a kernel function of your own, a restricted kernel), whose
rows can differ, they are averaged over every timepoint. When the
communities found at a feeding step would emit fewer than 4 columns (at most
half its input features; one when the shift null finds no structure), that step
emits ``WeightedStrength(split=False, signed=False, center=False)`` of the same
correlations instead (one column per input feature; uncentered pair weights,
without centering or clipping), so the feeding width never
drops below 4; a ``UserWarning`` is issued only when the shift null found no
structure, and ``return_info=True`` records every fallback. The fallback also
changes calls that completed without it: on 4 independent noise features
(T 100), ``D[1]`` from ``features(x, order=1, return_delta=True)`` had 1
community column and now has 4 weighted-strength columns, and ``X[2]`` of
``features(x, order=2)`` has 8 columns instead of 2. Summary columns whose
range over time is nonzero but at most ``1e-9 * max|column|`` are
round-off; they are set to their means exactly and reported by the
constant-output warning. An explicit ``Communities()`` keeps its
communities; ``delta_rfun="weighted_strength"`` takes any number of features. Steps reduced by a summary (with
``null_combine`` and ``cfun`` ``autofc``, ``isfc`` or ``wisfc``) are computed
in blocks of rows of the ``(K, K, T)`` correlation array, so the whole array is
held only when one block covers all K rows.

The default ``cfun=isfc`` correlates each subject's features with the other
subjects' and so reflects structure shared across subjects; ``cfun=autofc``
computes within-subject correlations. On ``simulate_nested`` data (108 nodes,
T 1000, 6 subjects per call, 12 seeds; mean +- standard error, control in
parentheses), with the fixed kernels of earlier versions
(``weights_params={"var": 100}, delta_weights_params={"var": 25}``) and the
other arguments default (with the automatic kernels, ``X[2]`` is the same
there and ``D[2]`` with ``isfc`` scores .560 +- .020, control .002 +- .025):

========================  ==========  ==============================  ==============================
data (``shared=``)        ``cfun``    ``X[2]`` score                  ``D[2]`` score
========================  ==========  ==============================  ==============================
``"order2"`` (default)    ``isfc``    -.031 +- .033 (.059 +- .020)    .199 +- .021 (.013 +- .015)
``"order2"``              ``autofc``  .552 +- .023 (-.008 +- .025)    .199 +- .021 (.013 +- .015)
``"order1"``              ``isfc``    .652 +- .026 (.078 +- .010)     .293 +- .027 (.012 +- .014)
``"order1"``              ``autofc``  .700 +- .012 (-.008 +- .025)    .293 +- .027 (.012 +- .014)
========================  ==========  ==============================  ==============================

Under ``shared="order2"`` the order-1 couplings are drawn per subject, and
``h`` was not recovered from ``X[2]`` with ``isfc``; the feeding features ``D[k]``
do not depend on ``cfun``.

.. currentmodule:: timecorr

Causal features
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeaturesModel

With ``causal=True`` (keyword-only, in ``timecorr``, ``features``,
``two_path_step``, and in ``wcorr``, ``autofc``, ``isfc`` and ``wisfc``), row
``t`` of every output is computed from rows ``0..t`` of the data only. The row
is unchanged, bit for bit, when later rows are replaced by any values (NaN and
inf included) and, with the built-in kernel functions, when later rows are
removed: a row is computed with the same operations whatever the length of the
series. With a kernel array or a kernel function of your own, removal is
documented to 1e-9 for outputs that depend continuously on the correlations (no
reduction, ``"strength"``, the summaries, and linear models such as PCA once
fitted); the eigenvector-based graph measures (``"eigenvector_centrality"``,
the PageRank measures) can then differ by more at a correlation matrix whose
leading eigenvalues coincide, and the t statistics of ``tstat_combine`` where
the subjects' correlations nearly coincide agree to about 1e-9 of their
magnitude.
With ``include_zero=False`` the row is also unchanged when row ``t`` itself
changes. The calls without ``causal=True`` are not changed by it.

.. code-block:: python

   x = np.random.default_rng(0).standard_normal((300, 6))
   kernels = dict(weights_params={"var": 25}, delta_weights_params={"var": 25})

   # reducers fitted on rows 0..199, then frozen; the fitted pipeline is returned
   X, model = tc.features(x[:250], order=2, causal=True, cfun=tc.autofc,
                          fit_timepoints=200, return_model=True, **kernels)
   later = model.transform(x)         # all 300 rows, nothing fitted
   # later[k][:250] equals X[k] bit for bit; X[k][:2] is NaN (warm-up), X[2][:4] too

   # the reducers fitted again every 50 rows, each time on all rows before
   X, info = tc.features(x, order=2, causal=True, cfun=tc.autofc, rfun="PCA",
                         delta_rfun="weighted_strength", fit_timepoints=200,
                         refit_every=50, return_info=True, **kernels)
   # info[0]["refit_rows"]: [200, 250]

- **Kernels.** Both paths use backward kernels (``include_timepoints`` may be
  left out; ``"forward"`` / ``"post"`` raises ``ValueError``), and their
  parameters must be given (in ``features`` for both paths and every order):
  widths are not chosen from the data, and ``weights_params=None`` raises
  ``ValueError`` (``{}`` for a kernel without parameters). ``eye_weights``
  raises ``ValueError`` (co-fluctuations use series z-scored over the whole
  recording), as does a kernel array, or a kernel function of your own, with
  weight above the diagonal.
- **Samples.** Infinite samples and samples above 1e150 in magnitude are
  missing data, like NaN.
- **Warm-up.** A timepoint with fewer than 3 observed timepoints in its kernel
  row is a row of NaN (0 without ``causal``). ``min_samples`` (default None)
  also asks for an effective sample size ``(sum |w|)**2 / sum w**2`` over those
  timepoints; a number that a built-in kernel of the call cannot reach raises
  ``ValueError``. In ``features`` the NaN rows of an order's feeding features
  are missing data for the next order, so each order has more of them;
  ``return_info`` counts them (``"warmup_rows"``).
- **Reducers.** None and the graph measures need no fit. A reducer that is
  fitted to data (a hypertools model, ``WeightedStrength``, ``Communities``,
  and ``"auto"``, the default of ``features``) needs ``fit_timepoints=n``: it
  is fitted on rows ``0..n-1`` and then applied, unchanged, to every row
  (without it: ``ValueError``). Row ``t >= n`` is then covered by the
  guarantee; a row ``t < n`` is unchanged when rows from ``n`` on are replaced
  or removed. A model must be a linear map applied to each row by itself. The
  guarantee covers ``"PCA"``, ``"IncrementalPCA"``, ``"TruncatedSVD"``,
  ``"FastICA"`` and ``"FactorAnalysis"`` (scikit-learn's classes of those
  names, by exact class). Any other model whose fitted estimator has
  ``components_``, reproduces its training rows to 1e-9 and reduces rows one
  at a time as it reduces them together is used with one ``UserWarning`` per
  call that the guarantee does not cover what its transform does. A model
  without ``components_``, or that fails a check, raises ``ValueError``
  (``"UMAP"``, ``"KernelPCA"``, ``"Isomap"``, ``"MDS"``, ``"TSNE"``,
  ``"SpectralEmbedding"``, ``"LocallyLinearEmbedding"``). ``"PCA"`` is fitted with ``svd_solver="full"`` unless a model
  dict gives a solver. An output column of a model that is constant on the
  training rows up to round-off is frozen: at any row, a value within that
  tolerance of the column's training mean is returned as exactly that mean,
  and other values (rows that do vary along it) are returned as computed. Every
  reducer (a model, a summary, a graph measure) reads the rows without any NaN
  or infinite value only; any other row, as a smoothed row (``cfun=None``)
  that is partly NaN, is left out of the fit and is a row of NaN in the
  reduced output.
  Summary columns that are constant up to round-off are not set to their means,
  and no warning reports them.
- **Refits.** ``refit_every=m`` (``features``) fits the reducers again at rows
  ``n + j * m``, each time on all rows before; the rows from one fit to the
  next are those of a call with that fit alone. The widths, the number of
  communities and the choice of an automatic feeding step are those of the
  first fit (a communities step looks for its number of communities of 2 or
  more nodes with the first fit's number of clusters and up to 5 more or
  fewer); a later fit that cannot have them is not used (the fit before
  stays; ``"refit_kept_previous"`` of ``return_info`` lists its row, and one
  ``UserWarning`` names it). The
  components of a model and the communities of ``Communities`` are matched to
  those of the fit before.
- **Not covered.** ``cfun=tc.wisfc`` raises ``ValueError`` (its default subject
  weights use the whole recording; ``functools.partial(tc.wisfc,
  subject_weights=...)`` is accepted); ``events`` must be boundaries (features
  to segment raise ``ValueError``), and that the boundaries are known at their
  own time is the caller's responsibility. A function of your own (``cfun``,
  ``combine``, ``rfun``, ``delta_rfun``) is called as without ``causal``, with one
  ``UserWarning`` that the guarantee does not cover what it does. The decoders
  have no causal form.

``features(..., return_model=True)`` returns the fitted pipeline as a
``FeaturesModel``; ``model.transform(data)`` computes the features of other
data with the same number of columns, causally and without fitting anything.
With ``refit_every`` the model holds the last fit. A model can be pickled when
the arguments of the call can (no lambdas or local functions). The
:doc:`causal tutorial <tutorial/causal>` shows these on daily stock returns.

Events
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   segment

``segment`` divides a timepoints-by-features timeseries into events with a
hidden Markov model (``hmmlearn``, installed on demand) and returns the block
starts, which ``timecorr`` accepts as ``events`` to confine its kernel to
blocks: the kernel gives no weight to timepoints outside the current block.
``events`` also accepts boundary indices directly, a features array (segmented
with ``events_params``), or a list with one entry per subject; in ``features``
it may be a dict keyed by order. The decoders accept ``events`` too:
``timepoint_decoder``, ``weighted_timepoint_decoder`` and
``weighted_timepoint_decoder_ec`` restrict the analysis kernel at levels 1 and
above and leave level 0 unchanged; ``pca_decoder`` restricts its level-0
smoothing kernel and takes one specification (not a dict keyed by order). In
the decoders the events are resolved once, on the full subject list, before the
folds are formed; a per-subject entry stays with its subject in every fold; the
boundaries are external information supplied by the caller, used as given in
every fold and not estimated from held-out data; and segmentation does not
advance NumPy's global random state. The :doc:`tutorial <tutorial/events>`
shows each form.

.. code-block:: python

   boundaries = tc.segment(embedding, n=6)                 # e.g. array([0, 48, ...])
   corrs = tc.timecorr(data, events=boundaries)
   corrs = tc.timecorr(data, events=[40, 70])              # blocks 0-39, 40-69, 70 to the end
   X = tc.features(data, order=2, events={1: boundaries, 2: [50]})

.. currentmodule:: timecorr.events

The building blocks are in ``timecorr.events``:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   check_boundaries
   block_labels
   resolve_events
   resolve_events_by_order
   mask_kernel
   distinct_event_sets

.. currentmodule:: timecorr

Kernels
-------

Each kernel function returns a ``T x T`` matrix whose row ``t`` is the kernel
used to estimate at timepoint ``t``. Every kernel function accepts
``direction`` (``None``, ``"backward"`` or ``"forward"``) and ``include_zero``.

:func:`timecorr`, :func:`features`, :func:`two_path_step` and the decoders do
not call the built-in kernel functions below. They compute the same weights
128 rows at a time, over the columns where those rows can have non-zero
weight, with ``include_timepoints``, ``include_zero``, ``exclude_timepoints``
and ``events`` applied, so with the correlation functions ``autofc``,
``isfc`` and ``wisfc`` (or ``cfun=None``) no ``T x T`` array is allocated and
memory grows linearly in ``T``. The correlations equal those computed from
the ``T x T`` matrix. Run time grows linearly in ``T`` once ``T`` is well
above the largest lag with a non-zero weight in float64 (192 samples for a
Gaussian of variance 25, 385 at variance 100); the uniform kernel has weight
at every lag, so its run time grows with ``T**2``. A kernel function of your
own is called, and its ``T x T`` matrix is used; a correlation function of
your own receives ``T x T`` matrices, built from the built-in kernels too.
The memory guards (``max_memory_gb``) and :func:`estimate_memory_gb` count
those matrices.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gaussian_weights
   laplace_weights
   mexican_hat_weights
   t_weights
   boxcar_weights
   uniform_weights
   eye_weights
   one_sided

.. code-block:: python

   w = tc.gaussian_weights(100, {"var": 10}, direction="backward", include_zero=False)
   backward = tc.timecorr(data, weights_params={"var": 10}, include_timepoints="backward")
   # UserWarning: 2 of 100 timepoints have fewer than 3 non-zero kernel weights;
   # their correlations are undefined and set to 0.

.. _default_kernel_params:

Default kernel parameters
-------------------------

Each kernel function uses its default dictionary when ``params`` is omitted,
and ``timecorr`` and ``features`` use it when ``weights_params`` is None.
Distances ``|t - tau|`` are in samples.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Value
     - Kernel and meaning of each key
   * - :data:`gaussian_params`
     - ``{"var": 100}``
     - :func:`gaussian_weights`.  ``var``: variance of the Gaussian, in
       squared samples (its standard deviation is ``sqrt(var)`` samples).
   * - :data:`laplace_params`
     - ``{"scale": 100}``
     - :func:`laplace_weights`.  ``scale``: scale ``b`` of the Laplace
       density, in samples; the weight is ``exp(-|t - tau| / b) / (2 b)``.
   * - :data:`eye_params`
     - ``{}``
     - :func:`eye_weights`.  No keys: the delta kernel has no parameters.
   * - :data:`t_params`
     - ``{"df": 100}``
     - :func:`t_weights`.  ``df``: degrees of freedom of Student's t
       density (unit scale, lags in samples); smaller values give heavier
       tails.
   * - :data:`mexican_hat_params`
     - ``{"sigma": 10}``
     - :func:`mexican_hat_weights`.  ``sigma``: width of the wavelet, in
       samples; weights are positive within ``sigma`` samples of ``t`` and
       negative beyond.
   * - :data:`uniform_params`
     - ``{}``
     - :func:`uniform_weights`.  No keys: the uniform kernel has no
       parameters.
   * - :data:`boxcar_params`
     - ``{"width": 10}``
     - :func:`boxcar_weights`.  ``width``: total width of the window, in
       samples; the weight is 1 where ``|t - tau| < width / 2`` and 0
       elsewhere.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gaussian_params
   laplace_params
   eye_params
   t_params
   mexican_hat_params
   uniform_params
   boxcar_params

The dictionaries are shared module-level objects: modifying one changes the
default for every later call in the session.  Passing a new dictionary (for
example ``weights_params={"var": 10}``) changes only that call.

Combining subjects
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   null_combine
   corrmean_combine
   mean_combine
   tstat_combine

Dimensionality reduction and smoothing
--------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reduce
   smooth

``reduce`` accepts a hypertools / scikit-learn model name (for example
``"PCA"``, ``"FastICA"``, ``"FactorAnalysis"``, ``"IncrementalPCA"``), a dict
such as ``{"model": "PCA", "params": {"whiten": True}}``, a graph measure
(``"eigenvector_centrality"``, ``"pagerank_centrality"``,
``"pagerank_centrality_abs"``, ``"strength"``), a
squared-correlation summary (below), or a callable that maps a list of arrays
to a list of arrays.  PageRank on signed correlations
(``"pagerank_centrality"``) is not always defined:
for some valid correlation matrices its linear system is singular.  Those
timepoints get the minimum-norm least-squares solution when it solves the
system, and otherwise the uniform vector 1/K, with one ``UserWarning`` per
``reduce`` call giving the counts (see ``reduce``).
``"pagerank_centrality_abs"`` is PageRank on the absolute values ``|r|``: it
is defined for every finite matrix, never falls back and never warns, and it
equals ``"pagerank_centrality"`` up to round-off when no weight is negative.

Squared-correlation summaries
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   WeightedStrength
   Communities

The summaries (module ``timecorr.summaries``) turn each subject's vectorized
correlations into a few columns per node (``WeightedStrength``: weighted sums
of squared correlations) or per community of nodes (``Communities``: mean
squared correlation within each community). They are given to ``reduce`` or
``timecorr`` as ``rfun``, as instances or as the strings
``"weighted_strength"``, ``"weighted_strength_split"`` and ``"communities"``;
``fit`` and ``transform`` fit the weights and communities to one set of
correlations and apply them to another (when the new list has as many
units as the fit, ``transform`` matches them by position: the unit at
position u gets the weights fitted to the unit at position u, whether or not
it is the same subject; with a different number of units every unit gets
the mean over the fitted units; ``transform(corrs, fit_index=...)`` names the
fitted unit instead: an int for every unit, a list with one fitted unit per
unit, or ``"mean"``), and ``fits_`` records the fits
(calling an instance keeps only that call's fit; ``fit`` and each step of
``features`` and ``two_path_step`` append one). ``WeightedStrength(center=False)``
uses the pair weights without centering. With ``center=True``, the default,
the pair weights are centered on the mean of their lower cluster (the lower
group of the split of the sorted weights into two groups with the least
within-group sum of squares; the split does not depend on the scale of the
weights, and near ties within a relative 1e-9 go to the smallest lower group)
and clipped at 0; with ``center="median"`` they are centered on their median.
``Communities`` centers its matrix ``C`` on the median. Both give zero
magnitude columns for K >= 4 nodes when the coupling is equal for every pair.
When more than half the pairs lie within one community, the median is a
weight of that community, so median centering removes most of its weights
and zeroes those of a more weakly coupled smaller community; the lower
cluster keeps both (see the ``WeightedStrength`` Notes for the measured
comparison). ``n_null`` sets the number of shift-null surrogates of the
automatic count of communities (default 5). ``Communities(signed=True)``
emits a magnitude and a signed column per community, both computed from the
correlations within that community; on the feeding path of ``features``, the
next order's correlations include the correlation between each community's
two columns.

.. code-block:: python

   corrs = tc.timecorr(data, cfun=tc.autofc, weights_params={"var": 25})
   comm = tc.Communities(n=6)
   cols = comm(corrs)              # one (T, c) array per subject
   labels = comm.fits_[0]["labels"]

Simulation
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   simulate_data
   simulate_multi_order
   simulate_nested
   simulate_hierarchy
   min_features
   carrier_reducer
   carrier_columns
   nested_template

.. code-block:: python

   sim, truth = tc.simulate_data(datagen="block", K=8, T=200, order=2,
                                 return_corrs=True, set_random_seed=0)
   # truth["corrs"]: (200, 10) correlations among the 4 block products

Decoders
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timepoint_decoder
   weighted_timepoint_decoder

.. currentmodule:: timecorr.decoding

Further decoder functions are in ``timecorr.decoding`` (and importable from
``timecorr.helpers``):

.. autosummary::
   :toctree: generated/
   :nosignatures:

   weighted_timepoint_decoder_ec
   optimize_weights
   folding_levels
   folding_levels_ec
   pca_decoder
   decoder
   weight_corrs
   get_xval_assignments

.. currentmodule:: timecorr

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mat2vec
   vec2mat
   format_data
   iseven
   isodd

.. currentmodule:: timecorr.helpers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   r2z
   z2r
   plot_weights

.. currentmodule:: timecorr.datasets

Example datasets
----------------

The data of the real-data tutorials are in the submodule ``timecorr.datasets``
(``tc.datasets`` after ``import timecorr as tc``). Files are downloaded once and
kept under the directory named by the environment variable ``TIMECORR_DATA``
(default ``~/timecorr_data``).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fetch
   cache_dir
   story_listening
   industry_returns

.. code-block:: python

   import timecorr as tc

   data = tc.datasets.story_listening()        # 36 participants, each (300, 100)
   returns = tc.datasets.industry_returns()    # DataFrame: trading days x 49 industries
   path = tc.datasets.fetch("http://mocap.cs.cmu.edu/subjects/86/86_02.amc",
                            "cmu_mocap/86_02.amc")

.. currentmodule:: timecorr

Videos and plot style
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   render_video
   show_video
   media_dir

``render_video`` draws the frames of a video in parallel worker processes and
encodes them as H.264; ``show_video`` displays the file in a notebook, linked
inside a Sphinx documentation tree and embedded elsewhere; ``media_dir`` gives
the directory for the two cases. They import joblib, imageio-ffmpeg and IPython
on first use (``pip install "timecorr[tutorials]"``).

.. code-block:: python

   def draw(frames, paths):                 # called once per run of frames, in a worker
       fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
       line, = ax.plot(x, y)
       for i, path in zip(frames, paths):
           line.set_data(x[:i + 1], y[:i + 1])
           fig.savefig(path)
       plt.close(fig)

   video = tc.render_video(draw, len(x), tc.media_dir() / "line.mp4", fps=30)
   tc.show_video(video)

The plot style of the tutorials (the Okabe-Ito colors, no top and right axes
lines, legends without frames) is the matplotlib style ``timecorr.tutorial``;
importing timecorr does not apply it:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.style.use("timecorr.tutorial")
   COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

.. currentmodule:: timecorr

See Also
--------

* :doc:`tutorials`: the tutorial gallery
* `GitHub repository <https://github.com/ContextLab/timecorr>`_: source code and issues
* `Owen, Chang & Manning (2021) <https://doi.org/10.1038/s41467-021-25876-x>`_: the paper describing the method
