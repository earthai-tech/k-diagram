==================================
Uncertainty & Diagnostic Commands
==================================

This page details the CLI commands related to visualizing uncertainty, 
drift, coverage, and other model diagnostics.

plot-interval-width
-------------------

Synopsis
^^^^^^^^

Visualize interval width (upper − lower) as a polar scatter plot. 
Optionally color by a third column.

.. code-block:: text

   kdiagram plot-interval-width INPUT
     --q-cols q10,q90
     [--z-col q50]
     [--theta-col THETA]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--figsize W,H]
     [--title TITLE]
     [--cmap CMAP]
     [--s SIZE]
     [--alpha ALPHA]
     [--show-grid | --no-show-grid]
     [--cbar | --no-cbar]
     [--mask-angle | --no-mask-angle]
     [--dropna | --no-dropna]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table (CSV/Parquet/…).

``--q-cols LOW,UP`` (or ``--q-cols LOW UP``)
  Two quantile columns (lower, upper).

``--z-col COL``
  Optional color column; defaults to interval width.

``--theta-col COL``
  Column for NaN-alignment (ordering currently by row index).

``--acov``
  Angular coverage span.

``--figsize``
  Figure size as W,H or WxH.

``--title``
  Plot title.

``--cmap``
  Matplotlib colormap.

``--s``
  Marker size.

``--alpha``
  Marker transparency.

``--show-grid / --no-show-grid``
  Toggle grid.

``--cbar / --no-cbar``
  Toggle colorbar.

``--mask-angle / --no-mask-angle``
  Toggle angular tick labels.

``--dropna / --no-dropna``
  Drop rows with NaN in essential columns.

``--savefig PATH``
  Save instead of show.

Notes
^^^^^

* Angles map linearly over the chosen span (``--acov``), using row order 
  after NaN filtering.
* ``--q-cols`` accepts either a single comma token or two space tokens.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-interval-width data.csv --q-cols q10,q90 --savefig iw.png

   kdiagram plot-interval-width data.parquet --q-cols q10 q90 --z-col q50 --cbar

plot-interval-consistency
-------------------------

Synopsis
^^^^^^^^

Compute temporal consistency (CV or Std) of interval widths across lists 
of lower/upper columns, then plot per location on polar axes.

.. code-block:: text

   kdiagram plot-interval-consistency INPUT
     --qlow-cols q10_2023 q10_2024 q10_2025
     --qup-cols  q90_2023 q90_2024 q90_2025
     [--q50-cols q50_2023 q50_2024 q50_2025]
     [--theta-col THETA]
     [--use-cv | --no-use-cv]
     [--cmap CMAP]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--title TITLE]
     [--figsize W,H]
     [--s SIZE]
     [--alpha ALPHA]
     [--show-grid | --no-show-grid]
     [--mask-angle | --no-mask-angle]
     [--dropna | --no-dropna]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table (CSV/Parquet/…).

``--qlow-cols`` (alias ``--q10-cols``)
  Lower columns list (CSV or space separated).

``--qup-cols`` (alias ``--q90-cols``)
  Upper columns list.

``--q50-cols``
  Optional median columns list (colors by average Q50 if present).

``--theta-col``
  Column for NaN-alignment (ordering currently by row index).

``--use-cv / --no-use-cv``
  Use coefficient of variation or standard deviation.

Remaining style/figure flags as in ``plot-interval-width``.

Notes
^^^^^

* ``--qlow-cols``/``--qup-cols`` must have the same length; if ``--q50-cols`` 
  provided, it must match too.
* Accepts CSV tokens (``a,b,c``) or space tokens (``a b c``).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-interval-consistency data.csv \
     --q10-cols q10,q10_2024,q10_2025 \
     --q90-cols q90,q90_2024,q90_2025 \
     --use-cv --savefig consistency.png

plot-anomaly-magnitude
----------------------

Synopsis
^^^^^^^^

Show anomalies where actual falls below the lower bound or above the upper 
bound, colored by magnitude of violation.

.. code-block:: text

   kdiagram plot-anomaly-magnitude INPUT
     --actual-col ACTUAL
     --q-cols LOW,UP
     [--theta-col THETA]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--title TITLE]
     [--figsize W,H]
     [--cmap-under CMAP] [--cmap-over CMAP]
     [--s SIZE] [--alpha ALPHA]
     [--show-grid | --no-show-grid]
     [--cbar | --no-cbar]
     [--mask-angle | --no-mask-angle]
     [--dropna | --no-dropna]
     [--verbose N]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--actual-col``
  Observed/ground-truth column.

``--q-cols LOW,UP``
  Lower/Upper bound columns.

Plus style/figure flags as shown.

Notes
^^^^^

* Under-predictions use one colormap (e.g., ``Blues``), 
  over-predictions another (e.g., ``Reds``).
* If no anomalies, the plot is empty with a notice 
  (and optional summary if verbose).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-anomaly-magnitude df.csv --actual-col y --q-cols q10,q90 --cbar

plot-temporal-uncertainty
-------------------------

Synopsis
^^^^^^^^

Plot one or more series as polar scatter (quantiles or arbitrary columns). 
Optional per-series min–max normalization.

.. code-block:: text

   kdiagram plot-temporal-uncertainty INPUT
     --q-cols colA colB [colC ...] | --q-cols auto
     [--theta-col THETA]
     [--names NAME1 NAME2 ...]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--figsize W,H]
     [--title TITLE]
     [--cmap CMAP]
     [--normalize | --no-normalize]
     [--show-grid | --no-show-grid]
     [--alpha ALPHA] [--s SIZE]
     [--dot-style MARKER]
     [--legend-loc LOC]
     [--mask-angle | --no-mask-angle]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--q-cols``
  ``auto`` (detect quantiles) or an explicit list of columns (CSV or space).

``--theta-col``
  Column for NaN-alignment (ordering currently by row index).

``--names``
  Legend names (CSV or space).

Styling options as listed.

Notes
^^^^^

* With ``--normalize``, each series is scaled to [0, 1] independently.
* Hide radial tick labels by default (values may be on different scales).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-temporal-uncertainty df.csv --q-cols q10 q50 q90 --normalize --savefig tu.png

plot-uncertainty-drift
----------------------

Synopsis
^^^^^^^^

Ring lines showing how (normalized) interval widths change across multiple time steps.

.. code-block:: text

   kdiagram plot-uncertainty-drift INPUT
     --qlow-cols q10_2023 q10_2024 q10_2025
     --qup-cols  q90_2023 q90_2024 q90_2025
     [--theta-col THETA]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--base-radius R0]
     [--band-height H]
     [--cmap CMAP] [--label LABEL]
     [--alpha ALPHA] [--figsize W,H]
     [--title TITLE]
     [--show-grid | --no-show-grid]
     [--show-legend | --no-show-legend]
     [--mask-angle | --no-mask-angle]
     [--dropna | --no-dropna]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--qlow-cols / --qup-cols``
  Time lists of lower and upper columns.

``--theta-col``
  NaN-alignment helper.

Geometry/styling as shown.

Notes
^^^^^

* Widths are normalized by the global maximum across all steps to compare rings.
* Legend title can be customized via ``--label``.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-uncertainty-drift df.csv \
     --qlow-cols q10_2023 q10_2024 \
     --qup-cols  q90_2023 q90_2024 \
     --title "Uncertainty Drift" --savefig drift_rings.png

plot-model-drift
----------------

Synopsis
^^^^^^^^

Polar bar chart summarizing how an uncertainty metric 
(default: mean width Q90−Q10) increases with forecast horizon.

.. code-block:: text

   kdiagram plot-model-drift INPUT
     [--q10-cols q10_h1 q10_h2 ... --q90-cols q90_h1 q90_h2 ...]
     | [--q-cols q10_h1,q90_h1 q10_h2,q90_h2 ...]
     [--horizons H1 H2 ...]
     [--color-metric-cols RMSE_h1 RMSE_h2 ...]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--value-label LABEL]
     [--cmap CMAP]
     [--figsize W,H]
     [--title TITLE]
     [--show-grid | --no-show-grid]
     [--annotate | --no-annotate]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

Either provide paired quantiles per horizon (``--q-cols``) or separate lists 
(``--q10-cols``/``--q90-cols``).

``--horizons``
  Labels for angular ticks; generated if omitted.

``--color-metric-cols``
  Color bars by average of these columns (instead of widths).

Remaining style options as shown.

Notes
^^^^^

* With partial spans (``--acov`` ≠ ``full``), radii are scaled to fit the sector.
* Annotations display raw mean widths (or metric).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-model-drift df.csv \
     --q10-cols q10_h1 q10_h2 q10_h3 \
     --q90-cols q90_h1 q90_h2 q90_h3 \
     --horizons 1 2 3 --savefig model_drift.png

plot-velocity
-------------

Synopsis
^^^^^^^^

Plot temporal velocity (first differences over consecutive columns) in polar.

.. code-block:: text

   kdiagram plot-velocity INPUT
     --q50-cols col_t1 col_t2 col_t3 ...
     [--theta-col THETA]
     [--cmap CMAP]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--normalize | --no-normalize]
     [--use-abs-color | --no-use-abs-color]
     [--figsize W,H]
     [--title TITLE]
     [--s SIZE] [--alpha ALPHA]
     [--show-grid | --no-show-grid]
     [--cbar | --no-cbar]
     [--mask-angle | --no-mask-angle]
     [--savefig PATH]

Arguments
^^^^^^^^^

``--q50-cols``
  Ordered columns used to compute successive differences.

``--use-abs-color``
  Color magnitude by absolute velocity.

Other styling as usual.

Notes
^^^^^

* Angles follow row order; values are derived from adjacent column differences.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-velocity df.csv --q50-cols q50_2023 q50_2024 q50_2025 --savefig vel.png

plot-coverage
-------------

Synopsis
^^^^^^^^

Compute and visualize aggregated coverage scores for one or more models 
(each model can be a single prediction column or a set of quantile columns).

.. code-block:: text

   kdiagram plot-coverage INPUT
     --y-true COL
     --model NAME:col1 --model NAME2:colA
     [--names N1 N2 ...]
     [--q-levels q1,q2,...]
     [--kind {line,bar,pie,radar}]
     [--cmap CMAP]
     [--figsize W H]
     [--title TITLE]
     [--savefig PATH]
     [-v VERBOSE]

Arguments
^^^^^^^^^

``INPUT``
  Input table (positional, or ``-i/--input``).

``--y-true`` (alias ``--true-col``)
  Ground-truth column.

``--model``
  Repeatable spec: ``NAME:col1[,col2,...]`` (point or quantile-set).

``--names``
  Override model names (CSV or space).

``--q-levels``
  Quantile levels if models are q-sets (e.g., ``0.1,0.5,0.9``).

``--kind``
  Chart type.

``--cmap``, ``--figsize``, ``--title``, ``--savefig``, ``-v``.

Notes
^^^^^

* If a model is a single column, it’s treated as a point estimate.
* If multiple columns are supplied for a model, coverage is computed 
vs the provided quantiles and ``--q-levels``.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-coverage data.csv \
     --y-true actual \
     --model M1:q10,q50,q90 \
     --model M2:q10_2024,q50,q90_2024 \
     --names M1 M2 --kind bar --savefig coverage.png

plot-coverage-diagnostic
------------------------

Synopsis
^^^^^^^^

Point-wise coverage diagnostic on polar (scatter or bars), with optional 
background gradient and average-coverage line.

.. code-block:: text

   kdiagram plot-coverage-diagnostic INPUT
     --actual-col ACTUAL
     --q-cols LOW,UP
     [--theta-col THETA]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--figsize W,H]
     [--title TITLE]
     [--show-grid]
     [--cmap CMAP]
     [--alpha ALPHA] [--s SIZE]
     [--as-bars]
     [--coverage-line-color COLOR]
     [--buffer-pts N]
     [--fill-gradient]
     [--gradient-size N]
     [--gradient-cmap CMAP]
     [--gradient-levels L1,L2,...]
     [--mask-angle]
     [--savefig PATH]
     [-v VERBOSE]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--actual-col`` (alias ``--actual``)
  Observed/ground-truth column.

``--q-cols LOW,UP``
  Interval columns.

Plot styling and gradient options as shown.

Notes
^^^^^

* Use ``--as-bars`` to switch from scatter to bars.
* ``--gradient-levels`` accepts comma-separated numeric thresholds for 
  contour-like shading.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-coverage-diagnostic df.csv --actual-col y --q-cols q10,q90 --fill-gradient --savefig diag.png

plot-radial-density-ring
------------------------

Synopsis
^^^^^^^^

Compute a radial density (KDE/histogram-like) ring from interval width, 
velocity, or a direct target series, and render as a circular band.

.. code-block:: text

   kdiagram plot-radial-density-ring INPUT
     --kind {width,velocity,direct}
     --target-cols C1 [C2 ...]
     [--title TITLE] [--r-label LABEL]
     [--figsize W,H] [--cmap CMAP]
     [--alpha ALPHA]
     [--cbar | --no-cbar]
     [--show-grid | --no-show-grid]
     [--mask-angle | --no-mask-angle]
     [--bandwidth BW]
     [--show-yticklabels | --no-show-yticklabels]
     [--savefig PATH] [--dpi DPI]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--kind``
  Choose data source (derived width, derived velocity, or direct series).

``--target-cols``
  One or more columns (CSV or space).

Remaining visual options as shown.

Notes
^^^^^

* With ``--kind width``, each pair in target-cols is treated as lower/upper; 
  with multiple columns you may compute composite density.
* ``--bandwidth`` controls KDE bandwidth (if applicable).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-radial-density-ring df.csv --kind direct --target-cols q50 --savefig ring.png

plot-polar-heatmap
------------------

Synopsis
^^^^^^^^

2D polar histogram/heatmap over radius and angle.

.. code-block:: text

   kdiagram plot-polar-heatmap INPUT
     --r-col R --theta-col THETA
     [--theta-period P]
     [--r-bins N] [--theta-bins M]
     [--statistic {count}]
     [--cbar-label LABEL]
     [--title TITLE] [--figsize W,H]
     [--cmap CMAP]
     [--mask-angle | --no-mask-angle]
     [--mask-radius | --no-mask-radius]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi DPI]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--r-col / --theta-col``
  Columns for polar coordinates.

``--theta-period``
  If provided, wraps theta to this period (e.g., ``2*pi`` or ``360`` in degrees if preconverted).

Binning, statistic, and style options as shown.

Notes
^^^^^

* Current statistic is ``count``; use bin sizes to control resolution.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-polar-heatmap df.csv --r-col r --theta-col theta --r-bins 30 --theta-bins 72 --savefig heat.png

plot-polar-quiver
-----------------

Synopsis
^^^^^^^^

Polar vector field (quiver). Each sample uses polar position (r, θ) and 
vector components (u, v).

.. code-block:: text

   kdiagram plot-polar-quiver INPUT
     --r-col R --theta-col THETA --u-col U --v-col V
     [--color-col C]
     [--theta-period P]
     [--title TITLE] [--figsize W,H]
     [--cmap CMAP]
     [--mask-angle | --no-mask-angle]
     [--mask-radius | --no-mask-radius]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi DPI]

   # (additional quiver styling may be supported via defaults)

Arguments
^^^^^^^^^

``INPUT``
  Input table.

Required columns: ``--r-col``, ``--theta-col``, ``--u-col``, ``--v-col``.

``--color-col``
  Optional argument to color arrows.

Other flags as shown.

Notes
^^^^^

* ``--theta-period`` can help if your theta is unwrapped or in a different period.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-polar-quiver df.csv --r-col r --theta-col theta --u-col u --v-col v --savefig quiver.png

plot-actual-vs-predicted
------------------------

Synopsis
^^^^^^^^

Compare actual vs predicted on polar axes (lines or points).

.. code-block:: text

   kdiagram plot-actual-vs-predicted INPUT
     --actual-col ACTUAL
     --pred-col PRED
     [--theta-col THETA]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--figsize W,H]
     [--title TITLE]
     [--line | --no-line]
     [--r-label LABEL]
     [--alpha ALPHA]
     [--show-grid | --no-show-grid]
     [--show-legend | --no-show-legend]
     [--mask-angle | --no-mask-angle]
     [--savefig PATH]

Arguments
^^^^^^^^^

``INPUT``
  Input table.

``--actual-col / --pred-col``
  Two columns to compare.

Standard polar/figure styling toggles.

Notes
^^^^^

* When ``--line`` is enabled, series are drawn as connected paths 
  (over index order).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-actual-vs-predicted df.csv --actual-col y --pred-col yhat --line --savefig avp.png