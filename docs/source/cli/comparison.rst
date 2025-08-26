==============================
Comparison & Calibration CLI
==============================

This page documents the model comparison and calibration
command-line interfaces shipped with kdiagram. These commands read
a flat table (CSV, Parquet, …), compute diagnostics, and render
figures to file or screen.

.. contents::
   :local:
   :depth: 2

General Notes
-------------

All commands accept a path to an input table either positionally or
via ``--input``. The parser infers the format from the extension
unless you override with ``--format``.

Many flags are offered in both primary and alias forms (e.g.
``--y-true`` and ``--true-col``).

You can pass model inputs as:

* ``--model NAME:col1[,col2,...]`` (repeat per model), or
* ``--pred colA[,colB,...]`` / ``--pred-cols colA[,colB,...]``
  (repeat), with optional ``--names`` to label the groups.

Use ``--savefig out.png`` to write the figure; omit to show
interactively.

plot-reliability-diagram
------------------------

Rectangular reliability (calibration) diagram. Compares predicted
probabilities to observed frequencies in bins.

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-reliability-diagram INPUT
     --y-true Y
     [--model NAME:col | --pred col[,col...] [--names ...]]...
     [--n-bins 10] [--strategy {uniform,quantile}]
     [--positive-label 1] [--class-index N]
     [--clip-probs 0,1] [--normalize-probs/--no-normalize-probs]
     [--error-bars {wilson,normal,none}] [--conf-level 0.95]
     [--show-diagonal/--no-show-diagonal]
     [--show-ece/--no-show-ece] [--show-brier/--no-show-brier]
     [--counts-panel {bottom,none}] [--counts-norm {fraction,count}]
     [--counts-alpha 0.35]
     [--figsize 9,7] [--title TITLE] [--cmap tab10]
     [--marker o] [--s 40] [--linewidth 2.0] [--alpha 0.9]
     [--connect/--no-connect] [--legend/--no-legend] [--legend-loc best]
     [--xlim 0,1] [--ylim 0,1]
     [--savefig out.png] [--dpi 300]

Required columns
^^^^^^^^^^^^^^^^

``Y``
  Ground-truth labels (binary for now).

For each model
  Either a single probability column (positive class), or a
  multi-column probability matrix from which a column is selected
  via ``--class-index`` (defaults to the last column).

Examples
^^^^^^^^

Two models, quantile binning, Wilson intervals, counts panel:

.. code-block:: bash

   kdiagram plot-reliability-diagram rel.csv \
     --y-true y \
     --pred p_m1 --pred p_m2 \
     --names "Wide" "Tight" \
     --strategy quantile --n-bins 12 \
     --error-bars wilson --counts-panel bottom \
     --show-ece --show-brier \
     --savefig reliability.png

Same with ``--model`` (names embedded):

.. code-block:: bash

   kdiagram plot-reliability-diagram rel.csv \
     --true-col y \
     --model M1:p_m1 --model M2:p_m2 \
     --savefig rel.png

Key options
^^^^^^^^^^^

``--strategy``
  ``uniform`` uses equal-width bins on [0,1]; ``quantile`` uses
  empirical quantiles of pooled predictions (falls back to
  ``uniform`` if edges collapse).

``--error-bars``
  ``wilson`` or ``normal`` CIs for observed frequency; use
  ``--conf-level`` to set confidence.

``--show-ece``, ``--show-brier``
  Append summary metrics to legend labels.

``--clip-probs`` + ``--normalize-probs``
  Gently repair near-range values before clipping to [0,1].

plot-polar-reliability
----------------------

Polar reliability (calibration spiral). Maps predicted probability
to angle (0°→90°) and observed frequency to radius (0→1). Perfect
calibration appears as a dashed spiral.

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-polar-reliability INPUT
     --y-true Y
     [--model NAME:col | --pred col ...]...
     [--n-bins 10] [--strategy {uniform,quantile}]
     [--title TITLE] [--figsize 8,8] [--cmap coolwarm]
     [--show-cbar/--no-show-cbar]
     [--show-grid/--no-show-grid] [--mask-radius/--no-mask-radius]
     [--savefig out.png] [--dpi 300]

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-polar-reliability rel.csv \
     --y-true y \
     --model Calibrated:p_m1 --model Over:p_m2 \
     --n-bins 15 --strategy uniform \
     --cmap coolwarm --savefig polar_reliability.png

Notes
^^^^^

* Diverging colormap highlights under-confidence vs
  over-confidence (observed minus predicted).
* Uses the same binning logic as the rectangular diagram.

plot-model-comparison
---------------------

Radar (spider) chart comparing multiple metrics across models.

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-model-comparison INPUT
     --y-true Y
     [--model NAME:col | --pred col]...
     [--metrics auto | MET1 [MET2 ...]]
     [--train-times t1 [t2 ...]]
     [--names N1 N2 ...]
     [--title TITLE] [--figsize 8,8]
     [--colors C1 C2 ...] [--alpha 0.7]
     [--legend/--no-legend] [--loc "upper right"]
     [--show-grid/--no-show-grid]
     [--scale {norm,min-max,std,standard,none}]
     [--lower-bound 0]
     [--savefig out.png] [--dpi 300]

Required columns
^^^^^^^^^^^^^^^^

``Y``
  Ground-truth numeric (regression) or class labels
  (classification).

One point prediction column per model
  Typical use: point estimates.

Examples
^^^^^^^^

Regression with explicit metrics and training times:

.. code-block:: bash

   kdiagram plot-model-comparison reg.csv \
     --true-col y \
     --model Lin:m1 --model Tree:m2 \
     --metrics r2 mae rmse \
     --train-times 0.1 0.5 \
     --scale norm \
     --title "Regression Model Comparison" \
     --savefig model_comparison.png

Auto metric selection (uses ``y`` type to choose sensible
defaults):

.. code-block:: bash

   kdiagram plot-model-comparison reg.csv \
     --y-true y \
     --pred m1 --pred m2 \
     --metrics auto \
     --savefig radar.png

Key options
^^^^^^^^^^^

``--metrics``
  * ``auto`` chooses defaults by target type (e.g., ``r2``,
    ``mae``, ``mape``, ``rmse`` for regression; ``accuracy``,
    ``precision``, ``recall`` for classification).
  * You can pass any scorers supported by your environment; custom
    callables are supported in the Python API (CLI uses names).

``--scale``
  * ``norm``/``min-max`` maps each axis to [0,1] across models.
  * ``std``/``standard`` uses Z-scores.
  * ``none`` plots raw values (be careful with differing scales).

``--train-times``
  Adds an extra axis (one value per model, or a single value
  broadcast to all).

plot-horizon-metrics
--------------------

Polar bar chart summarizing a primary metric (bar height) and
optional secondary metric (color) across horizons or categories
(one row per bar).

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-horizon-metrics INPUT
     --q-low COL1 [COL2 ...]
     --q-up  COL1 [COL2 ...]
     [--q50  COL1 [COL2 ...]]
     [--xtick-labels L1 [L2 ...]]
     [--normalize-radius/--no-normalize-radius]
     [--show-value-labels/--no-show-value-labels]
     [--cbar-label LABEL] [--r-label LABEL]
     [--cmap coolwarm] [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--title TITLE] [--figsize 8,8] [--alpha 0.85]
     [--show-grid/--no-show-grid] [--mask-angle/--no-mask-angle]
     [--savefig out.png] [--dpi 300] [--no-cbar]

Input expectations
^^^^^^^^^^^^^^^^^^

* Each row corresponds to a horizon/category to compare.
* ``--q-low`` and ``--q-up`` lists must have the same length; bars
  use the mean interval width across those columns for that row.
* If ``--q50`` is provided, the color encodes its row-wise mean;
  otherwise the color follows the bar height.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-horizon-metrics horizons.csv \
     --q-low  q10_s1 q10_s2 \
     --q-up   q90_s1 q90_s2 \
     --q50    q50_s1 q50_s2 \
     --xtick-labels H+1 H+2 H+3 H+4 H+5 H+6 \
     --title "Mean Interval Width Across Horizons" \
     --r-label "Mean (Q90 - Q10)" \
     --cbar-label "Mean Q50" \
     --savefig horizons.png

Input Schema Hints
------------------

Below is a minimal CSV sketch for the above commands:

.. code-block:: text

   # reliability (binary)
   y,p_m1,p_m2
   0,0.15,0.08
   1,0.62,0.44
   ...

   # model comparison (point predictions)
   y,m1,m2
   12.3,12.0,12.5
   ...

   # horizon metrics (row per horizon, columns are samples/realizations)
   q10_s1,q10_s2,q90_s1,q90_s2,q50_s1,q50_s2
   1.0,1.2,3.0,3.1,2.0,2.1
   ...

See Also
--------

:doc:`probabilistic` — PIT, CRPS, sharpness, credibility

:doc:`errors` — polar error bands, violins, ellipses

:doc:`relationship` — polar truth–prediction relationships

Feedback & Issues
-----------------

If a command’s behavior surprises you (e.g., binning fallback or
column selection), re-run with fewer options and verify input
columns. Feel free to file issues with a small CSV illustrating the
problem.