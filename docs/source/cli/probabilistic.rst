.. _cli_probabilistic:

================================
Probabilistic diagnostics (CLI)
================================

Polar diagnostics for probabilistic forecasts: PIT histograms, CRPS 
comparison, sharpness, and calibration–sharpness trade-offs.

These commands accept flexible column specs:

* **Ground truth**: ``--y-true`` or ``--true-col``.
* **Prediction groups** (choose one style, repeatable):
    * ``--model NAME:col1[,col2,...]`` (name + CSV list)
    * ``--pred col1 col2 col3`` (space-separated)
    * ``--pred-cols col1,col2,col3`` (single CSV token)
* **Quantile levels**: ``--q-levels`` or ``--quantiles`` (CSV string).
* **Names** (optional): ``--names`` supports CSV or space-separated.

All commands read a tabular file (CSV/Parquet/…) into a DataFrame. Pass 
the path as a positional input or via ``-i/--input``. Use ``--format`` 
to override detection if needed.

Shared conventions
------------------

* **Column lists** (via ``--pred``/``--pred-cols``/``--q-cols``): either 
  comma-separated in one token or multiple space-separated tokens 
  (flexible parsing).
* **Model specs** (via ``--model``): repeat the flag to add models, e.g. 
  ``--model M1:q10,q50,q90 --model M2:q10_2024,q50,q90_2024``.
* **Quantile levels**: one CSV string, e.g. ``--q-levels 0.1,0.5,0.9``. 
  Every model/group must have the same number of columns as the number of
   quantile levels.
* **Figure output**: add ``--savefig out.png`` to write the figure instead 
  of showing it interactively.
* **Booleans** use paired flags where available, e.g. 
  ``--show-grid`` / ``--no-show-grid``.


Commands
========

plot-pit-histogram
------------------

Synopsis
^^^^^^^^

.. code-block:: text

   kdiagram plot-pit-histogram INPUT
     --y-true Y_TRUE
     [--model NAME:Q1 | --pred Q1 Q2 ... | --pred-cols Q1,Q2,... | --q-cols Q1,Q2,...]
     --q-levels Q_LEVELS
     [--n-bins N] [--title TITLE] [--figsize W,H]
     [--color COLOR] [--edgecolor COLOR] [--alpha A]
     [--show-uniform-line | --no-show-uniform-line]
     [--show-grid | --no-show-grid]
     [--mask-radius | --no-mask-radius]
     [--dpi DPI] [--savefig PATH]
     [-i INPUT] [--format FMT]

Arguments
^^^^^^^^^

``INPUT / -i, --input``
  Input table path.

``--format``
  Explicit format override (e.g. ``csv``, ``parquet``).

``--y-true / --true-col``
  Ground-truth column.

One prediction group (choose one style; exactly one group)
  * ``--model NAME:Q1[,Q2,...]``
  * ``--pred Q1 Q2 ...``
  * ``--pred-cols Q1,Q2,...``
  * ``--q-cols Q1,Q2,...`` (legacy alias)

``--q-levels / --quantiles``
  Quantile levels CSV (e.g. ``0.1,0.5,0.9``).

Style

  * ``--n-bins`` (default: 10)
  * ``--title`` (default: "PIT Histogram")
  * ``--figsize`` (default: ``8,8``)
  * ``--color`` (default: "#3498DB")
  * ``--edgecolor`` (default: "black")
  * ``--alpha`` (default: 0.7)
  * ``--show-uniform-line / --no-show-uniform-line``
  * ``--show-grid / --no-show-grid``
  * ``--mask-radius / --no-mask-radius``
  * ``--dpi`` (default: 300)
  * ``--savefig PATH``

Notes
^^^^^
Computes the Probability Integral Transform (PIT) per observation from a 
single set of quantile forecasts and renders a polar histogram. The flat 
(uniform) reference line indicates perfect calibration.

Examples
^^^^^^^^

Using space-separated columns:

.. code-block:: bash

   kdiagram plot-pit-histogram demo.csv \
     --true-col actual \
     --pred q10 q50 q90 \
     --q-levels 0.1,0.5,0.9 \
     --savefig pit.png

Using a model spec:

.. code-block:: bash

   kdiagram plot-pit-histogram demo.csv \
     --y-true actual \
     --model PRED:q10,q50,q90 \
     --q-levels 0.1,0.5,0.9

plot-crps-comparison
--------------------

Synopsis
^^^^^^^^

.. code-block:: text

   kdiagram plot-crps-comparison INPUT
     --y-true Y_TRUE
     [--model NAME:Q1 ... | --pred Q1 Q2 ... --pred ...]
     --q-levels Q_LEVELS
     [--names N1 [N2 ...]]
     [--title TITLE] [--figsize W,H] [--cmap CMAP]
     [--marker M] [--s SIZE]
     [--show-grid | --no-show-grid]
     [--mask-radius | --no-mask-radius]
     [--dpi DPI] [--savefig PATH]
     [-i INPUT] [--format FMT]

Arguments
^^^^^^^^^

``INPUT / -i, --input / --format``
  As above.

``--y-true / --true-col``
  Ground-truth column.

One or more prediction groups, using either
  * repeated ``--model NAME:Q1[,Q2,...]``
  * repeated ``--pred …`` and/or ``--pred-cols …``

``--q-levels / --quantiles``
  Quantile levels CSV (must match each group’s number of columns).

``--names``
  Optional model names (CSV or space-separated). If not given, names 
  come from ``--model`` or defaults.

Style
  * ``--title`` (default: "Probabilistic Forecast Performance (CRPS)")
  * ``--figsize`` (default: ``8,8``)
  * ``--cmap`` (default: "viridis")
  * ``--marker`` (default: "o")
  * ``--s`` (default: 100)
  * ``--show-grid / --no-show-grid``
  * ``--mask-radius / --no-mask-radius``
  * ``--dpi`` (default: 300)
  * ``--savefig PATH``

Notes
^^^^^
Computes average CRPS per model (lower is better) and places each model 
as a point in polar coordinates (radius = CRPS). All models must share 
API the same quantile levels.

Examples
^^^^^^^^

Two ``--pred`` groups + explicit names:

.. code-block:: bash

   kdiagram plot-crps-comparison demo.csv \
     --true-col actual \
     --pred q10 q50 q90 \
     --pred q10_2024 q50 q90_2024 \
     --names M1 M2 \
     --q-levels 0.1,0.5,0.9 \
     --savefig crps.png

Two models via ``--model``:

.. code-block:: bash

   kdiagram plot-crps-comparison demo.csv \
     --y-true actual \
     --model M1:q10,q50,q90 \
     --model M2:q10_2024,q50,q90_2024 \
     --q-levels 0.1,0.5,0.9

plot-polar-sharpness
--------------------

Synopsis
^^^^^^^^

.. code-block:: text

   kdiagram plot-polar-sharpness INPUT
     [--model NAME:Q1 ... | --pred Q1 Q2 ... --pred ...]
     --q-levels Q_LEVELS
     [--names N1 [N2 ...]]
     [--title TITLE] [--figsize W,H] [--cmap CMAP]
     [--marker M] [--s SIZE]
     [--show-grid | --no-show-grid]
     [--mask-radius | --no-mask-radius]
     [--dpi DPI] [--savefig PATH]
     [-i INPUT] [--format FMT]

Arguments
^^^^^^^^^

``INPUT / -i, --input / --format``
  As above.

One or more prediction groups, via ``--model`` or ``--pred``.

``--q-levels / --quantiles``
  Shared quantile levels.

``--names``
  Optional model names (CSV or space-separated).

Style
  * ``--title`` (default: "Forecast Sharpness Comparison")
  * ``--figsize`` (default: ``8,8``)
  * ``--cmap`` (default: "viridis")
  * ``--marker`` (default: "o")
  * ``--s`` (default: 100)
  * ``--show-grid / --no-show-grid``
  * ``--mask-radius / --no-mask-radius``
  * ``--dpi`` (default: 300)
  * ``--savefig PATH``

Notes
^^^^^
Plots models by average interval width (sharpness) as the radial coordinate. 
Lower radius ⇒ sharper forecasts. Quantile levels are used to derive 
the lower/upper bounds per sample.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-polar-sharpness demo.csv \
     --model A:q10,q50,q90 \
     --model B:q10_2024,q50,q90_2024 \
     --q-levels 0.1,0.5,0.9 \
     --savefig sharpness.png

plot-calibration-sharpness
--------------------------

Synopsis
^^^^^^^^

.. code-block:: text

   kdiagram plot-calibration-sharpness INPUT
     --y-true Y_TRUE
     [--model NAME:Q1 ...]
     --q-levels Q_LEVELS
     [--names N1 [N2 ...]]
     [--title TITLE] [--figsize W,H] [--cmap CMAP]
     [--marker M] [--s SIZE]
     [--show-grid | --no-show-grid]
     [--mask-radius | --no-mask-radius]
     [--dpi DPI] [--savefig PATH]
     [-i INPUT] [--format FMT]

Arguments
^^^^^^^^^

``INPUT / -i, --input / --format``
  As above.

``--y-true / --true-col``
  Ground-truth column.

One or more models via ``--model NAME:Q1[,Q2,...]``.

``--q-levels / --quantiles``
  Quantile levels CSV.

``--names``
  Optional names (CSV or space-separated).

Style
  * ``--title`` (default: "Calibration vs. Sharpness Trade-off")
  * ``--figsize`` (default: ``8,8``)
  * ``--cmap`` (default: "viridis")
  * ``--marker`` (default: "o")
  * ``--s`` (default: 150)
  * ``--show-grid / --no-show-grid``
  * ``--mask-radius / --no-mask-radius``
  * ``--dpi`` (default: 300)
  * ``--savefig PATH``

Notes
^^^^^
Quarter-circle plot:

* **Angle (θ)** encodes calibration error (KS distance of PIT values from uniform). 
  0 ⇒ perfectly calibrated; larger ⇒ worse calibration.
* **Radius (r)** encodes sharpness (average width). Lower ⇒ sharper.

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-calibration-sharpness demo.csv \
     --true-col actual \
     --model Good:q10,q50,q90 \
     --model Wide:q10_2024,q50,q90_2024 \
     --q-levels 0.1,0.5,0.9 \
     --names Good Wide \
     --savefig cal_sharp.png

plot-credibility-bands
----------------------

Synopsis
^^^^^^^^

.. code-block:: text

   kdiagram plot-credibility-bands INPUT
     --q-cols LOW MED UP
     --theta-col THETA
     [--theta-period P] [--theta-bins K]
     [--title TITLE] [--figsize W,H] [--color COLOR]
     [--show-grid | --no-show-grid]
     [--mask-radius | --no-mask-radius]
     [--dpi DPI] [--savefig PATH]
     [-i INPUT] [--format FMT]

Arguments
^^^^^^^^^

``INPUT / -i, --input / --format``
  As above.

``--q-cols``
  Three columns (lower, median (Q50), upper), e.g. ``--q-cols q10 q50 q90`` 
  (CSV or space-separated accepted).

``--theta-col``
  Column to bin on for the angular axis (e.g., month, hour, or any cyclic driver).

``--theta-period``
  Wrap period for cyclic variables (e.g. 12 for months, 24 for hours). 
  If omitted, the column range is scaled to [0, 2π].

``--theta-bins``
  Number of angular bins (default: 24).

Style

  * ``--title`` (default: "Forecast Credibility Bands")
  * ``--figsize`` (default: ``8,8``)
  * ``--color`` (default: "#3498DB")
  * ``--show-grid / --no-show-grid``
  * ``--mask-radius / --no-mask-radius``
  * ``--dpi`` (default: 300)
  * ``--savefig PATH``

Notes
^^^^^
Displays mean median forecast per angular bin and a shaded band between mean 
lower/upper quantiles. Useful to visualize conditional structure and 
uncertainty versus a driver (seasonality, time of day, etc.).

Examples
^^^^^^^^

.. code-block:: bash

   kdiagram plot-credibility-bands demo.csv \
     --q-cols q10 q50 q90 \
     --theta-col month \
     --theta-period 12 \
     --theta-bins 12 \
     --title "Seasonal Forecast Credibility" \
     --savefig cred_bands.png

See also
========

:ref:`cli-uncertainty` for interval consistency, coverage, temporal 
uncertainty, and drift plots.

Python API: ``kdiagram.plot.probabilistic`` for the function docstrings 
and usage in notebooks or scripts.
