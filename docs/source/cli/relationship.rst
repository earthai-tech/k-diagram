==============================
Relationship (Polar) Commands
==============================

Overview
========
These commands visualize how predictions relate to the truth on
polar axes. They help you spot bias, heteroscedasticity, and other
systematic patterns in a compact, cyclical view.

Command summary
---------------

- ``plot-relationship``:
  Truth vs. one or more **point prediction** series on a polar
  scatter.

- ``plot-conditional-quantiles``:
  Truth vs. a **quantile set** (e.g., q10/q50/q90) with shaded
  uncertainty bands.

- ``plot-residual-relationship``:
  **Residuals vs. predictions** per model on polar axes.

- ``plot-error-relationship``:
  **Errors vs. true values** per model on polar axes.

Common CLI patterns
-------------------

- Ground truth can be passed with either flag::

    --y-true actual    # or: --true-col actual

- Provide point-pred columns with **either**:

  * Repeating ``--pred`` / ``--pred-cols`` (CSV or space-separated
    tokens), one group per model, e.g.::

      --pred q50           --pred q50_2024

  * Or named specs via ``--model`` (repeatable)::

      --model M1:q50      --model M2:q50_2024

- Provide quantile sets (for *conditional quantiles*) with one group
  of columns (e.g. ``q10,q50,q90``) and the matching levels via
  ``--q-levels`` / ``--quantiles`` (e.g. ``0.1,0.5,0.9``).

- ``--names`` accepts CSV **or** space-separated tokens and overrides
  auto names inferred from ``--model`` or group order.

- ``--figsize`` accepts ``W,H`` (e.g. ``8,8``) or ``WxH`` (e.g. ``8x8``).

- Any command can save a figure with ``--savefig out.png`` (and set
  DPI via ``--dpi``). If ``--savefig`` is omitted, the figure is shown.

.. _cli-plot-relationship:

plot-relationship
=================

Synopsis
--------
.. code-block:: bash

  kdiagram plot-relationship INPUT
    --y-true Y_TRUE
    [--pred COLS ... | --model NAME:COLS [--model ...] | --pred-cols COLS ...]
    [--names NAMES ...]
    [--theta-offset THETA]
    [--theta-scale {proportional,uniform}]
    [--acov {default,half_circle,quarter_circle,eighth_circle}]
    [--title TITLE] [--figsize W,H] [--cmap CMAP]
    [--s S] [--alpha ALPHA]
    [--legend | --no-legend]
    [--show-grid | --no-show-grid]
    [--xlabel XLABEL] [--ylabel YLABEL]
    [--z-values COL | --z-label LABEL]
    [--dpi DPI] [--savefig PATH]
    [-i INPUT] [--format FORMAT]

Arguments
---------

- **INPUT** / ``-i, --input``: Path to input table (CSV/Parquet, etc.).
  Use ``--format`` to force the loader.

- ``--y-true, --true-col`` (required): Name of the ground-truth column.

- ``--pred`` / ``--pred-cols`` (repeatable): Prediction columns per
  model (CSV or space-separated). Use multiple times to add models.

- ``--model`` (repeatable): Named spec in the form
  ``NAME:col1[,col2,...]`` for each model. For point predictions,
  a single column per model is typical (e.g. ``q50``).

- ``--names``: Model names (CSV or space-separated). Defaults to
  names derived from ``--model`` or auto-generated.

- ``--theta-offset``: Angular shift (radians) applied to all points
  after mapping.

- ``--theta-scale``: ``proportional`` (value-aware) or ``uniform``
  (index/order-based) angle mapping from ``y_true``.

- ``--acov``: Angular coverage (span). One of:
  ``default`` (``2π``), ``half_circle`` (``π``),
  ``quarter_circle`` (``π/2``), ``eighth_circle`` (``π/4``).

- ``--title`` / ``--figsize`` / ``--cmap`` / ``--s`` / ``--alpha``:
  Figure title, size, colormap, marker size, and marker alpha.

- ``--legend`` / ``--no-legend``: Toggle legend.

- ``--show-grid`` / ``--no-show-grid``: Toggle polar grid.

- ``--xlabel`` / ``--ylabel``: Axis labels. If omitted, sensible
  defaults are used.

- ``--z-values``: Column to use for angular tick labels (e.g., month).

- ``--z-label``: Text label describing the ``--z-values``.

- ``--dpi`` / ``--savefig``: Save configuration.

Notes
-----

- With ``theta_scale=proportional``, angles reflect the *value* of
  ``y_true`` over the chosen angular span. With ``uniform``, angles
  reflect *order* only.

- Radii are normalized per series so models with different scales are
  comparable in one view.

Examples
--------

Minimal two-model comparison (point predictions)::

  kdiagram plot-relationship data.csv \
    --y-true actual \
    --pred q50 --pred q50_2024 \
    --names M1 M2 \
    --savefig relationship.png

Half-circle, uniform angles, custom ticks and labels::

  kdiagram plot-relationship data.csv \
    --y-true actual \
    --pred q50 \
    --theta-scale uniform \
    --acov half_circle \
    --z-values month \
    --z-label "Month" \
    --title "Truth–Prediction (Half Circle)" \
    --savefig half.png


.. _cli-plot-conditional-quantiles:

plot-conditional-quantiles
==========================

Synopsis
--------
.. code-block:: bash

  kdiagram plot-conditional-quantiles INPUT
    --y-true Y_TRUE
    [--pred COLS ... | --pred-cols COLS ...]
    --q-levels Q_LEVELS
    [--bands PCTS]
    [--title TITLE] [--figsize W,H] [--cmap CMAP]
    [--alpha-min A_MIN] [--alpha-max A_MAX]
    [--show-grid | --no-show-grid]
    [--mask-radius | --no-mask-radius]
    [--dpi DPI] [--savefig PATH]
    [-i INPUT] [--format FORMAT]

Arguments
---------

- **INPUT** / ``-i, --input`` / ``--format``: As above.

- ``--y-true, --true-col`` (required): Ground-truth column.

- ``--pred`` / ``--pred-cols`` (required): **One** group of quantile
  columns (CSV or space-separated), e.g. ``q10,q50,q90``.

- ``--q-levels, --quantiles`` (required): Matching quantile levels,
  e.g. ``0.1,0.5,0.9``.

- ``--bands``: Interval percentages to shade (CSV), e.g. ``80,50``.
  If omitted, the widest available interval is shown.

- ``--alpha-min`` / ``--alpha-max``: Opacity range for bands
  (outer → inner).

- ``--title`` / ``--figsize`` / ``--cmap``: Figure options.

- ``--show-grid`` / ``--no-show-grid``: Toggle polar grid.

- ``--mask-radius`` / ``--no-mask-radius``: Hide/show radial tick
  labels.

- ``--dpi`` / ``--savefig``: Save configuration.

Notes
-----

- Data are sorted by ``y_true`` to produce a smooth radial spiral.

- Bands use the matching lower/upper quantiles implied by the
  requested percentage(s). The median (0.5) line is drawn when
  available.

Examples
--------

One set of quantiles + bands::

  kdiagram plot-conditional-quantiles data.csv \
    --y-true actual \
    --pred q10,q50,q90 \
    --q-levels 0.1,0.5,0.9 \
    --bands 80,50 \
    --savefig cond_quant.png


.. _cli-plot-residual-relationship:

plot-residual-relationship
==========================

Synopsis
--------
.. code-block:: bash

  kdiagram plot-residual-relationship INPUT
    --y-true Y_TRUE
    [--pred COLS ... | --model NAME:COLS ... | --pred-cols COLS ...]
    [--names NAMES ...]
    [--title TITLE] [--figsize W,H] [--cmap CMAP]
    [--s S] [--alpha ALPHA]
    [--show-zero-line | --no-show-zero-line]
    [--show-grid | --no-show-grid]
    [--dpi DPI] [--savefig PATH]
    [-i INPUT] [--format FORMAT]

Arguments
---------

- **INPUT** / ``-i, --input`` / ``--format``: As above.

- ``--y-true, --true-col`` (required): Ground-truth column.

- Predictions (choose a style):

  * ``--pred`` / ``--pred-cols`` (repeatable): one point-pred column
    per model; repeat to add models.

  * ``--model`` (repeatable): ``NAME:q50`` (one column per model).

- ``--names``: Model names, overrides defaults.

- ``--s`` / ``--alpha``: Marker size and alpha.

- ``--show-zero-line`` / ``--no-show-zero-line``: Toggle the
  zero-residual ring.

- ``--show-grid`` / ``--no-show-grid``: Toggle polar grid.

- ``--title`` / ``--figsize`` / ``--cmap`` / ``--dpi`` / ``--savefig``:
  Figure/save options.

Notes
-----

- Residuals are computed as ``actual - prediction``. The plot shifts
  radii to handle negative values; the zero line appears as a circle
  when enabled.

Examples
--------

Two models with explicit names::

  kdiagram plot-residual-relationship data.csv \
    --y-true actual \
    --pred q50 --pred q50_2024 \
    --names Baseline Wide \
    --show-zero-line \
    --savefig residuals.png


.. _cli-plot-error-relationship:

plot-error-relationship
=======================

Synopsis
--------
.. code-block:: bash

  kdiagram plot-error-relationship INPUT
    --y-true Y_TRUE
    [--pred COLS ... | --model NAME:COLS ... | --pred-cols COLS ...]
    [--names NAMES ...]
    [--title TITLE] [--figsize W,H] [--cmap CMAP]
    [--s S] [--alpha ALPHA]
    [--show-zero-line | --no-show-zero-line]
    [--show-grid | --no-show-grid]
    [--mask-radius | --no-mask-radius]
    [--dpi DPI] [--savefig PATH]
    [-i INPUT] [--format FORMAT]

Arguments
---------

- Same as :ref:`cli-plot-residual-relationship`, plus:

  * ``--mask-radius`` / ``--no-mask-radius``: Hide/show radial tick
    labels.

Notes
-----

- Errors are also ``actual - prediction``, but angles are based on the
  *true* value ordering; useful for seeing how error changes across the
  domain of ``y_true`` (e.g., heteroscedasticity with magnitude).

Examples
--------

Two models, hide radial ticks::

  kdiagram plot-error-relationship data.csv \
    --y-true actual \
    --model A:q50 --model B:q50_2024 \
    --mask-radius \
    --savefig error_rel.png

