Taylor Diagram (CLI)
====================

Overview
--------

The Taylor diagram summarizes how closely one or more prediction
series match a reference series by combining *standard deviation*
and *correlation* in a polar plot. kdiagram exposes three CLI
commands:

- ``plot-taylor-diagram``: Standard diagram (points/lines +
  reference arc).
- ``plot-taylor-diagram-in``: Diagram with a background
  colormap (pcolormesh) showing a chosen diagnostic.
- ``taylor-diagram``: Flexible entry-point that supports either
  precomputed statistics (*stats-mode*) or raw data columns
  (*data-mode*).

All commands save a figure when ``--savefig`` is given; otherwise
they display the plot interactively.

.. note::

   Correlation :math:`\rho` is mapped to the angle via
   :math:`\theta=\arccos(\rho)`. Radius encodes the prediction
   standard deviation. The reference standard deviation is drawn as
   an arc (or point/line).

Quick start
-----------

From a CSV with columns ``y``, ``m1``, and ``m2``:

.. code-block:: bash

   # Standard diagram
   kdiagram plot-taylor-diagram data.csv \
     --y-true y \
     --pred m1 --pred m2 \
     --names "Model A" "Model B" \
     --savefig taylor.png

   # Diagram with background
   kdiagram plot-taylor-diagram-in data.csv \
     --y-true y \
     --model "A:m1" --model "B:m2" \
     --cmap viridis --radial-strategy convergence \
     --cbar --savefig taylor_in.png

   # Stats-mode (no dataset needed)
   kdiagram taylor-diagram \
     --stddev 1.10 0.85 \
     --corrcoef 0.92 0.68 \
     --names A B \
     --draw-ref-arc --cmap plasma \
     --radial-strategy rwf \
     --savefig taylor_stats.png

Commands and options
--------------------

``plot-taylor-diagram``
~~~~~~~~~~~~~~~~~~~~~~~

Standard Taylor diagram comparing predictions to a reference.

**I/O**

- ``input`` (positional) or ``-i/--input``: Table path.
- ``--format``: Optional format override (``csv``, ``parquet``,
  ...).
- ``--y-true`` / ``--true-col``: Reference column (required).

**Predictions**

Provide one column per model:

- ``--model NAME:COL`` (repeatable)
- ``--pred COL`` (repeatable) or ``--pred-cols COL`` (alias)
- ``--names NAME1 NAME2 ...`` (optional)

**Diagram settings**

- ``--acov``: Angular coverage; one of ``default``, ``half_circle``
  (default: ``half_circle``).
- ``--zero-location``: Where correlation 1 sits. Choices:
  ``N``, ``NE``, ``E``, ``S``, ``SW``, ``W``, ``NW``, ``SE``.
  (default: ``W``)
- ``--direction``: Angle direction; ``1`` CCW or ``-1`` CW
  (default: ``-1``).
- ``--only-points``: Plot only markers (no radial lines).
- ``--ref-color``: Reference color (default: ``red``).
- ``--draw-ref-arc / --no-draw-ref-arc``: Show/hide reference arc.
- ``--angle-to-corr / --no-angle-to-corr``: Label angles with
  correlations or with degrees.
- ``--marker``: Marker style (default: ``o``).
- ``--corr-steps``: Correlation tick count (default: ``6``).

**Figure**

- ``--figsize W,H`` (e.g., ``10,8``), ``--title``, ``--dpi``,
  ``--savefig PATH``.

``plot-taylor-diagram-in``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Taylor diagram with a background colormap.

Same **I/O** and **Predictions** as above, plus:

**Diagram settings**

- ``--acov``: ``default`` or ``half_circle`` (default: ``None`` →
  ``half_circle`` internally).
- ``--zero-location`` (default: ``E``), ``--direction``,
  ``--only-points``, ``--draw-ref-arc``, ``--angle-to-corr``,
  ``--marker``, ``--corr-steps``.

**Background**

- ``--cmap``: Colormap (default: ``viridis``).
- ``--shading``: ``auto`` | ``gouraud`` | ``nearest`` (default:
  ``auto``).
- ``--shading-res``: Grid resolution (default: ``300``).
- ``--radial-strategy``:
  - ``convergence``: color ~ :math:`\cos(\theta)` (correlation).
  - ``norm_r``: color ~ normalized radius (std dev).
  - ``performance``: highlights near best model.
  - ``rwf`` / ``center_focus``: accepted, but degrade to
    ``performance`` with a warning in this CLI.
- ``--norm-c / --no-norm-c``: Normalize background values.
- ``--norm-range LO,HI``: Range when normalizing (e.g., ``0,1``).
- ``--cbar / --no-cbar``: Show/hide colorbar.

**Figure**

- ``--figsize W,H``, ``--title``, ``--dpi``, ``--savefig PATH``.

``taylor-diagram`` (flexible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two modes:

**A) Stats-mode (no dataset)**

- ``--stddev v1 v2 ...`` and ``--corrcoef r1 r2 ...`` (same length).
- Optional: ``--names ...``, ``--ref-std``, ``--draw-ref-arc``.
- Background (optional): ``--cmap``, ``--radial-strategy``
  (``rwf`` | ``convergence`` | ``center_focus`` | ``performance``),
  ``--norm-c``, ``--power-scaling``.
- Visual: ``--marker``, ``--tick-size``, ``--label-size``,
  ``--figsize W,H``, ``--title``, ``--dpi``, ``--savefig``.

**B) Data-mode (dataset)**

- Provide ``input`` (or ``--input``), ``--y-true``, and predictions
  via ``--model`` / ``--pred`` / ``--pred-cols``. You may also pass
  ``--names``.
- The remaining options mirror stats-mode for styling and background.

Examples
--------

Standard diagram (points + reference arc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   kdiagram plot-taylor-diagram data.csv \
     --y-true y \
     --pred m1 --pred m2 \
     --names "Model A" "Model B" \
     --acov half_circle \
     --zero-location W \
     --direction -1 \
     --savefig taylor_basic.png

Background: correlation field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   kdiagram plot-taylor-diagram-in data.csv \
     --y-true y \
     --model A:m1 --model B:m2 \
     --radial-strategy convergence \
     --cmap viridis --cbar \
     --savefig taylor_bg.png

Stats-mode with custom background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   kdiagram taylor-diagram \
     --stddev 1.05 0.88 0.75 \
     --corrcoef 0.91 0.72 0.60 \
     --names LR SVR RF \
     --draw-ref-arc \
     --cmap plasma \
     --radial-strategy rwf \
     --norm-c --power-scaling 1.2 \
     --savefig stats_mode.png

Tips
----

- **Predictions as columns**: each model must map to exactly one
  numeric column (use ``--model NAME:COL`` or ``--pred COL``).
- **Correlation ticks**: increase ``--corr-steps`` for finer angular
  labeling when using ``--angle-to-corr`` (default: on).
- **Orientation**: adjust ``--zero-location`` (e.g., ``E``, ``W``)
  and ``--direction`` (``1`` CCW or ``-1`` CW) to fit your reading
  convention.
- **Background strategies**: in the ``plot-taylor-diagram-in`` CLI,
  unsupported strategies (``rwf``/``center_focus``) downgrade to
  ``performance`` with a warning.

See also
--------

- :doc:`comparison` — radar comparison & reliability diagrams.
- :doc:`feature_based` — feature fingerprints and interactions.
- :doc:`errors` — error bands/violins/ellipses.

Reference
---------

- Taylor, K. E. (2001). Summarizing multiple aspects of model
  performance in a single diagram. *J. Geophysical Research*,
  **106**(D7), 7183–7192.

