.. _introduction_cli:
    
======================================
kdiagram command-line interface (CLI)
======================================

The **kdiagram** CLI lets you build the package’s polar diagnostics
straight from tabular data (CSV/Parquet/Feather/…)
without writing Python. It focuses on uncertainty, drift and
vector/scalar fields visualizations.

.. code-block:: text

   kdiagram --help
   kdiagram <command> --help

You can run it either via the console script:

.. code-block:: bash

   kdiagram --version

or via Python:

.. code-block:: bash

   python -m kdiagram --version

Quick start
-----------

Assume you have a table with at least ``q10``, ``q50``, ``q90``:

.. code-block:: bash

   kdiagram plot-interval-width data.csv \
     --q-cols q10,q90 \
     --z-col q50 \
     --savefig interval_width.png

Data input
----------

* **Input file**: most commands take the input path as the first
  positional argument (some also accept ``-i/--input``).
* **Format**: inferred from the extension or forced with ``--format``
  (e.g. ``csv``, ``parquet``, ``feather``).
* **NaNs**: many commands support a ``--dropna/--no-dropna`` toggle. By
  default we drop rows missing any **essential** columns for the plot.

Column notation (friendly parsers)
----------------------------------

To avoid brittle quoting, the CLI accepts both comma-separated and
space-separated styles.

* **Single column**
  Example: ``--actual-col actual`` or ``--y-true actual`` (synonyms are
  provided where helpful).

* **Pairs (lower,upper)**
  Use either one token with a comma, **or** two tokens:

  .. code-block:: bash

     --q-cols q10,q90
     # or
     --q-cols q10 q90

  Internally handled by a custom action, so both are equivalent.

* **Lists (multiple columns)**
  Again, one token with commas **or** several tokens:

  .. code-block:: bash

     --qlow-cols q10_2023,q10_2024,q10_2025
     # or
     --qlow-cols q10_2023 q10_2024 q10_2025

* **Model specs (coverage plots)**
  Coverage commands accept one or more ``--model`` specs:

  .. code-block:: bash

     --model NAME:col1[,col2,...]   # repeat for multiple models
     --names M1 M2                  # or: --names M1,M2

  Where a model may be a single prediction column (*point*) or a set of
  quantile columns (*q-set*).

.. tip::

   For long lists, prefer the “space tokens” form to avoid shell
   quoting issues on Windows.

Figure & styling options
------------------------

* **Figure size**: ``--figsize 8,8`` or ``--figsize 8x8``

* **Colormaps**: ``--cmap viridis`` (or any Matplotlib cmap), plus
  command-specific options like ``--cmap-under/--cmap-over``.

* **Toggles** follow the consistent ``--flag`` / ``--no-flag`` pattern:

  * ``--show-grid`` / ``--no-show-grid``
  * ``--cbar`` / ``--no-cbar``
  * ``--mask-angle`` / ``--no-mask-angle``
  * (and others, per command)

* **Saving vs showing**:
  ``--savefig out.png`` saves the figure; if omitted, the plot is shown
  interactively.

Angular coverage (polar span)
-----------------------------

Most polar plots accept ``--acov`` to control the angular span:
``default`` (full), ``half_circle``, ``quarter_circle``, ``eighth_circle``.

.. code-block:: bash

   --acov quarter_circle

Error handling
--------------

* Missing columns → clear error message listing the missing names.
* Non-numeric data where numeric is required → explicit type error.
* Invalid flag values → ``argparse`` error with usage help.

Command summary
-----------------

Below is the index of CLI commands you can explore. Use
``kdiagram <command> --help`` for full details.

* ``plot-actual-vs-predicted`` — Compare actual vs predicted in polar.
* ``plot-anomaly-magnitude`` — Magnitude of violations of prediction
  intervals (under/over).
* ``plot-coverage`` — Aggregated coverage scores for one or more models.
* ``plot-coverage-diagnostic`` — Point-wise coverage diagnostics on polar.
* ``plot-interval-consistency`` — Temporal consistency of interval widths
  (CV/Std) per location.
* ``plot-interval-width`` — Polar scatter of interval width (Qup − Qlow).
* ``plot-model-drift`` — Drift across forecast horizons (polar bars).
* ``plot-temporal-uncertainty`` — Multi-series polar scatter (quantiles
  or arbitrary columns), optional normalization.
* ``plot-uncertainty-drift`` — Ring lines showing how widths change over
  time steps.
* ``plot-velocity`` — Temporal velocity (first differences) on polar.
* ``plot-radial-density-ring`` — Radial density ring from widths/velocity
  or direct data.
* ``plot-polar-heatmap`` — 2D polar histogram/heatmap (r × θ).
* ``plot-polar-quiver`` — Polar vector field (quiver) with optional
  colorization.

Conventions & compatibility
-----------------------------

* We keep common synonyms to ease migration:

  * ``--y-true`` and ``--true-col`` are accepted where relevant.
  * Some legacy flags like ``--pred`` may be mapped to ``--model``
    internally.
* Column order and angular position: unless stated otherwise,
  angles follow row order after filtering/``dropna``.

What’s next
-----------

Head to the **Uncertainty** section for detailed usage and examples of:
``plot-interval-width``, ``plot-interval-consistency``,
``plot-anomaly-magnitude``, ``plot-temporal-uncertainty``,
``plot-uncertainty-drift``, and ``plot-model-drift``.