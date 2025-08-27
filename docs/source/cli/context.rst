=====================
Context plots (CLI)
=====================

This page documents the command-line interfaces for the context
plotting utilities. They live under the ``kdiagram`` CLI and read a
tabular file (CSV/Parquet/…) to produce a figure.

Common patterns
---------------

Input & format
^^^^^^^^^^^^^^

``INPUT`` (positional) or ``-i/--input``
  Path to a table.

``--format``
  Optional override (``csv``, ``parquet``, …). Usually inferred from
  the file extension.

Selecting columns
^^^^^^^^^^^^^^^^^

You can pass prediction columns in three interchangeable styles:

* **Repeat ``--pred`` tokens**:

  .. code-block:: bash

     --pred m1 --pred m2

* **CSV with ``--pred-cols``** (one or multiple groups):

  .. code-block:: bash

     --pred-cols m1,m2

* **Name a model and its column with ``--model``** (repeatable):

  .. code-block:: bash

     --model A:m1 --model B:m2

Optionally provide legend names with:

.. code-block:: bash

   --names A B

Booleans, sizes, and saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Boolean flags** are paired: ``--show-grid / --no-show-grid``, etc.
* **Figure sizes** use either ``W,H`` or ``WxH``, e.g. ``--figsize 10,6``.
* **Save** with ``--savefig out.png`` (and optionally ``--dpi 300``).

---

plot-time-series
----------------

Plot actuals and one or more forecasts across time, with an optional
uncertainty band.

Usage
^^^^^

.. code-block:: bash

   kdiagram plot-time-series INPUT
     [--x-col TIME] [--actual-col ACT]
     [--pred COL ... | --pred-cols CSV | --model NAME:COL ...]
     [--names NAME ...]
     [--q-lower-col QL] [--q-upper-col QU]
     [--title T] [--xlabel XL] [--ylabel YL]
     [--figsize W,H] [--cmap viridis]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi 300]
     [--format FMT]

Notes
^^^^^

* If ``--x-col`` is omitted, the DataFrame index is used.
* To draw an uncertainty band, pass both ``--q-lower-col`` and
  ``--q-upper-col``.

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-time-series data.csv \
     --x-col time --actual-col y \
     --pred-cols m1,m2 --names "Model-1" "Model-2" \
     --q-lower-col q10 --q-upper-col q90 \
     --cmap plasma --title "Forecast vs Actuals" \
     --savefig ts.png

plot-scatter-correlation
------------------------

Cartesian scatter of actual (x) vs pred (y), with an optional y=x
identity line.

Usage
^^^^^

.. code-block:: bash

   kdiagram plot-scatter-correlation INPUT
     --actual-col ACT
     [--pred COL ... | --pred-cols CSV | --model NAME:COL ...]
     [--names NAME ...]
     [--title T] [--xlabel XL] [--ylabel YL]
     [--figsize W,H] [--cmap viridis]
     [--s 50] [--alpha 0.7]
     [--show-identity-line | --no-show-identity-line]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi 300]
     [--format FMT]

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-scatter-correlation data.csv \
     --actual-col actual --pred-cols m1,m2 \
     --names A B --cmap plasma --s 35 --alpha 0.6 \
     --savefig scatter.png

plot-error-autocorrelation
--------------------------

Autocorrelation (ACF) of forecast errors (actual - pred) to check
for residual dependence.

Usage
^^^^^

.. code-block:: bash

   kdiagram plot-error-autocorrelation INPUT
     --actual-col ACT --pred-col PRED
     [--title T] [--xlabel XL] [--ylabel YL]
     [--figsize W,H]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi 300]
     [--format FMT]

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-error-autocorrelation data.csv \
     --actual-col actual --pred-col m1 \
     --title "ACF of Errors" --savefig acf.png

plot-qq
-------

Q–Q plot of forecast errors (actual - pred) against the normal
distribution.

Usage
^^^^^

.. code-block:: bash

   kdiagram plot-qq INPUT
     --actual-col ACT --pred-col PRED
     [--title T] [--xlabel XL] [--ylabel YL]
     [--figsize W,H]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi 300]
     [--format FMT]

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-qq data.csv \
     --actual-col actual --pred-col m1 \
     --title "Q-Q of Errors" --savefig qq.png

plot-error-pacf
---------------

Partial autocorrelation (PACF) of forecast errors. Requires
``statsmodels``.

Usage
^^^^^

.. code-block:: bash

   kdiagram plot-error-pacf INPUT
     --actual-col ACT --pred-col PRED
     [--title T] [--xlabel XL] [--ylabel YL]
     [--figsize W,H]
     [--show-grid | --no-show-grid]
     [--savefig PATH] [--dpi 300]
     [--format FMT]
     [--pacf-kw KEY=VAL ...]   # optional passthrough; see notes

Notes
^^^^^

* Internally we default to ``method='ywm'`` for stability unless
  you override via passthrough kwargs.
* If ``statsmodels`` is missing you’ll get an informative error.

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-error-pacf data.csv \
     --actual-col actual --pred-col m1 \
     --title "PACF of Errors" --savefig pacf.png

plot-error-distribution
-----------------------

Histogram + KDE of forecast errors.

Usage
^^^^^

.. code-block:: bash

   kdiagram plot-error-distribution INPUT
     --actual-col ACT --pred-col PRED
     [--title T] [--xlabel XL]
     [--savefig PATH] [--dpi 300]
     [--format FMT]
     [--bins 40] [--kde-color COLOR] [--figsize W,H] ...

Notes
^^^^^

* Additional histogram/KDE styling options are forwarded to the
  underlying helper (e.g. ``--bins 40``).

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-error-distribution data.csv \
     --actual-col actual --pred-col m1 \
     --title "Error Distribution" --bins 40 \
     --savefig err_dist.png

---

Tips & troubleshooting
----------------------

* If a command exits with “Missing columns: …”, check your column
  names and CSV separators.
* For datetime x-axes in ``plot-time-series``, keep your time
  column as an ISO8601 string or parse it to datetime before
  saving.
* For repeated flags (e.g., ``--pred``), order determines the
  legend order.
* Disable grids or identity lines with the ``--no-*`` variants of
  the flags.
* Most commands support ``--cmap`` to control color mapping when
  multiple series are plotted.