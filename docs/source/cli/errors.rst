=========================
Error Diagnostics (CLI)
=========================

Polar tools to explore forecast errors: compare full error
distributions, show error bands across a cyclic driver, and
visualize 2-D uncertainty with ellipses.

This page documents three commands:

* ``plot-error-violins`` — compare multiple error distributions.
* ``plot-error-bands`` — mean ± k·std error vs. a cyclic feature.
* ``plot-error-ellipses`` — 2-D uncertainty ellipses in polar coords.

All commands accept a positional INPUT (table path) and most also
support ``-i/--input`` and ``--savefig``. Tables are auto-detected
by extension (CSV/Parquet, etc.); override with ``--format`` if
needed.

plot-error-violins
------------------

Compare several one-dimensional error distributions as polar
“violins”. Each model occupies an angular sector; radial extent shows
error values and the violin width shows density.

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-error-violins INPUT
     [--error COL | --error COL1,COL2,...]...
     [--error-cols COL1,COL2,...]
     [--names NAME1 [NAME2 ...]]
     [--figsize WxH] [--cmap NAME] [--alpha A]
     [--show-grid | --no-show-grid]
     [--dpi 300] [--savefig PATH]

Key arguments
^^^^^^^^^^^^^

``--error``
  Add one or more error columns. Repeatable. Accepts a CSV list or
  tokens. Example: ``--error err_a,err_b`` or ``--error err_a``
  twice.

``--error-cols``
  Alias for a single list of error columns.

``--names``
  Display names aligned to the provided columns. Defaults to
  ``Model 1, Model 2, …``

Styling
^^^^^^^

``--figsize``, ``--cmap``, ``--alpha``, ``--show-grid``, ``--dpi``,
``--savefig``.

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-error-violins errors.csv \
     --error err_a --error err_b,err_c \
     --names "A" "B" "C" \
     --cmap plasma --alpha 0.7 \
     --savefig violins.png

plot-error-bands
----------------

Aggregate errors by angle bins of a cyclic or ordered driver (e.g.
month) and show mean ± k·std as a filled band in polar space.

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-error-bands INPUT
     --error-col ERR
     --theta-col COL
     [--theta-period P] [--theta-bins K]
     [--n-std S]
     [--color HEX] [--alpha A]
     [--mask-angle]
     [--figsize WxH] [--dpi 300] [--savefig PATH]

Key arguments
^^^^^^^^^^^^^

``--error-col``
  Column with error values (e.g. actual - predicted).

``--theta-col``
  Driver to bin on (mapped to angle). If ``--theta-period`` is
  given, values wrap modulo period; otherwise scaled min→max.

``--theta-period``
  Period of the cyclic driver (e.g. 12 for months). Optional.

``--theta-bins``
  Number of angular bins. Default is 24.

``--n-std``
  Band half-width as multiples of per-bin std. Default is 1.0.

Styling
^^^^^^^

``--color``, ``--alpha``, ``--mask-angle``, ``--figsize``, ``--dpi``,
``--savefig``.

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-error-bands errors.csv \
     --error-col err --theta-col month \
     --theta-period 12 --theta-bins 12 --n-std 1.5 \
     --color "#2980B9" --alpha 0.35 \
     --savefig error_bands.png

plot-error-ellipses
-------------------

Draw a filled ellipse for each point to represent 2-D uncertainty:
radial mean/std and angular mean/std (angles in degrees).

Synopsis
^^^^^^^^

.. code-block:: bash

   kdiagram plot-error-ellipses INPUT
     --r-col R --theta-col THETA_DEG
     --r-std-col RSTD --theta-std-col THSTD_DEG
     [--color-col COL]
     [--n-std S]
     [--cmap NAME] [--alpha A]
     [--edgecolor COLOR] [--linewidth W]
     [--mask-angle] [--mask-radius]
     [--figsize WxH] [--dpi 300] [--savefig PATH]

Key arguments
^^^^^^^^^^^^^

``--r-col``
  Mean radial position.

``--theta-col``
  Mean angular position in degrees.

``--r-std-col``
  Radial standard deviation.

``--theta-std-col``
  Angular standard deviation in degrees.

``--color-col``
  Optional column to color ellipses (otherwise uses radial std).

``--n-std``
  Ellipse size in standard deviations (e.g. 2.0 ≈ 95%). Default 2.0.

Styling
^^^^^^^

``--cmap``, ``--alpha``, ``--edgecolor``, ``--linewidth``,
``--mask-angle``, ``--mask-radius``, ``--figsize``, ``--dpi``,
``--savefig``.

Example
^^^^^^^

.. code-block:: bash

   kdiagram plot-error-ellipses errors.csv \
     --r-col r --theta-col theta_deg \
     --r-std-col r_std --theta-std-col theta_std_deg \
     --color-col priority --n-std 1.5 \
     --alpha 0.7 --edgecolor black --linewidth 0.5 \
     --savefig ellipses.png

Tips
----

* **Column names**
  Make sure column names in the flags match your table header.

* **Angle units**
  Ellipse angles must be in degrees (means and stds).

* **Saving vs. showing**
  When ``--savefig`` is omitted, figures are shown interactively.

* **Grids and masks**
  Use ``--show-grid/--no-show-grid`` or masking flags to
  declutter plots when needed.

* **Large files**
  For very large CSVs, Parquet input (``--format parquet``) can be
  much faster.
