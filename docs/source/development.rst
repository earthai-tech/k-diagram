.. _development:

=============================
Development Guide
=============================

This page explains how the package is structured, the API conventions to
follow, and the steps for adding new diagnostics. It complements the paper by
focusing on the *software artifact*—architecture, extensibility, testing, and
documentation practices.


Purpose and Scope
-----------------

``k-diagram`` targets *uncertainty diagnostics* first, with additional
plots (e.g, *evaluation*)  provided as optional, experimental views. The public surface
is deliberately small and stable; internals are modular and easy to extend.
This guide shows how to add a plot, write tests, and document the result
without breaking existing users.


Architecture at a Glance
------------------------

The package is a small stack of composable layers:

- ``kdiagram.plot`` – user functions grouped by task
  (``uncertainty``, ``errors``, ``probabilistic``, ``evaluation``,
  ``comparison``, ``feature_based``, ``relationship``, ``taylor_diagram``).
- ``kdiagram.utils`` – shared helpers for polar setup, colors, KDE, grids,
  validation, and tiny cross-cutting utilities.
- ``kdiagram.core`` – I/O and post-processing (column coercion, quantiles,
  binning, indexing, error policies).
- ``kdiagram.compat`` – shims for Matplotlib/Pandas version differences.
- ``kdiagram.cli`` – command-line front-ends that mirror the public API.
- ``kdiagram.datasets`` – small helpers to stage example assets for docs/tests.

Public API
----------

The public entry points are the functions under ``kdiagram.plot.*`` and the
CLI commands. Each plotting function:

- accepts a tidy ``pandas.DataFrame`` **or** arrays plus explicit selectors,
- validates inputs and shapes early,
- returns a **Matplotlib ``Axes``** (never hides the figure),
- takes an optional ``ax=``; if not provided, it creates one.

Example signature (typical):

.. code-block:: python

   ax = kd.plot_credibility_bands(
       df,
       q_cols=("q10", "q50", "q90"),
       theta_col="day_of_week",
       theta_period=7,
       theta_bins=7,
       # Polar grammar (see below)
       acov="default", zero_at="N", clockwise=True,
       theta_ticks=None, theta_ticklabels=None,
       # Aesthetics
       cmap="viridis", show_grid=True, figsize=(7, 7),
       # Integration
       ax=None, savefig=None, dpi=300,
   )

API Conventions
---------------

**Data-first.** Prefer DataFrames with explicit column names
(e.g., ``y_true='actual'``, ``y_pred='pred'``, ``q_cols=('q10','q50','q90')``).
When arrays are supported, shapes must be unambiguous; validation happens
immediately with informative errors.

**Polar grammar is explicit.** Angular coverage is ``acov`` (``default``,
``half_circle``, ``quarter_circle``, ``eighth_circle``). Orientation uses
``zero_at`` (``"N"|"E"|"S"|"W"``) and ``clockwise``. For periodic data, use
``theta_ticks`` and ``theta_ticklabels`` to label angles meaningfully
(e.g., weekdays).

**Evaluative plots offer parity.** Where community convention is Cartesian
(ROC/PR, classification reports), the plotting function provides
``kind="cartesian"|"polar"``. The default is **Cartesian**; polar is an
optional alternative.

**Return value.** Always return the ``Axes`` that was drawn on. This enables
downstream composition with standard Matplotlib.

Compatibility & Validation
--------------------------

Use ``kdiagram.compat`` to insulate user APIs from upstream changes
(e.g., a safe ``get_cmap`` wrapper). Prefer central validators and
decorators from ``kdiagram.utils`` to avoid ad-hoc checks:

- data frame checks and non-emptiness guards,
- quantile pairing and validation,
- consistent error policies (``errors='raise'|'warn'|'ignore'``).

Polar Setup & Shared Helpers
----------------------------

To keep figures consistent, rely on shared helpers:

- ``setup_polar_axes`` – create/orient a polar axes with standard styling.
- ``set_axis_grid`` – draw gridlines/ticks for the chosen ``acov``.
- ``_sample_colors`` / ``get_cmap`` – stable colormap access across MPL versions.
- ``map_theta_to_span`` – map a periodic variable to the configured angle span.

These keep "how to draw" separate from "what to draw" and make new plots small.

Adding a New Plot
-----------------

1. **Transform the data.**
   Coerce to a tidy table or validated arrays; resolve columns (``y_true``,
   ``y_pred``, ``q_cols``); aggregate or bin with vectorized NumPy/Pandas ops.

2. **Input validation.** 
   Decorators like ``@isdf, @check_non_emptiness`` (from ``kdiagram.decorators``), and 
   validators for quantile columns prevent ambiguous states early, with 
   informative error messages.

3. **Lay out the coordinates.**
   For polar, call ``setup_polar_axes`` and choose ``acov``, ``zero_at``,
   ``clockwise``. For periodic contexts, compute bin centers and set
   ``theta_ticks``/``theta_ticklabels`` when labels should be categorical.

4. **Render with Matplotlib primitives.**
   Bars, lines, patches, or violin envelopes; color with a colormap object
   from ``compat.get_cmap`` or ``matplotlib.colormaps``; add labels and a
   colorbar as needed.

5. **Return the Axes.**
   Respect any incoming ``ax=`` and return it. Support ``savefig=`` only as a
   thin convenience (do not hide the axes).

Minimal skeleton:

.. code-block:: python
   
   # 0) Make make sure the input is a dataframe
   @isdf 
   def plot_my_diagnostic(df, *, y_true="actual", y_pred="pred",
                          acov="default", zero_at="N", clockwise=True,
                          ax=None, **kws):
       # 1) validate/transform
       y = df[y_true].to_numpy()
       p = df[y_pred].to_numpy()

       # 2) layout
       fig, ax, span = setup_polar_axes(ax, acov=acov,
                                        zero_at=zero_at, clockwise=clockwise)

       # 3) render (example: angle = rank, radius = error magnitude)
       # ... compute theta, r ...
       ax.scatter(theta, r, **kws)
       ax.set_title("My Diagnostic")
       return ax

Kind Toggle (Cartesian vs Polar)
--------------------------------

If a plot also makes sense in Cartesian form, accept ``kind="cartesian"|"polar"``.
Share the **same transform** and implement two short renderers that differ only
in axes creation. Default to Cartesian for evaluative plots.

Testing & Coverage
------------------

- Use ``pytest`` and headless Matplotlib (Agg). Avoid pixel tests; assert on
  *semantics* (tick positions, labels, grid count, returned ``Axes`` type).
- Keep transforms and validators unit-tested (shapes, error messages, edge
  cases). Exercise rendering with smoke tests.
- Mock optional dependencies (e.g., HTML readers, downloaders) so tests do not
  depend on external services.
- Target high coverage for core code. Skip only glue files, version shims,
  and non-library examples.

Documentation
-------------

Add a short narrative example to the user guide and a compact example to the
gallery. Mirror the same call in the CLI docs when a CLI command exists.
Document new parameters with NumPy-style docstrings; keep names consistent
with the rest of the API (``y_true``, ``y_pred``, ``q_cols``, ``acov``,
``zero_at``, ``clockwise``, ``theta_ticks``, ``theta_ticklabels``, ``cmap``,
``show_grid``, ``figsize``, ``savefig``, ``ax``).

Performance Notes
-----------------

Data transforms are vectorized with NumPy/Pandas, and only compact arrays are
handed to Matplotlib. There is no hidden global state; each function depends
only on its inputs and returns an ``Axes``. This purity keeps rendering fast
and tests reliable.

Deprecation & Stability
-----------------------

Public behavior is stable. When changes are necessary, emit a
``PendingDeprecationWarning`` for one minor release, followed by a
``DeprecationWarning``; keep the old path working during that window and note
the migration in the changelog.

Local Development
-----------------

Create a fresh environment, install in editable mode with dev extras, and run
tests:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev]"
   pytest -q

Style & Docstrings
------------------

Follow PEP8 with Black/Ruff formatting. Use NumPy-style docstrings with clear
parameter/returns sections and examples. Keep lines ~70 characters where
practical for readable documentation.

Maintainer Checklist (PRs)
--------------------------

- The function returns an ``Axes`` and respects ``ax=``.
- Validation and errors are clear and tested.
- Polar controls/labels behave as documented.
- Semantics-based tests and docs/gallery entries are included.
- No new global state; vectorized transforms where feasible.


