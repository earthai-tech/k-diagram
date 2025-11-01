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
-------------------------

The package is a small stack of composable layers:

- ``kdiagram.plot``

  This is the main user-facing API, containing all plotting functions grouped
  by task. Each file represents a family of plots,
  such as ``probabilistic.py`` for PIT histograms and sharpness diagrams, 
  ``errors.py`` for violin and band plots, or ``anomaly.py`` for visualizing 
  forecast failures. Other modules like ``evaluation.py`` and ``context.py`` 
  provide evaluation and contextual plots and  ``taylor_diagram`` 
  for model evaluations.

- ``kdiagram.utils``

  This package provides the core machinery shared by all plots.
  It includes essential plot utilities like ``setup_polar_axes`` for creating
  the polar canvas and ``set_axis_grid`` for consistent styling. It also  
  contains robust data-handling logic, such as the quantile helpers in
  ``diagnose_q.py`` (e.g., ``validate_qcols``, ``detect_quantiles_in``)
  and metric calculation functions in ``mathext.py`` (e.g., ``compute_crps``).
  Filesystem operations are centralized in ``fs.py``, which provides the
  ``safe_savefig`` helper.

- ``kdiagram.core``

  This layer handles data input/output and internal post-processing.
  ``io.py`` provides the main ``read_data`` and ``write_data`` functions. 
  This is supported by ``_io_utils.py``, which contains
  internal helpers like ``_post_process`` (for ``fillna``, ``dropna``)
  and ``_get_valid_kwargs``. This ``_get_valid_kwargs``
  filter is used throughout the plotting modules (e.g., in ``errors.py``) 
  to safely pass user customizations to Matplotlib.

- ``kdiagram.compat``

  This is a crucial layer for ensuring stable behavior across different
  versions of Matplotlib, Pandas, and Scikit-learn.
  It provides compatibility shims, such as ``get_cmap``
  and ``get_colors`` for consistent color handling,
  and backports metrics like ``root_mean_squared_error``
  if they aren't available in the user's environment.

- ``kdiagram.cli``

  This package exposes the library's functionality to the command line. 
  The main ``__init__.py`` file builds the primary
  parser (``build_parser``) and registers all available plots as subcommands. 
  Each subcommand, like ``add_plot_anomalies`` or ``add_plot_taylord``, 
  links a CLI command (e.g., ``k-diagram plot-anomalies``) directly to the 
  corresponding Python plotting function.

- ``kdiagram.datasets``

  This provides utilities for documentation and testing.
  ``load.py`` contains functions to load real-world sample data, such as
  ``load_zhongshan_subsidence``. ``make.py`` contains
  generators for creating synthetic, reproducible datasets
  (e.g., ``make_regression_data``, ``make_cyclical_data``,
  ``make_fingerprint_data``), which are used extensively
  in the documentation gallery.
      

Public API
----------

The public entry points are the functions under ``kdiagram.plot.*`` and the
CLI commands. Each plotting function:

- accepts a tidy ``pandas.DataFrame`` **or** arrays plus explicit selectors.
- validates inputs and shapes early using helpers like ``exist_features``.
- returns a **Matplotlib ``Axes``** (never hides the figure).
- takes an optional ``ax=``; if not provided, it creates one using ``setup_polar_axes``.

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

The API is designed to be predictable and integrate smoothly with the
scientific Python ecosystem. We follow a **data-first** philosophy,
preferring that you pass a ``pandas.DataFrame`` and use explicit
column names (e.g., ``actual_col='actual'``,
``q_cols=('q10','q50','q90')``). This
approach avoids ambiguity and makes code more readable. When
passing raw arrays, we expect shapes to be unambiguous and perform
validation immediately to provide clear, informative errors.

A core feature is the **explicit "polar grammar"** used in all polar
plots. This gives you direct, repeatable
control over the plot's geometry. You can set the angular coverage with
``acov`` (like ``'half_circle'`` or ``'quarter_circle'``)
, define the plot's orientation using ``zero_at``
(e.g., ``'N'`` for North or ``'E'`` for East),
and set the rotational direction with ``clockwise``.
For cyclical or ordered data, you can provide ``theta_ticks`` and
``theta_ticklabels`` to map raw data values to meaningful labels, such
as ``{9.5: "Open 9:30"}``, making the plot self-explanatory.

We also provide **parity with conventional plots**.
For standard diagnostics like ROC curves or classification reports,
where a Cartesian view is the community standard, all functions
accept a ``kind="cartesian"`` argument. The
default is typically **polar** to encourage using the package's novel
visualizations, but you always have the option to fall back to a
traditional view.

Finally, the most important API contract is our **return value**.
Every plotting function, without exception, **returns the Matplotlib
``Axes``** object it drew on.
We never hide the figure or return a custom wrapper object. This ensures
that you can immediately use all your existing Matplotlib knowledge to
further customize, annotate, or combine plots into complex subplots,
making ``k-diagram`` a composable part of your existing workflows.


Compatibility & Validation
---------------------------

This layer is dedicated to making the package robust and easy to
debug. We insulate the public API from upstream changes in Matplotlib,
Pandas, or Scikit-learn by centralizing shims in the ``kdiagram.compat``
module. This is where we provide safe, version-aware wrappers like ``get_cmap``
or ``get_colors``,ensuring that our plots render consistently even as dependencies evolve.

Internally, we **prefer central validators and decorators** over ad-hoc
checks scattered throughout the code. Plotting
functions are decorated with ``@isdf``, ``@check_non_emptiness``, and /or scikit-learn 
``@validate_params``, to catch invalid inputs at the earliest possible moment. For more complex validation, we use
dedicated helpers from ``kdiagram.utils``. For example, ``exist_features``
is called at the beginning of most plot functions to confirm all
required columns are present in the DataFrame.
Similarly, ``diagnose_q.py`` provides functions like ``validate_qcols``
and ``build_qcols_multiple`` to robustly parse and pair quantile
column names. This consistent approach to validation makes the library 
more reliable and its error messages more informative.


Polar Setup & Shared Helpers
-----------------------------

To ensure every visualization has a consistent look, feel, and
orientation, all plot functions **rely on shared helpers** from
``kdiagram.utils.plot``. This is the key
to keeping the plotting functions themselves small, readable, and
focused on their specific logic.

Instead of creating an axis manually, a new plot function will almost
always call ``setup_polar_axes``.
This single helper is responsible for creating a new polar ``Axes`` (or
using an existing one passed via the ``ax`` parameter) and correctly
applying the ``acov``, ``zero_at``, and ``clockwise`` parameters. Immediately after,
``set_axis_grid`` is typically called to draw the grid lines and ticks
in a standardized way. When a plot needs to map data (like time of day or a feature value) to
an angular position, it uses ``map_theta_to_span``. This architecture cleanly
**separates "what to draw" (the plot's specific logic) from "how to
draw" (the boilerplate setup)**, making the code base much easier to
maintain and extend.

Adding a New Plot
-----------------

Adding a new diagnostic plot to ``k-diagram`` follows a consistent 5-step
pattern. This pattern ensures that your new plot correctly handles data, 
respects user parameters, uses shared helpers, and integrates cleanly 
with Matplotlib.

1. **Input Validation**
   First, it's crucial to validate the input ``DataFrame`` and the required 
   columns. This is standardized to provide clear, consistent errors. 
   You should apply the ``@isdf`` and ``@check_non_emptiness`` decorators 
   to the function signature. Inside the function, your very first action 
   should be to call ``exist_features`` to confirm all required columns 
   (e.g., ``actual_col``, ``pred_cols``) are present. If your plot compares 
   two arrays like ``y_true`` and ``y_pred``, use the ``validate_yy`` helper
   to align them and handle NaNs.

2. **Data Transformation**
   Next, transform the validated DataFrame columns into the final NumPy arrays
   needed for plotting. This is the core logic of your plot. This step might 
   involve calculating errors (``actual - predicted``), computing metrics 
   like ``clustered_anomaly_severity``, or aggregating data by binning, as 
   seen in the ``plot_feature_interaction`` function. The goal is to end up 
   with clean NumPy arrays for your coordinates (e.g., ``theta``, ``r``) and 
   visual properties (e.g., ``colors``, ``sizes``).

3. **Coordinate & Axes Layout**
   With your data ready, you prepare the Matplotlib axes. For any polar plot, 
   you must call the ``setup_polar_axes`` helper. This vital function respects 
   a user's incoming ``ax`` parameter and correctly applies the ``acov``, 
   ``zero_at``, and ``clockwise`` arguments. It returns the ``fig``, ``ax``, 
   and ``span`` (the angular coverage in radians). If your plot uses cyclical 
   or custom-ordered data, you can then map your feature to this span using 
   ``map_theta_to_span``.

4. **Render with Matplotlib Primitives**
   Now you draw on the axes. Use standard Matplotlib primitives based on 
   what your plot needs to show: ``ax.bar`` is used for polar bar charts, 
   ``ax.fill`` creates the violin shapes in ``plot_error_violins``, 
   ``ax.scatter`` is used for relationship plots, and ``ax.pcolormesh`` 
   or ``ax.contourf`` can create heatmaps. For colors, always use the 
   ``kdiagram.compat.get_cmap`` or ``kdiagram.utils.get_colors`` helpers 
   for consistent, version-safe color palettes.

5. **Finalize and Return**
   Finally, you conclude the function. Add titles, legends, and call 
   ``set_axis_grid`` for standardized gridlines. To handle saving, pass 
   the user's ``savefig`` path, ``fig`` object, and ``dpi`` to the 
   ``safe_savefig`` helper. This utility manages all file I/O, path creation, 
   and the logic for ``plt.show()`` vs. ``plt.close(fig)``. The most important 
   rule is to **always return the ``Axes`` object** (``ax``) so the user 
   can perform further customizations.

Minimal skeleton:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from kdiagram.decorators import isdf, check_non_emptiness
   from kdiagram.utils.validator import exist_features
   from kdiagram.utils.plot import setup_polar_axes, set_axis_grid
   from kdiagram.utils.fs import savefig as safe_savefig

   @isdf
   @check_non_emptiness
   def plot_my_diagnostic(
       df, *, my_col="default_val",
       acov="default", zero_at="N", clockwise=True,
       show_grid=True, grid_props=None,
       ax=None, savefig=None, dpi=300, **kws
   ):
       # 1) Input validation
       exist_features(df, features=[my_col])
       
       # 2) Data transformation
       data = df[my_col].dropna().to_numpy()
       # ... compute theta and r arrays ...
       theta = np.linspace(0, 2 * np.pi, len(data)) # example
       r = data # example

       # 3) Lay out the coordinates
       fig, ax, span = setup_polar_axes(
           ax, acov=acov,
           zero_at=zero_at,
           clockwise=clockwise
       )

       # 4) Render with Matplotlib primitives
       ax.scatter(theta, r, **kws)
       ax.set_title("My New Diagnostic Plot")
       
       # 5) Finalize and Return
       set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)
       
       # Use the helper to handle saving and figure closing
       final_path = safe_savefig(
           savefig,
           fig, 
           dpi=dpi,
           bbox_inches="tight",
       )
       
       if final_path is None: 
           # Only show if not saving
           plt.show() 
       else: 
           # Close if saving was successful
           plt.close(fig) 
           
       return ax
       

Kind Toggle (Cartesian vs Polar)
--------------------------------

For diagnostics that have a strong community standard in Cartesian
coordinates (like ROC/PR curves or classification reports),
we provide **API parity** by accepting a ``kind="cartesian"|"polar"``
parameter. This is a core
design philosophy: we share the **exact same data transformation**
logic for both plot types and then dispatch to one of two small,
separate rendering functions (e.g., ``_plot_pr_curve_cartesian``
).

Crucially, the ``kind`` parameter **defaults to "polar"**
. This is an intentional choice to
encourage users to try the package's novel visualizations, which are
often more compact, while always providing a familiar Cartesian
fallback. This entire switching logic is cleanly handled by the
``maybe_delegate_cartesian`` helper function, which you can see
used in ``plot_polar_roc`` and
``plot_polar_confusion_matrix``.


Testing & Coverage
------------------

Our testing philosophy is to **assert on semantics, not pixels**
. We use ``pytest`` and run all plots
with the headless Matplotlib ``Agg`` backend.
We explicitly avoid pixel-based snapshot tests, which are brittle
and fail with minor upstream rendering changes. Instead, our tests
assert on the properties of the returned ``Axes`` object: Does it
have the correct title? Are the tick labels set as expected? Are
the correct number of lines or bars present?

Our tests are split into two main categories. **Unit tests**
target core logic in ``kdiagram.utils``, such as data
transforms (e.g., ``compute_crps``) and validators
(e.g., ``validate_qcols``). These are tested
for edge cases, correct output shapes, and informative error messages. 
**Rendering tests** act as smoke tests for
the plotting functions themselves; they call
the plot function to ensure it runs without error, respects the ``ax``
parameter, and returns a valid ``Axes`` object.

We also **mock optional dependencies** to keep the core test
suite light. For example, ``plot_error_pacf``
is decorated with ``@ensure_pkg("statsmodels")``,
allowing it to be skipped if the heavy dependency isn't installed.
We target high test coverage for all core modules (``plot``,
``utils``, ``core``, ``compat``) and skip non-library files like
the ``cli`` and ``datasets`` loaders.


Documentation
-------------

Documentation is built from two primary sources: the narrative guides
(User Guide and Gallery) and the API reference, which is generated
directly from **NumPy-style docstrings**.

Every plot function's docstring is expected to be comprehensive,
including a `Parameters` section, a `Returns` section (which is
always an ``Axes``), a `Notes` section (often with LaTeX equations for
the underlying math), a copy-pastable ``Examples`` block, and a ``References`` 
section using ``.. footbibliography::``.

We strictly enforce **API consistency** to make the library
predictable. All plot functions should use
the following parameter names whenever possible:

- **Data:** ``y_true``, ``y_pred``,
  ``actual_col``, ``pred_col``,``q_cols``.
- **Polar Grammar:** ``acov``, ``zero_at``, ``clockwise``.
- **Ticks:** ``theta_ticks``, ``theta_ticklabels``, ``theta_tick_step``,
  ``r_ticks``, ``r_ticklabels``, ``r_tick_step``.
- **Aesthetics:** ``cmap``, ``colors``, ``show_grid``, ``grid_props``.
- **Integration:** ``figsize``, ``savefig``, ``dpi``, ``ax`` .
- **Behavior:** ``kind`` (for polar/cartesian toggle)
  and ``mode`` (for different plot styles).


Performance Notes
-----------------

We prioritize performance by ensuring all data transformations are
**vectorized with NumPy/Pandas** whenever possible, avoiding slow
Python loops. For example, aggregation
logic in plots like ``plot_feature_interaction`` relies on 
``pd.cut`` and ``groupby.agg``, and
metric calculations in ``mathext.py`` use ``np.mean``,
``np.where``, and ``np.diff`` for efficient computation
. Only these compact, aggregated arrays are
handed to Matplotlib for rendering.

Furthermore, the library is designed to be **stateless**.
There is **no hidden global state**; each plotting function
depends only on its inputs and returns an ``Axes`` **object**.
This functional purity makes rendering fast and, just as
importantly, makes our tests reliable and deterministic.


Deprecation & Stability
-----------------------

The public API is considered **stable**.
When breaking changes are unavoidable (e.g., to improve a
parameter's name), we follow a standard deprecation cycle. A new
parameter is introduced, and the old one is kept working for at
least one minor release, emitting a ``PendingDeprecationWarning``
or ``DeprecationWarning``.

For instance, when ``mask_angle`` was introduced to
``plot_radial_density_ring``, the old ``show_yticklabels``
parameter was kept but now issues a ``DeprecationWarning``,
guiding the user to the new API without breaking their existing
code. This process ensures that users'
code does not break unexpectedly.


Local Development
-----------------

To get started with local development, create a fresh virtual
environment using ``python -m venv .venv``. After activating
the environment, install the package in "editable" mode
(``-e``) along with all development dependencies (like
``pytest``) by running ``pip install -e ".[dev]"``. You can
then run the complete test suite from the root directory using
``pytest -q``.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev]"
   pytest -q


Style & Docstrings
------------------

We follow **PEP8** standards, with code formatting enforced
automatically by **Black** and **Ruff**.
All public functions and modules must have comprehensive
**NumPy-style docstrings**.

As seen across the ``kdiagram.plot`` modules, a good docstring
is extensive and includes:

- A clear ``Parameters`` section.
- A ``Returns`` section (which should always be ``ax : matplotlib.axes.Axes``).
- A ``Notes`` section for mathematical derivations (using LaTeX)
  or design rationale (like the story behind ``mode="cbueth"``).
- A copy-pastable ``Examples`` block that uses a synthetic
  dataset, ideally from ``kdiagram.datasets``.
- A ``See Also`` section linking to related functions.
- A ``References`` section using ``.. footbibliography::``.

Lines are kept to a practical length (around 70 characters) to
ensure docstrings render readably in terminals.


Maintainer Checklist (PRs)
--------------------------

When reviewing a Pull Request, ensure the following criteria are
met:

- **Returns an ``Axes`` and respects ``ax=``:** The function must
  integrate with existing Matplotlib figures and always return
  the ``Axes`` it drew on.
- **Clear Validation:** Inputs are validated early. This includes
  using ``@isdf`` and ``@check_non_emptiness``
  and calling ``exist_features`` for DataFrame
  checks.
- **Polar Grammar:** All polar-specific parameters
  (``acov``, ``zero_at``, ``clockwise``, ``theta_ticks``) are
  correctly passed to ``setup_polar_axes`` and
  behave as documented.
- **Tests and Docs:** The PR includes semantics-based tests for
  the new functionality and adds corresponding entries to the
  documentation (both the API docstring and the gallery).
- **No State:** The function is pure. It introduces no new global
  state and performs its data transforms using vectorized
  operations where feasible (e.g., using ``numpy``/``pandas``
  instead of ``for`` loops).
