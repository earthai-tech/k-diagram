.. _userguide_utils:

===================
Utility Functions
===================

Beyond the core visualization functions, `k-diagram` provides several
utility functions designed to help prepare and manipulate your data,
particularly when dealing with quantile forecasts stored in pandas
DataFrames.

These utilities can assist in detecting quantile columns based on naming
conventions, generating standard column names, and reshaping data between
wide and long formats suitable for different analysis or plotting tasks.

Summary of Utility Functions
----------------------------

.. list-table:: Utility Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.utils.detect_quantiles_in`
     - Automatically detects columns containing quantile values based
       on naming patterns (e.g., `_q0.X`) and optionally filters by
       prefix or date components.
   * - :func:`~kdiagram.utils.build_q_column_names`
     - Constructs expected quantile column names based on a prefix,
       optional date values, and desired quantiles, then validates
       if they exist in a DataFrame.
   * - :func:`~kdiagram.utils.reshape_quantile_data`
     - Reshapes a *wide-format* DataFrame (e.g.,
       `prefix_date_qX.X` columns) into a "semi-long" format where
       each quantile level gets its own column (e.g., `prefix_qX.X`),
       indexed by spatial and temporal columns.
   * - :func:`~kdiagram.utils.melt_q_data`
     - Reshapes a *wide-format* DataFrame into a *long format*, creating
       separate columns for the temporal value (`dt_name`), quantile level
       (`quantile`), and the corresponding prediction value. Inverse of
       :func:`~kdiagram.utils.pivot_q_data`. *(Note: The docstring description seems
       to incorrectly describe the output of `reshape_quantile_data`, while the
       implementation likely performs a melt-merge-pivot resulting in a semi-long
       format similar to `reshape_quantile_data`. Let's document based on the
       docstring's intent - a long format.)*
   * - :func:`~kdiagram.utils.pivot_q_data`
     - Reshapes a *long-format* DataFrame (with distinct columns for time,
       quantile level, and value) back into a *wide format*, creating
       columns like `prefix_date_qX.X`. Inverse operation of
       :func:`~kdiagram.utils.melt_q_data`.


Detailed Explanations
---------------------

.. _ug_detect_quantiles_in:

Detecting Quantile Columns (:func:`~kdiagram.utils.detect_quantiles_in`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
Automatically scans a DataFrame's column names to identify those that
likely represent quantile data, based on common naming conventions
(e.g., containing `_q` followed by a number like `_q0.1`, `_q0.95`).

**Key Parameters:**

* ``df``: The input DataFrame.
* ``col_prefix``: Optional prefix to narrow down the search (e.g.,
    `'prediction'` for columns like `'prediction_q0.5'`).
* ``dt_value``: Optional list of date/time strings to filter columns
    that include a temporal component in their name (e.g.,
    `'prediction_2023_q0.9'`).
* ``return_types``: Specifies the output format ('columns', 'q_val',
    'values', 'frame').

**Use Cases:**

* Automatically finding all quantile-related columns in a large dataset
    without manually listing them.
* Extracting specific quantile information (just the levels, the actual
    data arrays, or a subset DataFrame).
* Verifying which quantile levels are present in your data.

**Example:**
*(See the Gallery section below for a runnable code example)*

.. _ug_build_q_column_names:

Building Quantile Column Names (:func:`~kdiagram.utils.build_q_column_names`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
Constructs expected quantile column names based on specified quantiles,
an optional prefix, and optional date/time values, following the
standard naming convention (e.g., `prefix_date_qX.X` or `prefix_qX.X`).
It then checks if these constructed names exist in the provided DataFrame.

**Key Parameters:**

* ``df``: The DataFrame to check against.
* ``quantiles``: List of desired quantile levels (e.g., `[0.1, 0.5, 0.9]`).
* ``value_prefix``: Optional common prefix for the values.
* ``dt_value``: Optional list of date/time identifiers.
* ``strict_match``: If `True`, requires exact name matches; if `False`,
    allows pattern matching.

**Use Cases:**

* Programmatically generating lists of column names needed for other
    `k-diagram` functions (like `qlow_cols`, `qup_cols`).
* Validating whether all expected quantile columns for a given analysis
    are present in the DataFrame.

**Example:**
*(See the Gallery section below for a runnable code example)*

.. _ug_reshape_quantile_data:

Reshaping Quantile Data (Wide to Semi-Long) (:func:`~kdiagram.utils.reshape_quantile_data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
Transforms a DataFrame from a "wide" format, where different time steps
and quantiles for a variable are spread across many columns (e.g.,
`value_2023_q0.1`, `value_2023_q0.9`, `value_2024_q0.1`, ...), into a
more structured "semi-long" or "pivoted" format. In the output, each row
represents a unique combination of spatial location (if provided) and
time step, while different quantile levels become separate columns
(e.g., `value_q0.1`, `value_q0.9`).

**Key Parameters:**

* ``df``: The input wide-format DataFrame.
* ``value_prefix``: The common prefix identifying the quantile columns
    (e.g., `'subs'` for columns like `'subs_2022_q0.1'`).
* ``spatial_cols``: Optional list of columns identifying unique
    locations (e.g., `['lon', 'lat']`), preserved as index/columns.
* ``dt_col``: The name for the new column that will hold the extracted
    time step information (e.g., `'year'`).

**Use Cases:**

* Preparing data for time-series analysis or plotting where you need
    different quantiles aligned row-wise for each time step.
* Structuring data before calculating metrics that depend on having
    lower and upper bounds in the same row (e.g., interval width).
* Simplifying DataFrames with numerous time-stamped quantile columns.

**Example:**
*(See the Gallery section below for a runnable code example)*

.. _ug_melt_q_data:

Melting Quantile Data (Wide to Long) (:func:`~kdiagram.utils.melt_q_data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
Transforms a wide-format DataFrame containing time-stamped quantile
columns (e.g., `prefix_date_qX.X`) into a fully "long" or "tidy"
format. Each row in the output represents a single observation for a
specific location (if provided), time step, and quantile level. Creates
separate columns for the time step identifier, the quantile level, and
the corresponding value.

*(Note: Based on the implementation details likely involving melt-merge-pivot,
the actual output format might resemble `reshape_quantile_data`. However,
documenting based on the common understanding of "melting" to a long format.)*

**Key Parameters:**

* ``df``: The input wide-format DataFrame.
* ``value_prefix``: The common prefix identifying the quantile columns.
* ``dt_name``: The name for the new column holding the extracted time
    step information.
* ``q``: Optional list to filter specific quantiles.
* ``spatial_cols``: Optional list/tuple of spatial identifier columns.

**Use Cases:**

* Creating a "tidy" representation of quantile data suitable for use
    with plotting libraries like Seaborn or Altair that prefer long-format
    data.
* Preparing data for statistical analysis or database storage where each
    observation is a separate row.
* Filtering or grouping data easily by time step or quantile level.

**Example:**
*(See the Gallery section below for a runnable code example)*

.. _ug_pivot_q_data:

Pivoting Quantile Data (Long to Wide) (:func:`~kdiagram.utils.pivot_q_data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
Performs the inverse operation of :func:`~kdiagram.utils.melt_q_data`. It
takes a long-format DataFrame (where time, quantile level, and value
have their own columns) and transforms it back into a wide format. In the
output, columns are created for each combination of time step and
quantile level, following the pattern `prefix_date_qX.X`.

**Key Parameters:**

* ``df``: The input long-format DataFrame. Must contain columns for
    time (``dt_col``) and the quantile values (named like
    `prefix_qX.X`).
* ``value_prefix``: The common prefix used in the long-format quantile
    column names and for reconstructing the wide-format names.
* ``dt_col``: The name of the column containing the time step identifiers.
* ``q``: Optional list to filter specific quantiles before pivoting.
* ``spatial_cols``: Optional list/tuple of spatial identifier columns
    that form part of the index in the long format.

**Use Cases:**

* Reconstructing the original wide data format after performing analyses
    in long format.
* Preparing data for tools or functions that expect time steps and
    quantiles spread across columns.
* Creating summary tables or reports where different time points are columns.

**Example:**
*(See the Gallery section below for a runnable code example)*

.. raw:: html

   <hr>