.. _userguide_quantile_utils:

================================
Working with Quantile Data
================================

Probabilistic forecasts are often represented by a set of quantiles
stored in a "wide" format DataFrame, with separate columns for each
quantile and forecast horizon (e.g., ``'prediction_2023_q0.1'``,
``'prediction_2023_q0.9'``, etc.). While this format is common, it can
be cumbersome for analysis and visualization.

The :mod:`kdiagram.utils.q_utils` module provides a suite of powerful
helper functions designed to solve these common data wrangling
challenges. These utilities assist in automatically detecting quantile
columns based on naming conventions, generating standard column names,
and reshaping data between wide and long formats, preparing it for
use in `k-diagram`'s plotting functions or other analysis tasks.

This guide provides detailed explanations and practical examples for
each of these utility functions.

Summary of Quantile Utility Functions
-------------------------------------

.. list-table:: Quantile Utility Functions
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
       :func:`~kdiagram.utils.pivot_q_data`. 
   * - :func:`~kdiagram.utils.pivot_q_data`
     - Reshapes a *long-format* DataFrame (with distinct columns for time,
       quantile level, and value) back into a *wide format*, creating
       columns like `prefix_date_qX.X`. Inverse operation of
       :func:`~kdiagram.utils.melt_q_data`.

.. raw:: html

   <hr>
   
.. _ug_detect_quantiles_in:

Detecting Quantile Columns (:func:`~kdiagram.utils.detect_quantiles_in`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility automatically scans a DataFrame's column names to
identify those that likely represent quantile data. It is a
powerful convenience function for working with large datasets where
manually listing dozens of quantile columns would be tedious and
error-prone. The function works by recognizing common naming
conventions, such as columns containing ``_q`` followed by a
number (e.g., ``'prediction_q0.5'``).


**Key Parameters Explained:**

* **`col_prefix`**: An optional string to narrow down the search.
  For example, using `'prediction'` will only find columns that
  start with that prefix, like `'prediction_q0.5'`, and ignore
  others like `'temperature_q0.5'`.
* **`dt_value`**: An optional list of date/time strings (like
  ``['2023']``) to filter columns that include a temporal
  component in their name (e.g., `'prediction_2023_q0.9'`).
* **`return_types`**: This crucial parameter specifies the output
  format. You can choose to get back:
    
  - ``'columns'``: A list of the matching column names (default).
  - ``'q_val'``: A list of the unique quantile levels found.
  - ``'values'``: The raw data from the matching columns as a
    list of NumPy arrays.
  - ``'frame'``: A new DataFrame containing only the matching
    quantile columns.

**Conceptual Basis:**
The function operates by parsing column names using regular
expressions. It looks for specific patterns that indicate a
quantile forecast:

1.  **Non-temporal format**: ``{prefix}_q{value}``
    (e.g., ``'sales_q0.75'``)
2.  **Temporal format**: ``{prefix}_{date}_q{value}``
    (e.g., ``'temp_2023_q0.5'``)

The function also includes a ``mode`` parameter to handle cases
where quantile levels might be represented as values greater
than 1 (e.g., ``'risk_q150'`` for a 1.5 quantile). The adjustment
is formulated as:

.. math::
   :label: eq:quantile_detection

   q_{\text{adj}} = \begin{cases}
   \min(1, \max(0, q_{\text{raw}})) & \text{if } mode=\text{'soft'} \\
   q_{\text{raw}} & \text{if } q \in [0,1] \text{ and } mode=\text{'strict'}
   \end{cases}


**Example**
The following example demonstrates how to find columns based on a
prefix and a date, and how to return different types of output.

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu
   import pandas as pd
   import numpy as np

   # --- Sample Data ---
   df = pd.DataFrame({
       'site': ['A', 'B'],
       'value_2023_q0.1': [10, 11],
       'value_2023_q0.9': [20, 22],
       'temp_2023_q0.5': [15, 16],
       'value_2024_q0.1': [12, 13],
       'value_2024_q0.9': [23, 25],
       'notes': ['x', 'y']
   })

   # --- Usage ---
   print("Detecting 'value' columns for 2023:")
   q_cols_2023 = kdu.detect_quantiles_in(
       df, col_prefix='value', dt_value=['2023']
   )
   print(q_cols_2023)

   print("\nDetecting all quantile columns (returning levels):")
   q_levels = kdu.detect_quantiles_in(df, return_types='q_val')
   print(sorted(q_levels)) # Sort for consistent output

   print("\nDetecting 'temp' columns (returning frame):")
   temp_frame = kdu.detect_quantiles_in(
       df, col_prefix='temp', return_types='frame'
   )
   print(temp_frame)

.. code-block:: text
   :caption: Expected Output

   Detecting 'value' columns for 2023:
   ['value_2023_q0.1', 'value_2023_q0.9']

   Detecting all quantile columns (returning levels):
   [0.1, 0.5, 0.9]

   Detecting 'temp' columns (returning frame):
      temp_2023_q0.5
   0              15
   1              16
 

.. raw:: html

   <hr>
     
.. _ug_build_q_column_names:

Building Quantile Column Names (:func:`~kdiagram.utils.build_q_column_names`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility constructs a list of expected quantile column names
based on a specified prefix, optional date/time values, and a
list of desired quantiles. It then validates which of these
constructed names actually exist in the provided DataFrame. This
is a key function for programmatically gathering the correct
column names needed for other `k-diagram` plotting functions.


**Key Parameters Explained:**

* **`quantiles`**: A list of the quantile levels you are looking for
  (e.g., `[0.1, 0.5, 0.9]`).
* **`value_prefix`**: The common prefix for the forecast variable
  (e.g., `'precip'`).
* **`dt_value`**: An optional list of date or time identifiers to
  build names for specific horizons (e.g., `['2024']`).
* **`strict_match`**: If `True`, requires an exact name match. If
  `False`, allows for more flexible pattern matching.


**Conceptual Basis:**
The function programmatically constructs column names by assembling
the provided components according to a standard naming convention.
The general pattern is:

.. math::
   :label: eq:build_q_names

   \text{col\_name} = \begin{cases}
   \text{prefix}\_\text{date}\_q\text{quantile} & \text{if both exist} \\
   \text{prefix}\_q\text{quantile} & \text{if only prefix} \\
   \text{date}\_q\text{quantile} & \text{if only date} \\
   q\text{quantile} & \text{otherwise}
   \end{cases}

It then filters this generated list, returning only the names that
are actually present in the input DataFrame's columns.


**Example:**
The following example demonstrates how to build and validate column
names for different years. Note that for the year 2025, the
`precip_2025_q0.9` column is missing from the DataFrame, so it is
not included in the final output list.

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu
   import pandas as pd

   # --- Sample Data ---
   df = pd.DataFrame({
       'site': ['A', 'B'],
       'precip_2024_q0.1': [1, 2],
       'precip_2024_q0.9': [5, 6],
       'precip_2025_q0.1': [1.5, 2.5],
       # Missing 'precip_2025_q0.9'
   })

   # --- Usage ---
   print("Building names for 2024, quantiles 0.1, 0.9:")
   names_2024 = kdu.build_q_column_names(
       df, quantiles=[0.1, 0.9], value_prefix='precip', dt_value=['2024']
   )
   print(names_2024)

   print("\nBuilding names for 2025, quantiles 0.1, 0.9 (one missing):")
   names_2025 = kdu.build_q_column_names(
       df, quantiles=[0.1, 0.9], value_prefix='precip', dt_value=[2025]
   )
   print(names_2025)

.. code-block:: text
   :caption: Expected Output

   Building names for 2024, quantiles 0.1, 0.9:
   ['precip_2024_q0.1', 'precip_2024_q0.9']

   Building names for 2025, quantiles 0.1, 0.9 (one missing):
   ['precip_2025_q0.1']
 

.. raw:: html

   <hr>
     
.. _ug_reshape_quantile_data:

Reshaping Quantile Data (:func:`~kdiagram.utils.reshape_quantile_data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility transforms a DataFrame from a "wide" format, where
different time steps and quantiles are spread across many columns
(e.g., ``value_2023_q0.1``, ``value_2024_q0.1``), into a more
structured "semi-long" format. In the output, each row represents
a unique combination of a location and a time step, while the
different quantile levels become their own separate columns (e.g.,
``value_q0.1``, ``value_q0.9``). This is a crucial step for preparing
data for time-series analysis or for calculating metrics that
require lower and upper bounds to be in the same row.


**Key Parameters Explained:**

* **`value_prefix`**: The common prefix that identifies the quantile
  columns you want to reshape (e.g., ``'subs'`` for columns like
  ``'subs_2022_q0.1'``).
* **`spatial_cols`**: An optional list of columns that identify a
  unique location or sample (e.g., ``['lon', 'lat']``). These
  columns will be preserved.
* **`dt_col`**: The name for the new column that will hold the
  extracted time step information (e.g., ``'year'``).

**Conceptual Basis:**
This function reshapes the data by "melting" the wide-format
quantile columns into a long format and then "pivoting" them
back so that each unique quantile level becomes a new column.
The transformation can be conceptualized as:

.. math::
   :label: eq:reshape_q_data

   \mathbf{W}_{m \times n} \rightarrow \mathbf{L}_{p \times k}

where:

- :math:`\mathbf{W}` is the original wide DataFrame with :math:`m`
  rows and :math:`n` columns.
- :math:`\mathbf{L}` is the new semi-long DataFrame with :math:`p`
  rows, where :math:`p = m \times (\text{number of unique time steps})`.
- :math:`k` is the new number of columns, equal to the number of
  spatial columns + 1 (for the new time column) + the number of
  unique quantile levels.

**Example:**
The following example demonstrates how to transform a wide-format
DataFrame containing two years of subsidence forecasts into a
semi-long format, making it easier to analyze the data year by year.

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu
   import pandas as pd

   # --- Sample Wide Data ---
   wide_df = pd.DataFrame({
       'lon': [-118.25, -118.30],
       'lat': [34.05, 34.10],
       'subs_2022_q0.1': [1.2, 1.3],
       'subs_2022_q0.5': [1.5, 1.6],
       'subs_2023_q0.1': [1.7, 1.8],
       'subs_2023_q0.5': [1.9, 2.0],
   })
   print("Original Wide DataFrame:")
   print(wide_df)

   # --- Reshape the data ---
   semi_long_df = kdu.reshape_quantile_data(
       wide_df,
       value_prefix='subs',
       spatial_cols=['lon', 'lat'],
       dt_col='year' # Name for the new time column
   )
   print("\nReshaped (Semi-Long) DataFrame:")
   print(semi_long_df)

.. code-block:: text
   :caption: Expected Output

   Original Wide DataFrame:
         lon    lat  subs_2022_q0.1  subs_2022_q0.5  subs_2023_q0.1  subs_2023_q0.5
   0 -118.25  34.05             1.2             1.5             1.7             1.9
   1 -118.30  34.10             1.3             1.6             1.8             2.0

   Reshaped (Semi-Long) DataFrame:
         lon    lat  year  subs_q0.1  subs_q0.5
   0 -118.25  34.05  2022        1.2        1.5
   1 -118.30  34.10  2022        1.3        1.6
   2 -118.25  34.05  2023        1.7        1.9
   3 -118.30  34.10  2023        1.8        2.0
  

.. raw:: html

   <hr>
    
.. _ug_melt_q_data:

Melting Quantile Data (:func:`~kdiagram.utils.melt_q_data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility transforms a wide-format DataFrame containing
time-stamped quantile columns (e.g., ``prefix_date_qX.X``) into
a fully **"long"** or **"tidy"** format. Each row in the output
represents a single observation for a specific location (if
provided), time step, and quantile level. This process, often
called "melting" or "unpivoting," creates separate columns for
the time step identifier, the quantile level, and the
corresponding value.


**Key Parameters Explained:**

* **`value_prefix`**: The common prefix that identifies the
  quantile columns to be reshaped.
* **`dt_name`**: The name for the new column that will hold the
  extracted time step information (e.g., ``'year'``).
* **`spatial_cols`**: An optional list of identifier columns (like
  ``['lon', 'lat']``) that will be preserved and repeated for
  each new row.

**Conceptual Basis:**
This function implements the "melt" operation, a core principle
of creating tidy data :footcite:p:`Wickham2014`. It transforms a
wide table :math:`\mathbf{W}` into a long table :math:`\mathbf{L}`.

The source columns are named:

.. math::
   :label: eq:melt_q_source

   \mathrm{col}(t,\alpha) = \texttt{f"{value\_prefix}\_\{t\}\_q\{\alpha\}"}

The function unpivots these columns, creating new columns for the
time (:math:`t`), quantile level (:math:`\alpha`), and the value
itself (:math:`y_{s,t,\alpha}`).


**Example:**
The following example demonstrates how to convert a wide-format
DataFrame into a fully long, tidy format, which is ideal for use
with many modern plotting libraries and for detailed statistical
analysis.

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu
   import pandas as pd

   # --- Sample Wide Data ---
   wide_df = pd.DataFrame({
       'lon': [-118.25, -118.30],
       'lat': [34.05, 34.10],
       'subs_2022_q0.1': [1.2, 1.3],
       'subs_2022_q0.5': [1.5, 1.6],
       'subs_2023_q0.1': [1.7, 1.8],
   })
   print("Original Wide DataFrame:")
   print(wide_df)

   # --- Reshape the data into a long format ---
   long_df = kdu.melt_q_data(
       wide_df,
       value_prefix='subs',
       spatial_cols=('lon', 'lat'),
       dt_name='year'
   )
   print("\nMelted (Long) DataFrame:")
   print(long_df)

.. code-block:: text
   :caption: Expected Output

   Original Wide DataFrame:
         lon    lat  subs_2022_q0.1  subs_2022_q0.5  subs_2023_q0.1
   0 -118.25  34.05             1.2             1.5             1.7
   1 -118.30  34.10             1.3             1.6             1.8

   Melted (Long) DataFrame:
         lon    lat  year  quantile  subs
   0 -118.25  34.05  2022       0.1   1.2
   1 -118.30  34.10  2022       0.1   1.3
   2 -118.25  34.05  2022       0.5   1.5
   3 -118.30  34.10  2022       0.5   1.6
   4 -118.25  34.05  2023       0.1   1.7
   5 -118.30  34.10  2023       0.1   1.8
 

.. raw:: html

   <hr>
     
.. _ug_pivot_q_data:

Pivoting Quantile Data (:func:`~kdiagram.utils.pivot_q_data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility performs the inverse operation of melting, transforming
a DataFrame from a "long" or "semi-long" format back into a
**"wide" format**. It takes a table where different quantile
levels are in separate columns (e.g., ``'subs_q0.1'``,
``'subs_q0.5'``) and pivots it, creating a unique column for each
combination of a time step and a quantile level (e.g.,
``'subs_2022_q0.1'``). This is useful for reconstructing the
original data format or for creating summary tables.


**Key Parameters Explained:**

* **`value_prefix`**: The base name used in the input quantile
  columns (e.g., ``'subs'``) and for reconstructing the new,
  wide-format column names.
* **`dt_col`**: The name of the column in the input DataFrame that
  contains the time step identifiers (e.g., ``'year'``).
* **`spatial_cols`**: An optional list of identifier columns (like
  ``['lon', 'lat']``) that will be preserved as the index of the
  new wide-format DataFrame.


**Conceptual Basis:**
This function implements the "pivot" operation, which is the
reverse of melting. It takes a long or semi-long table
:math:`\mathbf{L}` and reconstructs the original wide table
:math:`\mathbf{W}`.

The input frame :math:`\mathbf{L}` is expected to have columns for
spatial identifiers, a time column, and a separate column for each
quantile level. The function then creates new columns in the
output frame :math:`\mathbf{W}` for each unique combination of a
time value :math:`t` and a quantile level :math:`\alpha`.

.. math::
   :label: eq:pivot_q_data

   \mathbf{L}\ \xrightarrow{\ \text{pivot}\ }\ \mathbf{W}


**Example:**
The following example demonstrates how to take a "semi-long"
DataFrame and pivot it back into a wide format, recreating the
original structure with time-stamped quantile columns.

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu
   import pandas as pd

   # --- Sample Long Data ---
   long_df = pd.DataFrame({
       'lon': [-118.25, -118.30, -118.25, -118.30],
       'lat': [34.05, 34.10, 34.05, 34.10],
       'year': [2022, 2022, 2023, 2023],
       'subs_q0.1': [1.2, 1.3, 1.7, 1.8],
       'subs_q0.5': [1.5, 1.6, 1.9, 2.0]
   })
   print("Original Long DataFrame:")
   print(long_df)

   # --- Pivot the data back to a wide format ---
   wide_df_reconstructed = kdu.pivot_q_data(
       long_df,
       value_prefix='subs',
       spatial_cols=('lon', 'lat'),
       dt_col='year'
   )
   print("\nPivoted (Wide) DataFrame:")
   # Sort columns for a consistent display
   print(wide_df_reconstructed.reindex(
       sorted(wide_df_reconstructed.columns), axis=1)
   )

.. code-block:: text
   :caption: Expected Output

   Original Long DataFrame:
         lon    lat  year  subs_q0.1  subs_q0.5
   0 -118.25  34.05  2022        1.2        1.5
   1 -118.30  34.10  2022        1.3        1.6
   2 -118.25  34.05  2023        1.7        1.9
   3 -118.30  34.10  2023        1.8        2.0

   Pivoted (Wide) DataFrame:
      lat      lon  subs_2022_q0.1  subs_2022_q0.5  subs_2023_q0.1  subs_2023_q0.5
   0  34.05 -118.250             1.2             1.5             1.7             1.9
   1  34.10 -118.300             1.3             1.6             1.8             2.0
   
.. raw:: html

   <hr>