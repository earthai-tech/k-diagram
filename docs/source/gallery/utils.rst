.. _gallery_utils:

=============================
Utility Function Examples
=============================

This section of the gallery demonstrates practical usage of the utility
functions provided within `k-diagram`. These functions are primarily
designed to help identify, validate, and reshape quantile data stored
in pandas DataFrames, preparing it for analysis or visualization.

Each example includes Python code using sample data and shows the
expected output printed to the console.

----------------------------
Detecting Quantile Columns
----------------------------

Uses :func:`~kdiagram.utils.detect_quantiles_in` to find columns matching
quantile naming patterns (e.g., ``prefix_date_qX.X`` or ``prefix_qX.X``).
This example shows detection based on prefix, date, and returning
different output types.

.. _gallery_detect_quantiles:

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu # Assuming utils are exposed here
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

--------------------------------
Building Quantile Column Names
--------------------------------

Uses :func:`~kdiagram.utils.build_q_column_names` to construct expected
quantile column names based on patterns and validate their existence in
a DataFrame.

.. _gallery_build_q_names:

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
   # Assuming strict_match=True (default)
   names_2024 = kdu.build_q_column_names(
       df, quantiles=[0.1, 0.9], value_prefix='precip', dt_value=['2024']
   )
   print(names_2024)

   print("\nBuilding names for 2025, quantiles 0.1, 0.9 (one missing):")
   # dt_value can often handle integers as years
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

---------------------------------------------
Reshaping Quantile Data (Wide to Semi-Long)
---------------------------------------------

Uses :func:`~kdiagram.utils.reshape_quantile_data` to transform
wide-format quantile data (e.g., ``prefix_date_qX.X`` columns) into a
format where each row is a location/time combination and different
quantiles become columns (e.g., ``prefix_qX.X``).

.. _gallery_reshape_q_data:

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

   # --- Usage ---
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

---------------------------------------
Melting Quantile Data (Wide to Long)
---------------------------------------

Uses :func:`~kdiagram.utils.melt_q_data` to convert a wide-format
DataFrame into a fully long ("tidy") format with separate columns for
time, quantile level, and the measurement value.

*(Note: The exact output structure of melt_q_data might depend on its specific
implementation; this example shows a typical "melted" structure.)*

.. _gallery_melt_q_data:

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

   # --- Usage ---
   long_df = kdu.melt_q_data(
       wide_df,
       value_prefix='subs',
       spatial_cols=('lon', 'lat'),
       dt_name='year' # Name for the time column
   )
   print("\nMelted (Long) DataFrame:")
   print(long_df)

.. code-block:: text
   :caption: Expected Output (Illustrative Long Format)

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


-----------------------------------------
Pivoting Quantile Data (Long to Wide)
-----------------------------------------

Uses :func:`~kdiagram.utils.pivot_q_data` to perform the inverse of
melting; converts a long-format DataFrame back into a wide format where
each time step and quantile combination becomes a separate column
(e.g., ``prefix_date_qX.X``).

.. _gallery_pivot_q_data:

.. code-block:: python
   :linenos:

   import kdiagram.utils as kdu
   import pandas as pd

   # --- Sample Long Data (output from reshape or similar) ---
   long_df = pd.DataFrame({
       'lon': [-118.25, -118.30, -118.25, -118.30],
       'lat': [34.05, 34.10, 34.05, 34.10],
       'year': [2022, 2022, 2023, 2023],
       'subs_q0.1': [1.2, 1.3, 1.7, 1.8], # Quantiles are columns
       'subs_q0.5': [1.5, 1.6, 1.9, 2.0]
   })
   print("Original Long DataFrame:")
   print(long_df)

   # --- Usage ---
   wide_df_reconstructed = kdu.pivot_q_data(
       long_df,
       value_prefix='subs',
       spatial_cols=('lon', 'lat'),
       dt_col='year' # Column containing time steps
   )
   print("\nPivoted (Wide) DataFrame:")
   # Sort columns for consistent output display
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
   0  34.10 -118.300             1.3             1.6             1.8             2.0
   1  34.05 -118.250             1.2             1.5             1.7             1.9