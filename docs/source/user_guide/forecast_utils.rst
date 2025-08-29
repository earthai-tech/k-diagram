.. _userguide_forecast_utils:

====================================
Forecast Utilities
====================================

Effective visualization begins with well-prepared data. The
:mod:`kdiagram.utils.forecast` module provides a suite of powerful
helper functions designed to handle common data preparation and
wrangling tasks associated with forecast evaluation.

These utilities are built to work seamlessly with pandas DataFrames and
bridge the gap between common data formats (like multi-column "wide"
DataFrames) and the specific NumPy array inputs required by many of
`k-diagram`'s plotting and mathematical functions. Using these helpers
can significantly reduce boilerplate code and ensure your data is
correctly structured for analysis.

Summary of Forecast Utility Functions
-------------------------------------

.. list-table:: Forecast Utility Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.utils.compute_forecast_errors`
     - Computes various forecast errors (raw, absolute, etc.) from
       actual and predicted values.
   * - :func:`~kdiagram.utils.compute_interval_width`
     - Calculates the width of one or more prediction intervals from
       pairs of quantile columns.
   * - :func:`~kdiagram.utils.calculate_probabilistic_scores`
     - Computes per-observation probabilistic scores (PIT, CRPS,
       sharpness) from quantile forecasts.
   * - :func:`~kdiagram.utils.pivot_forecasts_long`
     - Reshapes multi-horizon forecast data from a wide format to a
       convenient long format.
   * - :func:`~kdiagram.utils.bin_by_feature`
     - Bins data by a feature and computes aggregate statistics,
       powering conditional analysis plots.

.. raw:: html

    <hr>
    
.. _ug_compute_forecast_errors:

Computing Forecast Errors (:func:`~kdiagram.utils.compute_forecast_errors`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This is a core data preparation utility that calculates the
difference between true and predicted values for one or more models.
It supports several common error types and adds the results as new
columns to the DataFrame, making it easy to prepare data for the
diagnostic plots in the :mod:`kdiagram.plot.errors` module.

**Mathematical Concept:**
The forecast error (or residual), :math:`e_i`, for an observation
:math:`i` is the fundamental quantity for diagnosing model
performance. This function calculates it in several standard forms:

1.  **Raw Error**: The simple difference, which preserves the
    direction of the error (positive for under-prediction,
    negative for over-prediction).

    .. math::
       :label: eq:raw_error

       e_i = y_{true,i} - y_{pred,i}

2.  **Absolute Error**: The magnitude of the error, often used in
    metrics like Mean Absolute Error (MAE).

    .. math::
       :label: eq:abs_error

       e_{abs,i} = |y_{true,i} - y_{pred,i}|

3.  **Squared Error**: Penalizes larger errors more heavily and is
    the basis for metrics like Mean Squared Error (MSE).

    .. math::
       :label: eq:sq_error

       e_{sq,i} = (y_{true,i} - y_{pred,i})^2

4.  **Percentage Error**: Expresses the error as a percentage of
    the true value. Note that this can be unstable if
    :math:`y_{true,i}` is close to zero.

    .. math::
       :label: eq:pct_error

       e_{\%,i} = 100 \cdot \frac{y_{true,i} - y_{pred,i}}{y_{true,i}}


**Example**
The following example demonstrates how to compute both raw and
absolute errors for two different models.

.. code-block:: python
   :linenos:

   import pandas as pd
   import kdiagram.utils as kdu

   # Create a sample DataFrame
   df = pd.DataFrame({
       'actual': [10, 20, 30],
       'model_A_preds': [12, 18, 33],
       'model_B_preds': [10, 25, 28],
   })

   # Calculate raw errors for both models
   df_raw_errors = kdu.compute_forecast_errors(
       df, 'actual', 'model_A_preds', 'model_B_preds'
   )
   print("--- Raw Errors ---")
   print(df_raw_errors)

   # Calculate absolute errors with a custom prefix
   df_abs_errors = kdu.compute_forecast_errors(
       df, 'actual', 'model_A_preds', 'model_B_preds',
       error_type='absolute', prefix='abs_error_'
   )
   print("\n--- Absolute Errors ---")
   print(df_abs_errors)

.. code-block:: text
   :caption: Expected Output

   --- Raw Errors ---
      actual  model_A_preds  ...  error_model_A_preds  error_model_B_preds
   0      10             12  ...                   -2                    0
   1      20             18  ...                    2                   -5
   2      30             33  ...                   -3                    2

   [3 rows x 5 columns]

   --- Absolute Errors ---
      actual  model_A_preds  ...  abs_error_model_A_preds  abs_error_model_B_preds
   0      10             12  ...                        2                        0
   1      20             18  ...                        2                        5
   2      30             33  ...                        3                        2

   [3 rows x 5 columns]

.. raw:: html

    <hr>
     
.. _ug_compute_interval_width:

Computing Interval Width (:func:`~kdiagram.utils.compute_interval_width`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This is a fundamental data preparation utility that calculates the
width of one or more prediction intervals by taking the difference
between upper and lower quantile columns. The resulting interval
width is a key measure of a forecast's **sharpness**.

**Mathematical Concept:**
The width of a prediction interval is the most direct measure of
a forecast's **sharpness**, a key property of probabilistic
forecasts :footcite:p:`Gneiting2007b`. A smaller width
indicates a more precise, or sharper, forecast.

For a given observation :math:`i`, the interval width :math:`w_i`
is the simple difference between the upper and lower quantile
forecasts:

.. math::
   :label: eq:interval_width_calc

   w_i = q_{upper, i} - q_{lower, i}


**Example:**
The following example demonstrates how to compute both the 80%
(q10 to q90) and 90% (q05 to q95) interval widths for a model.

.. code-block:: python
   :linenos:

   import pandas as pd
   import kdiagram.utils as kdu

   # Create a sample DataFrame with quantile forecasts
   df = pd.DataFrame({
       'q10_model_A': [1, 2], 'q90_model_A': [10, 12],
       'q05_model_A': [0, 1], 'q95_model_A': [11, 13]
   })

   # Calculate the 80% and 90% interval widths
   widths_df = kdu.compute_interval_width(
       df,
       ['q10_model_A', 'q90_model_A'],
       ['q05_model_A', 'q95_model_A']
   )
   print(widths_df)

.. code-block:: text
   :caption: Expected Output

      q10_model_A  q90_model_A  ...  width_q90_model_A  width_q95_model_A
   0            1           10  ...                  9                 11
   1            2           12  ...                 10                 12

   [2 rows x 6 columns]
   

.. raw:: html

    <hr>
    
.. _ug_calculate_probabilistic_scores:

Calculating Probabilistic Scores (:func:`~kdiagram.utils.calculate_probabilistic_scores`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This utility provides a per-observation breakdown of three
fundamental scores for evaluating probabilistic forecasts: the
Probability Integral Transform (PIT), sharpness, and the
Continuous Ranked Probability Score (CRPS). It returns a
DataFrame where each row corresponds to an observation, making it
easy to analyze the distribution of these scores.

**Mathematical Concept:**
A good probabilistic forecast is judged by the joint properties
of **calibration** (reliability) and **sharpness** (precision)
:footcite:p:`Gneiting2007b`. This function calculates metrics
that capture these qualities.

1.  **Probability Integral Transform (PIT)**: This score assesses
    **calibration**. For each observation :math:`y_i`, the PIT is
    approximated as the fraction of forecast quantiles less than
    or equal to the observation.

    .. math::

       \text{PIT}_i = \frac{1}{M} \sum_{j=1}^{M}
       \mathbf{1}\{q_{i,j} \le y_i\}

2.  **Sharpness**: This score assesses **precision**. It is the
    width of the prediction interval between the lowest
    (:math:`q_{min}`) and highest (:math:`q_{max}`) provided
    quantiles for each observation :math:`i`.

    .. math::

       \text{Sharpness}_i = y_{i, q_{max}} - y_{i, q_{min}}

3.  **Continuous Ranked Probability Score (CRPS)**: This is an
    overall score that rewards both calibration and sharpness.
    It is approximated as the average of the **Pinball Loss**
    across all :math:`M` quantiles for each observation :math:`i`.

    .. math::

       \text{CRPS}_i \approx \frac{1}{M} \sum_{j=1}^{M}
       2 \mathcal{L}_{\tau_j}(q_{i,j}, y_i)

**Example:**
The following example demonstrates how to compute these three
scores for a set of quantile forecasts.

.. code-block:: python
   :linenos:

   import numpy as np
   from scipy.stats import norm
   import kdiagram.utils as kdu

   # Generate synthetic data
   np.random.seed(42)
   n_samples = 5
   y_true = np.random.normal(loc=10, scale=2, size=n_samples)
   quantiles = np.array([0.1, 0.5, 0.9])
   y_preds = norm.ppf(
       quantiles, loc=y_true[:, np.newaxis], scale=1.5
   )

   # Calculate the scores for each observation
   scores_df = kdu.calculate_probabilistic_scores(
       y_true, y_preds, quantiles
   )
   print(scores_df)

.. code-block:: text
   :caption: Expected Output

      pit_value  sharpness      crps
   0   0.666667   3.844655  0.128155
   1   0.666667   3.844655  0.128155
   2   0.666667   3.844655  0.128155
   3   0.666667   3.844655  0.128155
   4   0.666667   3.844655  0.128155
   

.. raw:: html

    <hr>
    
.. _ug_pivot_forecasts_long:

Pivoting Forecasts (:func:`~kdiagram.utils.pivot_forecasts_long`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This is a powerful data wrangling utility that reshapes multi-horizon
forecast data from a **wide** format to a **long** format. Wide-format
data, with separate columns for each horizon's quantiles (e.g.,
'q10_2023', 'q50_2023', 'q10_2024', etc.), is common but can be
inconvenient for plotting and analysis. This function transforms it
into a "long" format with dedicated columns like 'horizon', 'q_low',
and 'q_median', which is the standard for many visualization libraries.

**Example:**
The following example demonstrates how to convert a DataFrame with
two years of quantile forecasts into a tidy, long-format table.

.. code-block:: python
   :linenos:

   import pandas as pd
   import kdiagram.utils as kdu

   # Create a sample wide-format DataFrame
   df_wide = pd.DataFrame({
       'location_id': ['A', 'B'],
       'q10_2023': [10, 12], 'q50_2023': [15, 18], 'q90_2023': [20, 24],
       'q10_2024': [12, 14], 'q50_2024': [18, 21], 'q90_2024': [24, 28],
   })

   print("--- Original Wide DataFrame ---")
   print(df_wide)

   # Reshape the data into a long format
   df_long = kdu.pivot_forecasts_long(
       df_wide,
       qlow_cols=['q10_2023', 'q10_2024'],
       q50_cols=['q50_2023', 'q50_2024'],
       qup_cols=['q90_2023', 'q90_2024'],
       horizon_labels=['Year 2023', 'Year 2024'],
       id_vars='location_id'
   )

   print("\n--- Reshaped Long DataFrame ---")
   print(df_long)

.. code-block:: text
   :caption: Expected Output

   --- Original Wide DataFrame ---
     location_id  q10_2023  q50_2023  q90_2023  q10_2024  q50_2024  q90_2024
   0           A        10        15        20        12        18        24
   1           B        12        18        24        14        21        28

   --- Reshaped Long DataFrame ---
     location_id  q_low  q_median  q_high    horizon
   0           A     10        15      20  Year 2023
   1           B     12        18      24  Year 2023
   2           A     12        18      24  Year 2024
   3           B     14        21      28  Year 2024
  

.. raw:: html

    <hr>
     
.. _ug_bin_by_feature:

Binning by Feature (:func:`~kdiagram.utils.bin_by_feature`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This is a powerful data wrangling utility that groups a DataFrame
into bins based on the values in a specified column
(``bin_on_col``). It then calculates aggregate statistics (like
mean, std, etc.) for one or more target columns within each bin.
This is the core logic that powers conditional analysis plots like
:func:`~kdiagram.plot.errors.plot_error_bands`.

**Example**
The following example demonstrates how to calculate the mean and
standard deviation of a forecast's error, binned by the magnitude
of the forecast itself. This is a common technique for diagnosing
heteroscedasticity.

.. code-block:: python
   :linenos:

   import pandas as pd
   import kdiagram.utils as kdu

   # Create a sample DataFrame
   df = pd.DataFrame({
       'forecast_value': [10, 12, 20, 22, 30, 32],
       'error': [-1, 1.5, -2, 2.5, -3, 3.5]
   })

   # Calculate the mean and std dev of the error, binned by forecast value
   binned_stats = kdu.bin_by_feature(
       df,
       bin_on_col='forecast_value',
       target_cols='error',
       n_bins=3,
       agg_funcs=['mean', 'std']
   )
   print(binned_stats)

.. code-block:: text
   :caption: Expected Output

     forecast_value_bin error          
                         mean       std
   0    (9.978, 17.333]  0.25  1.767767
   1   (17.333, 24.667]  0.25  3.181981
   2     (24.667, 32.0]  0.25  4.596194
   
   
.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::