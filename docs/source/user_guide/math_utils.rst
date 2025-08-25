.. _userguide_math_utils:

========================
Mathematical Utilities
========================

While the core of `k-diagram` is visualization, quantitative
analysis is the foundation of any good forecast evaluation. The
:mod:`kdiagram.utils.mathext` module provides a suite of mathematical
and data extraction utilities designed to compute key performance
metrics and prepare data for analysis.

These functions provide the numerical backbone for the diagnostic plots,
allowing users to access the underlying scores for custom analysis,
reporting, or integration into other workflows. They handle common
tasks such as calculating proper scoring rules, assessing calibration,
and extracting data from pandas DataFrames into a format suitable for
numerical computation.

Summary of Mathematical Utility Functions
-----------------------------------------

.. list-table:: Mathematical Utility Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.utils.mathext.get_forecast_arrays`
     - Extracts and validates true and predicted values from a
       DataFrame into NumPy arrays or pandas objects.
   * - :func:`~kdiagram.utils.mathext.compute_coverage_score`
     - Calculates the empirical coverage of a prediction interval,
       with options to check for over- and under-prediction.
   * - :func:`~kdiagram.utils.mathext.compute_winkler_score`
     - Computes the Winkler score, which evaluates both the sharpness
       and calibration of a prediction interval.
   * - :func:`~kdiagram.utils.mathext.compute_pinball_loss`
     - Calculates the Pinball Loss for a single quantile forecast, the
       foundational metric for quantile evaluation.
   * - :func:`~kdiagram.utils.mathext.compute_crps`
     - Approximates the Continuous Ranked Probability Score (CRPS) by
       averaging the Pinball Loss over all quantiles.
   * - :func:`~kdiagram.utils.mathext.compute_pit`
     - Computes the Probability Integral Transform (PIT) value for each
       observation to assess calibration.
   * - :func:`~kdiagram.utils.mathext.calculate_calibration_error`
     - Quantifies the overall calibration error using the
       Kolmogorov-Smirnov statistic on PIT values.
   * - :func:`~kdiagram.utils.mathext.build_cdf_interpolator`
     - Creates a callable empirical CDF from a set of quantile
       forecasts.
   * - :func:`~kdiagram.utils.mathext.minmax_scaler`
     - Scales features to a specified range, robust to zero-variance
       features.
  
.. raw:: html

    <hr>
    
         
.. _ug_get_forecast_arrays:

Extracting Forecast Arrays (:func:`~kdiagram.utils.get_forecast_arrays`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This is a flexible extraction utility that serves as the primary
bridge between a DataFrame-centric workflow and the NumPy-based
mathematical and plotting functions in `k-diagram`. It handles the
critical tasks of selecting the correct columns, cleaning the data
by dropping or filling missing values, and converting the output
to the desired format (either NumPy arrays or pandas objects).
This function streamlines that process by  handling the critical tasks of:

* Selecting the correct columns for true values and predictions.
* Cleaning the data by dropping or filling missing values.
* Converting the output to the desired format (NumPy arrays or pandas objects).

**Key Parameters Explained:**
While the function has many options, a few key parameters control 
its main behavior:

* **`return_as`**: Determines the output type. Use `'numpy'` (default) 
  when you need raw arrays for mathematical computations. Use `'pandas'` 
  when you want to preserve the index and column names for further data 
  manipulation.
* **`drop_na`**: Controls how missing data is handled. By default, it removes 
  any row where the `actual_col` or any of the `pred_cols` are NaN.
* **`squeeze`**: When you request a single prediction column 
  (`pred_cols='column_name'`), `squeeze=True` (default) returns a 1D array 
  or Series. Set it to `False` to maintain a 2D column vector shape 
  `(n, 1)`, which is sometimes required for other libraries.


**Conceptual Workflow**
This function executes a sequence of data validation and
transformation steps to ensure the output is clean and correctly
formatted for downstream analysis.

1.  **Column Selection**: The function first identifies the full
    set of required columns based on the ``actual_col`` and
    ``pred_cols`` arguments and validates their existence in the
    input DataFrame.

2.  **Data Subsetting and Cleaning**:
    a. A subset of the DataFrame containing only the required
       columns is created.
    b. If ``fillna`` is specified, missing values are imputed
       using the provided strategy.
    c. If ``drop_na=True``, rows with remaining missing values
       are dropped according to the ``na_policy`` ('any' or 'all').

3.  **Type Coercion (Optional)**: If ``ensure_numeric=True``, the
    function attempts to convert all selected columns to a numeric
    data type, either raising an error or coercing invalid values
    to NaN based on the ``coerce_numeric`` flag.

4.  **Output Formatting**: The cleaned and validated data is then
    converted to the desired output format specified by
    ``return_as`` ('numpy' or 'pandas'). If a single prediction
    column is requested and ``squeeze=True``, the output is
    reduced to a 1D array or Series.
    
**Mathematical Formulation:**
The function can be understood as a sequence of data transformation
operations. Let :math:`\mathbf{DF}` be the input DataFrame,
:math:`c_a` be the name of the actual column, and
:math:`\mathbf{C}_p` be the set of prediction column names. The
process is as follows:

.. math::
   :label: eq:get_forecast_arrays_algo

   \begin{aligned}
     & \text{1. Subset:} & \mathbf{DF}_{sub} &\leftarrow \mathbf{DF}[c_a \cup \mathbf{C}_p] \\
     & \text{2. Clean:} & \mathbf{DF}_{clean} &\leftarrow \mathcal{C}(\mathbf{DF}_{sub}, \text{policy}) \\
     & \text{3. Extract:} & \mathbf{y}_{true} &\leftarrow \mathbf{DF}_{clean}[c_a] \\
     & & \mathbf{Y}_{pred} &\leftarrow \mathbf{DF}_{clean}[\mathbf{C}_p] \\
     & \text{4. Return:} & & (\mathbf{y}_{true}, \mathbf{Y}_{pred})
   \end{aligned}

where:

- :math:`\mathbf{DF}_{sub}` is the subset of the original DataFrame
  containing only the columns of interest.
- :math:`\mathcal{C}` is a cleaning operator that applies the
  ``fillna`` and ``dropna`` policies to the subsetted data.
- :math:`\mathbf{y}_{true}` is the final vector of true values and
  :math:`\mathbf{Y}_{pred}` is the final vector or matrix of
  predicted values, extracted from the cleaned DataFrame.
      

**Examples:**
The following example demonstrates how to extract true values and
a set of quantile predictions from a DataFrame that contains
missing values.

**Basic Extraction (NumPy Output):**
This example demonstrates the default behavior: extracting 
true values and a set of quantile predictions from a DataFrame that 
contains a missing value. The function automatically drops the row with 
the `NaN` before returning the clean NumPy arrays.

.. code-block:: python
   :linenos:

   import pandas as pd
   import numpy as np
   import kdiagram.utils as kdu

   # Create a sample DataFrame with a missing value
   df = pd.DataFrame({
      'actual': [10, 20, 30, 40, np.nan],
      'pred_point': [12, 18, 33, 42, 48],
      'q10': [8, 15, 25, 35, 45],
      'q90': [12, 25, 35, 45, 55],
   })

   # Extract the actual values and the Q10/Q90 predictions
   y_true, y_preds_q = kdu.get_forecast_arrays(
    df, actual_col='actual', pred_cols=['q10', 'q90']
   )

   print("--- True Values (NumPy) ---")
   print(y_true)
   print("\n--- Quantile Predictions (NumPy) ---")
   print(y_preds_q)

.. code-block:: text
   :caption: Expected Output

   --- True Values (NumPy) ---
   [10. 20. 30. 40.]

   --- Quantile Predictions (NumPy) ---
   [[ 8 12]
    [15 25]
    [25 35]
    [35 45]]

**Pandas Output with Index:**
This example shows how to extract a single point prediction as a pandas 
Series, keeping the original index and without dropping missing values.

.. code-block:: python
   :linenos:

   # Using the same DataFrame as above
   y_preds_series = kdu.get_forecast_arrays(
       df,
       pred_cols='pred_point',
       return_as='pandas',
       drop_na=False
   )

   print("\n--- Point Predictions (pandas Series) ---")
   print(y_preds_series)

.. code-block:: text
  :caption: Expected Output

  --- Point Predictions (pandas Series) ---
  0    12
  1    18
  2    33
  3    42
  4    48
  Name: pred_point, dtype: int64

.. raw:: html

    <hr>
    
.. _ug_compute_coverage_score:

Computing Coverage Scores (:func:`~kdiagram.utils.compute_coverage_score`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility calculates the empirical coverage of a prediction
interval. It is a fundamental metric for assessing the
**calibration** of a forecast's uncertainty bounds. A forecast is
well-calibrated if its :math:`(1-\alpha) \cdot 100\%` prediction
intervals contain the true observed value approximately
:math:`(1-\alpha) \cdot 100\%` of the time.

The function is versatile, allowing you to calculate not just the
standard coverage score (the proportion of true values *within* the
interval), but also the proportion of values falling *above* or
*below* the interval. This is crucial for diagnosing the
**direction of miscalibration**.

**Key Parameters Explained:**

* **`method`**: This parameter controls which type of coverage is
    calculated.
    
    - ``'within'``: This is the standard coverage. It tells you the
      fraction of time your forecast was "correct" in its
      uncertainty estimate.
    - ``'below'``: This calculates the fraction of times the true
      value was *lower* than your lower bound. A high value
      indicates your model's intervals are systematically too high.
    - ``'above'``: This calculates the fraction of times the true
      value was *higher* than your upper bound. A high value
      indicates your model's intervals are systematically too low.

* **`return_counts`**: By default, the function returns a
    proportion (a float between 0 and 1). Setting this to ``True``
    returns the raw integer count, which can be useful for reports
    or further statistical tests.


**Mathematical Concept:**
The empirical coverage is a key diagnostic for checking if a
model's prediction intervals are well-calibrated. For a given
:math:`(1-\alpha) \cdot 100\%` prediction interval, the
empirical coverage should be close to :math:`1-\alpha`.

The function calculates one of three scores for a set of :math:`N`
observations, where :math:`\mathbf{1}` is the indicator function:

1.  **Within-Interval Coverage** (``method='within'``):

    .. math::
       :label: eq:coverage_within

       \text{Coverage} = \frac{1}{N} \sum_{i=1}^{N}
       \mathbf{1}\{y_{lower,i} \le y_{true,i} \le y_{upper,i}\}

2.  **Below-Interval Rate** (``method='below'``):

    .. math::
       :label: eq:coverage_below

       \text{Rate}_{below} = \frac{1}{N} \sum_{i=1}^{N}
       \mathbf{1}\{y_{true,i} < y_{lower,i}\}

3.  **Above-Interval Rate** (``method='above'``):

    .. math::
       :label: eq:coverage_above

       \text{Rate}_{above} = \frac{1}{N} \sum_{i=1}^{N}
       \mathbf{1}\{y_{true,i} > y_{upper,i}\}


**Examples:**

**Basic Usage:**
The following example demonstrates how to compute the standard
coverage score, as well as the raw count of observations that fall
below the specified interval.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Create sample data
   y_true = np.array([1, 2, 3, 4, 5, 6])
   y_lower = np.array([0, 3, 2, 5, 4, 7])
   y_upper = np.array([2, 4, 4, 6, 6, 8])

   # Calculate the standard coverage (4 out of 6 are within)
   coverage = kdu.compute_coverage_score(y_true, y_lower, y_upper)
   print(f"Coverage Score: {coverage:.2f}")

   # Calculate the number of points below the interval
   count_below = kdu.compute_coverage_score(
       y_true, y_lower, y_upper, method='below', return_counts=True
   )
   print(f"Count below interval: {count_below}")

.. code-block:: text
   :caption: Expected Output

   Coverage Score: 0.67
   Count below interval: 2

**Diagnosing Miscalibration:**
A well-calibrated 80% prediction interval (e.g., from Q10 to Q90)
should have approximately 10% of observations below the lower bound
and 10% above the upper bound. We can use this function to check.

.. code-block:: python
   :linenos:

   # Simulate a model whose intervals are systematically too low
   np.random.seed(0)
   y_true = np.random.normal(loc=10, scale=2, size=1000)
   y_lower_biased = y_true - 3 # Lower bound is too low
   y_upper_biased = y_true + 1 # Upper bound is too low

   # Calculate the rates
   rate_within = kdu.compute_coverage_score(
       y_true, y_lower_biased, y_upper_biased, method='within'
   )
   rate_below = kdu.compute_coverage_score(
       y_true, y_lower_biased, y_upper_biased, method='below'
   )
   rate_above = kdu.compute_coverage_score(
       y_true, y_lower_biased, y_upper_biased, method='above'
   )

   print(f"Coverage (within interval): {rate_within:.2f}")
   print(f"Rate below interval: {rate_below:.2f}")
   print(f"Rate above interval: {rate_above:.2f}")

.. code-block:: text
   :caption: Expected Output

   Coverage (within interval): 0.69
   Rate below interval: 0.00
   Rate above interval: 0.31

The output clearly shows the miscalibration: far too many
observations (31%) are falling above the upper bound, confirming
that the prediction intervals are biased low.

.. raw:: html

    <hr>
    
.. _ug_compute_winkler_score:

Computing the Winkler Score (:func:`~kdiagram.utils.compute_winkler_score`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This utility calculates the **Winkler score**, a proper scoring
rule designed specifically for evaluating prediction intervals. It
is a powerful metric because it simultaneously rewards **sharpness**
(narrow intervals) while heavily penalizing for a lack of
**calibration** (when the true value falls outside the interval).
A lower score is better.


**Key Parameters Explained**

* **`alpha`**: This is the significance level of the prediction
    interval. It determines how heavily the score penalizes
    observations that fall outside the bounds. For a 90% prediction
    interval (from Q5 to Q95), the `alpha` would be 0.1. For an 80%
    interval (Q10 to Q90), the `alpha` would be 0.2.


**Mathematical Concept:**
The Winkler score :footcite:p:`Gneiting2007b` is designed to
evaluate both the **sharpness** and **calibration** of a
prediction interval simultaneously. The score for a single
observation :math:`y` and a :math:`(1-\alpha)` prediction
interval :math:`[l, u]` is defined as:

.. math::
   :label: eq:winkler_score

   S_{\alpha}(l, u, y) = (u - l) +
   \begin{cases}
     \frac{2}{\alpha}(l - y) & \text{if } y < l \\
     0 & \text{if } l \le y \le u \\
     \frac{2}{\alpha}(y - u) & \text{if } y > u
   \end{cases}

The first term, :math:`(u - l)`, is the interval width, which
rewards sharpness (narrower intervals). The second term is a
penalty that is applied only if the observation falls outside
the interval. The penalty increases as the observation gets
further from the violated bound. This function returns the
average of this score over all observations.


**Example:**
The following example demonstrates how to calculate the Winkler
score for a set of forecasts.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Create sample data
   y_true = np.array([1, 5, 12])
   y_lower = np.array([2, 4, 8])
   y_upper = np.array([8, 6, 10])

   # For a 90% interval, alpha = 0.1
   # Obs 1 (y=1): outside. Width=6. Penalty=(2/0.1)*(2-1)=20. Score=26.
   # Obs 2 (y=5): inside. Width=2. Penalty=0. Score=2.
   # Obs 3 (y=12): outside. Width=2. Penalty=(2/0.1)*(12-10)=40. Score=42.
   # Average = (26 + 2 + 42) / 3 = 23.33

   score = kdu.compute_winkler_score(
       y_true, y_lower, y_upper, alpha=0.1
   )
   print(f"Average Winkler Score (alpha=0.1): {score:.2f}")

.. code-block:: text
   :caption: Expected Output

   Average Winkler Score (alpha=0.1): 23.33
 

.. raw:: html

    <hr>
    
      
.. _ug_compute_pinball_loss:

Computing the Pinball Loss (:func:`~kdiagram.utils.compute_pinball_loss`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This utility calculates the **Pinball Loss**, a fundamental metric
used to evaluate the accuracy of a single quantile forecast. It is
the building block for the Continuous Ranked Probability Score
(CRPS). A lower score indicates a more accurate quantile forecast.

---
**Mathematical Concept**
The Pinball Loss, :math:`\mathcal{L}_{\tau}`, is a proper scoring
rule for a single quantile forecast :math:`q` at level
:math:`\tau` against an observation :math:`y`. Its key feature is
that it asymmetrically penalizes errors. It gives a weight of
:math:`\tau` to under-predictions (when :math:`y > q`) and a
weight of :math:`(1 - \tau)` to over-predictions (when :math:`y < q`).

.. math::
   :label: eq:pinball_loss_def

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

This function calculates the average of this loss over all
provided observations.

---
**Example**
The following example demonstrates how to calculate the average
Pinball Loss for a 90th percentile (Q90) forecast.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Create sample data
   y_true = np.array([10, 10, 5])
   y_pred_q90 = np.array([8, 12, 5]) # Under-predict, over-predict, exact
   quantile = 0.9

   # Loss for y=10, q=8: (10-8) * 0.9 = 1.8
   # Loss for y=10, q=12: (12-10) * (1-0.9) = 0.2
   # Loss for y=5, q=5: (5-5) * 0.9 = 0.0
   # Average = (1.8 + 0.2 + 0.0) / 3 = 0.667

   loss = kdu.compute_pinball_loss(y_true, y_pred_q90, quantile)
   print(f"Average Pinball Loss for Q90: {loss:.3f}")

.. code-block:: text
   :caption: Expected Output

   Average Pinball Loss for Q90: 0.667
  

.. raw:: html

    <hr>
    
     
.. _ug_compute_crps:

Computing the CRPS (:func:`~kdiagram.utils.compute_crps`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This utility approximates the **Continuous Ranked Probability Score
(CRPS)**, a proper scoring rule that provides a single, comprehensive
measure of a probabilistic forecast's quality. It generalizes the
Mean Absolute Error to a probabilistic setting and simultaneously
assesses both **calibration** and **sharpness**. A lower CRPS value
indicates a better forecast.

**Mathematical Concept**
The Continuous Ranked Probability Score (CRPS) is a widely used
metric for evaluating probabilistic forecasts
:footcite:p:`Gneiting2007b`. For a single observation :math:`y`
and a predictive CDF :math:`F`, it is defined as the integrated
squared difference between the forecast CDF and the empirical CDF
of the observation:

.. math::
   :label: eq:crps_integral

   \text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(x) -
   \mathbf{1}\{x \ge y\})^2 dx

where :math:`\mathbf{1}` is the Heaviside step function.

When the forecast is given as a set of :math:`M` quantiles, the
CRPS is approximated by averaging the **Pinball Loss**
:math:`\mathcal{L}_{\tau}` over all provided quantile levels
:math:`\tau`. The final score is the average over all
observations and all quantiles.


**Interpretation**
The CRPS provides a single number to summarize the overall
performance of a probabilistic forecast.

* **Lower is Better**: A model with a lower average CRPS is
    considered superior, as it indicates a better combination of
    calibration and sharpness.
* **Units**: The CRPS is expressed in the same units as the
    observed variable, making it easy to interpret.


**Use Cases**

* To get a single, high-level summary score for comparing the
    overall performance of multiple probabilistic models.
* To use as the primary objective function when tuning a
    probabilistic forecasting model.
* To use alongside diagnostic plots like the PIT Histogram and
    Sharpness Diagram to understand *why* one model has a better
    CRPS than another.


**Example**
The following example demonstrates how to calculate the average
CRPS for a set of quantile forecasts.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Define true values and quantile forecasts for 2 observations
   y_true = np.array([10, 25])
   quantiles = np.array([0.1, 0.5, 0.9])
   y_preds = np.array([
       [8, 11, 13],  # Forecast for y_true = 10
       [20, 22, 26]   # Forecast for y_true = 25
   ])

   # Calculate the average CRPS
   crps_score = kdu.compute_crps(y_true, y_preds, quantiles)
   print(f"Average CRPS: {crps_score:.3f}")

.. code-block:: text
   :caption: Expected Output

   Average CRPS: 1.467

.. raw:: html

    <hr>
    
.. _ug_compute_pit:

Computing PIT Values (:func:`~kdiagram.utils.compute_pit`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility computes the **Probability Integral Transform (PIT)**
value for each individual observation in a dataset. The PIT is a
fundamental score for assessing the **calibration** of a
probabilistic forecast. The output of this function is an array
of PIT values, which can then be visualized (e.g., with
:func:`~kdiagram.plot.probabilistic.plot_pit_histogram`) or used
to calculate summary statistics of calibration.

**Mathematical Concept**
The Probability Integral Transform (PIT) is a foundational concept
in forecast verification :footcite:p:`Gneiting2007b`. For a
continuous predictive distribution with a Cumulative Distribution
Function (CDF) denoted by :math:`F`, the PIT value for a given
observation :math:`y` is calculated as :math:`F(y)`.

When a predictive distribution is represented by a finite set of
:math:`M` quantiles, as is common in machine learning, the PIT
value for each observation :math:`y_i` is approximated. It is
calculated as the fraction of the forecast quantiles that are
less than or equal to the observed value:

.. math::
   :label: eq:pit_quantile_util

   \text{PIT}_i = \frac{1}{M} \sum_{j=1}^{M}
   \mathbf{1}\{q_{i,j} \le y_i\}

where :math:`q_{i,j}` is the :math:`j`-th quantile forecast for
observation :math:`i`, and :math:`\mathbf{1}` is the indicator
function. If a forecast is perfectly calibrated, the resulting
array of PIT values will be uniformly distributed on the
interval :math:`[0, 1]`.


**Example**
The following example demonstrates how to compute the PIT value
for each observation in a small dataset.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Define true values and quantile forecasts for 3 observations
   y_true = np.array([10, 1, 5.5])
   quantiles = np.array([0.1, 0.5, 0.9])
   y_preds = np.array([
       [8, 11, 13],  # Forecast for y_true = 10
       [0, 0.5, 2],  # Forecast for y_true = 1
       [4, 5, 6]     # Forecast for y_true = 5.5
   ])

   # Calculate the PIT value for each observation
   # - For y=10, 1/3 quantiles are <= 10 -> PIT = 0.333
   # - For y=1, 2/3 quantiles are <= 1 -> PIT = 0.667
   # - For y=5.5, 2/3 quantiles are <= 5.5 -> PIT = 0.667
   pit_values = kdu.compute_pit(y_true, y_preds, quantiles)
   print(pit_values)

.. code-block:: text
   :caption: Expected Output

   [0.33333333 0.66666667 0.66666667]
   
.. raw:: html

    <hr>
    
.. _ug_calculate_calibration_error:

Calculating Calibration Error (:func:`~kdiagram.utils.calculate_calibration_error`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility quantifies the overall **calibration error** of a
probabilistic forecast with a single numerical score. It works by
first computing the Probability Integral Transform (PIT) values
and then using the Kolmogorov-Smirnov (KS) statistic to measure
how much their distribution deviates from the ideal uniform
distribution. A lower score indicates better calibration.


**Mathematical Concept:**
This function provides a summary statistic for the PIT histogram.
A perfectly calibrated forecast produces PIT values that are
uniformly distributed on :math:`[0, 1]`. The calibration error is
quantified by measuring the maximum difference between the
empirical Cumulative Distribution Function (CDF) of the PIT
values and the CDF of a perfect uniform distribution.

This maximum difference is the **Kolmogorov-Smirnov (KS)
statistic**, :math:`D_n`.

.. math::
   :label: eq:ks_statistic

   D_n = \sup_{x} | F_{PIT}(x) - U(x) |

where:

- :math:`F_{PIT}(x)` is the empirical CDF of the calculated PIT
  values.
- :math:`U(x)` is the CDF of the standard uniform distribution
  (i.e., :math:`U(x) = x`).
- :math:`\sup_{x}` denotes the supremum of the set of distances.

The score is between 0 and 1, where 0 represents perfect
calibration.


**Example**
The following example demonstrates how to calculate the
calibration error for a well-calibrated model and a poorly
calibrated (overconfident) model.

.. code-block:: python
   :linenos:

   import numpy as np
   from scipy.stats import norm
   import kdiagram.utils as kdu

   # Generate synthetic data
   np.random.seed(42)
   n_samples = 500
   y_true = np.random.normal(loc=10, scale=5, size=n_samples)
   quantiles = np.linspace(0.05, 0.95, 19)

   # A well-calibrated forecast
   good_preds = norm.ppf(
       quantiles, loc=10, scale=5
   ).reshape(1, -1).repeat(n_samples, axis=0)

   # A poorly calibrated (overconfident) forecast
   bad_preds = norm.ppf(
       quantiles, loc=10, scale=2.5
   ).reshape(1, -1).repeat(n_samples, axis=0)

   # Calculate the calibration error for both models
   calib_error_good = kdu.calculate_calibration_error(
       y_true, good_preds, quantiles
   )
   calib_error_bad = kdu.calculate_calibration_error(
       y_true, bad_preds, quantiles
   )

   print(f"Calibration Error (Good Model): {calib_error_good:.3f}")
   print(f"Calibration Error (Bad Model): {calib_error_bad:.3f}")

.. code-block:: text
   :caption: Expected Output

   Calibration Error (Good Model): 0.035
   Calibration Error (Bad Model): 0.298
  
.. raw:: html

    <hr>
     
.. _ug_build_cdf_interpolator:

Building a CDF Interpolator (:func:`~kdiagram.utils.build_cdf_interpolator`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This is an advanced utility that constructs a callable **empirical
Cumulative Distribution Function (CDF)** from a set of quantile
forecasts. It returns a new function that can be used to find the
estimated cumulative probability for any given value. This is a
foundational tool for advanced probabilistic analysis, such as
calculating PIT values or the probability of exceeding a
critical threshold.


**Mathematical Concept:**
The Probability Integral Transform (PIT) is a key concept in
probabilistic forecast evaluation :footcite:p:`Gneiting2007b`.
For a continuous predictive CDF :math:`F`, the PIT of an
observation :math:`y` is :math:`F(y)`. This utility constructs
an empirical approximation of :math:`F` for each forecast.

The function works by creating a closure: the returned
``_interpolator`` function "remembers" the quantile forecasts it
was built with. For each observation :math:`y_i`, it performs a
**linear interpolation** using the corresponding forecast quantiles
:math:`\mathbf{q}_i = (q_{i,1}, ..., q_{i,M})` as the x-coordinates
and the quantile levels :math:`\mathbf{\tau} = (\tau_1, ..., \tau_M)`
as the y-coordinates. This allows you to estimate the cumulative
probability for any value of :math:`y_i`.


**Example:**
The following example demonstrates how to build the interpolator
from a set of forecasts and then use the resulting function to
calculate the PIT values for several new observations.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Forecasts for 3 observations at 3 quantiles (0.1, 0.5, 0.9)
   preds_quantiles = np.array([
       [8, 10, 12],
       [0, 1, 2],
       [4, 5, 6]
   ])
   quantiles = np.array([0.1, 0.5, 0.9])

   # Build the interpolator from the forecast distributions
   cdf_func = kdu.build_cdf_interpolator(preds_quantiles, quantiles)

   # Now, use the new function to find the PIT for 3 observations
   y_true = np.array([10.0, 0.5, 5.5])
   pit_values = cdf_func(y_true)
   print(pit_values)

.. code-block:: text
   :caption: Expected Output

   [0.5 0.3 0.7]
   

.. raw:: html

    <hr>
    
.. _ug_minmax_scaler:

Min-Max Scaling (:func:`~kdiagram.utils.minmax_scaler`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This utility scales features to a specified range, most commonly
[0, 1]. Min-Max scaling is a standard preprocessing step for many
machine learning algorithms that are sensitive to the magnitude of
input features, such as neural networks and distance-based
algorithms. This implementation is flexible to features with zero
variance by adding a small epsilon to the denominator to prevent
division-by-zero errors.


**Mathematical Concept:**
The Min-Max scaling transformation is a linear operation. For each
feature (column) in the input data :math:`\mathbf{X}`, the
transformation is calculated as described in the scikit-learn
documentation :footcite:p:`scikit-learn`:

.. math::
   :label: eq:minmax_scaler

   X_{\text{scaled}} = \text{min}_{\text{range}} +
   (\text{max}_{\text{range}} - \text{min}_{\text{range}})
   \cdot \frac{\mathbf{X} - \min(\mathbf{X})}
   {(\max(\mathbf{X}) - \min(\mathbf{X})) + \varepsilon}

where:

- :math:`\text{min}_{\text{range}}` and
  :math:`\text{max}_{\text{range}}` are the bounds of the
  ``feature_range``.
- :math:`\min(\mathbf{X})` and :math:`\max(\mathbf{X})` are the
  minimum and maximum values of the feature.
- :math:`\varepsilon` is a small epsilon to ensure numerical
  stability.


**Example:**
The following example demonstrates how to scale a 2D array to the
default [0, 1] range and to a custom [-1, 1] range.

.. code-block:: python
   :linenos:

   import numpy as np
   import kdiagram.utils as kdu

   # Create a sample 2D array
   X = np.array([[1, 10], [2, 20], [3, 30]])

   # Scale to the default [0, 1] range
   X_scaled_default = kdu.minmax_scaler(X)
   print("--- Scaled to [0, 1] ---")
   print(X_scaled_default)

   # Scale to a custom [-1, 1] range
   X_scaled_custom = kdu.minmax_scaler(X, feature_range=(-1, 1))
   print("\n--- Scaled to [-1, 1] ---")
   print(X_scaled_custom)

.. code-block:: text
   :caption: Expected Output

   --- Scaled to [0, 1] ---
   [[0.  0. ]
    [0.5 0.5]
    [1.  1. ]]

   --- Scaled to [-1, 1] ---
   [[-1. -1.]
    [ 0.  0.]
    [ 1.  1.]]