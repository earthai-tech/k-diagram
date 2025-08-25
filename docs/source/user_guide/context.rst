.. _userguide_context:

=============================
Contextual Diagnostic Plots
=============================

While the core of `k-diagram` is its specialized polar visualizations,
a complete forecast evaluation often benefits from standard, familiar
plots that provide essential context. The :mod:`kdiagram.plot.context`
module provides a suite of these fundamental diagnostic plots, designed
to be companions to the main polar diagrams.

These functions cover essential diagnostics such as time series
comparisons, scatter plots for correlation, and various checks on the
distribution and structure of forecast errors. They follow the same
consistent, DataFrame-centric API as the rest of the `k-diagram`
package, creating a cohesive and complete toolkit for forecast
evaluation.

Summary of Contextual Plotting Functions
----------------------------------------

.. list-table:: Contextual Diagnostic Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.context.plot_time_series`
     - Plots the actual and predicted values over time, with optional
       uncertainty bands.
   * - :func:`~k-diagram.plot.context.plot_scatter_correlation`
     - Creates a standard scatter plot of true vs. predicted values
       to assess correlation and bias.
   * - :func:`~kdiagram.plot.context.plot_error_distribution`
     - Visualizes the distribution of forecast errors with a
       histogram and KDE plot.
   * - :func:`~kdiagram.plot.context.plot_qq`
     - Generates a Q-Q plot to check if forecast errors are
       normally distributed.
   * - :func:`~kdiagram.plot.context.plot_error_autocorrelation`
     - Creates an ACF plot to check for remaining temporal patterns
       in the forecast errors.
   * - :func:`~kdiagram.plot.context.plot_error_pacf`
     - Creates a PACF plot to identify the specific structure of
       autocorrelation in the errors.

Common Plotting Parameters
--------------------------

Most plotting functions in `k-diagram` share a common set of
parameters for controlling the input data and the plot's
appearance. These are explained here once for brevity.

.. list-table:: Common Parameters
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``df``
     - The input :class:`pandas.DataFrame` containing the data.
   * - ``names``
     - A list of strings to use as labels for different models or
       prediction sets in the legend.
   * - ``title``, ``xlabel``, ``ylabel``
     - Strings to set the title and axis labels for the plot.
   * - ``figsize``
     - A tuple of ``(width, height)`` in inches for the figure size.
   * - ``cmap``
     - The name of the Matplotlib colormap to use for plots with
       multiple colors.
   * - ``show_grid`` & ``grid_props``
     - Controls the visibility and styling of the plot's grid lines.
   * - ``savefig`` & ``dpi``
     - The file path and resolution for saving the plot to a file.
     
.. raw:: html

   <hr>
   
.. _ug_plot_time_series:

Time Series Plot (:func:`~kdiagram.plot.context.plot_time_series`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This is the most fundamental contextual plot, providing a direct
visualization of the actual and predicted values over time. It is
an essential first step for understanding a model's performance,
showing how well it tracks the overall trend, seasonality, and
anomalies in the data. The function is flexible, allowing for the
comparison of multiple models and the inclusion of an uncertainty
interval.


**Key Parameters:**
In addition to the :ref:`common parameters <common_plotting_parameters>`,
this function uses:

* **`x_col`**: The column to use for the x-axis. If not provided,
  the DataFrame's index is used, which is ideal for time series
  data.
* **`actual_col`**: The column containing the ground truth values,
  typically plotted as a solid line for reference.
* **`pred_cols`**: A list of one or more columns containing the
  point forecasts from different models.
* **`q_lower_col` / `q_upper_col`**: Optional columns that define
  the bounds of a prediction interval, which will be visualized
  as a shaded band.


**Conceptual Basis:**
A time series plot is a direct visualization of one or more time-
dependent variables. It maps a time-like variable :math:`t` (from
``x_col`` or the index) to the x-axis and the value of a series
:math:`y` (from ``actual_col`` or ``pred_cols``) to the y-axis.

The plot visualizes the functions :math:`y_{true} = f(t)` and
:math:`y_{pred} = g(t)`, allowing for a direct comparison of their
behavior over the entire domain. The shaded uncertainty band
represents the interval :math:`[q_{lower}(t), q_{upper}(t)]`,
providing a visual representation of the forecast's uncertainty at
each point in time.


**Interpretation:**
The plot provides an immediate and intuitive overview of a
forecast's performance against the true observed values.

* **Tracking Performance**: A good forecast (dashed line) will
  closely follow the true values (solid line), capturing the
  major trends and seasonal patterns.
* **Bias**: A forecast that is consistently above or below the
  true value line has a clear systemic bias.
* **Uncertainty Bands**: The shaded gray area shows the prediction
  interval. A well-calibrated model should have the true value
  line fall within this band most of the time.


**Use Cases:**
* As the **first step** in any forecast evaluation to get a high-level
  sense of model performance.
* To visually compare the tracking ability of multiple models.
* To check if the prediction intervals are wide enough to contain the
  actual values and to see if the uncertainty changes over time.


**Example:**
See the gallery example and code: :ref:`gallery_plot_time_series`.

**Example**
The following example demonstrates how to plot the true values
against the forecasts of two different models. It also includes a
shaded uncertainty band for the "good" model.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np

   # --- Generate synthetic time series data ---
   np.random.seed(0)
   n_samples = 200
   time_index = pd.date_range("2023-01-01", periods=n_samples, freq='D')

   # A true signal with trend and seasonality
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))

   # A good forecast that tracks the signal well
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)
   # A biased forecast that misses the trend
   y_pred_biased = y_true * 0.8 + 5 + np.random.normal(0, 2, n_samples)

   df = pd.DataFrame({
       'time': time_index,
       'actual': y_true,
       'good_model': y_pred_good,
       'biased_model': y_pred_biased,
       'q10': y_pred_good - 5, # Uncertainty band for the good model
       'q90': y_pred_good + 5,
   })

   # --- Generate the plot ---
   kdc.plot_time_series(
       df,
       x_col='time',
       actual_col='actual',
       pred_cols=['good_model', 'biased_model'],
       q_lower_col='q10',
       q_upper_col='q90',
       title="Time Series Forecast Comparison"
   )
   
See the gallery :ref:`gallery_plot_time_series` for more examples.

.. raw:: html

   <hr>
   
.. _ug_plot_scatter_correlation:

Scatter Correlation Plot (:func:`~kdiagram.plot.context.plot_scatter_correlation`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This function creates a classic Cartesian scatter plot to visualize
the relationship between true observed values and model predictions.
It is an essential tool for assessing linear correlation, identifying
systemic bias, and spotting outliers. This plot serves as the
standard Cartesian counterpart to the polar relationship plots.


**Key Parameters Explained**
In addition to the common parameters, this function uses:

* **`actual_col`**: The column containing the ground truth values,
  which will be plotted on the x-axis.
* **`pred_cols`**: A list of one or more columns containing the
  point forecasts from different models, which will be plotted on
  the y-axis.
* **`show_identity_line`**: A boolean that controls the display of
  the dashed y=x line. This line is the reference for a perfect
  forecast.


**Mathematical Concept**
This plot directly visualizes the relationship between two variables
by plotting each observation :math:`i` as a point
:math:`(y_{true,i}, y_{pred,i})`.

The primary reference is the **identity line**, defined by the
equation:

.. math::
   :label: eq:identity_line

   y = x

For a perfect forecast, every predicted value would equal its
corresponding true value, and all points would fall exactly on this
line. Deviations from this line represent prediction errors.

**Interpretation:**
The plot provides a direct visual assessment of a point forecast's
performance.

* **Correlation**: If the points form a tight, linear cloud around
  the identity line, it indicates a strong positive correlation
  between the predictions and the true values.
* **Bias**: If the point cloud is systematically shifted above or
  below the identity line, it reveals a model bias. Points above
  the line are over-predictions, while points below are
  under-predictions.
* **Outliers**: Individual points that are far from the main cloud
  of points represent significant, one-off prediction errors.


**Use Cases:**

* To quickly assess the linear correlation between predictions and
  actuals.
* To diagnose systemic bias by observing how the point cloud
  deviates from the identity line.
* To identify individual outliers that are far from the main
  cluster of points.


**Example**
See the gallery example and code: :ref:`gallery_plot_scatter_correlation`.

.. raw:: html

   <hr>
   
.. _ug_plot_error_distribution:

Error Distribution Plot (:func:`~kdiagram.plot.context.plot_error_distribution`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a histogram and a Kernel Density Estimate
(KDE) plot of the forecast errors. It is a fundamental diagnostic
for checking if a model's errors are unbiased (centered at zero)
and normally distributed, which are key assumptions for many
statistical methods.

**Key Parameters:**
In addition to the :ref:`common parameters <common_plotting_parameters>`,
this function uses:

* **`actual_col`**: The column containing the ground truth values.
* **`pred_col`**: The column containing the point forecast values.
* **`**hist_kwargs`**: Additional keyword arguments (e.g., `bins`,
  `kde_color`) are passed directly to the underlying
  :func:`~kdiagram.utils.plot_hist_kde` function.

**Mathematical Concept:**
The plot visualizes the distribution of the forecast errors,
:math:`e_i = y_{true,i} - y_{pred,i}`, using two standard
non-parametric methods.

1.  **Histogram**: The range of errors is divided into a series
    of bins, and the height of each bar represents the frequency
    (or density) of errors that fall into that bin.
2.  **Kernel Density Estimate (KDE)**: This provides a smooth,
    continuous estimate of the error's probability density
    function, :math:`\hat{f}_h(e)`, based on the foundational
    work in density estimation :footcite:p:`Silverman1986`.

**Interpretation:**
The plot provides an immediate visual summary of the error
distribution's key characteristics.

* **Bias (Central Tendency)**: The location of the highest peak
    of the distribution. For an unbiased model, this peak should
    be centered at zero.
* **Variance (Spread)**: The width of the distribution. A narrow
    distribution indicates low-variance, consistent errors, while
    a wide distribution indicates high-variance, less reliable
    predictions.
* **Shape**: The overall shape of the curve. A symmetric "bell
    curve" suggests the errors are normally distributed. Skewness
    or multiple peaks (bimodality) can indicate that the model
    struggles with certain types of predictions.


**Use Cases:**

* To check if a model's errors are unbiased (i.e., have a mean of
    zero).
* To assess if the errors follow a normal distribution, which is a
    key assumption for constructing valid confidence intervals.
* To identify skewness or heavy tails in the error distribution,
    which might indicate that the model has systematic failings.


**Example**
See the gallery example and code:
:ref:`gallery_plot_error_distribution`.

.. raw:: html

   <hr>
   
.. _ug_plot_qq:

Q-Q Plot (:func:`~kdiagram.plot.context.plot_qq`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function generates a **Quantile-Quantile (Q-Q) plot**, a
standard graphical method for comparing a dataset's distribution
to a theoretical distribution (in this case, the normal
distribution). It is an essential tool for visually checking if
the forecast errors are normally distributed, which is a key
assumption for many statistical methods.

**Key Parameters:**
In addition to the :ref:`common parameters <common_plotting_parameters>`,
this function uses:

* **`actual_col`**: The column containing the ground truth values.
* **`pred_col`**: The column containing the point forecast values.
* **`**scatter_kwargs`**: Additional keyword arguments are passed
  to the underlying scatter plot for the data points.


**Mathematical Concept:**
A Q-Q plot is constructed by plotting the quantiles of two
distributions against each other. In this case, it compares the
quantiles of the empirical distribution of the forecast errors,
:math:`e_i = y_{true,i} - y_{pred,i}`, against the theoretical
quantiles of a standard normal distribution,
:math:`\mathcal{N}(0, 1)`.

If the two distributions are identical :eq:`eq:identity_line`, the resulting 
points will fall perfectly along the identity line :math:`y=x`.


**Interpretation:**
The plot provides a powerful visual diagnostic for checking the
normality assumption of a model's errors.

* **Reference Line (Blue Line)**: This line represents a perfect
  theoretical normal distribution.
* **Error Quantiles (Red Dots)**: Each dot represents a quantile from
  the actual error distribution plotted against the corresponding
  quantile from a theoretical normal distribution.
* **Alignment**: If the red dots fall closely along the straight blue
  reference line, it indicates that the error distribution is
  approximately normal.
* **Deviations**: Systematic deviations from the line indicate a
  departure from normality. For example, an "S"-shaped curve can
  indicate that the error distribution has "heavy tails" (more
  outliers than a normal distribution).


**Use Cases:**

* To visually verify the assumption that a model's errors are
  normally distributed.
* To diagnose specific types of non-normality, such as skewness or
  heavy tails.
* As a companion to the :func:`~kdiagram.plot.context.plot_error_distribution`
  to get a more rigorous check of the distribution's shape.


**Example:**
See the gallery example and code: :ref:`gallery_plot_qq`.

.. raw:: html

   <hr>
   
   
.. _ug_plot_error_autocorrelation:

Error Autocorrelation (ACF) Plot (:func:`~kdiagram.plot.context.plot_error_autocorrelation`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates an **Autocorrelation Function (ACF) plot**
of the forecast errors. It is a critical diagnostic for time series
models, used to check if there is any remaining temporal structure
(i.e., patterns) in the residuals. A well-specified model should
have errors that are uncorrelated over time, behaving like random
noise.

**Key Parameters:**
In addition to the :ref:`common parameters <common_plotting_parameters>`,
this function uses:

* **`actual_col`**: The column containing the ground truth values.
* **`pred_col`**: The column containing the point forecast values.
* **`**acf_kwargs`**: Additional keyword arguments are passed
  directly to the underlying ``pandas.plotting.autocorrelation_plot``
  function.

**Mathematical Concept:**
The Autocorrelation Function (ACF) at lag :math:`k` measures the
correlation between a time series and its own past values. For a
series of errors :math:`e_t`, the ACF is defined as:

.. math::
   :label: eq:acf

   \rho_k = \frac{\text{Cov}(e_t, e_{t-k})}{\text{Var}(e_t)}

This plot displays the values of :math:`\rho_k` for a range of
different lags :math:`k`. The plot also includes significance
bands (typically at 95% confidence), which provide a threshold
for determining if a correlation is statistically significant or
likely due to random chance.


**Interpretation:**
The plot is used to identify if predictable patterns remain in the
model's errors.

* **Significance Bands**: The horizontal lines or shaded area
    represent the significance threshold. Autocorrelations that
    fall **inside** this band are generally considered to be
    statistically insignificant from zero.
* **Significant Lags**: If one or more spikes extend **outside**
    the significance bands, it indicates that the errors are
    correlated with their past values at those lags. This means
    the model has failed to capture all the predictable
    information in the time series.


**Use Cases:**

* To check if a time series model's errors are independent over
    time (i.e., resemble white noise), which is a key assumption
    for a well-specified model.
* To identify remaining seasonality or trend in the residuals. If
    you see significant spikes at regular intervals (e.g., every
    12 lags for monthly data), it means your model has not fully
    captured the seasonal pattern.
* To guide model improvement. Significant autocorrelation suggests
    that the model could be improved by adding more lags or other
    time-based features.


**Example**
See the gallery example and code:
:ref:`gallery_plot_error_autocorrelation`.

.. raw:: html

   <hr>
   
.. _ug_plot_error_pacf:

Error Partial Autocorrelation (PACF) Plot (:func:`~kdiagram.plot.context.plot_error_pacf`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Partial Autocorrelation Function (PACF)
plot** of the forecast errors. It is a critical companion to the
ACF plot and is used to identify the *direct* relationship
between an error and its past values, after removing the effects
of the intervening lags.


**Key Parameters**
In addition to the :ref:`common parameters <common_plotting_parameters>`,
this function uses:

* **`actual_col`**: The column containing the ground truth values.
* **`pred_col`**: The column containing the point forecast values.
* **`**pacf_kwargs`**: Additional keyword arguments are passed
  directly to the underlying ``statsmodels.graphics.tsaplots.plot_pacf``
  function.


**Mathematical Concept:**
While the ACF at lag :math:`k` shows the total correlation between
:math:`e_t` and :math:`e_{t-k}`, the PACF shows the **partial
correlation**. It measures the correlation between :math:`e_t` and
:math:`e_{t-k}` after removing the linear dependence on the
intermediate observations :math:`e_{t-1}, e_{t-2}, ..., e_{t-k+1}`.

This helps to isolate the direct relationship at a specific lag,
making it a key tool for identifying the order of autoregressive
(AR) processes.


**Interpretation:**
The PACF plot is used in conjunction with the ACF plot to diagnose
the specific structure of any remaining patterns in the residuals.

* **Significance Band**: The shaded area represents the
    significance threshold. Spikes that extend **outside** this
    band are statistically significant.
* **Cut-off Pattern**: A key pattern to look for is a sharp
    "cut-off." If the PACF plot shows a significant spike at lag
    :math:`p` and non-significant spikes thereafter, it is a
    strong indication of an autoregressive (AR) process of order
    :math:`p`.


**Use Cases:**

* To identify the order of an autoregressive (AR) model that might
    be missing from your forecast model.
* To confirm that a model's errors are random and that no
    significant *direct* linear relationships between lagged errors
    remain.
* As a complementary tool to the ACF plot for a more complete
    diagnosis of time series residuals.


**Example**
See the gallery example and code:
:ref:`gallery_plot_error_pacf`.

.. raw:: html

   <hr>
   
.. rubric:: References

.. footbibliography::