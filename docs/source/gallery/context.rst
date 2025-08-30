.. _gallery_context:

===============================
Contextual Diagnostic Plots
===============================

While the core of `k-diagram` is its specialized polar visualizations,
a complete forecast evaluation often benefits from standard, familiar
plots that provide essential context. This gallery showcases the
functions in the :mod:`kdiagram.plot.context` module, which are
designed to be companions to the main polar diagnostics.

These plots cover fundamental diagnostics such as time series
comparisons, scatter plots, and error distribution analysis, all
following the consistent, DataFrame-centric API of the `k-diagram`
package.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots.
   

.. _gallery_plot_time_series:

------------------
Time Series Plot
------------------

This is the most fundamental contextual plot, providing a direct
visualization of the actual and predicted values over time. It is
an essential first step for understanding a model's performance,
showing how well it tracks the overall trend, seasonality, and
anomalies in the data.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(0)
   n_samples = 200
   time_index = pd.date_range("2023-01-01", periods=n_samples, freq='D')

   # A true signal with a trend and seasonality
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))

   # Model 1: A good forecast that tracks the signal well
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)

   # Model 2: A biased forecast that misses the trend
   y_pred_biased = y_true * 0.8 + 5 + np.random.normal(0, 2, n_samples)

   df = pd.DataFrame({
       'time': time_index,
       'actual': y_true,
       'good_model': y_pred_good,
       'biased_model': y_pred_biased,
       'q10': y_pred_good - 5, # Uncertainty band for the good model
       'q90': y_pred_good + 5,
   })

   # --- Plotting ---
   kdc.plot_time_series(
       df,
       x_col='time',
       actual_col='actual',
       pred_cols=['good_model', 'biased_model'],
       q_lower_col='q10',
       q_upper_col='q90',
       title="Time Series Forecast Comparison",
       savefig="gallery/images/gallery_plot_context_time_series_plot.png"
   )
   plt.close()

.. image:: ../images/gallery_plot_context_time_series_plot.png
   :alt: Example of a Time Series Plot
   :align: center
   :width: 90%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Time Series Plot** provides an immediate and intuitive
   overview of a forecast's performance against the true observed
   values.

   **Key Features:**

   * **Actual Values (Solid Black Line):** Represents the ground truth
     that the models are trying to predict.
   * **Predicted Values (Dashed Lines):** Each colored dashed line
     represents the point forecast from a different model.
   * **Uncertainty Interval (Shaded Gray Area):** Represents the
     prediction interval (e.g., from Q10 to Q90) for one of the models,
     visualizing its uncertainty.

   **🔍 In this Example:**

   * **Good Model (Purple):** The purple dashed line closely follows the
     solid black line, indicating that this model successfully captures
     both the upward trend and the seasonal cycles of the data. The
     uncertainty interval consistently contains the actual values.
   * **Biased Model (Yellow):** The yellow dashed line consistently
     deviates from the black line, especially at later time steps. It
     fails to capture the full extent of the upward trend, revealing a
     clear **systemic bias**.

   **💡 When to Use:**

   * As the **first step** in any forecast evaluation to get a high-level
     sense of model performance.
   * To visually compare the tracking ability of multiple models.
   * To check if the prediction intervals are wide enough to contain the
     actual values.

   
.. _gallery_plot_scatter_correlation:

---------------------------
Scatter Correlation Plot
---------------------------

This function creates a classic Cartesian scatter plot to visualize
the relationship between true observed values and model predictions.
It is an essential tool for assessing linear correlation, identifying
systemic bias, and spotting outliers.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using the same data as before) ---
   np.random.seed(0)
   n_samples = 200
   time_index = pd.date_range("2023-01-01", periods=n_samples, freq='D')
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)
   y_pred_biased = y_true * 0.8 + 5 + np.random.normal(0, 2, n_samples)

   df = pd.DataFrame({
       'time': time_index,
       'actual': y_true,
       'good_model': y_pred_good,
       'biased_model': y_pred_biased,
   })

   # --- Plotting ---
   kdc.plot_scatter_correlation(
       df,
       actual_col='actual',
       pred_cols=['good_model', 'biased_model'],
       title="Actual vs. Predicted Correlation",
       savefig="gallery/images/gallery_plot_context_time_scatter_corr.png"
   )
   plt.close()

.. image:: ../images/gallery_plot_context_time_scatter_corr.png
   :alt: Example of a Scatter Correlation Plot
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Scatter Correlation Plot** is a fundamental diagnostic for
   evaluating the performance of a point forecast.

   **Key Features:**

   * **Identity Line (Dashed Black Line):** This is the line of
     perfect correlation (y=x). For a perfect forecast, all points
     would lie directly on this line.
   * **Points:** Each point represents a single observation, with its
     x-coordinate being the true value and its y-coordinate being the
     predicted value.

   **🔍 In this Example:**

   * **Good Model (Purple):** The purple points are tightly clustered
     around the identity line. This indicates a strong linear
     correlation between the predictions and the true values, with
     low bias and low variance.
   * **Biased Model (Yellow):** The yellow points are more scattered
     and systematically deviate from the identity line. At low true
     values (left side), the points are above the line (over-prediction),
     while at high true values (right side), they fall below the line
     (under-prediction). This reveals a clear **systemic bias**.

   **💡 When to Use:**

   * To quickly assess the linear correlation between predictions and
     actuals.
   * To diagnose systemic bias by observing how the point cloud
     deviates from the identity line.
   * To identify individual outliers that are far from the main
     cluster of points.



.. _gallery_plot_error_distribution:

---------------------------
Error Distribution Plot
---------------------------

This function creates a histogram and a Kernel Density Estimate
(KDE) plot of the forecast errors. It is a fundamental diagnostic
for checking if a model's errors are unbiased (centered at zero)
and normally distributed, which are key assumptions for many
statistical methods.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using the same data as before) ---
   np.random.seed(0)
   n_samples = 200
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)

   df = pd.DataFrame({
       'actual': y_true,
       'good_model': y_pred_good,
   })

   # --- Plotting ---
   kdc.plot_error_distribution(
       df,
       actual_col='actual',
       pred_col='good_model',
       title="Error Distribution (Good Model)",
       savefig="gallery/images/gallery_plot_context_error_dist.png"
   )
   plt.close()

.. image:: ../images/gallery_plot_context_error_dist.png
   :alt: Example of an Error Distribution Plot
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Error Distribution Plot** is a crucial tool for validating
   the assumptions of a forecasting model.

   **Key Features:**

   * **Histogram (Blue Bars):** Shows the frequency of errors within
     specific bins.
   * **KDE Curve (Orange Line):** Provides a smooth estimate of the
     error's probability density function, making it easy to see
     the shape of the distribution.

   **🔍 In this Example:**

   * **Unbiased Errors:** The distribution is clearly centered around
     zero, which indicates that the "Good Model" has no significant
     systemic bias.
   * **Normal Distribution:** The shape of both the histogram and the
     KDE curve resembles a classic "bell curve," suggesting that the
     errors are approximately normally distributed. This is a desirable
     property for a well-behaved model.

   **💡 When to Use:**

   * To check if a model's errors are unbiased (i.e., have a mean of
     zero).
   * To assess if the errors follow a normal distribution, which is a
     key assumption for constructing valid confidence intervals.
   * To identify skewness or heavy tails in the error distribution,
     which might indicate that the model struggles with certain types
     of predictions.


.. _gallery_plot_qq:

-----------------------------
Q-Q Plot for Error Normality
-----------------------------

This function generates a Quantile-Quantile (Q-Q) plot, a standard
graphical method for comparing a dataset's distribution to a
theoretical distribution (in this case, the normal distribution). It is
an essential tool for visually checking if the forecast errors are
normally distributed.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using the same data as before) ---
   np.random.seed(0)
   n_samples = 200
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)

   df = pd.DataFrame({
       'actual': y_true,
       'good_model': y_pred_good,
   })

   # --- Plotting ---
   kdc.plot_qq(
       df,
       actual_col='actual',
       pred_col='good_model',
       title="Q-Q Plot of Errors (Good Model)",
       savefig="gallery/images/gallery_plot_context_qq_plot.png"
   )
   plt.close()

.. image:: ../images/gallery_plot_context_qq_plot.png
   :alt: Example of a Q-Q Plot
   :align: center
   :width: 70%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Q-Q Plot** is a powerful visual diagnostic for checking the
   normality assumption of a model's errors, which is a prerequisite
   for many statistical inference methods.

   **Key Features:**

   * **Reference Line (Blue Line):** This line represents a perfect
     normal distribution.
   * **Error Quantiles (Red Dots):** Each dot represents a quantile from
     the actual error distribution plotted against the corresponding
     quantile from a theoretical normal distribution.

   **🔍 In this Example:**

   * The red dots fall very closely along the straight blue reference
     line. This indicates that the distribution of the "Good Model's"
     errors is **approximately normal**.
   * There are minor deviations at the tails (the far left and right
     ends of the line), which is common with finite samples, but no
     strong, systematic pattern of deviation is visible.

   **💡 When to Use:**

   * To visually verify the assumption that a model's errors are
     normally distributed.
   * To diagnose specific types of non-normality. For example, an
     "S"-shaped curve in the points can indicate that the error
     distribution has "heavy tails" (more outliers than a normal
     distribution).
   * As a companion to the `plot_error_distribution` to get a more
     rigorous check of the distribution's shape.


   
.. _gallery_plot_error_autocorrelation:

------------------------------------
Error Autocorrelation (ACF) Plot
------------------------------------

This function creates an Autocorrelation Function (ACF) plot of the
forecast errors. It is a critical diagnostic for time series models,
used to check if there is any remaining temporal structure (i.e.,
patterns) in the residuals.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using the same data as before) ---
   np.random.seed(0)
   n_samples = 200
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)

   df = pd.DataFrame({
       'actual': y_true,
       'good_model': y_pred_good,
   })

   # --- Plotting ---
   kdc.plot_error_autocorrelation(
       df,
       actual_col='actual',
       pred_col='good_model',
       title="Error Autocorrelation (Good Model)",
       savefig="gallery/images/gallery_plot_context_error_autocorr_acf.png"
   )
   plt.close()

.. image:: ../images/gallery_plot_context_error_autocorr_acf.png
   :alt: Example of an Error Autocorrelation Plot
   :align: center
   :width: 85%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Error Autocorrelation Plot** is a key tool for verifying
   that a time series model has captured all the predictable patterns
   in the data.

   **Key Features:**

   * **Lag (X-axis):** Represents the time step separation. A lag of
     1 means the correlation between an error and the error from the
     previous time step.
   * **Autocorrelation (Y-axis):** Shows the correlation of the error
     series with its past values.
   * **Significance Bands (Shaded Area/Dashed Lines):** This area
     represents the threshold for statistical significance. Correlations
     that fall inside this band are generally considered to be noise.

   **🔍 In this Example:**

   * The plot shows that nearly all the autocorrelation values for
     different lags fall **within the significance bands**. This is the
     **ideal result**.
   * It indicates that the errors of the "Good Model" are behaving like
     random noise, with no significant temporal patterns left to be
     modeled.

   **💡 When to Use:**

   * To check if a time series model's errors are independent over time,
     which is a key assumption for a well-specified model.
   * To identify remaining seasonality or trend in the residuals. If you
     see significant spikes at regular intervals (e.g., every 12 lags
     for monthly data), it means your model has not fully captured the
     seasonal pattern.
   * To guide model improvement. Significant autocorrelation suggests that
     the model could be improved by adding more lags or other time-based
     features.

   
.. _gallery_plot_error_pacf:

------------------------------------------
Error Partial Autocorrelation (PACF) Plot
------------------------------------------

This function creates a Partial Autocorrelation Function (PACF) plot
of the forecast errors. It is a critical companion to the ACF plot,
used to identify the direct relationship between an error and its
past values, after removing the effects of intervening lags.

.. code-block:: python
   :linenos:

   import kdiagram.plot.context as kdc
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using the same data as before) ---
   np.random.seed(0)
   n_samples = 200
   y_true = (np.linspace(0, 20, n_samples) +
             10 * np.sin(np.arange(n_samples) * 2 * np.pi / 30) +
             np.random.normal(0, 2, n_samples))
   y_pred_good = y_true + np.random.normal(0, 1.5, n_samples)

   df = pd.DataFrame({
       'actual': y_true,
       'good_model': y_pred_good,
   })

   # --- Plotting ---
   # Note: Requires the 'statsmodels' package to be installed.
   try:
       kdc.plot_error_pacf(
           df,
           actual_col='actual',
           pred_col='good_model',
           title="Partial Autocorrelation of Forecast Errors",
           savefig="gallery/images/gallery_plot_context_error_partial_autocorr_pacf.png"
       )
   except ImportError:
       print("Skipping PACF plot: statsmodels is not installed.")
   finally:
       plt.close()

.. image:: ../images/gallery_plot_context_error_partial_autocorr_pacf.png
   :alt: Example of an Error Partial Autocorrelation Plot
   :align: center
   :width: 85%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Partial Autocorrelation Plot** is essential for diagnosing
   the specific type of remaining structure in a time series model's
   residuals.

   **Key Features:**

   * **Lag (X-axis):** Represents the time step separation.
   * **Partial Autocorrelation (Y-axis):** Shows the correlation
     between an error and its value at a specific lag, after
     removing the influence of the correlations at shorter lags.
   * **Significance Band (Shaded Blue Area):** Correlations that
     fall inside this band are not statistically significant from zero.

   **🔍 In this Example:**

   * The plot shows that, apart from the mandatory correlation at
     lag 0, all other partial autocorrelations fall **within the
     blue significance band**.
   * This is the **ideal result** for a well-specified model. It
     indicates that after accounting for the correlation at lag 1,
     there is no significant *direct* correlation between an error
     and its value at any subsequent lag.

   **💡 When to Use:**

   * In conjunction with the ACF plot to identify the order of an
     autoregressive (AR) model. A sharp cut-off in the PACF plot
     (e.g., a significant spike at lag `p` and non-significant
     spikes thereafter) is a classic signature of an AR(p) process.
   * To confirm that a model's errors are random and that no
     significant linear relationships between lagged errors remain.

.. raw:: html

   <hr>
