.. _gallery_uncertainty: 

==============================
Uncertainty Visualizations
==============================

This page showcases examples of plots specifically designed for
exploring, diagnosing, and communicating aspects of predictive
uncertainty using `k-diagram`.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``../images/gallery_actual_vs_predicted.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots (likely an ``images`` subdirectory relative to
   this file, e.g., `../images/`).

.. _gallery_plot_actual_vs_predicted: 

----------------------
Actual vs. Predicted
----------------------

Compares actual observed values against point predictions (e.g.,
Q50) sample-by-sample. Useful for assessing basic accuracy and
bias.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(66)
   n_points = 120
   df = pd.DataFrame({'sample': range(n_points)})
   signal = 20 + 15 * np.cos(np.linspace(0, 6 * np.pi, n_points))
   df['actual'] = signal + np.random.randn(n_points) * 3
   df['predicted'] = signal * 0.9 + np.random.randn(n_points) * 2 + 2

   # --- Plotting ---
   kd.plot_actual_vs_predicted(
       df=df,
       actual_col='actual',
       pred_col='predicted',
       title='Gallery: Actual vs. Predicted (Dots)',
       line=False, # Use dots instead of lines
       r_label="Value",
       actual_props={'s': 25, 'alpha': 0.7, 'color':'black'}, # Explicit color
       pred_props={'s': 35, 'marker': 'x', 'alpha': 0.7, 'color':'red'}, # Explicit color & size
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_actual_vs_predicted.png"
   )
   plt.close() # Close the plot window after saving

.. image:: ../images/gallery_actual_vs_predicted.png
   :alt: Actual vs. Predicted Plot Example
   :align: center
   :width: 70%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This polar plot provides a direct visual comparison between actual
   values (black dots) and model-predicted medians (red crosses, Q50)
   across a set of samples arranged angularly (by index). Each
   point's distance from the center (radius) corresponds to the
   magnitude of its value.

   **Key Insights:**

   * **Accuracy & Discrepancies:** Close alignment between black dots
     and red crosses indicates accurate predictions for that sample.
     Deviations highlight errors. The grey connecting lines (if
     ``line=True``) emphasize the error magnitude and direction.
   * **Systematic Bias:** Look for consistent patterns where red
     crosses are generally inside (under-prediction) or outside
     (over-prediction) the black dots.
   * **Outliers:** Samples with unusually large gaps between actual
     and predicted values are easily spotted.

   **🔍 In this Example:**

   * The points form clear cyclical patterns, matching the
     underlying cosine wave used in data generation.
   * Predictions (red crosses) generally track the actual values
     (black dots) but exhibit some scatter (noise) and slight
     magnitude differences, particularly near the peaks (outer radius)
     and troughs (inner radius) of the cycle.
   * There might be a subtle tendency for red crosses to be slightly
     closer to the center than black dots, suggesting mild
     underprediction or damping in the simulated model.

   **💡 When to Use:**

   Use this plot as a primary diagnostic tool to:

   * Get an initial visual assessment of point-forecast accuracy.
   * Quickly identify overall model bias (systematic over/under
     prediction).
   * Spot specific samples or regions (if angle is meaningful)
     with large prediction errors.
   * Complement numerical scores (MAE, RMSE) with an intuitive
     overview of model fit, especially for cyclical or ordered data.

.. raw:: html

    <hr>

.. _gallery_plot_anomaly_magnitude:

--------------------
Anomaly Magnitude
--------------------

Highlights instances where the actual value falls outside the
prediction interval [Qlow, Qup]. Shows the location (angle), type
(color), and severity (radius) of anomalies.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   n_points = 180
   df = pd.DataFrame({'sample_id': range(n_points)})
   df['actual'] = np.random.normal(loc=20, scale=5, size=n_points)
   df['q10'] = df['actual'] - np.random.uniform(2, 6, size=n_points)
   df['q90'] = df['actual'] + np.random.uniform(2, 6, size=n_points)
   # Add anomalies
   under_indices = np.random.choice(n_points, 20, replace=False)
   df.loc[under_indices, 'actual'] = df.loc[under_indices, 'q10'] - \
                                      np.random.uniform(1, 5, size=20)
   available = list(set(range(n_points)) - set(under_indices))
   over_indices = np.random.choice(available, 20, replace=False)
   df.loc[over_indices, 'actual'] = df.loc[over_indices, 'q90'] + \
                                     np.random.uniform(1, 5, size=20)

   # --- Plotting ---
   kd.plot_anomaly_magnitude(
       df=df,
       actual_col='actual',
       q_cols=['q10', 'q90'],
       title="Gallery: Prediction Anomaly Magnitude",
       cbar=True,
       s=30,
       verbose=0, # Keep output clean for gallery
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_anomaly_magnitude.png"
   )
   plt.close()

.. image:: ../images/gallery_anomaly_magnitude.png
   :alt: Anomaly Magnitude Plot Example
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Anomaly Magnitude Plot** provides valuable insights into
   prediction interval failures, showing how far actual values
   deviate when they fall outside the predicted bounds (e.g.,
   Q10 and Q90). Only points representing anomalies are plotted.

   **Key Features:**

   * **Angle (θ):** Represents the sample's position or index
     in the dataset, arranged circularly.
   * **Radius (r):** Directly corresponds to the **magnitude** of
     the anomaly (:math:`|y_{actual} - y_{bound}|`). Larger radii
     indicate more severe prediction interval failures.
   * **Color:** Distinguishes the **type** of anomaly using
     different colormaps (defaults: Blues for under-prediction,
     Reds for over-prediction).
   * **Color Intensity:** Further emphasizes the anomaly's
     **severity**, with darker/more intense colors typically
     representing larger magnitudes (larger radius).

   **🔍 In this Example:**

   * The plot clearly separates under-predictions (blue points,
     where `actual < q10`) and over-predictions (red points,
     where `actual > q90`).
   * Points further from the center represent larger deviations from
     the predicted [q10, q90] range. We can visually identify the
     most significant prediction failures.
   * The angular distribution shows where these failures occur within
     the sample order. Clusters might indicate problematic regimes.

   **💡 When to Use:**

   This plot is essential for diagnosing model uncertainty calibration
   and identifying high-risk predictions:

   * **Pinpoint Interval Failures:** Identify exactly which samples
     fall outside the expected range.
   * **Assess Anomaly Severity:** Quantify *how far* outside the bounds
     the actual values lie.
   * **Analyze Error Type:** Determine if the model tends to fail more
     often through under-prediction or over-prediction.
   * **Guide Model Refinement:** Focus attention on samples/regions
     with large anomalies where uncertainty estimation needs improvement.

   It offers a geographically or temporally focused investigation into
   where and how prediction *intervals* fail, complementing plots
   that assess point forecast accuracy.

.. raw:: html

    <hr>

.. _gallery_plot_overall_coverage:

--------------------
Overall Coverage
--------------------

Calculates and displays the overall empirical coverage rate(s)
compared to the nominal rate. Useful for comparing average
interval calibration across models. Shown here with a radar plot
for two simulated models.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   y_true = np.random.rand(100) * 10
   # Model 1 (e.g., ~80% coverage)
   y_pred_q1 = np.sort(np.random.normal(
       loc=y_true[:, np.newaxis], scale=1.5, size=(100, 2)), axis=1)
   # Model 2 (e.g., ~60% coverage - narrower intervals)
   y_pred_q2 = np.sort(np.random.normal(
       loc=y_true[:, np.newaxis], scale=0.8, size=(100, 2)), axis=1)
   q_levels = [0.1, 0.9] # Nominal 80% interval

   # --- Plotting ---
   kd.plot_coverage(
       y_true,
       y_pred_q1,
       y_pred_q2,
       names=['Model A (Wider)', 'Model B (Narrower)'],
       q=q_levels,
       kind='radar', # Use radar chart for profile comparison
       title='Gallery: Overall Coverage Comparison (Radar)',
       cov_fill=True,
       verbose=0,
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_coverage_radar.png"
   )
   plt.close()

.. image:: ../images/gallery_coverage_radar.png
   :alt: Overall Coverage Radar Plot Example
   :align: center
   :width: 70%

.. topic:: Analysis and Interpretation
    :class: hint

    This plot compares the **overall empirical coverage rate**
    between two simulated models using a radar plot. It helps assess
    the **interval calibration** across models, evaluating how well
    their predicted intervals (e.g., Q10 to Q90, implying 80%
    nominal coverage here) capture the actual values *on average*
    over the dataset.

    **Analysis and Interpretation:**

    In this example radar plot:

    * **Model A (Wider):** Exhibits a higher coverage rate
      (closer to the outer edge, likely near the target 80% or
      higher). This indicates its wider prediction intervals
      successfully encompass a larger fraction of the true values.
      While seemingly safer, it might suggest the model is
      conservative, potentially overestimating uncertainty.
    * **Model B (Narrower):** Shows a lower coverage rate (points
      closer to the center). Its narrower intervals fail to capture
      the true value more often. This model might seem more precise
      but likely underestimates uncertainty, increasing the risk of
      errors where reality falls outside the predicted range.

    The radar layout effectively contrasts the coverage profiles.
    Points closer to the outer boundary (radius 1.0) represent
    better average coverage relative to the defined interval.

    **When to Use This Plot:**

    * **Comparing Interval Calibration:** Ideal for a high-level
      comparison of how well different models' uncertainty estimates
      are calibrated (on average). Is one model consistently too wide
      (over-covered) or too narrow (under-covered)?
    * **Model Selection:** Aids in selecting a model based on risk
      tolerance. Model A might be preferred for risk-averse tasks,
      while Model B might be chosen if tighter (though less reliable)
      intervals are desired.
    * **Summarizing Reliability:** Provides a concise summary of the
      average reliability of prediction intervals.

.. raw:: html

    <hr>

.. _gallery_plot_coverage_diagnostic: 

----------------------
Coverage Diagnostic
----------------------

Visualizes coverage success (radius 1) or failure (radius 0) for
each individual data point. Helps diagnose *where* intervals fail.
The solid line shows the overall average coverage rate. Shown here
using bars.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(88)
   n_points = 200
   df = pd.DataFrame({'point_id': range(n_points)})
   df['actual_val'] = np.random.normal(loc=5, scale=1.5, size=n_points)
   df['q_lower'] = 5 - np.random.uniform(1, 3, n_points)
   df['q_upper'] = 5 + np.random.uniform(1, 3, n_points)
   # Some points deliberately outside
   df.loc[::15, 'actual_val'] = df.loc[::15, 'q_upper'] + 1

   # --- Plotting ---
   kd.plot_coverage_diagnostic(
       df=df,
       actual_col='actual_val',
       q_cols=['q_lower', 'q_upper'],
       title='Gallery: Point-wise Coverage Diagnostic (Bars)',
       as_bars=True, # Display as bars instead of scatter
       fill_gradient=True, # Show background gradient
       coverage_line_color='darkorange', # Example customization
       verbose=0,
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_coverage_diagnostic_bars.png"
   )
   plt.close()

.. image:: ../images/gallery_coverage_diagnostic_bars.png
   :alt: Coverage Diagnostic Plot Example (Bars)
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    This plot provides a **point-wise coverage diagnostic**, showing
    if the actual value for *each sample* falls within the
    prediction interval (e.g., Q10-Q90). Each bar (or point if
    ``as_bars=False``) represents one sample, arranged angularly
    by index.

    **🔍 Key Insights from this Example:**

    * **Bar Height/Radius:** Indicates coverage status. A bar
      reaching radius **1** means the actual value was *inside* the
      interval (success). A bar at radius **0** means the actual
      value was *outside* (failure).
    * **Color (Implied):** Although not the primary focus here,
      the points/bars are often colored by coverage status (e.g.,
      using the `cmap` parameter, green for 1, red for 0).
    * **Average Coverage Line:** The solid circular line (orange
      in this example code's customization) is drawn at the
      radius corresponding to the **overall coverage rate**
      (e.g., 0.75 if 75% of points are covered). This provides an
      immediate visual benchmark against the nominal target (e.g.,
      0.80 for a Q10-Q90 interval) and the plot boundaries (0 & 1).
    * **Patterns:** Look for clusters of bars at radius 0. These
      indicate ranges of samples (or specific conditions if the
      angle represented something else) where the model's intervals
      consistently fail.

    **💡 When to Use This Plot:**

    * **Diagnosing Interval Failures:** Go beyond the average score
      provided by ``plot_coverage`` to see *which specific samples*
      are missed by the prediction intervals.
    * **Identifying Systematic Errors:** Determine if coverage
      failures are random or concentrated in certain parts of the
      data distribution (represented by angles).
    * **Visual Calibration Assessment:** Get a detailed view of how
      well the empirical coverage matches the nominal rate point-
      by-point, complementing the overall average line.
    * **Guiding Model Improvement:** Pinpoint problematic samples
      or regimes where uncertainty quantification needs refinement.

.. raw:: html

    <hr>


.. _gallery_plot_interval_consistency: 

-------------------------
Interval Consistency
-------------------------

Analyzes the stability of the prediction interval width (Qup - Qlow)
for each location over multiple time steps. Radius shows
variability (CV or Std Dev); color often shows average Q50. High
radius means inconsistent width.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   n_points = 100
   n_years = 4
   years = list(range(2021, 2021 + n_years))
   df = pd.DataFrame({'id': range(n_points)})
   qlow_cols, qup_cols, q50_cols = [], [], []
   for i, year in enumerate(years):
       ql, qu, q50 = f'val_{year}_q10', f'val_{year}_q90', f'val_{year}_q50'
       qlow_cols.append(ql); qup_cols.append(qu); q50_cols.append(q50)
       base_low = np.random.rand(n_points)*5 + i*0.2
       width = np.random.rand(n_points)*3 + 1 + np.sin(
           np.linspace(0, np.pi, n_points))*i # Vary width
       df[ql] = base_low; df[qu] = base_low + width
       df[q50] = base_low + width/2 + np.random.randn(n_points)*0.5

   # --- Plotting ---
   kd.plot_interval_consistency(
       df=df,
       qlow_cols=qlow_cols,
       qup_cols=qup_cols,
       q50_cols=q50_cols, # Color by average Q50
       use_cv=True,       # Radius = Coefficient of Variation of width
       title='Gallery: Interval Width Consistency (CV)',
       acov='half_circle',
       cmap='viridis',
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_interval_consistency_cv.png"
   )
   plt.close()

.. image:: ../images/gallery_interval_consistency_cv.png
   :alt: Interval Consistency Plot Example
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot analyzes the **stability** of prediction interval
   widths (e.g., Q90 - Q10) over multiple time steps or forecast
   horizons for different samples (locations/indices arranged
   angularly).

   **Key Features:**

   * **Radius (r):** Corresponds to the **variability** of the
     interval width over time for each sample. By default
     (``use_cv=True``), it shows the **Coefficient of Variation (CV)**,
     representing relative variability. If ``use_cv=False``, it shows
     the standard deviation (absolute variability).
     * *Large Radius:* High inconsistency (width fluctuates a lot).
     * *Small Radius:* High consistency (width is stable).
   * **Color:** Typically represents the **average Q50** (median
     prediction) across the time steps for each sample, providing
     context about the prediction magnitude. Darker/cooler colors
     often indicate lower average Q50, brighter/warmer colors
     indicate higher average Q50 (depending on the `cmap`).
   * **Angle (θ):** Represents the sample index or location.

   **🔍 Key Insights from this Example:**

   * Points far from the center indicate locations where the model's
     uncertainty estimate (interval width) is **less stable** across
     the different years included in the data.
   * Points clustered near the center represent locations with
     **consistent** interval widths over time.
   * The color mapping (using `viridis`) shows whether high/low
     consistency (radius) correlates with high/low average predicted
     values (color). For instance, are the most inconsistent
     predictions (large radius) happening in areas predicted to have
     high values (yellow) or low values (purple)?

   **💡 When to Use This Plot:**

   * **Assess Model Stability:** Identify samples/locations where
     uncertainty predictions are erratic or stable over time/horizons.
   * **Diagnose Uncertainty Drift:** While other plots show average
     drift, this shows the *variability* aspect of drift for each
     point.
   * **Compare Relative vs. Absolute Variability:** Toggle `use_cv`
     to understand if large fluctuations are significant relative to
     the mean width (CV) or just large in absolute terms (Std Dev).
   * **Guide Risk Assessment:** Focus on predictions where interval
     widths are stable (low radius) for more reliable planning, and
     treat predictions with high variability (high radius) with more
     caution.

.. raw:: html

    <hr>

.. _gallery_plot_interval_width: 

-------------------
Interval Width
-------------------

Visualizes the magnitude of the prediction interval width (Qup - Qlow)
for each sample at a **single time point**. Radius directly represents
the width. Color can represent width or an optional third variable
(`z_col`), here showing the Q50 prediction.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(77)
   n_points = 150
   df = pd.DataFrame({'location': range(n_points)})
   df['elevation'] = np.linspace(100, 500, n_points) # Example feature
   df['q10_val'] = np.random.rand(n_points) * 20
   # Width depends on elevation in this synthetic example
   width = 5 + (df['elevation'] / 100) * np.random.uniform(0.5, 2, n_points)
   df['q90_val'] = df['q10_val'] + width
   df['q50_val'] = df['q10_val'] + width / 2 # Use as z_col

   # --- Plotting ---
   kd.plot_interval_width(
       df=df,
       q_cols=['q10_val', 'q90_val'],
       z_col='q50_val', # Color points by Q50 value
       title='Gallery: Interval Width (Colored by Q50)',
       cmap='plasma',
       cbar=True,
       s=30,
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_interval_width_z.png"
   )
   plt.close()

.. image:: ../images/gallery_interval_width_z.png
   :alt: Interval Width Plot Example
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot shows the **magnitude of predicted uncertainty**,
   represented by the interval width (e.g., Q90 - Q10), for each
   sample at a specific time point or forecast horizon.

   **Key Features:**

   * **Radius (r):** Directly proportional to the **interval width**.
     Larger radius means greater predicted uncertainty for that sample.
   * **Angle (θ):** Represents the sample index or location, arranged
     circularly.
   * **Color:** Represents the value of the column specified by the
     ``z_col`` parameter (here, the Q50 median prediction). If
     ``z_col`` is not provided, color defaults to representing the
     interval width (radius).

   **🔍 Key Insights from this Example:**

   * We can visually identify samples with the widest (points furthest
     from center) and narrowest (points closest to center) prediction
     intervals.
   * The `plasma` colormap colors points by their Q50 value (yellow =
     high Q50, purple = low Q50). By combining radius and color, we
     can assess if higher uncertainty (larger radius) tends to occur
     for samples with higher or lower median predictions (color). In
     this synthetic example, width was linked to 'elevation', which
     might also correlate with Q50, potentially revealing a pattern.

   **💡 When to Use This Plot:**

   * **Visualize Uncertainty Magnitude:** Get a direct overview of how
     much uncertainty the model predicts for each sample.
   * **Identify High/Low Uncertainty Samples:** Quickly spot the most
     and least certain predictions.
   * **Explore Correlations:** Use the ``z_col`` parameter to investigate
     if uncertainty width correlates with other factors like the
     magnitude of the prediction itself (Q50), actual values, or
     input features.
   * **Assess Spatial Patterns:** If the angle represented spatial
     location, this plot could reveal geographical areas of high/low
     predicted uncertainty.

.. raw:: html

    <hr>


.. _gallery_plot_model_drift: 

----------------
Model Drift
----------------

Shows how *average* uncertainty (mean interval width) evolves
across different forecast horizons using a polar bar chart. Helps
diagnose model degradation over lead time.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(0)
   years = [2023, 2024, 2025, 2026, 2027]
   n_samples = 50
   df = pd.DataFrame()
   q10_cols, q90_cols = [], []
   for i, year in enumerate(years):
       ql, qu = f'val_{year}_q10', f'val_{year}_q90'
       q10_cols.append(ql); q90_cols.append(qu)
       q10 = np.random.rand(n_samples)*5 + i*0.5 # Width tends to increase
       q90 = q10 + np.random.rand(n_samples)*2 + 1 + i*0.8
       df[ql]=q10; df[qu]=q90

   # --- Plotting ---
   kd.plot_model_drift(
       df=df,
       q10_cols=q10_cols,
       q90_cols=q90_cols,
       horizons=years, # Label bars with years
       acov='quarter_circle', # Use 90 degree span
       title='Gallery: Model Drift Across Horizons',
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_model_drift.png"
   )
   plt.close()

.. image:: ../images/gallery_model_drift.png
   :alt: Model Drift Plot Example
   :align: center
   :width: 70%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This **Model Drift** plot uses a polar bar chart to visualize
   how the **average uncertainty** (mean interval width, Q90-Q10)
   evolves across different **forecast horizons** (years in this
   case, arranged angularly).

   **Analysis and Interpretation:**

   * **Radius (Avg. Uncertainty Width):** The length of each bar
     (its radius) directly represents the average width of the
     prediction intervals for that specific horizon. Longer bars mean
     wider average intervals and thus higher average uncertainty for
     that forecast lead time.
   * **Angle (Horizon):** Each bar corresponds to a successive
     forecast horizon (e.g., 2023, 2024,...), arranged around the
     circle.
   * **Color Gradient:** The color often transitions (e.g., cool to
     warm colors via the default `coolwarm` cmap) along the angular
     axis, visually reinforcing the progression through forecast
     horizons.

   **🔍 Key Insights from this Example:**

   * The bars **increase in length** as we move from earlier years
     (e.g., 2023) to later years (e.g., 2027) along the angular
     axis. This clearly indicates **model drift**: the average
     uncertainty grows as the forecast horizon extends further into
     the future.
   * The **color transition** from blue/green towards red mirrors
     this increase in uncertainty over time.
   * This pattern is typical in forecasting, reflecting the
     increasing difficulty and accumulated error when predicting
     further ahead. The plot helps quantify this degradation rate.

   **💡 When to Use This Plot:**

   * **Assess Uncertainty Evolution:** Evaluate if and how quickly
     average forecast uncertainty increases with lead time.
   * **Monitor Model Degradation:** Identify horizons where the
     uncertainty becomes unacceptably large, indicating the limits
     of the model's reliable forecast range.
   * **Inform Retraining/Updates:** Significant drift can signal the
     need to retrain the model more frequently or incorporate
     time-dependent features.
   * **Communicate Forecast Reliability:** Show stakeholders how
     confidence in forecasts typically decreases for longer lead times.

.. raw:: html

    <hr>

.. _gallery_plot_temporal_uncertainty: 

-------------------------
Temporal Uncertainty
-------------------------

A general polar scatter plot for visualizing multiple data series.
Often used to show different quantiles (e.g., Q10, Q50, Q90) for a
*single* time step to illustrate the uncertainty spread across
samples.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(99)
   n_points = 80
   df = pd.DataFrame({'id': range(n_points)})
   base = 10 + 5*np.sin(np.linspace(0, 2*np.pi, n_points))
   df['val_q10'] = base - np.random.rand(n_points)*2 - 1
   df['val_q50'] = base + np.random.randn(n_points)*0.5
   df['val_q90'] = base + np.random.rand(n_points)*2 + 1
   # Ensure order for clarity in plot
   df['val_q50'] = np.maximum(df['val_q10'] + 0.1, df['val_q50'])
   df['val_q90'] = np.maximum(df['val_q50'] + 0.1, df['val_q90'])


   # --- Plotting ---
   kd.plot_temporal_uncertainty(
       df=df,
       q_cols=['val_q10', 'val_q50', 'val_q90'],
       names=['Q10', 'Q50', 'Q90'],
       title='Gallery: Uncertainty Spread (Q10, Q50, Q90)',
       normalize=False, # Show raw values
       cmap='coolwarm', # Use diverging map for bounds
       s=20,
       mask_angle=True,
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_temporal_uncertainty_quantiles.png"
   )
   plt.close()

.. image:: ../images/gallery_temporal_uncertainty_quantiles.png
   :alt: Temporal Uncertainty Plot Example (Quantiles)
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot uses a **polar scatter** format to visualize the
   **spread of uncertainty** at a single time point by plotting
   multiple related series, typically different **quantile
   predictions** (like Q10, Q50, Q90 shown here).

   **Analysis and Interpretation:**

   * **Angle (θ):** Each angular position represents a unique sample
     or location from the dataset (ordered by index here).
   * **Radius (r):** The distance from the center represents the
     **actual predicted value** for a specific quantile at that sample
     (since ``normalize=False`` was used).
   * **Color:** Each quantile series (Q10, Q50, Q90) is assigned a
     distinct color (using the `coolwarm` cmap here, blue for Q10,
     red for Q90), allowing visual differentiation.

   **🔍 Key Insights from this Example:**

   * The **radial distance between the blue (Q10) and red (Q90) points**
     at any given angle visually represents the **prediction interval
     width** (uncertainty magnitude) for that specific sample.
   * We can see how this **spread varies** around the circle. Some
     samples (angles) have a larger gap between blue and red points
     (higher uncertainty), while others have a smaller gap (lower
     uncertainty).
   * The grey points (Q50, median) trace the central tendency, lying
     between the Q10 and Q90 bounds.
   * The overall pattern follows the sinusoidal base signal used in
     the data generation.

   **💡 When to Use This Plot:**

   * **Visualize Interval Spread:** Show the range between lower and
     upper quantile bounds for each sample simultaneously at a specific
     time/horizon.
   * **Compare Multiple Series:** Plot predictions from different
     models side-by-side against the same angular axis.
   * **Identify Uncertainty Patterns:** See if uncertainty (spread
     between quantiles) correlates with sample index or location
     (angle) or with the magnitude of the prediction (radius/color).
   * **Check Quantile Ordering:** Visually verify that Q10 <= Q50 <= Q90
     holds for most samples.

.. raw:: html

    <hr>


.. _gallery_plot_uncertainty_drift:

--------------------
Uncertainty Drift
--------------------

Visualizes how the interval width pattern evolves across multiple time
steps using concentric rings. Each ring represents a time step,
showing the relative uncertainty width at each angle (location).

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(55)
   n_points = 90; n_years = 4; years = range(2020, 2020 + n_years)
   df = pd.DataFrame({'id': range(n_points)})
   qlow_cols, qup_cols = [], []
   for i, year in enumerate(years):
       ql, qu = f'value_{year}_q10', f'value_{year}_q90'
       qlow_cols.append(ql); qup_cols.append(qu)
       base_low = np.random.rand(n_points)*3 + i*0.1
       width = (np.random.rand(n_points)+0.5)*(1.5+i*0.3 + np.cos(
           np.linspace(0, 2*np.pi, n_points)))
       df[ql] = base_low; df[qu] = base_low + width
       df[qu] = np.maximum(df[qu], df[ql]) # Ensure non-negative width

   # --- Plotting ---
   kd.plot_uncertainty_drift(
       df=df,
       qlow_cols=qlow_cols,
       qup_cols=qup_cols,
       dt_labels=[str(y) for y in years],
       title='Gallery: Uncertainty Drift (Rings)',
       cmap='magma',
       base_radius=0.1, band_height=0.1,
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_uncertainty_drift_rings.png"
   )
   plt.close()

.. image:: ../images/gallery_uncertainty_drift_rings.png
   :alt: Uncertainty Drift Rings Plot Example
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot displays how the **prediction interval width pattern**
   (Q90-Q10) changes over multiple **time steps** (e.g., years)
   using **concentric rings**. Each ring represents a specific time
   step, ordered radially outwards.

   **Analysis and Interpretation:**

   * **Rings & Time:** Each colored ring corresponds to a time step
     (e.g., 2020 near center, 2023 further out). The legend links
     colors to time steps.
   * **Radius & Uncertainty:** The radius of a point on a specific
     ring represents the **relative interval width** for that sample
     at that time. The radius is calculated as a base offset for the
     ring plus a component scaled by the *globally normalized* width.
     Therefore, bulges or larger radii on a ring indicate higher
     relative uncertainty for those samples at that time.
   * **Comparing Rings:** Observe how the overall size and shape of
     the rings change from inner (earlier) to outer (later).
     Increasing average radius or increased "bumpiness" in outer
     rings suggests **uncertainty drift** - uncertainty grows or becomes
     more variable over time.
   * **Angular Patterns:** Consistent high/low radii at specific angles
     across multiple rings pinpoint locations/samples with persistently
     high/low relative uncertainty.

   **🔍 Key Insights from this Example:**

   * The concentric rings clearly separate the uncertainty patterns for
     different years (2020-2023).
   * Comparing the rings reveals how the spatial distribution and
     magnitude of relative uncertainty change over the forecast horizon.
     For instance, one might observe uncertainty increasing overall
     (outer rings generally larger) or becoming more pronounced in
     certain angular sectors (locations).
   * Potential cyclic patterns in width along the angular axis might
     suggest seasonal or location-based effects on uncertainty.

   **💡 When to Use This Plot:**

   * **Visualize Uncertainty Evolution:** Track how the *entire
     pattern* of uncertainty changes across multiple forecast periods.
   * **Identify Temporal Drift Patterns:** See if uncertainty increases
     uniformly, or only in specific regions/samples over time.
   * **Compare Uncertainty Maps:** Overlay and compare the "uncertainty
     map" (relative interval width vs. sample index/location) from
     different time steps in a single view.
   * **Assess Long-Term Reliability:** Evaluate if the model's
     uncertainty estimates remain stable or degrade significantly as
     forecasts extend further out.

.. raw:: html

    <hr>

.. _gallery_plot_prediction_velocity: 

----------------------
Prediction Velocity
----------------------

Visualizes the average rate of change (velocity) of the median (Q50)
prediction over consecutive time periods for each location. Radius
indicates velocity magnitude; color can indicate velocity or average
Q50.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(123)
   n_points = 100; years = range(2020, 2024)
   df = pd.DataFrame({'location_id': range(n_points)})
   q50_cols = []
   base_val = np.random.rand(n_points)*10
   trend = np.linspace(0, 5, n_points)
   for i, year in enumerate(years):
       q50_col = f'val_{year}_q50'
       q50_cols.append(q50_col)
       noise = np.random.randn(n_points)*0.5
       df[q50_col] = base_val + trend*i + noise

   # --- Plotting ---
   kd.plot_velocity(
       df=df,
       q50_cols=q50_cols,
       title='Gallery: Prediction Velocity (Colored by Avg Q50)',
       use_abs_color=True, # Color by magnitude of Q50
       normalize=True,     # Normalize radius (velocity)
       cmap='cividis',
       cbar=True,
       s=25,
       # Save the plot (adjust path relative to docs/source/)
       savefig="gallery/images/gallery_velocity_abs_color.png"
   )
   plt.close()

.. image:: ../images/gallery_velocity_abs_color.png
   :alt: Prediction Velocity Plot Example
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot visualizes the **average rate of change (velocity)**
   of the median (Q50) prediction across consecutive time periods.
   Each point represents a sample/location.

   **Analysis and Interpretation:**

   * **Radius (Velocity Magnitude):** The distance from the center
     indicates the **average speed** at which the Q50 prediction
     is changing over time for that sample. Larger radii mean
     faster average change (positive or negative); smaller radii
     mean more stable Q50 predictions. (Note: If ``normalize=False``,
     radius shows raw velocity).
   * **Angle (θ):** Represents the sample index/location, arranged
     circularly.
   * **Color (Context):** The color provides context.
   
     * If ``use_abs_color=True`` (default, as in this example):
       Color maps to the **average absolute Q50 value** across
       periods. This helps see if rapid changes (high radius)
       occur in high-value (e.g., yellow in `cividis`) or
       low-value (e.g., purple) regions.
     * If ``use_abs_color=False``: Color maps directly to the
       **velocity value**. Using a diverging colormap (like
       'coolwarm') distinguishes between positive velocity
       (increasing trend) and negative velocity (decreasing trend).

   **🔍 Key Insights from this Example:**

   * Points far from the center highlight locations where the median
     prediction is changing most rapidly on average between the years
     provided.
   * The `cividis` colormap shows the average magnitude of the Q50
     prediction at each location. We can observe if the high-velocity
     points (large radius) coincide with high-magnitude (yellow) or
     low-magnitude (purple) predictions.
   * Clustering of points with similar radius/color might indicate
     spatial patterns in the phenomenon's dynamics.

   **💡 When to Use This Plot:**

   * **Identify Dynamic Hotspots:** Find samples/locations where the
     central forecast trend is changing most quickly.
   * **Assess Prediction Stability:** Locate areas where predictions
     are relatively stable (low velocity) vs. dynamic (high velocity).
   * **Contextualize Change Rate:** Use ``use_abs_color=True`` to see
     if rapid changes are happening in already critical high/low value
     areas. Use ``use_abs_color=False`` with a diverging map to see
     the direction (increase/decrease) of the average change.
   * **Analyze Temporal Trends Spatially:** Understand the spatial
     distribution of the rate of change across different locations.

.. raw:: html

    <hr>


.. _gallery_plot_radial_density_ring:

--------------------------
Radial Density Ring
--------------------------

Visualizes the **1D probability distribution** of a metric using
Kernel Density Estimation (KDE). This plot is a unique way to
inspect the shape, peaks, and spread of a distribution, such as
prediction interval widths or forecast errors.

The key features are:

- **Radius (`r`)**: Represents the value of the metric.
- **Color**: Represents the probability density at that radius.
  Brighter/more intense colors indicate more common values.

.. _gallery_plot_density_ring_width:

Distribution of Interval Width (``kind='width'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows the distribution of the prediction interval
width (Q90 - Q10), a key measure of model uncertainty.

.. code-block:: python
    :linenos:

    import kdiagram as kd
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Data Generation (shared for all examples) ---
    np.random.seed(42)
    n_samples = 500
    df_test = pd.DataFrame({
        'q10': np.random.normal(10, 2, n_samples),
        'q90': np.random.normal(30, 3, n_samples),
        'value_2022': np.random.gamma(3, 5, n_samples),
        'value_2023': np.random.gamma(4, 5, n_samples),
        'error_metric': np.random.randn(n_samples) * 5,
    })
    # Ensure q90 is always greater than q10
    df_test['q90'] = df_test[['q10', 'q90']].max(axis=1) + \
        np.random.rand(n_samples) * 2

    # --- Plotting ---
    kd.plot_radial_density_ring(
        df=df_test,
        kind="width",
        target_cols=["q10", "q90"],
        title="Distribution of Prediction Interval Width",
        cmap="Blues",
        show_yticklabels=True,
        r_label="q90 − q10",
        savefig="gallery/images/gallery_plot_density_ring_prediction_interval.png"
    )
    plt.close()

.. image:: ../images/gallery_plot_density_ring_prediction_interval.png
    :alt: Radial Density Ring for Interval Width
    :align: center
    :width: 70%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    This plot reveals the distribution of the model's uncertainty estimates.

    **Key Insights:**

    * **Most Likely Uncertainty**: The brightest ring indicates the
      most common interval width. This represents the model's
      typical uncertainty range.
    * **Consistency**: A narrow, bright ring suggests the model
      produces highly consistent uncertainty estimates. A wide,
      diffuse ring indicates high variability in uncertainty.
    * **Multi-modality**: Multiple distinct bright rings would
      suggest the model operates in different uncertainty modes
      for different subsets of the data.

    **🔍 In this Example:**

    * The brightest part of the ring is centered around a radius
      of **20**. This means for most samples, the prediction
      interval (Q90 - Q10) has a width of about 20 units.
    * The distribution is relatively symmetric and bell-shaped,
      fading out for very narrow (<10) or very wide (>30)
      intervals.

    **💡 When to Use:**

    Use this plot to answer questions like:

    * "What is the typical range of my model's uncertainty?"
    * "Does my model produce consistent uncertainty estimates, or
      do they vary wildly?"
    * "Are there multiple, distinct levels of uncertainty in my
      predictions?"


.. _gallery_plot_density_ring_velocity:

Distribution of Change (``kind='velocity'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example visualizes the distribution of change between two
time points (e.g., year-over-year velocity).

.. code-block:: python
    :linenos:

    # Assumes df_test is already created from the previous block

    kd.plot_radial_density_ring(
        df=df_test,
        kind="velocity",
        target_cols=["value_2022", "value_2023"],
        title="Distribution of Value Change (2022 to 2023)",
        cmap="Reds",
        show_yticklabels=True,
        r_label="value_2023 − value_2022",
        savefig="gallery/images/gallery_plot_density_ring_distr_value.png"
    )
    plt.close()

.. image:: ../images/gallery_plot_density_ring_distr_value.png
    :alt: Radial Density Ring for Velocity
    :align: center
    :width: 70%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    This plot shows the distribution of the rate of change between
    two sets of values.

    **Key Insights:**

    * **Central Tendency**: The brightest ring shows the most
      common change or velocity. If it's centered at zero, it
      suggests stability; otherwise, it indicates a consistent
      positive or negative trend.
    * **Magnitude of Change**: The spread of the ring shows the
      variability in the rate of change. A tight ring means
      the change is consistent across all samples.

    **🔍 In this Example:**

    * The distribution is centered around a radius of **+5**. This
      indicates that the most common change from 2022 to 2023
      was a positive increase of 5 units.
    * The distribution has a longer tail towards higher values,
      suggesting that while a +5 change is most typical, some
      samples experienced a much larger increase.

    **💡 When to Use:**

    * To analyze the distribution of year-over-year changes in
      a forecast.
    * To study the distribution of differences between two model
      versions.
    * To visualize the distribution of treatment effects
      (post-treatment vs. pre-treatment values).



.. _gallery_plot_density_ring_direct:

Distribution of a Direct Metric (``kind='direct'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is the most general use case, visualizing the distribution
of any pre-calculated, single-column metric.

.. code-block:: python
    :linenos:

    # Assumes df_test is already created from the first block

    kd.plot_radial_density_ring(
        df=df_test,
        kind="direct",
        target_cols="error_metric",
        title="Distribution of a Pre-calculated Error Metric",
        cmap="Greens",
        show_yticklabels=True,
        r_label="error_metric",
        savefig="gallery/images/gallery_plot_density_ring_error_metric.png"
    )
    plt.close()

.. image:: ../images/gallery_plot_density_ring_error_metric.png
    :alt: Radial Density Ring for a Direct Metric
    :align: center
    :width: 70%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    This plot is a general-purpose tool for inspecting the shape
    of any continuous variable.

    **Key Insights:**

    * **Distribution Shape**: Immediately reveals if a
      distribution is symmetric, skewed, normal, or bimodal.
    * **Central Point**: The brightest ring highlights the mode
      (peak) of the distribution.
    * **Spread**: The width of the colored area indicates the
      variance or standard deviation of the metric.

    **🔍 In this Example:**

    * The synthetic `error_metric` was generated from a standard
      normal distribution, and the plot reflects this perfectly.
    * The brightest ring is at a radius of **0**, indicating an
      unbiased error distribution centered at zero.
    * The density is symmetric around zero and fades smoothly,
      as expected for a Gaussian (bell-curve) distribution.

    **💡 When to Use:**

    * To visualize the distribution of model residuals or errors.
    * To inspect the distribution of a feature before modeling.
    * To present the distribution of any summary statistic in a
      visually engaging format.

.. raw:: html

    <hr>

.. _gallery_plot_polar_heatmap:

--------------------------
Polar Heatmap
--------------------------

Visualizes the 2D density of data points on a polar grid, showing
the concentration of a radial variable against a cyclical or ordered
angular variable.

.. code-block:: python
    :linenos:

    import kdiagram.plot.uncertainty as kdu
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Data Generation ---

    np.random.seed(42)
    n_points = 5000

    # Simulate hour of day with more events in the afternoon

    hour = np.concatenate([
        np.random.normal(15, 2, int(n_points * 0.7)),
        np.random.normal(5, 2, int(n_points * 0.3))
    ]) % 24

    # Simulate rainfall, correlated with afternoon hours

    rainfall = np.random.gamma(2, 5, n_points) +  
    (hour > 12) * np.random.gamma(3, 5, n_points)

    df_weather = pd.DataFrame({'hour': hour, 'rainfall_mm': rainfall})

    # --- Plotting ---

    kdu.plot_polar_heatmap(
        df=df_weather,
        r_col='rainfall_mm',
        theta_col='hour',
        theta_period=24,
        r_bins=25,
        theta_bins=24,
        cmap='plasma',
        title='Rainfall Intensity vs. Hour of Day',
        cbar_label='Event Count',
        savefig="gallery/images/gallery_plot_polar_heatmap.png"
    )
    plt.close()

.. image:: ../images/gallery_plot_polar_heatmap.png
    :alt: Example of a Polar Heatmap
    :align: center
    :width: 75%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    The **Polar Heatmap** is a powerful tool for finding patterns
    between a cyclical feature (like time) and a magnitude.

    **Key Features:**

    * **Angle (θ):** Represents the cyclical variable (e.g., hour of the day).
    * **Radius (r):** Represents the magnitude variable (e.g., rainfall amount).
    * **Color:** Shows the density or count of data points. Bright, hot
      colors indicate a high concentration of events in that specific
      angle-radius bin.

    **🔍 In this Example:**

    * The brightest colors (yellow) are concentrated between roughly
      180° and 270° (corresponding to the afternoon hours, 12:00 to 18:00)
      and at a moderate radius (rainfall intensity).
    * This immediately reveals the pattern in the simulated data: heavy
      rainfall events are most frequent in the afternoon. The rest of the
      plot is dark, indicating few events at other times or intensities.

    **💡 When to Use:**

    * To find correlations between a cyclical feature (time, season) and
      an event's magnitude or error.
    * To identify "hot spots" in your data where specific conditions
      (e.g., time of day and error size) frequently co-occur.

.. raw:: html

    <hr>

.. _gallery_plot_polar_quiver:

--------------------------
Polar Quiver Plot
--------------------------

Visualizes vector data (magnitude and direction) at specific points
on a polar grid. It's ideal for showing changes, revisions, or error
vectors.

.. code-block:: python
    :linenos:

    import kdiagram.plot.uncertainty as kdu
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Data Generation ---

    np.random.seed(0)
    n_points = 50
    locations = np.linspace(0, 360, n_points, endpoint=False)
    initial_forecast = 10 + 5 * np.sin(np.deg2rad(locations) * 3)

    # Simulate forecast revisions

    radial_change = np.random.normal(0, 1.5, n_points)
    tangential_change = np.random.normal(0, 0.1, n_points)

    df_forecasts = pd.DataFrame({
    'location_angle': locations,
    'initial_value': initial_forecast,
    'update_radial': radial_change,
    'update_tangential': tangential_change,
    })

    # --- Plotting ---

    kdu.plot_polar_quiver(
        df=df_forecasts,
        r_col='initial_value',
        theta_col='location_angle',
        u_col='update_radial',
        v_col='update_tangential',
        theta_period=360,
        title='Forecast Revisions for Spatial Locations',
        cmap='coolwarm',
        scale=25,
        savefig="gallery/images/gallery_uncertainty_vector_revisions.png"
    )
    plt.close()

.. image:: ../images/gallery_uncertainty_vector_revisions.png
    :alt: Example of a Polar Quiver Plot
    :align: center
    :width: 75%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    The **Polar Quiver Plot** shows the direction and magnitude of
    change at different points in a system.

    **Key Features:**

    * **Arrow Position:** The base of each arrow is located at a
      point (:math:`(r, \theta)`) on the polar grid.
    * **Arrow Direction & Length:** The arrow points in the direction
      of the vector, and its length represents the vector's magnitude.
      Color is also often used to represent magnitude.

    **🔍 In this Example:**

    * The base of each arrow represents an initial forecast value for a
      specific location (angle).
    * The arrow itself shows the revision to that forecast. An arrow
      pointing outward indicates the forecast was revised upward. An arrow
      pointing inward indicates a downward revision.
    * The color and length of the arrows show the magnitude of the
      revision. The long, dark red arrow near 180° represents the largest
      single forecast update in the dataset.

    **💡 When to Use:**

    * To visualize forecast updates and assess model stability.
    * To plot error vectors (e.g., where the vector shows the direction
      and magnitude of error from the true value).
    * To visualize flow fields or other vector data in a polar context.
