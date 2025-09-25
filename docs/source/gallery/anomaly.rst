.. _gallery_anomaly:

=============================
Anomaly Diagnostics Gallery
=============================

This gallery page showcases specialized plots from the
:mod:`kdiagram.plot.anomaly` module, designed for the
in-depth diagnosis of prediction interval failures. These
visualizations move beyond simple coverage scores to assess the
key characteristics of forecast anomalies: their **magnitude** (how
severe is the failure?), their **type** (was it an over- or
under-prediction?), and their **clustering** (are the failures
systematic?).

The plots provide intuitive diagnostics for visualizing anomaly
severity through polar scatter plots, stylized "fiery ring"
profiles, information-rich glyphs, and layered Cartesian
profiles, allowing for a deeper understanding of a model's
failure modes.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots.

.. _gallery_plot_anomaly_severity:

----------------------------------------------------
Polar Anomaly Severity Plot (Scatter Version)
----------------------------------------------------

The :func:`~kdiagram.plot.anomaly.plot_anomaly_severity` function is
a primary diagnostic tool for moving beyond simple coverage metrics.
It focuses exclusively on **forecast failures** (anomalies) and
visualizes four key dimensions of these failures simultaneously: their
location, magnitude, type, and clustering.

First, let's break down the components of this diagnostic plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** The angular axis represents the **sample index**,
     showing *where* in the dataset the failure occurred. A cluster
     of points in one angular region indicates a systematic, localized
     problem.
   * **Radius (r):** The radius of each point shows the **Anomaly
     Magnitude**—the absolute distance from the true value to the
     nearest violated interval bound. Points far from the center are
     severe failures.
   * **Color:** The color of a point represents its **Local Anomaly Density**.
     "Hotter" colors (e.g., bright yellow) indicate that
     the anomaly is part of a dense cluster of other failures—a "hotspot."
   * **Marker Shape:** The marker's shape distinguishes the **Type** of
     anomaly. By default, circles (`o`) are over-predictions (risk
     underestimated), while `X`'s are under-predictions.

With this framework, let's explore how to use this plot to diagnose
different, common types of forecast failures.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: The Ideal Case - Minor, Scattered Anomalies**

The first use case is to identify a well-behaved model where the few
prediction interval failures are small and randomly distributed. This
serves as the benchmark for a low-risk forecast profile.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   np.random.seed(42)
   n_samples = 400

   # 1) Start from a stable signal and define a fairly wide PI (±8)
   base   = np.random.normal(loc=50, scale=10, size=n_samples)
   q_low  = base - 8
   q_up   = base + 8

   # 2) Actuals mostly inside the band + a small random noise
   actual = base + np.random.normal(0, 2, size=n_samples)

   # 3) Inject a few *small* interval failures on random indices
   k = 24  # ~6% scattered anomalies
   over_idx  = np.random.choice(n_samples, size=k//2, replace=False)
   rest      = np.setdiff1d(np.arange(n_samples), over_idx)
   under_idx = np.random.choice(rest, size=k - k//2, replace=False)

   # Just outside the band by 0.2–2.0 units (minor misses)
   actual[over_idx]  = q_up[over_idx]  + np.random.uniform(0.2, 2.0, size=over_idx.size)
   actual[under_idx] = q_low[under_idx] - np.random.uniform(0.2, 2.0, size=under_idx.size)

   df = pd.DataFrame({"actual": actual, "q_low": q_low, "q_up": q_up})

   # 4) Plot
   ax = kd.plot_anomaly_severity(
       df,
       actual_col="actual",
       q_low_col="q_low",
       q_up_col="q_up",
       window_size=21,
       title="Use Case 1: A Well-Behaved Model (Minor, Scattered Anomalies)",
       # savefig="gallery/images/gallery_anomaly_severity_good.png",
   )


.. figure:: ../images/anomaly/gallery_anomaly_severity_good.png
   :align: center
   :width: 70%
   :alt: A polar scatter plot with a few, cool-colored points near the center.

   A sparse plot showing a few scattered anomalies with low
   magnitudes (close to the center) and cool colors (low density).

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot is the signature of a **well-behaved model**. The few
   anomalies that exist are scattered randomly around the circle,
   indicating no systematic pattern to the failures. Furthermore, they
   are all close to the center (low magnitude) and are colored dark
   purple (low local density), meaning they are minor, isolated
   events. This is the ideal result for this diagnostic.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Diagnosing a "Hotspot" of Severe Failures**

A more dangerous failure mode occurs when a model is systematically
wrong in a specific region of the data. The CAS score and this plot
are designed to detect exactly this kind of "hotspot."

Let's simulate a model that performs well overall but has a cluster of
large, severe failures in one particular segment of the data.

.. code-block:: python
   :linenos:

   # --- 1. Simulate data with a failure hotspot ---
   np.random.seed(0)
   n_samples = 400
   y_true = 100 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
   y_qlow = y_true - 10
   y_qup = y_true + 10
   
   # Introduce a cluster of large over-prediction anomalies
   cluster_slice = slice(100, 140)
   y_true[cluster_slice] = (
       y_qup[cluster_slice] 
       + np.random.uniform(10, 25, 40)
   )
   df = pd.DataFrame({
       "actual": y_true, "q_low": y_qlow, "q_up": y_qup
   })
   
   # --- 2. Plotting ---
   kd.plot_anomaly_severity(
       df,
       actual_col="actual",
       q_low_col="q_low",
       q_up_col="q_up",
       window_size=31,
       acov="half_circle",
       title="Use Case 2: A Model with a Failure Hotspot",
       savefig="gallery/images/gallery_anomaly_severity_hotspot.png",
   )
   plt.close()

.. figure:: ../images/anomaly/gallery_anomaly_severity_hotspot.png
   :align: center
   :width: 70%
   :alt: A polar scatter plot showing a dense cluster of bright, high-magnitude points.

   The plot reveals a clear "hotspot" where a cluster of severe
   anomalies (high radius, bright color) has occurred.

.. topic:: 🧠 Interpretation
   :class: hint

   This plot instantly reveals a critical, systematic failure.
   There is a dense **cluster of bright yellow points** in one
   angular region. This tells us two things: the failures are
   **clustered** (indicated by the hot color from the high local
   density), and they are **severe** (indicated by their large
   distance from the center). This is a clear "hotspot" of poor
   performance that requires immediate investigation, a finding that
   would be completely hidden by an overall coverage score.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

For a deeper understanding of the CAS score and its components,
please refer back to the main :ref:`ug_plot_anomaly_severity` or
:ref:`ug_cluster_aware_severity_score` section in the metrics
guide.

.. _gallery_plot_anomaly_profile:

--------------------------------------
Polar Anomaly Profile ("Fiery Ring")
--------------------------------------

The :func:`~kdiagram.plot.anomaly.plot_anomaly_profile` function
offers a stylized and aesthetically focused visualization of
forecast failures. It is designed to be a high-impact figure for
scientific papers, using the powerful metaphor of a "fiery ring"
to represent hotspots of clustered anomalies, with individual
failures erupting from it as "flares."

First, let's break down the components of this unique plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** The angular axis represents the **sample index**,
     showing *where* in the dataset the failure occurred.
   * **Central Ring:** This colored ring sits at a fixed radius and
     visualizes the **Local Anomaly Density**. "Hotter" colors
     (e.g., bright yellow) on the ring indicate a "hotspot" where
     failures are highly clustered.
   * **Flares:** Each individual anomaly is drawn as a line or "flare"
     extending from the central ring. The flare's properties encode
     two dimensions of the failure:
     
     * **Length:** Represents the **Anomaly Magnitude**. Longer flares
       are more severe failures.
     * **Direction:** Represents the **Type** of anomaly. Outward
       flares are over-predictions (risk underestimated), while
       inward flares are under-predictions.

With this framework, let's explore how to use this plot to diagnose a
model with complex, mixed failure modes.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: Diagnosing Mixed and Clustered Failure Modes**

A use of this plot is to identify if a model suffers from
different systematic biases in different parts of the data. A model
might underestimate risk in one scenario (e.g., high-demand
periods) and overestimate it in another.

Let's simulate a forecast that has two distinct, separate clusters of
failures: one of over-predictions and one of under-predictions.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   # --- 1. Simulate data with two distinct failure clusters ---
   np.random.seed(30)
   n_samples = 500
   y_true = np.sin(np.linspace(0, 6 * np.pi, n_samples)) * 10 + 20
   y_qlow = y_true - 5
   y_qup = y_true + 5
   
   # Cluster 1: Over-predictions (true value > upper bound)
   y_true[100:130] += np.random.uniform(6, 12, 30)
   # Cluster 2: Under-predictions (true value < lower bound)
   y_true[300:330] -= np.random.uniform(6, 12, 30)
   
   df = pd.DataFrame({
       "actual": y_true, "q10": y_qlow, "q90": y_qup
   })
   
   # --- 2. Plotting ---
   kd.plot_anomaly_profile(
       df,
       actual_col="actual",
       q_low_col="q10",
       q_up_col="q90",
       window_size=31,
       title="Use Case: Profile with Mixed Failure Types",
       savefig="gallery/images/gallery_anomaly_profile.png",
   )
   plt.close()

.. figure:: ../images/anomaly/gallery_anomaly_profile.png
   :align: center
   :width: 70%
   :alt: A polar anomaly profile with a fiery ring and flares.

   A "fiery ring" plot where the ring's color shows anomaly
   hotspots, and flares show the magnitude and type of failures.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot instantly diagnoses a complex failure pattern. The
   **central ring** is brightly colored in two distinct angular
   regions, confirming that the forecast failures are not random
   but are **systematically clustered** in two separate "hotspots."

   The **flares** reveal the nature of these failures. The hotspot
   on the right consists entirely of **outward red flares**,
   indicating a systematic **underestimation of risk** (over-
   prediction) in this part of the dataset. Conversely, the
   hotspot on the left consists entirely of **inward blue flares**,
   revealing a systematic **overestimation of risk** (under-
   prediction). This powerful visualization uncovers two separate,
   opposing biases in the model that would be impossible to see
   with a single aggregate score.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the CAS score and its components,
please refer back to the main :ref:`ug_plot_anomaly_profile` or 
:ref:`ug_cluster_aware_severity_score` section in the metrics
guide.

.. _gallery_plot_glyphs:

--------------------------
Polar Anomaly Glyph Plot
--------------------------

The :func:`~kdiagram.plot.anomaly.plot_glyphs` function creates a
highly informative and scientifically rigorous diagnostic plot.
Instead of simple dots, each forecast failure (anomaly) is
represented by a **glyph** (a custom symbol). The glyph's
properties—its location, shape, and color—encode multiple
characteristics of the anomaly simultaneously.

First, let's break down the components of this advanced plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** The angular position is determined by the
     `sort_by` parameter, providing a meaningful order to the
     data points (e.g., by time or a spatial coordinate).
   * **Radius (r):** The radius is determined by the metric
     specified in the `radius` parameter (e.g., 'magnitude',
     'severity'), normalized to [0, 1] for visual consistency.
   * **Glyph Color:** The color is determined by the metric
     specified in the `color_by` parameter. By default, this is
     the 'local_density', so "hotter" colors indicate a
     "hotspot" of clustered failures.
   * **Glyph Shape:** The shape of the marker provides an
     intuitive metaphor for the **Type** of anomaly:
     
     * `▲` (up-triangle): **Over-prediction**, where the true
       value was higher than the upper bound (risk underestimated).
     * `▼` (down-triangle): **Under-prediction**, where the true
       value was lower than the lower bound (risk overestimated).

This plot provides a dense, multi-dimensional summary of forecast
failures, making it ideal for detailed analysis and publication.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: Diagnosing a Failure Hotspot with High Granularity**

This plot is most powerful when you need to understand not just
that a cluster of failures occurred, but also the specific nature
of each failure within that cluster.

Let's simulate a model that fails during a specific period (e.g.,
a summer heatwave that was not well-predicted), and use the glyph
plot to dissect the resulting hotspot.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   # --- 1. Simulate temperature forecast data with a hotspot ---
   np.random.seed(0)
   n_samples = 365
   time = pd.to_datetime(pd.date_range(
       "2024-01-01", periods=n_samples)
   )
   y_true = 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
   y_qlow = y_true - 2
   y_qup = y_true + 2
   # Add a summer heatwave the model misses (over-prediction)
   y_true[180:210] += np.random.uniform(2.5, 5, 30)
   
   df = pd.DataFrame({
       "time": time, "actual_temp": y_true,
       "q10_temp": y_qlow, "q90_temp": y_qup
   })
   
   # --- 2. Plotting ---
   kd.plot_glyphs(
       df,
       actual_col="actual_temp",
       q_low_col="q10_temp",
       q_up_col="q90_temp",
       sort_by=time,
       radius="magnitude",
       color_by="local_density",
       title="Use Case: Glyph Plot of Seasonal Anomalies",
       savefig="gallery/images/gallery_anomaly_glyphs.png",
   )
   plt.close()

.. figure:: ../images/anomaly/gallery_anomaly_glyphs.png
   :align: center
   :width: 70%
   :alt: A polar glyph plot showing forecast anomalies.

   A polar glyph plot where each triangle represents a forecast
   failure. Its position, shape, and color reveal the failure's
   location, type, magnitude, and clustering.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This glyph plot immediately reveals a systematic, seasonal
   failure in the forecast model. There is a distinct **cluster of
   bright yellow, outward-pointing triangles (`▲`)** in the angular
   region corresponding to the summer months. This tells us several
   things at once:
   
   1.  The failures are **clustered** (indicated by the hot color).
   2.  They are all **over-predictions** (indicated by the `▲`
       shape), meaning the model systematically underestimated the
       summer temperatures.
   3.  The **large radius** of these glyphs shows that the
       magnitudes of these failures were significant.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

For a deeper understanding of the CAS score and its components,
please refer back to the main :ref:`ug_plot_glyphs` or 
:ref:`ug_cluster_aware_severity_score` section in the metrics
guide.

.. _gallery_plot_cas_profile:

------------------------------------
Cartesian Anomaly Profile
------------------------------------

The :func:`~kdiagram.plot.anomaly.plot_cas_profile` function
creates a **Cartesian Anomaly Profile**, a powerful non-polar
alternative for diagnosing forecast failures. It is highly
effective for sequential data, such as time series, where the
x-axis can represent time or sample index. The plot visualizes an
anomaly's location, magnitude, type, and clustering density.

First, let's break down the components of this diagnostic plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **X-axis:** The horizontal axis represents the **sample
     index**, showing *when* or *where* in the sequence the
     failure occurred.
   * **Y-axis:** The vertical axis represents the **Anomaly
     Magnitude**. The height of a point directly shows the
     severity of the failure.
   * **Color:** The color of a point represents its **Local Anomaly
     Density**. "Hotter" colors (e.g., bright yellow) indicate
     that the anomaly is part of a dense cluster of other
     failures—a "hotspot."
   * **Marker Shape:** The marker's shape distinguishes the **Type**
     of anomaly, using an intuitive metaphor:
     
       * `▲` (up-triangle): **Over-prediction** (risk underestimated).
       * `▼` (down-triangle): **Under-prediction** (risk overestimated).

This plot provides a direct, sequential view of forecast
failures, making it easy to spot trends or regime changes in model
performance over time.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: Diagnosing a Regime Change or Model Degradation**

This plot is ideal for time-ordered data where you suspect a
model's performance may not be stationary. A model that performs
well on historical data may fail when the underlying process
changes.

Let's simulate a forecast for a process that is stable for a long
period and then suddenly becomes more volatile. We want to see if
the model's prediction intervals can adapt to this new regime.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   # --- 1. Simulate a time series with a failure hotspot ---
   np.random.seed(0)
   n_samples = 400
   y_true = np.zeros(n_samples)
   y_qlow = y_true - 10
   y_qup = y_true + 10
   # Introduce a cluster of severe failures toward the end
   y_true[300:340] += np.random.uniform(12, 20, 40)
   
   df = pd.DataFrame({
       "actual": y_true, "q10": y_qlow, "q90": y_qup
   })
   
   # --- 2. Plotting ---
   kd.plot_cas_profile(
       df,
       actual_col="actual",
       q_low_col="q10",
       q_up_col="q90",
       window_size=21,
       title="Use Case: Diagnosing a Failure Hotspot Over Time",
       savefig="gallery/images/gallery_cas_profile.png",
   )
   plt.close()

.. figure:: ../images/anomaly/gallery_cas_profile.png
   :align: center
   :width: 80%
   :alt: A Cartesian anomaly profile plot.

   A Cartesian plot showing forecast failures over time, where
   the y-axis is the failure magnitude, and the color reveals
   failure hotspots.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This profile plot clearly reveals a change in the model's
   performance over time. For the first ~300 samples, there are no
   anomalies, indicating the model's prediction intervals were
   successful. However, a distinct **cluster of bright yellow,
   high-magnitude, upward-pointing triangles (`▲`)** appears
   towards the end of the series.

   This provides a critical insight: the model was reliable during
   the stable period but began to systematically fail when the
   process became more volatile. The plot pinpoints exactly *when*
   this degradation occurred and shows that the failures were both
   **severe** (high on the y-axis) and **clustered** (bright color),
   signaling a potential regime change that the model could not adapt
   to.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the CAS score and its components,
please refer back to the main :ref:`ug_plot_cas_profile` or 
:ref:`ug_cluster_aware_severity_score` section in the metrics
guide.

.. _gallery_plot_cas_layers:

------------------------------------
Layered Anomaly Profile
------------------------------------

The :func:`~kdiagram.plot.anomaly.plot_cas_layers` function
creates the most comprehensive diagnostic in the anomaly suite.
It visualizes a forecast's prediction interval, the true values,
and the calculated anomaly characteristics in a set of **layered,
stacked panels**. This is particularly effective for sequential
data, providing a clear, contextualized story of model
performance.

First, let's break down the components of this multi-layered plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Top Panel (Forecast Context):** This panel displays the
     primary forecast.
     
     * A **shaded gray area** shows the prediction interval
       (`q_low_col` to `q_up_col`).
     * A **dark line** shows the true values (`actual_col`).
     * **Anomalies** are marked with colored triangles (`▲` for
       over-predictions, `▼` for under-predictions). The color
       intensity of the marker corresponds to its **severity
       score**.

   * **Bottom Panel (Severity Breakdown):** This panel, linked by a
     shared x-axis, explains *why* the anomalies are severe.
     
     * **Vertical bars** show the per-sample **severity score**.
       Tall, hot-colored bars pinpoint the most critical
       failures.
     * A **solid black line** (`show_density=True`) traces the
       **local anomaly density**, highlighting the "hotspot"
       regions that contribute most to the severity scores.

This plot decomposes the CAS diagnostic into layers, providing a
clear, sequential view of model performance.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Diagnosing Performance During a Regime Shift**

This plot's primary strength is showing how a model's performance
and failure modes change over time or in response to specific
events. It can clearly visualize a model's breakdown during a
"regime shift," where the underlying data-generating process changes.

Let's simulate a financial forecast where a model is stable during
normal market conditions but fails catastrophically during a sudden
market shock.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   # --- 1. Simulate a time series with a failure hotspot ---
   np.random.seed(0)
   n_samples = 400
   x_axis = np.arange(n_samples)
   y_true = 20 * np.sin(x_axis * np.pi / 100)
   y_qlow = y_true - 10
   y_qup = y_true + 10
   # Introduce a cluster of severe failures (a "market shock")
   y_true[180:220] += np.random.uniform(12, 20, 40)
   
   df = pd.DataFrame({
       "x": x_axis, "actual": y_true,
       "q10": y_qlow, "q90": y_qup
   })
   
   # --- 2. Plotting ---
   axes = kd.plot_cas_layers(
       df,
       actual_col="actual",
       q_low_col="q10",
       q_up_col="q90",
       sort_by="x",
       title="Use Case 1: Model Failure During a Market Shock",
       savefig="gallery/images/gallery_cas_layers_shock.png",
   )
   plt.close()

.. figure:: ../images/anomaly/gallery_cas_layers_shock.png
   :align: center
   :width: 80%
   :alt: A layered Cartesian plot of anomalies and severity.

   A two-panel plot. The top shows a time series forecast with
   anomalies. The bottom shows the severity of those anomalies.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This layered plot provides a complete narrative of the model's
   performance. The **top panel** shows that the model's
   prediction interval (shaded area) successfully contains the
   true value (black line) during the stable periods. However,
   during the "market shock" (around index 200), the true value
   dramatically breaks out of the predicted band, and this region
   is marked by a cluster of bright, hot-colored `▲` markers.

   The **bottom panel** explains the severity of this event. The
   vertical bars show a massive spike in the **severity score**
   that is perfectly aligned with the breakout event above. The
   black line, tracing the **local density**, confirms that this
   is a true "hotspot" of clustered failures. This plot provides
   clear, visual evidence that the model is not robust to regime
   changes.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Comparing Anomaly Severity Across Categories**

While ideal for time series, the x-axis can be ordered by any
meaningful feature. This allows us to compare the severity of
anomalies across different categories.

Let's simulate product sales forecasts where a model performs
well for one category but produces severe, clustered failures for
another.

.. code-block:: python
   :linenos:

   # --- 1. Simulate data for two product categories ---
   np.random.seed(1)
   n_cat_A = 150
   n_cat_B = 150
   # Category A: Well-behaved
   y_true_A = np.random.normal(100, 10, n_cat_A)
   y_qlow_A = y_true_A - 20
   y_qup_A = y_true_A + 20
   
   # Category B: Has severe over-prediction anomalies
   y_true_B = np.random.normal(150, 15, n_cat_B)
   y_qlow_B = y_true_B - 25
   y_qup_B = y_true_B + 25
   y_true_B[50:80] += np.random.uniform(30, 50, 30)
   
   df = pd.DataFrame({
       "category": ["A"] * n_cat_A + ["B"] * n_cat_B,
       "actual": np.concatenate([y_true_A, y_true_B]),
       "q10": np.concatenate([y_qlow_A, y_qlow_B]),
       "q90": np.concatenate([y_qup_A, y_qup_B]),
   })
   
   # --- 2. Plotting (sorted by category) ---
   axes = kd.plot_cas_layers(
       df,
       actual_col="actual",
       q_low_col="q10",
       q_up_col="q90",
       sort_by="category",
       title="Use Case 2: Anomaly Severity by Product Category",
       savefig="gallery/images/gallery_cas_layers_category.png",
   )
   # Customize x-axis for categorical data
   ax, ax2 = axes
   ax.set_xticks([n_cat_A / 2, n_cat_A + n_cat_B / 2])
   ax.set_xticklabels(["Category A", "Category B"])
   ax.figure.savefig(
       "gallery/images/gallery_cas_layers_category.png",
       bbox_inches="tight"
   )
   plt.close()

.. figure:: ../images/anomaly/gallery_cas_layers_category.png
   :align: center
   :width: 80%
   :alt: A layered Cartesian plot comparing anomalies by category.

   The plot is sorted by category, revealing that all severe
   anomalies are concentrated in Category B.

.. topic:: 🧠 Interpretation
   :class: hint

   By sorting the data by `category`, the plot is divided into
   two distinct sections. The left side, corresponding to
   **Category A**, shows no anomalies in the top panel and zero
   severity in the bottom panel. In contrast, the right side,
   representing **Category B**, shows a clear cluster of severe,
   hot-colored anomalies in the top panel. The bottom panel
   confirms this with a large, concentrated spike in the severity
   score exclusively within the Category B section. This plot
   makes it immediately obvious that the model's reliability is
   not uniform but is conditional on the product category.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the CAS score and its components,
please refer back to the main :ref:`ug_plot_cas_layers` or 
:ref:`ug_cluster_aware_severity_score` section in the metrics
guide.

.. _gallery_application_anomaly_diagnostics:

------------------------------------------------------------
Application: Diagnosing Spatiotemporal Geohazard Forecasts
------------------------------------------------------------

In high-stakes fields like geohazard forecasting, an aggregate
metric of a model's performance is not enough. A model that is
"correct on average" can still be dangerously unreliable if all
its failures are concentrated in the most vulnerable areas.
Decision-makers need to understand the **structure of forecast
failures** to manage risk effectively.

This application demonstrates how to combine the **Polar Anomaly
Glyph Plot** and the **Layered Anomaly Profile** into a single
dashboard to conduct a comprehensive diagnosis of a model's
reliability and biases.

**The Problem: Forecasting Land Subsidence**

.. admonition:: Practical Example

   A municipal government in a coastal city is using probabilistic
   forecasts to identify areas at high risk of land subsidence.
   This is a critical task for prioritizing infrastructure
   maintenance and implementing zoning regulations. An interval
   failure, or "anomaly," has severe consequences:

   - **Over-prediction (Risk Underestimation)**: If the model
     predicts less subsidence than actually occurs, critical
     infrastructure could be damaged without warning. This is the
     most dangerous type of error.
   - **Under-prediction (Risk Overestimation)**: If the model
     predicts more subsidence than occurs, it could lead to the
     unnecessary and costly reinforcement of safe infrastructure.

   The city is evaluating two models: a new, complex **Deep
   Learning (DL) model** that is highly reliable (well-calibrated)
   but sometimes imprecise, and a simpler **Machine Learning (ML)
   model** that is much sharper (more precise) but is suspected of
   having a systematic bias.

**Translating the Problem into a Visual Dashboard**

To make an informed decision, we need to move beyond simple
coverage scores and dissect the *nature* of each model's
failures. A 2x2 dashboard will allow us to compare both the
polar and Cartesian views of the anomaly profiles for each model
side-by-side.

The following code simulates the models' performance and creates
the diagnostic dashboard.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   # --- 1. Simulate Land Subsidence Data ---
   np.random.seed(42)
   n_locations = 500
   # Sort by a spatial feature, e.g., distance from coast
   distance_from_coast = np.linspace(0, 20, n_samples)
   
   # True subsidence is higher further from the coast
   y_true = 10 + (distance_from_coast**1.5) + np.random.normal(
       0, 3, n_locations
   )

   # --- 2. Simulate Forecasts from Two Models ---
   # DL Model: Reliable but not very sharp (wide intervals)
   dl_qlow = y_true - 15
   dl_qup = y_true + 15
   
   # ML Model: Sharp (narrow intervals) but systematically biased
   # It underestimates the risk for high-subsidence areas
   ml_qlow = y_true - 4
   ml_qup = y_true + 4
   # Introduce the failure hotspot for high-subsidence areas
   hotspot_mask = distance_from_coast > 15
   y_true[hotspot_mask] += np.random.uniform(5, 15, hotspot_mask.sum())

   df_dl = pd.DataFrame({
       'distance': distance_from_coast, 'actual': y_true,
       'q_low': dl_qlow, 'q_up': dl_qup
   })
   df_ml = pd.DataFrame({
       'distance': distance_from_coast, 'actual': y_true,
       'q_low': ml_qlow, 'q_up': ml_qup
   })

   # --- 3. Create the 2x2 Dashboard ---
   fig = plt.figure(figsize=(18, 16))
   gs = fig.add_gridspec(2, 2)
   ax1 = fig.add_subplot(gs[0, 0], projection='polar')
   ax2 = fig.add_subplot(gs[0, 1], projection='polar')
   ax3 = fig.add_subplot(gs[1, 0])
   ax4 = fig.add_subplot(gs[1, 1])
   
   fig.suptitle("Land Subsidence Forecast Anomaly Dashboard",
                fontsize=24, y=0.98)

   # Top Row: Polar Glyph Plots for each model
   kd.plot_glyphs(
       df_dl, 'actual', 'q_low', 'q_up', ax=ax1,
       sort_by='distance', title="DL Model (Reliable but Imprecise)"
   )
   kd.plot_glyphs(
       df_ml, 'actual', 'q_low', 'q_up', ax=ax2,
       sort_by='distance', title="ML Model (Sharp but Biased)"
   )

   # Bottom Row: Cartesian Anomaly Profiles for each model
   kd.plot_cas_profile(
       df_dl, 'actual', 'q_low', 'q_up', ax=ax3,
   )
   kd.plot_cas_profile(
       df_ml, 'actual', 'q_low', 'q_up', ax=ax4,
   )

   fig.tight_layout(pad=2.0)
   fig.savefig("gallery/images/gallery_anomaly_dashboard.png")
   plt.close(fig)

.. figure:: ../images/anomaly/gallery_anomaly_dashboard.png
   :align: center
   :width: 100%
   :alt: A 2x2 dashboard showing different anomaly diagnostic plots.

   A comprehensive dashboard using polar glyphs and Cartesian
   profiles to diagnose the failure modes of two competing
   geohazard forecast models.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This dashboard provides a clear, evidence-based narrative for
   choosing the right model for the job.

   1.  **DL Model (Left Column):** This model is "boringly"
       good. Both the polar glyph plot (top-left) and the
       Cartesian profile (bottom-left) are nearly empty. This
       indicates there are **almost no forecast failures**. Its
       wide prediction intervals successfully capture the true
       outcome nearly every time. The model is highly **reliable**,
       but as a consequence, it is not very **sharp** or precise.

   2.  **ML Model (Right Column):** This model tells a much more
       dangerous story. The polar glyph plot (top-right) reveals
       a **hotspot of bright yellow, outward-pointing triangles (`▲`)**
       at the top of the circle. The Cartesian profile (bottom-
       right) confirms this, showing a large cluster of severe
       anomalies at the high end of the x-axis (which corresponds
       to the largest distance from the coast).

   **Conclusion:** The dashboard makes the trade-off explicit. The
   **DL Model** is the safe, reliable choice; its uncertainty bounds
   can be trusted, but they are not very precise. The **ML Model**,
   while appearing sharper, has a critical, systematic flaw: it
   **severely underestimates risk** in the most vulnerable,
   high-subsidence areas. For a geohazard application, this is an
   unacceptable bias. The dashboard provides the clear evidence
   needed to reject the sharper but biased ML model in favor of the
   more reliable DL model.

.. admonition:: Best Practice
   :class: hint

   When evaluating probabilistic forecasts, never rely on a single
   metric. A model with high sharpness (narrow intervals) may seem
   appealing, but it is useless if it is not also well-calibrated.
   Always use a combination of diagnostics. The CAS score and its
   associated visualizations are designed to be used alongside
   standard metrics like coverage and PIT histograms to get a
   complete picture of a model's performance.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the mathematical concepts behind the CAS
score, please refer to the main User Guide
:ref:`userguide_metrics`.