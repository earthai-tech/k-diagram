.. _gallery_hist:

============================
Histogram Gallery
============================

This gallery page showcases histogram-based plots from `k-diagram`
designed for visualizing the distribution of one-dimensional data,
which is fundamental for evaluating forecast errors and uncertainty.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots.

.. _gallery_plot_hist_kde:

----------------------------------
Histogram with KDE Overlay
----------------------------------

Uses :func:`~kdiagram.utils.hist.plot_hist_kde` to visualize the
distribution of a variable. It combines a traditional histogram with a
smooth Kernel Density Estimate (KDE) curve to provide a detailed
view of the data's shape, central tendency, and spread.

.. code-block:: python
    :linenos:

    import kdiagram.utils.hist as kdh
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Data Generation ---
    # Simulate sensor readings with a normal distribution
    np.random.seed(42)
    sensor_readings = np.random.normal(loc=100, scale=15, size=1000)

    # --- Plotting ---
    kdh.plot_hist_kde(
        sensor_readings,
        bins=40,
        title="Distribution of Sensor Readings",
        x_label="Temperature (°C)",
        kde_color="#FF5733", # Custom reddish-orange
        # Save the plot (adjust path relative to docs/source/)
        savefig="gallery/images/gallery_utils_sensor_reading.png"
    )
    plt.close() # Close plot after saving

.. image:: ../images/gallery_utils_sensor_reading.png
    :alt: Example of a Histogram with a KDE Overlay
    :align: center
    :width: 70%

.. topic:: 🧠 Analysis and Interpretation
    :class: hint

    The **Histogram with KDE Overlay** is a classic and effective
    tool for inspecting the distribution of any continuous variable,
    such as prediction errors or interval widths.

    **Analysis and Interpretation:**

    * **Histogram (Blue Bars):** The height of each bar shows the
      frequency (or density) of data points falling within that
      specific bin or range. It gives a discretized view of the
      distribution.
    * **KDE Curve (Orange Line):** The Kernel Density Estimate
      provides a smooth, continuous line that estimates the
      underlying probability density function of the data. It helps
      to identify the shape, peaks (modes), and skewness of the
      distribution more clearly than the histogram alone.
    * **Combined View:** Together, they confirm the data's
      characteristics. The KDE should generally follow the contour
      of the histogram bars.

    **🔍 Key Insights from this Example:**

    * The data (simulated sensor readings) appears to be **normally
      distributed**, which is confirmed by the classic "bell curve"
      shape of both the histogram and the KDE.
    * The **central peak** is located around 100, indicating that
      this is the most frequent sensor reading (the mean or mode).
    * The distribution is largely **symmetrical**, with tails
      tapering off evenly on both sides of the peak.

    **💡 When to Use:**

    * **Error Analysis:** To check if your model's prediction errors
      (``actual - predicted``) are normally distributed and centered
      at zero, which is a common assumption and sign of a well-calibrated
      model.
    * **Feature Distribution:** To understand the distribution of
      input features before modeling.
    * **Uncertainty Characterization:** To analyze the distribution
      of prediction interval widths. A narrow, single-peaked
      distribution is often desirable, indicating consistent
      uncertainty estimates.
