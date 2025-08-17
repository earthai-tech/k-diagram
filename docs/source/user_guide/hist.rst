.. _userguide_hist:

======================================= 
Visualizing 1D Distributions
======================================= 

Understanding the distribution of a single variable is a cornerstone of
data analysis and forecast evaluation. Before applying complex
visualizations, it's often crucial to inspect the fundamental
characteristics of key metrics, such as prediction errors or the
width of uncertainty intervals. The histogram and its smoothed
counterpart, the Kernel Density Estimate (KDE), are primary tools
for this task.

The :mod:`kdiagram.utils.hist` module provides straightforward functions
for creating these essential distribution plots.

Summary of Histogram Functions
-------------------------------

.. list-table:: Distribution Visualization Functions
    :widths: 40 60
    :header-rows: 1

    *   - Function
        - Description
    *   - :func:`~kdiagram.utils.plot_hist_kde`
        - Plots a histogram combined with a smooth Kernel Density
          Estimate (KDE) curve to visualize a 1D distribution.

Detailed Explanations
-----------------------

Let's explore the `plot_hist_kde` function in detail.

.. _ug_plot_hist_kde:

Histogram and KDE (:func:`~kdiagram.utils.plot_hist_kde`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**

This function provides a comprehensive visualization of a one-dimensional
data distribution. It combines a **histogram**, which groups data into
bins, with a **Kernel Density Estimate (KDE)**, which provides a smooth,
continuous estimate of the probability density function. This dual view
is highly effective for understanding the shape, central tendency, and
spread of a variable.

**Mathematical Concept:**

1. **Histogram**: The data range is divided into a series of intervals,
   or **bins**. The plot displays bars where the height of each bar
   corresponds to the number of data points that fall into that bin. When
   `density=True`, the bar heights are normalized so that the total area
   of the histogram equals 1.

2. **Kernel Density Estimate (KDE)**: The KDE is a non-parametric way
   to estimate the probability density function of a random variable. It
   creates a smooth curve by placing a kernel function (typically a
   Gaussian, or "bell curve") on each data point, and then summing all
   these kernels. The resulting curve, :math:`\hat{f}_h(x)`, is a smooth
   estimate of the data's distribution.

   .. math::

      \hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

   Here, :math:`K` is the kernel function, :math:`h` is the bandwidth
   (a smoothing parameter), and :math:`n` is the number of data points.

**Interpretation:**

* **Shape:** The overall shape of the histogram and KDE curve reveals
  the nature of the distribution. Is it symmetric (like a normal
  distribution), skewed to one side, or does it have multiple peaks
  (bimodal or multimodal)?
* **Central Tendency:** The location of the highest peak(s) indicates
  the mode(s) of the data—the most frequently occurring values.
* **Spread:** The width of the distribution indicates the variability
  or dispersion of the data. A narrow plot signifies low variance,
  while a wide plot signifies high variance.
* **Outliers:** Data points that fall far from the central mass of the
  distribution can be identified in the tails of the plot.

**Use Cases:**

* **Forecast Error Analysis:** This is a primary use case. Plotting the
  distribution of prediction errors (:math:`y_{true} - \hat{y}_{pred}`)
  is crucial. A good model often has errors that are normally
  distributed and centered at zero.
* **Uncertainty Assessment:** Visualize the distribution of prediction
  interval widths (:math:`Q_{up} - Q_{low}`). A narrow, unimodal
  distribution is often desirable, as it suggests the model produces
  consistent uncertainty estimates.
* **Feature Inspection:** Before building a model, inspect the
  distribution of input features to identify skewness or other
  characteristics that might require transformation.

**Example:**
(See the :ref:`Histogram with KDE Overlay <gallery_plot_hist_kde>`
in the Gallery for code and a plot example)

.. rubric:: References

.. bibliography::
   :style: plain
   :filter: cited