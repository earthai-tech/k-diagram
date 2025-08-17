.. _userguide_errors:

=======================================
Visualizing Forecast Errors
=======================================

A crucial part of model evaluation is the direct analysis of its
errors. While uncertainty visualizations focus on the predicted range,
error visualizations focus on the discrepancy between the point
forecast and the actual outcome (:math:`e = y_{true} - \hat{y}_{pred}`).
A thorough error analysis can reveal systemic biases, inconsistencies,
and opportunities for model improvement.

The :mod:`kdiagram.plot.errors` module provides specialized polar plots
to diagnose and compare model errors in an intuitive, visual manner.

Summary of Error Visualization Functions
------------------------------------------

.. list-table:: Error Visualization Functions
    :widths: 40 60
    :header-rows: 1

    *   - Function
        - Description
    *   - :func:`~kdiagram.plot.errors.plot_error_bands`
        - Visualizes mean error (bias) and error variance as a function
          of a cyclical or ordered feature.
    *   - :func:`~kdiagram.plot.errors.plot_error_violins`
        - Compares the full error distributions of multiple models on a
          single polar plot.
    *   - :func:`~kdiagram.plot.errors.plot_error_ellipses`
        - Displays two-dimensional uncertainty using error ellipses,
          ideal for spatial or positional errors.

Detailed Explanations
-----------------------

Let's explore these functions in detail.

.. _ug_plot_error_bands:

Systemic vs. Random Error (:func:`~kdiagram.plot.errors.plot_error_bands`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This plot is designed to decompose a model's error into two components:
**systemic error (bias)** and **random error (variance)**. It achieves
this by aggregating errors across bins of an angular variable (like
the month of the year or hour of the day) and displaying the mean and
standard deviation of the errors in each bin (:cite:t:`kouadiob2025`).

**Mathematical Concept:**
The function first partitions the dataset into :math:`K` bins,
:math:`B_k`, based on the ``theta_col`` values.

1. **Mean Error (Bias):** For each bin :math:`B_k`, the mean error
   :math:`\mu_{e,k}` is calculated. This represents the average bias
   of the model under the conditions of that bin.

   .. math::

      \mu_{e,k} = \frac{1}{|B_k|} \sum_{i \in B_k} e_i

   where :math:`e_i` is the error of sample :math:`i`. This is plotted
   as the central black line.

2. **Error Variance:** The standard deviation of the error,
   :math:`\sigma_{e,k}`, is calculated for each bin. This measures the
   consistency or random scatter of the errors.

   .. math::

      \sigma_{e,k} = \sqrt{\frac{1}{|B_k|-1} \sum_{i \in B_k} (e_i - \mu_{e,k})^2}

3. **Error Band:** A shaded band is drawn around the mean error line,
   with its boundaries defined as:

   .. math::

      \text{Bounds}_k = \mu_{e,k} \pm n_{std} \cdot \sigma_{e,k}

   The width of this band is a direct visualization of the model's
   random error.

**Interpretation:**

* **Mean Error Line (Bias):** If this line deviates from the "Zero Error"
  reference circle, the model has a systemic bias in that angular region.
  An outward deviation means over-prediction on average; an inward
  deviation means under-prediction.
* **Shaded Band (Variance):** A wide band indicates high variance, meaning
  the model's predictions are inconsistent and unreliable in that region.
  A narrow band indicates consistent, low-variance errors.

**Use Cases:**

* Diagnosing if a model's bias is dependent on a cyclical feature like
  seasonality or time of day.
* Identifying conditions under which a model's performance becomes
  unstable or inconsistent.
* Separating reducible systemic errors (bias) from irreducible random
  errors (variance) to guide model improvement efforts.

**Example:**
(See :ref:`Gallery <gallery_plot_error_bands>` for code and plot examples)

.. raw:: html

    <hr>

.. _ug_plot_error_violins:

Comparing Error Distributions (:func:`~kdiagram.plot.errors.plot_error_violins`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function provides a direct visual comparison of the **full error
distributions** for multiple models on a single polar plot. It adapts the 
traditional violin plot (:cite:t:`Hintze1998`) to a polar coordinate system, 
to show the shape, bias, and variance of each model's errors, making it an
excellent tool for model selection.

**Mathematical Concept:**
For each model's error data, a **Kernel Density Estimate (KDE)** is
computed to create a smooth representation of its probability density
function, :math:`\hat{f}_h(x)`.

.. math::

   \hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

This density curve is then plotted symmetrically around a radial axis to
form the "violin" shape. The width of the violin at any error value
:math:`x` is proportional to the probability density :math:`\hat{f}_h(x)`.
Each model is assigned its own angular sector on the polar plot.

**Interpretation:**

* **Bias (Centering):** The location of the widest part of the violin
  relative to the "Zero Error" circle reveals the model's bias. A violin
  centered on the circle is unbiased. A violin shifted outward indicates
  a positive bias (over-prediction), while a shift inward indicates a
  negative bias (under-prediction).
* **Variance (Width/Height):** A short, wide violin signifies a
  high-variance model with inconsistent errors. A tall, narrow violin
  signifies a low-variance model with consistent performance.
* **Shape:** The shape of the violin reveals further details. An
  asymmetric shape indicates skewed errors. Multiple wide sections
  (bimodality) suggest the model makes two or more common types of errors.

**Use Cases:**

* Directly comparing the overall performance of multiple candidate models.
* Selecting a model based on a holistic view of its error profile
  (e.g., choosing a slightly biased but highly consistent model over an
  unbiased but inconsistent one).
* Presenting a summary of comparative model performance to stakeholders.

**Example:**
(See :ref:`Gallery <gallery_plot_error_violins>` for code and plot examples)

.. raw:: html

    <hr>

.. _ug_plot_polar_error_ellipses:

Visualizing 2D Uncertainty (:func:`~kdiagram.plot.errors.plot_error_ellipses`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function is designed for visualizing **two-dimensional uncertainty**,
a concept explored in (:cite:t:`kouadiob2025`),  which is common in spatial
or positional forecasting. It draws an ellipse for each data point, where 
the ellipse's size and orientation represent the uncertainty in both the 
radial and angular directions.

**Mathematical Concept:**
For each data point :math:`i`, we have a mean position
:math:`(\mu_{r,i}, \mu_{\theta,i})` and the standard deviations of the
errors in those directions, :math:`\sigma_{r,i}` and
:math:`\sigma_{\theta,i}`.

The ellipse is defined by its half-width (in the radial direction) and
half-height (in the tangential direction):

.. math::

   \text{width} &= n_{std} \cdot \sigma_{r,i} \\
   \text{height} &= n_{std} \cdot (\mu_{r,i} \cdot \sin(\sigma_{\theta,i}))

The ellipse is then rotated by the angle :math:`\mu_{\theta,i}` and
translated to its mean position on the polar plot. The area of the
ellipse represents the confidence region (e.g., :math:`n_{std}=2`
approximates a 95% confidence region).

**Interpretation:**

* **Ellipse Position:** The center of the ellipse marks the mean predicted
  location.
* **Ellipse Size:** A larger ellipse indicates greater overall positional
  uncertainty.
* **Ellipse Shape (Eccentricity):** The shape reveals the nature of the
  uncertainty. A circular ellipse means the error is similar in all
  directions. An elongated ellipse indicates that the error is much
  larger in one direction (e.g., radial) than the other (e.g., angular).

**Use Cases:**

* Visualizing the uncertainty in tracking applications (e.g., predicting
  the future position of a vehicle or storm).
* Understanding the directionality of spatial forecast errors.
* Assessing the positional accuracy of simulation models.

**Example:**
(See :ref:`Gallery <gallery_plot_polar_error_ellipses>` for code and plot examples)


.. raw:: html

    <hr>
    
.. rubric:: References

.. bibliography::
   :style: plain
   :filter: cited