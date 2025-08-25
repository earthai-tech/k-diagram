.. _userguide_probabilistic:

====================================
Evaluating Probabilistic Forecasts
====================================

While prediction intervals provide a crucial view of uncertainty, a full
**probabilistic forecast** offers a more complete picture by assigning a
probability to all possible future outcomes. Evaluating these predictive
distributions requires moving beyond simple interval checks to assess two
fundamental and often competing qualities: **calibration** and **sharpness**
:footcite:p:`Gneiting2007b`.

* **Calibration** (or reliability) refers to the statistical consistency
  between the probabilistic forecasts and the observed outcomes. A
  well-calibrated forecast is "honest" about its own uncertainty.
* **Sharpness** refers to the concentration of the predictive distribution.
  A sharp forecast provides narrow, highly specific prediction intervals.

An ideal forecast is one that is both perfectly calibrated and maximally
sharp. The :mod:`kdiagram.plot.probabilistic` module provides a suite of
specialized polar plots to diagnose these two key properties.

.. admonition:: From Theory to Practice: A Real-World Case Study
   :class: hint

   The visualization methods described in this guide were developed to solve
   practical challenges in interpreting complex, high-dimensional forecasts.
   For a detailed case study demonstrating how these plots are used to
   analyze the spatiotemporal uncertainty of a deep learning model for
   land subsidence forecasting, please refer to our research paper
   :footcite:p:`kouadiob2025`.

Summary of Probabilistic Diagnostic Functions
---------------------------------------------

.. list-table:: Probabilistic Visualization Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.probabilistic.plot_pit_histogram`
     - Assesses forecast calibration using a Polar Probability
       Integral Transform (PIT) histogram.
   * - :func:`~kdiagram.plot.probabilistic.plot_polar_sharpness`
     - Compares the sharpness (average interval width) of one or
       more models.
   * - :func:`~kdiagram.plot.probabilistic.plot_crps_comparison`
     - Provides an overall performance score using the Continuous
       Ranked Probability Score (CRPS).
   * - :func:`~kdiagram.plot.probabilistic.plot_credibility_bands`
     - Visualizes how the forecast's median and credibility bands
       change as a function of another feature.
   * - :func:`~kdiagram.plot.probabilistic.plot_calibration_sharpness`
     - Visualizes the direct trade-off between calibration and
       sharpness for multiple models.

.. raw:: html

   <hr>
   
.. _ug_plot_pit_histogram:

PIT Histogram (:func:`~kdiagram.plot.probabilistic.plot_pit_histogram`)
-----------------------------------------------------------------------

**Purpose:**
Creates a **Polar Probability Integral Transform (PIT) histogram**, a
primary diagnostic for assessing the **calibration** (reliability) of a
probabilistic forecast. It answers: *Are the predicted probability
distributions statistically consistent with the observed outcomes?*

**Mathematical Concept**
The Probability Integral Transform (PIT) is foundational in forecast
verification :footcite:p:`Gneiting2007b`. For a continuous predictive
distribution with CDF :math:`F`, the PIT value for an observation
:math:`y` is :math:`F(y)`. If forecasts are perfectly calibrated, PIT
values across observations are i.i.d. uniform on :math:`[0,1]`.

When only a finite set of :math:`M` quantiles is available (common in ML
workflows), the PIT for observation :math:`y_i` can be approximated by
the fraction of forecast quantiles less than or equal to :math:`y_i`:

.. math::
   :label: eq:pit_quantile

   \mathrm{PIT}_i \;=\; \frac{1}{M} \sum_{j=1}^{M}
   \mathbf{1}\{\, q_{i,j} \le y_i \,\},

where :math:`q_{i,j}` is the :math:`j`-th quantile forecast for
observation :math:`i`, and :math:`\mathbf{1}` is the indicator function.
The histogram is then formed from the set of :math:`\mathrm{PIT}_i`
values.

**Interpretation:**

In the polar plot, PIT bins map to the **angle**; frequencies map to the
**radius**.

* **Perfect calibration:** A uniform PIT histogram. In polar form, bars
  lie on a **perfect circle**, matching the dashed “Uniform” reference.
* **Over-confidence (too narrow intervals):** **U-shaped** histogram:
  large counts near 0 and 1, few in the middle.
* **Under-confidence (too wide intervals):** **Hump-shaped** histogram:
  excess mass near the center.
* **Systemic bias:** Sloped or skewed histogram indicating forecasts are
  consistently too high or too low.

**Use Cases:**

* Visual assessment of probabilistic **calibration**.
* Diagnose **overconfidence**, **underconfidence**, or **bias**.
* Compare calibration across models before evaluating **sharpness**.

**Example:**

See the gallery example and code: :ref:`gallery_plot_pit_histogram`.

.. raw:: html

   <hr>
   
.. _ug_plot_polar_sharpness:

Polar Sharpness Diagram (:func:`~kdiagram.plot.probabilistic.plot_polar_sharpness`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This function creates a **Polar Sharpness Diagram** to visually
compare the **sharpness** (or precision) of one or more probabilistic
forecasts. While calibration assesses a forecast's reliability,
sharpness measures the concentration of its predictive distribution.
An ideal forecast is not only calibrated but also as sharp as
possible. This plot directly answers the question: *"Which model
provides the most precise (narrowest) forecast intervals?"*

**Mathematical Concept**
Sharpness is a property of the forecast alone and does not depend on
the observed outcomes :footcite:p:`Gneiting2007b`. It is typically
quantified by the average width of the prediction intervals.

1.  **Interval Width**: For each model and each observation :math:`i`,
    the width of the central prediction interval is calculated using
    the lowest (:math:`q_{min}`) and highest (:math:`q_{max}`)
    provided quantiles.

    .. math::
       :label: eq:interval_width

       w_i = y_{i, q_{max}} - y_{i, q_{min}}

2.  **Sharpness Score**: The sharpness score :math:`S` for each model
    is the average of these interval widths over all :math:`N`
    observations. This score is used as the **radial coordinate**
    in the polar plot. A **lower score is better**, indicating a
    sharper, more concentrated forecast.

    .. math::
       :label: eq:sharpness

       S = \frac{1}{N} \sum_{i=1}^{N} w_i


**Interpretation**
The plot assigns each model its own angular sector for clear
separation, with the radial distance from the center representing
its sharpness.

* **Radius**: The distance from the center directly corresponds to
  the average prediction interval width. **Points closer to the
  center represent sharper, more desirable forecasts.**
* **Comparison**: The plot allows for an immediate visual comparison
  of the relative sharpness of different models.

**Use Cases:**

* To directly compare the precision (average interval width) of
  multiple forecasting models.
* To use in conjunction with a calibration plot (like the PIT
  Histogram) to understand the crucial **trade-off between a model's
  reliability and its sharpness**. A model might be very sharp but
  poorly calibrated, or vice-versa.
* To select a model that provides the best balance of sharpness and
  calibration for a specific application.


**Example**
See the gallery example and code: :ref:`gallery_plot_polar_sharpness`.

.. _ug_plot_crps_comparison:

.. raw:: html

   <hr>
   
CRPS Comparison (:func:`~kdiagram.plot.probabilistic.plot_crps_comparison`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar CRPS Comparison Diagram** to provide a
high-level summary of a model's overall probabilistic skill. It uses
the Continuous Ranked Probability Score (CRPS), a proper scoring rule
that assesses both **calibration** and **sharpness** simultaneously.
This plot answers the question: *"Which model performs best overall
when considering both reliability and precision?"*

**Mathematical Concept:**
The Continuous Ranked Probability Score (CRPS) is a widely used
metric for evaluating probabilistic forecasts that generalizes the
Mean Absolute Error :footcite:p:`Gneiting2007b`. For a single
observation :math:`y` and a predictive CDF :math:`F`, it is defined as:

.. math::
   :label: eq:crps_integral

   \text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(x) -
   \mathbf{1}\{x \ge y\})^2 dx

where :math:`\mathbf{1}` is the Heaviside step function. A lower
CRPS value indicates a better forecast.

When the forecast is given as a set of :math:`M` quantiles
:math:`\{q_1, ..., q_M\}`, the CRPS can be approximated by
averaging the pinball loss :math:`\mathcal{L}_{\tau}` over the
quantile levels :math:`\tau \in \{ \tau_1, ..., \tau_M \}`. The
pinball loss for a single quantile forecast :math:`q` at level
:math:`\tau` is:

.. math::
   :label: eq:pinball_loss

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

This function calculates the average CRPS over all observations for
each model and plots this final score as the radial coordinate.


**Interpretation:**
The plot assigns each model its own angular sector, with the radial
distance from the center representing its overall performance.

* **Radius**: The distance from the center directly corresponds to
  the average CRPS. **Points closer to the center represent
  better-performing models.**
* **Comparison**: The plot provides an immediate visual summary of
  the relative performance of different models. It is a "bottom-line"
  metric but does not explain *why* one model is better (i.e.,
  whether due to superior calibration or superior sharpness).


**Use Cases**

* To get a quick, high-level summary of which model performs best
  overall when considering both calibration and sharpness.
* To use as a final comparison plot after using the PIT histogram
  and sharpness diagram to understand the components of the CRPS score.
* For model selection when a single, proper scoring rule is the
  primary decision criterion.


**Example**
See the gallery example and code: :ref:`gallery_plot_crps_comparison`.

.. raw:: html

   <hr>
   
.. _ug_plot_credibility_bands:

Polar Credibility Bands (:func:`~kdiagram.plot.probabilistic.plot_credibility_bands`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Credibility Bands** plot to
visualize the structure of a model's forecast distribution as a
function of another variable. It is a descriptive tool that answers
the question: *"How do my model's median prediction and its
uncertainty (interval width) change depending on a specific
feature?"*

**Mathematical Concept:**
This plot visualizes the conditional expectation of the forecast
quantiles. It is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`.

1.  **Binning**: The data is first partitioned into :math:`K` bins,
    :math:`B_k`, based on the values in ``theta_col``.

2.  **Conditional Means**: For each bin :math:`B_k`, the mean
    of the lower quantile (:math:`\bar{q}_{low,k}`), median
    quantile (:math:`\bar{q}_{med,k}`), and upper quantile
    (:math:`\bar{q}_{up,k}`) are calculated.

    .. math::
       :label: eq:mean_quantiles

       \bar{q}_{j,k} = \frac{1}{|B_k|} \sum_{i \in B_k} q_{j,i}

    where :math:`j \in \{\text{low, med, up}\}`.

3.  **Visualization**: The plot displays:

    - A central line representing the mean median forecast
      (:math:`\bar{q}_{med,k}`).
    - A shaded band between the mean lower and upper bounds
      (:math:`\bar{q}_{low,k}` and :math:`\bar{q}_{up,k}`). The
      width of this band represents the average forecast
      sharpness for that bin.

**Interpretation:**
The plot reveals how the forecast distribution's center and spread
are related to the feature on the angular axis.

* **Central Line (Mean Median)**: The position of this line shows
  the average central tendency of the forecast for each bin.
  Trends in this line reveal if the model's predictions are
  correlated with the binned feature.
* **Shaded Band (Credibility Band)**: The width of this band
  visualizes the average forecast sharpness. If the band's width
  changes at different angles, it is a clear sign of
  **heteroscedasticity**—meaning the model's uncertainty is not
  constant but depends on the binned feature.

**Use Cases:**

* To diagnose if a model's uncertainty changes predictably with
  another feature (e.g., time, or the magnitude of the forecast
  itself).
* To visually inspect the conditional mean of a forecast.
* To communicate how the forecast distribution is expected to behave
  under different conditions.


**Example:**
See the gallery example and code: :ref:`gallery_plot_credibility_bands`.

.. raw:: html

   <hr>
   
.. _ug_plot_calibration_sharpness:

Calibration-Sharpness Diagram (:func:`~kdiagram.plot.probabilistic.plot_calibration_sharpness`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Calibration-Sharpness Diagram**, a
powerful summary visualization that plots the fundamental trade-off
between a forecast's **calibration** (reliability) and its
**sharpness** (precision). Each model is represented by a single
point, allowing for an immediate and intuitive comparison of overall
probabilistic performance. The ideal forecast is located at the
center of the plot.

**Mathematical Concept:**
This plot synthesizes two key aspects of a probabilistic forecast
into a single point for each model. It is a novel visualization
developed as part of the analytics framework
:footcite:p:`kouadiob2025`.

1.  **Sharpness (Radius)**: The radial coordinate represents the
    forecast's sharpness, calculated as the average width of the
    prediction interval between the lowest and highest provided
    quantiles. A smaller radius is better (sharper).

    .. math::
       :label: eq:sharpness_score

       S = \frac{1}{N} \sum_{i=1}^{N} (y_{i, q_{max}} - y_{i, q_{min}})

2.  **Calibration Error (Angle)**: The angular coordinate
    represents the forecast's calibration error. This is
    quantified by first calculating the Probability Integral
    Transform (PIT) values for each observation. The
    Kolmogorov-Smirnov (KS) statistic is then used to measure
    the maximum distance between the empirical CDF of these PIT
    values and the CDF of a perfect uniform distribution.

    .. math::
       :label: eq:calib_error

       E_{calib} = \sup_{x} | F_{PIT}(x) - U(x) |

    An error of 0 indicates perfect calibration. The angle is
    mapped such that :math:`\theta = E_{calib} \cdot \frac{\pi}{2}`,
    so 0° is perfect and 90° is the worst possible calibration.

**Interpretation:**
The plot provides a high-level summary of probabilistic forecast
quality, with the ideal model located at the center (origin).

* **Radius (Sharpness)**: The distance from the center. **Models
  closer to the center are sharper** (more precise).
* **Angle (Calibration Error)**: The angle from the 0° axis.
  **Models with a smaller angle are better calibrated**.
* **Overall Performance**: The best model is the one closest to the
  origin, as it represents the optimal balance of both low
  calibration error and high sharpness.


**Use Cases:**

* To quickly compare the overall quality of multiple probabilistic
  models in a single, decision-oriented view.
* To visualize the trade-off between a model's reliability and its
  precision. For example, one model might be very sharp but poorly
  calibrated, while another is well-calibrated but not very sharp.
* For model selection when a balanced performance between
  calibration and sharpness is the primary goal.


**Example**
See the gallery example and code:
:ref:`gallery_plot_calibration_sharpness`.

.. raw:: html

   <hr>
   
.. rubric:: References

.. footbibliography::