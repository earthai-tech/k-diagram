.. _gallery_probabilistic:

===================================
Probabilistic Diagnostics Gallery
===================================

This gallery page showcases plots from ``k-diagram`` designed for the
comprehensive evaluation of probabilistic forecasts. These visualizations
move beyond simple interval checks to assess the two key qualities of a
probabilistic forecast: **calibration** (is the forecast reliable?) and
**sharpness** (is the forecast precise?).

The plots provide intuitive diagnostics for PIT histograms, forecast
sharpness, overall performance (CRPS), and conditional uncertainty,
allowing for a deeper understanding of a model's predictive distributions.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots.

.. _gallery_plot_pit_histogram:

-----------------------------
PIT Histogram (Calibration)
-----------------------------

The :func:`~kdiagram.plot.probabilistic.plot_pit_histogram` function is
the primary diagnostic tool for assessing the **statistical calibration**
of a probabilistic forecast. It visualizes the Probability Integral
Transform (PIT) distribution, which for a perfectly calibrated model,
should be uniform. This polar version transforms that ideal uniform
histogram into a perfect circle, making deviations—and thus specific
types of miscalibration—immediately obvious.

First, let's break down the components of this core diagnostic plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** The angular axis is divided into bins representing
     the **PIT value**, spanning from 0 to 1. Each sector corresponds
     to a range of PIT values (e.g., 0.0 to 0.1).
   * **Radius (r):** The radius of each bar shows the **frequency** or
     count of PIT values that fall into that angular bin. Since the
     radial axis represents a simple count, its tick labels can
     optionally be hidden with ``mask_radius=True`` to simplify the
     visual.
   * **Reference Line:** The dashed red circle shows the **expected
     frequency** for a perfectly uniform (calibrated) distribution. The
     goal is for the blue bars to align with this circle.

With this framework, let's explore how to use this plot to diagnose
different, common types of forecast miscalibration.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: The Ideal Case - A Well-Calibrated Forecast**

The first and most important use case is to identify a well-calibrated
forecast. This serves as the benchmark against which all other models
are compared.

Let's simulate a probabilistic forecast for daily temperatures where the
model's predicted uncertainty perfectly matches the true, underlying
stochasticity of the weather.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from scipy.stats import norm
   import matplotlib.pyplot as plt

   # --- 1. Data Generation (Shared for all use cases) ---
   np.random.seed(42)
   n_samples = 2000
   true_mean = 15
   true_scale = 5.0 # The "true" uncertainty
   y_true = np.random.normal(loc=true_mean, scale=true_scale, size=n_samples)
   quantiles = np.linspace(0.05, 0.95, 19) # 19 quantiles from 5% to 95%

   # --- 2. Create a Well-Calibrated Forecast ---
   # The model's predicted scale matches the true scale
   calibrated_preds = norm.ppf(quantiles, loc=true_mean, scale=true_scale)
   # We need to broadcast it to match the shape of y_true
   calibrated_preds = np.tile(calibrated_preds, (n_samples, 1))

   # --- 3. Plotting ---
   kd.plot_pit_histogram(
       y_true, calibrated_preds, quantiles,
       title="Use Case 1: A Well-Calibrated Forecast",
       savefig="gallery/images/gallery_pit_histogram_calibrated.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_pit_histogram_calibrated.png
   :align: center
   :width: 70%
   :alt: A polar PIT histogram where the bars form a nearly perfect circle.

   The blue bars of the histogram align almost perfectly with the
   dashed red reference circle, indicating a uniform PIT distribution.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot is the signature of a **perfectly calibrated** forecast. The
   blue bars, representing the frequency of PIT values in each bin, are
   all very close in height to the dashed red reference circle. This
   indicates that the PIT values are uniformly distributed, which means
   the model's predicted probability distributions are statistically
   consistent with the observed outcomes. This is the ideal result we
   strive for.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Diagnosing an Overconfident Forecast**

A very common failure mode for modern machine learning models is
**overconfidence**. The model produces prediction intervals that are
systematically too narrow, failing to account for the true level of
uncertainty. The PIT histogram has a classic "tell" for this condition.

Let's simulate a model that underestimates the true volatility of the
weather, producing overly sharp forecasts.

.. code-block:: python
   :linenos:

   # --- 1. Data Generation (uses y_true and quantiles from previous step) ---
   # --- 2. Create an Overconfident Forecast ---
   # The model's predicted scale is smaller than the true scale
   overconfident_scale = 2.5 # Half of the true scale
   overconfident_preds = norm.ppf(quantiles, loc=true_mean, scale=overconfident_scale)
   overconfident_preds = np.tile(overconfident_preds, (n_samples, 1))

   # --- 3. Plotting ---
   kd.plot_pit_histogram(
       y_true, overconfident_preds, quantiles,
       title="Use Case 2: An Overconfident Forecast",
       color="#E74C3C", # Use red to indicate a problem
       savefig="gallery/images/gallery_pit_histogram_overconfident.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_pit_histogram_overconfident.png
   :align: center
   :width: 70%
   :alt: A U-shaped polar PIT histogram.

   The histogram bars are very high at the extremes (0.0 and 0.9 bins)
   and very low in the middle, forming a distinct U-shape.

.. topic:: 🧠 Interpretation
   :class: hint

   This plot shows a distinct **U-shaped** (or bowl-shaped) histogram.
   The bars in the lowest and highest PIT bins (near 0.0 and 0.9) are
   much taller than the reference circle, while the bars in the middle
   are much shorter. This is the classic signature of an
   **overconfident** model. It means the true observed values are
   frequently falling in the extreme tails of the predicted
   distribution—or outside of it entirely. The model's forecast
   intervals are too narrow, and it is not accounting for enough
   uncertainty.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 3: Diagnosing an Underconfident or Biased Forecast**

The opposite problem is **underconfidence**, where a model's prediction
intervals are systematically too wide. We can also use the PIT
histogram to diagnose a simple **bias**, where the model's central
tendency is consistently wrong.

Let's create side-by-side plots: one for an underconfident model and
one for a biased model, to see their distinct signatures.

.. code-block:: python
   :linenos:

   # --- 1. Data Generation (uses y_true and quantiles from previous steps) ---
   # --- 2. Create Underconfident and Biased Forecasts ---
   # Underconfident model: predicted scale is larger than true scale
   underconfident_scale = 10.0 # Double the true scale
   underconfident_preds = norm.ppf(quantiles, loc=true_mean, scale=underconfident_scale)
   underconfident_preds = np.tile(underconfident_preds, (n_samples, 1))
   # Biased model: central tendency is wrong
   biased_loc = 12.0 # True mean is 15
   biased_preds = norm.ppf(quantiles, loc=biased_loc, scale=true_scale)
   biased_preds = np.tile(biased_preds, (n_samples, 1))

   # --- 3. Create a figure with two polar subplots ---
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

   # --- 4. Plot each diagnostic on its dedicated axis ---
   kd.plot_pit_histogram(
       y_true, underconfident_preds, quantiles, ax=ax1,
       title="Underconfident Forecast", color="#F1C40F" # Yellow
   )
   kd.plot_pit_histogram(
       y_true, biased_preds, quantiles, ax=ax2,
       title="Biased Forecast", color="#9B59B6" # Purple
   )

   fig.suptitle('Use Case 3: Diagnosing Other Miscalibrations', fontsize=16)
   fig.tight_layout(rect=[0, 0, 1, 0.95])
   fig.savefig("gallery/images/gallery_pit_histogram_other.png")
   plt.close(fig)

.. figure:: ../images/probabilistic/gallery_pit_histogram_other.png
   :align: center
   :width: 90%
   :alt: Side-by-side comparison of a hump-shaped and a sloped PIT histogram.

   The left plot shows a hump-shaped histogram (underconfidence). The
   right plot shows a sloped histogram (bias).

.. topic:: 🧠 Interpretation
   :class: hint

   This side-by-side comparison reveals two different failure modes. The
   **Underconfident Forecast** (left) produces a **hump-shaped**
   histogram. The bars in the middle bins are much taller than the
   reference circle, while the bars at the extremes are too short. This
   means the true values are clustering too often in the center of the
   predicted distribution; the forecast intervals are too wide and
   overly conservative.

   The **Biased Forecast** (right) produces a **sloped** histogram. The
   bars are systematically too high on one side and too low on the other.
   This indicates a consistent bias in the central tendency of the
   forecast—in this case, the model is consistently predicting a lower
   temperature than what is observed.

.. admonition:: Best Practice
   :class: best-practice

   The PIT histogram should be your **first** diagnostic for any
   probabilistic forecast. A model that is not well-calibrated (i.e.,
   does not produce a uniform PIT histogram) cannot be trusted, even if
   it appears to be sharp or has a good overall score on other metrics.
   Always check for calibration first.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical theory behind the
Probability Integral Transform, please refer back to the main
:ref:`ug_plot_pit_histogram` section.

   
.. _gallery_plot_polar_sharpness:

-----------------------
Polar Sharpness Diagram
-----------------------

While the PIT histogram assesses a forecast's reliability, the
:func:`~kdiagram.plot.probabilistic.plot_polar_sharpness` function
evaluates its **sharpness**, or precision. A calibrated forecast that is
too wide (e.g., "tomorrow's temperature will be between -10°C and
40°C") is reliable but not very useful. This plot directly compares the
average width of prediction intervals from one or more models, helping
to identify which forecast is the most decisive.

Let's begin by understanding the components of this comparative plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular sector is assigned to a different
     **model** or prediction set. This is purely for visual separation;
     the angle itself has no numerical meaning. The angular tick labels
     are therefore hidden by default.
   * **Radius (r):** Directly corresponds to the **average prediction
     interval width**, which serves as the **sharpness score**. A
     **smaller radius is better**, as it indicates a sharper, more
     precise, and more useful forecast.
   * **Azimuth:** The azimuth, or the circular path, represents a line
     of constant sharpness. The plot's grid lines are drawn at specific
     sharpness levels to serve as a reference for comparing the models.

Now, let's apply this plot to a real-world model selection problem.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: The Sharpness-Calibration Trade-off**

The most important use of the sharpness diagram is in conjunction with a
calibration plot like the PIT histogram. A model can easily achieve high
sharpness by being overconfident, so we must evaluate both properties
together.

Let's continue our weather forecasting scenario. We have three models:
one is well-calibrated, one is overconfident (too sharp), and one is
underconfident (not sharp). This plot will quantify their precision.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from scipy.stats import norm
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Three models with different properties ---
   np.random.seed(42)
   n_samples = 2000
   true_mean = 15
   true_scale = 5.0
   y_true = np.random.normal(loc=true_mean, scale=true_scale, size=n_samples)
   quantiles = np.linspace(0.05, 0.95, 19)

   # Model A: Well-calibrated
   calibrated_preds = norm.ppf(quantiles, loc=true_mean, scale=true_scale)
   calibrated_preds = np.tile(calibrated_preds, (n_samples, 1))

   # Model B: Overconfident (too sharp)
   overconfident_preds = norm.ppf(quantiles, loc=true_mean, scale=2.5)
   overconfident_preds = np.tile(overconfident_preds, (n_samples, 1))

   # Model C: Underconfident (not sharp)
   underconfident_preds = norm.ppf(quantiles, loc=true_mean, scale=10.0)
   underconfident_preds = np.tile(underconfident_preds, (n_samples, 1))

   # --- 2. Plotting ---
   kd.plot_polar_sharpness(
       calibrated_preds,
       overconfident_preds,
       underconfident_preds,
       quantiles=quantiles,
       names=['A (Calibrated)', 'B (Overconfident)', 'C (Underconfident)'],
       title="Use Case: The Sharpness-Calibration Trade-off",
       savefig="gallery/images/gallery_polar_sharpness_basic.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_polar_sharpness_basic.png
   :align: center
   :width: 70%
   :alt: A polar sharpness diagram comparing three models.

   A polar plot with three points, each representing a model. The
   overconfident model is closest to the center (sharpest), while the
   underconfident model is farthest away.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a clear ranking of the models based on their
   precision. The **"Overconfident" model** is closest to the center,
   meaning it produces the **sharpest** (narrowest) forecast intervals.
   The **"Underconfident" model** is the farthest from the center,
   indicating its intervals are the widest and least precise. The
   **"Calibrated" model** sits in the middle.

   This highlights the critical trade-off: the sharpest forecast is not
   always the best. The PIT Histogram for the "Overconfident" model
   would show a U-shape, revealing its poor calibration. Therefore, the
   "Calibrated" model, which has both good calibration (from the PIT
   plot) and reasonable sharpness, is the best overall choice.

.. admonition:: Best Practice
   :class: best-practice

   **Never evaluate sharpness in isolation.** A model can always appear
   sharper by becoming more overconfident. Always use this plot in
   conjunction with the :func:`~kdiagram.plot.probabilistic.plot_pit_histogram`
   to ensure you are selecting a model that is both sharp *and* reliable.

.. admonition:: See Also
   :class: seealso

   The :func:`~kdiagram.plot.probabilistic.plot_crps_comparison` function
   and/or :func:`~kdiagram.plot.probabilistic.plot_calibration_sharpness` is 
   designed to combine both calibration and sharpness into a single,
   overall score, making it a great final step after analyzing the two
   properties separately.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind sharpness
and proper scoring rules, please refer back to the main
:ref:`ug_plot_polar_sharpness` section.

.. _gallery_plot_crps_comparison:


---------------------------------
CRPS Comparison (Overall Score)
---------------------------------

After analyzing a forecast's reliability (calibration) and precision
(sharpness) separately, we often need a single, overall score to make a
final decision. The :func:`~kdiagram.plot.probabilistic.plot_crps_comparison`
function provides this summary. It uses the Continuous Ranked Probability
Score (CRPS), a proper scoring rule that simultaneously rewards both
calibration and sharpness, to give a final verdict on which model
performs best.

Let's begin by understanding the components of this summary plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular sector is assigned to a different
     **model** or prediction set. This is purely for visual separation;
     the angle itself has no numerical meaning.
   * **Radius (r):** Directly corresponds to the **average CRPS** for
     that model. The CRPS is an error metric, so a **smaller radius is
     better**, indicating a more skillful probabilistic forecast.
   * **Azimuth:** The azimuth, or the circular path, represents a line
     of constant CRPS. The plot's grid lines serve as a reference for
     comparing the models' scores. Since the radius is the key metric,
     its labels are important, but can be hidden with ``mask_radius=True``
     for a cleaner look.

With this in mind, let's conclude our wind power forecasting case study
by using the CRPS to select the winning model.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: The Final Verdict in Model Selection**

The most powerful use of this plot is as the final step in a
probabilistic forecast evaluation. It synthesizes the complex trade-offs
between calibration and sharpness into a single, easy-to-interpret score.

Let's revisit our three wind power forecasting models: one
well-calibrated, one overconfident (too sharp), and one underconfident
(not sharp). The CRPS will penalize the miscalibrated models and reward
the one that achieves the best balance.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from scipy.stats import norm
   import matplotlib.pyplot as plt

   # --- 1. Data Generation (consistent with previous examples) ---
   np.random.seed(42)
   n_samples = 2000
   true_mean = 15
   true_scale = 5.0
   y_true = np.random.normal(loc=true_mean, scale=true_scale, size=n_samples)
   quantiles = np.linspace(0.05, 0.95, 19)

   # Model A: Well-calibrated
   calibrated_preds = norm.ppf(quantiles, loc=true_mean, scale=true_scale)
   calibrated_preds = np.tile(calibrated_preds, (n_samples, 1))

   # Model B: Overconfident (too sharp)
   overconfident_preds = norm.ppf(quantiles, loc=true_mean, scale=2.5)
   overconfident_preds = np.tile(overconfident_preds, (n_samples, 1))

   # Model C: Underconfident (not sharp)
   underconfident_preds = norm.ppf(quantiles, loc=true_mean, scale=10.0)
   underconfident_preds = np.tile(underconfident_preds, (n_samples, 1))

   # --- 2. Plotting ---
   kd.plot_crps_comparison(
       y_true,
       calibrated_preds,
       overconfident_preds,
       underconfident_preds,
       quantiles=quantiles,
       names=['A (Calibrated)', 'B (Overconfident)', 'C (Underconfident)'],
       title="Final Verdict: Overall Forecast Performance (CRPS)",
       savefig="gallery/images/gallery_crps_comparison.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_crps_comparison.png
   :align: center
   :width: 70%
   :alt: A polar CRPS diagram comparing three models.

   A polar plot with three points representing the final CRPS score for
   each model. The well-calibrated model is closest to the center,
   indicating the best overall performance.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides the final, summary judgment on our models. Since
   a lower CRPS is better, the model closest to the center is the
   overall winner. In this case, **"Model A (Calibrated)"** has the
   lowest CRPS and is closest to the origin. The CRPS correctly
   penalizes both the "Overconfident" model for its poor reliability
   and the "Underconfident" model for its lack of precision. Even though
   the overconfident model appeared "sharper" in the previous plot, its
   frequent, large errors when the truth falls outside its narrow
   intervals result in a higher (worse) CRPS score than the
   well-calibrated model. This confirms that Model A provides the best
   balance of both calibration and sharpness.

.. admonition:: Best Practice
   :class: best-practice

   The CRPS is an excellent "bottom-line" metric, but it should be used
   as the **final step** of an analysis, not the only step. Always use
   the :func:`~kdiagram.plot.probabilistic.plot_pit_histogram` and
   :func:`~kdiagram.plot.probabilistic.plot_polar_sharpness` plots
   first to understand *why* one model has a better CRPS than another.
   This allows you to diagnose if the improvement comes from better
   calibration, better sharpness, or both.

.. admonition:: See Also
   :class: seealso

   The :func:`~kdiagram.plot.probabilistic.plot_calibration_sharpness`
   diagram provides an alternative summary view. While this plot gives a
   single overall score (radius), the calibration-sharpness diagram
   plots the two components (calibration error on the angle, sharpness
   on the radius) separately, which can be useful for visualizing the
   trade-off more explicitly.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical theory behind the
Continuous Ranked Probability Score, please refer back to the main
:ref:`ug_plot_crps_comparison` section.


.. _gallery_plot_calibration_sharpness:

-------------------------------
Calibration-Sharpness Diagram
-------------------------------

The :func:`~kdiagram.plot.probabilistic.plot_calibration_sharpness`
function provides the ultimate summary for probabilistic model
selection. It distills the two most important, and often competing,
qualities of a forecast—**calibration** (reliability) and **sharpness**
(precision)—into a single, decision-oriented visualization. Each model
is represented by a single point, making it immediately clear which one
achieves the best overall balance.

Let's begin by understanding the components of this powerful summary plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the **calibration error** of the forecast,
     calculated using the Kolmogorov-Smirnov statistic on the PIT values.
     An angle of **0° is perfect calibration**, and the error increases as
     the angle approaches 90°.
   * **Radius (r):** Represents the **sharpness** of the forecast, measured
     as the average width of the prediction interval. A **smaller radius is
     better**, indicating a sharper, more precise forecast.
   * **Ideal Point:** The ideal forecast is located at the **center of the
     plot (the origin)**, as this represents the perfect combination of
     zero calibration error and zero interval width (perfect sharpness).
     The best real-world model is the one closest to this point.

With this framework, let's apply the plot to a final model selection
problem, showing how it can guide us to the best choice.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: The Three-Model Trade-off**

The most common use of this plot is to visualize the classic trade-offs
between different types of probabilistic models and select the most
balanced performer.

Let's return to our weather forecasting scenario. An agency has three
competing models for predicting temperature:

- **Model A (Balanced):** A well-regarded model that aims for a good compromise.
- **Model B (Sharp & Biased):** A newer, aggressive model that produces very 
  tight predictions but is suspected of being poorly calibrated.
- **Model C (Calibrated & Wide):** An older, conservative model that is 
  reliable but often produces impractically wide prediction intervals.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from scipy.stats import norm
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Three models with different trade-offs ---
   np.random.seed(42)
   n_samples = 2000
   y_true = np.random.normal(loc=15, scale=5, size=n_samples)
   quantiles = np.linspace(0.05, 0.95, 19)

   # Model A (Balanced)
   model_A = norm.ppf(quantiles, loc=y_true[:, np.newaxis], scale=5)
   # Model B (Sharp but biased/overconfident)
   model_B = norm.ppf(quantiles, loc=y_true[:, np.newaxis] - 1, scale=3)
   # Model C (Calibrated but wide/underconfident)
   model_C = norm.ppf(quantiles, loc=y_true[:, np.newaxis], scale=8)

   model_names = ["A (Balanced)", "B (Sharp/Biased)", "C (Calibrated/Wide)"]

   # --- 2. Plotting ---
   kd.plot_calibration_sharpness(
       y_true,
       model_A, model_B, model_C,
       quantiles=quantiles,
       names=model_names,
       cmap='plasma',
       title='Use Case: Model Selection Trade-off',
       savefig="gallery/images/gallery_calibration_sharpness_basic.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_calibration_sharpness_basic.png
   :align: center
   :width: 70%
   :alt: A calibration-sharpness diagram comparing three models.

   A polar plot showing three points, with the "Balanced" model being
   closest to the ideal point at the center of the plot.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a clear and decisive summary for model selection.
   The **"Balanced" model (A)** is positioned closest to the ideal
   point at the center, indicating it achieves the best overall
   compromise between low calibration error (small angle) and good
   sharpness (small radius).

   The plot also diagnoses the specific failings of the other models.
   **Model B** has the smallest radius, making it the **sharpest**, but
   its large angle shows it is **poorly calibrated**. Conversely,
   **Model C** has a very small angle, indicating excellent **calibration**,
   but its large radius means it suffers from poor sharpness. For most
   applications, the balanced model is the superior choice.

.. admonition:: See Also
   :class: seealso

   This diagram is the culminating plot of a probabilistic forecast
   evaluation. It synthesizes the information from the
   :func:`~kdiagram.plot.probabilistic.plot_pit_histogram` (which
   measures calibration) and the
   :func:`~kdiagram.plot.probabilistic.plot_polar_sharpness` plot
   into a single, decision-oriented graphic.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical theory behind calibration,
sharpness, and proper scoring rules, please refer back to the main
:ref:`ug_plot_calibration_sharpness` section.

.. _gallery_plot_credibility_bands:

------------------------
Polar Credibility Bands
------------------------

The :func:`~kdiagram.plot.probabilistic.plot_credibility_bands` function
is a  descriptive tool for understanding the **conditional
behavior** of a probabilistic forecast. It answers the question: "How do
my model's median prediction and its uncertainty change depending on a
specific feature, like the time of year or a categorical input?" By
binning the data based on this feature, it creates a clear picture of
the forecast's structure.

Let's begin by understanding the components of this diagnostic plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the binned values of the feature specified
     by ``theta_col``. If the feature is cyclical (e.g., month of the
     year), the plot wraps around seamlessly to show the full cycle.
   * **Radius (r):** Represents the **magnitude of the forecast value**.
   * **Central Line:** This solid black line shows the **average of the
     median (Q50) forecast** for each angular bin. Its position reveals
     the forecast's central tendency under each condition.
   * **Shaded Band:** The area between the **average of the lower and
     upper quantiles**. The width of this band directly visualizes the
     average forecast **sharpness** for each bin, making it an excellent
     tool for diagnosing heteroscedasticity.

Now, let's apply this plot to a real-world forecasting problem to see
how it uncovers conditional patterns.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Diagnosing Seasonal Uncertainty**

A primary use case for this plot is to analyze how a forecast's central
tendency and uncertainty evolve over a seasonal or cyclical period.

Let's simulate a forecast for monthly product sales. We expect both the
sales volume and the forecast uncertainty to follow a strong seasonal
pattern, with higher and more volatile sales during the holiday season.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Seasonal Sales Forecast ---
   np.random.seed(0)
   n_points = 1000
   # Simulate a cyclical feature (month of the year)
   month = np.random.randint(1, 13, n_points)
   # Forecast median follows a seasonal pattern (peaks in summer/winter)
   median_forecast = 50 + 25 * np.sin((month - 3) * np.pi / 6)
   # Uncertainty (interval width) is also seasonal (widest in winter)
   interval_width = 15 + 10 * np.cos(month * np.pi / 3)**2

   df_seasonal = pd.DataFrame({
       'month': month,
       'q50_sales': median_forecast + np.random.randn(n_points) * 3,
       'q10_sales': median_forecast - interval_width,
       'q90_sales': median_forecast + interval_width,
   })

   # --- 2. Plotting ---
   kd.plot_credibility_bands(
       df=df_seasonal,
       q_cols=('q10_sales', 'q50_sales', 'q90_sales'),
       theta_col='month',
       theta_period=12, # A 12-month cycle
       theta_bins=12,
       title="Use Case 1: Seasonal Sales Forecast Uncertainty",
       color="#8E44AD", # A nice purple
       savefig="gallery/images/gallery_credibility_bands_seasonal.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_credibility_bands_seasonal.png
   :align: center
   :width: 70%
   :alt: A polar credibility bands plot showing seasonal patterns.

   The central line (median) and the width of the shaded band both
   show a clear cyclical pattern as the angle (month) changes.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot perfectly visualizes the seasonal structure of the
   forecast. The **central black line** shows that the mean of the median
   sales forecast follows a distinct seasonal trend, peaking in the
   spring/summer (top-right) and hitting a low in the autumn/winter
   (bottom-left). More importantly, the **width of the shaded band is not
   constant**. It is narrowest during the summer months and becomes
   significantly wider during the winter months. This is a clear visual
   diagnosis of **heteroscedasticity**: the model correctly predicts that
   its forecast is much more uncertain during the volatile winter
   season.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Comparing Uncertainty Across Categories**

This plot is not limited to time-based features. It is an ideal
tool for comparing the forecast distribution across any set of discrete
categories.

Let's analyze a model that predicts shipping costs. We want to see if
the model's uncertainty is different for three distinct shipping
methods: "Air", "Sea", and "Ground".

.. code-block:: python
   :linenos:

   # --- 1. Data Generation: Shipping Cost by Method ---
   np.random.seed(42)
   n_points = 900
   # Assign each sample to a shipping method
   method_map = {0: 'Air', 1: 'Sea', 2: 'Ground'}
   shipping_method_code = np.random.randint(0, 3, n_points)
   shipping_method_name = [method_map[c] for c in shipping_method_code]

   # Define different uncertainty profiles for each method
   median_forecast = np.zeros(n_points)
   interval_width = np.zeros(n_points)
   median_forecast[shipping_method_code == 0] = 150 # Air is expensive
   median_forecast[shipping_method_code == 1] = 70  # Sea is mid-range
   median_forecast[shipping_method_code == 2] = 40  # Ground is cheap
   interval_width[shipping_method_code == 0] = 30 # Air is predictable
   interval_width[shipping_method_code == 1] = 50 # Sea is highly unpredictable
   interval_width[shipping_method_code == 2] = 10 # Ground is very predictable

   df_shipping = pd.DataFrame({
       'method_code': shipping_method_code,
       'q50_cost': median_forecast + np.random.randn(n_points),
       'q10_cost': median_forecast - interval_width,
       'q90_cost': median_forecast + interval_width,
   })

   # --- 2. Plotting ---
   # Note: Since theta_col is categorical, we don't set theta_period.
   # The plot will map the unique categories to the angular space.
   kd.plot_credibility_bands(
       df=df_shipping,
       q_cols=('q10_cost', 'q50_cost', 'q90_cost'),
       theta_col='method_code', # Bin by the numerical code
       theta_bins=3, # We have 3 categories
       title="Use Case 2: Shipping Cost Uncertainty by Method",
       savefig="gallery/images/gallery_credibility_bands_categorical.png",
   )
   plt.close()

.. figure:: ../images/probabilistic/gallery_credibility_bands_categorical.png
   :align: center
   :width: 70%
   :alt: A polar credibility bands plot comparing three categories.

   The plot is divided into three distinct angular sectors, one for each
   shipping method, each showing a different median value and
   interval width.

.. topic:: 🧠 Interpretation
   :class: hint

   This plot provides an excellent summary of the forecast distribution
   for each shipping category. We can immediately see three distinct
   regimes. The **"Air"** category has the highest average median cost (largest
   radius of the central line). The **"Sea"** category not only has a mid-range
   cost but also has by far the **widest credibility band**, indicating
   that its shipping costs are the most uncertain and difficult to
   predict. Finally, the **"Ground"** category has the lowest cost and a
   very narrow band, showing that its costs are highly predictable.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 3: Side-by-Side Comparison of Conditional Uncertainty**

A truly powerful application of this plot is to compare the conditional
uncertainty structures of two competing models. Does a newer, more
complex model produce more realistic uncertainty estimates under
different conditions than an older, simpler one? A side-by-side
comparison provides a clear and decisive answer.

.. admonition:: Best Practice
   :class: best-practice

   To compare two models, create a multi-panel figure using
   ``matplotlib.pyplot.subplots`` and then pass each ``ax`` object to
   ``plot_credibility_bands``. This is the recommended workflow for a
   direct, visual comparison of model behavior.

Let's tackle a common problem in energy forecasting: predicting solar
power output, where uncertainty is highly dependent on cloud cover.

.. admonition:: Practical Example

   A renewable energy company is evaluating a new AI-based model for
   forecasting solar power output against their older "Baseline Model".
   The baseline model has a known weakness: it assumes a constant level
   of uncertainty regardless of the weather. The new AI model is
   supposed to have learned that forecasts are much less certain on
   heavily overcast days.

   We will create a side-by-side credibility band plot, with both plots
   binned by cloud cover. The left panel will show the Baseline Model's
   naive uncertainty, and the right panel will show the AI Model's more
   sophisticated, condition-dependent uncertainty.

   .. code-block:: python
      :linenos:

      import kdiagram as kd
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt

      # --- 1. Data Generation: Solar Power Forecast ---
      np.random.seed(1)
      n_points = 2000
      # Cloud cover from 0% to 100%
      cloud_cover = np.random.uniform(0, 100, n_points)
      # Median forecast decreases with more clouds
      median_forecast = 150 * np.exp(-cloud_cover / 50) + np.random.normal(0, 5, n_points)

      df_solar = pd.DataFrame({'cloud_cover': cloud_cover, 'q50_ai': median_forecast, 'q50_baseline': median_forecast})

      # --- 2. Generate Predictions for Two Models ---
      # Baseline Model: constant, naive uncertainty
      width_baseline = np.ones(n_points) * 30
      df_solar['q10_baseline'] = df_solar['q50_baseline'] - width_baseline
      df_solar['q90_baseline'] = df_solar['q50_baseline'] + width_baseline

      # AI Model: uncertainty correctly grows with cloud cover
      width_ai = 10 + (df_solar['cloud_cover'] / 100) * 50
      df_solar['q10_ai'] = df_solar['q50_ai'] - width_ai
      df_solar['q90_ai'] = df_solar['q50_ai'] + width_ai

      # --- 3. Create a figure with two polar subplots ---
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   subplot_kw={'projection': 'polar'})

      # --- 4. Plot each model's diagnostic on its dedicated axis ---
      kd.plot_credibility_bands(
          df=df_solar, ax=ax1,
          q_cols=('q10_baseline', 'q50_baseline', 'q90_baseline'),
          theta_col='cloud_cover',
          theta_bins=10, # Bin cloud cover into 10 groups
          title='Baseline Model (Naive Uncertainty)',
          color='crimson'
      )
      kd.plot_credibility_bands(
          df=df_solar, ax=ax2,
          q_cols=('q10_ai', 'q50_ai', 'q90_ai'),
          theta_col='cloud_cover',
          theta_bins=10,
          title='AI Model (Conditional Uncertainty)',
          color='teal'
      )

      fig.suptitle('Use Case 3: Comparing Conditional Uncertainty Structures', fontsize=16)
      fig.tight_layout(rect=[0, 0.03, 1, 0.95])
      fig.savefig("gallery/images/gallery_credibility_bands_side_by_side.png")
      plt.close(fig)

.. figure:: ../images/probabilistic/gallery_credibility_bands_side_by_side.png
   :align: center
   :width: 90%
   :alt: Side-by-side credibility bands for a baseline and an AI model.

   A two-panel figure. The left plot (Baseline Model) shows a
   credibility band with a constant width. The right plot (AI Model)
   shows a band that is narrow for low cloud cover and wide for high
   cloud cover.

.. topic:: 🧠 Interpretation
   :class: hint

   The side-by-side comparison provides a verdict on the models'
   sophistication. The **Baseline Model** (left) displays a shaded
   credibility band that has a **constant width** at all angles (all
   levels of cloud cover). This visually confirms its critical weakness:
   it uses a naive, one-size-fits-all approach to uncertainty.

   In stark contrast, the **AI Model** (right) shows a credibility band
   that is **narrow** for low cloud cover (angles near 0°) and becomes
   progressively **wider** for high cloud cover (angles near 360°). This
   provides clear, data-driven evidence that the new AI Model is more
   robust and has successfully learned the realistic relationship
   between cloud cover and forecast uncertainty.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind conditional
distributions and heteroscedasticity, please refer back to the main
:ref:`ug_plot_credibility_bands` section.