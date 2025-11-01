.. _gallery_errors:

======================
Error Visualizations
======================

Diagnosing and understanding forecast errors is a critical step in
model evaluation. This gallery showcases specialized polar plots
from the `k-diagram` package designed to visualize different aspects
of model errors, from systemic biases to multi-dimensional uncertainty.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots (e.g.,
   ``images/gallery_plot_error_bands.png``).

.. _gallery_plot_error_bands:

----------------------
Polar Error Bands
----------------------

The :func:`~kdiagram.plot.errors.plot_error_bands` function is a
diagnostic tool designed to decompose a model's forecast error
into two fundamental components: **systemic error (bias)** and **random
error (variance)**. By aggregating errors as a function of a cyclical or
ordered feature (like the month of the year), it reveals conditional
patterns in a model's performance that a single error score would miss.

First, let's break down the components of this diagnostic plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the binned values of the feature from the
     ``theta_col`` (e.g., month, hour). This allows you to see how
     performance changes as this feature's value changes.
   * **Radius (r):** Represents the **magnitude of the forecast error**.
     The dashed red circle at a radius of 0 is the crucial "Zero Error"
     reference line.
   * **Mean Error (Black Line):** This line tracks the **average error**
     for each angular bin. A consistent deviation of this line from the
     zero-circle reveals a systemic bias.
   * **Shaded Band:** The width of this band is proportional to the
     **standard deviation** of the error in each bin. A wide band
     indicates high variance and inconsistent performance.

Now, let's apply this plot to a real-world problem to see how it can be
used to generate critical insights.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Basic Seasonal Error Analysis**

The most common application of this plot is to diagnose if a model's
performance changes with the seasons. A good forecast should be reliable
all year round, but many models struggle during specific periods.

Let's simulate a forecast where a model has a clear seasonal bias,
over-predicting in the summer and under-predicting in the winter, and is
also more inconsistent during the winter months.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: A forecast with seasonal error patterns ---
   np.random.seed(42)
   n_points = 2000
   day_of_year = np.arange(n_points) % 365
   month = (day_of_year // 30) + 1
   # Create a bias (positive error) in summer and more noise in winter
   seasonal_bias = np.sin((day_of_year - 90) * np.pi / 180) * 10
   seasonal_noise = 4 + 3 * np.cos(day_of_year * np.pi / 180)**2
   errors = seasonal_bias + np.random.normal(0, seasonal_noise, n_points)

   df_seasonal_errors = pd.DataFrame({'month': month, 'forecast_error': errors})

   # --- 2. Plotting ---
   kd.plot_error_bands(
       df=df_seasonal_errors,
       error_col='forecast_error',
       theta_col='month',
       theta_period=12,
       theta_bins=12,
       n_std=1.5,
       title='Use Case 1: Seasonal Forecast Error Analysis',
       color='#2980B9',
       alpha=0.25,
       savefig="gallery/images/gallery_plot_error_bands_basic.png"
   )
   plt.close()

.. figure:: ../images/errors/gallery_plot_error_bands_basic.png
   :align: center
   :width: 70%
   :alt: A polar error band plot showing clear seasonal patterns in bias and variance.

   A polar plot where the black line (mean error) oscillates around
   the red zero-error circle, and the blue shaded band (variance)
   changes width, indicating seasonal patterns.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot reveals two critical, seasonal patterns in the model's
   error. First, the **mean error line (black)** is not centered on the
   red "Zero Error" circle. It is clearly outside the circle (positive
   bias, or over-prediction) in the spring/summer months and inside the
   circle (negative bias, or under-prediction) in the autumn/winter.
   Second, the **width of the shaded band** is not constant; it is much
   wider during the winter months, indicating that the model's
   predictions are far more inconsistent and variable during that
   season.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Comparing Competing Models (Bias vs. Variance)**

A more advanced use case is to compare two competing models to understand
not just which is "better," but *how* they differ in their failure modes.
One model might be consistently wrong (biased), while another might be
right on average but highly unpredictable (high variance).

Let's consider a city's electricity provider evaluating two models for
forecasting energy demand. They need to know which model is more
reliable during the critical, high-demand afternoon hours.

.. code-block:: python
   :linenos:

   # --- 1. Data Generation: Two models with different error profiles ---
   np.random.seed(10)
   n_points = 5000
   hour = np.random.randint(0, 24, n_points)
   # Model A is consistently wrong (biased) in the afternoon but has low variance
   bias_A = np.where((hour >= 15) & (hour <= 19), 20, 0)
   error_A = bias_A + np.random.normal(0, 5, n_points)
   # Model B is right on average (unbiased) but highly inconsistent in the afternoon
   noise_B = np.where((hour >= 15) & (hour <= 19), 25, 5)
   error_B = np.random.normal(0, noise_B, n_points)

   df_model_A = pd.DataFrame({'hour': hour, 'error': error_A})
   df_model_B = pd.DataFrame({'hour': hour, 'error': error_B})

   # --- 2. Create side-by-side plots for comparison ---
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

   kd.plot_error_bands(
       df=df_model_A, ax=ax1, error_col='error', theta_col='hour',
       theta_period=24, theta_bins=24, title='Model A (Biased but Consistent)'
   )
   kd.plot_error_bands(
       df=df_model_B, ax=ax2, error_col='error', theta_col='hour',
       theta_period=24, theta_bins=24, title='Model B (Unbiased but Inconsistent)',
       color='darkgreen', alpha=0.2
   )
   fig.suptitle('Use Case 2: Comparing Model Error Profiles (Bias vs. Variance)', fontsize=16)
   fig.tight_layout(rect=[0, 0.03, 1, 0.95])
   fig.savefig("gallery/images/gallery_plot_error_bands_compare.png")
   plt.close(fig)


.. figure:: ../images/errors/gallery_plot_error_bands_compare.png
   :align: center
   :width: 90%
   :alt: Side-by-side error bands comparing a biased vs. an inconsistent model.

   Two plots showing different failure modes. The left plot shows a
   mean error line far from the center but a narrow band. The right
   plot shows a mean error line near the center but a very wide band.

.. topic:: 🧠 Interpretation
   :class: hint

   This side-by-side comparison reveals the distinct failure modes of
   the two models. **Model A (left)** is clearly **biased** during the
   afternoon peak (15:00-19:00), as its mean error line pushes far away
   from the red zero-error circle. However, its shaded band is narrow,
   indicating it is consistently wrong in a predictable way. In
   contrast, **Model B (right)** is **unbiased** on average—its mean error
   line stays close to the zero-error circle at all times. However, its
   shaded band becomes extremely wide during the afternoon, indicating
   it is **highly inconsistent and unreliable** during these critical
   hours. This analysis shows that neither model is perfect and the
   "best" choice depends on the business need: is it easier to correct a
   predictable bias or to manage unpredictable volatility?

.. admonition:: Best Practice
   :class: best-practice

   Use this plot not just to see *if* a model is wrong, but to
   understand *how* it is wrong. Distinguishing between a predictable
   systemic bias (which can sometimes be corrected with post-processing)
   and high random error (which indicates fundamental model
   instability) is crucial for effective model improvement.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind bias and
variance in forecasting, please refer back to the main
:ref:`ug_plot_error_bands` section.

.. _gallery_plot_error_violins:

---------------------
Polar Error Violins
---------------------

The :func:`~kdiagram.plot.errors.plot_error_violins` function provides a
rich, comparative view of the **full error distributions** for multiple
models. By adapting the traditional violin plot to a polar layout, it
allows for an immediate visual assessment of each model's bias,
variance, and overall error shape, making it a premier tool for model
selection.

First, let's break down how to interpret these informative shapes.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular sector is dedicated to a different
     **model** being compared. The angle itself is for separation and has
     no numeric meaning.
   * **Radius (r):** Represents the **forecast error value**. The dashed
     black circle at a radius of 0 is the "Zero Error" reference line.
   * **Violin Shape:** The **width** of the violin at any given radius
     shows the **probability density** of errors at that value. Wide
     sections indicate common error values, while narrow sections
     indicate rare ones. The overall shape reveals the error
     distribution's character (e.g., symmetric, skewed, etc.).

Now, let's apply this plot to a real-world model selection problem.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: The Classic Trade-off (Bias vs. Variance)**

The most common use of this plot is to visualize the classic trade-off
between a model that is consistently wrong (biased) and a model that is
right on average but highly unpredictable (high variance).

Let's imagine a financial firm has three models for predicting a stock's
price. They need to choose the one with the most desirable error
profile for their trading strategy.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Three models with different error profiles ---
   np.random.seed(0)
   n_points = 2000
   df_model_errors = pd.DataFrame({
       # Low bias and low variance
       'Error Model A': np.random.normal(loc=0.5, scale=1.5, size=n_points),
       # Strong negative bias
       'Error Model B': np.random.normal(loc=-4.0, scale=1.5, size=n_points),
       # Unbiased but high variance
       'Error Model C': np.random.normal(loc=0, scale=4.0, size=n_points),
   })

   # --- 2. Plotting ---
   kd.plot_error_violins(
       df_model_errors,
       'Error Model A', 'Error Model B', 'Error Model C',
       names=['A (Balanced)', 'B (Biased)', 'C (Inconsistent)'],
       title='Use Case 1: Comparing Model Error Distributions',
       cmap='plasma',
       savefig="gallery/images/gallery_plot_error_violins_basic.png"
   )
   plt.close()

.. figure:: ../images/errors/gallery_plot_error_violins_basic.png
   :align: center
   :width: 70%
   :alt: A polar violin plot comparing a good, a biased, and an inconsistent model.

   Three violins showing different error profiles: one is centered and
   narrow (good), one is shifted off-center (biased), and one is centered
   but very wide (inconsistent).

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a rich comparison that goes far beyond a single
   error metric. The violin for **Model A** is the best overall: it is
   narrow, indicating low variance (consistent errors), and its widest
   part is centered close to the "Zero Error" circle, indicating low
   bias. In contrast, **Model B** is clearly **biased**; its entire
   distribution is shifted to a negative error value, meaning it
   systematically under-predicts. Finally, **Model C** is **unbiased** on
   average (its distribution is centered on zero), but it is dangerously
   **inconsistent**. Its wide shape indicates a high variance, meaning it
   is prone to making very large errors in both directions. For most
   applications, Model A would be the superior choice.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Uncovering Skewed and Bimodal Error Distributions**

Standard error metrics like Mean Absolute Error assume that errors are
symmetrically distributed. However, this is often not the case. A model
might be prone to making very large errors in one direction but not the
other (skew) or have two different common types of errors (bimodality).
The violin plot is the perfect tool to diagnose these complex error shapes.

Let's simulate two new models for our stock prediction task:

- A model with **skewed error**: it rarely makes large positive errors 
  but is prone to "crash" predictions with large negative errors.
- A model with **bimodal error**: it is either very accurate (errors near zero)
  or very inaccurate, with few errors in between.

.. code-block:: python
   :linenos:

   # --- 1. Data Generation: Complex Error Distributions ---
   np.random.seed(42)
   # Skewed errors (e.g., from a log-normal distribution)
   skewed_errors = 5 - np.random.lognormal(mean=1, sigma=0.5, size=n_points)
   # Bimodal errors (mixture of two normal distributions)
   bimodal_errors = np.concatenate([
       np.random.normal(loc=-5, scale=1, size=n_points // 2),
       np.random.normal(loc=5, scale=1, size=n_points // 2)
   ])
   df_complex_errors = pd.DataFrame({
       'Skewed Model': skewed_errors,
       'Bimodal Model': bimodal_errors
   })

   # --- 2. Plotting ---
   kd.plot_error_violins(
       df_complex_errors,
       'Skewed Model', 'Bimodal Model',
       title='Use Case 2: Diagnosing Skewed and Bimodal Errors',
       cmap='viridis',
       savefig="gallery/images/gallery_plot_error_violins_complex.png"
   )

.. figure:: ../images/errors/gallery_plot_error_violins_complex.png
   :align: center
   :width: 70%
   :alt: A polar violin plot showing a skewed and a bimodal distribution.

   Two violins showing complex shapes. The "Skewed Model" violin has a
   long tail in one direction. The "Bimodal Model" violin has two distinct
   wide sections, with a narrow part in the middle.

.. topic:: 🧠 Interpretation
   :class: hint

   This plot reveals error structures that simple metrics would miss.
   The **"Skewed Model"** violin is clearly asymmetric. Its density is
   concentrated at small positive errors, but it has a long, thin tail
   extending to large negative errors. This indicates the model is
   prone to occasional, severe under-predictions. The **"Bimodal
   Model"** has two distinct "lobes" or wide sections, one at -5 and one
   at +5, with a very narrow section in the middle around the zero-error
   line. This is the signature of a model that is either very right or
   very wrong, with no middle ground—a critical insight for understanding
   its behavior.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 3: Reviewer-Inspired Overlay — a Two-Model Face-Off**

.. topic:: The Story Behind ``mode="cbueth"``
   :class: hint

   This plot's default mode has a special origin. During the
   peer review for the ``kdiagram`` JOSS paper, a reviewer
   (GitHub user **cbueth**) provided critical feedback. They
   noted that the original design (now ``mode="basic"``)
   could be difficult to interpret when comparing just two
   models, especially if their distributions were skewed.
   The reviewer asked: *"wouldn't it be easier to just plot
   them on top of each other with transparency?"*
   This single question inspired a complete redesign. The new
   mode splits the violin into positive and negative error
   lobes (to show bias/skew) and maps error *magnitude* to
   the radius. To honor this transformative suggestion, the new
   mode was named ``"cbueth"`` and made the default. This
   use case shows their exact suggestion in action.

This view implements ``mode="cbueth"`` by applying the reviewer's
suggestion directly. When you compare only a few models (here, k=2),
``overlay="auto"`` places them on a single spoke with transparency
so differences are visible at a glance. Positive and negative
errors form two lobes around the spoke (asymmetry ≈ skew). Summary
stats (median, skew) stay **in the legend**, and the zero-error
reference is a dot at the center.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data: two contrasting models ---

   np.random.seed(0)
   n_points = 2000
   df_two = pd.DataFrame({
   "Error Model A": np.random.normal(loc=0.5,  scale=1.5, size=n_points),
   "Error Model B": np.random.normal(loc=-4.0, scale=1.5, size=n_points),
   })

   # --- Plot: overlay with transparency; stats in legend ---

   kd.plot_error_violins(
     df_two,
     "Error Model A", "Error Model B",
     names=["A (Balanced)", "B (Biased)"],
     title="Two-Model Overlay",
     mode="cbueth",
     overlay="auto",         # overlay when k <= 2
     show_stats=True,        # (median, skew) in legend
     cmap="plasma",
     savefig="gallery/images/errors/gallery_plot_error_violins_cbueth_overlay.png",
   )
   plt.close()


.. figure:: ../images/errors/gallery_plot_error_violins_cbueth_overlay.png
   :align: center
   :width: 80%
   :alt: Two models overlaid on a single polar spoke with transparent violins.

   Two transparent violins share one spoke. The center dot marks zero
   error; the legend reports median and skew, keeping the plot uncluttered.

.. topic:: 🧠 Interpretation
   :class: hint

   Overlay makes local differences obvious. Here, **Model B** is clearly
   shifted (negative bias, confirmed by med=-4.02) while **Model A**
   stays closer to zero (med=0.48) with a tighter spread.
   Asymmetry of each lobe hints at skew; the legend confirms it
   numerically. Use this view when the audience needs a quick,
   direct comparison of a small set of models.

.. admonition:: Try it
   :class: tip

   * Rotate the shared spoke: ``overlay_angle=0`` (horizontal) or
     ``np.pi/2`` (vertical).
   * Smooth more/less with ``bw_method="scott"`` or a float like ``0.3``.
   * Force split-spokes by setting ``overlay=False`` (see Use Case 4).

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">margin: 30px 0;">

**Use Case 4: Three-Model Split-Spokes — Outside Labels, Clean Plot**

For 3+ models, ``mode="cbueth"`` switches to split-spokes. Model names
are drawn **outside the circle** to keep the plot readable; the legend
continues to carry compact statistics.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data: balanced, biased, high-variance ---

   np.random.seed(0)
   n_points = 2000
   df_three = pd.DataFrame({
      "Error Model A": np.random.normal(loc=0.5,  scale=1.5, size=n_points),
      "Error Model B": np.random.normal(loc=-4.0, scale=1.5, size=n_points),
      "Error Model C": np.random.normal(loc=0.0,  scale=4.0, size=n_points),
   })

   # --- Plot: split spokes + outside labels; stats in legend ---

   kd.plot_error_violins(
      df_three,
      "Error Model A", "Error Model B", "Error Model C",
      names=["A (Balanced)", "B (Biased)", "C (Inconsistent)"],
      title="Three-Model Comparison (cbueth split-spokes)",
      mode="cbueth",
      overlay=False,           # split spokes; model labels outside rim
      show_stats=True,
      colors = ["green", "red", "blue"], # take precedence over cmap
      cmap="viridis",
      savefig="gallery/images/errors/gallery_plot_error_violins_cbueth_split.png",
   )

.. figure:: ../images/errors/gallery_plot_error_violins_cbueth_split.png
   :align: center
   :width: 80%
   :alt: Three polar violins on separate spokes with model labels outside the rim.

   Each model occupies its own spoke; labels sit just outside the rim.
   The center dot is the zero-error reference; legend shows median and
   skew for quick comparison.

.. topic:: 🧠 Interpretation
   :class: hint

   **Model A** remains closest to zero with modest spread (good baseline).
   **Model B** is systematically negative (bias), and **Model C** is
   widest (variance risk). The two-lobe shapes still reveal skew

.. admonition:: Try it
   :class: tip

   * Toggle outside labels by switching ``overlay`` between ``False`` and
     ``"auto"``.
   * Compare palettes with ``cmap="plasma"`` or pass custom ``colors=...``.
   * Stress-test variance: increase ``scale`` for one model and observe
     the radial extent and lobe widths grow.
  
.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind
probability distributions and Kernel Density Estimation, please refer
back to the main :ref:`ug_plot_error_violins` section.

.. _gallery_plot_error_ellipses:

----------------------
Polar Error Ellipses
----------------------

The :func:`~kdiagram.plot.errors.plot_error_ellipses` function is a
specialized tool for visualizing **two-dimensional uncertainty**. In many
real-world problems, particularly in spatial or positional forecasting,
error is not a single number but has components in multiple
directions. This plot represents the uncertainty for each data point as
an ellipse, where the ellipse's size and shape reveal the magnitude and
directionality of the error.

Let's begin by understanding the components of this advanced plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the **mean angular position** of a data
     point, as specified by ``theta_col``. For cyclical data like
     degrees in a circle, this wraps around seamlessly when a
     ``theta_period`` (e.g., 360) is provided.
   * **Radius (r):** Represents the **mean radial position** of a data
     point, as specified by ``r_col``.
   * **Ellipse Shape:** The size and orientation of each ellipse are
     determined by the standard deviations in the radial (``r_std_col``)
     and tangential (``theta_std_col``) directions. A large, elongated
     ellipse indicates high and directional uncertainty, while a small,
     circular ellipse indicates low and uniform uncertainty.

Now, let's apply this plot to a real-world scientific problem where 2D
uncertainty is a critical factor.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Visualizing Positional Uncertainty in Tracking**

The primary application of this plot is in tracking problems, where the
goal is to predict an object's future position.

Imagine an air traffic control system that uses a model to predict the
position of aircraft. For each aircraft, the model outputs a predicted
location (distance and angle from the control tower) and an estimate of
the uncertainty in both of those dimensions. Visualizing this
uncertainty is critical for maintaining safe separation between aircraft.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   np.random.seed(1)
   n_aircraft = 20

   # Generate distance first so we can reuse it
   distance_km = np.random.uniform(20, 80, n_aircraft)

   # Std in degrees grows with distance; same length as distance_km
   angle_std_deg = 2 + np.random.uniform(5, 10, n_aircraft) * (distance_km / 80.0)

   df_tracking = pd.DataFrame({
       "angle_deg":     np.linspace(0, 360, n_aircraft, endpoint=False),
       "distance_km":   distance_km,
       "distance_std":  np.random.uniform(2, 4, n_aircraft),
       # convert to radians for the plotting function
       "angle_std_rad": np.deg2rad(angle_std_deg),
       "aircraft_type": np.random.randint(1, 4, n_aircraft),
   })

   kd.plot_error_ellipses(
       df=df_tracking,
       r_col="distance_km",
       theta_col="angle_deg",        # degrees are fine; function maps internally
       r_std_col="distance_std",
       theta_std_col="angle_std_rad",# radians expected
       color_col="aircraft_type",
       n_std=2.0,
       title="Use Case 1: 2-Sigma Positional Uncertainty for Aircraft",
       cmap="cividis",
       alpha=0.7,
       edgecolor="black",
       linewidth=0.5,
       savefig="gallery/images/gallery_plot_error_ellipses_basic.png"
   )
   plt.close()

.. figure:: ../images/errors/gallery_plot_error_ellipses_basic.png
   :align: center
   :width: 70%
   :alt: A polar plot with ellipses of different shapes and sizes.

   Each ellipse represents the 95% confidence region for a predicted
   aircraft position, revealing the magnitude and directionality of the
   uncertainty.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a rich, multi-faceted view of the model's
   positional uncertainty. Each ellipse represents the predicted 95%
   confidence region for an aircraft. We can see that the **shape and
   size** of the ellipses vary significantly. Some are nearly circular,
   indicating uniform uncertainty in all directions. Others are highly
   **elongated**, such as the ones at a large radius (far from the
   tower). This indicates that for distant aircraft, the uncertainty in
   their angular position is much greater than the uncertainty in their
   distance. The color, representing the aircraft type, could be used
   to see if this effect is stronger for certain types of planes.

.. admonition:: Best Practice
   :class: best-practice

   The ``n_std`` parameter is key to interpreting this plot correctly.
   Setting ``n_std=1.0``, ``n_std=2.0``, or ``n_std=3.0`` corresponds to
   visualizing the 68%, 95%, and 99.7% confidence regions, respectively 
   (assuming a normal error distribution). Choosing the appropriate
   level is crucial for risk assessment in your specific application.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the mathematical concepts behind
two-dimensional uncertainty and confidence ellipses, please refer back
to the main :ref:`ug_plot_error_ellipses` section.

.. _practical_app_error_evaluation:

--------------------------
Practical Application
--------------------------

While the sections above demonstrate each function in isolation, the
true power of ``k-diagram`` lies in using these tools together to conduct
a comprehensive, multi-faceted analysis. A thorough model evaluation
is not just about a single score; it's about building a deep
understanding of a model's behavior, its strengths, and its hidden
weaknesses.

This case study will walk you through a realistic workflow, showing how
the different plots from the ``errors`` module can be combined to move
from a high-level model comparison to a detailed, actionable diagnosis.

.. admonition:: Case Study: Selecting a Drone Navigation System
   :class: best-practice

   **The Business Problem:** A new logistics company, "AeroDeliver," is
   finalizing the design for its fleet of delivery drones. The most
   critical component is the navigation system responsible for the final
   landing phase. An accurate and reliable landing is paramount for safety
   and customer satisfaction.

   **The Models:** The engineering team is evaluating two competing systems:
   
   1. **"Standard GPS":** A reliable, cost-effective model based on traditional GPS.
   2. **"AI Vision":** A new, more expensive model that fuses GPS with computer
      vision to improve accuracy, especially under challenging conditions.

   **The Core Questions:** The team needs to answer three key questions
   to make a data-driven decision:
   
   1. Which model has the best **overall error profile** in terms of 
      bias and consistency?
   2. Does the performance of the chosen model degrade under specific, 
      predictable conditions, like the **time of day** (which affects 
      lighting for the vision system)?
   3. What is the precise **two-dimensional positional uncertainty** of the 
      final, chosen system for a safety and compliance report?

Let's use ``k-diagram`` to answer each of these questions in turn.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Step 1: Overall Performance Comparison with Error Violins**

Our first step is a high-level comparison. We need to look at the entire
distribution of landing errors for both models. A simple metric like
"average error" could be misleading. One model might have a zero average
error but make occasional, catastrophic mistakes. The polar violin plot
is the perfect tool for this initial, holistic comparison.

.. admonition:: Practical Example

   We will simulate the final landing error (in meters) for hundreds of
   test flights for both the "Standard GPS" and "AI Vision" systems.

   .. code-block:: python
      :linenos:

      import kdiagram as kd
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt

      # --- 1. Data Generation: Landing Errors ---
      np.random.seed(0)
      n_landings = 2000
      # The Standard GPS has a slight positive bias and moderate variance
      gps_errors = np.random.normal(loc=0.5, scale=1.0, size=n_landings)
      # The AI Vision model is unbiased and more consistent (lower variance)
      ai_vision_errors = np.random.normal(loc=0.0, scale=0.6, size=n_landings)

      df_errors = pd.DataFrame({
          'Standard GPS': gps_errors,
          'AI Vision': ai_vision_errors
      })

      # --- 2. Plotting ---
      kd.plot_error_violins(
          df_errors,
          'Standard GPS', 'AI Vision',
          names = ['Standard GPS', 'AI Vision'],
          title='Step 1: Overall Landing Error Comparison',
          savefig="gallery/images/casestudy_error_violins.png"
      )
      plt.close()

   .. figure:: ../images/errors/casestudy_error_violins.png
      :align: center
      :width: 70%
      :alt: A polar violin plot comparing the error distributions of two drone navigation models.

   **Quick Interpretation:**
    The violin plot provides a clear verdict on overall performance. The
    violin for the **"Standard GPS"** model is wider and its peak is
    visibly shifted slightly outside the "Zero Error" circle, indicating
    a small positive bias and higher variance. In contrast, the
    **"AI Vision"** model's violin is significantly narrower and is
    perfectly centered on zero. This indicates it is both **unbiased**
    and **more consistent**. Based on this initial analysis, the AI
    Vision model is the superior system.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Step 2: Diagnosing Conditional Performance with Error Bands**

Now that we've selected the AI Vision model as the better performer, we
need to stress-test it. We suspect that its performance might degrade at
dawn and dusk, when difficult lighting conditions could challenge the
computer vision algorithms. The polar error band plot is the ideal tool
to investigate if the model's error is conditional on the time of day.

.. admonition:: Practical Example

   We will simulate the AI model's landing errors across a full 24-hour
   cycle, introducing a slight degradation in performance (higher bias
   and variance) during sunrise (around 6:00) and sunset (around 18:00).

   .. code-block:: python
      :linenos:

      # --- 1. Data Generation: Time-Dependent Errors for the AI Model ---
      np.random.seed(42)
      n_landings = 3000
      hour_of_day = np.random.uniform(0, 24, n_landings)
      # Introduce a bias and higher noise during dawn (5-7) and dusk (17-19)
      is_twilight = ((hour_of_day > 5) & (hour_of_day < 7)) | ((hour_of_day > 17) & (hour_of_day < 19))
      bias = np.where(is_twilight, 0.3, 0)
      noise_scale = np.where(is_twilight, 1.0, 0.5)
      errors = bias + np.random.normal(0, noise_scale, n_landings)

      df_hourly_errors = pd.DataFrame({'hour': hour_of_day, 'landing_error': errors})

      # --- 2. Plotting ---
      kd.plot_error_bands(
          df=df_hourly_errors,
          error_col='landing_error',
          theta_col='hour',
          theta_period=24,
          theta_bins=24,
          n_std=2.0,
          title='Step 2: AI Model Error vs. Time of Day',
          savefig="gallery/images/casestudy_error_bands.png"
      )
      plt.close()

   .. figure:: ../images/errors/casestudy_error_bands.png
      :align: center
      :width: 70%
      :alt: A polar error band plot showing AI model error by hour of day.

   **Quick Interpretation:**
    This plot reveals a subtle but important conditional pattern. For
    most of the day, the black "Mean Error" line is flat on the red
    "Zero Error" circle, and the blue shaded band is very narrow,
    confirming the model's excellent performance. However, in the
    angular sectors corresponding to the **dawn and dusk hours**, the
    mean error line pushes slightly outwards, and the **shaded band becomes
    noticeably wider**. This is a critical finding: while the AI model
    is excellent overall, its performance degrades slightly in challenging
    lighting conditions, becoming both slightly biased and less consistent.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Step 3: Visualizing 2D Positional Uncertainty with Error Ellipses**

Finally, for the safety and compliance report, AeroDeliver needs to
provide a clear visualization of the AI model's 2D landing uncertainty.
The error isn't just one number; it has a North-South and an East-West
component. The polar error ellipse plot is designed to show exactly this.

.. admonition:: Practical Example

   We will simulate the predicted landing positions and the associated
   uncertainties (standard deviations) in both the radial (distance from
   target) and tangential (directional) axes for several landing sites.

   .. code-block:: python
      :linenos:

      # --- 1. Data Generation: 2D Positional Uncertainty ---
      np.random.seed(1)
      n_sites = 15
      df_tracking = pd.DataFrame({
          'angle_deg': np.linspace(0, 360, n_sites, endpoint=False),
          'distance_km': np.random.uniform(2, 8, n_sites),
          'distance_std_m': np.random.uniform(0.2, 0.8, n_sites), # Radial error in meters
          'angle_std_deg': np.random.uniform(0.5, 1.5, n_sites), # Angular error in degrees
          'site_priority': np.random.randint(1, 4, n_sites)
      })

      # --- 2. Plotting ---
      kd.plot_error_ellipses(
          df=df_tracking,
          r_col='distance_km',
          theta_col='angle_deg',
          r_std_col='distance_std_m',
          theta_std_col='angle_std_deg',
          color_col='site_priority',
          n_std=2.5, # Plot a 2.5-sigma (approx. 99%) confidence ellipse
          title='Step 3: 99% Confidence Landing Ellipses for AI Model',
          savefig="gallery/images/casestudy_error_ellipses.png"
      )
      plt.close()

   .. figure:: ../images/errors/casestudy_error_ellipses.png
      :align: center
      :width: 70%
      :alt: Polar error ellipses showing the 2D landing uncertainty for the AI model.

   **Quick Interpretation:**
    This final plot provides a clear, actionable summary of the AI
    model's spatial uncertainty. Each ellipse represents the 99%
    confidence region for a drone landing at a specific site. We can
    see that the ellipses are all small and nearly circular, indicating
    that the positional error is **low and uniform** in all directions.
    This visualization would be a key figure in a safety report, as it
    provides a clear and honest depiction of the system's expected
    landing precision.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind these
advanced error diagnostics, please refer back to the main
:ref:`userguide_errors` section.