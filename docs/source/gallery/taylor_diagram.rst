.. _gallery_taylor_diagram:

====================
Taylor Diagrams
====================

This gallery page focuses on Taylor Diagrams, which provide a concise
visual summary of model performance. They compare key statistics like
correlation, standard deviation, and Centered Root Mean Square Difference
(CRMSD) between one or more models (or predictions) and a reference
(observed) dataset.

A Taylor Diagram :footcite:p:`Taylor2001` conveys three key
statistics on a 2D plot, allowing for a dense, quantitative
comparison of model performance against a reference dataset (often
called "observations"). This geometric relationship is not just an 
analogy; it is a direct consequence of the mathematical definitions of
the statistics involved. The Law of Cosines provides the geometric framework to
visualize a fundamental statistical identity.

**Deriving the Relationship**

Let's start with the definition of the CRMSD. It is calculated on data 
where the means have been subtracted (denoted by primes: :math:`m'` and :math:`r'`).

1.  The squared CRMSD is the mean of the squared differences:

    .. math::

       CRMSD^2 = \frac{1}{N}\sum_{k=1}^{N} (m'_k - r'_k)^2

2.  Expanding the squared term :math:`(a-b)^2 = a^2 + b^2 - 2ab` gives:

    .. math::

       CRMSD^2 = \frac{1}{N}\sum(m'_k)^2 + \frac{1}{N}\sum(r'_k)^2 - \frac{2}{N}\sum m'_k r'_k

3.  We can recognize each term in this expansion:

    * :math:`\frac{1}{N}\sum(m'_k)^2` is the variance of the model (:math:`\sigma_{model}^2`).
    * :math:`\frac{1}{N}\sum(r'_k)^2` is the variance of the reference (:math:`\sigma_{ref}^2`).
    * :math:`\frac{1}{N}\sum m'_k r'_k` is the covariance between the model and reference.

4.  Substituting these back, we get:

    .. math::

       CRMSD^2 = \sigma_{model}^2 + \sigma_{ref}^2 - 2 \cdot \text{cov}(m, r)

5.  Finally, we use the definition of the correlation coefficient,
    :math:`R = \frac{\text{cov}(m, r)}{\sigma_{model} \sigma_{ref}}`.
    Rearranging this gives :math:`\text{cov}(m, r) = R \cdot \sigma_{model} \sigma_{ref}`.
    Substituting this into our equation yields the final relationship:

    .. math::
       :label: eq:taylor_cossine

       CRMSD^2 = \sigma_{model}^2 + \sigma_{ref}^2 - 2 \sigma_{model} \sigma_{ref} R

**The Connection to the Law of Cosines**

Now, compare this final equation to the Law of Cosines for a triangle
with sides *a*, *b*, *c*, and an angle :math:`\gamma` between sides
*a* and *b*:

.. math::

   c^2 = a^2 + b^2 - 2ab \cos(\gamma)

The two equations have the exact same form. By mapping the statistical
terms to the geometric terms, we get:

* Side *a* :math:`\rightarrow` :math:`\sigma_{ref}` (Standard deviation of reference)
* Side *b* :math:`\rightarrow` :math:`\sigma_{model}` (Standard deviation of model)
* Side *c* :math:`\rightarrow` **CRMSD**
* :math:`\cos(\gamma)` :math:`\rightarrow` **R** (Correlation Coefficient)

**From Mathematics to Visualization**

This remarkable equivalence is the conceptual foundation of the Taylor
Diagram. It allows us to take an abstract statistical relationship and
plot it in a simple, intuitive geometric space.

In the diagram, the standard deviations of the reference and the model
are plotted as two sides of a triangle originating from the origin. The
angle between them is set as :math:`\theta = \arccos(R)`. Because of
the Law of Cosines, the length of the third side of the triangle—the
line connecting the model point to the reference point—is guaranteed to
be equal to the CRMSD.

This transforms a complex, multi-metric evaluation into a simple visual
task: finding the point on the diagram that is closest to the reference.
See the full details in the user guide section: :ref:`userguide_taylor_diagram`.
Now, let's break down the components of this  diagram and its variants.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Reference Point:** A single point, typically plotted on the
     horizontal axis, that represents the "perfect" model. Its radial
     distance is the standard deviation of the reference data, and its
     correlation is, by definition, 1.0.
   * **Azimuthal Angle (θ):** The angle from the horizontal axis
     represents the **Pearson Correlation Coefficient** (*R*). A smaller
     angle indicates a higher correlation between the model and the
     reference. The axis is typically scaled with :math:`\arccos(R)`.
   * **Radial Distance (r):** The distance from the origin (the point
     (0,0)) represents the **Standard Deviation** (:math:`\sigma`) of the
     model's predictions. The plot often includes one or more circular
     arcs to show isocontours of standard deviation.
   * **Distance from Reference:** The distance between any model's point
     and the reference point represents the **Centered Root Mean Square
     Difference (CRMSD)**. This is a measure of the overall model
     skill; the closer a model's point is to the reference point, the
     better its performance.
   * **Model Points:** Each colored marker on the plot corresponds to a
     different model or prediction set, allowing for simultaneous
     evaluation.

With these concepts in mind, let's explore several practical applications
and gallery examples.


.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``images/gallery_taylor_diagram_rwf.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots (likely an ``images`` subdirectory relative to
   this file).
   

.. _gallery_plot_taylor_diagram_basic: 

-----------------------------
Taylor Diagram (Basic Plot)
-----------------------------

The :func:`~kdiagram.plot.taylor_diagram.plot_taylor_diagram` is a basic 
form. It is a more standard Taylor Diagram layout without
background shading, focusing purely on the positions of the model
points relative to the reference. Uses a half-circle layout (90
degrees, showing positive correlations only) with default West
orientation for Corr=1.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (reusing from previous example) ---
   np.random.seed(101)
   n_points = 150
   reference = np.random.normal(0, 1.0, n_points)
   pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
   pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
   pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)
   y_preds = [pred_a, pred_b, pred_c]
   names = ["Model A", "Model B", "Model C"]

   # --- Plotting ---
   kd.plot_taylor_diagram(
       *y_preds,
       reference=reference,
       names=names,
       acov='half_circle',      # Use 90-degree layout
       zero_location='W',       # Place Corr=1 at the Left (West)
       direction=-1,            # Clockwise angles
       title='Gallery: Basic Taylor Diagram (Half Circle)',
       # Save the plot (adjust path relative to this file)
       savefig="images/gallery_taylor_diagram_basic.png"
   )
   plt.close()

.. image:: ../images/gallery_taylor_diagram_basic.png
   :alt: Basic Taylor Diagram Example
   :align: center
   :width: 80%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This **basic Taylor Diagram** presents a clean comparison of model
   skill without background shading, using a 90-degree arc
   (``acov='half_circle'``) focused on positive correlations. Perfect
   correlation (1.0) is on the left (West axis, ``zero_location='W'``),
   and correlation decreases clockwise (``direction=-1``).

   **Analysis and Interpretation:**

   * **Reference Arc:** The red arc shows the standard deviation of
     the reference data (approx. 1.0).
   * **Model Positions:**
   
     * **Model A** (Red Dot): High correlation (small angle relative
       to West axis), standard deviation below the reference arc
       (~0.8). Underestimates variability.
     * **Model B** (Blue Dot): Lower correlation (larger angle),
       standard deviation above the reference arc (~1.2).
       Overestimates variability and has poorer pattern match.
     * **Model C** (Green Dot): Highest correlation (smallest angle),
       standard deviation almost exactly on the reference arc (~1.0).
       Best overall model in this comparison.
   * **RMSD:** Model C is closest to the reference point (at radius
     ~1.0 on the West axis), indicating the lowest centered RMS
     difference. Model B is furthest away.

   **💡 When to Use:**

   * Use this basic plot for a clear, uncluttered view focused purely
     on the standard deviation and correlation metrics.
   * Ideal when comparing many models where background shading might
     become too busy.
   * Suitable for publications preferring a standard, minimalist
     Taylor Diagram representation.
    

.. _gallery_plot_taylor_diagram_flexible: 

----------------------------------------------
Taylor Diagram (Flexible Input & Background)
----------------------------------------------

The :func:`~kdiagram.plot.taylor_diagram.taylor_diagram` is a variant 
of a series of Taylor diagrams implemented by ``k-diagram``. It
shows its flexibility by accepting raw data arrays and adding a
background colormap based on the 'rwf' (Radial Weighting Function)
strategy, emphasizing points with good correlation and reference-like
standard deviation.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(101)
   n_points = 150
   reference = np.random.normal(0, 1.0, n_points) # Ref std dev approx 1.0

   # Model A: High correlation, slightly lower std dev
   pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
   # Model B: Lower correlation, higher std dev
   pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
   # Model C: Good correlation, similar std dev
   pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)

   y_preds = [pred_a, pred_b, pred_c]
   names = ["Model A", "Model B", "Model C"]

   # --- Plotting ---
   kd.taylor_diagram(
       y_preds=y_preds,
       reference=reference,
       names=names,
       cmap='Blues',             # Add background shading
       radial_strategy='rwf',    # Use RWF strategy for background
       norm_c=True,              # Normalize background colors
       title='Gallery: Taylor Diagram (RWF Background)',
       # Save the plot (adjust path relative to this file)
       savefig="images/gallery_taylor_diagram_rwf.png"
   )
   plt.close()

.. image:: ../images/gallery_taylor_diagram_rwf.png
   :alt: Taylor Diagram with RWF Background Example
   :align: center
   :width: 80%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Taylor Diagram** summarizes model skill by plotting
   standard deviation (radius) vs. correlation (angle) relative
   to a reference (red marker/arc at reference std dev = 1.0,
   angle = 0). Points closer to the reference point indicate
   better overall performance (lower centered RMSD).

   This implementation uses the **Radial Weighting Function (RWF)**
   strategy for the background colormap (normalized blues).

   **Analysis and Interpretation:**

   * **Reference Point:** The red marker at radius ~1.0 on the
     horizontal axis represents the reference data's variability.
   * **Background (RWF):** Darker blue shades highlight regions
     with both high correlation (small angle) and standard
     deviation close to the reference (radius near 1.0).
   * **Model Performance:**

     * **Model A** (Red Dot): High correlation (~0.85), slightly
       low std dev (~0.8). Good pattern match, slightly low variability.
     * **Model B** (Blue Dot): Low correlation (~0.5), high std
       dev (~1.2). Poor pattern match and wrong variability.
     * **Model C** (Green Dot): Very high correlation (~0.95),
       std dev very close to reference (~1.0). Best overall fit,
       landing in the darkest blue region.

   **💡 When to Use:**

   * Use this plot (`taylor_diagram`) when you need flexibility:
     you can provide pre-calculated stats or raw data.
   * The background (`cmap` + `radial_strategy`) adds context.
     'rwf' specifically helps identify models that match both
     correlation *and* standard deviation well.
   * Ideal for comparing multiple models against observations in
     fields like climate science or hydrology.


.. _gallery_plot_taylor_diagram_background_shading_focus: 

-------------------------------------------
Taylor Diagram (Background Shading Focus)
-------------------------------------------

The :func:`~kdiagram.plot.taylor_diagram.plot_taylor_diagram_in` is an alternative 
plot with background shaing focus. It highlights the background colormap  feature, 
using the 'convergence' strategy where color intensity relates directly to the
correlation coefficient. It also demonstrates changing the plot
orientation (Corr=1 at North, angles increase counter-clockwise).

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (reusing from previous example) ---
   np.random.seed(101)
   n_points = 150
   reference = np.random.normal(0, 1.0, n_points)
   pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
   pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
   pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)
   y_preds = [pred_a, pred_b, pred_c]
   names = ["Model A", "Model B", "Model C"]

   # --- Plotting ---
   kd.plot_taylor_diagram_in(
       *y_preds,                     # Pass predictions as separate args
       reference=reference,
       names=names,
       radial_strategy='convergence',# Background color shows correlation
       cmap='viridis',
       zero_location='N',            # Place Corr=1 at the Top (North)
       direction=1,                  # Counter-clockwise angles
       cbar=True,                    # Show colorbar for correlation
       title='Gallery: Taylor Diagram (Correlation Background, N-oriented)',
       # Save the plot (adjust path relative to this file)
       savefig="images/gallery_taylor_diagram_in_conv.png"
   )
   plt.close()

.. image:: ../images/gallery_taylor_diagram_in_conv.png
   :alt: Taylor Diagram with Correlation Background Example
   :align: center
   :width: 80%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This version (`plot_taylor_diagram_in`) emphasizes the
   **background color map** and offers flexible **orientation**.
   Here, the background uses the `viridis` colormap with the
   `'convergence'` strategy, meaning color directly maps to the
   correlation value (yellow = high, purple = low). The plot is
   oriented with perfect correlation (1.0) at the top ('N').

   **Analysis and Interpretation:**

   * **Orientation:** Correlation decreases as the angle increases
     counter-clockwise from the top 'N' position. Standard
     deviation increases radially outwards. The red reference arc is
     at radius ~1.0.
   * **Background (Convergence):** The yellow region near the top
     indicates correlations close to 1.0. Colors shift towards
     green/blue/purple as correlation decreases (angle increases).
   * **Model Performance:**
   
     * **Model A** (Red Dot): Good correlation (in greenish-yellow
       zone), std dev slightly below reference arc.
     * **Model B** (Blue Dot): Low correlation (in blue/purple
       zone), std dev slightly above reference arc.
     * **Model C** (Green Dot): Excellent correlation (in bright
       yellow zone), std dev very close to reference arc.

   **💡 When to Use:**

   * Choose `plot_taylor_diagram_in` when you want a strong visual
     guide for correlation levels provided by the background shading.
   * Useful for presentations where the background color helps direct
     the audience's focus to high-correlation regions.
   * Use the orientation options (`zero_location`, `direction`) to
     match specific conventions or visual preferences.


.. _gallery_plot_taylor_diagram_in_variant1: 

-----------------------------------------------------
Taylor Diagram (NE Orientation, Convergence BG)
-----------------------------------------------------

Another variant using :func:`~kdiagram.plot.taylor_diagram.plot_taylor_diagram_in`,
this time placing perfect correlation (1.0) in the North-East ('NE')
quadrant, with angles increasing counter-clockwise (`direction=1`).
The background uses the 'convergence' strategy with the 'Purples'
colormap, where color intensity maps directly to the correlation
value, and includes a colorbar.

.. code-block:: python
   :linenos:

   import kdiagram.plot.evaluation as kde
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using same data as previous examples) ---
   np.random.seed(42) # Use same seed for consistency if desired
   reference = np.random.normal(0, 1, 100)
   y_preds = [
       reference + np.random.normal(0, 0.3, 100), # Model A (close)
       reference * 0.9 + np.random.normal(0, 0.8, 100) # Model B (worse corr/std)
   ]
   names = ['Model A', 'Model B']

   # --- Plotting ---
   kde.plot_taylor_diagram_in(
       *y_preds,
       reference=reference,
       names=names,
       acov='half_circle', # 90 degree span
       zero_location='NE', # Corr = 1.0 at North-East
       direction=1,        # Angles increase counter-clockwise
       fig_size=(8, 8),
       cbar=True,          # Show colorbar for correlation
       cmap='Purples',       # Use Purples colormap for background
       radial_strategy='convergence', # Color based on correlation
       title='Gallery: Taylor Diagram (NE, CCW, Convergence BG)',
       # Save the plot (adjust path relative to this file)
       savefig="images/gallery_taylor_diagram_in_ne_ccw_conv.png"
   )
   plt.close()

.. image:: ../images/gallery_taylor_diagram_in_ne_ccw_conv.png
   :alt: Taylor Diagram NE Orientation Convergence BG Example
   :align: center
   :width: 80%

.. topic:: 🧠 Analysis and Interpretation Note
    :class: hint

    Compare this plot's orientation to previous examples. Here, the
    point of perfect correlation (1.0) is at the top-right (45 degrees).
    The angles increase counter-clockwise, so points further "left"
    along the arc have lower correlation. The background color intensity
    directly reflects the correlation value based on the 'Purples' map.



.. _gallery_plot_taylor_diagram_in_variant2: 

------------------------------------------------------
Taylor Diagram (SW Orientation, Performance BG)
------------------------------------------------------

This variant uses :func:`~kdiagram.plot.taylor_diagram.plot_taylor_diagram_in`
with perfect correlation (1.0) placed in the South-West ('SW')
quadrant, counter-clockwise angle increase (`direction=1`), and the
'performance' background strategy. The 'performance' strategy uses an
exponential decay centered on the *best performing model* in the input
(closest correlation and std dev to reference), highlighting the region
around it. Uses 'gouraud' shading for a smoother background and hides
the colorbar.

.. code-block:: python
   :linenos:

   import kdiagram.plot.taylor_diagram as kde
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (using same data as previous examples) ---
   np.random.seed(42) # Use same seed for consistency
   reference = np.random.normal(0, 1, 100)
   y_preds = [
       reference + np.random.normal(0, 0.3, 100), # Model A (close)
       reference * 0.9 + np.random.normal(0, 0.8, 100) # Model B (worse corr/std)
   ]
   names = ['Model A', 'Model B']

   # --- Plotting ---
   kde.plot_taylor_diagram_in(
       *y_preds,
       reference=reference,
       names=names,
       acov='half_circle',     # 90 degree span
       zero_location='SW',     # Corr = 1.0 at South-West
       direction=1,            # Angles increase counter-clockwise
       fig_size=(8, 8),
       cbar=False,             # Hide colorbar
       cmap='twilight_shifted',# Use a cyclic map 
       shading='gouraud',      # Smoother shading
       radial_strategy='performance', # Color based on best model proximity
       title='Gallery: Taylor Diagram (SW, CCW, Performance BG)',
       # Save the plot (adjust path relative to this file)
       savefig="images/gallery_taylor_diagram_in_sw_ccw_perf.png"
   )
   plt.close()

.. image:: ../images/gallery_taylor_diagram_in_sw_ccw_perf.png
   :alt: Taylor Diagram SW Orientation Performance BG Example
   :align: center
   :width: 80%

.. topic:: 🧠 Analysis and Interpretation Note
    :class: hint

    Notice the different orientation with Corr=1.0 now at the bottom-left.
    The 'performance' background strategy creates a "hotspot" (brighter
    color with this cmap) centered around the best input model (Model A in
    this case), visually guiding the eye to the top performer relative
    to the provided dataset. 'gouraud' shading smooths the background
    colors.
    
.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">
   
.. _gallery_application_taylor_diagram:

---------------------------------------------------------------
Practical Application: Evaluating Climate Models
---------------------------------------------------------------
A primary and classic use of Taylor Diagrams is in climate science for
evaluating the performance of Global Climate Models (GCMs). A research
group might want to assess how well several competing GCMs reproduce
the historical seasonal cycle of surface temperatures for a critical
region, like the Amazon basin.

The diagram allows them to see, in a single glance, not just which model
is "best," but to diagnose the specific nature of each model's errors.
Does a model capture the timing of the seasons correctly (good
correlation) but get the magnitude wrong (incorrect standard
deviation)? Or does it have the right amount of variability but at the
wrong times?

Let's simulate this scenario with one set of observations and three
different models.

.. admonition:: Practical Example

   A team of climatologists is faced with a critical task: validating a new
   generation of Global Climate Models (GCMs). Before these computationally
   expensive models can be trusted to project future climate scenarios, they
   must first prove their ability to accurately reproduce the known climate
   of the past. A simple error score is insufficient; the team needs to
   understand the *nature* of each model's biases.
   The team chooses to focus on a critical and sensitive region: the Amazon
   basin. Their goal is to assess how well three competing GCMs simulate the
   historical seasonal cycle of monthly surface temperatures. The key
   scientific questions are:

   1.  Does the model correctly capture the **timing** of the seasons
       (the pattern, measured by correlation)?
   2.  Does the model correctly capture the **intensity** of the seasons
       (the magnitude of temperature swings, measured by standard
       deviation)?
   3.  Which model provides the best overall fidelity to the observed
       climate record?

A Taylor Diagram is the ideal tool for this multi-faceted evaluation,
as it can represent all three statistics in a single, concise plot.

To perform this comparative analysis, the team's workflow is simulated
in the following code. First, a synthetic "observed" temperature record
is created, representing the known seasonal cycle. Then, outputs from
three different models are generated, each with a distinct performance
profile: one that is well-calibrated, one that dampens the seasonal
swings, and one that exaggerates them. Finally, these datasets are
plotted on a Taylor Diagram.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Simulate Climate Data ---
   # Represents 20 years of monthly average temperatures
   n_points = 20 * 12
   time = np.linspace(0, 20 * 2 * np.pi, n_points)

   # Observed Data: A clear seasonal cycle with some natural noise
   observed_temps = 25 + 5 * np.sin(time) + np.random.normal(0, 0.5, n_points)

   # Model A (Good): Captures the pattern and magnitude well
   model_a = 25 + 4.8 * np.sin(time) + np.random.normal(0, 0.6, n_points)

   # Model B (Dampened): Underestimates the seasonal swings
   model_b = 25 + 2.5 * np.sin(time) + np.random.normal(0, 0.8, n_points)

   # Model C (Exaggerated): Overestimates the seasonal swings
   model_c = 25 + 6.5 * np.sin(time) + np.random.normal(0, 0.7, n_points)

   y_preds = [model_a, model_b, model_c]
   names = ["Model A (Good)", "Model B (Dampened)", "Model C (Exaggerated)"]
   
   fig, axes = plt.subplots(
       1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"}
   )

   # --- 2. Plotting ---
   kd.plot_taylor_diagram(
       *y_preds,
       reference=observed_temps,
       names=names,
       acov='half_circle',
       zero_location='E',
       direction=-1,
       ax=axes[0],        # <- draw on ax1   
       # title='Climate Model Evaluation: Amazon Seasonal Temperature Cycle',
       savefig="images/gallery_taylor_diagram_climate.png"
   )
   # Right: shaded background + colorbar
   kd.plot_taylor_diagram_in(
       *y_preds,
       reference=obs,
       names=names,
       acov="half_circle",
       zero_location="E",
       direction=-1,
       radial_strategy="performance",
       cmap="magma",
       norm_c=True,
       cbar="on",
       title="Taylor Diagram (Background Field)",
       ax=axes[1],          # <- draw into right axes
   )
   # Global title with safe top margin
   fig.suptitle(
       "Climate Model Evaluation: Amazon Seasonal Temperature Cycle",
       y=1.0, fontsize=14
   )


   plt.close()

.. figure:: ../images/gallery_taylor_diagram_climate.png
   :align: center
   :width: 80%
   :alt: A Taylor Diagram used to evaluate three climate models.

   Taylor Diagram comparing the performance of three simulated climate
   models against observed seasonal temperature data for the Amazon.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The Taylor Diagrams provide a visual verdict on each model's
   performance, allowing for a clear diagnosis of their specific
   strengths and weaknesses. The left panel shows the standard plot, while
   the right panel adds a background field where brightness corresponds
   to higher correlation, visually guiding the eye to the best-performing
   regions.

   * **Reference (Ground Truth):** The red arc represents the "ground
     truth"—the standard deviation of the observed historical
     temperatures (approx. 2.5). The ideal model would lie exactly where
     this arc intersects the horizontal axis.
   * **Model A (Good):** This point is the clear winner. It sits closest to
     the reference point, indicating the lowest overall error (CRMSD). Its
     position reveals a **near-perfect correlation (R > 0.98)**, meaning
     it correctly captures the *phenology* (timing) of the seasons. Its
     radial distance is almost exactly on the red arc, showing it also
     reproduces the correct **amplitude** (intensity) of temperature
     swings. In the right-hand plot, it falls squarely in the brightest
     "hotspot," visually confirming its superior performance.
   * **Model B (Dampened):** This model is diagnosed as "sluggish." It has
     a significantly lower correlation **(R ≈ 0.8)** and its radial distance
     **(Std. Dev. ≈ 1.5)** is far inside the reference arc. This tells
     us the model fails on two fronts: it struggles with the seasonal
     timing and **severely underestimates climate variability**.
   * **Model C (Exaggerated):** This model is "overly sensitive." It
     achieves a high correlation **(R ≈ 0.95)**, correctly simulating the
     seasonal *timing*. However, its point lies far outside the reference
     arc **(Std. Dev. ≈ 3.5)**, indicating its **variability is too high**.
     The model exaggerates the seasonal cycle, a significant bias that
     could stem from flawed feedback mechanisms.

   In conclusion, the diagram provides a definitive report. The team can
   confidently select **Model A** for future projections. More
   importantly, they can provide specific, actionable feedback to the
   developers of the other models: Model B's core issue is its weak
   amplitude and poor timing, while Model C's primary flaw is its
   excessive amplitude.
   
.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind these
diagrams, please refer back to the main
:ref:`userguide_taylor_diagram` section.

.. raw:: html

   <hr>
   
.. rubric:: References

.. footbibliography::