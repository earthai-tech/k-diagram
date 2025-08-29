.. _gallery_taylor_diagram:

====================
Taylor Diagrams
====================

This gallery page focuses on Taylor Diagrams, which provide a concise
visual summary of model performance. They compare key statistics like
correlation, standard deviation, and centered Root Mean Square Difference
(RMSD) between one or more models (or predictions) and a reference
(observed) dataset.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``images/gallery_taylor_diagram_rwf.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots (likely an ``images`` subdirectory relative to
   this file).


.. _gallery_plot_taylor_diagram_flexible: 

----------------------------------------------
Taylor Diagram (Flexible Input & Background)
----------------------------------------------

Uses :func:`~kdiagram.plot.evaluation.taylor_diagram`. This example
shows its flexibility by accepting raw data arrays and adding a
background colormap based on the 'rwf' (Radial Weighting Function)
strategy, emphasizing points with good correlation and reference-like
standard deviation.

.. code-block:: python
   :linenos:

   # Assuming plot functions are in kd.plot.evaluation
   import kdiagram.plot.taylor_diagram as kde
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
   kde.taylor_diagram(
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

.. raw:: html

    <hr>

.. _gallery_plot_taylor_diagram_background_shading_focus: 

-------------------------------------------
Taylor Diagram (Background Shading Focus)
-------------------------------------------

Uses :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`. This
example highlights the background colormap feature, here using the
'convergence' strategy where color intensity relates directly to the
correlation coefficient. It also demonstrates changing the plot
orientation (Corr=1 at North, angles increase counter-clockwise).

.. code-block:: python
   :linenos:

   import kdiagram.plot.taylor_diagram as kde
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
   kde.plot_taylor_diagram_in(
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

.. raw:: html

    <hr>


.. _gallery_plot_taylor_diagram_basic: 

-----------------------------
Taylor Diagram (Basic Plot)
-----------------------------

Uses :func:`~kdiagram.plot.evaluation.plot_taylor_diagram`. This
example shows a more standard Taylor Diagram layout without
background shading, focusing purely on the positions of the model
points relative to the reference. Uses a half-circle layout (90
degrees, showing positive correlations only) with default West
orientation for Corr=1.

.. code-block:: python
   :linenos:

   import kdiagram.plot.taylor_diagram as kde
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
   kde.plot_taylor_diagram(
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
    

.. raw:: html

    <hr> 

.. _gallery_plot_taylor_diagram_in_variant1: 

-----------------------------------------------------
Taylor Diagram (NE Orientation, Convergence BG)
-----------------------------------------------------

Another variant using :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`,
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


.. raw:: html

    <hr>

.. _gallery_plot_taylor_diagram_in_variant2: 

------------------------------------------------------
Taylor Diagram (SW Orientation, Performance BG)
------------------------------------------------------

This variant uses :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`
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

   <hr>