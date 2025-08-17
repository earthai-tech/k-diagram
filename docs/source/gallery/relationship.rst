.. _gallery_relationship:

============================
Relationship Visualization
============================

This gallery page showcases the ``plot_relationship`` function, which
provides a unique polar perspective on the relationship between true
observed values and model predictions.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots.

.. _gallery_plot_relationship:

----------------------------------
True vs. Predicted Relationship
----------------------------------

Uses :func:`~kdiagram.plot.relationship.plot_relationship` to map
true values to the angular axis and normalized predicted values to the
radial axis (:cite:t:`kouadiob2025`). This creates a spiral-like plot that 
reveals the consistency and correlation of model predictions across the entire
range of true values.

.. code-block:: python
   :linenos:

   import kdiagram.plot.relationship as kdr
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   n_points = 150
   # Create a clear, non-linear true signal
   y_true = np.linspace(0, 10, n_points)**1.5 + np.sin(
       np.linspace(0, 10, n_points)
   ) * 2

   # Model 1: Good fit with some noise
   y_pred1 = y_true + np.random.normal(0, 1.5, n_points)
   # Model 2: Worse fit, under-predicts high values
   y_pred2 = y_true * 0.8 + np.random.normal(0, 2.5, n_points)

   # --- Plotting ---
   kdr.plot_relationship(
       y_true,
       y_pred1,
       y_pred2,
       names=["Good Model", "Biased Model"],
       title="Gallery: True vs. Predicted Relationship",
       theta_scale="proportional",  # Map angle to y_true value
       acov="default",
       s=40,
       # Save the plot (adjust path relative to this file)
       savefig="gallery/images/gallery_plot_relationship.png",
   )
   plt.close()

.. image:: ../images/gallery_plot_relationship.png
   :alt: Example of a Polar Relationship Plot
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Relationship Plot** offers a novel way to visualize the
   correlation between true values and model predictions, moving beyond
   a standard Cartesian scatter plot.

   **Key Features:**

   * **Angle (θ):** The angular position is directly proportional to the
     **true value** (``y_true``). The plot spirals outwards from the
     lowest true value to the highest.
   * **Radius (r):** The radial distance is the **normalized predicted
     value** (``y_pred``), scaled to the range [0, 1].
   * **Points:** Each point represents a single sample. Different
     colors distinguish between different models.

   **🔍 In this Example:**

   * **Good Model (Blue):** The blue points form a relatively tight,
     consistent spiral. As the angle increases (meaning ``y_true``
     increases), the radius also tends to increase, showing a strong
     positive correlation. The scatter around the spiral path represents
     the prediction noise.
   * **Biased Model (Orange):** The orange points are more scattered and
     form a less defined spiral. Critically, at larger angles (higher
     true values), the orange points are consistently at a smaller
     radius than the blue points, visually demonstrating the model's
     tendency to under-predict high values.

   **💡 When to Use:**

   * To get an intuitive feel for the correlation and consistency of a
     model's predictions across the entire data range.
   * To visually compare the performance of multiple models. A "tighter"
     spiral indicates a better, more consistent model.
   * To identify non-linear biases, where a model might perform well for
     low values but poorly for high values (or vice versa).


.. raw:: html

   <hr>
   
.. rubric:: References

.. bibliography::
   :style: plain
   :filter: cited