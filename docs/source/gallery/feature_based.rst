.. _gallery_feature_based:

========================================
Feature-Based Visualization Gallery
========================================

This gallery page showcases plots from `k-diagram` focused on
understanding feature influence and importance. Currently, it features
the Feature Importance Fingerprint plot.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``images/gallery_feature_fingerprint.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots (likely an ``images`` subdirectory relative to
   this file).

.. _gallery_plot_feature_fingerprint: 

--------------------------------
Feature Importance Fingerprint
--------------------------------

Uses :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`.
This radar chart compares the importance profiles ("fingerprints") of
several features across different groups or layers (e.g., different
years or models). This example shows raw (unnormalized) importance
values comparing feature influence across three years.

.. code-block:: python
   :linenos:

   # Assuming plot function is in kd.plot.feature_based
   import kdiagram.plot.feature_based as kdf
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   features = ['Rainfall', 'Temperature', 'Wind Speed',
               'Soil Moisture', 'Solar Radiation', 'Topography']
   n_features = len(features)
   years = ['2022', '2023', '2024']
   n_layers = len(years)

   # Generate importance scores (rows=years, cols=features)
   # Make them slightly different per year
   np.random.seed(123)
   importances = np.random.rand(n_layers, n_features) * 0.5
   importances[0, 0] = 0.8 # Rainfall important in 2022
   importances[1, 3] = 0.9 # Soil Moisture important in 2023
   importances[2, 1] = 0.7 # Temperature important in 2024
   importances[2, 4] = 0.75# Solar Radiation also important in 2024

   # --- Plotting ---
   kdf.plot_feature_fingerprint(
       importances=importances,
       features=features,
       labels=years,
       normalize=False, # Show raw importance scores
       fill=True,
       cmap='Pastel1',
       title="Gallery: Feature Importance Fingerprint (Yearly)",
       # Save the plot relative to this file's location
       savefig="images/gallery_feature_fingerprint.png"
   )
   plt.close()

.. image:: ../images/gallery_feature_fingerprint.png
   :alt: Feature Importance Fingerprint Plot Example
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Feature Importance Fingerprint** (a radar plot) visually
   represents the importance of various features across different
   groups or "layers". Each axis corresponds to a feature, and each
   colored polygon represents a layer (here, different years). The
   distance from the center along an axis indicates that feature's
   importance for that specific layer.

   **Analysis and Interpretation:**

   * **Axes:** Represent Rainfall, Temperature, Wind Speed, Soil
     Moisture, Solar Radiation, and Topography.
   * **Layers (Colors/Polygons):** Represent the years 2022, 2023,
     and 2024, showing how feature importance changes annually.
   * **Radius:** Since ``normalize=False``, the radius shows the
     *raw* importance score. Larger extensions along an axis mean
     higher importance.
   * **Shape ("Fingerprint"):** The overall shape of each polygon
     gives a unique "fingerprint" of feature influence for that year.

   **🔍 Key Insights from this Example:**

   * **2022:** The polygon extends furthest along the **Rainfall**
     axis, indicating it was the dominant feature in that year's
     model or context.
   * **2023:** The **Soil Moisture** axis shows the largest value,
     suggesting a shift in primary drivers compared to 2022.
   * **2024:** **Temperature** and **Solar Radiation** show the
     highest importance, indicating another change in the factors
     influencing the outcome for this year.
   * **Comparison:** We can easily see that the relative importance
     of features is not static but changes from year to year.

   **💡 When to Use This Plot:**

   * **Compare Feature Importance:** Visualize differences between
     models, time periods, or groups (e.g., spatial zones).
   * **Identify Dominant Features:** Quickly see which features have
     the most impact for each layer.
   * **Analyze Importance Drift:** Track how feature influence evolves
     over time, as shown in this yearly example.
   * **Model Interpretation:** Understand and communicate the key
     drivers behind model predictions for different scenarios.