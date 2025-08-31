.. _userguide_feature_based:

==================================
Feature Importance Visualization
==================================

Understanding which input features most significantly influence a model's
predictions is crucial for interpretation, debugging, and building
trust in forecasting models. While overall importance scores are useful,
visualizing how these importances compare across different contexts
(e.g., different models, time periods, spatial regions) can reveal
deeper insights :footcite:p:`Lim2021, scikit-learn`.

``k-diagram`` provides a specialized radar chart, the "Feature
Fingerprint," to effectively visualize and compare these multi-
dimensional feature importance profiles.

Summary of Feature-Based Functions
-------------------------------------

This section focuses on functions for visualizing how model
predictions and performance are influenced by input features, either
individually or in combination.

.. list-table:: Feature-Based Visualization Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`
     - Creates a radar chart comparing feature importance profiles
       across different groups or layers.
   * - :func:`~kdiagram.plot.feature_based.plot_feature_interaction`
     - Creates a polar heatmap to visualize how a target variable is
       affected by the interaction between two features.
       
       
Detailed Explanations
-----------------------

Let's explore the feature based plots.

.. _ug_feature_fingerprint:

Feature Importance Fingerprint (:func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function generates a polar radar chart designed to visually
compare the importance or contribution profiles of multiple features
across different groups, conditions, or models (referred to as "layers").
Each layer is represented by a distinct colored polygon on the chart,
creating a unique "fingerprint" of feature influence for that layer
:footcite:p:`kouadiob2025`. It allows for easy identification of dominant
features, relative-importance patterns, and shifts in influence across
the layers being compared. When feature scores originate from model-
agnostic tools (e.g., permutation importance) or model-specific methods
(e.g., gradient/attention based for TFT), the fingerprint helps synthesize
those signals into a single comparative view :footcite:p:`Lim2021, scikit-learn`.

**Mathematical Concept:**
Let :math:`\mathbf{R}` be the input `importances` matrix of shape
:math:`(M, N)`, where :math:`M` is the number of layers and :math:`N`
is the number of features.

1.  **Angle Assignment:** Each feature :math:`j` is assigned an axis on
    the radar chart at an evenly spaced angle:
    
    .. math::
        \theta_j = \frac{2 \pi j}{N}, \quad j = 0, 1, \dots, N-1

2.  **Radial Value (Importance):** For each layer :math:`i` and feature
    :math:`j`, the radial distance :math:`r_{ij}` represents the
    importance value from the input matrix :math:`\mathbf{R}`.

3.  **Normalization (Optional):** If ``normalize=True``, the importances
    within each layer (row) :math:`i` are scaled independently to the
    range [0, 1]:
    
    .. math::
        r'_{ij} = \frac{r_{ij}}{\max_{k}(r_{ik})}
        
    If the maximum importance in a layer is zero or less, the normalized
    values for that layer are set to zero. The radius plotted is then
    :math:`r'_{ij}`. If ``normalize=False``, the raw radius :math:`r_{ij}`
    is used.

4.  **Plotting:** Points :math:`(r, \theta)` are plotted for each feature
    and connected to form a polygon for each layer. The shape is closed
    by connecting the last feature's point back to the first. The area
    can optionally be filled (``fill=True``).

**Interpretation:**

* **Axes:** Each angular axis corresponds to a specific input feature.
* **Polygons (Layers):** Each colored polygon represents a different
  layer (e.g., Model A vs. Model B, or Zone 1 vs. Zone 2).
* **Radius:** The distance from the center along a feature's axis
  indicates the importance of that feature for a given layer.
* **Shape (Normalized View):** When ``normalize=True``, compare the
  *shapes* of the polygons. This highlights the *relative* importance
  patterns. Which features are *most* important within each layer,
  regardless of overall magnitude? Do different layers rely on vastly
  different feature subsets?
* **Size (Raw View):** When ``normalize=False``, compare the overall
  *size* of the polygons. A larger polygon indicates that the layer
  generally assigns higher absolute importance scores across features
  compared to a smaller polygon (though interpretation depends on the
  nature of the importance metric).
* **Dominant Features:** Features corresponding to axes where polygons
  extend furthest are the most influential for those respective layers.

**Use Cases:**

* **Comparing Model Interpretations:** Visualize and contrast feature
  importance derived from different model types (e.g., Random Forest vs.
  Gradient Boosting) trained on the same data.
* **Analyzing Importance Drift:** Plot importance profiles calculated
  for different time periods or spatial regions to see if feature
  influence changes.
* **Identifying Characteristic Fingerprints:** Understand the typical
  pattern of feature reliance for a specific system or model setup.
* **Debugging and Validation:** Check if the feature importance profile
  aligns with domain knowledge or expectations.

**Advantages (Polar/Radar Context):**

* Excellent for simultaneously comparing multiple multi-dimensional
  profiles (feature importance vectors) against a common set of axes
  (features).
* The closed polygon shape provides a distinct visual "fingerprint" for
  each layer.
* Makes it easy to spot the most dominant features (those axes with the
  largest radial values) for each layer.
* Normalization allows comparing relative patterns effectively, even if
  absolute importance scales differ significantly between layers.


Understanding which features a model relies on is a cornerstone of
interpretation and trust. While a simple bar chart can show feature
importance for a single model, the real insights often come from
comparing these patterns across different models or contexts. This
"Feature Fingerprint" plot is designed for exactly that kind of
comparative analysis.

.. admonition:: Practical Example

   A telecommunications company has two models competing to predict
   customer churn: a classic ``Logistic Regression`` model and a more
   complex ``Gradient Boosting`` model. To trust and deploy one of
   them, the company needs to understand their decision-making
   processes. Which features does each model consider most important?
   Do they rely on the same information, or do they have fundamentally
   different "views" of the problem?

   This plot will create a unique "fingerprint" for each model,
   visualizing their feature importance profiles on the same set of
   axes for a direct comparison.

   .. code-block:: pycon

      >>> import numpy as np
      >>> import kdiagram as kd
      >>>
      >>> # --- 1. Define feature names and model importance scores ---
      >>> features = [
      ...     'tenure', 'monthly_charges', 'total_charges',
      ...     'data_usage', 'support_calls', 'contract_type'
      ... ]
      >>> labels = ['Logistic Regression', 'Gradient Boosting']
      >>>
      >>> # Logistic Regression relies heavily on a few key features
      >>> logreg_importances = [0.8, 0.9, 0.7, 0.1, 0.2, 0.6]
      >>> # Gradient Boosting uses a wider range of features
      >>> boosting_importances = [0.5, 0.6, 0.6, 0.8, 0.7, 0.4]
      >>>
      >>> importances = np.array([logreg_importances, boosting_importances])
      >>>
      >>> # --- 2. Generate the plot ---
      >>> ax = kd.plot_feature_fingerprint(
      ...     importances,
      ...     features=features,
      ...     labels=labels,
      ...     title='Churn Model Feature Importance Fingerprints'
      ... )

   .. figure:: ../images/userguide_plot_feature_fingerprint.png
      :align: center
      :width: 80%
      :alt: A feature fingerprint radar chart comparing two models.

      A polar radar chart comparing the feature importance profiles
      ("fingerprints") of a Logistic Regression and a Gradient
      Boosting model for customer churn prediction.

   This plot allows for an immediate visual comparison of the models'
   internal logic. By comparing the shapes of the colored polygons, we
   can see which features dominate each model's decision-making.

   **Quick Interpretation:**
    The plot reveals the distinctly different "fingerprints" of the two
    models. The ``Logistic Regression`` model (blue) has a spiky
    profile, indicating it relies heavily on a few core features like
    ``tenure``, ``monthly_charges``, and ``total_charges``, while paying
    little attention to others. In contrast, the ``Gradient Boosting``
    model (brown) displays a more well-rounded fingerprint, showing
    that it has learned to incorporate a wider array of information,
    assigning significant importance to features like ``data_usage`` and
    ``support_calls`` as well.

This ability to compare feature importance profiles is crucial for
model selection, debugging, and ensuring alignment with domain
knowledge. To see the full implementation, please explore the gallery.

**Example:**
See the gallery example and code: :ref:`gallery_plot_feature_fingerprint`.

.. raw:: html

   <hr>

.. _ug_plot_feature_interaction:

Feature Interaction Plot (:func:`~kdiagram.plot.feature_based.plot_feature_interaction`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Feature Interaction Plot** to
visualize how a target variable is affected by the interaction
between two features. It is a powerful diagnostic tool for moving
beyond one-dimensional feature importance to understand complex,
non-linear relationships that a model may have learned.


**Mathematical Concept:**
This plot is a polar heatmap, a novel visualization method
developed as part of the analytics framework
:footcite:p:`kouadiob2025`. It displays the conditional expectation
of a target variable, :math:`z`, given the values of two
features, one mapped to an angular coordinate, :math:`\theta`, and
the other to a radial coordinate, :math:`r`.

1.  **Coordinate Mapping and Binning**: The 2D feature space is
    first mapped to polar coordinates. The data is then
    partitioned into a grid of :math:`K_r \times K_{\theta}` polar
    bins, where :math:`K_r` is ``r_bins`` and :math:`K_{\theta}` is
    ``theta_bins``.

2.  **Aggregation**: For each bin, :math:`B_{ij}`, which corresponds
    to a specific range of values for ``r_col`` and ``theta_col``,
    an aggregate statistic (e.g., the mean) of the target
    variable, ``color_col`` (:math:`z`), is computed.

    .. math::
       :label: eq:feature_interaction_agg

       C_{ij} = \text{statistic}(\{z_k \mid (r_k, \theta_k) \in B_{ij}\})

    The resulting value, :math:`C_{ij}`, determines the color of
    the corresponding polar sector on the heatmap.


**Interpretation:**
The plot reveals how the two features jointly influence the target.

* **Angle (θ)**: Represents the first feature (``theta_col``). If
  the feature is cyclical (like the hour of the day), the plot
  will wrap around seamlessly.
* **Radius (r)**: Represents the second feature (``r_col``), with
  lower values near the center and higher values at the edge.
* **Color**: The color of each polar sector shows the average
  value of the target variable (``color_col``). "Hot spots"
  (bright, intense colors) indicate a strong interaction effect,
  where a specific combination of the two features leads to a
  notable outcome.

**Use Cases:**

* To diagnose how **pairs of features** interact to affect a
  model's prediction or error, moving beyond simple feature
  importance.
* To identify non-linear relationships and conditional patterns
  in your data.
* To visually confirm that a model has learned an expected
  physical or logical interaction (e.g., high solar output
  only occurs at midday with low cloud cover).

While individual feature importances are revealing, they do not tell
the whole story. In many complex systems, the most powerful predictive
signals come from the **interaction** between two or more features.
This polar heatmap is designed to move beyond one-dimensional analysis
and uncover these crucial two-way feature interactions.

.. admonition:: Practical Example

   An energy analyst is modeling the power output of a solar farm.
   They know that the output depends on both the **time of day** and
   the **cloud cover**. However, the effect is not simply additive;
   these two features interact strongly. High energy output is only
   possible when it is both midday AND cloud cover is low. At night,
   the level of cloud cover is completely irrelevant.

   This plot will visualize this interaction by mapping the time of
   day to the angle, cloud cover to the radius, and the resulting
   energy output to the color, revealing the "hot spot" of peak
   performance.

   .. code-block:: pycon

      >>> import numpy as np
      >>> import pandas as pd
      >>> import kdiagram as kd
      >>>
      >>> # --- 1. Simulate solar farm output data ---
      >>> np.random.seed(42)
      >>> n_points = 5000
      >>> df = pd.DataFrame({
      ...     'hour_of_day': np.random.uniform(0, 24, n_points),
      ...     'cloud_cover_pct': np.random.uniform(0, 100, n_points)
      ... })
      >>> # Output depends on time (peaks at noon) AND low cloud cover
      >>> time_effect = np.sin(df['hour_of_day'] * np.pi / 24)**2
      >>> cloud_effect = (100 - df['cloud_cover_pct']) / 100
      >>> df['energy_output_kw'] = 150 * time_effect * cloud_effect + np.random.randn(n_points) * 5
      >>>
      >>> # --- 2. Generate the plot ---
      >>> ax = kd.plot_feature_interaction(
      ...     df,
      ...     theta_col='hour_of_day',
      ...     r_col='cloud_cover_pct',
      ...     color_col='energy_output_kw',
      ...     theta_period=24,
      ...     title='Solar Energy Output (kW) vs. Time and Cloud Cover'
      ... )

   .. figure:: ../images/userguide_plot_feature_interaction.png
      :align: center
      :width: 80%
      :alt: A polar heatmap showing a two-way feature interaction.

      A polar heatmap visualizing the interaction between the hour of
      the day (angle) and cloud cover (radius) on solar energy
      output (color).

   This plot translates a complex, three-dimensional relationship into
   an intuitive 2D visualization. The location of the most intense
   colors reveals the conditions that lead to the strongest outcomes.

   **Quick Interpretation:**
    This polar heatmap clearly visualizes the strong interaction between
    the time of day (angle) and cloud cover (radius) on energy output
    (color). The most intense energy generation, shown by the bright
    yellow "hot spot," occurs only under a specific combination of
    conditions: near midday (top of the plot) **and** with very low
    cloud cover (close to the center). The plot also confirms that
    during the night (bottom of the plot), energy output is near zero
    regardless of cloud cover, effectively demonstrating that the two
    features are not merely additive but have a powerful interactive
    effect.

Understanding feature interactions is key to unlocking deeper insights
from your data and models. To see the full code for this example,
please visit the gallery.

**Example**
See the gallery example and code:
:ref:`gallery_plot_feature_interaction`.

.. raw:: html

   <hr>
   
.. rubric:: References

.. footbibliography::