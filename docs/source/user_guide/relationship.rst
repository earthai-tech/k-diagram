.. _userguide_relationship:

=============================
Visualizing Relationships
=============================

Understanding the relationship between observed (true) values and model
predictions is fundamental to evaluation (see :footcite:t:`Murphy1993What, Jolliffe2012`).
While standard scatter plots are common, visualizing this relationship in a polar
context can sometimes reveal different patterns or allow for comparing multiple
prediction series against the true values in a compact format (see also the wider
discussion on calibration and sharpness in probabilistic evaluation,
:footcite:p:`Gneiting2007b`).

`k-diagram` provides the ``plot_relationship`` function to explore these
connections using a flexible polar scatter plot where the angle is
derived from the true values and the radius from the predicted values
:footcite:p:`kouadiob2025`.

Summary of Relationship Functions
---------------------------------

This section focuses on functions for visualizing the relationships
between the core components of a forecast: true values, model
predictions, and forecast errors.

.. list-table:: Relationship Visualization Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.relationship.plot_relationship`
     - Creates a polar scatter plot mapping true values to angle and
       (normalized) predicted values to radius.
   * - :func:`~kdiagram.plot.relationship.plot_conditional_quantiles`
     - Visualizes how the full predicted distribution (quantile bands)
       changes as a function of the true value.
   * - :func:`~kdiagram.plot.relationship.plot_error_relationship`
     - Plots the forecast error against the true value to diagnose
       conditional biases.
   * - :func:`~kdiagram.plot.relationship.plot_residual_relationship`
     - Plots the forecast error (residual) against the predicted value
       to diagnose issues like heteroscedasticity.
       

Detailed Explanations
---------------------

Let's dive into the :mod:`kdiagram.plot.relationship` function.

.. _ug_plot_relationship:

True vs. Predicted Polar Relationship (:func:`~kdiagram.plot.relationship.plot_relationship`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function generates a polar scatter plot designed to visualize the
relationship between a single set of true (observed) values and one or
more sets of corresponding predicted values. It maps the true values to
the angular position and the predicted values (normalized) to the radial
position, allowing comparison of how different predictions behave across
the range of true values :footcite:p:`kouadiob2025` ( see foundational ideas
on forecast evaluation and reliability :footcite:p:`Murphy1993What, Jolliffe2012`).

**Mathematical Concept:**

1.  **Angular Mapping** ( :math:`\theta` ): Let's consider :math:`\upsilon` as 
    the ``angular_angle``. The angle :math:`\theta_i` for each
    data point :math:`i` is determined by its corresponding true value 
    :math:`y_{\text{true}_i}` based on the ``theta_scale`` parameter:
    
    * ``'proportional'`` (Default): Linearly maps the range of
      `y_true` values to the specified angular coverage (`acov`).
        
      .. math::
          \theta_i = \theta_{offset} + \upsilon \cdot
          \frac{y_{\text{true}_i} - \min(y_{\text{true}})}
          {\max(y_{\text{true}}) - \min(y_{\text{true}})}
            
    * ``'uniform'``: Distributes points evenly across the angular
      range based on their index :math:`i`, ignoring the actual
      `y_true` value for positioning (useful if `y_true` isn't
      strictly ordered or continuous).
        
      .. math::
          \theta_i = \theta_{offset} + \upsilon \cdot
          \frac{i}{N-1}

    Where :math:`\upsilon` is determined by `acov` (e.g., :math:`2\pi`
    for 'default', :math:`\pi` for 'half_circle') and :math:`\theta_{offset}`
    is an optional rotation.

2.  **Radial Mapping** :math:`r`: For *each* prediction series `y_pred`, its
    values are independently normalized to the range [0, 1] using min-max
    scaling. This normalized value determines the radius :math:`r_i` for
    that prediction series at angle :math:`\theta_i`.
    
    .. math::
        r_i = \frac{y_{\text{pred}_i} - \min(y_{\text{pred}})}
        {\max(y_{\text{pred}}) - \min(y_{\text{pred}})}

3.  **Custom Angle Labels** :math:`z_{values}`: If :math:`z_{values}` are provided,
    the angular tick labels are replaced with these values (scaled to
    match the angular range), providing a way to label the angular axis
    with a variable other than the raw `y_true` values used for positioning.

**Interpretation:**

* **Angle:** Represents the position within the range of `y_true` values
  (if `theta_scale='proportional'`) or simply the sample index (if
  `theta_scale='uniform'`). If `z_values` are used, the tick labels
  refer to that variable.
* **Radius:** Represents the **normalized** predicted value for a specific
  model/series. A radius near 1 means the prediction was close to the
  *maximum prediction* made by *that specific model*. A radius near 0
  means it was close to the *minimum prediction* made by *that model*.
* **Comparing Models:** Look at points with similar angles (i.e., similar
  `y_true` values). Compare the radial positions of points from
  different models (different colors). Does one model consistently
  predict higher *normalized* values than another at certain `y_true`
  ranges (angles)?
* **Relationship Pattern:** Observe the overall pattern. Does the radius
  (normalized prediction) tend to increase as the angle (`y_true`)
  increases? Is the relationship linear, cyclical, or scattered? How
  does the pattern differ between models?

**Use Cases:**

* Comparing the *relative* response patterns of multiple models across
  the observed range of true values, especially when absolute scales
  differ.
* Visualizing potential non-linear relationships between true values
  (angle) and normalized predictions (radius).
* Exploring data using alternative angular representations by providing
  custom labels via `z_values`.
* Displaying cyclical relationships if `y_true` represents a cyclical
  variable (e.g., day of year, hour of day) and `acov='default'`.

**Advantages (Polar Context):**

* Can effectively highlight cyclical patterns when `y_true` is mapped
  proportionally to a full circle (`acov='default'`).
* Allows overlaying multiple normalized prediction series against a
  common angular axis derived from the true values.
* Flexible angular labeling using `z_values` provides context beyond the
  raw `y_true` mapping.
* Normalization focuses the comparison on response *patterns* rather than
  absolute prediction magnitudes.

**Example:**
(See the :ref:`Gallery <gallery_plot_relationship>` section
below for a runnable code example and plot)


.. raw:: html

   <hr>

.. _ug_plot_conditional_quantiles:

Conditional Quantile Bands (:func:`~kdiagram.plot.relationship.plot_conditional_quantiles`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Conditional Quantile Plot** to
visualize how the entire predicted conditional distribution
(represented by quantile bands) changes as a function of the true
observed value. It is a powerful diagnostic tool for identifying
**heteroscedasticity**—i.e., whether the forecast uncertainty is
constant or changes with the magnitude of the target variable.

**Mathematical Concept:**
This plot provides an intuitive view of the conditional predictive
distribution, a novel visualization developed as part of the
analytics framework:footcite:p:`kouadiob2025`.

1.  **Coordinate Mapping**: The function first sorts the data based
    on the true values, :math:`y_{true}`, to ensure a continuous
    spiral. The sorted true values are then mapped to the
    angular coordinate, :math:`\theta`, in the range :math:`[0, 2\pi]`.

    .. math::
       :label: eq:angle_map_cond_q

       \theta_i \propto y_{true,i}^{\text{(sorted)}}

    The predicted quantiles, :math:`q_{i, \tau}`, for each
    observation :math:`i` and quantile level :math:`\tau` are
    mapped directly to the radial coordinate, :math:`r`.

2.  **Band Construction**: For a given prediction interval (e.g.,
    80%), the corresponding lower (:math:`\tau=0.1`) and
    upper (:math:`\tau=0.9`) quantile forecasts are used to
    define the boundaries of a shaded band. The function can
    plot multiple, nested bands to give a more complete picture
    of the distribution's shape. The median forecast
    (:math:`\tau=0.5`) is drawn as a solid central line.


**Interpretation:**
The plot reveals how the forecast distribution's center and spread
are related to the true value on the angular axis.

* **Central Line (Median Forecast)**: The position of this line
    shows the central tendency of the forecast. If it consistently
    deviates from a perfect spiral, it may indicate a conditional
    bias.
* **Shaded Bands (Prediction Intervals)**: The **width** of these
    bands is the most important feature.
    
    - If the band has a **constant width** as the angle increases,
      the model's uncertainty is **homoscedastic** (constant).
    - If the band's width **changes** (e.g., gets wider), the
      model's uncertainty is **heteroscedastic**, meaning the
      forecast precision depends on the magnitude of the true value.

**Use Cases:**

* To diagnose if a model's uncertainty is constant or if it
    changes with the magnitude of the target variable.
* To visually inspect the full predicted distribution, not just a
    point estimate, across the range of outcomes.
* To identify if a model is consistently over- or under-confident
    for specific ranges of the true value by observing the band widths.

**Example:**
See the gallery example and code: :ref:`gallery_plot_conditional_quantiles`.


.. raw:: html

   <hr>
   
.. _ug_plot_error_relationship:

Error vs. True Value Relationship (:func:`~kdiagram.plot.relationship.plot_error_relationship`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This function creates a **Polar Error vs. True Value Plot**, a
powerful diagnostic tool for understanding if a model's errors are
correlated with the magnitude of the actual outcome. The angle is
proportional to the **true value**, and the radius represents the
**forecast error**. It is designed to reveal conditional biases and
heteroscedasticity.


**Mathematical Concept:**
This plot is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`. It helps
diagnose if the model's error is independent of the true value,
a key assumption in many statistical models.

1.  **Error (Residual) Calculation**: For each observation
    :math:`i`, the error is the difference between the true and
    predicted value.

    .. math::
       :label: eq:error_calc_true

       e_i = y_{true,i} - y_{pred,i}

2.  **Angular Mapping**: The angle :math:`\theta_i` is made
    proportional to the true value :math:`y_{true,i}`,
    after sorting, to create a continuous spiral.

    .. math::
       :label: eq:angle_map_true

       \theta_i \propto y_{true,i}^{\text{(sorted)}}

3.  **Radial Mapping**: The radius :math:`r_i` represents the
    error :math:`e_i`. To handle negative error values on a
    polar plot, an offset is added to all radii so that the
    zero-error line becomes a reference circle.


**Interpretation:**
The plot reveals how the error distribution changes as the true
value increases.

* **Conditional Bias**: A well-behaved model should have its
    error points scattered symmetrically around the "Zero Error"
    circle at all angles. If the center of the point cloud
    consistently drifts away from this circle at certain angles,
    it reveals a **conditional bias** (e.g., the model only
    under-predicts high values).
* **Heteroscedasticity**: The vertical spread of the points
    (the width of the spiral) shows the error variance. If this
    spread changes as the angle increases, it indicates
    **heteroscedasticity** (i.e., the model is more or less
    certain for different true values).

**Use Cases:**

* To check the fundamental assumption in many models that errors
    are independent of the true value.
* To diagnose if a model has a conditional bias (e.g., it only
    performs poorly for high or low values).
* To visually inspect for heteroscedasticity, where the variance
    of the error changes across the range of true values.

**Example:**
See the gallery example and code: :ref:`gallery_plot_error_relationship`.

.. raw:: html

   <hr>
   
.. _ug_plot_residual_relationship:

Residual vs. Predicted Relationship (:func:`~kdiagram.plot.relationship.plot_residual_relationship`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Residual vs. Predicted Plot**, a
fundamental diagnostic for assessing model performance. The angle is
proportional to the **predicted value**, and the radius represents
the **forecast error** (residual). It is a powerful tool for
identifying if a model's errors are correlated with its own
predictions, which can reveal issues like heteroscedasticity.

.. admonition:: Key Distinction: Error vs. Residual Plots
   :class: hint

   This plot is a companion to
   :func:`~kdiagram.plot.relationship.plot_error_relationship`.
   The key difference is the variable mapped to the angle:

   - **Error vs. True Value Plot**: Angle is based on ``y_true``. It
     answers: *"Are my errors related to the actual outcome?"*
   - **Residual vs. Predicted Plot**: Angle is based on ``y_pred``. It
     answers: *"Are my errors related to what my model is predicting?"*

   Both are crucial for a complete diagnosis.

**Mathematical Concep:t**
This plot is a novel visualization developed as part of the
analytics framework in :footcite:p:`kouadiob2025`.

1.  **Error (Residual) Calculation**: For each observation
    :math:`i`, the error is the difference between the true and
    predicted value.

    .. math::
       :label: eq:residual_calc

       e_i = y_{true,i} - y_{pred,i}

2.  **Angular Mapping**: The angle :math:`\theta_i` is made
    proportional to the **predicted value** :math:`y_{pred,i}`,
    after sorting, to create a continuous spiral.

    .. math::
       :label: eq:angle_map_pred

       \theta_i \propto y_{pred,i}^{\text{(sorted)}}

3.  **Radial Mapping**: The radius :math:`r_i` represents the
    error :math:`e_i`. An offset is added to handle negative
    values, making the "Zero Error" line a reference circle.


**Interpretation:**
The plot reveals how the error distribution changes as the
model's own prediction magnitude increases.

* **Heteroscedasticity**: A well-behaved model should have a
    random scatter of points with a constant vertical spread
    (width of the spiral). If the spread of points forms a
    **cone or fan shape**, getting wider as the angle increases,
    it is a clear sign of **heteroscedasticity**. This means the
    model's error variance grows as its predictions get larger.
* **Conditional Bias**: If the center of the point cloud
    consistently drifts away from the "Zero Error" circle at
    certain angles, it reveals a bias dependent on the
    prediction's magnitude (e.g., the model is only biased when
    it predicts high values).


**Use Cases:**

* To check the assumption that the variance of the model's errors
    is constant across the range of its predictions.
* To diagnose if a model is becoming more or less confident in
    itself as its predictions change.
* To identify non-linear patterns in the residuals that might
    suggest a missing feature or an incorrect model specification.

**Example:**
See the gallery example and code:
:ref:`gallery_plot_residual_relationship`.

.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::