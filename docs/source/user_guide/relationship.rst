.. _userguide_relationship:

=============================
Visualizing Relationships
=============================

Understanding the relationship between observed (true) values and model
predictions is fundamental to evaluation. While standard scatter plots
are common, visualizing this relationship in a polar context can
sometimes reveal different patterns or allow for comparing multiple
prediction series against the true values in a compact format.

`k-diagram` provides the `plot_relationship` function to explore these
connections using a flexible polar scatter plot where the angle is
derived from the true values and the radius from the predicted values.

Summary of Relationship Functions
---------------------------------

This section focuses on the primary function for visualizing these
true-vs-predicted relationships in polar coordinates:

.. list-table:: Relationship Visualization Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.relationship.plot_relationship`
     - Creates a polar scatter plot mapping true values to angle and
       (normalized) predicted values to radius.


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
the range of true values. 

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
(See the :ref:`Gallery <gallery_plot_relationship>` section below for a runnable code example and plot)


.. raw:: html

   <hr>

