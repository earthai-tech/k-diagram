.. _userguide_uncertainty:

=======================================
Visualizing Forecast Uncertainty
=======================================

Effective forecasting involves more than just predicting a single future
value; it requires understanding the inherent **uncertainty** surrounding
that prediction. Point forecasts alone can be misleading, especially
when making critical decisions based on them. `k-diagram` provides a
suite of specialized polar visualizations designed to dissect and
illuminate various facets of forecast uncertainty.

Why Polar Plots for Uncertainty?
------------------------------------

Traditional Cartesian plots can become cluttered when visualizing
multiple aspects of uncertainty across many data points or locations.
`k-diagram` leverages the polar coordinate system to:

* Provide a **compact overview** of uncertainty characteristics
  across the entire dataset (represented angularly).
* Highlight **patterns** in uncertainty related to temporal or
  spatial dimensions (if mapped to the angle).
* Visually emphasize **drift**, **anomalies**, and **coverage**
  in intuitive ways using radial distance and color.

This page details the functions within `k-diagram` focused on
evaluating prediction intervals, diagnosing coverage failures,
analyzing anomaly severity, and tracking how uncertainty evolves.

Summary of Uncertainty Functions
--------------------------------

The following functions provide different perspectives on forecast
uncertainty and related diagnostics:

.. list-table:: Uncertainty Visualization Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.uncertainty.plot_actual_vs_predicted`
     - Compares actual vs. predicted point values point-by-point.
   * - :func:`~kdiagram.plot.uncertainty.plot_anomaly_magnitude`
     - Visualizes magnitude and type of prediction anomalies.
   * - :func:`~kdiagram.plot.uncertainty.plot_coverage`
     - Calculates and plots overall interval coverage scores.
   * - :func:`~kdiagram.plot.uncertainty.plot_coverage_diagnostic`
     - Diagnoses interval coverage point-by-point on a polar plot.
   * - :func:`~kdiagram.plot.uncertainty.plot_interval_consistency`
     - Shows consistency/variability of interval widths over time.
   * - :func:`~kdiagram.plot.uncertainty.plot_interval_width`
     - Visualizes the width of prediction intervals across samples.
   * - :func:`~kdiagram.plot.uncertainty.plot_model_drift`
     - Tracks how average uncertainty width changes over horizons.
   * - :func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty`
     - General plot for visualizing multiple series (e.g., quantiles).
   * - :func:`~kdiagram.plot.uncertainty.plot_uncertainty_drift`
     - Visualizes drift of uncertainty using concentric rings over time.
   * - :func:`~kdiagram.plot.uncertainty.plot_velocity`
     - Shows the rate of change (velocity) of median predictions.
   * - :func:`~kdiagram.plot.uncertainty.plot_radial_density_ring`
     - Shows a unique visualization of the probability distribution.

Detailed Explanations
---------------------

Let's explore some of these functions in detail.

.. _ug_actual_vs_predicted:

Actual vs. Predicted Comparison (:func:`~kdiagram.plot.uncertainty.plot_actual_vs_predicted`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This plot provides a direct visual comparison between the actual
observed ground truth values and the model's point predictions
(typically the median forecast, Q50) for each sample or location.
It's a fundamental diagnostic for assessing basic model accuracy and
identifying systematic biases.

**Mathematical Concept:**
For each data point :math:`i`, we have an actual value :math:`y_i` and a
predicted value :math:`\hat{y}_i`. The plot displays both values radially
at a corresponding angle :math:`\theta_i`. The difference, or error,
:math:`e_i = y_i - \hat{y}_i`, is implicitly visualized by the gap
between the plotted points/lines for actual and predicted values. Often,
gray lines connect :math:`y_i` and :math:`\hat{y}_i` at each angle to
emphasize the error magnitude and direction.

**Interpretation:**

* **Closeness:** How close are the points or lines representing actual
  and predicted values? Closer alignment indicates better point-forecast
  accuracy.
* **Systematic Bias:** Does the prediction line/dots consistently sit
  inside or outside the actual line/dots? This indicates a systematic
  under- or over-prediction bias.
* **Error Magnitude:** The length of the connecting gray lines (if shown)
  or the radial distance between points directly shows the prediction
  error for each sample. Large gaps indicate poor predictions for those
  points.
* **Angular Patterns:** If the angle :math:`\theta` represents a meaningful
  dimension (like time index, season, or spatial grouping), look for
  patterns in accuracy or bias around the circle. Does the model perform
  better or worse at certain "angles"?

**Use Cases:**

* **Initial Performance Check:** Get a quick overview of how well the
  point forecast aligns with reality across the dataset.
* **Bias Detection:** Easily spot systematic over- or under-prediction.
* **Identifying Problematic Regions:** If using angles meaningfully,
  locate specific periods or areas where point predictions are poor.
* **Communicating Basic Accuracy:** Provides a simple visual for
  stakeholders before diving into complex uncertainty measures.

**Advantages of Polar View:**

* Provides a compact, circular overview of performance across many samples.
* Can make cyclical patterns (if angle relates to time, like month or
  hour) more apparent than a standard time series plot.

**Example:**
(See :ref:`Gallery <gallery_plot_actual_vs_predicted>` for code and plot examples)

.. _ug_anomaly_magnitude:

Anomaly Magnitude Analysis (:func:`~kdiagram.plot.uncertainty.plot_anomaly_magnitude`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This diagnostic specifically focuses on **prediction interval failures**.
It identifies instances where the actual observed value falls *outside*
the predicted range [Qlow, Qup] and visualizes the **location**, **type**
(under- or over-prediction), and **severity** (magnitude) of these
anomalies. It answers: "When my model's uncertainty bounds are wrong,
*how wrong* are they, and where?"

**Mathematical Concept:**
An anomaly exists if the actual value :math:`y_i` is outside the
interval defined by the lower (:math:`Q_{low,i}`) and upper
(:math:`Q_{up,i}`) quantiles.

* **Under-prediction:** :math:`y_i < Q_{low,i}`
* **Over-prediction:** :math:`y_i > Q_{up,i}`

The **magnitude** (:math:`r_i`) of the anomaly is the absolute distance
from the actual value to the *nearest violated bound*:

.. math::

   r_i =
   \begin{cases}
     Q_{low,i} - y_i & \text{if } y_i < Q_{low,i} \\
     y_i - Q_{up,i} & \text{if } y_i > Q_{up,i} \\
     0              & \text{otherwise}
   \end{cases}

Only points where :math:`r_i > 0` are plotted. The radial coordinate of
a plotted point is :math:`r_i`.

**Interpretation:**

* **Presence/Absence:** Points only appear if an anomaly occurred. A sparse
  plot indicates good interval coverage. Dense clusters indicate regions
  of poor uncertainty estimation.
* **Radius:** The distance from the center directly represents the
  **severity** of the anomaly. Points far from the center are large
  errors relative to the predicted bounds.
* **Color:** Distinct colors (e.g., blues for under-prediction, reds for
  over-prediction) immediately classify the type of failure. Color
  intensity often also maps to the magnitude :math:`r_i`.
* **Angular Position:** Shows *where* (which samples, locations, or times,
  based on the angle representation) these failures occur. Look for
  clustering at specific angles.

**Use Cases:**

* **Risk Assessment:** Identify predictions where the actual outcome might
  be significantly worse than the uncertainty bounds suggested.
* **Model Calibration Check:** Assess if the prediction intervals are
  meaningful. Frequent or large anomalies suggest poor calibration.
* **Pinpointing Failure Modes:** Determine if the model tends to fail more
  by under-predicting or over-predicting, and under what conditions
  (angles).
* **Targeting Investigation:** Guide further analysis or data collection
  efforts towards the specific samples/locations exhibiting the most
  severe anomalies.

**Advantages of Polar View:**

* Provides a focused view solely on prediction interval failures.
* Radial distance intuitively maps to error magnitude/severity.
* Color effectively separates under- vs. over-prediction types.
* Circular layout helps identify patterns or concentrations of anomalies
  across the angular dimension.

**Example:**
(Refer to :ref:`Gallery <gallery_plot_anomaly_magnitude>` and runnable code examples)


.. raw:: html

   <hr>


.. _ug_coverage:

Overall Coverage Scores (:func:`~kdiagram.plot.uncertainty.plot_coverage`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function calculates and visualizes the **overall empirical
coverage rate** for one or more sets of predictions. It answers the
fundamental question: "Across the entire dataset, what fraction of the
time did the true observed values fall within the specified prediction
interval bounds (e.g., Q10 to Q90)?" It allows for comparing this
aggregate performance across different models or prediction sets using
various chart types.

**Mathematical Concept:**
The empirical coverage for a given prediction interval
:math:`[Q_{low,i}, Q_{up,i}]` and actual values :math:`y_i` over
:math:`N` samples is calculated as:

.. math::

   \text{Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{Q_{low,i} \le y_i \le Q_{up,i}\}

Where :math:`\mathbf{1}\{\cdot\}` is the indicator function, which is 1
if the condition (actual value :math:`y_i` is within the bounds) is
true, and 0 otherwise.

For point predictions :math:`\hat{y}_i`, coverage typically measures
exact matches (often resulting in very low scores unless data is
discrete): :math:`\text{Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{y_i = \hat{y}_i\}`.

**Interpretation:**

* **Compare to Nominal Rate:** The primary use is to compare the
  calculated empirical coverage rate against the **nominal coverage rate**
  implied by the quantiles used. For example, a Q10-Q90 interval has a
  nominal coverage of 80% (0.8).
  
  * If Empirical Coverage ≈ Nominal Coverage: The intervals are well-
    calibrated on average.
  * If Empirical Coverage > Nominal Coverage: The intervals are too wide
    (conservative) on average.
  * If Empirical Coverage < Nominal Coverage: The intervals are too narrow
    (overconfident) on average.
    
* **Model Comparison:** When plotting multiple models, directly compare
  their coverage scores. A model closer to the nominal rate is generally
  better calibrated in terms of its average interval performance.
* **Chart Type:**

  * `bar` or `line`: Good for direct comparison of scores between models.
  * `pie`: Shows the proportion of coverage relative to the sum (less common
    for direct calibration assessment).
  * `radar`: Provides a profile view comparing multiple models across the
    same metric (coverage).

**Use Cases:**

* Quickly assessing the average calibration of prediction intervals for
  one or multiple models.
* Comparing the overall reliability of uncertainty estimates from different
  forecasting methods.
* Summarizing interval performance for reporting.

**Advantages:**

* Provides a single, easily interpretable summary statistic for average
  interval performance per model.
* Offers multiple visualization options (`kind` parameter) for flexible
  comparison.

**Example:**
(See :ref:`Gallery <gallery_plot_overall_coverage>` for code and plot examples)

.. _ug_coverage_diagnostic:

Point-wise Coverage Diagnostic (:func:`~kdiagram.plot.uncertainty.plot_coverage_diagnostic`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
While :func:`~kdiagram.plot.uncertainty.plot_coverage` gives an overall
average, this function provides a **granular, point-by-point diagnostic**
of prediction interval coverage on a polar plot. It visualizes *where*
(at which specific sample, location, or time, represented angularly)
the prediction intervals succeeded or failed to capture the actual value.

**Mathematical Concept:**
For each data point :math:`i`, a binary coverage indicator :math:`c_i` is
calculated:

.. math::

   c_i = \mathbf{1}\{Q_{low,i} \le y_i \le Q_{up,i}\}

Each point :math:`i` is then plotted at an angle :math:`\theta_i`
(determined by its index or an optional feature) and a **radius**
:math:`r_i = c_i`. This means:

* Covered points (:math:`c_i=1`) are plotted at radius **1**.
* Uncovered points (:math:`c_i=0`) are plotted at radius **0**.

The plot also typically shows the overall coverage rate
:math:`\bar{c} = \frac{1}{N} \sum c_i` as a prominent reference circle.

**Interpretation:**

* **Radial Position:** Instantly separates successes (radius 1) from
  failures (radius 0).
* **Angular Clusters:** Look for clusters of points at radius 0. Such
  clusters indicate specific regions, times, or conditions (depending on
  what the angle represents) where the model's prediction intervals
  systematically fail. Randomly scattered points at radius 0 suggest less
  systematic issues.
* **Average Coverage Line:** The solid circular line drawn at radius
  :math:`\bar{c}` represents the overall empirical coverage rate. Compare
  its position to:
  
  * The nominal coverage rate (e.g., 0.8 for an 80% interval).
  * Reference grid lines (often shown at 0.2, 0.4, 0.6, 0.8, 1.0).
  
* **Background Gradient (Optional):** If enabled, the shaded gradient
  extending from the center to the average coverage line provides a strong
  visual cue for the overall performance level.
* **Point/Bar Color:** Color (e.g., green for covered, red for uncovered
  using the default 'RdYlGn' cmap) reinforces the binary status.

**Use Cases:**

* **Diagnosing Coverage Failures:** Go beyond the average rate to see
  *where* and *how often* intervals fail.
* **Identifying Systematic Issues:** Detect if failures are concentrated
  in specific segments of the data (angles).
* **Visual Calibration Assessment:** Provides a more intuitive feel for
  calibration than just a single number. Is the coverage rate met because
  most points are covered, or are there many failures balanced by overly
  wide intervals elsewhere?
* **Debugging Model Uncertainty:** Pinpoint areas needing improved
  uncertainty quantification.

**Advantages (Polar Context):**

* Excellent for visualizing the status of many points compactly.
* The radial mapping (0 or 1) provides a very clear visual separation
  of coverage success/failure.
* Angular clustering of failures is easily identifiable.
* The average coverage line acts as an immediate visual benchmark against
  the plot boundaries (0 and 1) and reference grid lines.

**Example:**
(See :ref:`Gallery <gallery_plot_coverage_diagnostic>` or function docstring for code and plot examples)

.. _ug_interval_consistency:

Interval Width Consistency (:func:`~kdiagram.plot.uncertainty.plot_interval_consistency`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This plot analyzes the **temporal stability** of the predicted
uncertainty range. It visualizes how much the **width** of the
prediction interval (:math:`Q_{up} - Q_{low}`) fluctuates for each
location or sample across multiple time steps or forecast horizons.
It answers: "Are the model's uncertainty estimates stable over time for
a given location, or do they vary wildly?"

**Mathematical Concept:**
For each location/sample :math:`j`, the interval width is calculated
for each available time step :math:`t`:

.. math::

   w_{j,t} = Q_{up,j,t} - Q_{low,j,t}

The plot then visualizes the *variability* of these widths :math:`w_{j,t}`
over the time steps :math:`t` for each location :math:`j`. The radial
coordinate :math:`r_j` typically represents either:

1.  **Standard Deviation:** :math:`r_j = \sigma_t(w_{j,t})` - Measures the
    absolute variability of the width.
2.  **Coefficient of Variation (CV):** :math:`r_j = \frac{\sigma_t(w_{j,t})}{\mu_t(w_{j,t})}`
    - Measures the relative variability (standard deviation relative to the
    mean width). Set via the ``use_cv=True`` parameter.

Each location :math:`j` is plotted at an angle :math:`\theta_j` (based
on index) and radius :math:`r_j`. The color of the point often represents
the *average median prediction* :math:`\mu_t(Q_{50,j,t})` across the time
steps, providing context.

**Interpretation:**

* **Radius:** Points far from the center indicate locations where the
  prediction interval width is **inconsistent** or varies significantly
  across the different time steps/horizons considered. Points near the
  center have stable interval width predictions over time.
  
* **CV vs. Standard Deviation (`use_cv`):**

  * If `use_cv=False` (default), radius shows *absolute* standard
    deviation. A large radius means large absolute fluctuations in width.
  * If `use_cv=True`, radius shows *relative* variability (CV). A large
    radius means the width fluctuates significantly *compared to its
    average width*. This helps compare consistency across locations that
    might have very different average interval widths.
    
* **Color (Context):** If `q50_cols` are provided, color typically shows
  the average Q50 value. This helps answer questions like: "Does high
  inconsistency (large radius) tend to occur in locations with high or low
  average predicted values?"
  
* **Angular Clusters:** Clusters of points with high/low radius might indicate
  spatial patterns in the stability of uncertainty predictions.

**Use Cases:**

* **Assessing Model Reliability Over Time:** Identify locations where
  uncertainty estimates are unstable across forecast horizons.
* **Diagnosing Temporal Effects:** Understand if interval predictions
  become more or less variable further into the future.
* **Comparing Relative vs. Absolute Stability:** Use `use_cv` to
  distinguish between large absolute fluctuations and large relative
  fluctuations.
* **Identifying Locations for Scrutiny:** Points with high inconsistency
  might warrant further investigation into why the uncertainty estimate
  is so variable for those locations/conditions.

**Advantages (Polar Context):**

* Compactly displays the consistency profile across many locations.
* Radial distance provides an intuitive measure of inconsistency
  (variability).
* Allows visual identification of clusters based on consistency levels.
* Color adds valuable context about the average prediction level associated
  with different consistency levels.

**Example:**
(See :ref:`Gallery <gallery_plot_interval_consistency>` or function docstring for code and plot examples)

.. raw:: html

   <hr>

.. _ug_interval_width:

Prediction Interval Width Visualization (:func:`~kdiagram.plot.uncertainty.plot_interval_width`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a polar scatter plot focused solely on the
**magnitude of predicted uncertainty**. It visualizes the **width** of
the prediction interval (:math:`Q_{up} - Q_{low}`) for each individual
sample or location, typically at a single snapshot in time or for a
specific forecast horizon. It answers: "How wide is the predicted
uncertainty range for each point in my dataset?"

**Mathematical Concept:**
For each data point :math:`i`, the interval width is calculated:

.. math::

   w_i = Q_{up,i} - Q_{low,i}

The point is plotted at an angle :math:`\theta_i` (based on index) and a
**radius** :math:`r_i = w_i`. Optionally, a third variable :math:`z_i`
from a specified ``z_col`` can determine the color of the point; otherwise,
the color typically represents the width :math:`w_i` itself.

**Interpretation:**

* **Radius:** The radial distance directly corresponds to the width of
  the prediction interval. Points far from the center represent samples
  with high predicted uncertainty (wide intervals). Points near the
  center have low predicted uncertainty (narrow intervals).
* **Color (with `z_col`):** If a ``z_col`` (e.g., the median prediction
  Q50, or the actual value) is provided, the color allows you to see how
  interval width relates to that variable. For example, are wider
  intervals (larger radius) associated with higher or lower median
  predictions (color)?
* **Color (without `z_col`):** If no ``z_col`` is given, color usually
  maps to the width itself, reinforcing the radial information.
* **Angular Patterns:** Look for regions around the circle (representing
  subsets of data based on index order or a future `theta_col`
  implementation) that exhibit consistently high or low interval widths.

**Use Cases:**

* Identifying samples or locations with the largest/smallest predicted
  uncertainty ranges at a specific time/horizon.
* Visualizing the overall distribution of uncertainty magnitudes across
  the dataset.
* Exploring potential relationships between uncertainty width and other
  factors (e.g., input features, predicted value magnitude) by using
  the ``z_col`` option.
* Assessing if uncertainty is relatively uniform or highly variable
  across samples.

**Advantages (Polar Context):**

* Provides a compact overview of uncertainty magnitude for many points.
* The radial distance offers a direct, intuitive mapping for interval
  width.
* Facilitates the visual identification of angular patterns or clusters
  related to uncertainty levels.
* Allows simultaneous visualization of location (angle), uncertainty
  width (radius), and a third variable (color via ``z_col``).

**Example:**
(See :ref:`Gallery <gallery_plot_interval_width>` or function docstring for code and plot examples)

.. _ug_model_drift:

Model Forecast Drift (:func:`~kdiagram.plot.uncertainty.plot_model_drift`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This visualization focuses on **model degradation over forecast
horizons**. It creates a polar *bar* chart to show how the *average*
prediction uncertainty (specifically, the mean interval width
:math:`\mathbb{E}[Q_{up} - Q_{low}]`) changes as the forecast lead time
increases. It helps diagnose *concept drift* or *model aging* effects
related to uncertainty.

**Mathematical Concept:**
For each distinct forecast horizon :math:`h` (e.g., 1-step ahead, 2-steps
ahead), the average interval width across all :math:`N` samples is
calculated:

.. math::

   \bar{w}_h = \frac{1}{N} \sum_{j=1}^{N} (Q_{up,j,h} - Q_{low,j,h})

Each horizon :math:`h` is assigned a distinct angle :math:`\theta_h` on
the polar plot. A bar is drawn at this angle with a height (radius)
proportional to the average width :math:`\bar{w}_h`. The color of the
bar typically also reflects this average width, or potentially another
aggregated metric for that horizon if ``color_metric_cols`` is used.

**Interpretation:**

* **Radial Growth:** The key aspect is the change in bar height (radius)
  as the angle (horizon) progresses. A noticeable increase in radius for
  later horizons indicates that, on average, the model's prediction
  intervals widen significantly as it forecasts further into the future.
  This signifies increasing uncertainty or *model drift*.
* **Bar Height Comparison:** Directly compare the heights of bars for
  different horizons to quantify the average increase in uncertainty.
  Annotations usually display the exact average width :math:`\bar{w}_h`
  for each horizon.
* **Stability:** Bars of relatively similar height across horizons suggest
  that the model's average uncertainty level is stable over the forecast
  lead times considered.

**Use Cases:**

* **Detecting Model Degradation:** Identify if forecast uncertainty grows
  unacceptably large at longer lead times.
* **Assessing Forecast Reliability Horizon:** Determine the practical
  limit of how far ahead the model provides reasonably certain forecasts.
* **Informing Retraining Strategy:** Significant drift might indicate the
  need for more frequent model retraining or incorporating features that
  capture evolving dynamics.
* **Comparing Model Stability:** Generate plots for different models to
  compare how their uncertainty characteristics drift over time.

**Advantages (Polar Context):**

* The polar bar chart format makes the "outward drift" of average
  uncertainty across increasing horizons (angles) very intuitive to grasp.
* Provides a concise summary comparing average uncertainty levels across
  multiple forecast lead times.

**Example:**
(See :ref:`Gallery <gallery_plot_model_drift>` or function docstring for code and plot examples)

.. _ug_temporal_uncertainty:

General Polar Series Visualization (:func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This is a **general-purpose** polar scatter plot utility within the
uncertainty module, designed for visualizing and comparing **multiple
data series** (columns from a DataFrame) simultaneously. While flexible,
a common application in uncertainty analysis is to plot different quantile
predictions (e.g., Q10, Q50, Q90) for the *same* forecast horizon to
visualize the **uncertainty spread** at that specific point in time across
all samples.

**Mathematical Concept:**
For each data series :math:`k` (corresponding to a column in ``q_cols``)
and each sample :math:`i`, the value :math:`v_{i,k}` is plotted at an
angle :math:`\theta_i` (based on index) and radius :math:`r_{i,k} = v_{i,k}`.

If ``normalize=True``, each series :math:`k` is independently scaled
to the range [0, 1] before plotting using min-max scaling:
:math:`r_{i,k} = (v_{i,k} - \min_j(v_{j,k})) / (\max_j(v_{j,k}) - \min_j(v_{j,k}))`.
Each series :math:`k` is assigned a distinct color.

**Interpretation:**

* **Series Comparison:** Observe the relative radial positions of points
  belonging to different series (colors) at the same angle.
* **Uncertainty Spread (Quantile Use Case):** When plotting Q10, Q50,
  and Q90 for a single horizon:
  
  * The **radial distance** between the points for Q10 (e.g., blue) and
    Q90 (e.g., red) at a specific angle represents the **interval width**
    (uncertainty) for that sample.
  * Look for how this spread varies around the circle (across samples).
  * The position of the Q50 points (e.g., green) shows the central tendency
    relative to the bounds.
    
* **Normalization Effect:** If ``normalize=True``, the plot emphasizes the
  *relative shapes* and *overlap* of the series, regardless of their
  original scales. This is useful for comparing patterns but loses
  information about absolute magnitudes. If ``normalize=False``, the
  radial axis reflects the actual data values.
  
* **Angular Patterns:** Observe if specific series tend to be higher or lower
  at certain angles (samples/locations).

**Use Cases:**

* **Visualizing Uncertainty Intervals:** Plot Qlow, Qmid, Qup for a *single*
  time step/horizon to see the uncertainty band across samples.
* **Comparing Multiple Models:** Plot the point predictions (e.g., Q50)
  from several different models to compare their outputs side-by-side.
* **Plotting Related Variables:** Visualize any set of related numerical
  columns from your DataFrame in a polar layout.

**Advantages (Polar Context):**

* Allows overlaying multiple related data series in a single, compact plot.
* Effective for visualizing the *spread* or *range* between different
  series (like quantiles) at each angular position.
* Normalization option facilitates shape comparison for series with
  different scales.
* Can reveal shared cyclical patterns among the plotted series.

**Example:**
(See :ref:`Gallery <gallery_plot_temporal_uncertainty>` or function docstring for code and plot examples)

.. raw:: html

   <hr>


.. _ug_uncertainty_drift:

Multi-Time Uncertainty Drift Rings (:func:`~kdiagram.plot.uncertainty.plot_uncertainty_drift`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This plot offers a dynamic view of how the **spatial pattern of
prediction uncertainty** (interval width) evolves across **multiple time
steps** (e.g., years) for all locations simultaneously. Unlike
:func:`~kdiagram.plot.uncertainty.plot_model_drift`, which averages
across locations for each horizon, this function plots each time step
as a distinct **concentric ring**, allowing direct comparison of the
uncertainty "map" at different times.

**Mathematical Concept:**
For each location :math:`j` and time step :math:`t`, the interval width
is calculated: :math:`w_{j,t} = Q_{up,j,t} - Q_{low,j,t}`. These widths
are typically **normalized globally** across all locations and times:
:math:`w'_{j,t} = w_{j,t} / \max_{j',t'}(w_{j',t'})`.

Each location :math:`j` corresponds to an angle :math:`\theta_j`. For a
given time step :math:`t`, the radius :math:`r_{j,t}` for location
:math:`j` is determined by a base offset for that ring plus the scaled
normalized width:

.. math::

   r_{j,t} = R_t + H \cdot w'_{j,t}

Where :math:`R_t` is the base radius for ring :math:`t` (increasing
with time, controlled by ``base_radius``) and :math:`H` is a scaling
factor (``band_height``) controlling the visual impact of the width.
Each ring :math:`t` receives a distinct color from the specified
``cmap``.

**Interpretation:**

* **Concentric Rings:** Each colored ring represents a specific time
  step, with inner rings typically corresponding to earlier times and
  outer rings to later times.
* **Ring Shape & Radius Variations:** The deviations of a single ring
  from a perfect circle show the spatial variability of uncertainty
  *at that specific time step*. Points on a ring that bulge outwards
  represent locations with higher relative uncertainty (wider intervals)
  at that time.
* **Comparing Rings:** Examine how the overall radius and "bumpiness"
  change from inner rings (earlier times) to outer rings (later times).
  If outer rings are consistently larger or more irregular, it suggests
  that uncertainty generally increases and/or becomes more spatially
  variable over time.
* **Angular Patterns:** Trace specific angles (locations) across multiple
  rings. Does the radius consistently increase (growing uncertainty at
  that location)? Is it consistently large or small (persistently
  high/low uncertainty location)?

**Use Cases:**

* Tracking the **full spatial pattern** of uncertainty as it evolves
  over multiple forecast periods.
* Identifying specific locations where uncertainty grows or shrinks most
  dramatically over time.
* Comparing the uncertainty landscape between different forecast horizons
  (e.g., visualizing the difference in uncertainty patterns between a
  1-year and a 5-year forecast).
* Complementing :func:`~kdiagram.plot.uncertainty.plot_model_drift` by
  showing detailed spatial variations instead of just the average trend.

**Advantages (Polar Context):**

* Uniquely effective at overlaying multiple temporal snapshots of the
  uncertainty field in a single, comparative view.
* Concentric rings provide clear visual separation between time steps.
* Radial variations within each ring clearly highlight spatial differences
  in relative uncertainty at that time.
* Color coding aids in distinguishing and tracking specific time steps.

**Example:**
(See :ref:`Gallery <gallery_plot_uncertainty_drift>` or function docstring for code and plot examples)

.. _ug_velocity:

Prediction Velocity Visualization (:func:`~kdiagram.plot.uncertainty.plot_velocity`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This plot visualizes the **rate of change**, or **velocity**, of the
central forecast prediction (typically the median, Q50) over consecutive
time periods for each individual location or sample. It helps understand
the predicted dynamics of the phenomenon being forecast, answering: "How
fast is the predicted median value changing from one period to the next
at each location?"

**Mathematical Concept:**
For each location :math:`j`, the change in the median prediction between
consecutive time steps :math:`t` and :math:`t-1` is calculated:
:math:`\Delta Q_{50,j,t} = Q_{50,j,t} - Q_{50,j,t-1}`. The average velocity
for location :math:`j` over all time steps is the mean of these changes:

.. math::

   v_j = \mathbb{E}_t [ \Delta Q_{50,j,t} ]

The point for location :math:`j` is plotted at angle :math:`\theta_j`
(based on index) and radius :math:`r_j = v_j`. The radius can be
normalized to [0, 1] if ``normalize=True``. The color of the point can
represent either the velocity :math:`v_j` itself, or the average
absolute magnitude of the Q50 predictions
:math:`\mathbb{E}_t [ |Q_{50,j,t}| ]` (controlled by ``use_abs_color``).

**Interpretation:**

* **Radius:** Directly represents the average velocity (rate of change)
  of the Q50 prediction.
  
  * Points far from the center indicate locations with **high average
    velocity** (rapidly changing predictions).
  * Points near the center indicate locations with **low average
    velocity** (stable predictions).
  * If normalized, the radius shows relative velocity across locations.
  
* **Color (Mapped to Velocity):** If ``use_abs_color=False``, color
  directly reflects the velocity value :math:`v_j`. Using a diverging
  colormap (like 'coolwarm') helps distinguish between positive average
  change (e.g., red/warm colors for increasing values) and negative
  average change (e.g., blue/cool colors for decreasing values).
    
* **Color (Mapped to Q50 Magnitude):** If ``use_abs_color=True``, color
  shows the average absolute value of the Q50 predictions themselves.
  This provides context: Is high velocity (large radius) associated
  with high or low absolute predicted values (color)?
    
* **Angular Patterns:** Look for clusters of points with similar radius
  (velocity) or color at specific angles, which might indicate spatial
  patterns in the predicted dynamics.

**Use Cases:**

* Identifying spatial "hotspots" where the predicted phenomenon is changing
  most rapidly.
* Locating areas of predicted stability or stagnation.
* Analyzing and visualizing the spatial distribution of predicted trends or
  rates of change.
* Contextualizing velocity with the underlying magnitude of the prediction
  (e.g., are flood level predictions rising faster in already high areas?).

**Advantages (Polar Context):**

* Provides a compact overview comparing the rate of change across many
  locations or samples.
* Radial distance gives an intuitive sense of the magnitude of change
  (velocity).
* Color adds a critical second layer of information, either directional change
  or contextual magnitude.
* Facilitates spotting spatial patterns or clusters related to the dynamics
  of the prediction.

**Example:**
(See :ref:`Gallery <gallery_plot_prediction_velocity>` or function docstring for code and plot examples)

.. raw:: html

   <hr>


.. _ug_radial_density_ring:

Radial Density Ring (:func:`~kdiagram.plot.uncertainty.plot_radial_density_ring`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This plot provides a unique visualization of the **one-dimensional
probability distribution** of a continuous variable. It uses Kernel
Density Estimation (KDE) to create a smooth representation of the data's
distribution, answering the question: "What is the shape of this
data's distribution, and where are its most common values?"

**Mathematical Concept:**
The function first derives a one-dimensional data vector :math:`\mathbf{x}`
based on the ``kind`` and ``target_cols`` parameters. For instance, with
``kind='width'``, :math:`x_i = Q_{up,i} - Q_{low,i}`.

It then computes the Probability Density Function (PDF),
:math:`\hat{f}_h(x)`, using a Gaussian kernel. This is an estimate of the
true probability distribution from which the data samples are drawn.

The calculated PDF is then normalized to the range ``[0, 1]`` for
visual mapping to a color gradient:

.. math::

   \text{PDF}_{\text{norm}}(x) = \frac{\hat{f}_h(x)}{\max(\hat{f}_h)}

In the plot, the radial distance from the center corresponds to the
value :math:`x`, and the color at that radius is determined by
:math:`\text{PDF}_{\text{norm}}(x)`.

**Interpretation:**

* **Radius:** The radial axis represents the **value** of the
  metric being analyzed. The center corresponds to the minimum
  value in the data range, and the outer edge to the maximum.
* **Color:** The color at any given radius represents the
  **probability density** for that value. Intense, saturated colors
  indicate high density, corresponding to peaks (modes) in the
  distribution where data is most concentrated. Faint, light colors
  indicate low density, corresponding to the tails of the distribution.
* **Angle:** The angular dimension is purely for aesthetic effect and
  carries no information. The density is repeated around the
  full circle to create the "ring" visual.

**Use Cases:**

* **Error Distribution Analysis:** Plot the distribution of forecast
  errors (e.g., :math:`y_i - \hat{y}_i`). An ideal distribution is
  often a sharp peak centered at zero.
* **Uncertainty Characterization:** Visualize the distribution of
  prediction interval widths. A narrow, single-peaked distribution
  suggests the model produces consistent uncertainty estimates. A wide
  or multi-modal distribution suggests variability.
* **Velocity/Change Analysis:** Analyze the distribution of year-over-
  year changes or other calculated velocities to understand the
  typical magnitude and spread of change.
* **General Distribution Inspection:** Quickly understand the shape
  (e.g., normal, skewed, bimodal) of any continuous variable.

**Advantages of Polar View:**

* Provides a visually striking and compact representation of a 1D
  distribution.
* Avoids the binning choices and jagged appearance of a traditional
  histogram.
* The "ring" metaphor can be an intuitive way to view the entirety of a
  distribution's shape at once.

**Example:**
(See :ref:`Gallery <gallery_plot_radial_density_ring>` for code and plot examples)

.. raw:: html

   <hr>


.. _ug_plot_polar_heatmap:

2D Density Analysis (:func:`~kdiagram.plot.uncertainty.plot_polar_heatmap`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**

This function creates a **polar heatmap** to visualize the two-dimensional
density distribution of data points. It is particularly powerful for
uncovering relationships between a linear variable (mapped to the radius)
and a cyclical or ordered variable (mapped to the angle). It answers the
question: "Do high or low values of one metric tend to concentrate at
specific times, seasons, or categories?"

**Mathematical Concept:**
The plot is a 2D histogram in polar coordinates.

1. **Coordinate Mapping:** The data is mapped to polar coordinates. The
   radial variable :math:`r` is taken from ``r_col``. The angular
   variable :math:`\theta_{data}` from ``theta_col`` is converted to
   radians :math:`[0, 2\pi]`. If a period :math:`P` is provided (e.g.,
   24 for hours), the mapping is:

   .. math::

      \theta_{rad} = \left( \frac{\theta_{data} \pmod P}{P} \right) \cdot 2\pi

2. **Binning and Counting:** The polar space is divided into a grid of
   bins defined by ``r_bins`` and ``theta_bins``. The function then
   counts the number of data points that fall into each polar sector
   :math:`(r_j, \theta_k)`. The result is a count matrix :math:`\mathbf{C}`.

**Interpretation:**

* **Angle:** Represents the cyclical or ordered feature (e.g., hour of
  the day, month of the year).
* **Radius:** Represents the magnitude of the second variable (e.g.,
  prediction error, rainfall amount).
* **Color:** The color intensity of each polar bin corresponds to the
  **count** or density of data points within it. "Hot" or bright
  colors indicate a high concentration of data, revealing a strong
  relationship between the radial and angular variables in that region.

**Use Cases:**

* **Error Analysis:** Identify if large forecast errors (radius) are
  more frequent at certain times of the day (angle).
* **Feature Correlation:** Discover patterns between a cyclical feature
  and a measurement, like finding the time of day when wind speeds
  are highest.
* **Identifying "Hot Spots":** Pinpoint specific conditions where events
  of a certain magnitude are most likely to occur.

**Advantages of Polar View:**

* Makes cyclical patterns immediately obvious, which can be harder to
  spot in a standard Cartesian heatmap.
* Provides a compact and intuitive overview of a 2D distribution.

**Example:**
(See :ref:`Gallery <gallery_plot_polar_heatmap>` for code and plot examples)

.. raw:: html

    <hr>

.. _ug_plot_polar_quiver:

Visualizing Vector Fields (:func:`~kdiagram.plot.uncertainty.plot_polar_quiver`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**

This function creates a **polar quiver plot** to visualize vector data.
Each arrow on the plot represents a vector, showing both its **magnitude**
and **direction**. It is an excellent tool for understanding dynamic
processes like forecast revisions, error vectors, or physical flows.

**Mathematical Concept:**
Each arrow is a vector defined at an origin point in polar coordinates.

1. **Vector Origin:** The tail of each vector :math:`i` is placed at the
   polar coordinate :math:`(r_i, \theta_i)`, determined by the `r_col`
   and `theta_col`.

2. **Vector Components:** The vector itself is defined by its components
   in the local radial and tangential directions.

   * :math:`u_i` (from `u_col`) is the vector's component in the
     **radial** direction (pointing away from the center).
   * :math:`v_i` (from `v_col`) is the vector's component in the
     **tangential** direction (perpendicular to the radial line).

3. **Magnitude:** The color and/or length of the arrow typically
   represents the vector's Euclidean magnitude, :math:`M_i`.

   .. math::

      M_i = \sqrt{u_i^2 + v_i^2}

**Interpretation:**

  * **Arrow Position:** The base of the arrow shows the location where the
    vector originates.
  * **Arrow Direction:** The arrow points in the direction of the vector.
    For forecast revisions, an arrow pointing outward means the forecast
    was revised upward; an inward arrow means a downward revision.
  * **Arrow Length & Color:** The size and color of the arrow represent
    the magnitude of the vector. Longer, more intense arrows indicate
    stronger flows or larger changes.

**Use Cases:**

  * **Forecast Stability:** Visualize how much forecasts change between
    updates. Small, randomly oriented arrows suggest a stable model.
    Large, consistently oriented arrows might indicate model drift.
  * **Error Vector Analysis:** Plot the error as a vector pointing from
    the predicted value to the actual value.
  * **Flow Visualization:** Model physical phenomena like wind or ocean
    currents in a polar context.

**Advantages of Polar View:**

  * Provides an intuitive way to visualize vector fields that have a
    natural central point or cyclical nature.
  * Can reveal large-scale rotational or radial patterns in the vector
    data.

**Example:**
(See :ref:`Gallery <gallery_plot_polar_quiver>` for code and plot examples)
