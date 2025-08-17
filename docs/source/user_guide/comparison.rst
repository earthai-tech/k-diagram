.. _userguide_comparison:

==================================
Model Comparison Visualization 
==================================

Comparing the performance of different forecasting or simulation models
is a common task in model development and selection. Often, evaluation requires
looking at multiple performance metrics simultaneously to understand
the trade-offs and overall suitability of each model for a specific
application.

The :mod:`kdiagram.plot.comparison` module provides tools specifically
for this purpose, currently featuring radar charts for multi-metric,
multi-model comparisons.

Summary of Comparison Functions
-------------------------------

.. list-table:: Model Comparison Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.comparison.plot_model_comparison`
     - Generates a radar chart comparing multiple models across
       various performance metrics (e.g., R2, MAE, Accuracy).
   * - :func:`~kdiagram.plot.comparison.plot_reliability_diagram`
     - Draws a reliability (calibration) diagram to assess how
       well predicted probabilities match observed frequencies.
   * - :func:`~kdiagram.plot.comparison.plot_horizon_metrics`
     - Draw a polar bar chart to visually compare key metrics across
       a set of distinct categories.

Detailed Explanations
---------------------

Let's explore the model comparison function.

.. _ug_plot_model_comparison:

Multi-Metric Model Comparison (:func:`~kdiagram.plot.comparison.plot_model_comparison`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function generates a **radar chart** (also known as a spider
or star chart) to visually compare the performance of **multiple
models** across **multiple evaluation metrics** simultaneously. It
provides a holistic snapshot of model strengths and weaknesses,
making it easier to select the best model based on criteria beyond
a single score. Optionally, training time can be included as an
additional comparison axis.

**Mathematical Concept:**

For each model :math:`k` (with predictions :math:`\hat{y}_k`) and
each chosen metric :math:`m`, a score :math:`S_{m,k}` is calculated
using the true values :math:`y_{true}`:

.. math::
    S_{m,k} = \text{Metric}_m(y_{true}, \hat{y}_k)

The metrics used can be standard ones (like R2, MAE, Accuracy, F1)
or custom functions. If `train_times` are provided, they are
treated as another dimension.

The scores for each metric :math:`m` are typically **scaled** across
the models (using `scale='norm'` for Min-Max or `scale='std'` for
Standard Scaling) before plotting, to bring potentially different
metric ranges onto a comparable radial axis:

.. math::
   S'_{m,k} = \text{Scale}(S_{m,1}, S_{m,2}, ..., S_{m,n_{models}})_k

Each metric :math:`m` is assigned an angle :math:`\theta_m` on the
radar chart, and the scaled score :math:`S'_{m,k}` determines the
radial distance along that axis for model :math:`k`. These points
are connected to form a polygon representing each model's overall
performance profile.

**Interpretation:**

* **Axes:** Each axis radiating from the center represents a
  different performance metric (e.g., 'r2', 'mae', 'accuracy',
  'train_time_s').
* **Polygons:** Each colored polygon corresponds to a different model,
  as indicated by the legend.
* **Radius:** The distance from the center along a metric's axis
  shows the model's (potentially scaled) score for that metric.
  
  * **Important:** By default (`scale='norm'` with internal inversion
    for error metrics), a **larger radius generally indicates
    better performance** (higher score for accuracy/R2, lower score
    for MAE/RMSE/MAPE/time after inversion during scaling). Check
    the `scale` parameter used. If `scale=None`, interpret radius
    based on the raw metric values.
* **Shape Comparison:** Compare the overall shapes and sizes of the
  polygons. A model with a consistently large polygon across multiple
  desirable metrics might be considered the best overall performer.
  Different shapes highlight trade-offs (e.g., one model might excel
  in R2 but be slow, while another is fast but has lower R2).

**Use Cases:**

* **Multi-Objective Model Selection:** Choose the best model when
  performance needs to be balanced across several, potentially
  conflicting, metrics (e.g., high accuracy vs. low error vs.
  fast training time).
* **Visualizing Strengths/Weaknesses:** Quickly identify which metrics
  a particular model excels or struggles with compared to others.
* **Communicating Comparative Performance:** Provide stakeholders with
  an intuitive visual summary of how different candidate models stack
  up against each other based on chosen criteria.
* **Comparing Regression and Classification:** Use appropriate default
  or custom metrics to compare models for either task type.

**Advantages (Radar Context):**

* Effectively displays multiple performance dimensions (>2) for
  multiple entities (models) in a single, relatively compact plot.
* Allows direct comparison of the *profiles* of different models
  – are they generally good/bad, or strong in some areas and weak
  in others?
* Facilitates the identification of trade-offs between different metrics.

**Example:**
(See the :ref:`Model Comparison Example <gallery_plot_model_comparison>`
in the Gallery)

.. raw:: html

   <hr>


.. _ug_plot_reliability:

Reliability Diagram (:func:`~kdiagram.plot.comparison.plot_reliability_diagram`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function draws a **reliability (calibration) diagram**, a standard
method in forecast verification :cite:t:`Jolliffe2012`, to assess how
well **predicted probabilities** match **observed frequencies**. It supports
one or many models on the same figure, multiple binning strategies, optional
error bars (e.g., Wilson intervals), and a counts panel for diagnosing data
sparsity across probability ranges.

**Mathematical Concept:**
Given binary labels :math:`y_j \in \{0,1\}` and predicted probabilities
:math:`p_j \in [0,1]` (optionally with per-sample weights
:math:`w_j \ge 0`), probabilities are partitioned into bins via a
binning rule :math:`b(\cdot)` (uniform or quantile).

For bin :math:`i`, define the (weighted) bin weight

.. math::
   W_i \;=\; \sum_{j=1}^{N} w_j \, \mathbf{1}\{ b(p_j) = i \}, 
   \qquad
   W \;=\; \sum_{i} W_i \;=\; \sum_{j=1}^{N} w_j.

Within each bin, compute the **mean confidence** (x–axis) and **observed
frequency** (y–axis):

.. math::
   \mathrm{conf}_i \;=\; 
   \frac{1}{W_i} \sum_{j=1}^{N} w_j \, p_j \, \mathbf{1}\{ b(p_j)=i \},
   \qquad
   \mathrm{acc}_i \;=\;
   \frac{1}{W_i} \sum_{j=1}^{N} w_j \, y_j \, \mathbf{1}\{ b(p_j)=i \}.

Each bin yields a point :math:`(\mathrm{conf}_i, \mathrm{acc}_i)`. A perfectly
calibrated model satisfies :math:`\mathrm{acc}_i \approx \mathrm{conf}_i` for
all bins, i.e., points lie on the diagonal :math:`y=x`.

**Uncertainty in observed frequency.**
When :math:`W_i` is sufficiently large, a normal approximation can be used for
:math:`\mathrm{acc}_i` with standard error

.. math::
   \mathrm{SE}_i \;\approx\; 
   \sqrt{ \frac{\mathrm{acc}_i \, (1-\mathrm{acc}_i)}{W_i} }.

Alternatively, the **Wilson interval** (95%) for a binomial proportion with
:math:`z = 1.96` provides a more stable interval, especially for small counts:

.. math::
   \hat{p} \;=\; \mathrm{acc}_i, \quad
   n \;=\; W_i, \quad
   \tilde{p} \;=\; \frac{\hat{p} + \frac{z^2}{2n}}
                         {1 + \frac{z^2}{n}}, \quad
   \mathrm{half\_width} \;=\;
   \frac{z}{1+\frac{z^2}{n}} 
   \sqrt{ \frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2} }.

.. math::
   \mathrm{CI}_i \;=\; 
   \Big[\, \tilde{p} - \mathrm{half\_width},\;
           \tilde{p} + \mathrm{half\_width} \,\Big].

(With sample weights, :math:`n` is treated as an **effective count**.)

**Aggregate calibration metrics.**

* **Expected Calibration Error (ECE)** (L1 form):

  .. math::
     \mathrm{ECE} \;=\; \sum_{i} \frac{W_i}{W} 
     \;\big|\mathrm{acc}_i - \mathrm{conf}_i\big|.

* **Maximum Calibration Error (MCE)** (optional concept):

  .. math::
     \mathrm{MCE} \;=\; \max_i \;\big|\mathrm{acc}_i - \mathrm{conf}_i\big|.

* **Brier score** (mean squared error on probabilities):

  .. math::
     \mathrm{Brier} \;=\; 
     \frac{1}{W}\sum_{j=1}^{N} w_j \, (p_j - y_j)^2.
  
Lower ECE/MCE/Brier indicate better calibration (and accuracy for Brier).

**Interpretation:**

* **Diagonal (:math:`y=x`):** Reference for perfect calibration.

  * Points **above** diagonal :math:`(\mathrm{acc}_i > \mathrm{conf}_i)`
    ⇒ model is **under-confident** in that bin.
  * Points **below** diagonal :math:`(\mathrm{acc}_i < \mathrm{conf}_i)`
    ⇒ model is **over-confident** in that bin.
* **Counts panel:** A histogram of :math:`p_j` per bin reveals data
  coverage; sparse bins tend to have larger uncertainty intervals.
* **Multiple models:** Curves are overlaid; compare proximity to
  the diagonal and reported ECE/Brier in the legend.

**Binning strategies:**

* **Uniform:** fixed-width bins on :math:`[0,1]` (e.g., 10 bins).
* **Quantile:** bins formed so each has (approximately) equal counts.
  This stabilizes variance of :math:`\mathrm{acc}_i` but can yield
  irregular edges if many identical scores occur.

**Use Cases:**

* **Calibrating classifiers** that output probabilities (logistic regression,
  gradient boosting, neural nets).
* **Comparing models or calibration methods** (e.g., Platt scaling vs.
  isotonic regression).
* **Communicating reliability**: the diagram shows at a glance if a model
  is systematically over-/under-confident and where.

**Advantages:**

* **Local view** of calibration (per bin) instead of a single scalar.
* **Uncertainty-aware** via bin-wise intervals.
* **Distribution-aware** with the counts panel, showing score sharpness
  and data coverage.

**Example:**
(See the :ref:`Gallery example <gallery_plot_reliability>` for a complete,
runnable snippet that saves an image and returns per-bin statistics.)



.. raw:: html

   <hr>

.. _ug_plot_horizon_metrics:

Comparing Metrics Across Horizons (:func:`~kdiagram.plot.comparison.plot_horizon_metrics`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a polar bar chart, a novel visualization
developed as part of the analytics framework in :cite:t:`kouadiob2025`,
to visually compare key metrics across a set of distinct categories,
most commonly different forecast horizons (e.g., H+1, H+2, etc.).
It is designed to answer questions like: "How does my model's
uncertainty (interval width) and central tendency (median prediction)
evolve as it forecasts further into the future?"

**Mathematical Concept:**
The plot summarizes metrics for :math:`N` horizons (corresponding to
the rows in the input `df`) using data from :math:`M` samples
(corresponding to the provided columns for each quantile). Let the
input data be represented by matrices for the lower, upper, and
median quantiles: :math:`\mathbf{L}`, :math:`\mathbf{U}`, and
:math:`\mathbf{Q50}`, all of shape :math:`(N, M)`.

1.  **Interval Width Calculation**: First, a matrix of interval
    widths :math:`\mathbf{W}` of shape :math:`(N, M)` is computed by
    element-wise subtraction. Each element :math:`W_{j,i}`
    represents the interval width for horizon :math:`j` and sample
    :math:`i`.

    .. math::

        W_{j,i} = U_{j,i} - L_{j,i}

2.  **Radial Value (Bar Height)**: The primary metric plotted as the
    bar height (radial value :math:`r_j`) for each horizon :math:`j`
    is the **mean** of its interval widths across all :math:`M`
    samples.

    .. math::

        r_j = \frac{1}{M} \sum_{i=0}^{M-1} W_{j,i}

    If `normalize_radius=True`, these values are then min-max scaled
    to the range `[0, 1]`.

3.  **Color Value**: The secondary metric, encoded as color, is the
    **mean of the Q50 values** for each horizon :math:`j`.

    .. math::

        c_j = \frac{1}{M} \sum_{i=0}^{M-1} Q50_{j,i}

    If `q50_cols` are not provided, the color value defaults to the
    radial value, :math:`c_j = r_j`. These color values are then mapped
    to a colormap via a standard normalization.

**Interpretation:**

* **Angle:** Each angular segment represents a different horizon or
  category, as specified by the ``xtick_labels`` parameter. The plot typically
  starts at the top (12 o'clock) and proceeds clockwise.
* **Radius (Bar Height):** The length of each bar indicates the
  magnitude of the primary metric (e.g., **mean interval width**).
  Longer bars signify larger values.
* **Color:** The color of each bar represents the magnitude of the
  secondary metric (e.g., **mean Q50 value**). The color bar on the
  side of the plot provides the scale for this metric.

**Use Cases:**

* **Analyzing Uncertainty Drift:** Track how a model's predictive
  uncertainty (interval width) grows or shrinks over a forecast horizon.
* **Comparing Forecast Magnitudes:** Simultaneously visualize how the
  central tendency (Q50) of the forecast changes along with its
  uncertainty.
* **Comparing Models:** Generate this plot for multiple models to
  compare their uncertainty profiles over time. A model with shorter,
  more stable bars may be preferable.
* **Categorical Performance:** The "horizons" can represent any set of
  categories, such as different geographic regions or model configurations,
  to compare aggregated metrics.

**Advantages (Polar Bar Context):**

* **Intuitive Comparison:** The circular layout allows for easy comparison
  of values across sequential categories.
* **Two-Dimensional Insight:** It effectively encodes two different
  metrics (bar height and bar color) for each category in a single,
  compact plot.
* **Highlights Trends:** Trends across horizons, such as consistently
  increasing uncertainty, are immediately apparent.

**Example:**
(See the :ref:`Horizon Metrics Example <gallery_plot_horizon_metrics>`
in the Gallery)

.. raw:: html

   <hr>

.. rubric:: References

.. bibliography::
   :style: plain
   :filter: cited