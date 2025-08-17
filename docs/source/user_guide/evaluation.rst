.. _userguide_evaluation:

=============================================
Model Evaluation with Taylor Diagrams
=============================================

Evaluating the performance of forecast or simulation models often
requires considering multiple aspects simultaneously. How well does the
model capture the overall variability (standard deviation) of the observed
phenomenon? How well does the pattern of the model's output correlate
with the observed pattern? A Taylor Diagram, developed by Karl E. Taylor
:cite:t:`Taylor2001`, provides an elegant solution by graphically summarizing these
key statistics in a single, concise plot.

Taylor diagrams are widely used, particularly in climate science and
meteorology, but are applicable to any field where model outputs need
rigorous comparison against a reference dataset (observations). They
allow for the simultaneous assessment of correlation, standard
deviation, and (implicitly) the centered root-mean-square difference
(RMSD) between different models and the reference.

`k-diagram` provides flexible functions to generate these informative
diagrams.

Summary of Evaluation Functions
----------------------------------

The following functions generate variations of the Taylor Diagram:

.. list-table:: Taylor Diagram Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.evaluation.taylor_diagram`
     - Flexible Taylor Diagram plotter; accepts pre-computed statistics
       (std. dev., correlation) or raw prediction/reference arrays.
       Includes options for background shading based on different
       weighting strategies.
   * - :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`
     - Taylor Diagram plotter featuring a background colormap encoding
       correlation or performance zones, with specific shading strategies.
       Requires raw prediction/reference arrays.
   * - :func:`~kdiagram.plot.evaluation.plot_taylor_diagram`
     - A potentially simpler interface for plotting Taylor Diagrams,
       requiring raw prediction/reference arrays. (May share features
       with the other functions).


Interpreting Taylor Diagrams
-------------------------------

Regardless of the specific function used, interpreting a Taylor Diagram
involves looking at the position of points (representing models or
predictions) relative to the reference point and the axes:

* **Reference Point/Arc:** Typically marked on the horizontal axis (at
  angle 0) or as an arc. Its radial distance from the origin represents
  the standard deviation of the reference (observed) data (:math:`\sigma_r`).
* **Radial Axis (Distance from Origin):** Represents the standard
  deviation of the prediction (:math:`\sigma_p`). Models with standard
  deviations similar to the reference will lie near the reference arc.
* **Angular Axis (Angle from Horizontal/Reference):** Represents the
  correlation coefficient (:math:`\rho`) between the prediction and the
  reference, usually via the relation :math:`\theta = \arccos(\rho)`.
  Points closer to the horizontal axis (smaller angle) have higher
  correlations.
* **Distance to Reference Point:** The *straight-line distance* between a
  model point and the reference point is proportional to the centered
  Root Mean Square Difference (RMSD) between the prediction and the
  reference.
* **Overall Skill:** Generally, models plotted closer to the reference
  point are considered more skillful, indicating a better balance of
  correlation and amplitude of variations (standard deviation).

Detailed Explanations
------------------------

Let's explore the specific functions.

.. _ug_taylor_diagram:

Flexible Taylor Diagram (:func:`~kdiagram.plot.evaluation.taylor_diagram`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function provides a highly flexible way to generate Taylor Diagrams.
It uniquely accepts either **pre-computed statistics** (standard
deviations and correlation coefficients) or the **raw data arrays**
(predictions and reference) from which it calculates these statistics
internally. It also offers several strategies for adding an optional
**background color mesh** to highlight specific regions of the diagram.

**Mathematical Concept:**
The plot is based on the geometric relationship between the standard
deviations of the reference (:math:`\sigma_r`) and prediction
(:math:`\sigma_p`), their correlation coefficient (:math:`\rho`), and the
centered Root Mean Square Difference (RMSD):

.. math::

   RMSD^2 = \sigma_p^2 + \sigma_r^2 - 2\sigma_p \sigma_r \rho

On the diagram:

 * Radius (distance from origin) = :math:`\sigma_p`
 * Angle (from reference axis) = :math:`\theta = \arccos(\rho)`
 * Distance from Reference Point = RMSD

**Interpretation:**

* Evaluate model points based on their proximity to the reference point
    (lower RMSD is better), their angular position (lower angle means
    higher correlation), and their radial position relative to the
    reference arc/point (matching standard deviation is often desired).
* If `cmap` is used, the background shading provides additional context
  based on the `radial_strategy`:
    
  * `'rwf'`: Emphasizes points with high correlation *and* standard
    deviation close to the reference.
  * `'convergence'` / `'norm_r'`: Simple radial gradients.
  * `'center_focus'`: Highlights a central region.
  * `'performance'`: Highlights the area around the best-performing
    point based on correlation and std. dev. matching the reference.

**Use Cases:**

* Comparing multiple model results when only summary statistics
  (std. dev., correlation) are available.
* Generating standard Taylor diagrams from raw model output and
  observation arrays.
* Creating visually enhanced diagrams with background shading to guide
  interpretation towards specific performance criteria.
* Customizing the appearance of the reference marker and plot labels.

**Advantages:**

* High flexibility in accepting either pre-computed statistics or raw
  data arrays.
* Offers multiple strategies for informative background shading to
  enhance interpretation.
* Provides options for customizing reference display and label sizes.


**Example:** :ref:`View Gallery Example <gallery_plot_taylor_diagram_flexible>`

.. _ug_plot_taylor_diagram_in:

Taylor Diagram with Background Shading (:func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function specializes in creating Taylor Diagrams with a prominent
**background color mesh** that visually encodes the correlation domain or
other performance metrics. It requires raw prediction and reference arrays
as input and offers specific strategies for generating the background.

**Mathematical Concept:**
Same fundamental relationship as `taylor_diagram`: maps standard
deviation (:math:`\sigma_p`) to radius and correlation (:math:`\rho`) to
angle (:math:`\theta = \arccos(\rho)`). The key feature is the generation
of the background color field `CC` based on `radial_strategy`:

* `'convergence'`: :math:`CC = \cos(\theta)` (directly maps correlation).
* `'norm_r'`: :math:`CC = r / \max(r)` (maps normalized radius).
* `'performance'`: :math:`CC = \exp(-(\sigma_p - \sigma_{best})^2 / \epsilon_\sigma) \cdot \exp(-(\theta - \theta_{best})^2 / \epsilon_\theta)`
    (Gaussian-like function centered on the best model point).

**Interpretation:**

* Interpret model points relative to the reference point/arc as described
  in the general interpretation guide.
* The **background color** provides context:

  * With `'convergence'`, colors directly map to correlation values
    (e.g., warmer colors for higher correlation).
  * With `'norm_r'`, colors show relative standard deviation.
  * With `'performance'`, the brightest color highlights the region
    closest to the best-performing input model.
* The `zero_location` and `direction` parameters change the orientation
  of the plot, affecting where correlation=1 appears and whether angles
  increase clockwise or counter-clockwise.

**Use Cases:**

* Creating visually rich Taylor diagrams where the background emphasizes
  correlation levels or proximity to the best model.
* Comparing models when a strong visual cue for correlation or relative
  performance across the diagram space is desired.
* Generating diagrams with specific orientations (e.g., correlation=1 at
  the top North position).

**Advantages:**

* Provides built-in, visually informative background shading options
  focused on correlation or performance.
* Offers fine control over plot orientation (`zero_location`, `direction`).

**Example:** :ref:`View Gallery Example <gallery_plot_taylor_diagram_background_shading_focus>`

.. _ug_plot_taylor_diagram:

Basic Taylor Diagram (:func:`~kdiagram.plot.evaluation.plot_taylor_diagram`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function appears to offer a potentially simpler interface for
generating a standard Taylor Diagram, requiring raw prediction and
reference arrays as input. It compares models based on standard
deviation (radius) and correlation (angle).

**Mathematical Concept:**
Utilizes the same core principles as the other Taylor diagram functions,
mapping standard deviation (:math:`\sigma_p`) to the radial coordinate
and correlation (:math:`\rho`) to the angular coordinate
(:math:`\theta = \arccos(\rho)`).

**Interpretation:**

* Interpret points based on their standard deviation (radius),
  correlation (angle), and distance to the reference point (RMSD) as
  outlined in the general interpretation guide above.
* Customization options like `zero_location`, `direction`, and
  `angle_to_corr` allow tailoring the plot's appearance and labeling.

**Use Cases:**

* Generating standard Taylor diagrams for model evaluation when background
  shading is not required.
* Comparing multiple predictions against a common reference based on
  correlation and standard deviation.

**Advantages:**

* May offer a more streamlined interface if fewer customization options
  are needed compared to `taylor_diagram` or `plot_taylor_diagram_in`.

**Example:** :ref:`View Gallery Example <gallery_plot_taylor_diagram_basic>`

.. raw:: html

   <hr>

.. rubric:: References

.. bibliography::
   :style: plain