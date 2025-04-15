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
*(Note: Ensure the label `_gallery_plot_model_comparison` exists before
the corresponding example in your gallery file, likely
`gallery/evaluation.rst` or `gallery/comparison.rst` if you create one.)*