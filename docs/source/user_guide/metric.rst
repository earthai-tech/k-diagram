.. _userguide_metrics:

=====================================
Specialized Forecasting Metrics
=====================================

Effective forecast evaluation requires metrics that move beyond
simple accuracy to capture the nuanced behavior of a model's
predictions, especially for probabilistic forecasts. The
:mod:`kdiagram.metrics` module provides specialized scoring
functions designed to quantify complex aspects of forecast
performance, such as the structure and severity of prediction
interval failures.

.. admonition:: From Theory to Practice: A Real-World Case Study
   :class: hint

   The metric described in this guide was developed to solve a
   practical challenge: quantifying forecast failures in a way that
   accounts for both their magnitude and their spatiotemporal
   clustering. For a detailed case study demonstrating how this
   metric is used to reveal model trade-offs, please refer to our
   research paper :footcite:p:`kouadioc2025`.

Summary of Anomaly Severity Metrics
-----------------------------------

.. list-table:: Anomaly Severity Metric Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.metrics.cluster_aware_severity_score`
     - A Scikit-learn compliant scorer that quantifies the
       severity and clustering of prediction interval failures.
   * - :func:`~kdiagram.metrics.clustered_anomaly_severity`
     - A simplified helper for direct CAS score calculation, often
       used internally by plotting functions.

.. raw:: html

   <hr>

.. _ug_cluster_aware_severity_score:

Cluster-Aware Severity (CAS) Score (:func:`~kdiagram.metrics.cluster_aware_severity_score`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function calculates the **Cluster-Aware Severity (CAS)
score**, a novel diagnostic metric designed to evaluate the
reliability of probabilistic forecasts. It moves beyond simple
coverage checks by penalizing not just the **magnitude** of
prediction interval failures (anomalies) but also their
**concentration** or clustering. It answers the critical
question: *"Are my model's failures small and random, or are
they large and systematic?"* A lower CAS score is better.

**Mathematical Concept**
The CAS score is a composite metric that integrates two key
components of forecast failure.

1.  **Anomaly Magnitude** (:math:`m_i`): This measures the
    severity of an individual failure. For a true value
    :math:`y_i` and a prediction interval
    :math:`[\hat{y}_{i,q_{lower}}, \hat{y}_{i,q_{upper}}]`, the
    magnitude is the absolute distance to the nearest violated
    bound.

    .. math::
       :label: cas_metric_mag

       m_i =
       \begin{cases}
         \hat{y}_{i,q_{lower}} - y_i & \text{if } y_i < \hat{y}_{i,q_{lower}} \\
         y_i - \hat{y}_{i,q_{upper}} & \text{if } y_i > \hat{y}_{i,q_{upper}}
       \end{cases}

    Optionally, this raw magnitude can be normalized by the
    prediction interval width (`normalize='band'`) or the
    Median Absolute Deviation (`normalize='mad'`) to assess
    relative severity.

2.  **Local Cluster Density** (:math:`d_i`): This quantifies
    the concentration of anomalies. The "clustering" is
    assessed along a meaningful sequence defined by the
    `sort_by` parameter (e.g., time or a spatial coordinate).
    The density is calculated using a centered rolling window of
    size `window_size` over a source vector (either anomaly
    magnitudes or a simple anomaly indicator).

3.  **Composite Score**: The per-sample **severity**
    :math:`s_i` combines magnitude and density. The overall CAS
    score is the weighted average of these severities.

    .. math::
       :label: cas_metric_sev

       s_i = m_i \cdot (1 + \lambda \cdot d_i^{\gamma})

    where :math:`\lambda` and :math:`\gamma` are parameters to
    control the penalty for clustering.

**Interpretation:**
A low CAS score is desirable, indicating that forecast failures
are infrequent, small in magnitude, and randomly scattered. A high
CAS score signals a potential issue:

* **High Magnitude**: The model produces large errors when its
  intervals fail.
* **High Density**: The model's failures are not random but are
  concentrated in specific regions of the data (e.g., during
  certain time periods or in particular spatial areas), suggesting
  a systematic bias.

**Use Cases:**

* To get a more nuanced evaluation of prediction intervals than
  simple coverage scores.
* To diagnose if a model's failures are systematic (clustered)
  or random (scattered).
* For model selection in high-stakes applications where clustered,
  severe errors are more costly than random noise.

While standard metrics tell us *if* a model is accurate on
average, the CAS score tells us *how* it fails. A model with a
good average score can still be untrustworthy if all its errors
are concentrated in one catastrophic, high-risk scenario. The CAS
score is designed specifically to detect this kind of structural,
systematic risk.

.. admonition:: Practical Example

   An insurance company uses a model to predict claim costs,
   providing an 80% confidence interval for each claim. A standard
   coverage metric shows the model is 82% accurate, which seems
   good. However, the CAS score is very high. Why?

   By visualizing the components of the score (using a plot like
   :func:`~kdiagram.plot.anomaly.plot_glyphs`), they discover
   that while the model is correct most of the time, all of its
   failures are concentrated on a specific type of high-value
   claim, and the magnitudes of these failures are catastrophic.
   The CAS score successfully flagged this hidden, systematic
   risk that the aggregate coverage metric missed.

   .. code-block:: pycon

      >>> import numpy as np
      >>> from kdiagram.metrics import cluster_aware_severity_score
      >>>
      >>> # --- 1. Simulate forecast data ---
      >>> y_true = np.array([10, 5, 10, 10, 25, 30])
      >>> # Anomalies are at index 1 and 4
      >>> y_pred = np.array(
      ...     [[8, 12], [6, 7], [8, 12], [8, 12], [26, 27], [28, 32]]
      ... )
      >>> # A sort vector that will group the anomalies together
      >>> sort_by_vec = np.array([10, 2, 30, 40, 3, 50])
      >>>
      >>> # --- 2. Calculate CAS score ---
      >>> # Score is low when anomalies are scattered in default order
      >>> score_scattered = cluster_aware_severity_score(
      ...     y_true, y_pred, window_size=3
      ... )
      >>> # Score is higher when anomalies are grouped by sorting
      >>> score_clustered = cluster_aware_severity_score(
      ...     y_true, y_pred, sort_by=sort_by_vec, window_size=3
      ... )
      >>>
      >>> print(f"Scattered Score: {score_scattered:.4f}")
      Scattered Score: 0.1111
      >>> print(f"Clustered Score: {score_clustered:.4f}")
      Clustered Score: 0.1778

   This example demonstrates the core function of the CAS score.
   Even with the same set of anomalies, the score increases when
   the `sort_by` parameter reveals that they are clustered in the
   feature space.

   **Quick Interpretation:**
   The `sort_by` parameter allows the CAS score to detect hidden
   patterns. The initial `score_scattered` is low because the
   anomalies at index 1 and 4 are far apart in the default data
   order. However, when sorted by `sort_by_vec`, these two points
   become neighbors. The `score_clustered` is consequently higher,
   correctly identifying that these failures are not random but
   are concentrated among samples with low `sort_by` values.

This ability to diagnose the structure of errors is crucial for
building trustworthy models. To see the full implementation and
explore visualizations of this metric, please visit the gallery.

**Example:**
See the gallery example and code at :ref:`gallery_plot_cas_layers`.

.. raw:: html

   <hr>

.. _ug_clustered_anomaly_severity_helper:

CAS Score Helper Function (:func:`~kdiagram.metrics.clustered_anomaly_severity`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function is a simplified helper for calculating the
**Clustered Anomaly Severity (CAS) score**. It offers a direct
and convenient interface for computing the score and its
intermediate components, making it ideal for use within plotting
functions or for quick, interactive analysis. It supports two
convenient input patterns: providing raw array-like objects, or
providing a pandas DataFrame along with the relevant column names.

**Mathematical Concept:**
This function is a simplified wrapper that computes the CAS
score with a fixed set of robust default settings. The core
calculation relies on two primary components:

1.  **Anomaly Magnitude** (:math:`m_i`): The severity of an
    individual forecast failure, measured as the distance from
    the true value to the nearest violated interval bound 
    (See Eq. :eq:`cas_metric_mag`).

2.  **Local Cluster Density** (:math:`d_i`): The concentration
    of failures, calculated using a rolling average of anomaly
    magnitudes over a specified `window_size`.

The final severity for each point is :math:`s_i = m_i \cdot d_i`,
and the overall CAS score is the mean of these severities. For a
more detailed mathematical breakdown and advanced configuration,
please see the main scorer function,
:func:`~kdiagram.metrics.cluster_aware_severity_score`.

**Interpretation:**
A low CAS score is desirable, indicating that forecast failures
are infrequent, small in magnitude, and randomly scattered. A high
CAS score suggests that the model's failures are either very
large, highly concentrated in specific regions of the data, or
both—a sign of systematic bias.

**Use Cases:**

* As an internal helper function for creating diagnostic plots like
  :func:`~kdiagram.plot.anomaly.plot_glyphs`.
* For rapid, interactive data exploration in a notebook environment
  where the full Scikit-learn API is not required.
* To quickly get the detailed intermediate calculations
  (`magnitude`, `local_density`, etc.) for custom analysis by
  setting `return_details=True`.

.. admonition:: Practical Example

   An analyst wants to quickly diagnose a forecast stored in a pandas
   DataFrame. Instead of extracting each column into a NumPy array,
   they can use this helper to directly reference the columns by
   name and get both the final CAS score and the detailed, per-sample
   breakdown for further plotting or investigation.

   .. code-block:: pycon

      >>> import numpy as np
      >>> import pandas as pd
      >>> from kdiagram.metrics import clustered_anomaly_severity
      >>>
      >>> # --- 1. Create a sample DataFrame ---
      >>> y_true = np.array([10, 25, 30, 45, 50])
      >>> y_qlow = np.array([8, 24, 32, 44, 48])
      >>> y_qup = np.array([12, 26, 33, 46, 52])
      >>> df = pd.DataFrame({
      ...     'actual': y_true,
      ...     'lower_bound': y_qlow,
      ...     'upper_bound': y_qup
      ... })
      >>>
      >>> # --- 2. Calculate CAS score and get details ---
      >>> cas_score, details_df = clustered_anomaly_severity(
      ...     'actual', 'lower_bound', 'upper_bound',
      ...     data=df, window_size=3, return_details=True
      ... )
      >>>
      >>> print(f"CAS Score: {cas_score:.4f}")
      CAS Score: 0.2222
      >>> print(details_df[['is_anomaly', 'magnitude', 'local_density']])
         is_anomaly  magnitude  local_density
      0       False        0.0       0.000000
      1       False        0.0       0.666667
      2        True        2.0       0.666667
      3       False        0.0       0.666667
      4       False        0.0       0.000000

**Example:**
See the gallery example and code at :ref:`gallery_plot_glyphs`.

.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::

