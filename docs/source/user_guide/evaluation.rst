.. _userguide_evaluation:

====================================
Evaluating Classification Models
====================================

Evaluating the performance of classification models is a crucial
step in the machine learning workflow. Beyond a single accuracy
score, a thorough evaluation requires understanding a model's
ability to distinguish between classes, its performance on
imbalanced data, and its specific types of errors.

The :mod:`kdiagram.plot.evaluation` module provides a suite of
visualizations for this purpose, featuring novel polar adaptations
of standard, powerful diagnostic tools like the ROC curve, the
Precision-Recall curve, and the confusion matrix. These plots
provide an intuitive and aesthetically engaging way to compare
the performance of multiple models and diagnose their strengths
and weaknesses.

Summary of Evaluation Functions
-------------------------------

.. list-table:: Classification Evaluation Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.plot.evaluation.plot_polar_roc`
     - Draws a Polar Receiver Operating Characteristic (ROC) curve
       to assess a model's discriminative ability.
   * - :func:`~kdiagram.plot.evaluation.plot_polar_pr_curve`
     - Draws a Polar Precision-Recall (PR) curve, ideal for
       evaluating models on imbalanced datasets.
   * - :func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix`
     - Visualizes the four components of a binary confusion matrix
       as a polar bar chart.
   * - :func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix_in`
     - Visualizes a multiclass confusion matrix using a grouped
       polar bar chart to show per-class predictions.
   * - :func:`~kdiagram.plot.evaluation.plot_polar_classification_report`
     - Displays a detailed per-class report of Precision, Recall,
       and F1-Score on a polar plot.
   * - :func:`~kdiagram.plot.evaluation.plot_pinball_loss`
     - Visualizes the Pinball Loss for each quantile of a
       probabilistic forecast.
   * - :func:`~kdiagram.plot.evaluation.plot_regression_performance`
     - Visualizes the performance of multiple regression models through 
       grouped polar bar chart. 
       
.. _ug_plot_polar_roc:

Polar ROC Curve (:func:`~kdiagram.plot.evaluation.plot_polar_roc`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Receiver Operating Characteristic
(ROC) Curve**, a novel visualization for evaluating the performance
of binary classification models. It adapts the standard ROC curve,
a fundamental tool in machine learning, to a more intuitive and
aesthetically engaging polar format :footcite:p:`scikit-learn`.

**Mathematical Concept:**
A Receiver Operating Characteristic (ROC) curve is a standard
tool for evaluating binary classifiers :footcite:p:`Powers2011`.
It is created by plotting the **True Positive Rate (TPR)** against
the **False Positive Rate (FPR)** at various threshold settings.

.. math::
   :label: eq:tpr_fpr

   \text{TPR} = \frac{TP}{TP + FN} \quad , \quad
   \text{FPR} = \frac{FP}{FP + TN}

The novelty of this plot, developed as part of the analytics
framework in :footcite:p:`kouadiob2025`, lies in its
transformation of these Cartesian coordinates into a polar system.
The mapping is defined as:

.. math::
   :label: eq:roc_polar_transform

   \begin{aligned}
     \theta &= \text{FPR} \cdot \frac{\pi}{2} \\
     r &= \text{TPR}
   \end{aligned}

This transformation maps the standard ROC space onto a 90-degree
polar quadrant:

- The **angle (θ)** is mapped to the False Positive Rate,
  spanning from 0 at 0° to 1 at 90°.
- The **radius (r)** is mapped to the True Positive Rate,
  spanning from 0 at the center to 1 at the edge.

Under this transformation, the standard y=x "no-skill" line becomes
a perfect Archimedean spiral.

**Interpretation:**
The plot provides an intuitive visual assessment of a classifier's
discriminative power.

* **No-Skill Spiral (Dashed Line)**: This is the polar equivalent
  of the y=x diagonal in a standard ROC plot. A model with no
  discriminative power would lie on this line.
* **Model Curve**: Each colored line represents a model. A better
  model will have a curve that bows outwards, away from the
  no-skill spiral, maximizing the area under the curve (AUC).
* **Performance**: A model is superior if its curve achieves a
  high True Positive Rate (large radius) for a low False
  Positive Rate (small angle).

**Use Cases:**

* To evaluate and compare the overall discriminative power of
  binary classification models.
* To select an optimal classification threshold based on the
  desired balance between the True Positive Rate and False
  Positive Rate.
* To create a more visually engaging and compact representation of
  ROC performance for reports and presentations.


**Example**
See the gallery example and code: :ref:`gallery_plot_polar_roc`.

.. _ug_plot_polar_pr_curve:

Polar Precision-Recall Curve (:func:`~kdiagram.plot.evaluation.plot_polar_pr_curve`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Precision-Recall (PR) Curve**, a
novel visualization for evaluating binary classification models. It
is particularly useful for tasks with **imbalanced classes** (e.g.,
fraud detection, medical diagnosis), where ROC curves can sometimes
provide an overly optimistic view of performance.


**Mathematical Concept:**
A Precision-Recall curve is a standard tool for evaluating binary
classifiers :footcite:p:`Powers2011`. It is created by plotting
**Precision** against **Recall** at various threshold settings.

.. math::
   :label: eq:pr_curve

   \text{Precision} = \frac{TP}{TP + FP} \quad , \quad
   \text{Recall} = \frac{TP}{TP + FN}

The novelty of this plot, developed as part of the analytics
framework in :footcite:p:`kouadiob2025`, lies in its
transformation of these Cartesian coordinates into a polar system.
The mapping is defined as:

.. math::
   :label: eq:pr_polar_transform

   \begin{aligned}
     \theta &= \text{Recall} \cdot \frac{\pi}{2} \\
     r &= \text{Precision}
   \end{aligned}

This transformation maps the standard PR space onto a 90-degree
polar quadrant:

- The **angle (θ)** is mapped to **Recall**, spanning from 0 at
  0° to 1 at 90°.
- The **radius (r)** is mapped to **Precision**, spanning from 0
  at the center to 1 at the edge.

A "no-skill" classifier is represented by a circle at a radius
equal to the proportion of positive samples in the dataset.


**Interpretation:**
The plot provides an intuitive visual assessment of a classifier's
performance on the positive class.

* **No-Skill Circle (Dashed Line)**: Represents a random
  classifier. A good model's curve should be far outside this
  circle.
* **Model Curve**: Each colored line represents a model. A better
  model will have a curve that bows outwards towards the top-right
  of the plot, maximizing the area under the curve (Average
  Precision).
* **Performance**: A model is superior if it maintains a high
  Precision (large radius) as it achieves a high Recall (wide
  angular sweep).


**Use Cases:**

* To evaluate and compare binary classifiers on **imbalanced
  datasets** where the number of negative samples far outweighs
  the positive samples.
* To understand the trade-off between a model's ability to
  correctly identify positive cases (Recall) and its ability to
  avoid false alarms (Precision).
* To compare models based on their Average Precision (AP) score,
  which is summarized by the area under the PR curve.


**Example**
See the gallery example and code: :ref:`gallery_plot_polar_pr_curve`.

.. _ug_plot_polar_confusion_matrix:

Polar Confusion Matrix (:func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**
This function creates a **Polar Confusion Matrix**, a novel
visualization for the four key components of a binary confusion
matrix: True Positives (TP), False Positives (FP), True Negatives
(TN), and False Negatives (FN). It provides an intuitive,
at-a-glance summary of a classifier's performance and allows for
the direct comparison of multiple models.


**Mathematical Concept:**
The confusion matrix is a fundamental tool for evaluating a
classifier's performance by summarizing the counts of correct and
incorrect predictions for each class. This plot maps these
four components onto a polar bar chart.

- **True Positives (TP)**: Correctly predicted positive cases.
- **False Positives (FP)**: Negative cases incorrectly predicted as positive (Type I error).
- **True Negatives (TN)**: Correctly predicted negative cases.
- **False Negatives (FN)**: Positive cases incorrectly predicted as negative (Type II error).

Each of these four categories is assigned its own angular sector,
and the height (radius) of the bar in that sector represents the
count or proportion of samples in that category.


**Interpretation:**
The plot provides an immediate visual summary of a binary
classifier's strengths and weaknesses.

* **Angle**: Each of the four angular sectors represents a
  component of the confusion matrix.
* **Radius**: The length of each bar represents the **proportion**
  (if normalized) or **count** of samples in that category.
* **Ideal Performance**: A good model will have very **long bars**
  in the "True Positive" and "True Negative" sectors and very
  **short bars** in the "False Positive" and "False Negative"
  sectors.


**Use Cases:**

* To get a quick, visual summary of a binary classifier's
  performance.
* To directly compare the error types (False Positives vs. False
  Negatives) of multiple models.
* To create a more visually engaging and intuitive representation
  of a confusion matrix for reports and presentations.


**Example**
See the gallery example and code:
:ref:`gallery_plot_polar_confusion_matrix`.

.. _ug_plot_polar_confusion_matrix_in:

Multiclass Polar Confusion Matrix (:func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix_in`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Grouped Polar Bar Chart** to visualize
the performance of a multiclass classifier. It provides an
intuitive, at-a-glance summary of the confusion matrix by
showing how samples from each true class are distributed among
the predicted classes :footcite:p:`scikit-learn`.


**Mathematical Concept**
This plot is a novel visualization of the standard confusion
matrix, :math:`\mathbf{C}`, a fundamental tool for evaluating a
classifier's performance. Each element :math:`C_{ij}` of the
matrix contains the number of observations known to be in class
:math:`i` but predicted to be in class :math:`j`.

This function maps this matrix to a polar plot:

1.  **Angular Sectors**: The polar axis is divided into :math:`K`
    sectors, where :math:`K` is the number of classes. Each
    sector corresponds to a **true class** :math:`i`.

2.  **Grouped Bars**: Within each sector for true class :math:`i`,
    a set of :math:`K` bars is drawn. The height (radius) of the
    :math:`j`-th bar corresponds to the value of :math:`C_{ij}`,
    representing the count or proportion of samples from true
    class :math:`i` that were predicted as class :math:`j`.

**Interpretation:**
The plot makes it easy to identify a model's strengths and
weaknesses on a per-class basis.

* **Angle**: Each major angular sector represents a **True
    Class** (e.g., "True Class A").
* **Bars**: Within each sector, the different colored bars show
    how the samples from that true class were **predicted**. The
    legend indicates which color corresponds to which predicted
    class.
* **Radius**: The length of each bar represents the **proportion**
    (if normalized) or **count** of samples.
* **Ideal Performance**: A good model will have tall bars that
    match the sector's true class (e.g., the "Predicted Class A"
    bar is tallest in the "True Class A" sector) and very short
    bars for all other predicted classes.

**Use Cases:**

* To get a detailed, visual summary of a multiclass
    classifier's performance.
* To quickly identify which classes a model struggles with the
    most.
* To understand the specific patterns of confusion between
    classes (e.g., "Is Class A more often confused with B or C?").


**Example**
See the gallery example and code:
:ref:`gallery_plot_polar_confusion_matrix_in`.

.. _ug_plot_polar_classification_report:

Polar Classification Report (:func:`~kdiagram.plot.evaluation.plot_polar_classification_report`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Classification Report**, a novel
visualization that displays the key performance metrics—Precision,
Recall, and F1-Score—for each class in a multiclass
classification problem. It provides a more detailed and
interpretable summary than a confusion matrix alone, making it
easy to diagnose a model's per-class performance at a glance.


**Mathematical Concept:**
This plot visualizes the three most common metrics for evaluating
a multiclass classifier on a per-class basis
:footcite:p:`Powers2011`.

1.  **Precision**: The ability of the classifier not to label as
    positive a sample that is negative. It answers: *"Of all the
    predictions for this class, how many were correct?"*

    .. math::
       :label: eq:precision

       \text{Precision} = \frac{TP}{TP + FP}

2.  **Recall (Sensitivity)**: The ability of the classifier to
    find all the positive samples. It answers: *"Of all the
    actual samples of this class, how many did the model find?"*

    .. math::
       :label: eq:recall

       \text{Recall} = \frac{TP}{TP + FN}

3.  **F1-Score**: The harmonic mean of Precision and Recall,
    providing a single score that balances both metrics.

    .. math::
       :label: eq:f1_score

       \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

Each class is assigned an angular sector, and within that sector,
three bars are drawn, with their heights (radii) corresponding
to the scores for these metrics.

**Interpretation:**
The plot provides a granular, per-class breakdown of a
classifier's performance, making it easy to spot imbalances and
trade-offs.

* **Angle**: Each major angular sector represents a **True
    Class** (e.g., "Class Alpha").
* **Bars**: Within each sector, the three colored bars represent
    the key metrics: **Precision**, **Recall**, and **F1-Score**.
* **Radius**: The length of each bar represents the score for
    that metric, from 0 at the center to 1 at the edge. A good
    model will have consistently tall bars across all metrics and
    classes.

**Use Cases:**

* To get a detailed, per-class summary of a multiclass
    classifier's performance beyond a single accuracy score.
* To diagnose the Precision vs. Recall trade-off for each class.
* To identify which specific classes a model is struggling to
    predict correctly, especially in imbalanced datasets.


**Example**
See the gallery example and code:
:ref:`gallery_plot_polar_classification_report`.

.. _ug_plot_pinball_loss:

Pinball Loss Plot (:func:`~kdiagram.plot.evaluation.plot_pinball_loss`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Pinball Loss Plot** to provide a
granular, per-quantile assessment of a probabilistic forecast's
accuracy :footcite:p:`Gneiting2007b`. While the CRPS gives a single 
score for the overall performance, this plot breaks that score down 
and shows the model's performance at *each individual quantile level*.

**Mathematical Concept**
The Pinball Loss, :math:`\mathcal{L}_{\tau}`, is a proper scoring
rule for evaluating a single quantile forecast :math:`q` at level
:math:`\tau` against an observation :math:`y`. It asymmetrically
penalizes errors, giving a weight of :math:`\tau` to
under-predictions and :math:`(1 - \tau)` to over-predictions.

.. math::
   :label: eq:pinball_loss_plot

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

This plot calculates the average Pinball Loss for each provided
quantile and visualizes these scores on a polar axis, where the
angle represents the quantile level and the radius represents the
loss.

**Interpretation:**
The plot provides a detailed breakdown of a probabilistic
forecast's performance across its entire distribution.

* **Angle**: Represents the **Quantile Level**, sweeping from 0
    to 1 around the circle.
* **Radius**: The radial distance from the center represents the
    **Average Pinball Loss** for that quantile. A **smaller radius
    is better**, indicating a more accurate forecast for that
    specific quantile.
* **Shape**: A good forecast will have a small and relatively
    symmetrical shape close to the center. An asymmetrical shape
    can reveal if the model is better at predicting the lower
    tail of the distribution than the upper tail, or vice-versa.


**Use Cases:**

* To get a granular, per-quantile view of a model's performance,
    which is more detailed than an overall score like the CRPS.
* To diagnose if a model is better at predicting the center of a
    distribution (e.g., the median, q=0.5) versus its tails
    (e.g., q=0.1 or q=0.9).
* To compare the per-quantile performance of multiple models by
    overlaying their plots.


**Example**
See the gallery example and code: :ref:`gallery_plot_pinball_loss`.

.. _ug_plot_regression_performance:

Polar Performance Chart (:func:`~kdiagram.plot.evaluation.plot_regression_performance`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:**
This function creates a **Polar Performance Chart**, a grouped polar
bar chart designed to visually compare the performance of multiple
regression models across several evaluation metrics simultaneously.
It provides a holistic snapshot of model strengths and weaknesses,
making it easier to select the best model based on criteria beyond
a single score :footcite:p:`scikit-learn`.


**Mathematical Concept**
The plot visualizes a set of performance scores, which are
processed in three main steps:

1.  **Score Calculation**: For each model :math:`k` and each metric
    :math:`m`, a score :math:`S_{m,k}` is calculated. The function
    is designed to assume that a **higher score is always better**.
    To achieve this:
    
    - Standard scikit-learn error metrics are automatically
      negated (e.g., it uses ``neg_mean_absolute_error``).
    - The ``higher_is_better`` parameter allows the user to
      explicitly tell the function whether a lower value is better
      for any given metric (e.g., ``{'my_custom_error': False}``).
      The function will then negate the scores for that metric.

2.  **Normalization**: To make scores with different scales
    comparable, the scores for each metric are independently
    scaled to the range [0, 1] using Min-Max normalization. For a
    given metric :math:`m`, the normalized score for model :math:`k`
    is:

    .. math::
       :label: eq:perf_norm

       S'_{m,k} = \frac{S_{m,k} - \min(\mathbf{S}_m)}{\max(\mathbf{S}_m) - \min(\mathbf{S}_m)}

    where :math:`\mathbf{S}_m` is the vector of scores for all
    models on metric :math:`m`. A score of 1 represents the
    best-performing model for that metric, and a score of 0
    represents the worst.

3.  **Polar Mapping**:

    - Each metric is assigned its own angular sector, :math:`\theta_m`.
    - The normalized score, :math:`S'_{m,k}`, is mapped to the
      **radius** (height) of the bar for model :math:`k` within
      that sector.


**Interpretation:**
The plot provides a holistic, multi-metric view of model
performance, making it easy to identify trade-offs.

* **Angle**: Each angular sector represents a different
  **evaluation metric** (e.g., R², MAE, RMSE).
* **Bars**: Within each sector, the different colored bars represent
  the different models being compared.
* **Radius**: The length of each bar represents the model's
  **normalized score** for that metric. The green circle at the
  edge is the "Best Performance" line (a score of 1), and the
  red dashed circle is the "Worst Performance" line (a score of 0).
* **Shape**: The overall shape of a model's bars reveals its
  performance profile. A model with consistently long bars is a
  strong all-around performer.


**Use Cases:**

* To get a quick, visual summary of how multiple models perform
  across a range of different metrics.
* To identify the strengths and weaknesses of each model (e.g., "Is
  this model biased or just noisy?").
* For model selection when you need to balance trade-offs between
  different performance criteria.

---
**Example**
See the gallery example and code:
:ref:`gallery_plot_regression_performance`.