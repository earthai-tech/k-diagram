.. _gallery_evaluation:

============================
Model Evaluation Gallery
============================

This gallery page showcases plots from the `k-diagram` package
designed for the evaluation of classification models. It features
novel polar adaptations of standard, powerful diagnostic tools like
the ROC curve and the Precision-Recall curve.

These visualizations provide an intuitive and aesthetically engaging
way to compare the performance of multiple models, assess their
discriminative power, and understand their behavior, especially on
imbalanced datasets.

.. note::
   **Polar vs. Cartesian rendering (``kind``)**
   
   Many evaluation plots accept ``kind={'polar','cartesian'}``
   (default is ``'polar'`` unless stated otherwise). When
   ``kind='cartesian'``, the function **delegates** to a Cartesian
   renderer while preserving common styling (``figsize``, ``colors``,
   ``show_grid``). Polar-only options (e.g., ``acov``, ``zero_at``,
   ``clockwise``) are ignored in Cartesian mode. The return value is
   always the actual ``Axes`` used.

   *Use Cartesian* when you want the conventional reading for ROC/PR and
   classification plots (FPR/TPR on x/y, Precision/Recall on y/x,
   grouped bars). *Use Polar* when you want compact overviews, periodic
   angles, or comparative radial layouts. For ROC/PR in polar, a
   quarter-circle is used for readability.

   **Examples**

   .. code-block:: python

      # Delegates to Cartesian
      ax = kd.plot_polar_pr_curve(y_true, y_pred1, y_pred2,
                                  names=['A', 'B'],
                                  kind='cartesian')

      # Polar with angular coverage controls
      ax = kd.plot_polar_confusion_matrix(y_true, y_pred,
                                          kind='polar', acov='default')
   Invalid values are rejected with:
   ``ValueError("kind must be 'polar' or 'cartesian'.")``
                                          
   Furthermore, you need to run the code snippets locally to generate the plot
   images referenced below. Ensure the image paths in the
   ``.. image::`` directives match where you save the plots.


.. _gallery_plot_polar_roc:

------------------------------------------------------
Polar Receiver Operating Characteristic (ROC) Curve
------------------------------------------------------

The :func:`~kdiagram.plot.evaluation.plot_polar_roc` function visualizes
the performance of binary classifiers. It adapts the standard Receiver
Operating Characteristic (ROC) analysis to a polar coordinate system,
plotting the True Positive Rate against the False Positive Rate in an
intuitive quarter-circle format to assess a model's ability to
distinguish between classes.

To appreciate how this visualization conveys such rich information,
we must first understand its anatomy.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the **False Positive Rate (FPR)**, which
     is the fraction of negative instances incorrectly classified as
     positive. The angle sweeps from 0 to 1 as it moves from 0° to
     90°. **A smaller angle is better**.
   * **Radius (r):** Represents the **True Positive Rate (TPR)**, also
     known as **Recall** or **Sensitivity**. The radius extends from 0 at
     the center to 1 at the edge. **A larger radius is better**.
   * **No-Skill Line:** The dashed diagonal spiral represents a random
     classifier with an Area Under the Curve (AUC) of 0.5. A skillful
     model's curve should bow outwards, far away from this baseline.

With this framework in mind, let's apply the plot to a practical
scenario: evaluating a new medical screening test.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Comparing Medical Screening Algorithms**

A biotech company has developed a new "Experimental Algorithm" to screen
for a moderately common disease. They need to compare its performance
against their existing "Standard Algorithm." Since the dataset of
patients is relatively balanced, the ROC curve is an appropriate tool to
visualize which algorithm provides a better trade-off between correctly
identifying sick patients (TPR) and incorrectly flagging healthy
patients (FPR).

The following code simulates this clinical evaluation by generating
prediction scores from two overlapping distributions for each model—a
hallmark of realistic classification problems.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt

   # --- 1. Simulate Balanced Medical Data ---
   X, y_true = make_classification(
       n_samples=2000,
       n_classes=2,
       weights=[0.5, 0.5], # Balanced dataset
       flip_y=0,
       n_informative=5,
       random_state=42
   )

   # --- 2. Simulate Realistic Prediction Scores ---
   def generate_scores(y_true, pos_mean, scale):
       """Generate scores from two overlapping normal distributions."""
       scores = np.zeros_like(y_true, dtype=float)
       pos_mask = (y_true == 1)
       neg_mask = (y_true == 0)
       scores[pos_mask] = np.random.normal(
           loc=pos_mean, scale=scale, size=pos_mask.sum())
       scores[neg_mask] = np.random.normal(
           loc=0.5, scale=scale, size=neg_mask.sum())
       return np.clip(scores, 0, 1)

   # Experimental model has better separation between classes
   y_pred_experimental = generate_scores(y_true, pos_mean=0.65, scale=0.15)
   # Standard model has more overlap
   y_pred_standard = generate_scores(y_true, pos_mean=0.58, scale=0.18)


   # --- 3. Plotting ---
   kd.plot_polar_roc(
       y_true,
       y_pred_experimental,
       y_pred_standard,
       names=["Experimental Algorithm", "Standard Algorithm"],
       title="Medical Screening Algorithm Comparison (Polar ROC)",
       savefig="gallery/images/gallery_evaluation_plot_polar_roc.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_plot_polar_roc.png
   :align: center
   :width: 75%
   :alt: Polar ROC Curve comparing two medical screening algorithms.

   The "Experimental Algorithm" (blue) has a curve that bows out much
   further than the "Standard Algorithm" (orange), indicating a higher
   AUC and superior performance.

The generated plot provides an immediate visual verdict
on the performance of the two algorithms.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The generated plot provides an immediate visual verdict on the
   performance of the two algorithms.
   The **"Experimental Algorithm"** (dark blue curve) is demonstrably
   superior across all performance thresholds. Its curve consistently
   bows further outwards than the standard model, achieving a much
   larger Area Under the Curve (AUC) of **0.77**. This means that for
   any given False Positive Rate (angle), it correctly identifies a
   significantly higher proportion of true cases (larger radius).
   The **"Standard Algorithm"** (light blue curve), while better than
   random chance, is clearly outperformed. Its lower AUC of **0.61**
   reflects a poorer ability to distinguish between sick and healthy
   patients.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

While the first use case identified a clear overall winner, real-world
decisions are often more nuanced. Let's now consider a scenario where the
'best' model depends entirely on a specific operational constraint.

**Use Case 2: Optimizing for a Specific Clinical Need**

In a different clinical setting, the follow-up test for a disease is
extremely expensive and invasive. A hospital's primary goal is to
**strictly limit false positives to below 10% (FPR < 0.1)**. They are
evaluating two high-performing models: one with the best overall AUC,
and another specifically designed to be highly confident at low FPRs.

.. code-block:: python
   :linenos:

   # --- 1. Use the same balanced data ---
   # (Assuming y_true from the previous example is available)

   # --- 2. Simulate Predictions for two specialized models ---
   # Model A (High AUC): Generally excellent across all thresholds
   y_pred_high_auc = generate_scores(y_true, pos_mean=0.7, scale=0.2)
   # Model B (Low-FPR Specialist): Negative scores are tightly clustered
   # at a low value, making false positives at high thresholds rare.
   scores_b = np.zeros_like(y_true, dtype=float)
   pos_mask_b = (y_true == 1)
   neg_mask_b = (y_true == 0)
   scores_b[pos_mask_b] = np.random.normal(0.65, 0.25, pos_mask_b.sum())
   scores_b[neg_mask_b] = np.random.normal(0.3, 0.1, neg_mask_b.sum())
   y_pred_low_fpr = np.clip(scores_b, 0, 1)


   # --- 3. Plotting ---
   kd.plot_polar_roc(
       y_true,
       y_pred_high_auc,
       y_pred_low_fpr,
       names=["Model A (High AUC)", "Model B (Low-FPR Specialist)"],
       title="Optimizing for Low False Positive Rate",
       cmap='viridis',
       savefig="gallery/images/gallery_evaluation_roc_specialist.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_roc_specialist.png
   :align: center
   :width: 75%
   :alt: A Polar ROC Curve showing a trade-off at low FPRs.

   While Model A has a better overall AUC, Model B's curve starts
   higher (larger radius), showing it performs better under a strict
   low-FPR constraint.

.. topic:: 💡 Interpretation
   :class: hint

   This plot reveals that one model is exceptionally well-suited to the
   hospital's specific, low-risk requirement. The **"Model B (Low-FPR
   Specialist)"** (yellow curve) is the unambiguous winner. Its curve
   rises almost vertically at the start, indicating it achieves a very
   high True Positive Rate (large radius) while maintaining a near-zero
   False Positive Rate (very small angle).
   Its outstanding AUC of **0.90** confirms it is a far more powerful
   discriminative model than **Model A**, which has a much lower AUC of
   **0.76**. For the hospital's goal of keeping the FPR below 10%,
   Model B is not just the better choice, it is an outstanding one,
   offering high accuracy that perfectly aligns with their operational
   constraints.
   
.. admonition:: See Also
   :class: seealso

   While the ROC curve is a standard tool for balanced datasets, for
   problems with significant **class imbalance** (e.g., fraud detection),
   the :ref:`gallery_plot_polar_pr_curve` is often a more informative
   visualization of model performance.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the mathematical concepts behind ROC analysis,
please refer to the main :ref:`ug_plot_polar_pr_curve` section.


.. _gallery_plot_polar_pr_curve:

----------------------------------
Polar Precision-Recall Curve
----------------------------------

The :func:`~kdiagram.plot.evaluation.plot_polar_pr_curve` function
visualizes the trade-off between **Precision** and **Recall**. By mapping
these metrics to a polar coordinate system, it provides a clear view
of classifier performance. This is especially critical when dealing
with imbalanced datasets where other metrics, like the ROC curve, can be
misleadingly optimistic.

To understand how this plot visualizes this crucial trade-off, let's
first examine its components.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents **Recall** (True Positive Rate). This
     sweeps from 0 at the right (0°) to 1 at the top (90°). A wider
     angular sweep indicates the model's ability to find more of the
     true positive cases.
   * **Radius (r):** Represents **Precision**. The distance from the
     center (0) to the edge (1). A larger radius means that when the
     model predicts a positive, it is more likely to be correct.
   * **No-Skill Line:** The dashed circle represents the performance of
     a random classifier, where precision is equal to the prevalence of
     the positive class. A skillful model's curve should extend far
     beyond this baseline.

With this framework in mind, let's apply the Polar PR Curve to a classic
real-world problem where it excels: fraud detection.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Detecting Fraudulent Transactions**

A financial institution is developing a machine learning model to detect
fraudulent credit card transactions. This is a classic imbalanced data
problem: the vast majority of transactions are legitimate, and only a
tiny fraction are fraudulent (the positive class). The bank needs to
compare a new, sophisticated model against a simpler baseline to see if it's
better at catching fraud without overwhelming investigators with false alarms.

The following code simulates this scenario by generating prediction
scores from two overlapping distributions for each model—a hallmark of
realistic classification problems.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt

   # --- 1. Simulate Imbalanced Fraud Data ---
   X, y_true = make_classification(
       n_samples=2000, n_classes=2, weights=[0.95, 0.05], # 5% positive class
       flip_y=0.01, n_informative=5, random_state=42
   )

   # --- 2. Simulate Realistic Prediction Scores ---
   def generate_scores(y_true, pos_mean, scale):
       """Generate scores from two overlapping normal distributions."""
       scores = np.zeros_like(y_true, dtype=float)
       pos_mask = (y_true == 1)
       neg_mask = (y_true == 0)
       scores[pos_mask] = np.random.normal(
           loc=pos_mean, scale=scale, size=pos_mask.sum())
       scores[neg_mask] = np.random.normal(
           loc=0.4, scale=scale, size=neg_mask.sum())
       return np.clip(scores, 0, 1)

   # A good model with better separation between classes
   y_pred_good = generate_scores(y_true, pos_mean=0.75, scale=0.15)
   # A weak model with significant overlap
   y_pred_bad = generate_scores(y_true, pos_mean=0.55, scale=0.2)


   # --- 3. Plotting ---
   kd.plot_polar_pr_curve(
       y_true,
       y_pred_good,
       y_pred_bad,
       names=["Good Model", "Weak Model"],
       title="Fraud Detection Model Comparison (Polar PR Curve)",
       savefig="gallery/images/gallery_evaluation_plot_polar_pr_curve.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_plot_polar_pr_curve.png
   :align: center
   :width: 75%
   :alt: Example of a Polar Precision-Recall Curve for fraud detection.

   The "Good Model" (blue) shows a curve that bows out towards high
   precision and recall, while the "Weak Model" (orange) hugs the
   no-skill baseline.

The generated plot provides an immediate visual verdict on the models'
capabilities.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The plot clearly differentiates the two models' capabilities. The
   **"Good Model"** (purple curve) maintains a respectable radius (precision)
   as it sweeps to a wider angle, showing it can identify a large
   fraction of fraudulent cases without raising an excessive number of
   false alarms. Its Average Precision (AP) score of **0.68** is
   substantially better than the no-skill baseline (0.05).

   In stark contrast, the **"Weak Model"** (yellow curve) barely rises
   above the no-skill line, achieving a very low AP of **0.15**. This
   indicates that its performance is only marginally better than random
   guessing on this imbalanced dataset.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

While the first example showed a clear winner, real-world decisions
often involve choosing between two competent models with different
strengths. This is where the PR curve's ability to visualize strategic
trade-offs becomes invaluable.

**Use Case 2: Choosing Between Strategic Trade-offs**

Let's return to our fraud detection example. The bank has developed two
final candidate models:

* **Model A (The "Sniper"):** A high-precision model designed to
  minimize false alarms. Every alert it generates is highly likely to
  be true fraud, but it might miss some subtle cases.
* **Model B (The "Dragnet"):** A high-recall model designed to catch as
  much fraud as possible, even if it generates more false alarms.

The Polar PR curve is the perfect tool to help the bank make an informed
business decision by visualizing this strategic trade-off.

.. code-block:: python
   :linenos:

   # --- 1. Use the same imbalanced data ---
   # (Assuming y_true is available from the previous example)

   # --- 2. Simulate predictions for two specialized models ---
   # Model A (High-Precision): Tight negative distribution
   scores_a = np.zeros_like(y_true, dtype=float)
   pos_mask_a = (y_true == 1)
   neg_mask_a = (y_true == 0)
   scores_a[pos_mask_a] = np.random.normal(0.7, 0.2, pos_mask_a.sum())
   scores_a[neg_mask_a] = np.random.normal(0.3, 0.1, neg_mask_a.sum())
   y_pred_precision = np.clip(scores_a, 0, 1)

   # Model B (High-Recall): Positive distribution shifted higher
   scores_b = np.zeros_like(y_true, dtype=float)
   pos_mask_b = (y_true == 1)
   neg_mask_b = (y_true == 0)
   scores_b[pos_mask_b] = np.random.normal(0.8, 0.2, pos_mask_b.sum())
   scores_b[neg_mask_b] = np.random.normal(0.4, 0.2, neg_mask_b.sum())
   y_pred_recall = np.clip(scores_b, 0, 1)


   # --- 3. Plotting ---
   kd.plot_polar_pr_curve(
       y_true,
       y_pred_precision,
       y_pred_recall,
       names=["Model A (High-Precision)", "Model B (High-Recall)"],
       title="PR Curve: Visualizing Strategic Model Trade-offs",
       savefig="gallery/images/gallery_evaluation_pr_curve_tradeoff.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_pr_curve_tradeoff.png
   :align: center
   :width: 75%
   :alt: A Polar PR Curve showing a trade-off between two models.

   The plot shows Model A achieving high precision at low recall, while
   Model B achieves high recall at the cost of lower precision.

.. topic:: 💡 Interpretation
   :class: hint

   The plot instantly visualizes the different philosophies of the two
   models. In this case, while both models offer a strategic choice, one
   is a markedly stronger performer overall.

   * **Model A (Purple):** This is an outstanding "sniper" model. Its
     curve starts with a very **large radius** (near-perfect precision)
     and maintains it for the first 40-50% of recall. Its exceptionally
     high AP score of **0.90** makes it the superior choice for almost any
     business case where accuracy is valued.

   * **Model B (Yellow):** This "dragnet" model sweeps out to a **wider
     final angle**, achieving a slightly higher maximum recall. However,
     this comes at a great cost to precision, as shown by its much
     **smaller radius** across the board and significantly lower AP of
     **0.59**.

   The PR curve clarifies the choice. While Model B could be considered
   in a rare scenario where finding every last positive case is the only
   priority, Model A's high-precision and high-performance profile make
   it the clear winner for any strategy that balances accuracy and
   completeness.

.. admonition:: Best Practice
   :class: hint

   For classification tasks with a significant class imbalance, the
   **Precision-Recall Curve should be your primary evaluation tool** over
   the ROC Curve. The ROC curve's inclusion of True Negatives can paint
   a deceptively optimistic picture when the negative class is vast.


.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">
   
For a deeper dive into the mathematical concepts behind Precision and
Recall, please refer to the main :ref:`ug_plot_polar_pr_curve`.


.. _gallery_plot_polar_confusion_matrix:

-----------------------------
Polar Confusion Matrix
-----------------------------

The :func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix` function
provides a visually engaging alternative to the traditional grid-based
confusion matrix. It visualizes the four core components of binary
classification performance (TP, FP, TN, FN) as bars on a polar plot,
making it an excellent tool for at-a-glance model comparison.

To see how this plot transforms a simple table of numbers into an
intuitive graphic, let's first deconstruct its components.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each of the four angular sectors is dedicated to one
     component of the confusion matrix: **True Positive (TP)**, **False
     Positive (FP)**, **True Negative (TN)**, and **False Negative (FN)**.
   * **Radius (r):** The length of a bar represents the number of
     samples in that category. This can be displayed as a **proportion**
     of the total (if ``normalize=True``) or as **raw counts**.
     Ideally, bars in the TP and TN sectors should be long, while bars
     in the FP and FN sectors should be short.
   * **Model Comparison:** Different models are represented by different
     colored bars within each of the four sectors, allowing for direct,
     side-by-side comparison of performance and error types.

With this framework in mind, we can now apply the plot to a practical
scenario.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Comparing Spam Detection Models**

A cybersecurity team is evaluating two new spam detection algorithms.
They need to understand not just their overall accuracy, but the
specific *types* of errors each one makes. A "False Positive" (flagging
a legitimate email as spam) is highly undesirable as it can disrupt
communication, while a "False Negative" (letting a spam email through)
is a nuisance.

The Polar Confusion Matrix allows the team to visually compare these
error trade-offs. The following code simulates the evaluation of a
"Balanced Model" against a more "Aggressive" filter.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt

   # --- 1. Simulate Email Classification Data ---
   X, y_true = make_classification(
       n_samples=2000, n_classes=2, weights=[0.7, 0.3], flip_y=0.1,
       random_state=42
   ) # 1 = Spam, 0 = Not Spam

   # --- 2. Simulate Realistic Prediction Scores ---
   def generate_scores(y_true, pos_mean, scale):
       """Generate scores from two overlapping normal distributions."""
       scores = np.zeros_like(y_true, dtype=float)
       pos_mask = (y_true == 1); neg_mask = (y_true == 0)
       scores[pos_mask] = np.random.normal(loc=pos_mean, scale=scale, size=pos_mask.sum())
       scores[neg_mask] = np.random.normal(loc=0.4, scale=scale, size=neg_mask.sum())
       return np.clip(scores, 0, 1)

   # A balanced model with decent performance
   y_pred_balanced = generate_scores(y_true, pos_mean=0.65, scale=0.15)
   # An aggressive model biased towards flagging spam (higher scores overall)
   y_pred_aggressive = generate_scores(y_true, pos_mean=0.75, scale=0.2)

   # --- 3. Plotting ---
   kd.plot_polar_confusion_matrix(
       y_true,
       y_pred_balanced,
       y_pred_aggressive,
       names=["Balanced Model", "Aggressive Filter"],
       normalize=True, # Show results as proportions
       title="Spam Detection Model Comparison",
       savefig="gallery/images/gallery_evaluation_plot_polar_confusion_matrix.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_plot_polar_confusion_matrix.png
   :align: center
   :width: 75%
   :alt: Polar Confusion Matrix comparing two spam detection models.

   The plot shows the "Aggressive Filter" (orange) has a higher True
   Positive rate but also a higher False Positive rate than the
   "Balanced Model" (blue).

The generated plot provides an immediate visual summary of each model's
behavior.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The plot reveals the distinct trade-offs made by each model.
   The **"Balanced Model"** (purple bars) excels at correctly
   identifying legitimate emails, as shown by its very long bar in the
   **True Negative** sector. It maintains a good balance between catching
   spam (True Positive) and its errors.

   The **"Aggressive Filter"** (yellow bars) tells a different story. It
   catches slightly more spam (a longer **True Positive** bar), but this
   comes at a significant cost: its **False Positive** bar is much
   longer, indicating it incorrectly flags far more legitimate emails
   as spam. This visual evidence allows the team to make an informed
   decision, likely favoring the Balanced Model unless the goal is to
   block spam at all costs.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

While normalized proportions are excellent for comparing relative
performance, some applications require the exact counts. The plot can
be customized for this and other presentation needs, as our next use
case shows.

**Use Case 2: Customizing for a Clinical Trial Report**

A medical research team is evaluating a new diagnostic test. For their
clinical report, they must present the exact **number** of patients
correctly and incorrectly classified. Furthermore, they want to tailor
the visualization to emphasize the most critical outcomes for patient
care: **False Negatives** (missed diagnoses) and **True Positives**
(correct diagnoses).

By setting ``normalize=False`` and reordering the sectors with the
``categories`` parameter, they can create a more impactful report figure.

.. code-block:: python
   :linenos:

   # --- 1. Use the same data as the previous example ---
   # (Assuming y_true, y_pred_balanced, y_pred_aggressive are available)

   # --- 2. Plotting with Customizations ---
   kd.plot_polar_confusion_matrix(
       y_true,
       y_pred_balanced,
       y_pred_aggressive,
       names=["Balanced Model", "Aggressive Filter"],
       normalize=False, # Show raw counts instead of proportions
       title="Diagnostic Test Results (Patient Counts)",
       # Reorder categories to group by predicted outcome
       categories=["TP", "FP",
                   "TN", "FN"],
       # Use custom colors for the report
       colors=['#003f5c', '#ffa600'],
       savefig="gallery/images/gallery_evaluation_cm_custom.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_cm_custom.png
   :align: center
   :width: 75%
   :alt: A customized Polar Confusion Matrix showing raw counts.

   The plot now displays absolute patient counts and has been reordered
   to place the most critical metrics (TP and FN) side-by-side.

.. topic:: 💡 Interpretation
   :class: hint

   This customized plot answers the researchers' specific questions
   more directly. The y-axis now clearly shows the **absolute number of
   patients** in each category, providing concrete numbers for the report.

   By reordering the sectors, the two most critical outcomes for patient
   health are now adjacent in the top half of the plot. The audience can
   immediately compare the number of correctly identified cases
   (**TP**) against the number of dangerously missed diagnoses (**FN**)
   for both the **Balanced Model** (teal) and the **Aggressive Filter**
   (orange). This customization transforms the plot from a general
   evaluation tool into a focused narrative device, tailored to the
   high-stakes concerns of a medical audience.
   
.. admonition:: See Also
   :class: seealso

   This plot is designed for binary classification. For tasks with
   three or more classes, a different visualization is required. See
   the :func:`~kdiagram.plot.evaluation.plot_polar_confusion_multiclass`
   function for an alternative designed for multiclass problems.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the concepts of confusion matrices, please refer
to the main :ref:`ug_plot_polar_confusion_matrix`.

.. _gallery_plot_polar_confusion_matrix_in:

-----------------------------------
Multiclass Polar Confusion Matrix
-----------------------------------

The :func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix_in`
function, also available as :func:`~kdiagram.plot.evaluation.plot_polar_confusion_multiclass`,
deconstructs a multiclass confusion matrix into an intuitive
visual format. By dedicating an angular sector to each "true" class,
it uses grouped bars to show how those samples were predicted, making it
easy to spot which classes are well-predicted and which are commonly
confused.

To see how this plot transforms a complex grid of numbers into an
interpretable graphic, let's first examine its components.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each major angular sector is dedicated to a single
     **True Class** (e.g., the actual category of a sample).
   * **Bars Within a Sector:** The different colored bars *within* a
     True Class's sector show the distribution of the model's
     **Predicted Classes**. In a perfect model, each sector would contain
     only a single, long bar corresponding to the correct prediction.
   * **Radius (r):** The length of each bar represents the number of
     samples. This can be displayed as a **proportion** of the total
     (if ``normalize=True``) or as **raw counts**.

With this framework in mind, we can now apply the plot to a practical
scenario in image classification.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Diagnosing an Image Classifier**

An AI team has trained a Convolutional Neural Network (CNN) to classify
animal images into four categories: 'Cat', 'Dog', 'Fox', and 'Wolf'. A
simple accuracy score isn't enough; they need to diagnose the model's
behavior. Which animals does it struggle with? Does it have a specific
bias, like confusing visually similar animals such as dogs and wolves?

The following code simulates the model's predictions, introducing a
plausible confusion between canid species, and then visualizes the
results.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt

   # --- 1. Simulate Image Classification Data ---
   class_labels = ["Cat", "Dog", "Fox", "Wolf"]
   # Create y_true with integer labels 0, 1, 2, 3
   X, y_true = make_classification(
       n_samples=2000, n_classes=4, weights=[0.25, 0.35, 0.15, 0.25],
       flip_y=0.05, n_informative=8, n_clusters_per_class=1, random_state=42
   )

   # --- 2. Simulate Realistic Predictions with Confusion ---
   y_pred = y_true.copy()
   # Confuse 30% of Dogs (1) as Wolves (3)
   dog_mask = (y_true == 1) & (np.random.rand(2000) < 0.30)
   y_pred[dog_mask] = 3
   # Confuse 20% of Foxes (2) as Cats (0)
   fox_mask = (y_true == 2) & (np.random.rand(2000) < 0.20)
   y_pred[fox_mask] = 0

   # --- 3. Plotting ---
   kd.plot_polar_confusion_matrix_in(
       y_true,
       y_pred,
       class_labels=class_labels,
       normalize=True, # Show results as proportions
       title="Animal Image Classifier Performance",
       savefig="gallery/images/gallery_evaluation_multiclass_cm.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_multiclass_cm.png
   :align: center
   :width: 75%
   :alt: Polar Confusion Matrix for an animal image classifier.

   The plot reveals that the model performs well on 'Cats' and 'Wolves'
   but frequently confuses 'Dogs' with 'Wolves'.

The generated plot provides an immediate diagnostic report on the model's
behavior.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The plot instantly reveals the classifier's strengths and, more
   importantly, a critical weakness. The model shows **excellent
   performance** in the **"True Cat"** sector, where the "Predicted
   Cat" bar (blue) reaches a proportion of 1.0, indicating near-perfect
   classification for that class. Performance on **"True Fox"** (pink)
   and **"True Wolf"** (cyan) is also reasonably good.

   However, the **"True Dog"** sector highlights a catastrophic
   failure. The "Predicted Dog" bar (red) is almost non-existent, and no
   other bar is significantly large. This demonstrates that the model
   is not just confusing dogs with other animals; it is **systematically
   failing to identify dogs at all**. This insight is crucial for the AI
   team, as it points to a severe issue with the 'Dog' class in the
   training data or model architecture that must be addressed.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

While normalized proportions are great for understanding error *rates*,
some applications depend on the absolute *number* of errors. The plot can
be customized for this and other presentation needs.

**Use Case 2: Customizing for an Inventory Management Report**

A retail company uses an automated system to classify products into
categories. Misclassifying a few expensive 'Electronics' items as
'Groceries' can be a costly error. The logistics team needs a report
showing the raw **count** of misclassified items. For their weekly
meeting, they want a plot that orients the most problematic category,
'Electronics', at the top for immediate focus.

.. code-block:: python
   :linenos:

   # --- 1. Simulate Inventory Data ---
   # (Using the same y_true and y_pred logic from Use Case 1,
   # but with different labels for the new context)
   inventory_labels = ["Electronics", "Apparel", "Home Goods", "Groceries"]

   # --- 2. Plotting with Customizations ---
   kd.plot_polar_confusion_matrix_in(
       y_true,
       y_pred,
       class_labels=inventory_labels,
       normalize=False, # Show raw item counts
       title="Inventory Misclassification (Weekly Report)",
       cmap='Set2', # Use a different color palette
       # Place the first class ('Electronics') at the North position
       zero_at='N',
       savefig="gallery/images/gallery_evaluation_multiclass_cm_custom.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_multiclass_cm_custom.png
   :align: center
   :width: 75%
   :alt: A customized Multiclass Polar Confusion Matrix showing raw counts.

   The plot now displays absolute item counts and has been oriented to
   place the "True Electronics" category at the top for emphasis.

.. topic:: 💡 Interpretation
   :class: hint

   This customized plot directly addresses the logistics team's needs.
   The radial axis now shows the **absolute number of items**,
   transforming abstract proportions into tangible business metrics. By
   setting ``zero_at='N'``, the **"True Electronics"** sector is
   placed at the top, focusing the weekly meeting on this key category,
   where the model performs exceptionally well.

   The plot also serves as an immediate high-priority alert. While
   performance on 'Home Goods' (yellow) and 'Groceries' (gray) is
   adequate, the **"True Apparel"** sector shows a near-total failure,
   with almost no items being correctly classified. Seeing this near-zero
   bar for "Predicted Apparel" instantly tells the team that an entire
   product category is being systematically mismanaged, providing a
   clear, data-driven directive to investigate and fix the classification
   error for 'Apparel'.

.. admonition:: See Also
   :class: seealso

   This plot is designed for multiclass classification. For tasks with
   only two classes, the binary version,
   :func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix`,
   provides a more specialized visualization.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the concepts of confusion matrices, please refer
to the :ref:`ug_plot_polar_confusion_matrix_in` section.

   
.. _gallery_plot_polar_classification_report:

-----------------------------
Polar Classification Report
-----------------------------

The :func:`~kdiagram.plot.evaluation.plot_polar_classification_report`
function provides a detailed, per-class performance breakdown for a
multiclass classifier. It moves beyond a single accuracy score to
visualize the key metrics of **Precision**, **Recall**, and **F1-Score** for
each class in an intuitive polar bar chart, making it easy to diagnose
a model's specific strengths and weaknesses.

To appreciate how this plot effectively summarizes a standard
classification report, let's first deconstruct its components.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each major angular sector is dedicated to a single
     **Class** from the dataset (e.g., "Class Alpha").
   * **Bars Within a Sector:** The three different colored bars *within* a
     class's sector represent the key performance metrics: **Precision**,
     **Recall**, and the **F1-Score**.
   * **Radius (r):** The length of each bar represents the score for
     that metric, on a scale from 0 (at the center) to 1 (at the edge).
     Taller bars indicate better performance for that specific metric
     and class.

With this framework in mind, let's apply the plot to a common challenge
in machine learning: evaluating a model trained on imbalanced data.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Diagnosing an Imbalanced Classifier**

A data science team is classifying customer support tickets into three
categories: 'Technical Issue', 'Billing Inquiry', and 'General
Feedback'. The dataset is naturally imbalanced—most tickets are
'Technical', while 'General Feedback' is rare. A high overall accuracy
score could be misleading if the model is simply ignoring the minority
class.

The Polar Classification Report is the perfect tool to diagnose this
per-class performance and uncover any hidden weaknesses.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt

   # --- 1. Simulate Imbalanced Support Ticket Data ---
   class_labels = ["Technical Issue", "Billing Inquiry", "General Feedback"]
   X, y_true = make_classification(
       n_samples=2000, n_classes=3, weights=[0.6, 0.3, 0.1], # Imbalanced
       flip_y=0.1, n_informative=10, n_clusters_per_class=1, random_state=42
   )

   # --- 2. Simulate predictions where model struggles with minority class ---
   y_pred = y_true.copy()
   # Confuse 50% of the rare 'General Feedback' class (2) as 'Technical' (0)
   feedback_mask = (y_true == 2) & (np.random.rand(2000) < 0.5)
   y_pred[feedback_mask] = 0

   # --- 3. Plotting ---
   kd.plot_polar_classification_report(
       y_true,
       y_pred,
       class_labels=class_labels,
       title="Support Ticket Classifier Performance (Initial Model)",
       savefig="gallery/images/gallery_evaluation_class_report.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_class_report.png
   :align: center
   :width: 75%
   :alt: Polar Classification Report for an imbalanced dataset.

   The plot shows high scores for the majority class ('Technical Issue')
   but very poor scores, especially Recall, for the minority class
   ('General Feedback').

The generated plot immediately highlights the problem.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The report visually confirms the team's suspicion about the model's
   performance on the imbalanced dataset. The model performs very well on
   the majority class, **"Technical Issue"**, with all three metric bars
   (Precision, Recall, F1-Score) being very tall, with scores above 0.8.
   Performance on **"Billing Inquiry"** is also strong, although it shows a
   trade-off: its Precision is high (blue bar ≈ 0.9), but its Recall is
   lower (brown bar ≈ 0.6), meaning it is accurate but misses some cases.

   However, the **"General Feedback"** sector reveals the model's critical
   flaw. The **Recall** bar is extremely short (≈ 0.2), while the
   **Precision** bar is moderately high. This indicates that while the
   model rarely misclassifies other tickets as 'General Feedback', it
   fails to find most of the actual 'General Feedback' tickets. The low
   **F1-Score** (cyan bar ≈ 0.3) confirms this poor overall performance,
   providing a clear directive to improve the model's handling of this
   minority class.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

This plot is not just a static diagnostic tool; it is also invaluable for
demonstrating the impact of model improvements, as our next use case shows.

**Use Case 2: Comparing Models Before and After Tuning**

After diagnosing the problem, the team retrains their model, this time
using class weights to force it to pay more attention to the minority
'General Feedback' class. To showcase their success to stakeholders, they
need a clear, side-by-side comparison of the model's performance before
and after this tuning.

By passing an `ax` object, we can create subplots to generate a powerful
comparative visualization.

.. code-block:: python
   :linenos:

   # --- 1. Use y_true and the initial y_pred from Use Case 1 ---
   y_pred_before = y_pred

   # --- 2. Simulate improved predictions after tuning ---
   y_pred_after = y_true.copy()
   # Now, only 20% of 'General Feedback' are confused
   feedback_mask_after = (y_true == 2) & (np.random.rand(2000) < 0.2)
   y_pred_after[feedback_mask_after] = 0

   # --- 3. Plotting side-by-side ---
   fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                            subplot_kw={'projection': 'polar'})

   # Plot Before
   kd.plot_polar_classification_report(
       y_true, y_pred_before, class_labels=class_labels,
       title="Before Tuning", ax=axes[0],
       # Use custom colors for metrics for visual consistency
       colors=['#1f77b4', '#ff7f0e', '#2ca02c']
   )
   # Plot After
   kd.plot_polar_classification_report(
       y_true, y_pred_after, class_labels=class_labels,
       title="After Tuning (with Class Weights)", ax=axes[1],
       colors=['#1f77b4', '#ff7f0e', '#2ca02c']
   )
   fig.savefig("gallery/images/gallery_evaluation_class_report_comparison.png")
   plt.close(fig)

.. figure:: ../images/evaluation/gallery_evaluation_class_report_comparison.png
   :align: center
   :width: 100%
   :alt: Side-by-side comparison of a model before and after tuning.

   The side-by-side plots clearly show the significant improvement in
   Recall and F1-Score for the 'General Feedback' class after tuning.

.. topic:: 💡 Interpretation
   :class: hint

   This side-by-side comparison provides a compelling narrative of model
   improvement. The **"Before Tuning"** plot on the left serves as the
   baseline, clearly showing the poor Recall (~0.2) on 'General Feedback'.

   The **"After Tuning"** plot on the right demonstrates the success of
   the team's intervention. The **Recall** bar (orange) in the 'General
   Feedback' sector is now significantly taller, jumping from roughly 0.2
   to 0.7. This directly boosts the **F1-Score** (green bar), confirming
   that the model is now much better at correctly identifying tickets from
   the rare category. This improvement comes with a slight, acceptable
   dip in precision for the majority class, a common trade-off when
   optimizing for fairness on imbalanced data. This comparative
   visualization is far more impactful than a table of numbers, making
   it an effective tool for communicating progress.

.. admonition:: Best Practice
   :class: hint

   Use this plot in conjunction with a
   :ref:`gallery_plot_polar_confusion_matrix_in`. The classification
   report shows you *how well* a model performs on a class, while the
   confusion matrix shows you *where the errors are going*—the specific
   patterns of misclassification.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the definitions of Precision, Recall, and
F1-Score, please refer to the main :ref:`ug_plot_polar_classification_report` 
section.


.. _gallery_application_classification_metrics:

--------------------------------------------------------
Application: A Holistic View of Classifier Performance
--------------------------------------------------------

Individual evaluation plots are excellent for diagnosing specific aspects
of a model's performance. However, their true power is unlocked when they
are used together as a visual dashboard to build a complete, holistic
understanding of a classifier's behavior.

This application demonstrates how to combine the polar confusion matrix,
classification report, and PR curve to solve a realistic business
problem, leading to a nuanced and data-driven decision.

**The Problem: Classifying E-Commerce Support Tickets**

.. admonition:: Practical Example

   An e-commerce company uses an AI model to automatically classify
   incoming customer support emails into three categories - **'Returns'**,
   **'Shipping Inquiry'**, and **'Product Feedback'**. The business has
   specific, and sometimes conflicting, operational needs:

   1.  **'Returns'** are time-sensitive and costly if misclassified. They
       must be identified with the **highest possible Recall**, even if it
       means some other tickets are incorrectly flagged as returns.
   2.  **'Shipping Inquiry'** tickets must be routed to the correct
       department. High **Precision** is critical to avoid sending customers
       down the wrong path and increasing resolution time.
   3.  **'Product Feedback'** is a lower priority and can tolerate more
       errors.

The dataset is highly imbalanced, with 'Returns' being the rarest
category. The team needs to evaluate two models—a baseline Logistic
Regression and a more complex Random Forest—to determine which one best
meets these complex business requirements.

**Translating the Problem into a Visual Dashboard**

To get a complete picture, we will generate a three-panel dashboard.
This will allow us to move from a high-level overview of errors to a
detailed, per-class metric analysis, and finally to a focused comparison
on the most critical business task.

The following code simulates the models' performance and creates this
diagnostic dashboard.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from sklearn.datasets import make_classification
   import matplotlib.pyplot as plt

   # --- 1. Simulate Imbalanced E-Commerce Support Data ---
   class_labels = ["Shipping Inquiry", "Product Feedback", "Returns"]
   # Class 2 ('Returns') is the rare, critical class
   X, y_true = make_classification(
       n_samples=3000, n_classes=3, weights=[0.45, 0.45, 0.1],
       flip_y=0.1, n_informative=12, n_clusters_per_class=1, random_state=42
   )

   # --- 2. Simulate Realistic Predictions from Two Models ---
   def generate_scores(y_true, class_means, class_scales):
       """Generate scores from class-specific normal distributions."""
       n_classes = len(class_means)
       scores = np.zeros((len(y_true), n_classes))
       for i in range(n_classes):
           mask = (y_true == i)
           scores[mask, :] = np.random.normal(
               loc=class_means[i], scale=class_scales[i], size=(mask.sum(), n_classes)
           )
       return np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

   # Logistic Regression: Modest performance
   lr_scores = generate_scores(y_true,
       class_means=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
       class_scales=[0.8, 0.8, 1.2]
   )
   y_pred_lr = np.argmax(lr_scores, axis=1)

   # Random Forest: Better overall, especially at identifying 'Returns'
   rf_scores = generate_scores(y_true,
       class_means=[[2, 0, 0.5], [0, 2, 0.5], [0.5, 0, 3]],
       class_scales=[0.5, 0.5, 0.8]
   )
   y_pred_rf = np.argmax(rf_scores, axis=1)

   # --- 3. Create the 2x2 Dashboard ---
   fig, axes = plt.subplots(2, 2, figsize=(18, 18),
                            subplot_kw={'projection': 'polar'})
   fig.suptitle("E-Commerce Classifier Evaluation Dashboard", fontsize=24, y=1.02)

   # Top-Left: Confusion Matrix for the best model (Random Forest)
   kd.plot_polar_confusion_matrix_in(
       y_true, y_pred_rf, class_labels=class_labels, ax=axes[0, 0],
       title="Random Forest: Confusion Patterns", normalize=False,
       colors=['#1a5f7a', '#57c5b6', '#ffc93c']
   )

   # Top-Right: Classification Report for the Random Forest
   kd.plot_polar_classification_report(
       y_true, y_pred_rf, class_labels=class_labels, ax=axes[0, 1],
       title="Random Forest: Per-Class Metrics",
       colors=['#003f5c', '#bc5090', '#ffa600']
   )

   # Bottom-Left: PR Curve for the critical 'Returns' class (Class 2)
   # We treat this as a one-vs-rest problem for the PR curve
   y_true_returns = (y_true == 2).astype(int)
   # Use the probability of the 'Returns' class for the PR curve
   lr_scores_returns = lr_scores[:, 2]
   rf_scores_returns = rf_scores[:, 2]
   kd.plot_polar_pr_curve(
       y_true_returns, rf_scores_returns, lr_scores_returns,
       names=["Random Forest", "Logistic Regression"], ax=axes[1, 0],
       title="PR Curve for 'Returns' Class", 
   )
   # Hide the unused subplot in the bottom-right
   fig.delaxes(axes[1, 1])
   fig.savefig("gallery/images/gallery_evaluation_dashboard_2x2.png")
   plt.close(fig)

.. figure:: ../images/evaluation/gallery_evaluation_dashboard_2x2.png
   :align: center
   :width: 100%
   :alt: A three-panel dashboard showing different polar evaluation plots.

   A comprehensive evaluation dashboard using three polar plots to
   provide a holistic view of classifier performance.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This dashboard provides a complete story, allowing the team to make
   a nuanced, evidence-based decision by examining the model's
   performance from three different perspectives.

   1.  **Panel 1 (Top-Left:Confusion Matrix):** This plot gives a high-level
       view of the Random Forest model's errors. We can see it
       performs exceptionally well on **'Shipping Inquiry'** (top
       sector), with a very long bar for correct predictions (over
       1200 tickets) and minimal confusion. The model is also effective
       at identifying **'Returns'** (left sector). Its most significant
       weakness is in classifying **'Product Feedback'**, where it
       correctly identifies most cases but also frequently
       misclassifies them as 'Shipping Inquiry'.

   2.  **Panel 2 (Top-Right:Classification Report):** This plot quantifies the
       business trade-offs. For **'Shipping Inquiry'**, both the
       **Precision** and **Recall** bars are very high (around 0.9),
       meeting the business need for accurate routing. For the
       critical **'Returns'** class, the **Recall** bar (pink) is the
       highest of its three metrics (around 0.85), confirming the model
       is very effective at finding these important tickets. The lower
       precision for 'Returns' is an acceptable trade-off, as per the
       initial requirements.

   3.  **Panel 3 (Bottom-Left: PR Curve):** This plot provides the final verdict on
       the most critical task. When comparing the models' ability to
       identify 'Returns', the **Random Forest** (purple curve) is
       unambiguously superior, achieving a near-perfect Average
       Precision (AP) score of **0.99**. The **Logistic Regression**
       (yellow curve) performs far worse, with an AP of only **0.50**,
       making it unsuitable for this key task.

   **Conclusion:** The dashboard provides a clear recommendation. The
   **Random Forest** model should be deployed. It dramatically
   outperforms the baseline on the most critical task (Panel 3) and
   meets the specific, nuanced precision and recall goals for the
   different ticket categories (Panel 2), all while having a clear and
   understandable error pattern (Panel 1).
   

.. _gallery_plot_pinball_loss:

-----------------------------
Polar Pinball Loss
-----------------------------

The :func:`~kdiagram.plot.evaluation.plot_pinball_loss` function
provides a granular view of a probabilistic forecast's performance by
visualizing the **Pinball Loss** for each predicted quantile. While a
single score like the CRPS gives an overall average error, this plot
diagnoses *where* in the distribution a model is accurate and where it
struggles.

To understand how this plot reveals a model's predictive characteristics,
let's first deconstruct its components.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the **Quantile Level**, sweeping from 0
     to 1 around the circle. For example, the 0.5 quantile (the median)
     is typically at the bottom of the plot.
   * **Radius (r):** The radial distance from the center represents the
     **Average Pinball Loss** for that specific quantile. Unlike other
     plots, here **a smaller radius is better**, indicating a more
     accurate forecast for that quantile level.
   * **Shape:** The overall shape of the resulting polygon is highly
     informative. A symmetrical "butterfly" shape often indicates a
     well-calibrated model that is more certain about the median than
     the tails, while a lopsided shape can reveal a systematic bias in
     the forecast.

With this framework in mind, let's apply the plot to a practical
forecasting problem.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Diagnosing a Temperature Forecast Model**

A meteorology team has developed a new model to predict the next day's
temperature range. Instead of a single value, it predicts a full
probability distribution, which is summarized by various quantiles (e.g.,
the 10th, 50th, and 90th percentiles). A key question is whether the model
is equally good at predicting the median temperature as it is at
predicting the extreme cold or hot temperatures in the tails of the
distribution.

The following code simulates a common scenario: a model that is very
accurate at predicting the median but less certain about the extremes.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from scipy.stats import norm
   import matplotlib.pyplot as plt

   # --- 1. Simulate True Values and Quantile Predictions ---
   np.random.seed(0)
   n_samples = 1000
   y_true = np.random.normal(loc=15, scale=5, size=n_samples) # Daily temps
   quantiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

   # Simulate a model that is better at the median, worse at the tails
   # This is done by varying the scale of the normal distribution
   scales = np.array([8, 6, 4, 3, 4, 6, 8])
   y_preds = norm.ppf(
       quantiles, loc=y_true[:, np.newaxis], scale=scales
   )

   # --- 2. Plotting ---
   kd.plot_pinball_loss(
       y_true,
       y_preds,
       quantiles,
       title="Temperature Forecast Performance by Quantile",
       savefig="gallery/images/gallery_evaluation_plot_pinball_loss.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_evaluation_plot_pinball_loss.png
   :align: center
   :width: 75%
   :alt: Example of a Polar Pinball Loss Plot for a weather forecast.

   The plot shows a "butterfly" shape, with the smallest loss (radius)
   at the 0.5 quantile and the largest losses at the extreme tails.

The generated plot provides an immediate diagnostic report on the model's
behavior.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The distinct "butterfly" shape of the plot instantly reveals the
   model's performance profile. The radius is smallest at the **0.5 
   quantile**, indicating that the pinball loss is lowest for the median 
   forecast. This means the model is highly skilled at predicting the 
   central tendency of the next day's temperature.

   Conversely, the radii are largest at the extreme tails shown (e.g.,
   the **0.12** and **0.88** quantiles). This shows that the model is
   much less accurate when predicting unusually cold or hot days. The
   slight asymmetry, with slightly higher losses on the lower quantiles,
   suggests the model finds it a bit harder to predict colder extremes
   than warmer ones. This is a critical insight, telling meteorologists
   that while their median forecast is reliable, the uncertainty range
   for extreme weather may be underestimated.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

While a single plot is great for diagnosis, the next step is often to
compare a new model against an existing one. For such comparisons,
focusing on the *shape* of the loss profile can be more insightful than
the exact loss values.

**Use Case 2: Comparing Model Performance Profiles**

The meteorology team now wants to compare their new, sophisticated
model against a simpler baseline model. For their report, they want a
side-by-side visualization that emphasizes the difference in the
*performance shape* of the two models. They decide to mask the radial
tick labels to focus the audience's attention on the contrasting shapes
of the loss curves.

.. code-block:: python
   :linenos:

   # --- 1. Use data from Use Case 1 for the sophisticated model ---
   # (Assuming y_true, quantiles, and y_preds are available)
   y_preds_sophisticated = y_preds

   # --- 2. Simulate a simpler baseline model ---
   # This model has a constant, larger uncertainty across all quantiles
   y_preds_baseline = norm.ppf(
       quantiles, loc=y_true[:, np.newaxis], scale=7
   )

   # --- 3. Plotting side-by-side ---
   fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                            subplot_kw={'projection': 'polar'})

   # Plot Sophisticated Model
   kd.plot_pinball_loss(
       y_true, y_preds_sophisticated, quantiles,
       title="Sophisticated Model Profile",
       ax=axes[0], 
       colors='r'
   )
   # Plot Baseline Model and mask the radius labels
   kd.plot_pinball_loss(
       y_true, y_preds_baseline, quantiles,
       title="Baseline Model Profile",
       mask_radius=True, # Focus on the shape
       ax=axes[1], 
       colors='r', 
   )
   fig.savefig("gallery/images/gallery_evaluation_pinball_comparison.png")
   plt.close(fig)

.. figure:: ../images/evaluation/gallery_evaluation_pinball_comparison.png
   :align: center
   :width: 100%
   :alt: Side-by-side comparison of two forecast models' loss profiles.

   The side-by-side plots contrast the specialized "butterfly" shape of
   the sophisticated model with the more uniform, circular loss profile
   of the baseline model.

.. topic:: 💡 Interpretation
   :class: hint

   This side-by-side comparison effectively highlights the behavioral
   differences between the two models.

   The **Sophisticated Model** (left) is a *specialist*. Its profile
   shows a very low loss (small radius) for the median forecast, proving
   it allocates its predictive power to delivering a highly accurate
   forecast for the most likely outcomes. This specialization, however,
   comes at the cost of much higher errors for the tails.

   The **Baseline Model** (right) is a *generalist*. Its loss profile is
   flatter, with a smaller performance gap between the median and the
   tails. However, its loss at the median is visibly much higher (worse)
   than that of the sophisticated model. This visual comparison makes it
   clear that the Sophisticated model is far superior for its primary
   goal of accurately predicting the median temperature.

.. admonition:: Best Practice
   :class: hint

   Pinball Loss is the only strictly **proper scoring rule** for
   evaluating quantile forecasts. Unlike Mean Squared Error, it correctly
   penalizes under-prediction and over-prediction asymmetrically, in
   proportion to the quantile level, making it the industry standard for
   this task.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the mathematical definition of the Pinball Loss
function, please refer to the main :ref:`ug_plot_pinball_loss`.
     

.. _gallery_plot_regression_performance:

-----------------------------
Polar Performance Chart
-----------------------------

The :func:`~kdiagram.plot.evaluation.plot_regression_performance`
function provides a holistic, multi-metric dashboard for comparing
regression models. It uses a grouped polar bar chart to visualize
several performance scores at once, making it an exceptional tool for
understanding the unique strengths, weaknesses, and trade-offs of each
model at a single glance.

To appreciate how this plot can distill a complex comparison into a
clear visual summary, let's first deconstruct its components.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each major angular sector is dedicated to a single
     **Evaluation Metric**, such as R², MAE, or RMSE.
   * **Bars Within a Sector:** The different colored bars *within* a
     metric's sector represent the different **Models** being compared.
   * **Radius (r):** The length of a bar represents the model's
     **Normalized Score** for that metric. For this plot, all metrics
     are scaled so that **a longer bar is always better**.
   * **Reference Rings:** The plot includes two rings for context. The
     outer solid green ring is the **"Best Performance"** line (a
     normalized score of 1), while the inner dashed red ring is the
     **"Worst Performance"** line (a score of 0).

With this framework in mind, let's apply the plot to a common challenge
in machine learning: diagnosing the nature of model errors.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">


Default Metrics & Custom Metric Addition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case 1: Diagnosing Model Error Types**

A financial firm is building a model to predict house prices. After
training, they have three candidate models with very different behaviors:

1.  A **"Good Model"** that serves as a solid baseline.
2.  A **"Biased Model"** that is consistently off by a fixed amount
    (e.g., always predicts $10k too low).
3.  A **"High Variance Model"** whose predictions are on average correct,
    but individual errors are large and unpredictable.

The team needs to diagnose and quantify these issues. They start by
visualizing the standard regression metrics.

* *Default Metrics Analysis*


.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Simulate Housing Price Data and Predictions ---
   np.random.seed(0)
   n_samples = 200
   y_true = np.random.rand(n_samples) * 500 # Price in $1000s

   y_pred_good = y_true + np.random.normal(0, 25, n_samples)
   y_pred_biased = y_true - 50 + np.random.normal(0, 10, n_samples)
   y_pred_variance = y_true + np.random.normal(0, 75, n_samples)
   model_names = ["Good Model", "Biased Model", "High Variance"]

   # --- 2. Plotting with Default Metrics ---
   kd.plot_regression_performance(
       y_true,
       y_pred_good, y_pred_biased, y_pred_variance,
       names=model_names,
       title="Performance with Default Metrics",
       metric_labels={'r2': 'R²', 'neg_mean_absolute_error': 'MAE',
                      'neg_root_mean_squared_error': 'RMSE'},
       colors = ["g", "b", "r"],
       savefig="gallery/images/gallery_plot_regression_performance_default.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_default.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart with Default Metrics

   The plot shows the "Biased Model" performing best on MAE but worst
   on R², revealing its specific error profile.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The initial plot already tells a rich story. It provides an immediate 
   diagnosis of each model's behavior. The **"Good Model"** (green) is the 
   clear winner, with the longest bars (best performance) across all three 
   default metrics: R², MAE, and RMSE.
   However, the error profiles of the other two models are also evident. The
   **"Biased Model"** (blue) performs very poorly on R² and RMSE, which
   are metrics that heavily penalize systematic bias. The **"High
   Variance Model"** (red) also performs poorly, particularly on MAE,
   indicating its large, unpredictable errors are detrimental across
   the board.
   

* *Adding a Custom Metric for Deeper Insight*

Now, the team suspects the "High Variance" model is particularly affected by
a few extreme outliers. To investigate this, they add a more robust
metric, **Median Absolute Error (MedAE)**, which is less sensitive to
outliers than MAE or RMSE.

.. code-block:: python
   :linenos:

   from sklearn.metrics import median_absolute_error

   # --- 1. Use the same data as above ---
   # (Assuming y_true, y_preds, and model_names are available)

   # --- 2. Define a custom scorer function ---
   # Note: Scikit-learn convention is "higher is better," so we negate errors.
   def median_abs_error_scorer(y_true, y_pred):
       return -median_absolute_error(y_true, y_pred)

   # --- 3. Plotting with Added Custom Metric ---
   kd.plot_regression_performance(
       y_true,
       y_pred_good, y_pred_biased, y_pred_variance,
       names=model_names,
       metrics=[median_abs_error_scorer], # Add the custom metric
       add_to_defaults=True,           # Keep the default metrics
       title="Performance with Added Custom Metric",
       metric_labels={'r2': 'R²', 'neg_mean_absolute_error': 'MAE',
                      'neg_root_mean_squared_error': 'RMSE',
                      'median_abs_error_scorer': 'MedAE'},
       bp_padding=0.98, # Out the best performance to the main circle.
       colors = ["g", "b", "r"],
       savefig="gallery/images/gallery_plot_regression_performance_custom.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_custom.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart with a Custom Metric

   The addition of the MedAE metric provides a more complete picture of
   each model's error characteristics.


.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The addition of the **MedAE** axis provides a crucial, nuanced
   insight that the other metrics missed. It is the *only* metric
   where the **"Biased Model"** (blue) is the top performer. This is
   because the Median Absolute Error is robust to both bias and
   outliers, highlighting the Biased Model's low underlying error
   variance.
   This combined view confirms that the **"Good Model"** (green) offers
   the best overall balance, while also showing how the choice of
   metric can change a model's ranking. If robustness to outliers were
   the single most important criterion, the "Biased Model" might warrant
   a second look.
   
.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

While the previous use case focused on *relative* performance, sometimes
we must judge models against a fixed, *absolute* standard.

**Use Case 2: Evaluating Against Production Benchmarks**

The housing price prediction team now wants to evaluate two new
"Challenger" models against their current "Champion" model, which is
already in production. The company has established minimum performance
criteria for production models (e.g., R² must be > 0.8).

To do this, they use the function's "values mode" by passing
pre-computed scores, and they set ``norm='global'`` to compare all
models against a fixed, absolute scale.

.. code-block:: python
   :linenos:

   # --- 1. Define pre-computed scores and model names ---
   model_names = ["Champion", "Challenger A", "Challenger B"]
   metric_values = {
       'r2': [0.92, 0.95, 0.78],  # R² (higher is better)
       'neg_mean_absolute_error': [-15.5, -18.2, -12.1] # MAE (negated)
   }

   # --- 2. Define absolute bounds for normalization ---
   # These are the business-defined ranges for performance.
   global_bounds = {
       'r2': (0.80, 1.0), # Min acceptable R² is 0.8
       'neg_mean_absolute_error': (-25.0, -10.0) # Acceptable MAE is 10-25
   }

   # --- 3. Plotting with Global Normalization ---
   kd.plot_regression_performance(
       names=model_names,
       metric_values=metric_values,
       norm='global',
       global_bounds=global_bounds,
       title="Evaluating Challengers Against Production Benchmarks",
       metric_labels={'r2': 'R²', 'neg_mean_absolute_error': 'MAE'},
       savefig="gallery/images/gallery_regression_perf_global_norm.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_regression_perf_global_norm.png
   :align: center
   :width: 75%
   :alt: Polar chart using global normalization to compare against benchmarks.

   The plot shows absolute performance, revealing that Challenger B
   fails to meet the minimum R² threshold.


.. topic:: 💡 Interpretation
   :class: hint

   Because we used ``norm='global'`` , the length of the bars now
   represents absolute performance against the business benchmarks, not
   just a relative ranking.
   The plot reveals a clear trade-off. **Challenger A** (blue) surpasses
   the **Champion** (green) on the R² metric, but this comes at the cost
   of a worse MAE. Conversely, **Challenger B** (red) offers the best
   MAE, but its bar on the R² axis is extremely short, indicating it
   fails to meet the minimum standard for overall model fit. The plot
   makes the decision clear: Challenger A is a viable but imperfect
   replacement, while Challenger B is unsuitable for production despite
   its strong MAE performance.
   
.. admonition:: Best Practice
   :class: hint

   * Use ``norm='per_metric'`` (the default) for exploratory analysis
     to quickly identify the relative strengths and weaknesses of a set
     of candidate models.
   * Use ``norm='global'`` for model monitoring or when comparing
     candidates against established, fixed performance benchmarks.


.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">


Pre-calculated & Overriding Metrics Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the previous use case focused on *relative* performance, sometimes
we must judge models against a fixed, *absolute* standard or visualize
scores that have already been computed.

**Use Case 3: Plotting Pre-calculated Scores**

Often, performance metrics are generated by an automated pipeline or a
colleague and exist in a table or report. The analyst's job is not to
re-run the models, but to create a compelling visualization from these
existing scores.

This example shows how to use the ``metric_values`` parameter to plot a
dictionary of pre-calculated scores directly, decoupling the
visualization from the model execution.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import matplotlib.pyplot as plt

   # --- 1. Assume these scores came from a report ---
   precalculated_scores = {
       'R²': [0.85, 0.55, 0.65],
       'MAE': [-4.0, -10.5, -12.0], # Negated errors
       'RMSE': [-5.0, -11.0, -15.0] # Negated errors
   }
   model_names = ["Good Model", "Biased Model", "High Variance"]

   # --- 2. Plotting ---
   kd.plot_regression_performance(
       metric_values=precalculated_scores,
       names=model_names,
       title="Performance from Pre-calculated Scores",
       cmap='Set2',
       # Optional: Mute axis labels for a cleaner look
       metric_labels={'R²':'R²', 'MAE': 'MAE', 'RMSE': 'RMSE'},
       colors = ["g", "b", "r"],
       savefig="gallery/images/gallery_plot_regression_performance_precalc.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_precalc.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart from Pre-calculated Values

   The chart accurately visualizes the pre-calculated scores, providing
   an instant comparison of the three models.


.. topic:: 💡 Interpretation
   :class: hint

   This workflow is highly efficient, allowing for rapid visualization
   of existing results. The chart accurately reflects the provided
   scores, showing the **"Good Model"** (green) is the best all-rounder,
   with the longest bars on all three metrics. The **"Biased Model"**
   (blue) and **"High Variance Model"** (red) both perform poorly in
   comparison. This demonstrates the function's flexibility as a
   standalone visualization tool.
   
.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

The function's flexibility extends to how it interprets metrics
themselves, ensuring correct visualization even for non-standard,
user-defined functions.

**Use Case 4: Overriding Metric Behavior**

A data scientist creates a new, domain-specific error metric called
``custom_deviation``. Because the function name does not contain "error"
or "loss," the plotting function would incorrectly assume a *higher*
score is better. This would lead to a completely inverted and misleading
visualization for that metric.

This use case demonstrates how the crucial ``higher_is_better``
parameter is used to give the function explicit instructions, ensuring
the plot correctly represents the metric's intent.

.. code-block:: python
   :linenos:

   # --- 1. Use data from the first example ---
   # (Assuming y_true, y_pred_good, y_pred_biased, etc. 
   # are available from previous examples)

   # --- 2. A custom error metric with a neutral name ---
   def custom_deviation(y_true, y_pred):
       return np.mean(np.abs(y_true - y_pred)) # Lower is better

   # --- 3. Plotting with the override ---
   kd.plot_regression_performance(
       y_true,
       y_pred_good, y_pred_biased,
       names=["Good Model", "Biased Model"],
       metrics=['r2', custom_deviation],
       title="Ensuring Correct Metric Interpretation",
       metric_labels={'r2': 'R²', 'custom_deviation': 'Custom Deviation'},
       # Explicitly tell the function lower is better for our metric
       higher_is_better={'custom_deviation': False},
       colors = ["g", "b"],
       savefig="gallery/images/gallery_plot_regression_performance_override.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_override.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart with Overridden Metric Behavior

   By explicitly setting `higher_is_better` to `False`, the plot
   correctly shows the "Biased Model" as the top performer on the
   custom error metric.

.. topic:: 💡 Interpretation
   :class: hint

   This example highlights a critical feature for robust analysis.
   Although the "Good Model" has a lower (better) raw score on the
   ``custom_deviation`` metric, the plot must show it with a longer
   bar.By setting ``higher_is_better={'custom_deviation': False}``, we
   instruct the function to correctly invert this error metric during
   normalization. As a result, the plot correctly visualizes the
   **"Good Model"** (green) with the longest bar on both the R² and
   "Custom Deviation" axes, confirming its superior performance. Without
   this override, the plot would have been dangerously misleading.
   
.. admonition:: Best Practice
   :class: hint
   
   Always use ``higher_is_better`` to manually specify the behavior
   of custom error metrics with ambiguous names to *ensure your
   visualizations are correct*.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">
   
Controlling Normalization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case 5: Controlling Perspective with Normalization**

Beyond adding or removing metrics, one of this plot's most powerful
features is its ability to change the entire analytical "perspective"
using the ``norm`` parameter. This controls how raw scores are scaled
into bar lengths, allowing you to seamlessly switch between asking
"Which model is relatively better?" and "Does this model meet our
absolute quality standards?".

To demonstrate this, we will generate data for a "Good Model" and a
"Biased Model" and visualize their performance using all three
normalization strategies.


.. code-block:: python
   :linenos:

   import kdiagram as kd
   import matplotlib.pyplot as plt
   import numpy as np

   # --- 1. Define distinct model error profiles ---
   np.random.seed(0)
   n_samples = 200
   y_true = np.random.rand(n_samples) * 50

   y_pred_good = y_true + np.random.normal(0, 5, n_samples)
   y_pred_biased = y_true - 10 + np.random.normal(0, 2, n_samples)
   model_names = ["Good Model", "Biased Model"]

   # --- 2. Define consistent labels for all plots ---
   metric_labels = {'r2': 'R²', 'neg_mean_absolute_error': 'MAE',
                    'neg_root_mean_squared_error': 'RMSE'}
                    
   colors =['green', 'blue']

Perspective 1: Relative Comparison (`norm="per_metric"`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the default and most common mode. It scales each metric
independently, setting the best-performing model for that metric to 1
("Best") and the worst to 0 ("Worst"). This is ideal for quickly
understanding the relative strengths and weaknesses of the models you are
comparing.

.. code-block:: python
   :linenos:

   kd.plot_regression_performance(
       y_true, y_pred_good, y_pred_biased,
       names=model_names,
       metric_labels=metric_labels,
       norm="per_metric",
       title="Relative Performance (Per-Metric Norm)",
       colors = colors, 
       savefig="gallery/images/gallery_plot_regression_performance_per_metric.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_per_metric.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart with Per-Metric Normalization

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a clear verdict on the models' relative ranking.
   The **"Good Model"** (green) is demonstrably superior, with its bars
   reaching the outer "Best Performance" ring for all three metrics:
   R², MAE, and RMSE.
   Conversely, the **"Biased Model"** (blue) is the worst performer on
   every metric, so its bars are consistently at the inner "Worst
   Performance" ring. This view is perfect for an initial exploratory
   analysis, making it immediately obvious that there is no trade-off
   to consider; the "Good Model" is dominant.

Perspective 2: Absolute Benchmarks (`norm="global"`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A relative ranking is useful, but in production, models often need to
meet fixed quality standards. This mode compares models against a
predefined, absolute scale that you define with ``global_bounds``.

.. code-block:: python
   :linenos:

   # Define a benchmark for what "good" and "bad" means for each metric
   global_bounds = {
       "r2": (0.0, 1.0),
       "neg_mean_absolute_error": (-15.0, 0.0),
       "neg_root_mean_squared_error": (-20.0, 0.0),
   }

   kd.plot_regression_performance(
       y_true, y_pred_good, y_pred_biased,
       names=model_names,
       metric_labels=metric_labels,
       norm="global",
       global_bounds=global_bounds,
       title="Absolute Performance (Global Norm)",
       colors = colors, 
       savefig="gallery/images/gallery_plot_regression_performance_global.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_global.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart with Global Normalization

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   In this view, the bar lengths represent absolute performance against
   the predefined benchmarks, not just a relative ranking. The bars no
   longer necessarily touch the edges.
   The **"Good Model"** (green) performs very well against the absolute
   standards, with long bars for R², MAE, and RMSE. In contrast, the
   **"Biased Model"** (blue) has a very short bar for R², accurately
   reflecting its poor performance against the 0.0 to 1.0 benchmark.
   This perspective is essential for determining if a model meets
   production-ready criteria.

Perspective 3: Raw Scores (`norm="none"`) or Expert Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, for expert analysis or technical reports, you may need to see
the un-scaled metric values directly. This mode provides the most
direct, unfiltered view, but requires careful interpretation as each
axis has a different scale.

.. warning::
   :class: critical

   In Expert Mode, do not visually compare the length of a bar
   for one metric to the length of a bar for another. This view is for
   reading the **exact numerical scores**, not for comparing shapes.
   
.. code-block:: python
   :linenos:

   kd.plot_regression_performance(
       y_true, y_pred_good, y_pred_biased,
       names=model_names,
       metric_labels=metric_labels,
       norm="none",
       title="Raw Performance Scores (No Norm)",
       colors = colors, 
       savefig="gallery/images/gallery_plot_regression_performance_none.png"
   )
   plt.close()

.. figure:: ../images/evaluation/gallery_plot_regression_performance_none.png
   :align: center
   :width: 75%
   :alt: Polar Performance Chart with No Normalization

.. topic:: 🧠 Analysis and Interpretation (Expert Mode)
   :class: hint

   This mode provides the most direct, unfiltered view of the raw
   performance scores, but it requires careful interpretation because each
   metric axis exists on its own unique scale. The key is to **read
   each metric axis independently**, like separate bar charts radiating
   from the center.

   For example, on the **MAE** axis, the "Good Model's" bar (green)
   reaches a raw score of approximately **-4.7**, while the "Biased
   Model's" bar (blue) reaches about **-10**. Since -4.7 is a better
   (higher) score, the "Good Model" is the clear winner. Similarly, on
   the **R²** axis, the "Good Model's" score of **0.88** is vastly
   superior.

   **CRITICAL WARNING:** Do not visually compare the length of a bar for
   one metric to the length of a bar for another (e.g., comparing an
   R² bar to an RMSE bar). This view is for reading the **exact
   numerical scores**, not for comparing shapes.
   

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the definitions of these regression metrics and
normalization strategies, please refer to the 
:ref:`ug_plot_regression_performance` section.


.. _gallery_application_regression_metrics:

-----------------------------------------------------------
Application: Evaluating Probabilistic Energy Forecasts
-----------------------------------------------------------

In many critical industries, a single-point forecast is not enough.
Decision-makers need to understand the full range of potential
outcomes—the uncertainty—to manage risk effectively. This is especially
true in energy markets.

This application demonstrates how to combine the Polar Pinball Loss plot
and the Polar Performance Chart into a single diagnostic dashboard to
conduct a comprehensive evaluation of probabilistic forecasts.

**The Problem: Forecasting National Grid Electricity Demand**


.. admonition:: Practical Example

    A national grid operator needs to forecast the next day's electricity
    demand. This is a high-stakes problem with significant financial and
    societal consequences — **Under-prediction** If demand is higher than 
    predicted, the  operator may need to purchase emergency power at exorbitant
    prices or, in the worst case, initiate rolling blackouts. This means the model's
    performance at **high quantiles** (e.g., the 95th percentile,
    representing peak demand) is critical — **Over-prediction** If demand is 
    lower than predicted, costly power generation is wasted. This makes performance 
    at **low quantiles** important as well.

The operator is evaluating two new probabilistic forecasting models: a
complex **Deep Learning (N-BEATS)** model and a robust **Quantile
Regression Forest (QRF)**. A comprehensive evaluation must assess both
the accuracy of the median (point) forecast and the reliability of the
full predicted distribution.

**Translating the Problem into a Visual Dashboard**

To get a complete picture, we will generate a dashboard that allows us to
move from a granular, quantile-by-quantile diagnosis to a holistic
comparison of standard performance metrics. A 2x2 layout provides a
compact and effective way to arrange these plots.

The following code simulates the models' performance on this task and
creates the diagnostic dashboard.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   from scipy.stats import norm
   import matplotlib.pyplot as plt

   # --- 1. Simulate Electricity Demand Data ---
   np.random.seed(42)
   n_samples = 2000
   y_true = 50 + 10 * np.sin(np.arange(n_samples) * np.pi / 12) \
            + np.random.normal(0, 3, n_samples) # in Gigawatts (GW)

   quantiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
   q_map = {q: i for i, q in enumerate(quantiles)}

   # --- 2. Simulate Quantile Predictions from Two Models ---
   # N-BEATS: Excellent at the median, slightly less certain at tails
   nbeats_scales = np.array([5, 4, 3, 2, 3, 4, 5])
   y_preds_nbeats = norm.ppf(
       quantiles, loc=y_true[:, np.newaxis], scale=nbeats_scales)

   # QRF: Very robust at the tails, slightly less accurate at the median
   qrf_scales = np.array([4.5, 3.8, 3.2, 2.5, 3.2, 3.8, 4.5])
   y_preds_qrf = norm.ppf(
       quantiles, loc=y_true[:, np.newaxis] + 0.5, scale=qrf_scales)

   # Extract the median (0.5 quantile) as the point forecast
   y_pred_nbeats_median = y_preds_nbeats[:, q_map[0.5]]
   y_pred_qrf_median = y_preds_qrf[:, q_map[0.5]]

   # --- 3. Create the 2x2 Dashboard ---
   fig, axes = plt.subplots(2, 2, figsize=(18, 18),
                            subplot_kw={'projection': 'polar'})
   fig.suptitle("Electricity Demand Forecast Evaluation Dashboard",
                fontsize=24, y=0.98)

   # Top Row: Pinball Loss profiles for each model
   kd.plot_pinball_loss(
       y_true, y_preds_nbeats, quantiles, ax=axes[0, 0],
       title="Pinball Loss Profile (N-BEATS)", colors=['#8a2be2']
   )
   kd.plot_pinball_loss(
       y_true, y_preds_qrf, quantiles, ax=axes[0, 1],
       title="Pinball Loss Profile (QRF)", colors=['#de3163']
   )

   # Bottom-Left: Performance of the median forecasts
   kd.plot_regression_performance(
       y_true, y_pred_nbeats_median, y_pred_qrf_median,
       names=["N-BEATS (Median)", "QRF (Median)"], ax=axes[1, 0],
       title="Median Forecast Performance",
       metric_labels={'r2': 'R²', 'neg_mean_absolute_error': 'MAE',
                      'neg_root_mean_squared_error': 'RMSE'}, 
       colors = ['#8a2be2', '#de3163'],
   )

   # Hide the unused subplot in the bottom-right
   fig.delaxes(axes[1, 1])

   fig.tight_layout(pad=2.0)
   fig.savefig("gallery/images/gallery_evaluation_regression_dashboard.png")
   plt.close(fig)

.. figure:: ../images/evaluation/gallery_evaluation_regression_dashboard.png
   :align: center
   :width: 100%
   :alt: A 2x2 dashboard showing different polar regression evaluation plots.

   A comprehensive dashboard using Pinball Loss and standard regression
   metrics to provide a holistic view of probabilistic forecast skill.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This dashboard provides a complete story, allowing the grid operator
   to make a clear, evidence-based decision.

   1.  **Top Row (Pinball Loss Profiles):** These plots compare the
       quantile-specific accuracy. The **N-BEATS model** (top-left,
       purple) demonstrates superior performance across the entire
       distribution. Its polygon is visibly smaller than the QRF's,
       indicating a lower (better) pinball loss at every quantile. Its
       profile is sharpest at the 0.5 quantile, showing it is an
       exceptional *specialist* at predicting the median demand. However, the
       **QRF model** (top-right, pink) is clearly less accurate, with
       higher losses across all quantiles.

   2.  **Bottom-Left (Median Forecast Performance)** confirms
       the overwhelming superiority of the N-BEATS model's point
       forecast. The **N-BEATS model** achieves the "Best
       Performance" on all three metrics (R², MAE, and RMSE), while the
       **QRF model** (yellow bars) registers the "Worst Performance" on
       all three.

   **Conclusion:** The dashboard provides a clear and decisive
   recommendation. The **N-BEATS model** is unequivocally superior in
   every aspect measured. It provides more accurate quantile forecasts
   across the entire distribution and a vastly more accurate median
   forecast. The plots empower the operator to confidently select the
   N-BEATS model for deployment.
   
.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the mathematical concepts behind these evaluation
metrics, please refer to the main **User Guide** :ref:`userguide_evaluation`.