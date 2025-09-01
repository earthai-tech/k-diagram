.. _gallery_comparison:

============================
Model Comparison Gallery
============================

This gallery page showcases plots from `k-diagram` designed for
comparing the performance of multiple models across various metrics,
primarily using radar charts.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``images/gallery_model_comparison.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots (likely an ``images`` subdirectory relative to
   this file).

.. _gallery_plot_model_comparison: 

---------------------------------
Multi-Metric Model Comparison
---------------------------------

The :func:`~kdiagram.plot.comparison.plot_model_comparison` function is
a tool for moving beyond single-score evaluations. It creates
a polar radar chart to visualize and compare multiple models across
several performance metrics simultaneously, providing a holistic
"fingerprint" of each model's strengths and weaknesses.

First, let's break down the components of this comparative plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular axis represents a different
     **performance metric** (e.g., R², MAE, Training Time).
   * **Radius (r):** Corresponds to the **normalized performance score**
     for that metric, typically scaled to the range [0, 1]. To maintain
     consistency, all metrics are scaled such that a **larger radius is
     always better** (e.g., lower MAE or faster training time results
     in a larger radius).
   * **Polygon:** Each colored polygon represents a **model**, with its
     vertices showing its performance on each metric. The overall shape
     and size of the polygon provide an at-a-glance summary of the
     model's performance profile.

With this framework, we can now apply the plot to a real-world model
selection problem, progressing from a standard regression task to a more
nuanced classification task.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Standard Regression Model Comparison**

The most common use for this plot is to select the best model for a
standard regression task by balancing accuracy, error, and efficiency.

Let's imagine an analytics team at an e-commerce company has built
three different models to predict sales revenue: a fast but simple
`Ridge` regression, a `Lasso` model that performs feature selection,
and a more complex `Decision Tree`. They need to choose the best
all-around performer.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Sales Revenue Forecast ---
   np.random.seed(42)
   n_samples = 100
   y_true_reg = np.random.rand(n_samples) * 50 + 10 # True revenue
   # Model 1 (Ridge): Good fit, fast
   y_pred_r1 = y_true_reg + np.random.normal(0, 4, n_samples)
   # Model 2 (Lasso): Similar fit, slightly slower
   y_pred_r2 = y_true_reg * 0.98 + 1 + np.random.normal(0, 4.5, n_samples)
   # Model 3 (Tree): Overfit, slower, poor on some metrics
   y_pred_r3 = y_true_reg + np.random.normal(2, 8, n_samples)

   times = [0.1, 0.3, 0.8] # Training times in seconds
   names = ['Ridge', 'Lasso', 'Tree']

   # --- 2. Plotting ---
   # Using default regression metrics: ['r2', 'mae', 'mape', 'rmse']
   kd.plot_model_comparison(
       y_true_reg,
       y_pred_r1, y_pred_r2, y_pred_r3,
       train_times=times,
       names=names,
       title="Use Case 1: E-Commerce Sales Model Comparison",
       scale='norm',
       savefig="gallery/images/gallery_model_comparison_regression.png"
   )
   plt.close()

.. figure:: ../images/comparison/gallery_model_comparison_regression.png
   :align: center
   :width: 70%
   :alt: A radar chart comparing three regression models.

   A radar chart showing the performance profiles of Ridge, Lasso, and
   Decision Tree models across five different metrics.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot reveals a classic trade-off between performance and
   efficiency. The **Ridge model (blue)** is the clear winner on
   **all predictive performance metrics** (``r2``, ``mae``, ``mape``,
   and ``rmse``), as its polygon has the largest overall area and extends
   furthest on these axes. However, the **Lasso model (orange)**, while
   slightly less accurate, is the fastest to train, as shown by its
   superior score on the ``Train Time (s)`` axis. The Tree model is not
   visible, indicating its performance was the lowest on all metrics. The
   choice is clear: use the Ridge model for the highest accuracy, or the
   Lasso model for a good balance of speed and performance.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Evaluating a Classification Task with a Custom Metric**

This plot is equally usefull for classification. The default metrics
will automatically switch to `['accuracy', 'precision', 'recall', 'f1']`,
but we can also provide our own custom metrics to evaluate performance
on criteria that are specific to our business problem.

Let's consider a medical diagnosis model that predicts whether a patient
has a rare disease. In this case, **Recall** (correctly identifying sick
patients) is far more important than Precision. We can create a custom,
weighted F-beta score to reflect this and add it to our plot.

.. code-block:: python
   :linenos:

   from sklearn.metrics import fbeta_score

   # --- 1. Data Generation: Medical Diagnosis ---
   np.random.seed(0)
   n_samples = 200
   y_true_clf = np.array([0] * 180 + [1] * 20) # Imbalanced data
   # Model A: High precision, but misses sick patients (low recall)
   y_pred_A = np.copy(y_true_clf)
   y_pred_A[np.random.choice(np.where(y_true_clf==1)[0], 12, False)] = 0
   # Model B: Lower precision, but finds most sick patients (high recall)
   y_pred_B = np.copy(y_true_clf)
   y_pred_B[np.random.choice(np.where(y_true_clf==0)[0], 20, False)] = 1

   # --- 2. Define a custom metric that prioritizes Recall ---
   # An F-beta score with beta=2 weighs recall higher than precision
   f2_score = lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2)
   f2_score.__name__ = "F2-Score (Recall Focus)" # Give it a nice name for the plot

   # --- 3. Plotting with default and custom metrics ---
   kd.plot_model_comparison(
       y_true_clf,
       y_pred_A,
       y_pred_B,
       names=['Model A (High Precision)', 'Model B (High Recall)'],
       metrics=['accuracy', 'precision', 'recall', f2_score], # Add our custom metric
       title="Use Case 2: Medical Diagnosis Classifier Comparison",
       scale='norm',
       savefig="gallery/images/gallery_model_comparison_classification.png"
   )
   plt.close()

.. figure:: ../images/comparison/gallery_model_comparison_classification.png
   :align: center
   :width: 70%
   :alt: A radar chart comparing two classification models with a custom metric.

   A radar chart showing how two classifiers perform on standard
   metrics as well as a custom "F2-Score" that prioritizes recall.

.. topic:: 🧠 Interpretation
   :class: hint

   The radar chart illustrates a stark and mutually exclusive trade-off
   between the two classifiers. **Model A (blue)** achieves **perfect
   scores** on the ``accuracy`` and ``precision`` axes but completely fails
   on ``recall`` and our custom ``F2-Score``, with scores of zero.
   Conversely, **Model B (orange)** shows the exact opposite profile: it
   scores perfectly on ``recall`` and the ``F2-Score`` but fails
   completely on accuracy and precision.

   For a medical diagnosis where failing to identify a sick patient
   (low recall) is a critical error, Model B is the only viable choice.
   The custom ``F2-Score (Recall Focus)`` axis correctly identifies it
   as the superior model for this specific, high-stakes application.

.. admonition:: Best Practice
   :class: best-practice

   Don't rely solely on default metrics. For real-world problems,
   business needs often dictate that some errors are more costly than
   others. Adding custom metrics to the ``plot_model_comparison``
   function, as shown in this use case, is a powerful way to ensure your
   model evaluation aligns with your specific goals.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind these
evaluation metrics, please refer back to the main
:ref:`ug_plot_model_comparison` section.
     
     
.. _gallery_plot_reliability:

-----------------------------------------
Model Reliability (Calibration) Diagram
-----------------------------------------

The :func:`~kdiagram.plot.comparison.plot_reliability_diagram` is the
industry-standard tool for assessing the calibration of a binary
classifier. It answers a crucial question: "When my model predicts a
70% probability of an event, does that event actually happen 70% of the
time?" A model whose probabilities accurately reflect real-world
frequencies is considered "well-calibrated" and is essential for making
trustworthy, risk-based decisions.

Let's begin by breaking down the components of this fundamental plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **X-Axis (Mean Predicted Probability):** For each bin, this is the
     average of the probabilities predicted by the model. This is also
     referred to as the forecast's **confidence**.
   * **Y-Axis (Observed Frequency):** For each bin, this is the actual
     fraction of positive cases observed in the data. This is also
     referred to as the forecast's **accuracy**.
   * **Diagonal Line** (:math:`y=x`): This is the line of **perfect
     calibration**. A model whose points fall on this line is perfectly
     calibrated.
   * **Counts Panel (Bottom):** A histogram showing the number of
     predictions that fall into each probability bin, which helps in
     diagnosing if the model is timid (most predictions near 0.5) or
     decisive (most predictions near 0 or 1).

With this in mind, let's explore how to use this plot to diagnose and
compare the reliability of different models.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Basic Calibration Check with Uniform Bins**

The most common use case is to get a quick, initial assessment of a
single model's calibration. For this, we can use the default `uniform`
binning strategy, which creates equally spaced bins across the [0, 1]
probability range.

Let's evaluate a model trained to predict customer churn, where a "1"
means the customer is likely to leave.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Customer Churn Predictions ---
   np.random.seed(0)
   n_customers = 2000
   # True outcome: ~30% of customers churn
   y_true = (np.random.rand(n_customers) < 0.3).astype(int)
   # A reasonably good, but not perfect, model
   y_pred = np.clip(y_true * 0.4 + 0.3 + np.random.normal(0, 0.15, n_customers), 0.01, 0.99)

   # --- 2. Plotting ---
   kd.plot_reliability_diagram(
       y_true, y_pred,
       names=['Churn Model'],
       n_bins=10,
       strategy="uniform", # Default, but explicit here
       title='Use Case 1: Basic Calibration Check',
       savefig="gallery/images/gallery_reliability_diagram_basic.png"
   )
   plt.close()


.. figure:: ../images/comparison/gallery_reliability_diagram_basic.png
   :align: center
   :width: 70%
   :alt: A basic reliability diagram showing a single model's calibration.

   A reliability diagram showing the model's calibration curve relative
   to the perfect diagonal. The counts panel below shows the
   distribution of its predictions.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a clear, initial diagnosis. The model's calibration
   curve (blue line) generally follows the dashed diagonal reference
   line, suggesting it is reasonably well-calibrated. However, for
   higher predicted probabilities (confidence > 0.6), the curve dips
   slightly below the diagonal, indicating a tendency towards
   **over-confidence** in this range—when it is highly confident that a
   customer will churn, the actual churn rate is slightly lower. The
   **counts panel** at the bottom shows that the model is quite decisive,
   with most of its predictions falling into the bins near 0.2 and 0.7.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Comparing Models with Quantile Binning**

A more advanced task is to compare the reliability of multiple competing
models. For this, `quantile` binning is often superior to `uniform`
binning, as it ensures that each bin contains an equal number of
samples, providing a more stable estimate of the observed frequency.

Let's compare our "Churn Model" to a new "Calibrated Model" that has been
post-processed to improve its reliability.

.. code-block:: python
   :linenos:

   # --- 1. Data Generation (uses y_true and y_pred from previous step) ---
   # Create a second, better-calibrated model's predictions
   y_pred_calibrated = np.clip(y_true * 0.35 + 0.32 + np.random.normal(0, 0.1, n_customers), 0.01, 0.99)

   # --- 2. Plotting ---
   kd.plot_reliability_diagram(
       y_true, y_pred, y_pred_calibrated,
       names=['Original Model', 'Calibrated Model'],
       n_bins=12,
       strategy="quantile", # Use quantile binning for a stable comparison
       error_bars="wilson",  # Add Wilson confidence intervals
       title='Use Case 2: Comparing Model Reliability',
       savefig="gallery/images/gallery_reliability_diagram_compare.png"
   )
   plt.close()


.. figure:: ../images/comparison/gallery_reliability_diagram_compare.png
   :align: center
   :width: 70%
   :alt: A reliability diagram comparing two models using quantile binning.

   Two calibration curves are shown. The "Calibrated Model" (orange)
   hugs the diagonal line more closely than the "Original Model" (blue).

.. topic:: 🧠 Interpretation
   :class: hint

   This side-by-side comparison on the same axes reveals the distinct
   calibration profiles of the two models. The **Original Model** (blue)
   clearly deviates from the diagonal, exhibiting significant
   **under-confidence** for predicted probabilities between 0.4 and 0.6.
   The **"Calibrated Model"** (orange) shows a different pattern of
   miscalibration, with a noticeable "S" shape where it is first
   under-confident and then over-confident.

   Interestingly, the quantitative metrics in the legend confirm this
   visual assessment: the attempted calibration was not successful in this
   case, as the "Calibrated Model" has a slightly **worse (higher) ECE
   score** than the original. This is a perfect example of why
   reliability diagrams are so crucial—they provide a nuanced diagnostic
   that goes beyond simple labels and reveals the true behavior of a
   model's probability outputs.

.. admonition:: Best Practice
   :class: best-practice

   When comparing multiple models, using ``strategy="quantile"`` is
   highly recommended. It prevents bins from being empty and provides
   more stable and reliable estimates of the observed frequencies, leading
   to a fairer comparison between models. Also, including error bars
   (e.g., ``error_bars="wilson"``) provides crucial context about the
   statistical uncertainty of your assessment.

.. admonition:: See Also
   :class: seealso

   For an alternative, and often more intuitive, way to visualize model
   calibration, see the :func:`~kdiagram.plot.comparison.plot_polar_reliability`
   function. It transforms this Cartesian plot into a polar spiral,
   which can make miscalibration patterns even easier to spot.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 3: Weighted Calibration for High-Value Segments**

In many real-world business problems, not all prediction errors are
created equal. An error on a high-value customer can be far more costly
than an error on a standard customer. A model might appear well-calibrated
overall, but this aggregate view can hide poor performance on the most
critical segments. The ``sample_weight`` parameter is a powerful tool
for diagnosing this exact problem.

.. admonition:: Best Practice
   :class: best-practice

   When the business impact of your model's predictions is not uniform
   across all samples, always perform a weighted calibration analysis.
   Use the ``sample_weight`` parameter to assign higher importance to
   high-value customers, critical events, or costly failure modes to
   ensure your model is reliable where it matters most.

Let's tackle a common problem in customer retention: ensuring our churn
model is reliable for our most valuable "premium" subscribers.

.. admonition:: Practical Example

   A streaming service uses a model to predict the probability that a
   subscriber will churn (cancel their subscription). The model's
   overall calibration appears to be good. However, the business is most
   concerned about retaining its "premium" subscribers, as they account
   for a disproportionate amount of revenue. Is the model's churn
   probability trustworthy *specifically for this high-value segment*?

   We will create a side-by-side comparison. The left plot will show
   the standard, unweighted reliability, while the right plot will use
   ``sample_weight`` to give 10x more importance to the premium
   subscribers, revealing the model's true performance for this
   critical group.

   .. code-block:: python
      :linenos:

      import kdiagram as kd
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt

      # --- 1. Data Generation: Churn with a high-value segment ---
      np.random.seed(10)
      n_customers = 5000
      # True churn status
      y_true = (np.random.rand(n_customers) < 0.2).astype(int)
      # Create sample weights: 10% are "premium" customers with 10x weight
      sample_weight = np.ones(n_customers)
      premium_indices = np.random.choice(n_customers, 500, replace=False)
      sample_weight[premium_indices] = 10

      # --- 2. Create biased predictions FOR THE PREMIUM SEGMENT ---
      # The model is well-calibrated for standard users but overconfident
      # for premium users (predicts lower churn probability than is real)
      y_pred = np.clip(y_true * 0.5 + 0.15 + np.random.normal(0, 0.1, n_customers), 0.01, 0.99)
      # Introduce the bias for the premium segment
      y_pred[premium_indices] *= 0.5

      # --- 3. Create side-by-side plots ---
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

      kd.plot_reliability_diagram(
          y_true, y_pred,
          ax=ax1,
          names=['Churn Model'],
          title='Use Case 3a: Unweighted Reliability (All Customers)',
          savefig=None # Prevent saving from the first call
      )
      kd.plot_reliability_diagram(
          y_true, y_pred,
          ax=ax2,
          sample_weight=sample_weight, # Apply the crucial sample weights
          names=['Churn Model'],
          title='Use Case 3b: Weighted Reliability (Premium Focus)',
          savefig=None # Prevent saving from the second call
      )

      fig.suptitle('Diagnosing Hidden Bias with Weighted Calibration', fontsize=16)
      fig.tight_layout(rect=[0, 0, 1, 0.95])
      fig.savefig("gallery/images/gallery_reliability_diagram_weighted.png")
      plt.close(fig)

.. figure:: ../images/comparison/gallery_reliability_diagram_weighted.png
   :align: center
   :width: 90%
   :alt: Side-by-side reliability diagrams, one unweighted and one weighted.

   A two-panel figure. The left plot (unweighted) shows a reasonably
   well-calibrated model. The right plot (weighted by customer value)
   reveals the same model is severely overconfident for its most
   important customers.

.. topic:: 🧠 Interpretation
   :class: hint

   This side-by-side comparison reveals a critical, hidden flaw that
   would be missed by a standard analysis. The **Unweighted Reliability**
   plot (left) suggests the model is acceptable. Because the premium
   subscribers are only 10% of the data, their poor calibration is
   masked by the good performance on the majority of standard users.

   However, the **Weighted Reliability** plot (right) tells a completely
   different and more important story. By giving more weight to the
   premium segment, the curve is now dragged far below the diagonal. This
   shows that for high-value customers, the model is **severely
   overconfident**. It consistently underestimates their churn risk,
   which could lead the business to neglect retention efforts for its
   most important user base. This analysis demonstrates that the model is
   not yet fit for its intended business purpose.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical theory behind calibration
and proper scoring rules, please refer back to the main
:ref:`ug_plot_reliability` section.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">


.. _gallery_plot_polar_reliability:

------------------------------------------------
Polar Reliability Diagram (Calibration Spiral)
------------------------------------------------

The :func:`~kdiagram.plot.comparison.plot_polar_reliability` function
provides a novel and highly intuitive visualization of model calibration.
It transforms the traditional reliability diagram into a "calibration
spiral," where deviations from a perfect spiral immediately reveal the
nature and location of a model's miscalibrations through diagnostic
coloring.

First, let's break down the components of this innovative plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the **mean predicted probability**
     (:math:`\bar{p}_k`) for each bin, sweeping from 0.0 at 0° to 1.0
     at 90°. This is the model's *confidence*.
   * **Radius (r):** Represents the **observed frequency** of the
     event (:math:`\bar{y}_k`) for each bin. This is the *actual outcome*.
   * **Perfect Calibration Spiral:** The dashed black line represents the
     ideal case where :math:`r = \frac{2\theta}{\pi}`
     (:math:`\bar{y}_k = \bar{p}_k`). A model's spiral should lie
     directly on this line.
   * **Color:** The color of the model's spiral is a diagnostic tool,
     representing the calibration error (:math:`\bar{y}_k - \bar{p}_k`).
     Colors on one side of the colormap's center (e.g., reds) indicate
     over-confidence, while colors on the other side (e.g., blues)
     indicate under-confidence.

With this in mind, let's apply the plot to a real-world problem to see
how it uncovers different types of miscalibration.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Diagnosing an Over-Confident Model**

A common failure mode for classifiers, especially on complex tasks, is
**overconfidence**. The model assigns high probabilities to its
predictions, but its real-world accuracy doesn't match this high level
of certainty.

Let's simulate a scenario in medical diagnostics, where a model is
trained to predict the probability of a disease. An overconfident model
might predict a 90% probability of disease when the actual rate for
such patients is only 70%, which could lead to unnecessary treatments.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: A Well-Calibrated vs. Over-Confident Model ---
   np.random.seed(0)
   n_samples = 2000
   # The true probability of the event is 0.4
   y_true = (np.random.rand(n_samples) < 0.4).astype(int)

   # A well-calibrated model's probabilities are realistic
   calibrated_preds = np.clip(0.4 + np.random.normal(0, 0.15, n_samples), 0, 1)

   # An over-confident model pushes probabilities towards the extremes of 0 and 1
   overconfident_preds = np.clip(0.4 + np.random.normal(0, 0.3, n_samples), 0, 1)

   # --- 2. Plotting ---
   kd.plot_polar_reliability(
       y_true,
       calibrated_preds,
       overconfident_preds,
       names=["Well-Calibrated", "Over-Confident"],
       n_bins=15,
       cmap='coolwarm',
       title="Use Case 1: Diagnosing an Over-Confident Model",
       savefig="gallery/images/gallery_polar_reliability_overconfident.png"
   )
   plt.close()

.. figure:: ../images/comparison/gallery_polar_reliability_overconfident.png
   :align: center
   :width: 70%
   :alt: A polar reliability diagram showing one well-calibrated and one over-confident model.

   The "Well-Calibrated" model's spiral closely follows the dashed
   reference line, while the "Over-Confident" model's spiral falls
   inside the reference.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot makes the models' behaviors easy to distinguish. The
   **"Well-Calibrated"** model's spiral (not shown with a separate legend
   entry but represented by the line segments colored near the neutral
   center of the colormap) adheres very closely to the dashed "Perfect
   Calibration" spiral. This is the signature of a reliable model.

   In stark contrast, the **"Over-Confident"** model's spiral deviates
   significantly. In the region of higher predicted probabilities
   (larger angles), its spiral falls **inside** the dashed reference
   line, and the diagnostic coloring turns red. This is a clear visual
   indication of over-confidence: the observed frequency (radius) is
   systematically lower than the predicted probability (angle).

.. admonition:: See Also
   :class: seealso

   This plot is the polar-coordinate counterpart to the traditional
   Cartesian :func:`~kdiagram.plot.comparison.plot_reliability_diagram`.
   While both show the same underlying data, the spiral format can often
   make deviations and the nature of miscalibration more intuitive to
   see at a glance.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical theory behind calibration
and reliability, please refer back to the main
:ref:`ug_plot_polar_reliability` section.
   
.. _gallery_plot_horizon_metrics:

--------------------------------------
Comparing Metrics Across Horizons
--------------------------------------

The :func:`~kdiagram.plot.comparison.plot_horizon_metrics` function
creates a polar bar chart designed to compare two key metrics across a
set of distinct categories, such as different forecast horizons. It's a
powerful tool for visualizing how a model's uncertainty (bar height)
and central tendency (bar color) evolve over time or differ between
groups.

First, let's break down the components of this two-dimensional summary plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular sector represents a distinct **category**
     or **horizon** (e.g., "H+1", "H+2"), corresponding to a row in the
     input DataFrame. The labels for these sectors are provided via the
     ``xtick_labels`` parameter.
   * **Radius (r):** The height of each bar represents the **average
     value of a primary metric**. By default, this is the mean prediction
     interval width (:math:`Q_{upper} - Q_{lower}`).
   * **Color:** The color of each bar visualizes a **secondary metric**.
     By default, this is the mean of the median (Q50) predictions for
     that category, adding another layer of information to the comparison.

With this in mind, let's apply the plot to a classic forecasting
problem.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Standard Forecast Horizon Analysis**

The most common use of this plot is to see how a model's uncertainty
and central prediction change as it forecasts further into the future.
It's a typical and expected behavior for uncertainty to grow over longer
lead times, and this plot quantifies that drift.

Let's simulate a multi-step forecast where both the predicted value and
its uncertainty increase for each step.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Multi-step Forecast ---
   # Each row represents a forecast horizon (H+1 to H+6)
   # Each column is a different sample of that forecast
   horizons = ["H+1", "H+2", "H+3", "H+4", "H+5", "H+6"]
   df = pd.DataFrame(index=horizons)
   q10_cols, q90_cols, q50_cols = [], [], []

   for i in range(len(horizons)):
       # Both median and width increase with the horizon
       median = 10 + 5 * i
       width = 5 + 3 * i
       # Create two samples for each horizon
       df[f'q10_s{i}_1'] = median - width/2 + np.random.randn()
       df[f'q90_s{i}_1'] = median + width/2 + np.random.randn()
       df[f'q50_s{i}_1'] = median + np.random.randn()
       df[f'q10_s{i}_2'] = median - width/2 + np.random.randn()
       df[f'q90_s{i}_2'] = median + width/2 + np.random.randn()
       df[f'q50_s{i}_2'] = median + np.random.randn()
       q10_cols.extend([f'q10_s{i}_1', f'q10_s{i}_2'])
       q90_cols.extend([f'q90_s{i}_1', f'q90_s{i}_2'])
       q50_cols.extend([f'q50_s{i}_1', f'q50_s{i}_2'])

   # Reshape for the function: rows are horizons, cols are samples
   df_horizons = pd.DataFrame(index=horizons)
   for i in range(len(horizons)):
       df_horizons.loc[f"H+{i+1}", 'q10_s1'] = df.loc[f"H+{i+1}", f'q10_s{i}_1']
       df_horizons.loc[f"H+{i+1}", 'q90_s1'] = df.loc[f"H+{i+1}", f'q90_s{i}_1']
       df_horizons.loc[f"H+{i+1}", 'q50_s1'] = df.loc[f"H+{i+1}", f'q50_s{i}_1']
       df_horizons.loc[f"H+{i+1}", 'q10_s2'] = df.loc[f"H+{i+1}", f'q10_s{i}_2']
       df_horizons.loc[f"H+{i+1}", 'q90_s2'] = df.loc[f"H+{i+1}", f'q90_s{i}_2']
       df_horizons.loc[f"H+{i+1}", 'q50_s2'] = df.loc[f"H+{i+1}", f'q50_s{i}_2']

   # --- 2. Plotting ---
   kd.plot_horizon_metrics(
       df=df_horizons,
       qlow_cols=['q10_s1', 'q10_s2'],
       qup_cols=['q90_s1', 'q90_s2'],
       q50_cols=['q50_s1', 'q50_s2'],
       title="Use Case 1: Mean Interval Width Across Horizons",
       xtick_labels=horizons,
       r_label="Mean Interval Width",
       cbar_label="Mean Q50 Value",
       savefig="gallery/images/gallery_horizon_metrics_basic.png"
   )
   plt.close()

.. figure:: ../images/comparison/gallery_horizon_metrics_basic.png
   :align: center
   :width: 70%
   :alt: A polar bar chart showing increasing bar height and changing color.

   A polar bar chart where both the height of the bars (uncertainty)
   and their color (median prediction) increase progressively across
   the forecast horizons.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot provides a two-dimensional summary of the forecast's
   drift. The **bar height (radius)** clearly increases as we move
   clockwise from horizon "H+1" to "H+6". This is a direct visualization
   of growing uncertainty; the model's average prediction interval
   width gets larger as it forecasts further into the future.
   Simultaneously, the **color of the bars** shifts from blue (lower
   values) to red (higher values), showing that the model's central
   prediction (the mean Q50 value) is also trending upwards across the
   horizons.

.. admonition:: See Also
   :class: seealso

   This plot is closely related to
   :func:`~kdiagram.plot.uncertainty.plot_model_drift`. While both
   visualize drift over horizons with polar bars, this function is more
   general-purpose. It can be used to compare any set of distinct
   categories (not just time horizons) and offers more direct control
   over the data columns used for the radius and color calculations.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind analyzing
forecasts over different horizons, please refer back to the main
:ref:`ug_plot_horizon_metrics` section.


.. _gallery_plot_combined_analysis:

---------------------------------------------------
Combined Analysis: Reliability and Horizon Drift
---------------------------------------------------

Evaluating a sophisticated forecasting model often requires more than a
single plot. A comprehensive analysis involves using multiple,
complementary visualizations to diagnose different aspects of performance.
This tutorial showcases a workflow, combining
:func:`~kdiagram.plot.comparison.plot_polar_reliability` and
:func:`~kdiagram.plot.comparison.plot_horizon_metrics` to perform a
two-part evaluation of a weather forecast.

First, let's re-introduce the anatomy of the two plots we will be using
in our combined analysis.

.. admonition:: Plot Anatomy (Polar Reliability)
   :class: anatomy

   * **Angle (θ):** Represents the **mean predicted probability** of an
     event (e.g., rain), sweeping from 0.0 to 1.0.
   * **Radius (r):** Represents the **observed frequency** of that event.
   * **Reference:** The dashed black spiral is the line of perfect
     calibration. A good model's curve should follow this spiral.

.. admonition:: Plot Anatomy (Horizon Metrics)
   :class: anatomy

   * **Angle (θ):** Represents distinct **forecast horizons** (e.g.,
     "H+6", "H+12").
   * **Radius (r):** The height of each bar represents the **average
     prediction interval width** (uncertainty).
   * **Color:** The color of each bar represents the **average median
     (Q50) prediction** (e.g., the expected amount of rain).

Now, let's apply these two diagnostics to a challenging, real-world
forecasting problem.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case: A Holistic Evaluation of a Weather Forecast Model**

A meteorological agency has a new weather model that produces two key
outputs for a 24-hour period:

1. The **probability** that it will rain at all (a binary event).
2. A **probabilistic forecast** of the total rainfall amount (in mm).

To validate this new model, we need to answer two critical questions:

- **Is the model reliable?** When it predicts a 70% chance of rain, is it trustworthy?
- **How does its uncertainty grow over time?** Is the forecast for rainfall
  amount sharp and useful for the next 6 hours, but too uncertain for 
  the full 24-hour period?

We will perform a combined analysis by creating a side-by-side plot to
answer both questions at once.

.. admonition:: Practical Example

   .. code-block:: python
      :linenos:

      import kdiagram as kd
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt

      # --- 1. Data Generation ---
      np.random.seed(1)
      n_days = 1000

      # --- Part A: Data for Reliability Plot (Probability of Rain) ---
      # True events: It rains on 40% of days
      y_true_rain_event = (np.random.rand(n_days) < 0.4).astype(int)
      # A slightly over-confident model for predicting the event
      y_pred_rain_prob = np.clip(0.4 + np.random.normal(0, 0.3, n_days), 0, 1)

      # --- Part B: Data for Horizon Metrics Plot (Amount of Rain) ---
      horizons = ["H+6", "H+12", "H+18", "H+24"]
      df_horizons = pd.DataFrame(index=horizons)
      # For each horizon, we have multiple samples (e.g., from different days)
      n_samples = 50
      q10_cols, q90_cols, q50_cols = [], [], []

      for i in range(len(horizons)):
          # Both median rainfall and uncertainty increase with the horizon
          median = 5 + 5 * i
          width = 3 + 4 * i
          # Create two samples for each horizon
          df_horizons.loc[f"H+{6*(i+1)}", 'q10_s1'] = median - width/2 + np.random.randn()
          df_horizons.loc[f"H+{6*(i+1)}", 'q90_s1'] = median + width/2 + np.random.randn()
          df_horizons.loc[f"H+{6*(i+1)}", 'q50_s1'] = median + np.random.randn()
          df_horizons.loc[f"H+{6*(i+1)}", 'q10_s2'] = median - width/2 + np.random.randn()
          df_horizons.loc[f"H+{6*(i+1)}", 'q90_s2'] = median + width/2 + np.random.randn()
          df_horizons.loc[f"H+{6*(i+1)}", 'q50_s2'] = median + np.random.randn()

      # --- 2. Create a figure with two polar subplots ---
      fig = plt.figure(figsize=(18, 9))
      ax1 = fig.add_subplot(1, 2, 1, projection='polar')
      ax2 = fig.add_subplot(1, 2, 2, projection='polar')

      # --- 3. Plot each diagnostic on its dedicated axis ---
      kd.plot_polar_reliability(
          y_true_rain_event, y_pred_rain_prob,
          ax=ax1,
          names=["Forecast Model"],
          title='Part A: Is the Rain Probability Forecast Reliable?'
      )
      kd.plot_horizon_metrics(
          df=df_horizons,
          ax=ax2,
          qlow_cols=['q10_s1', 'q10_s2'],
          qup_cols=['q90_s1', 'q90_s2'],
          q50_cols=['q50_s1', 'q50_s2'],
          xtick_labels=horizons,
          title='Part B: How Does Rainfall Uncertainty Evolve?',
          r_label="Mean Interval Width (mm)",
          cbar_label="Mean Predicted Rainfall (mm)"
      )

      fig.suptitle('Combined Analysis of a Weather Forecast Model', fontsize=18)
      fig.tight_layout(rect=[0, 0.03, 1, 0.95])
      fig.savefig("gallery/images/gallery_comparison_combined.png")
      plt.close(fig)

.. figure:: ../images/comparison/gallery_comparison_combined.png
   :align: center
   :width: 90%
   :alt: Side-by-side plots showing reliability and horizon metrics.

   A two-panel figure providing a complete model evaluation. The left
   plot diagnoses the calibration of the rain probability forecast, while
   the right plot shows how the uncertainty of the rainfall amount
   forecast grows over time.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This combined view provides a comprehensive performance summary that
   would be impossible to get from a single plot.

   The **Reliability Spiral** on the left diagnoses the model's ability
   to predict *if* it will rain. The model's curve falls slightly
   inside the dashed reference spiral, particularly for higher
   probabilities. This indicates the model is slightly **over-confident**:
   when it predicts a high probability of rain, the actual frequency is
   a bit lower.

   The **Horizon Metrics** plot on the right shows a clear drift in the
   forecast for rainfall *amount*. The height of the bars (mean interval
   width) increases steadily from the 6-hour to the 24-hour forecast,
   indicating that the model's uncertainty grows significantly over
   longer lead times. The color also shifts from blue to red, showing
   that the median predicted rainfall amount also increases.

   **Overall Conclusion:** By combining these two plots, we can conclude
   that while the model is slightly over-confident in predicting *if* it
   will rain, its primary weakness is a rapid degradation in the
   *precision* of its forecast for *how much* it will rain at longer
   lead times. This is a critical insight for anyone using this model
   for operational planning.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind these
evaluation techniques, please refer back to the main
:ref:`userguide_comparison` and :ref:`userguide_probabilistic` sections.