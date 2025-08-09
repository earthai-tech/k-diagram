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

--------------------------------
Multi-Metric Model Comparison
--------------------------------

Uses :func:`~kdiagram.plot.comparison.plot_model_comparison` to generate
a radar chart comparing multiple models across several performance
metrics (R2, MAE, RMSE, MAPE by default for regression) and includes
training time as an additional axis. Scores are normalized for visual
comparison.

.. code-block:: python
   :linenos:

   import kdiagram.plot.comparison as kdc
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   rng = np.random.default_rng(42)
   n_samples = 100
   y_true_reg = np.random.rand(n_samples) * 20 + 5 # True values
   # Model 1: Good fit
   y_pred_r1 = y_true_reg + np.random.normal(0, 2, n_samples)
   # Model 2: Slight bias, more noise
   y_pred_r2 = y_true_reg * 0.9 + 3 + np.random.normal(0, 3, n_samples)
   # Model 3: Less correlated
   y_pred_r3 = np.random.rand(n_samples) * 25 + rng.normal(0, 4, n_samples)

   times = [0.2, 0.8, 0.5] # Example training times
   names = ['Ridge', 'Lasso', 'Tree'] # Example model names

   # --- Plotting ---
   ax = kdc.plot_model_comparison(
       y_true_reg,
       y_pred_r1,
       y_pred_r2,
       y_pred_r3,
       train_times=times,
       names=names,
       # metrics=['r2', 'mae'] # Optionally specify metrics
       title="Gallery: Multi-Metric Model Comparison (Regression)",
       scale='norm', # Normalize scores to [0, 1] (higher is better)
       # Save the plot (adjust path relative to this file)
       savefig="images/gallery_model_comparison.png"
   )
   plt.close() # Close plot after saving

.. image:: ../images/gallery_model_comparison.png
   :alt: Example Multi-Metric Model Comparison Radar Chart
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Multi-Metric Model Comparison** plot uses a radar chart to
   provide a holistic view of performance across several metrics for
   multiple models.

   **Analysis and Interpretation:**

   * **Axes:** Each axis represents a performance metric (e.g., R2,
     MAE, RMSE, MAPE, Train Time). Note that error metrics like MAE
     and time are internally inverted during normalization, so a
     **larger radius always indicates better performance** on that
     axis (higher R2, lower MAE, lower time).
   * **Polygons:** Each colored polygon represents a model.
   * **Performance Profile:** The shape and size of a model's
     polygon reveal its strengths and weaknesses. A large, balanced
     polygon generally indicates good overall performance. Comparing
     polygons shows relative performance across all chosen metrics.

   **🔍 Key Insights from this Example:**

   * We can directly compare 'Ridge', 'Lasso', and 'Tree' models.
   * Look at the 'r2' axis: the model whose polygon extends furthest
     has the highest R-squared value.
   * Look at the 'mae' axis: the model whose polygon extends furthest
     here had the *lowest* MAE (since lower error is better and was
     inverted during scaling).
   * Look at the 'Train Time (s)' axis: the model extending furthest
     was the *fastest* to train.
   * By examining the overall shape, we can identify trade-offs (e.g.,
     one model might have the best R2 but be the slowest).

   **💡 When to Use:**

   * **Model Selection:** When choosing between models based on multiple,
     potentially conflicting, performance criteria.
   * **Performance Summary:** To create a concise visual summary of
     comparative model performance for reports or presentations.
   * **Identifying Trade-offs:** Clearly visualize if improving one
     metric comes at the cost of another (e.g., accuracy vs. speed).
     
     
.. _gallery_plot_reliability:

-----------------------------------------
Model Reliability (Calibration) Diagram
-----------------------------------------

Uses :func:`~kdiagram.plot.comparison.plot_reliability_diagram` to
compare **predicted probabilities** to **observed frequencies** across
probability bins. A perfectly calibrated model lies on the
:math:`y=x` diagonal.

.. code-block:: python
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   import kdiagram.plot.comparison as kdc

   # --- Binary data generation (slightly miscalibrated vs. tighter) ---
   np.random.seed(0)
   n = 1000
   y = (np.random.rand(n) < 0.4).astype(int)

   # Model 1: wider/noisier probabilities around 0.4
   p1 = 0.4 * np.ones_like(y) + 0.15 * np.random.rand(n)
   # Model 2: tighter probabilities around 0.4
   p2 = 0.4 * np.ones_like(y) + 0.05 * np.random.rand(n)

   # --- Plotting ---
   ax, data = kdc.plot_reliability_diagram(
       y, p1, p2,
       names=["Wide", "Tight"],
       n_bins=12,
       strategy="quantile",      # quantile-based binning
       error_bars="wilson",      # 95% Wilson CIs per bin
       counts_panel="bottom",    # show counts histogram below
       show_ece=True,            # compute & display ECE
       show_brier=True,          # compute & display Brier score
       title="Gallery: Reliability Diagram (Quantile + Wilson CIs)",
       savefig="images/gallery_reliability_diagram.png",
       return_data=True,         # get per-bin stats back
   )
   plt.close()

.. image:: ../images/gallery_reliability_diagram.png
   :alt: Example Reliability (Calibration) Diagram
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   The **Reliability Diagram** assesses probability calibration by
   plotting per-bin **observed event frequency** against the **mean
   predicted probability**. The closer the curve is to the diagonal
   line :math:`y=x`, the better the calibration.

   **How to read it:**
   
   * **Diagonal (reference):** Perfect calibration. Points above the
     diagonal indicate the model is *under-confident* (events happen
     more often than predicted); points below indicate *over-confidence*.
   * **Markers & line:** Each marker is a bin; its x-position is the
     average predicted probability in that bin, its y-position is the
     observed fraction of positives in that bin.
   * **Error bars:** Per-bin uncertainty (e.g., Wilson intervals) for
     the observed frequency.
   * **Counts panel:** Shows how many samples fall into each bin (or the
     fraction, if normalized), helping diagnose data sparsity.

   **Reported metrics (optional):**
   
   * **ECE (Expected Calibration Error):**
   
     :math:`\mathrm{ECE} = \sum_{i} w_i \, \lvert \mathrm{acc}_i -
     \mathrm{conf}_i \rvert`, where :math:`\mathrm{acc}_i` is the
     observed frequency, :math:`\mathrm{conf}_i` is the mean predicted
     probability in bin :math:`i`, and :math:`w_i` is the bin weight
     (optionally sample-weighted).
     
   * **Brier score:** Mean squared error on probabilities
     :math:`\frac{1}{N}\sum_{j=1}^{N}(p_j - y_j)^2` (optionally
     weighted). Lower is better.

   **🔍 Key insights from this example:**
   
   * *Wide* shows larger deviations from the diagonal—more miscalibration.
   * *Tight* hugs the diagonal more closely—better calibrated.
   * The counts panel reveals how well the binning covers the probability
     range and whether any bins are data-sparse (wider error bars).

   **💡 When to use:**
   
   * **Probability calibration checks** for binary classifiers producing
     scores/probabilities.
   * **Comparing multiple models** (or post-calibration methods like
     Platt scaling/Isotonic regression) on the same dataset.
   * **Communicating** both calibration quality (curve vs. diagonal) and
     **data support** (counts per bin).
