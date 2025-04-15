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