.. _lab_quickstart:

=============
Quick Start
=============

.. topic:: Our Goal: Comparing Two Models with a Radar Chart
   :class: hint

   Imagine you've developed a new "Upgraded" model and you need to
   prove it's better than the old "Baseline". A single accuracy score
   might not tell the whole story. How can you show the trade-offs
   across multiple performance metrics quickly and visually?

   This guide provides a fast, hands-on example to do just that using
   ``k-diagram``'s polar radar chart.

Welcome! This is a complete, copy-pasteable example to
take you from synthetic data generation to a final, publication-quality
plot in just a few minutes. We'll show you how to do it with both the
**Python API** and the **Command-Line Interface (CLI)**.

---------------------------
Part 1: The Python API
---------------------------

The most flexible way to use ``k-diagram`` is within a Python script
or notebook. The code below will generate a complete, illustrative
example.

**Code**
^^^^^^^^
.. code-block:: python
   :linenos:

   import kdiagram as kd
   import matplotlib.pyplot as plt

   # 1. Define distinct profiles for the models we want to compare.
   # This ensures they have different performance characteristics, which
   # will make our comparison plot much more interesting.
   model_profiles = {
       "Baseline": {"bias": -8.0, "noise_std": 3.0},
       "Upgraded": {"bias": 0.5, "noise_std": 4.0},
   }

   # 2. Generate the dataset using these specific profiles.
   print("Generating synthetic data for two models...")
   data = kd.datasets.make_regression_data(
       model_profiles=model_profiles,
       seed=42,
       as_frame=True  # We ask for a pandas DataFrame as output
   )

   # 3. Save the data so we can use it with the CLI in Part 2
   data.to_csv("quickstart_data.csv", index=False)
   print("Sample data created and saved to quickstart_data.csv")

   # 4. Create the comparison plot and display it.
   print("\nGenerating comparison plot from Python API...")
   ax = kd.plot_regression_performance(
       # Pass the true values and each prediction series as NumPy arrays
       data['y_true'].values,
       data['pred_Baseline'].values,
       data['pred_Upgraded'].values,
       
       names=["Baseline", "Upgraded"],
       title="Model Comparison: Baseline vs. Upgraded",
       # Use metric_labels for a cleaner plot
       metric_labels={
           'r2': 'R²',
           'neg_mean_absolute_error': 'MAE',
           'neg_root_mean_squared_error': 'RMSE',
       }, 
   )
   
   plt.show()

**Expected Output**
^^^^^^^^^^^^^^^^^^^
Running this script will print a confirmation message and then display
a polar radar chart similar to this one, clearly showing the
performance difference between the two models.

.. image:: /images/quickstart_radar_chart.png
   :alt: Example Regression Performance Radar Chart
   :align: center
   :width: 80%

------------------------------------------
Part 2: The Command-Line (CLI) Alternative
------------------------------------------

Prefer the command line for quick tasks? You can create the exact same
plot without writing any Python. Since we already saved our data to
``quickstart_data.csv``, just run this command in your terminal:

.. code-block:: bash

   k-diagram plot-regression-performance quickstart_data.csv \
     --y-true y_true \
     --pred pred_Baseline pred_Upgraded \
     --names "Baseline" "Upgraded" \
     --title "Model Comparison (from CLI)" \
     --metric-label "r2:R²" "neg_mean_absolute_error:MAE" \
       "neg_root_mean_squared_error:RMSE" \
     --savefig quickstart_cli_plot.png

This will save the plot directly to a file named
``quickstart_cli_plot.png``.

-------------------------
Interpreting the Plot
-------------------------

This radar chart provides a rich, holistic view of model performance.
Here’s how to read it.

.. topic:: How to Read the Chart

   * **Axes**: Each axis represents a different performance metric (like
     R², MAE, RMSE). To make all metrics comparable, scores are always
     normalized so that **outward is always better**.
   * **Radius**: The length of a bar shows the model's normalized score
     on that metric. A bar reaching the outer green circle represents
     the best possible performance among the models being compared.
   * **Shape**: Each colored shape represents a model. A model with a
     **larger overall area** is a better all-around performer.

**Putting It All Together: Our Analysis**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The plot tells a very clear story: the **"Upgraded" model is a significant 
improvement** over the "Baseline".

* The **Upgraded model** (green) has long bars on all three axes, forming 
  a large, balanced shape. This indicates it is a strong all-around performer 
  with a high R² (good fit) and low MAE and RMSE (low errors).

* The **Baseline model** (blue) reveals a critical flaw. While its MAE bar 
  is not zero, its bars for **R² and RMSE are extremely short**. This is a 
  classic sign of a model with a **severe, systematic bias**. The large, 
  consistent errors are heavily penalized by the R² and RMSE metrics, 
  indicating the model has poor predictive power despite its seemingly 
  acceptable average error.

In conclusion, the chart instantly shows us that the "Upgraded" model 
successfully fixed the critical bias issue present in the "Baseline", 
making it the clear winner.

This is just one example of what this chart can reveal. To see more
advanced use cases, including how to add custom metrics or control the
normalization, check out the detailed examples in our gallery.

*See more examples in* :ref:`gallery_plot_regression_performance`

-----------------------------------
From Synthetic to Real-World Data
-----------------------------------

This quick start guide uses a synthetic dataset to demonstrate the plotting
workflow in a simple and reproducible way.

However, the true power of ``k-diagram`` shines when analyzing the
complex uncertainty of real-world forecasts. To see these same diagnostic
charts applied to an environmental forecasting challenge, we highly
recommend exploring our detailed case study on land subsidence.

*See the full analysis in* :ref:`case_history_zhongshan`

-------------
Next Steps
-------------

Congratulations! You've created your first k-diagram plot and seen
how easy it is to compare models.

* Explore more plot types and their capabilities in the
  :doc:`Plot Gallery <gallery/index>`.
* Learn about the concepts behind the visualizations in the
  :doc:`User Guide <user_guide/index>`.
* Refer to the :doc:`API Reference <api>` for detailed function
  signatures and parameters.