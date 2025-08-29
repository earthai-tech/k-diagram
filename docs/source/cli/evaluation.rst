.. _cli_evaluation:

==================
Evaluation Plots
==================

This page documents the command-line interfaces for a suite of model
evaluation plots. These tools help you move beyond simple error scores
and visualize the performance of classifiers and regressors, often
using polar coordinates to make multi-model comparisons intuitive.

.. list-table:: Available Commands
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Description
   * - :ref:`plot-polar-roc <cli_plot_polar_roc>`
     - Draws a Polar Receiver Operating Characteristic (ROC) curve.
   * - :ref:`plot-polar-pr-curve <cli_plot_polar_pr_curve>`
     - Draws a Polar Precision-Recall (PR) curve for imbalanced data.
   * - :ref:`plot-polar-cm <cli_plot_polar_confusion_matrix>`
     - Visualizes a binary confusion matrix on a polar plot.
   * - :ref:`plot-polar-cm-in <cli_plot_polar_confusion_matrix_in>`
     - Visualizes a multiclass confusion matrix.
   * - :ref:`plot-polar-cr <cli_plot_polar_classification_report>`
     - Displays a per-class report of Precision, Recall, and F1-Score.
   * - :ref:`plot-pinball-loss <cli_plot_pinball_loss>`
     - Visualizes the Pinball Loss for each quantile of a forecast.
   * - :ref:`plot-regression-performance <cli_plot_regression_performance>`
     - Visualizes multiple regression models across several metrics at once.
     
----------------------
Common CLI Conventions
----------------------

Most commands on this page share a common set of options for handling
data and customizing the output.

- **Input/Output**: All commands read a tabular data file (like
  ``data.csv``) and will save the resulting plot if you provide the
  ``--savefig`` flag.
- **Column Selection**:

  - ``--y-true``: **(Required)** The column containing the true,
    observed values.
  - ``--y-pred`` or ``--model NAME:COL``: Use one of these patterns to
    specify the prediction columns you want to evaluate.
  - ``--names``: Provide clean, readable names for the legend.
  
- **Styling**: Flags like ``--title``, ``--figsize``, and ``--cmap``
  give you control over the plot's final appearance.


.. _cli_plot_polar_roc:

-----------------
plot-polar-roc
-----------------

The ROC (Receiver Operating Characteristic) curve is a fundamental
tool for binary classifiers. This command visualizes it on a polar
axis, where the angle represents the False Positive Rate (FPR) and
the radius represents the True Positive Rate (TPR). A perfect model
"reaches" for the top-left corner.

The command is used as follows:

.. code-block:: bash

   k-diagram plot-polar-roc [INPUT]
     --y-true YTRUE
     [--model NAME:COL]... | [--pred COL]...
     [--names NAME1 NAME2 ...]
     [common figure options]

For example, to compare two models, ``m1`` and ``m2``, against the
ground truth in column ``y``:

.. code-block:: bash

   k-diagram plot-polar-roc data.csv \
     --y-true y \
     --pred m1 --pred m2 \
     --names "Logistic Regression" "Random Forest" \
     --title "ROC Curve Comparison (polar)" \
     --savefig polar_roc.png

.. note::
   This plot is a polar version of the output from
   :func:`sklearn.metrics.roc_curve`.


.. _cli_plot_polar_pr_curve:

-------------------------
plot-polar-pr-curve
-------------------------

While the ROC curve is useful, the Precision-Recall (PR) curve can be
more informative on imbalanced datasets. Here, the angle represents
Recall and the radius represents Precision, making it easy to see the
trade-off :footcite:p:`Powers2011`.

Its usage is very similar to the ROC plot:

.. code-block:: bash

   k-diagram plot-polar-pr-curve [INPUT]
     --y-true YTRUE
     [--model NAME:COL]... | [--pred COL]...
     [--names NAME1 NAME2 ...]
     [common figure options]

Here is a practical example:

.. code-block:: bash

   k-diagram plot-polar-pr-curve data.csv \
     --y-true y \
     --pred m1 --pred m2 \
     --names "Good Model" "Baseline" \
     --title "Precision-Recall Comparison (polar)" \
     --savefig polar_pr.png


.. _cli_plot_polar_confusion_matrix:

-----------------------
plot-polar-cm
-----------------------

A confusion matrix provides a direct look at a classifier's mistakes.
This command renders a binary confusion matrix (TP, FP, TN, FN) as a
quartet of polar bars for each model being compared, which makes
visual comparison of error types straightforward.

To generate the plot, you can use this structure:

.. code-block:: bash

   k-diagram plot-polar-cm [INPUT]
     --y-true YTRUE
     [--model NAME:COL]... | [--pred COL]...
     [--names NAME1 NAME2 ...]
     [--normalize / --no-normalize]

Here's an example comparing a "Latest" and "Previous" model, with
normalized results:

.. code-block:: bash

   k-diagram plot-polar-cm data.csv \
     --y-true y \
     --pred m1 --pred m2 \
     --names "Latest" "Prev" \
     --normalize \
     --savefig polar_cm_binary.png

Transitioning from binary to multiclass problems, the next command
offers a more detailed view.


.. _cli_plot_polar_confusion_matrix_in:

-------------------------
plot-polar-cm-in
-------------------------

For multiclass problems, this command displays a confusion matrix as
grouped polar bars. Each angular sector represents a *true class*,
and the colored bars within it show how the model *predicted* samples
from that class. It's a powerful way to spot specific inter-class
confusion :footcite:p:`scikit-learn`.

The command takes a single prediction column and the true column:

.. code-block:: bash

   k-diagram plot-polar-cm-in [INPUT]
     --y-true YTRUE
     --y-pred YPRED
     [--class-labels L1 L2 ...]
     [--normalize / --no-normalize]

Let's visualize the performance on a 4-class problem:

.. code-block:: bash

   k-diagram plot-polar-cm-in data.csv \
     --y-true true_labels \
     --y-pred predicted_labels \
     --class-labels Apple Banana Orange Grape \
     --savefig polar_cm_multi.png

An alias for this command is ``plot-polar-cm-multiclass``.


.. _cli_plot_polar_classification_report:

-------------------
plot-polar-cr
-------------------

This command provides a polar visualization of a classification
report, showing per-class Precision, Recall, and F1-score as grouped
bars. This helps to quickly identify which classes the model struggles
with :footcite:p:`Powers2011`.

The synopsis is straightforward:

.. code-block:: bash

   k-diagram plot-polar-cr [INPUT]
     --y-true YTRUE
     --y-pred YPRED
     [--class-labels L1 L2 ...]

Here's how you'd use it for a 3-class model:

.. code-block:: bash

   k-diagram plot-polar-cr data.csv \
     --y-true yt \
     --y-pred yp \
     --class-labels A B C \
     --title "Per-class Metrics" \
     --savefig polar_cls_report.png

Now, let's shift from classification to quantile regression.


.. _cli_plot_pinball_loss:

---------------------
plot-pinball-loss
---------------------

For quantile regression models, the pinball loss is a key metric. This
command plots the average pinball loss for each predicted quantile on
a polar axis. The angle corresponds to the quantile level, and the
radius shows the loss, making it easy to see if your model is more or
less accurate at different quantiles.

To use it, you provide the true values and the corresponding quantile
forecast columns:

.. code-block:: bash

   k-diagram plot-pinball-loss [INPUT]
     --y-true YTRUE
     --q-cols Q10,Q25,Q50,Q75,Q90
     --quantiles 0.10,0.25,0.50,0.75,0.90

Here is a typical example:

.. code-block:: bash

   k-diagram plot-pinball-loss data.csv \
     --y-true y \
     --q-cols q10,q25,q50,q75,q90 \
     --quantiles 0.10,0.25,0.50,0.75,0.90 \
     --title "Pinball Loss by Quantile" \
     --savefig pinball_loss.png


.. _cli_plot_regression_performance:

-------------------------------
plot-regression-performance
-------------------------------

This command creates a  "radar chart" for comparing multiple
regression models across several metrics at once. It's the an ideal
tool for moving beyond a single score and getting a holistic, visual
understanding of your models' trade-offs :footcite:p:`kouadiob2025`.

The command is flexible, but don't let that intimidate you! It's
built around two simple modes for providing your data and offers
several powerful strategies for viewing the results.

The command's general structure showcases its dual-mode capability
and new normalization controls:

.. code-block:: text

   k-diagram plot-regression-performance [INPUT]

     # --- Data Mode (provide one of these) ---
     --y-true TRUE_COL --pred PRED_1 PRED_2 ...
     --y-true TRUE_COL --model "M1:p1" --model "M2:p2" ...

     # --- OR Values Mode ---
     --metric-values "r2:0.8,0.7" "mae:2.1,2.5" ...
     
     # --- Normalization Control ---
     --norm [per_metric|global|none]
     [--global-bounds "metric:min,max" ...]

     # --- Labels & Styling ---
     [--names "Model 1" "Model 2" ...]
     [--metric-label "neg_mean_absolute_error:MAE"]
     [--title "My Plot Title"]
     [--savefig my_plot.png]

**Putting It Into Practice: Two Common Scenarios**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**1. The Standard Workflow (Data-mode)**
The most common way to use this command is by pointing it at your data
and letting it do the hard work of calculating the metrics for you.
Let's say you want to compare a Linear Regression model against a
Gradient Boosting model:

.. code-block:: bash

   k-diagram plot-regression-performance data.csv \
     --y-true y \
     --pred m1 m2 \
     --names "Linear Regression" "Gradient Boosting" \
     --metrics r2 neg_mean_absolute_error \
     --metric-label "neg_mean_absolute_error:MAE" \
     --title "Overall Model Performance" \
     --savefig reg_perf_data.png

**2. Using Pre-Computed Scores (Values-mode)**
What if you already have your performance scores, maybe from a report
or a different experiment? No problem. You can feed the scores
directly to the command without needing the original dataset.

Here's how you'd plot the same kind of chart using pre-computed values:

.. code-block:: bash

   k-diagram plot-regression-performance \
     --metric-values "r2:0.82,0.74" "mae:-3.2,-3.6" \
     --names "Model A" "Model B" \
     --metric-label "mae:MAE" \
     --title "Performance from Pre-computed Scores" \
     --savefig reg_perf_values.png

As you can see, getting a plot is easy. But to truly unlock its power,
it helps to understand how the data is scaled.

**Choosing Your Perspective: Normalization Strategies**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make different metrics (like R², which is near 1, and RMSE, which
can be large) comparable on the same plot, their scores are normalized
to determine the length of the bars. The ``--norm`` flag gives you
precise control over this process.

* **``--norm per_metric`` (Default): The Relative Comparison**
  **Use this when you want to know:** *"Which of my models is
  relatively better or worse on each metric?"*
  This strategy scales each metric independently. The best model for
  a given metric gets a bar of length 1, and the worst gets a bar of
  length 0. It's perfect for quickly spotting the relative strengths
  of each model.

* **``--norm global``: The Absolute Benchmark**
  **Use this when you want to know:** *"Do my models meet a
  predefined standard of 'good'?"*
  This strategy scales each metric against fixed, absolute bounds
  that you provide. It's ideal for comparing models against a
  consistent benchmark. For example, let's judge R² on a fixed scale
  of 0 to 1, and MAE on a scale of -10 (bad) to 0 (perfect).

  .. code-block:: bash

     k-diagram plot-regression-performance data.csv \
       --y-true y --pred m1 m2 --names M1 M2 \
       --norm global \
       --global-bounds "r2:0,1" "neg_mean_absolute_error:-10,0" \
       --savefig reg_perf_global.png

* **``--norm none``: The Expert's View**
  **Use this when you need to know:** *"What are the exact, raw score
  values?"*
  This plots the raw metric scores directly. Be careful with this
  option, as metrics with very different scales can make the plot
  difficult to interpret visually, but it's useful for seeing the
  un-scaled numbers.
  
-------------------------
Troubleshooting & Tips
-------------------------

- **"Missing columns" error?** Double-check that the column names in
  your command exactly match the headers in your data file.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options.
- **See Also**: The commands on this page are often used together.
  For example, after finding a good model with
  :ref:`cli_plot_regression_performance`, you might use the plots from
  the :doc:`context/` page to inspect its errors more closely.
  
  
.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::