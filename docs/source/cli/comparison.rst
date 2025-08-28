.. _cli_comparison:

==============================
Comparison & Calibration 
==============================

This page covers the command-line tools for model comparison and
calibration diagnostics. These commands help you answer key questions:
*"Are my model's probabilities trustworthy?"* and *"Which of my models
is best, and in what ways?"* Like other tools in this suite, they
read a tabular data file (CSV, Parquet, etc.) and generate insightful
visualizations.

.. list-table:: Available Commands
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Description
   * - :ref:`plot-reliability-diagram <cli_plot_reliability_diagram>`
     - Compares predicted probabilities to observed frequencies.
   * - :ref:`plot-polar-reliability <cli_plot_polar_reliability>`
     - A polar version of the reliability diagram.
   * - :ref:`plot-model-comparison <cli_plot_model_comparison>`
     - A radar chart for multi-metric model comparison.
   * - :ref:`plot-horizon-metrics <cli_plot_horizon_metrics>`
     - Compares metrics across different forecast horizons.


-------------------
Common Conventions
-------------------

The commands on this page share a few common patterns.

- **Input Data**: A path to your data table can be passed as the first
  argument or with the ``--input`` flag. The format is detected from
  the file extension.
- **Specifying Models**: You can provide prediction columns using
  either the named ``--model NAME:COL`` syntax or by repeating the
  ``--pred`` flag and supplying ``--names`` for the legend.
- **Saving Plots**: All commands will save the figure to a file if you
  provide the ``--savefig out.png`` flag; otherwise, they display it
  in an interactive window.


.. _cli_plot_reliability_diagram:

--------------------------
plot-reliability-diagram
--------------------------

A reliability diagram is the go-to tool for assessing if a model's
predicted probabilities are well-calibrated :footcite:p:`Jolliffe2012`. 
It plots the observed frequency of an event against the predicted probability. 
For a perfectly calibrated model, all points should lie on the diagonal
line.

The command offers a rich set of options for customization:

.. code-block:: bash

   k-diagram plot-reliability-diagram INPUT
     --y-true Y
     [--model NAME:col | --pred col[,col...] [--names ...]]...
     [--n-bins 10]
     [--strategy {uniform,quantile}]
     [--positive-label 1] [--class-index N]
     [--error-bars {wilson,normal,none}] [--conf-level 0.95]
     [--show-diagonal/--no-show-diagonal]
     [--show-ece/--no-show-ece] [--show-brier/--no-show-brier]
     [--counts-panel {bottom,none}] [--counts-norm {fraction,count}]
     [--counts-alpha 0.35]
     [--connect/--no-connect] [--legend/--no-legend] [--legend-loc best]
     [--xlim 0,1] [--ylim 0,1]
     

For example, let's compare two models using quantile binning, add
Wilson confidence intervals for the error bars, and show a panel with
the counts in each bin:

.. code-block:: bash

   k-diagram plot-reliability-diagram rel.csv \
     --y-true y \
     --pred p_m1 p_m2 \
     --names "Wide Model" "Tight Model" \
     --strategy quantile \
     --n-bins 12 \
     --error-bars wilson \
     --counts-panel bottom \
     --show-ece --show-brier \
     --savefig reliability.png

A key option is ``--strategy``, which controls the binning method.
Use ``uniform`` for equal-width bins or ``quantile`` to ensure each
bin has a similar number of samples.


.. _cli_plot_polar_reliability: 

--------------------------
plot-polar-reliability
--------------------------

This command presents the same calibration data in a different light,
mapping it onto a polar "spiral." The predicted probability is mapped
to the angle (from 0° to 90°), and the observed frequency is mapped to
the radius. A perfectly calibrated model will trace the dashed spiral
perfectly :footcite:p:`kouadiob2025`. This view makes over- and 
under-confidence immediately apparent.

The usage is simpler than its rectangular counterpart:

.. code-block:: bash

   k-diagram plot-polar-reliability INPUT
     --y-true Y
     [--model NAME:col | --pred col ...]
     [--n-bins 10]
     [--strategy {uniform,quantile}]

Here's an example comparing a calibrated model with one that is
over-confident:

.. code-block:: bash

   k-diagram plot-polar-reliability rel.csv \
     --y-true y \
     --model Calibrated:p_m1 --model Over-confident:p_m2 \
     --n-bins 15 \
     --strategy uniform \
     --cmap coolwarm \
     --savefig polar_reliability.png


.. _cli_plot_model_comparison: 

-----------------------
plot-model-comparison
-----------------------

This command generates a classic radar (or spider) chart, providing a
holistic, multi-metric comparison of several models. It's an excellent
way to visualize the trade-offs between different models across various
performance axes like accuracy, speed, and error metrics.

The command can automatically select metrics or use ones you provide:

.. code-block:: bash

   k-diagram plot-model-comparison INPUT
     --y-true Y
     [--model NAME:col | --pred col]...
     [--metrics auto | MET1 [MET2 ...]]
     [--train-times t1 [t2 ...]]
     [--scale {norm,min-max,std,none}]

Here, we compare two regression models on R², MAE, and RMSE, also
including their training times as a performance axis:

.. code-block:: bash

   k-diagram plot-model-comparison reg.csv \
     --true-col y \
     --model "Linear Model":m1 --model "Tree Model":m2 \
     --metrics r2 mae rmse \
     --train-times 0.1 0.5 \
     --scale norm \
     --title "Regression Model Comparison" \
     --savefig model_comparison.png


.. _cli_plot_horizon_metrics: 

------------------------
plot-horizon-metrics
------------------------

This plot is designed to summarize how a metric changes across different
forecast horizons or categories. It uses a polar bar chart where each
bar's height represents a primary metric (like mean interval width),
and its color can represent an optional secondary metric (like mean
error).

To use it, you provide columns corresponding to each horizon/category:

.. code-block:: bash

   k-diagram plot-horizon-metrics INPUT
     --q-low COL1 [COL2 ...]
     --q-up  COL1 [COL2 ...]
     [--q50  COL1 [COL2 ...]]
     [--xtick-labels L1 [L2 ...]]

In this example, we visualize how the mean prediction interval width
(bar height) and the mean median forecast (color) change across six
forecast horizons:

.. code-block:: bash

   k-diagram plot-horizon-metrics horizons.csv \
     --q-low  q10_s1 q10_s2 \
     --q-up   q90_s1 q90_s2 \
     --q50    q50_s1 q50_s2 \

     --xtick-labels "H+1" "H+2" "H+3" "H+4" "H+5" "H+6" \
     --title "Mean Interval Width Across Horizons" \
     --r-label "Mean (Q90 - Q10)" \
     --cbar-label "Mean Q50" \
     --savefig horizons.png

-------------------------
Troubleshooting & Tips
-------------------------

- **"Missing columns" error?** Double-check that the column names in
  your command exactly match the headers in your data file.
- **Unexpected binning behavior?** In reliability plots, the
  ``quantile`` strategy can fall back to ``uniform`` if there are too
  few unique prediction values. Check your data's distribution.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options and their descriptions.
- **See Also**: After comparing models, you might want to explore the
  best one's error properties using the tools in :doc:`errors` or
  examine its probabilistic forecasts with the tools in
  :doc:`probabilistic`.

If a command's behavior surprises you (e.g., binning fallback or
column selection), re-run with fewer options and verify input
columns. Feel free to file `issues <https://github.com/earthai-tech/k-diagram/issues>`_
with a small CSV illustrating the problem.

.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::