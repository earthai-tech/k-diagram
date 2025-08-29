.. _cli_probabilistic:

=========================
Probabilistic Diagnostics
=========================

While point forecasts are useful, probabilistic (or quantile)
forecasts provide a much richer picture of uncertainty. The commands
on this page are designed to diagnose the quality of these forecasts,
helping you assess their calibration (reliability) and sharpness
(precision) using specialized polar plots.

.. list-table:: Available Commands
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Description
   * - :ref:`plot-pit-histogram <cli_plot_pit_histogram>`
     - Assesses forecast calibration using a Polar PIT Histogram.
   * - :ref:`plot-polar-sharpness <cli_plot_polar_sharpness>`
     - Compares the sharpness (precision) of multiple models.
   * - :ref:`plot-crps-comparison <cli_plot_crps_comparison>`
     - Provides an overall performance score using the CRPS.
   * - :ref:`plot-credibility-bands <cli_plot_credibility_bands>`
     - Visualizes how forecast uncertainty changes with a feature.
   * - :ref:`plot-calibration-sharpness <cli_plot_calibration_sharpness>`
     - Shows the trade-off between calibration and sharpness.

-------------------
Common Conventions
-------------------

Evaluating probabilistic forecasts requires specifying multiple columns
for each model. These commands use a flexible system to handle this.

- **Ground Truth**: Provide the true value with ``--y-true`` or
  ``--true-col``.
- **Quantile Levels**: You must specify the quantile levels
  corresponding to your prediction columns with ``--q-levels`` or
  ``--quantiles`` (e.g., ``--q-levels 0.1,0.5,0.9``).
- **Prediction Columns**: Provide the quantile forecast columns for
  each model using one of these styles:
  - ``--model NAME:q10,q50,q90`` (repeatable for multiple models)
  - ``--pred q10 q50 q90`` (repeatable for multiple models)
  - ``--pred-cols q10,q50,q90`` (repeatable for multiple models)
- **Saving**: To save a plot, add the ``--savefig out.png`` flag.


.. _cli_plot_pit_histogram:

--------------------
plot-pit-histogram
--------------------

The Probability Integral Transform (PIT) histogram is a key tool for
assessing calibration. It checks whether the observed outcomes are
consistent with the predicted distributions. For a perfectly
calibrated forecast, the PIT histogram should be flat (uniform).
Deviations from a flat shape indicate systematic biases, such as the
forecasts being too narrow (U-shaped) or too wide (hump-shaped) 
:footcite:p:`Gneiting2007b`.

The command's general structure is:

.. code-block:: bash

   k-diagram plot-pit-histogram INPUT
     --y-true Y_TRUE
     --q-levels 0.1,0.5,0.9
     --pred q10 q50 q90
     [--n-bins 10]
     [--show-uniform-line]

Here is an example using space-separated prediction columns:

.. code-block:: bash

   k-diagram plot-pit-histogram demo.csv \
     --true-col actual \
     --pred q10 q50 q90 \
     --q-levels 0.1,0.5,0.9 \
     --title "PIT Calibration Histogram" \
     --savefig pit.png

.. _cli_plot_polar_sharpness:

----------------------
plot-polar-sharpness
----------------------

A sharp forecast is a confident one, meaning its prediction intervals
are narrow. While sharpness is desirable, it must be balanced with
calibration. This command isolates the sharpness component by plotting
the average interval width for each model as a point in polar space.
The radius corresponds to the width, so models closer to the center
are sharper (more precise) :footcite:p:`Gneiting2007b`.

The usage is very similar to the CRPS comparison:

.. code-block:: bash

   k-diagram plot-polar-sharpness INPUT
     --q-levels 0.1,0.5,0.9
     --model M1:q10a,q50a,q90a
     --model M2:q10b,q50b,q90b

This example compares the sharpness of two models, A and B:

.. code-block:: bash

   k-diagram plot-polar-sharpness demo.csv \
     --model A:q10_a,q50_a,q90_a \
     --model B:q10_b,q50_b,q90_b \
     --q-levels 0.1,0.5,0.9 \
     --savefig sharpness_comparison.png


.. _cli_plot_crps_comparison:

----------------------
plot-crps-comparison
----------------------

The Continuous Ranked Probability Score (CRPS) is a popular metric
that summarizes the overall skill of a probabilistic forecast into a
single number (where lower is better) :footcite:p:`scikit-learn`. This command 
calculates the average CRPS for one or more models and plots them as points in polar
space, where the radius is equal to the CRPS. It's a great way to get
a quick, quantitative comparison of competing models :footcite:p:`Gneiting2007b, Jolliffe2012`.

To use it, provide the quantile forecasts for each model you want to
compare:

.. code-block:: bash

   k-diagram plot-crps-comparison INPUT
     --y-true Y_TRUE
     --q-levels 0.1,0.5,0.9
     --model M1:q10a,q50a,q90a
     --model M2:q10b,q50b,q90b

Here's a practical example comparing two models, "M1" and "M2":

.. code-block:: bash

   k-diagram plot-crps-comparison demo.csv \
     --true-col actual \
     --pred q10_m1 q50_m1 q90_m1 \
     --pred q10_m2 q50_m2 q90_m2 \
     --names "Model 1" "Model 2" \
     --q-levels 0.1,0.5,0.9 \
     --savefig crps_comparison.png

.. _cli_plot_credibility_bands:

------------------------
plot-credibility-bands
------------------------

This command helps you visualize how a model's uncertainty changes in
response to a cyclical driver, like seasonality or time of day. It
plots the mean median forecast as a line and shades the area between
the mean lower and upper quantiles, creating a "credibility band" that
can reveal conditional patterns in the forecast's uncertainty 
:footcite:p:`Gneiting2007b, kouadiob2025`.

To generate the plot, you provide three quantile columns and the
cyclic feature:

.. code-block:: bash

   k-diagram plot-credibility-bands INPUT
     --q-cols LOW_Q MED_Q HIGH_Q
     --theta-col CYCLIC_FEATURE
     [--theta-period 12]
     [--theta-bins 12]

Here's an example showing seasonal forecast credibility, binned by month:

.. code-block:: bash

   k-diagram plot-credibility-bands demo.csv \
     --q-cols q10 q50 q90 \
     --theta-col month \
     --theta-period 12 \
     --theta-bins 12 \
     --title "Seasonal Forecast Credibility" \
     --savefig credibility_bands.png


.. _cli_plot_calibration_sharpness:

----------------------------
plot-calibration-sharpness
----------------------------

This plot visualizes the fundamental trade-off between calibration and
sharpness. It places each model on a quarter-circle where the **angle
(θ)** represents the calibration error (0° is perfect) and the
**radius (r)** represents the sharpness (lower is sharper). The ideal
model would be located at the bottom-left corner (low radius, near-zero
angle).

The command synopsis is as follows:

.. code-block:: bash

   k-diagram plot-calibration-sharpness INPUT
     --y-true Y_TRUE
     --q-levels 0.1,0.5,0.9
     --model M1:q10a,q50a,q90a
     --model M2:q10b,q50b,q90b

Let's compare a "Good" model with a "Wide" (but possibly well-calibrated) model:

.. code-block:: bash

   k-diagram plot-calibration-sharpness demo.csv \
     --true-col actual \
     --model Good:q10_good,q50_good,q90_good \
     --model Wide:q10_wide,q50_wide,q90_wide \
     --q-levels 0.1,0.5,0.9 \
     --savefig calibration_vs_sharpness.png


-------------------------
Troubleshooting & Tips
-------------------------

- **Column Count Mismatch?** Ensure that for every model you provide,
  the number of prediction columns exactly matches the number of
  levels in your ``--q-levels`` flag.
- **Understanding the Plots**: A perfectly calibrated model has a
  flat PIT histogram. A skillful model has a low CRPS. A sharp model
  has narrow intervals. A good model balances both calibration and
  sharpness.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options.
- **See Also**: These plots provide a deep dive into probabilistic
  skill. For other ways to look at uncertainty, see the commands in
  the :doc:`uncertainty` guide.
  

.. raw:: html

    <hr>
    
.. rubric:: References

.. footbibliography::