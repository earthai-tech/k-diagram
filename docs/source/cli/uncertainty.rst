.. _cli_uncertainty:

=========================
Uncertainty & Diagnostics
=========================

A good forecast provides more than just a single number; it communicates
a level of uncertainty. The powerful commands on this page are
dedicated to visualizing and diagnosing this uncertainty. They allow
you to inspect the width and consistency of prediction intervals, spot
anomalies, analyze how uncertainty changes over time, and check if your
intervals are providing the coverage you expect.

.. list-table:: Avilable Commands
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Description
   * - :ref:`plot-actual-vs-predicted <cli_plot_actual_vs_predicted>`
     - A polar plot comparing true vs. predicted point values.
   * - :ref:`plot-anomaly-magnitude <cli_plot_anomaly_magnitude>`
     - Visualizes the severity and type of interval failures.
   * - :ref:`plot-coverage <cli_plot_coverage>`
     - Calculates and plots overall interval coverage scores.
   * - :ref:`plot-coverage-diagnostic <cli_plot_coverage_diagnostic>`
     - Diagnoses interval coverage on a point-by-point basis.
   * - :ref:`plot-interval-consistency <cli_plot_interval_consistency>`
     - Shows the variability of interval widths over time.
   * - :ref:`plot-interval-width <cli_plot_interval_width>`
     - Visualizes the magnitude of prediction interval widths.
   * - :ref:`plot-model-drift <cli_plot_model_drift>`
     - Tracks how average uncertainty changes across horizons.
   * - :ref:`plot-temporal-uncertainty <cli_plot_temporal_uncertainty>`
     - Visualizes the evolution of different series together.
   * - :ref:`plot-uncertainty-drift <cli_plot_uncertainty_drift>`
     - Visualizes the evolution of uncertainty patterns over time.
   * - :ref:`plot-velocity <cli_plot_velocity>`
     - Shows the rate of change of median predictions.
   * - :ref:`plot-radial-density-ring <cli_plot_radial_density_ring>`
     - Visualizes a 1D distribution as a polar density ring.
   * - :ref:`plot-polar-heatmap <cli_plot_polar_heatmap>`
     - Creates a 2D density plot on a polar grid.
   * - :ref:`plot-polar-quiver <cli_plot_polar_quiver>`
     - Draws vector arrows to show forecast revisions or flows.

.. _cli_plot_actual_vs_predicted:

--------------------------
plot-actual-vs-predicted
--------------------------

This command provides a straightforward polar comparison of an actual
(true) series and a predicted series. It can render the data as either
a scatter plot or as connected line plots, providing a simple but
effective visual diagnostic. The angle is mapped from the data's
index, creating a spiral that makes temporal patterns easy to spot 
:footcite:p:`Murphy1993What, Jolliffe2012`.

The command's synopsis is:

.. code-block:: bash

   k-diagram plot-actual-vs-predicted INPUT
     --actual-col ACTUAL
     --pred-col PRED
     [--theta-col THETA]
     [--acov {default,half_circle,quarter_circle,eighth_circle}]
     [--line | --no-line]
     [--r-label LABEL]
     [--alpha ALPHA]

Here is an example that compares the true values in column ``y`` with
the predictions in column ``yhat``, drawing them as connected lines to
visualize their trajectories over time:

.. code-block:: bash

   k-diagram plot-actual-vs-predicted data.csv \
     --actual-col y \
     --pred-col yhat \
     --line \
     --title "Actual vs. Predicted Trajectory" \
     --savefig actual_vs_predicted.png
     
.. _cli_plot_anomaly_magnitude:

------------------------
plot-anomaly-magnitude
------------------------

Prediction intervals are designed to "capture" the true value a certain
percentage of the time. This command visualizes the exceptions,
plotting only the anomalies where the actual value fell outside the
prediction interval. The points are colored by the magnitude of the
violation, making it easy to see not just *that* you had outliers, but
*how severe* they were.

The command requires the actual values and the interval bounds:

.. code-block:: bash

   k-diagram plot-anomaly-magnitude INPUT
     --actual-col ACTUAL
     --q-cols LOWER_BOUND,UPPER_BOUND
     [--cmap-under Blues]
     [--cmap-over Reds]

This example visualizes anomalies against an 80% prediction interval:

.. code-block:: bash

   k-diagram plot-anomaly-magnitude data.csv \
     --actual-col y_true \
     --q-cols q10,q90 \
     --cbar \
     --savefig anomaly_magnitude.png


.. _cli_plot_coverage:

---------------
plot-coverage
---------------

Does your 80% prediction interval actually contain the true value 80%
of the time? This command computes and visualizes the aggregated
coverage score for one or more models. It supports several chart
types, including bar and radar plots, to easily compare the empirical
coverage of your models against their nominal targets.

To use it, specify your models and, if they are quantile-based, the
quantile levels:

.. code-block:: bash

   k-diagram plot-coverage INPUT
     --y-true ACTUAL
     --model M1:q10a,q50a,q90a
     --model M2:q10b,q50b,q90b
     --q-levels 0.1,0.5,0.9
     [--kind bar]

This example compares the coverage of two models using a bar chart:

.. code-block:: bash

   k-diagram plot-coverage data.csv \
     --y-true actual \
     --model M1:q10,q50,q90 \
     --model M2:q10_alt,q50_alt,q90_alt \
     --q-levels 0.1,0.5,0.9 \
     --kind bar \
     --savefig coverage_comparison.png

.. _cli_plot_coverage_diagnostic:

--------------------------
plot-coverage-diagnostic
--------------------------

This command provides a point-wise coverage diagnostic. Instead of one
aggregated score, it plots a point for every single forecast, showing
whether the actual value fell inside or outside the prediction
interval. This detailed view can reveal patterns, such as whether
coverage failures are clustered or random.

The command needs the actual values and the interval bounds:

.. code-block:: bash

   k-diagram plot-coverage-diagnostic INPUT
     --actual-col ACTUAL
     --q-cols LOWER_BOUND,UPPER_BOUND
     [--fill-gradient]
     [--as-bars]

This example creates a diagnostic plot with a background gradient to
help guide the eye:

.. code-block:: bash

   k-diagram plot-coverage-diagnostic data.csv \
     --actual-col y_true \
     --q-cols q10,q90 \
     --fill-gradient \
     --savefig coverage_diagnostic.png

.. _cli_plot_interval_consistency:

---------------------------
plot-interval-consistency
---------------------------

This plot assesses the temporal consistency of your forecast's
uncertainty. For a set of forecasts made over time (e.g., for 2023,
2024, and 2025), it calculates how much the interval width varies at
each location or time step. The result is a polar scatter plot where
the radius shows the variation (either standard deviation or CV),
helping you find forecasts with unstable uncertainty estimates.

To use it, provide lists of the lower and upper quantile columns over time:

.. code-block:: bash

   k-diagram plot-interval-consistency INPUT
     --qlow-cols q10_2023,q10_2024,q10_2025
     --qup-cols  q90_2023,q90_2024,q90_2025
     [--use-cv]

Here is an example that uses the coefficient of variation (CV) to measure consistency:

.. code-block:: bash

   k-diagram plot-interval-consistency data.csv \
     --qlow-cols q10_2023,q10_2024,q10_2025 \
     --qup-cols  q90_2023,q90_2024,q90_2025 \
     --use-cv \
     --savefig interval_consistency.png
     
     
.. _cli_plot_interval_width: 

---------------------
plot-interval-width
---------------------

How wide are your prediction intervals? Are they consistent, or do they
vary wildly? This command creates a polar scatter plot to visualize
the width (upper bound - lower bound) of your intervals for every
forecast. This can help you spot outliers or identify regions of high
and low uncertainty :footcite:p:`kouadiob2025`.

The command's general structure is:

.. code-block:: bash

   k-diagram plot-interval-width INPUT
     --q-cols LOWER_QUANTILE,UPPER_QUANTILE
     [--z-col COLOR_VARIABLE]
     [--acov half_circle]

Here is a basic example plotting the 80% interval width:

.. code-block:: bash

   k-diagram plot-interval-width data.csv \
     --q-cols q10,q90 \
     --savefig interval_width.png

You can also color the points by another variable, like the median
forecast, to see if wider intervals correlate with higher predictions:

.. code-block:: bash

   k-diagram plot-interval-width data.parquet \
     --q-cols q10,q90 \
     --z-col q50 \
     --cbar \
     --savefig interval_width_colored.png

.. _cli_plot_model_drift:

------------------
plot-model-drift
------------------

This command creates a polar bar chart to summarize how an uncertainty
metric, like the mean interval width, increases as the forecast
horizon gets longer. Each bar represents a different horizon (e.g.,
1-day ahead, 2-days ahead), making it easy to visualize how quickly
your forecast uncertainty grows.

Provide the quantile columns for each forecast horizon:

.. code-block:: bash

   k-diagram plot-model-drift INPUT
     --q10-cols q10_h1,q10_h2,q10_h3
     --q90-cols q90_h1,q90_h2,q90_h3
     [--horizons 1 2 3]

This example visualizes drift across three forecast horizons:

.. code-block:: bash

   k-diagram plot-model-drift data.csv \
     --q10-cols q10_h1 q10_h2 q10_h3 \
     --q90-cols q90_h1 q90_h2 q90_h3 \
     --horizons "H+1" "H+2" "H+3" \
     --annotate \
     --savefig model_drift.png
     
.. _cli_plot_temporal_uncertainty:

---------------------------
plot-temporal-uncertainty
---------------------------

This is a general-purpose command for plotting one or more time series
(such as a set of quantiles) as a polar scatter plot. It is useful
for visualizing the evolution of different series together. With the
normalize option, you can compare the shapes of series that have very
different scales.

You can specify columns manually or use ``auto`` to detect quantiles:

.. code-block:: bash

   k-diagram plot-temporal-uncertainty INPUT
     --q-cols [colA colB ... | auto]
     [--names "Series A" "Series B" ...]
     [--normalize]

Here, we plot the 10th, 50th, and 90th quantiles, normalizing each to
the range [0, 1] to compare their temporal patterns:

.. code-block:: bash

   k-diagram plot-temporal-uncertainty data.csv \
     --q-cols q10 q50 q90 \
     --normalize \
     --savefig temporal_uncertainty.png

.. _cli_plot_uncertainty_drift:

------------------------
plot-uncertainty-drift
------------------------

This command creates a beautiful "ring plot" that shows how the width
of your prediction intervals changes over multiple time steps or model
versions. Each ring represents a different forecast (e.g., for 2023,
2024), and its radius at each point shows the normalized interval
width. It's a powerful way to see if your model's uncertainty is
growing or shrinking over time :footcite:p:`kouadiob2025, Gneiting2007b`.

You provide lists of lower and upper quantile columns for each time step:

.. code-block:: bash

   k-diagram plot-uncertainty-drift INPUT
     --qlow-cols q10_2023,q10_2024
     --qup-cols  q90_2023,q90_2024

Here's an example showing the drift between two years:

.. code-block:: bash

   k-diagram plot-uncertainty-drift data.csv \
     --qlow-cols q10_2023,q10_2024 \
     --qup-cols  q90_2023,q90_2024 \
     --title "Uncertainty Drift (2023 vs 2024)" \
     --savefig uncertainty_drift.png


.. _cli_plot_velocity:

---------------
plot-velocity
---------------

This command calculates and plots the temporal "velocity" of a time
series. It computes the first difference (value_t - value_t-1)
across a sequence of columns and displays it as a polar scatter plot.
This can help you identify periods of rapid change in your forecasts.

You provide an ordered list of columns representing the series over time:

.. code-block:: bash

   k-diagram plot-velocity INPUT
     --q50-cols col_t1,col_t2,col_t3

Here, we visualize the velocity of the median forecast from 2023 to 2025:

.. code-block:: bash

   k-diagram plot-velocity data.csv \
     --q50-cols q50_2023,q50_2024,q50_2025 \
     --title "Velocity of Median Forecast" \
     --savefig forecast_velocity.png


.. _cli_plot_radial_density_ring: 

----------------------------
plot-radial-density-ring
----------------------------

This command creates a beautiful radial density plot, shaped like a
ring. It can compute the density (similar to a histogram or KDE :footcite:p:`Silverman1986`) 
of your forecast's interval widths, its velocity, or any other target
column you provide. The result is a circular band where the color
intensity shows the density, helping you see the most common values
at a glance.

You must specify the ``kind`` of data and the target columns:

.. code-block:: bash

   k-diagram plot-radial-density-ring INPUT
     --kind [width|velocity|direct]
     --target-cols C1 [C2 ...]

Here, we plot the density of the median forecast (q50):

.. code-block:: bash

   k-diagram plot-radial-density-ring data.csv \
     --kind direct \
     --target-cols q50 \
     --title "Density Ring of Median Forecast" \
     --savefig density_ring.png

.. _cli_plot_polar_heatmap:

--------------------
plot-polar-heatmap
--------------------

This command creates a 2D histogram, or heatmap, on polar axes. It's
a powerful tool for visualizing the density of data points in a polar
coordinate system. By binning the data by both radius and angle, it
can reveal clusters and patterns that are not obvious in other plot
types.

The general usage is as follows:

.. code-block:: bash

   k-diagram plot-polar-heatmap INPUT
     --r-col RADIUS_COLUMN
     --theta-col ANGLE_COLUMN
     [--r-bins 30]
     [--theta-bins 60]
     [--theta-period 360]

Here is an example that creates a heatmap with 30 radial bins and 72
angular bins:

.. code-block:: bash

   k-diagram plot-polar-heatmap data.csv \
     --r-col distance \
     --theta-col bearing_degrees \
     --r-bins 30 \
     --theta-bins 72 \
     --savefig polar_heatmap.png

.. note::
   If your angular data is cyclical (like degrees from 0-360), use
   the ``--theta-period`` flag to ensure it wraps around correctly.

.. _cli_plot_polar_quiver:

-------------------
plot-polar-quiver
-------------------

A quiver plot is used to visualize a vector field. This command plots
arrows on a polar grid, where each arrow's position is given by a
radius and angle, and its direction and magnitude are given by vector
components (u, v). It's a specialized plot, often used in scientific
and engineering fields to show things like wind patterns or fluid
dynamics.

The command requires four main columns for position and vector components:

.. code-block:: bash

   k-diagram plot-polar-quiver INPUT
     --r-col RADIUS
     --theta-col ANGLE
     --u-col U_COMPONENT
     --v-col V_COMPONENT
     [--color-col COLOR_VARIABLE]

Here is a basic example of its use:

.. code-block:: bash

   k-diagram plot-polar-quiver vector_field_data.csv \
     --r-col r \
     --theta-col theta \
     --u-col u \
     --v-col v \
     --savefig polar_quiver.png

---

--------------------------
plot-actual-vs-predicted
--------------------------

This command provides a straightforward polar comparison of an actual
(true) series and a predicted series. It can render the data as either
a scatter plot or as connected line plots, providing a simple but
effective visual diagnostic. The angle is mapped from the data's
index, creating a spiral that makes temporal patterns easy to spot.

The command's synopsis is:

.. code-block:: bash

   k-diagram plot-actual-vs-predicted INPUT
     --actual-col ACTUAL
     --pred-col PREDICTED
     [--line | --no-line]

Here is an example that compares the true values in column ``y`` with
the predictions in column ``yhat``, drawing them as connected lines:

.. code-block:: bash

   k-diagram plot-actual-vs-predicted data.csv \
     --actual-col y \
     --pred-col yhat \
     --line \
     --title "Actual vs. Predicted Trajectory" \
     --savefig actual_vs_predicted.png

-------------------------
Troubleshooting & Tips
-------------------------

- **"Missing columns" error?** This is the most common issue.
  Double-check that the column names in your command exactly match
  the headers in your data file.
- **Column lists**: For commands that take lists of columns (like
  ``--qlow-cols``), ensure you provide the same number of columns in
  each corresponding list.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options and their descriptions.
- **See Also**: The tools on this page provide a comprehensive look
  at uncertainty. They pair well with the tools in
  :doc:`probabilistic` for a complete picture of your probabilistic
  forecast's quality.

.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::