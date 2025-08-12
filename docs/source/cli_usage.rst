.. _cli_usage:

=============================
Command-Line Interface (CLI)
=============================

For users who prefer working in the terminal or want to integrate
plotting into scripts without writing Python code, `k-diagram`
provides a command-line interface (CLI) tool named ``k-diagram``.

This tool allows you to generate many of the diagnostic plots directly
by providing data through CSV files and specifying options via command-
line arguments.

Getting Help
--------------

You can get help on the available commands and general options using
the ``--help`` flag:

.. code-block:: bash

   k-diagram --help

This will output a summary of the main command and list all available
sub-commands (plot types):

.. code-block:: text
   :emphasize-lines: 1, 5

   usage: k-diagram [-h] [--version] <command> ...

   K-Diagram: CLI for Forecasting Uncertainty Visualization.

   Available commands:
     <command>
       plot_coverage         Plot coverage diagnostic (requires NumPy arrays from CSVs).
       plot_model_drift      Plot model drift over time (requires DataFrame from CSV).
       plot_velocity         Plot velocity diagnostic (requires DataFrame from CSV).
       plot_interval_consistency
                             Plot prediction interval consistency (requires DataFrame from CSV).
       plot_anomaly_magnitude
                             Plot anomaly magnitude polar plot (requires DataFrame from CSV).
       plot_uncertainty_drift
                             Plot uncertainty drift polar plot (requires DataFrame from CSV).
       

   options:
     -h, --help     show this help message and exit
     --version      show program's version number and exit

   Use '<command> --help' for specific command options.
   ...

To get help specific to a particular plot command, use ``--help`` after
the command name. For example:

.. code-block:: bash

   k-diagram plot_anomaly_magnitude --help

This will show all the arguments and options accepted by the
``plot_anomaly_magnitude`` command, such as required input files,
column specifications, and plotting adjustments:

.. code-block:: text
   :emphasize-lines: 1, 5, 6, 7

   usage: k-diagram plot_anomaly_magnitude [-h] --actual_col ACTUAL_COL
                                           --q_cols LOWER_Q_COL UPPER_Q_COL
                                           [--theta_col THETA_COL]
                                           [--acov {default,half_circle,quarter_circle,eighth_circle}]
                                           [--title TITLE] [--figsize FIGSIZE]
                                           [--cmap_under CMAP_UNDER] [--cmap_over CMAP_OVER]
                                           [--s S] [--alpha ALPHA] [--show_grid | --no_show_grid]
                                           [--verbose VERBOSE] [--cbar] [--mask_angle]
                                           [--savefig FILEPATH]
                                           filepath

   Plot anomaly magnitude polar plot (requires DataFrame from CSV).

   positional arguments:
     filepath              Path to the CSV file containing the data.

   options:
     -h, --help            show this help message and exit
     --actual_col ACTUAL_COL
                           The column containing actual values. (required)
     --q_cols LOWER_Q_COL UPPER_Q_COL
                           Two column names: the lower and upper quantiles (e.g., Q10 Q90). (required)
     --theta_col THETA_COL
                           Optional column name for angular position.
     # ... (other options listed) ...
     --savefig FILEPATH    Save the plot to the specified file path instead of showing it.

Checking Version
------------------

You can check your installed version of `k-diagram` using:

.. code-block:: bash

    k-diagram --version

General Usage Pattern
-----------------------

The basic structure of a `k-diagram` CLI command is:

.. code-block:: bash

   k-diagram <plot_command> [input_files...] [options...]

* **`<plot_command>`:** The name of the plot you want to generate
  (e.g., `plot_anomaly_magnitude`, `plot_coverage`).
* **`[input_files...]`:** One or more positional arguments specifying
  the path(s) to your input CSV data file(s). The number and meaning
  of these files depend on the specific command (check its `--help`).
* **`[options...]`:** Flags and arguments starting with `--` or `-`
  used to specify necessary information (like column names) or customize
  the plot appearance. Common options include:

  * `--actual_col`, `--q_cols`, `--q10_cols`, etc.: Specify which
    columns in your CSV contain the relevant data. **These are often
    required.**
  * `--title`: Set a custom plot title.
  * `--savefig`: Save the output plot to a file instead of displaying it.
  * Options specific to the plot type (e.g., `--acov`, `--cmap`, `--normalize`).

Command Examples
------------------

Here are a few examples demonstrating common use cases:

Anomaly Magnitude Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate an anomaly magnitude plot from `results.csv`, using the
`'observed'` column for actual values and `'p10'`, `'p90'` for the
prediction interval bounds.

.. code-block:: bash

   k-diagram plot_anomaly_magnitude results.csv \
       --actual_col observed \
       --q_cols p10 p90 \
       --title "Anomaly Magnitude for Experiment X" \
       --cbar \
       --savefig anomaly_plot.pdf

* `results.csv`: The input data file (positional argument).
* `--actual_col observed`: Specifies the column with true values.
* `--q_cols p10 p90`: Specifies the lower and upper quantile columns.
* `--title ...`: Sets the plot title.
* `--cbar`: Adds a color bar.
* `--savefig anomaly_plot.pdf`: Saves the output as a PDF file.

Anomaly Magnitude Example (Reiteration/Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*(This provides another example for plot_anomaly_magnitude, perhaps
focusing on different options than the one already present in
cli_usage.rst, or you can replace the existing one if preferred)*

Generate an anomaly magnitude plot from `forecast_results.csv`. Uses
the `actual_flow` column and the interval `flow_q10` - `flow_q90`.
Includes a color bar and saves the plot as a PDF.

.. code-block:: bash

   k-diagram plot_anomaly_magnitude forecast_results.csv \
       --actual-col actual_flow \
       --q-cols flow_q10 flow_q90 \
       --title "Flow Forecast Anomaly Magnitude (Q10-Q90)" \
       --cmap-under coolwarm \
       --cmap-over magma \
       --cbar \
       --s 40 \
       --savefig anomaly_magnitude.pdf

* `forecast_results.csv`: Positional argument for input CSV.
* `--actual-col actual_flow`: Column with true values.
* `--q-cols ...`: Lower and upper quantile columns.
* `--title "..."`: Custom plot title.
* `--cmap-under/--cmap-over ...`: Customizes colormaps.
* `--cbar`: Adds a color bar for anomaly magnitude.
* `--s 40`: Sets marker size.
* `--savefig ...`: Saves the plot as PDF.

Coverage Example (Multiple Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate an overall coverage plot (using radar style) comparing two
prediction models against true values.

.. code-block:: bash

   k-diagram plot_coverage true_values.csv model_A_preds.csv model_B_preds.csv \
       --q 0.05 0.95 \
       --names "Model A" "Model B" \
       --kind radar \
       --cov_fill \
       --title "Coverage Comparison (5%-95% Interval)" \
       --savefig coverage_radar.png

* `true_values.csv`: File with actual values (first positional argument).
* `model_A_preds.csv model_B_preds.csv`: Files with predictions for
  each model (subsequent positional arguments for `y_preds_files`).
  *Note: Assumes these CSVs contain just the lower and upper bounds needed.*
* `--q 0.05 0.95`: Specifies the quantile levels used for the interval (90%).
* `--names ...`: Provides labels for the models in the legend.
* `--kind radar`: Selects the radar chart type.
* `--cov_fill`: Fills the area under the radar lines.
* `--savefig ...`: Saves the plot.


Coverage Diagnostic Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a coverage diagnostic plot using bars, checking if values
in the `observed` column fall between the `lower_bound` and
`upper_bound` columns from `validation_data.csv`. Restricts the view
to a half-circle and saves the output.

.. code-block:: bash

   k-diagram plot_coverage_diagnostic validation_data.csv \
       --actual-col observed \
       --q-cols lower_bound upper_bound \
       --as-bars \
       --acov half_circle \
       --title "Coverage Diagnostic (Q_low to Q_up)" \
       --savefig coverage_diagnostic_plot.png

* `validation_data.csv`: Positional argument for the input CSV file.
* `--actual-col observed`: Specifies the column with the true values.
* `--q-cols ...`: Specifies the lower and upper quantile columns.
* `--as-bars`: Displays coverage as bars instead of points.
* `--acov half_circle`: Restricts the plot to 180 degrees.
* `--title "..."`: Sets a custom plot title.
* `--savefig ...`: Saves the plot to a file.

Model Drift Example
~~~~~~~~~~~~~~~~~~~~~

Generate a model drift plot showing how average uncertainty width
changes across forecast horizons defined in `multi_horizon_preds.csv`.

.. code-block:: bash

   k-diagram plot_model_drift multi_horizon_preds.csv \
       --q10_cols forecast_h1_q10 forecast_h2_q10 forecast_h3_q10 \
       --q90_cols forecast_h1_q90 forecast_h2_q90 forecast_h3_q90 \
       --horizons "1-Step" "2-Steps" "3-Steps" \
       --acov quarter_circle \
       --title "Forecast Uncertainty Drift" \
       --savefig model_drift.svg

* `multi_horizon_preds.csv`: The input data file.
* `--q10_cols ...`: Lists the columns containing the lower quantile
  for each horizon.
* `--q90_cols ...`: Lists the columns containing the upper quantile
  for each horizon.
* `--horizons ...`: Provides labels for each horizon shown on the plot.
* `--acov quarter_circle`: Restricts the plot to a 90-degree arc.
* `--savefig ...`: Saves the plot as an SVG file.


Velocity Example
~~~~~~~~~~~~~~~~~~
Generate a velocity plot from `median_predictions.csv`, calculating the
average rate of change across Q50 predictions from three time steps.
Colors points by velocity and saves as SVG.

.. code-block:: bash

   k-diagram plot_velocity median_predictions.csv \
       --q50-cols q50_yr1 q50_yr2 q50_yr3 \
       --use-velocity-color \
       --cmap coolwarm \
       --title "Prediction Velocity (Color by Change)" \
       --cbar \
       --savefig prediction_velocity.svg

* `median_predictions.csv`: Positional argument for input CSV.
* `--q50-cols ...`: Specifies the Q50 columns for consecutive periods.
* `--use-velocity-color`: Colors points by the calculated velocity
  value (positive/negative change) instead of absolute Q50 magnitude.
* `--cmap coolwarm`: Uses a diverging colormap suitable for velocity.
* `--title "..."`: Custom plot title.
* `--cbar`: Adds a color bar (representing velocity).
* `--savefig ...`: Saves the plot as SVG.

Taylor Diagram Example (from Arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a Taylor Diagram using the flexible `taylor_diagram` command,
providing raw data arrays and using background shading.

.. code-block:: bash

   k-diagram taylor_diagram \
       --reference-file observed_data.csv \
       --y-preds-files model_X_output.csv model_Y_output.csv \
       --names "Model X" "Model Y" \
       --cmap plasma \
       --radial-strategy performance \
       --norm-c \
       --title "Model Performance Comparison" \
       --savefig taylor_diagram.png

* `--reference-file observed_data.csv`: Specifies the reference data.
* `--y-preds-files ...`: Specifies the files containing predictions
  from Model X and Model Y. Assumes each CSV contains a single column
  of values.
* `--names ...`: Assigns names for the legend.
* `--cmap plasma --radial-strategy performance --norm-c`: Configures
  the background shading to highlight the best-performing region and
  normalizes its colors.
* `--savefig ...`: Saves the output.

Taylor Diagram Example (Shaded Background & Orientation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a Taylor Diagram using `plot_taylor_diagram_in`, focusing on
the background shading and changing the plot orientation.

.. code-block:: bash

   k-diagram plot_taylor_diagram_in observed_data.csv \
       model_X_output.csv model_Y_output.csv \
       --names "Model X" "Model Y" \
       --radial-strategy convergence \
       --cmap viridis \
       --zero-location N \
       --direction 1 \
       --cbar True \
       --title "Taylor Diagram (Corr Background, N-Oriented)" \
       --savefig taylor_in_plot.svg

* `observed_data.csv`, `model_X_output.csv`, `model_Y_output.csv`:
  Positional arguments for reference and prediction files.
* `--radial-strategy convergence --cmap viridis`: Sets the background
  color to represent the correlation coefficient.
* `--zero-location N --direction 1`: Orients the plot with Corr=1 at
  the top (North) and angles increasing counter-clockwise.
* `--cbar True`: Displays the color bar for the background map.
* `--savefig ...`: Saves the output.

Feature Fingerprint Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a feature fingerprint (radar) plot from an importance matrix
stored in `feature_importances.csv`. Assume the CSV contains only the
numeric importance values (rows=layers, columns=features).

.. code-block:: bash

   k-diagram plot_feature_fingerprint feature_importances.csv \
       --features Rainfall Temp WindMoisture Radiation Topo \
       --labels Method1 Method2 Method3 \
       --no-normalize \
       --fill \
       --title "Feature Importance Comparison" \
       --savefig fingerprint.png

* `feature_importances.csv`: Positional argument for the CSV file
  containing the importance matrix.
* `--features ...`: Provides names for the columns (axes).
* `--labels ...`: Provides names for the rows (layers/polygons).
* `--no-normalize`: Plots the raw importance scores instead of
  scaling each layer to [0, 1].
* `--fill`: Fills the area under the polygons.
* `--savefig ...`: Saves the output.

Relationship Plot Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a relationship plot comparing two prediction models against
true values, mapping true values proportionally to the angle over a
half circle.

.. code-block:: bash

   k-diagram plot_relationship \
       true_data.csv \
       predictions_model_A.csv \
       predictions_model_B.csv \
       --names "Model A" "Model B" \
       --theta-scale proportional \
       --acov half_circle \
       --title "True vs Predicted Relationship (Normalized)" \
       --savefig relationship_plot.pdf

* `true_data.csv`: Positional argument for the true values file.
* `predictions_model_A.csv`, `predictions_model_B.csv`: Positional
  arguments for the prediction files.
* `--names ...`: Assigns names for the legend.
* `--theta-scale proportional`: Maps the angle based on the range of
  true values.
* `--acov half_circle`: Restricts the plot to 180 degrees.
* `--savefig ...`: Saves the output as PDF.


Saving Plots
---------------

By default, generated plots are displayed in an interactive window. To
save a plot to a file instead, use the ``--savefig`` option followed by
the desired file path and name (including the extension, e.g., `.png`,
`.pdf`, `.svg`):

.. code-block:: bash

   k-diagram <plot_command> [inputs...] [options...] --savefig path/to/your/plot.png
