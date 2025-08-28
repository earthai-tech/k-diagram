.. _cli_context:

===============
Context plots
===============

This page covers the essential tools for initial data exploration and
basic model diagnostics. These commands help you get a feel for your
data by visualizing time series, checking correlations, and inspecting
the fundamental properties of forecast errors. Every command reads a
standard tabular file (like a CSV or Parquet) and produces a plot.

.. contents:: Table of Contents
   :local:
   :depth: 1

---------------
Common Patterns
---------------

The commands on this page follow a few simple patterns for specifying
data and saving your work.

- **Input & Format**: Provide the path to your data table as the first
  argument (e.g., ``data.csv``). The format is detected automatically,
  but you can force it with ``--format``.
- **Selecting Columns**: When plotting forecasts, you can specify
  prediction columns with ``--pred``, ``--pred-cols``, or the named
  ``--model NAME:COL`` format. Use ``--names`` to add legend labels.
- **Saving**: By default, plots are shown in a window. To save a plot
  to a file, just add the ``--savefig out.png`` flag.

---

------------------
plot-time-series
------------------

This is often the first plot you'll make. It displays your actual
(true) values over time, overlaid with one or more sets of forecasts.
It's perfect for visually assessing how well your models track the
data, and you can even add an uncertainty band.

The general usage is as follows:

.. code-block:: bash

   k-diagram plot-time-series INPUT
     --x-col TIME
     --actual-col ACTUAL
     --pred FORECAST_1 FORECAST_2 ...
     [--q-lower-col QL] [--q-upper-col QU]
     [--names NAME1 NAME2 ...]

For instance, to plot two models ("Model-1", "Model-2") along with an
80% prediction interval (q10 to q90):

.. code-block:: bash

   k-diagram plot-time-series data.csv \
     --x-col time \
     --actual-col y \
     --pred-cols m1,m2 \
     --names "Model-1" "Model-2" \
     --q-lower-col q10 \
     --q-upper-col q90 \
     --title "Forecast vs. Actuals" \
     --savefig ts.png

.. note::
   If ``--x-col`` is omitted, the DataFrame's index is used. To draw
   an uncertainty band, both ``--q-lower-col`` and ``--q-upper-col``
   must be provided.

---

--------------------------
plot-scatter-corr
--------------------------

This command creates a classic scatter plot of actual values versus
predicted values. By including an identity line (y=x), you can
instantly spot systematic biases—points consistently above the line
indicate under-prediction, while points below indicate
over-prediction.

To generate the plot, you can use this structure:

.. code-block:: bash

   k-diagram plot-scatter-corr INPUT
     --actual-col ACTUAL
     --pred FORECAST_1 FORECAST_2 ...
     [--names NAME1 NAME2 ...]

Here's an example comparing two models, A and B:

.. code-block:: bash

   k-diagram plot-scatter-correlation data.csv \
     --actual-col actual \
     --pred-cols m1,m2 \
     --names A B \
     --s 35 --alpha 0.6 \
     --savefig scatter.png

After checking the direct correlation, it's often useful to analyze
the errors themselves, which the following plots help with.

---

----------------------------
plot-error-autocorr
----------------------------

This plot helps you answer the question: "Are my model's errors
predictable?" It shows the autocorrelation (ACF) of the forecast
errors. For a good model, the errors should be like random noise, with
no significant correlation at any lag.

The command is simple, requiring just the actual and prediction
columns:

.. code-block:: bash

   k-diagram plot-error-autocorr INPUT
     --actual-col ACTUAL
     --pred-col PREDICTION

Here is a basic example:

.. code-block:: bash

   k-diagram plot-error-autocorrelation data.csv \
     --actual-col actual \
     --pred-col m1 \
     --title "ACF of Model Errors" \
     --savefig acf.png

---

---------
plot-qq
---------

A Q-Q (Quantile-Quantile) plot is used to assess if the forecast
errors follow a normal distribution. If the errors are normally
distributed, the points on the plot will lie closely along the
straight diagonal line.

You can generate it easily with:

.. code-block:: bash

   k-diagram plot-qq INPUT
     --actual-col ACTUAL
     --pred-col PREDICTION

Here is an example:

.. code-block:: bash

   k-diagram plot-qq data.csv \
     --actual-col actual \
     --pred-col m1 \
     --title "Q-Q Plot of Model Errors" \
     --savefig qq.png

---

-----------------
plot-error-pacf
-----------------

Similar to the ACF plot, the partial autocorrelation (PACF) plot also
investigates the temporal structure of errors. It's particularly
useful for identifying the order of an autoregressive (AR) process if
you were trying to model the leftover error.

The command usage is as follows:

.. code-block:: bash

   k-diagram plot-error-pacf INPUT
     --actual-col ACTUAL
     --pred-col PREDICTION
     [--pacf-kw KEY=VAL ...]

Here's a simple use case:

.. code-block:: bash

   k-diagram plot-error-pacf data.csv \
     --actual-col actual \
     --pred-col m1 \
     --title "PACF of Model Errors" \
     --savefig pacf.png

.. note::
   This command requires the ``statsmodels`` library to be installed.
   You'll get a helpful error message if it's missing.

---

-------------------
plot-error-dist
-------------------

What does the landscape of your model's errors look like? Are they
centered around zero, or is there a systematic bias? Are they tightly
packed or widely spread? This command helps you answer these questions
by plotting a histogram of the forecast errors (actual - predicted),
giving you an immediate visual sense of their central tendency,
variance, and shape.

By default, it also overlays a smooth Kernel Density Estimate (KDE)
curve, which provides a clearer view of the distribution's underlying
shape, a fundamental technique in data analysis :footcite:p:`Silverman1986`.

The command's general structure is:

.. code-block:: bash

   kdiagram plot-error-dist INPUT
     --actual-col ACTUAL
     --pred-col PREDICTION
     [--bins 30]
     [--kde | --no-kde]
     [--density | --no-density]
     [--title "My Plot Title"]
     [--savefig my_plot.png]

Here is an example that plots the distribution of model errors,
customizing the number of bins and the title:

.. code-block:: bash

   kdiagram plot-error-dist model_results.csv \
     --actual-col y_true \
     --pred-col y_pred \
     --bins 40 \
     --title "Distribution of Model Errors" \
     --savefig error_distribution.png

.. note::
   By default, the histogram's y-axis is normalized to show density,
   making it comparable to the KDE curve. If you'd rather see the raw
   counts in each bin, use the ``--no-density`` flag.
   
---

-------------------------
Troubleshooting & Tips
-------------------------

- **"Missing columns" error?** Double-check that the column names in
  your command exactly match the headers in your data file.
- **Datetime issues?** For ``plot-time-series``, ensure your time
  column is in a standard, parsable format like ISO 8601.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options.
- **See Also**: The error diagnostic plots on this page
  (``plot-error-autocorrelation``, ``plot-qq``, etc.) are excellent
  next steps after you've looked at the main ``plot-time-series``
  and decided which model's errors you want to investigate.
  
.. raw:: html

   <hr>

.. rubric:: References

.. footbibliography::