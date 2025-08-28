.. _cli_relationship:

=============================
Relationship (Polar) Commands
=============================

How do your model's predictions relate to the ground truth? The
commands on this page are designed to answer that question by
visualizing these relationships on polar axes. They provide a compact,
cyclical view that is excellent for spotting bias, heteroscedasticity
(errors that change with magnitude), and other systematic patterns
that might be missed in a standard Cartesian plot.

.. contents:: Table of Contents
   :local:
   :depth: 1

-------------------
Common Conventions
-------------------

The commands here share a common approach for specifying the essential
data columns.

- **Ground Truth**: The true, observed values are always required and
  can be passed with either ``--y-true`` or its alias ``--true-col``.
- **Point Predictions**: For commands that use point predictions
  (like the median), you can specify model columns with the named
  ``--model NAME:COL`` syntax or by repeating ``--pred``.
- **Quantile Predictions**: For ``plot-conditional-quantiles``, you
  must provide the set of quantile columns (e.g., ``q10,q50,q90``) and
  the corresponding quantile levels via ``--q-levels``.
- **Saving**: To save a plot, add the ``--savefig out.png`` flag.

---

.. _cli-plot-relationship:

-------------------
plot-relationship
-------------------

This command creates a polar scatter plot of true values versus one
or more point prediction series. The angle is determined by the true
values, while the radius is determined by the predictions. It's a great
way to see if different models behave differently across the range of
the target variable.

The command offers a flexible set of options for controlling the
angular mapping:

.. code-block:: bash

   k-diagram plot-relationship INPUT
     --y-true Y_TRUE
     [--pred COLS ... | --model NAME:COLS ...]
     [--names NAMES ...]
     [--theta-scale {proportional,uniform}]
     [--acov {default,half_circle,quarter_circle}]

Here is a simple example comparing two models, ``M1`` and ``M2``:

.. code-block:: bash

   k-diagram plot-relationship data.csv \
     --y-true actual \
     --pred q50_m1 q50_m2 \
     --names "Model 1" "Model 2" \
     --savefig relationship.png

In this next example, we map the true values to a half-circle and use
a "month" column to label the angular ticks, which can help reveal
seasonality in the relationship:

.. code-block:: bash

   k-diagram plot-relationship data.csv \
     --y-true actual \
     --pred q50 \
     --theta-scale uniform \
     --acov half_circle \
     --z-values month \
     --z-label "Month of Year" \
     --title "Truth–Prediction Relationship by Month" \
     --savefig half_circle_relationship.png

---

.. _cli-plot-conditional-quantiles:

----------------------------
plot-conditional-quantiles
----------------------------

This plot visualizes a full set of quantile predictions against the
true values. The data is sorted by the true values to produce a smooth
spiral, with shaded bands representing the uncertainty intervals (e.g.,
the 80% interval between Q10 and Q90). It's a fantastic tool for
assessing if your model's uncertainty estimates are reasonable across
the entire range of outcomes.

To use this command, you provide one set of quantile columns and their
corresponding levels:

.. code-block:: bash

   k-diagram plot-conditional-quantiles INPUT
     --y-true Y_TRUE
     --pred q10,q50,q90
     --q-levels 0.1,0.5,0.9
     [--bands 80,50]

Here, we plot a model's forecasts and shade the 80% and 50% prediction
intervals:

.. code-block:: bash

   k-diagram plot-conditional-quantiles data.csv \
     --y-true actual \
     --pred q10,q50,q90 \
     --q-levels 0.1,0.5,0.9 \
     --bands 80,50 \
     --savefig conditional_quantiles.png

---

.. _cli-plot-residual-relationship:

----------------------------
plot-residual-relationship
----------------------------

This command helps you find patterns in your model's mistakes. It
creates a polar scatter plot of the **residuals (actual - prediction)
versus the predicted values**. The angle is based on the predicted
value, while the radius shows the residual see :footcite:t:`Murphy1993What, Jolliffe2012`). 
An ideal plot would show points randomly scattered around the zero-residual 
circle, indicating that the errors are not dependent on the magnitude 
of the prediction.

The synopsis is as follows:

.. code-block:: bash

   k-diagram plot-residual-relationship INPUT
     --y-true Y_TRUE
     [--pred COLS ... | --model NAME:COLS ...]
     [--show-zero-line]

Let's compare the residuals for two models, "Baseline" and "Wide":

.. code-block:: bash

   k-diagram plot-residual-relationship data.csv \
     --y-true actual \
     --pred q50_baseline q50_wide \
     --names "Baseline" "Wide" \
     --show-zero-line \
     --savefig residuals_vs_predicted.png

---

.. _cli-plot-error-relationship:

-------------------------
plot-error-relationship
-------------------------

This plot is subtly different from the residual plot but equally
important. It shows the **errors (actual - prediction) versus the
true values**. Here, the angle is based on the true value. This view
is essential for diagnosing heteroscedasticity—a common issue where
the variance of the error changes as the true outcome value changes.

The command usage is very similar to the residual plot:

.. code-block:: bash

   k-diagram plot-error-relationship INPUT
     --y-true Y_TRUE
     [--pred COLS ... | --model NAME:COLS ...]
     [--show-zero-line]

Here is an example comparing two models, A and B:

.. code-block:: bash

   k-diagram plot-error-relationship data.csv \
     --y-true actual \
     --model A:q50_a --model B:q50_b \
     --mask-radius \
     --savefig error_vs_true.png

---

-------------------------
Troubleshooting & Tips
-------------------------

- **"Missing columns" error?** Double-check that the column names in
  your command exactly match the headers in your data file.
- **Plot looks strange?** The ``--theta-scale`` option in
  ``plot-relationship`` can have a big impact. Try switching between
  ``proportional`` and ``uniform`` to see which reveals more insight.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options and their descriptions.
- **See Also**: These plots are powerful on their own, but they pair
  well with the tools in :doc:`errors` for a complete picture of your
  model's error behavior.
  
.. raw:: html

    <hr>
    
.. rubric:: References

.. footbibliography::