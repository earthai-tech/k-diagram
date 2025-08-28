.. _cli_errors:

===================
Error Diagnostics
===================

Once you've identified a promising model, the next step is to
understand the nature of its errors. The commands on this page offer
advanced polar visualizations to move beyond simple error metrics.
They help you compare the full error distributions of multiple models,
see how errors vary with cyclical drivers like seasonality, and even
visualize 2-D uncertainty.

.. contents:: Table of Contents
   :local:
   :depth: 1

-------------------
Common Conventions
-------------------

The tools on this page read a tabular data file (e.g., ``errors.csv``)
specified as a positional argument or via ``-i/--input``. The file
format is auto-detected, but you can override it with ``--format``. To
save a plot, simply add the ``--savefig out.png`` flag.

---

--------------------
plot-error-violins
--------------------

This command allows you to compare several one-dimensional error
distributions side-by-side using polar "violins." Each model is given
its own angular sector, where the radial extent shows the range of
errors and the width of the violin shows the density. It's a powerful
way to quickly compare the bias, variance, and shape of different
models' errors :footcite:p:`Hintze1998`.

The command is used as follows:

.. code-block:: bash

   k-diagram plot-error-violins INPUT
     [--error COL | --error COL1,COL2,...]...
     [--names NAME1 [NAME2 ...]]
     [--figsize WxH]
     [--cmap NAME]
     [--alpha A]

For example, to compare the error distributions from three models,
``err_a``, ``err_b``, and ``err_c``:

.. code-block:: bash

   k-diagram plot-error-violins errors.csv \
     --error err_a --error err_b,err_c \
     --names "Model A" "Model B" "Model C" \
     --cmap plasma \
     --alpha 0.7 \
     --savefig violins.png

.. note::
   You can specify error columns by repeating the ``--error`` flag or
   providing a comma-separated list. Use ``--names`` to give them
   readable labels for the plot legend.

---

------------------
plot-error-bands
------------------

Often, a model's errors aren't uniform; they might be larger during
certain months or times of day. This command helps you visualize such
patterns by plotting the mean error ± some multiple of the standard
deviation against a cyclical feature (like the month of the year).
The result is a shaded band in polar space that reveals conditional
biases or variance.

You can generate the plot with this structure:

.. code-block:: bash

   k-diagram plot-error-bands INPUT
     --error-col ERR
     --theta-col CYCLIC_FEATURE
     [--theta-period P]
     [--theta-bins K]
     [--n-std S]

Here's an example that shows the mean error ± 1.5 standard deviations
for each month of the year:

.. code-block:: bash

   k-diagram plot-error-bands errors.csv \
     --error-col err \
     --theta-col month \
     --theta-period 12 \
     --theta-bins 12 \
     --n-std 1.5 \
     --color "#2980B9" \
     --alpha 0.35 \
     --savefig error_bands.png

---

---------------------
plot-error-ellipses
---------------------

This command is for visualizing 2-D uncertainty. It draws a filled
ellipse for each data point, where the ellipse's shape and orientation
are defined by the mean and standard deviation in both the radial and
angular directions. This is particularly useful for tasks like object
tracking, where you have uncertainty in both distance and angle.

The command requires columns for the mean and standard deviation of
both polar coordinates (radius and theta):

.. code-block:: bash

   k-diagram plot-error-ellipses INPUT
     --r-col R
     --theta-col THETA_DEG
     --r-std-col R_STD
     --theta-std-col THETA_STD_DEG
     [--color-col C]
     [--n-std S]

For example, to plot 1.5-std ellipses for a set of observations,
coloring them by a "priority" column:

.. code-block:: bash

   k-diagram plot-error-ellipses errors.csv \
     --r-col r \
     --theta-col theta_deg \
     --r-std-col r_std \
     --theta-std-col theta_std_deg \
     --color-col priority \
     --n-std 1.5 \
     --alpha 0.7 \
     --edgecolor black \
     --linewidth 0.5 \
     --savefig ellipses.png

---
   
-------------------------
Troubleshooting & Tips
-------------------------

- **"Missing columns" error?** Make sure the column names you provide
  in the flags exactly match the headers in your data file.
- **Angle Units**: For ``plot-error-ellipses``, the mean and standard
  deviation for the angle (``--theta-col``, ``--theta-std-col``) must
  be in **degrees**.
- **Large Files**: If you're working with a very large CSV file,
  converting it to Parquet first can significantly speed up data
  loading.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options and their descriptions.
- **See Also**: After diagnosing errors with these tools, you might
  want to explore the :doc:`relationship` plots to see how errors
  correlate with true or predicted values.
  
.. raw:: html

    <hr>
    
.. rubric:: References

.. footbibliography::