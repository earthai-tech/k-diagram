.. _cli_taylor:

===============
Taylor Diagram
===============

How can you summarize multiple aspects of a model's performance in a
single, intuitive plot? The Taylor diagram is a classic solution,
brilliantly summarizing how closely a model's predictions match a
reference series by plotting their standard deviation and correlation
on a polar axis :footcite:p:`Taylor2001`. This allows for a quick
assessment of model fidelity.

The ``k-diagram`` library provides three commands for creating them:

- ``plot-taylor-diagram``: For a standard, clean diagram.
- ``plot-taylor-diagram-in``: Adds a colored background for context.
- ``taylor-diagram``: A flexible command that can take either raw data
  or pre-computed statistics.

.. list-table:: Available Commands
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Description
   * - :ref:`plot-taylor-diagram <cli_plot_taylor_diagram>`
     - Generates a standard Taylor Diagram from pre-calculated statistics.
   * - :ref:`plot-taylor-diagram-in <cli_plot_taylor_diagram_in>`
     - Generates a Taylor Diagram with background shading from raw data.
   * - :ref:`taylor-diagram <cli_taylor_diagram>`
     - An alias for the `plot-taylor-diagram` command.

-------------------
Common Conventions
-------------------

All commands on this page read a tabular data file (e.g., ``data.csv``)
and require you to specify the ground-truth column with ``--y-true``.
You can provide prediction columns using the ``--pred`` or the named
``--model NAME:COL`` syntax. To save a plot, simply add the
``--savefig out.png`` flag.

.. note::
   In a Taylor diagram, the correlation :math:`\rho` is mapped to the
   angle via :math:`\theta=\arccos(\rho)`, while the model's standard
   deviation is mapped to the radius.


.. _cli_plot_taylor_diagram:

-----------------------
plot-taylor-diagram
-----------------------

This is the primary command for creating a standard Taylor diagram. It
plots each model as a point and includes a reference arc representing
the standard deviation of the true data, making it easy to see which
models are closest to the reference.

The command usage is as follows:

.. code-block:: bash

   k-diagram plot-taylor-diagram INPUT
     --y-true Y_TRUE
     [--pred COL | --model NAME:COL]...
     [--names NAME1 NAME2 ...]
     [--acov half_circle]
     [--zero-location W]
     [--direction -1]

Here's a typical example comparing two models, "Model A" and "Model B":

.. code-block:: bash

   k-diagram plot-taylor-diagram data.csv \
     --y-true y_actual \
     --pred model_a_preds model_b_preds \
     --names "Model A" "Model B" \
     --acov half_circle \
     --savefig taylor_basic.png


.. _cli_plot_taylor_diagram_in:

--------------------------
plot-taylor-diagram-in
--------------------------

This command enhances the standard diagram by adding a shaded
background colormap. The color can represent a diagnostic metric, such
as the correlation itself, providing an extra layer of visual context
for interpreting the model points.

To generate this plot, you can add background-specific flags:

.. code-block:: bash

   k-diagram plot-taylor-diagram-in INPUT
     --y-true Y_TRUE
     [--pred COL | --model NAME:COL]...
     [--radial-strategy convergence]
     [--cmap viridis]
     [--cbar]

For example, let's create a diagram where the background color shows
the correlation field:

.. code-block:: bash

   k-diagram plot-taylor-diagram-in data.csv \
     --y-true y \
     --model A:m1 --model B:m2 \
     --radial-strategy convergence \
     --cmap viridis \
     --cbar \
     --savefig taylor_with_background.png


.. _cli_taylor_diagram:

------------------
taylor-diagram
------------------

This is a highly flexible command that can operate in two distinct
modes, making it useful in a wide variety of situations.

**1. Data-mode (from a dataset)**
This mode works just like the other commands, calculating statistics
directly from your data columns.

**2. Stats-mode (from pre-computed values)**
This mode is incredibly useful when you don't have the raw data but
already know the statistics (standard deviation and correlation). It
allows you to generate a Taylor diagram without needing an input file.

Here is an example of using **stats-mode** to plot the performance of
three models for which we have pre-computed scores:

.. code-block:: bash

   k-diagram taylor-diagram \
     --stddev 1.05 0.88 0.75 \
     --corrcoef 0.91 0.72 0.60 \
     --names "Linear Regression" "SVR" "Random Forest" \
     --draw-ref-arc \
     --cmap plasma \
     --radial-strategy rwf \
     --savefig taylor_from_stats.png

---

-------------------------
Troubleshooting & Tips
-------------------------

- **Orientation**: The diagram's orientation can be confusing at
  first. Use the ``--zero-location`` (where correlation=1 sits, e.g.,
  'E' for East) and ``--direction`` (``-1`` for clockwise) flags to
  match your preferred convention.
- **Correlation Labels**: By default, the angular axis is labeled
  with correlation values. If you'd rather see degrees, use the
  ``--no-angle-to-corr`` flag.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options.
- **See Also**: The Taylor diagram is a great summary tool. For more
  detailed comparisons, you might use the radar charts in
  :doc:`comparison` or dive into feature analysis with the tools in
  :doc:`feature_based`.

.. raw:: html

    <hr>
    
.. rubric:: References

.. footbibliography::