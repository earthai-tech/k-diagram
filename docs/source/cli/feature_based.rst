.. _cli_feature_based:

=====================
Feature-Based Plots
=====================

Beyond evaluating model predictions, it's often crucial to understand
the features themselves. The commands on this page provide powerful,
feature-centric visualizations. They help you explore how features
interact to influence a target and compare feature importance
profiles across different models or datasets :footcite:p:`kouadiob2025.

.. contents:: Table of Contents
   :local:
   :depth: 1

-------------------
Common Conventions
-------------------

The tools on this page read a tabular data file (e.g., ``data.csv``)
specified as a positional argument or via ``-i/--input``. To save a
plot, simply add the ``--savefig out.png`` flag. For detailed help on
any command, run it with the ``-h`` flag.

---

--------------------------
plot-feature-interaction
--------------------------

How do two features jointly affect an outcome? This command helps you
answer that by creating a polar heatmap. One feature is mapped to the
angle, a second is mapped to the radius, and the color of each cell
shows the aggregated value of a third, target column. It's especially
powerful for visualizing interactions with cyclical features like the
hour of the day or month of the year :footcite:p:`Wickham2014`.

The general usage for this command is:

.. code-block:: bash

   k-diagram plot-feature-interaction INPUT
     --theta-col CYCLIC_FEATURE
     --r-col OTHER_FEATURE
     --color-col TARGET_VARIABLE
     [--statistic mean]
     [--theta-period 24]
     [--theta-bins 24] [--r-bins 10]

For example, to see how solar panel output is affected by the
interaction of the hour of the day and the amount of cloud cover:

.. code-block:: bash

   k-diagram plot-feature-interaction data/solar.csv \
     --theta-col hour \
     --r-col cloud_cover \
     --color-col panel_output \
     --theta-period 24 \
     --theta-bins 24 \
     --r-bins 8 \
     --statistic mean \
     --cmap inferno \
     --title "Solar Output by Hour and Cloud Cover" \
     --savefig solar_interaction.png

---

--------------------------
plot-feature-fingerprint
--------------------------

This command creates a radar (or "spider") chart to visualize and
compare the feature importance profiles of different models or groups.
Each polygon on the chart represents a "layer" (like a model), and
each axis represents a feature. This "fingerprint" makes it easy to
see which features are most important for each model and how these
profiles differ :footcite:p:`Lim2021, scikit-learn`.

The command is highly flexible, allowing you to shape the input data
in different ways:

.. code-block:: bash

   k-diagram plot-feature-fingerprint INPUT
     --cols f1,f2,f3,...
     [--labels L1 L2 ... | --labels-col NAME_COL]
     [--features F1 F2 ...]
     [--transpose]
     [--normalize / --no-normalize]
     [--fill / --no-fill]

**Example 1: Standard Orientation**
By default, each row in your data is treated as a layer (a model).
Here, we get the layer names from the "layer" column.

.. code-block:: bash

   k-diagram plot-feature-fingerprint data/importances.csv \
     --cols feature_1,feature_2,feature_3,feature_4,feature_5 \
     --labels-col layer_name \
     --title "Model Importance Fingerprints" \
     --cmap tab10 \
     --savefig fingerprint_layers.png

**Example 2: Explicit Labels**
You can also provide labels for both the layers (models) and the
features (axes) directly on the command line.

.. code-block:: bash

   k-diagram plot-feature-fingerprint data/importances.csv \
     --cols f1,f2,f3,f4,f5,f6 \
     --labels "Model A" "Model B" "Model C" \
     --features "Temp" "Wind" "Pressure" "Humidity" "Solar" "Time" \
     --normalize \
     --fill \
     --savefig fingerprint_with_labels.png

**Example 3: Transposed Data**
If your data is arranged with features in rows and models in columns,
just add the ``--transpose`` flag.

.. code-block:: bash

   k-diagram plot-feature-fingerprint data/transposed_importances.csv \
     --cols Model_A,Model_B,Model_C \
     --labels-col feature_name \
     --transpose \
     --cmap Set3 \
     --title "Transposed Fingerprint" \
     --savefig fingerprint_transposed.png

---

-------------------------
Troubleshooting & Tips
-------------------------

- **Readability**: For fingerprints with many features, the axis
  labels can get crowded. Consider using shorter feature names or
  generating a larger figure with ``--figsize``.
- **Color Choice**: When preparing figures for publication, use a
  colorblind-friendly palette like ``--cmap tab10`` or ``--cmap viridis``.
- **Need more help?** Run any command with the ``-h`` or ``--help``
  flag to see its full list of options and their descriptions.
- **See Also**: After examining feature importance with a fingerprint,
  you might use ``plot-feature-interaction`` to dive deeper into how
  the top two features interact.
  

.. raw:: html

    <hr>
    
.. rubric:: References

.. footbibliography::