===================
Feature-Based CLI
===================

Command-line tools for feature-centric polar visualizations. These
commands live under the top-level ``kdiagram`` executable. Run any
command with ``-h`` to see its full help.

---

plot-feature-interaction
------------------------

A polar heatmap shows how a target varies with two features—one
mapped to angle, one to radius.

### Synopsis

.. code-block:: bash

   kdiagram plot-feature-interaction INPUT
     --theta-col <col>
     --r-col <col>
     --color-col <col>
     [--statistic <agg>] [--theta-period <p>]
     [--theta-bins <n>] [--r-bins <m>]
     [--title <str>] [--figsize W,H] [--cmap <map>]
     [--show-grid / --no-show-grid]
     [--mask-radius / --no-mask-radius]
     [--dpi <int>] [--savefig <path>]
     [--format <csv|parquet|...>]
     [-i | --input <path>]

### Required arguments

``INPUT``
  Path to a table (CSV, Parquet, …). You can also pass it via
  ``--input``.

``--theta-col``
  The column mapped to the angular axis (θ). Cyclical features like
  **hour** or **month** work well.

``--r-col``
  The column mapped to the radial axis (r).

``--color-col``
  The target column whose aggregated value colors each sector.

### Common options

``--statistic`` (default: mean)
  The aggregation applied within each 2D bin (e.g., **mean**,
  **median**, **std**). Any pandas reducer name is accepted.

``--theta-period``
  The period of the angular feature (e.g., **24** for hours, **12**
  for months). This ensures proper wrap-around.

``--theta-bins`` (default: 24), ``--r-bins`` (default: 10)
  The number of bins along the angle and radius.

``--cmap`` (default: viridis)
  The Matplotlib colormap for the heatmap.

``--figsize`` (default: 8,8), ``--title``, ``--dpi``
  Standard figure settings.

``--show-grid / --no-show-grid`` (default: show)
  Toggles the polar grid.

``--mask-radius / --no-mask-radius`` (default: no)
  Hides radial tick labels.

``--savefig``
  Saves the figure to a file instead of showing it.

### Example

.. code-block:: bash

   # Hour vs. cloud cover interaction on panel output
   kdiagram plot-feature-interaction data/solar.csv \
     --theta-col hour --r-col cloud --color-col output \
     --theta-period 24 --theta-bins 24 --r-bins 8 \
     --statistic mean --cmap inferno \
     --title "Solar Output by Hour × Cloud" \
     --savefig out/interaction.png

**Python API**:
``:mod:kdiagram.plot.feature_based.plot_feature_interaction``

---

plot-feature-fingerprint
------------------------

A radar (polar) chart compares feature-importance profiles across
different layers (e.g., models, years). Each polygon is a layer, and
each axis is a feature.

### Synopsis

.. code-block:: bash

   kdiagram plot-feature-fingerprint INPUT
     --cols <c1[,c2,...]>
     [--labels L1 [L2 ...] | --labels-col <col>]
     [--features F1 [F2 ...]]
     [--transpose]
     [--normalize / --no-normalize]
     [--fill / --no-fill]
     [--cmap <map>] [--title <str>] [--figsize W,H]
     [--show-grid / --no-show-grid]
     [--dpi <int>] [--savefig <path>]
     [--format <csv|parquet|...>] [-i | --input <path>]

### Required arguments

``INPUT``
  Path to a table (CSV, Parquet, …).

``--cols``
  A comma-separated list of numeric columns that form the importance
  matrix.

### Orientation & labels

**Default orientation (rows = layers, columns = features)**
  Each row is treated as a layer (polygon). The columns specified in
  ``--cols`` serve as the feature axes.

``--transpose``
  Swaps the interpretation: rows become features, and the columns in
  ``--cols`` become the layer columns.

``--labels``
  Explicit layer names, provided as space-separated values. The
  length should match the number of layers.

``--labels-col``
  Takes layer names (default) or feature names (with
  ``--transpose``) from a specified column.

``--features``
  Explicit names for the feature axes. If omitted, generic names
  like "Feature 1" are used.

### Appearance

``--normalize / --no-normalize`` (default: normalize)
  Normalizes each layer to its row-wise maximum, so that the shapes
  are comparable.

``--fill / --no-fill`` (default: fill)
  Fills the polygons with translucent colors.

``--cmap`` (default: tab10)
  The colormap or palette for the different layers.

``--show-grid / --no-show-grid`` (default: show)
  Toggles the polar grid.

``--title``, ``--figsize``, ``--dpi``, ``--savefig``
  Standard figure controls.

### Examples

.. code-block:: bash

   # Layers in rows, labels from a column, default normalization and fill
   kdiagram plot-feature-fingerprint data/imp_layers.csv \
     --cols f1,f2,f3,f4,f5,f6 \
     --labels-col layer \
     --title "Model Importance Fingerprints" \
     --cmap tab10 \
     --savefig out/fingerprint_layers.png

.. code-block:: bash

   # Explicit labels & feature names, keep normalization & fill
   kdiagram plot-feature-fingerprint data/imp_layers.csv \
     --cols f1,f2,f3,f4,f5,f6 \
     --labels A B C \
     --features F1 F2 F3 F4 F5 F6 \
     --normalize --fill \
     --savefig out/fingerprint_labels.png

.. code-block:: bash

   # Transposed: rows are features, layer columns in --cols
   kdiagram plot-feature-fingerprint data/imp_features.csv \
     --cols L1,L2,L3 \
     --labels-col feature \
     --transpose \
     --cmap Set3 \
     --title "Transposed Fingerprint" \
     --savefig out/fingerprint_transpose.png

**Python API**:
``:mod:kdiagram.plot.feature_based.plot_feature_fingerprint``

---

Tips
----

* Clean **NaNs** before plotting, or rely on the commands’ internal
    NaN handling; empty bins are simply omitted in the heatmap.
* For readability with many features or layers, consider using
    shorter axis labels and placing the legend outside the plot area.
* Use colorblind-friendly palettes (like **tab10** or **tab20**) for any
    printed materials. 