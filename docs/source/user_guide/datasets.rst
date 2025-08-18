.. _userguide_datasets:

============
Datasets 
============

The :mod:`kdiagram.datasets` module provides convenient functions
to access sample datasets included with the package (like the
Zhongshan subsidence data) and to generate various synthetic datasets
on the fly.

These datasets are invaluable for:

* Running examples provided in the documentation and gallery.
* Testing `k-diagram`'s plotting functions with predictable data structures.
* Exploring different scenarios of uncertainty, drift, or model comparison.

Most functions allow you to retrieve data either as a standard
:class:`pandas.DataFrame` or as a :class:`~kdiagram.bunch.Bunch` object
(using the ``as_frame`` parameter). The Bunch object conveniently packages
the DataFrame along with metadata like feature/target names, relevant
column lists, and a description of the dataset's origin or generation
parameters.

Function Summary
------------------

.. list-table:: Dataset Loading and Generation Functions
   :widths: 35 65
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~kdiagram.datasets.load_uncertainty_data`
     - Generates synthetic multi-period quantile data with trends,
       noise, and anomalies. Ideal for drift/consistency plots.
   * - :func:`~kdiagram.datasets.load_zhongshan_subsidence`
     - Loads the included Zhongshan subsidence prediction sample dataset.
   * - :func:`~kdiagram.datasets.make_taylor_data`
     - Generates a reference series and multiple prediction series with
       controlled correlation/standard deviation for Taylor Diagrams.
   * - :func:`~kdiagram.datasets.make_multi_model_quantile_data`
     - Generates quantile predictions from multiple simulated models
       for a single time period. Useful for model comparison plots.
   * - :func:`~kdiagram.datasets.make_cyclical_data`
     - Generates data with true and predicted series exhibiting
       cyclical/seasonal patterns.
   * - :func:`~kdiagram.datasets.make_fingerprint_data`
     - Generates a synthetic feature importance matrix for feature
       fingerprint (radar) plots.


Quantile Naming & Basic Notation
---------------------------------

Synthetic quantile columns follow a consistent pattern:

.. math::

   \text{colname} \;=\;
   \texttt{<prefix>}\_{\texttt{year}}\_\texttt{q}\alpha,
   \qquad \alpha \in \{0.1, 0.5, 0.9\}.

For an observation :math:`i` at period :math:`t`, the **interval width** and
**pointwise coverage indicator** are

.. math::

   w_{i,t} \;=\; Q90_{i,t} - Q10_{i,t},
   \qquad
   \mathbb{1}\!\left\{ Q10_{i,t} \le y_{i} \le Q90_{i,t} \right\}.

Aggregate empirical coverage across :math:`n` points is

.. math::

   \widehat{c}
   \;=\;
   \frac{1}{n}
   \sum_{i=1}^n
   \mathbb{1}\!\left\{ Q10_{i,t} \le y_i \le Q90_{i,t} \right\},
   \qquad
   \text{compare against the nominal level.}

.. tip::
   Keep seeds fixed (``seed=...``) to obtain **deterministic** examples in
   documentation and tests.
   
.. raw:: html

    <hr>
    
Usage Examples
----------------

Below are examples demonstrating how to use each function.

Loading Zhongshan Subsidence Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:func:`~kdiagram.datasets.load_zhongshan_subsidence` loads the packaged
sample (coordinates, targets for 2022/2023, quantiles 2022–2026). You can
**subset by year and quantile** at load time.

.. code-block:: python
   :linenos:

   from kdiagram.datasets import load_zhongshan_subsidence
   import warnings

   # Suppress potential download warnings if data exists locally
   warnings.filterwarnings("ignore", message=".*already exists.*")

   # Load as DataFrame, subsetting years and quantiles
   try:
       df_zhongshan_subset = load_zhongshan_subsidence(
           as_frame=True,
           years=[2023, 2025],
           quantiles=[0.1, 0.9],
           include_target=False, # Exclude 'subsidence_YYYY' cols
           download_if_missing=True # Allow download if not packaged/cached
       )
       print("Loaded Zhongshan Subset DataFrame:")
       print(df_zhongshan_subset.head(3))
       print("\nColumns:")
       print(df_zhongshan_subset.columns)

   except FileNotFoundError as e:
       print(f"Error loading Zhongshan data: {e}")
       print("Ensure the package data was installed correctly or "
             "download is enabled/possible.")
   except Exception as e:
        print(f"An unexpected error occurred: {e}")

.. code-block:: text
   :caption: Example Output (Structure, assuming load successful)

   Loaded Zhongshan Subset DataFrame:
        longitude   latitude  subsidence_2023_q0.1  subsidence_2023_q0.9  subsidence_2025_q0.1  subsidence_2025_q0.9
   0   113.237984  22.494591              ...              ...              ...              ...
   1   113.220802  22.513592              ...              ...              ...              ...
   2   113.225632  22.530231              ...              ...              ...              ...

   Columns:
   Index(['longitude', 'latitude', 'subsidence_2023_q0.1',
          'subsidence_2023_q0.9', 'subsidence_2025_q0.1',
          'subsidence_2025_q0.9'], dtype='object')

Loading Uncertainty Datasets (Synthetic vs Semi-Realistic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are **two** uncertainty-oriented helpers with different purposes:

**1) Fully synthetic generator —**
:func:`~kdiagram.datasets.make_uncertainty_data`

*What it is.* Programmatically **constructs** multi-period quantiles
(Q10/Q50/Q90) with controllable median trend and interval-width dynamics,
plus an optional fraction of **injected coverage failures** in the *first*
period for testing diagnostics.

*When to use.* Benchmarks, tutorials, and unit-style checks where you want
repeatable behavior and knobs (``trend_strength``, ``interval_width_*``,
``anomaly_frac``). Ideal for coverage summaries, pointwise diagnostics,
and drift/consistency analyses :footcite:p:`Gneiting2007b, Jolliffe2012`.

.. code-block:: python
   :linenos:

   from kdiagram.datasets import make_uncertainty_data
   ds = make_uncertainty_data(
       n_samples=200, n_periods=5,
       trend_strength=1.2, interval_width_trend=0.4,
       anomaly_frac=0.2, seed=7
   )
   df = ds.frame


**2) Packaged semi-realistic sample —**
:func:`~kdiagram.datasets.load_uncertainty_data`

*What it is.* Loads a **compact, ready-to-use sample** that **mimics the
schema** and “feel” of the Zhongshan-style quantile outputs (years as
periods, Q10/Q50/Q90 columns, and a single “actual” baseline), but without
having to fetch the full Zhongshan dataset. Think of it as a **toy clone**
of the real structure for quick demos.

*When to use.* You need data that “looks like” the Zhongshan project’s
outputs (column naming and period layout) without network access or large
files—e.g., to wire up gallery pages or quick API examples.

.. code-block:: python
   :linenos:
   
   from kdiagram.datasets import load_uncertainty_data
   toy = load_uncertainty_data(as_frame=False)  # Bunch with metadata
   toy.frame.head()

   # Generate as Bunch (default)
   data_bunch = load_uncertainty_data(
       n_samples=10, n_periods=2, seed=1, prefix="flow"
       )

   print("--- Bunch Object ---")
   print(f"Keys: {list(data_bunch.keys())}")
   print(f"Description:\n{data_bunch.DESCR[:200]}...") # Print start of DESCR
   print("\nDataFrame Head:")
   print(data_bunch.frame.head(3))
   print("\nQ10 Columns:")
   print(data_bunch.q10_cols)

.. code-block:: text
   :caption: Example Output (Structure)

   --- Bunch Object ---
   Keys: ['frame', 'feature_names', 'target_names', 'target', 'quantile_cols', 'q10_cols', 'q50_cols', 'q90_cols', 'n_periods', 'prefix', 'start_year', 'DESCR']
   Description:
   Synthetic Multi-Period Uncertainty Dataset for k-diagram

   **Description:**
   Generates synthetic data simulating quantile forecasts (Q10,
   Q50, Q90) for 'flow' over 2 periods starting
   from 2022 across 10 samples/lo...

   DataFrame Head:
      location_id  longitude   latitude   elevation  flow_actual  ...
   0            0 -116.8388    35.094262  366.807627    16.816179  ...
   1            1 -117.8696    34.045590  247.216119     9.508103  ...
   2            2 -119.749534  35.488999  353.628218     5.439137  ...

   Q10 Columns:
   ['flow_2022_q0.1', 'flow_2023_q0.1']


Quantile naming and the empirical coverage definition follow the
conventions in :footcite:p:`Gneiting2007b, Jolliffe2012`:

.. math::

   w_{i,t} = Q90_{i,t} - Q10_{i,t}, \qquad
   \widehat{c} =
   \frac{1}{n}\sum_{i=1}^n
   \mathbb{1}\{Q10_{i,t} \le y_i \le Q90_{i,t}\}.


Generating Taylor Diagram Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Taylor diagrams summarize **correlation** and **standard deviation** in a
single polar plot :footcite:t:`Taylor2001`. Use
:func:`~kdiagram.datasets.make_taylor_data` to synthesize a reference and
several model series with controllable spread and correlation (bias added
but irrelevant to centered Taylor metrics) :footcite:p:`Jolliffe2012`.


.. code-block:: python
   :linenos:

   from kdiagram.datasets import make_taylor_data

   taylor_data = make_taylor_data(n_models=2, n_samples=50, seed=101)

   print("--- Taylor Data Bunch ---")
   print(f"Reference shape: {taylor_data.reference.shape}")
   print(f"Number of prediction series: {len(taylor_data.predictions)}")
   print(f"Prediction shapes: {[p.shape for p in taylor_data.predictions]}")
   print("\nCalculated Stats:")
   print(taylor_data.stats)
   print(f"\nActual Reference Std Dev: {taylor_data.ref_std:.4f}")

.. code-block:: text
   :caption: Example Output

   --- Taylor Data Bunch ---
   Reference shape: (50,)
   Number of prediction series: 2
   Prediction shapes: [(50,), (50,)]

   Calculated Stats:
              stddev  corrcoef
   Model_A  0.729855  0.835114
   Model_B  1.029889  0.508220

   Actual Reference Std Dev: 0.9404


Generating Multi-Model Quantile Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:func:`~kdiagram.datasets.make_multi_model_quantile_data` simulates **several
models** producing quantiles for the **same horizon**. Each model gets its
own median bias and overall interval width, supporting calibration/coverage
comparisons across models :footcite:p:`Gneiting2007b, Jolliffe2012`.


.. code-block:: python
   :linenos:

   from kdiagram.datasets import make_multi_model_quantile_data

   # Get as DataFrame
   df_multi_model = make_multi_model_quantile_data(
       n_samples=5, n_models=2, seed=5, as_frame=True,
       quantiles=[0.1, 0.5, 0.9]
   )

   print("--- Multi-Model Quantile DataFrame ---")
   print(df_multi_model)

.. code-block:: text
   :caption: Example Output

   --- Multi-Model Quantile DataFrame ---
      y_true  feature_1  feature_2  pred_Model_A_q0.1  pred_Model_A_q0.5  pred_Model_A_q0.9  pred_Model_B_q0.1  pred_Model_B_q0.5  pred_Model_B_q0.9
   0  50.853502   0.533165   5.108194          43.514661          49.740457          54.158097          36.189075          46.430960          58.077600
   1  46.300911   0.639037   1.962088          41.607881          45.545123          51.889254          35.546803          41.932122          51.628643
   2  44.874897   0.138801   5.689870          42.241030          44.652911          49.972431          37.209904          42.587300          50.182159
   3  52.396877   0.948104   2.990119          45.163347          52.437158          57.719859          45.359873          54.715327          60.382700
   4  53.938741   0.776598   5.808982          43.275494          53.397751          61.104506          39.947971          52.309521          63.340564


Generating Cyclical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~
:func:`~kdiagram.datasets.make_cyclical_data` produces a “true” sinusoid plus
one or more **phase-shifted / amplitude-scaled** prediction series with noise,
useful when **angle encodes phase** (e.g., seasonal cycle). This is convenient
for relationship plots and multi-series polar overlays.

.. code-block:: python
   :linenos:

   from kdiagram.datasets import make_cyclical_data

   # Get as Bunch
   cycle_bunch = make_cyclical_data(
       n_samples=12, n_series=1, cycle_period=12, seed=5,
       amplitude_true=5, offset_true=10
   )

   print("--- Cyclical Data Bunch ---")
   print(f"Frame shape: {cycle_bunch.frame.shape}")
   print(f"Series names: {cycle_bunch.series_names}")
   print(cycle_bunch.frame[['time_step', 'y_true', 'model_A']].head())

.. code-block:: text
   :caption: Example Output

   --- Cyclical Data Bunch ---
   Frame shape: (12, 3)
   Series names: ['model_A']
      time_step     y_true    model_A
   0          0   9.830655   9.801473
   1          1  14.369168  14.775036
   2          2  14.989960  15.554347
   3          3   9.668771  10.262745
   4          4   4.783064   5.812793


Generating Fingerprint Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:func:`~kdiagram.datasets.make_fingerprint_data` creates a **layer × feature**
matrix of importances with optional sparsity and structure, for
:func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`. This supports
comparisons of **feature influence profiles** across models or periods—an
interpretability aid complementary to verification metrics
:footcite:p:`Jolliffe2012`.

.. code-block:: python
   :linenos:

   from kdiagram.datasets import make_fingerprint_data

   # Get as DataFrame
   fp_df = make_fingerprint_data(
       n_layers=3, n_features=5, seed=303, as_frame=True,
       sparsity=0.2, add_structure=True
   )

   print("--- Fingerprint Data Frame ---")
   print(fp_df)

.. code-block:: text
   :caption: Example Output

   --- Fingerprint Data Frame ---
              Feature_1  Feature_2  Feature_3  Feature_4  Feature_5
   Layer_A     0.941006   0.000000   0.000000   0.000000   0.000000
   Layer_B     0.130220   0.870414   0.456472   0.769115   0.322668
   Layer_C     0.391512   0.139630   1.022977   0.000000   0.000000

.. raw:: html

    <hr>
    
Integrated Plotting Example
------------------------------

This example shows how to generate a dataset using a `load_` or
`make_` function (requesting the DataFrame directly with
``as_frame=True``) and immediately pass it to a relevant `k-diagram`
plotting function. Here, we generate uncertainty data and create an
anomaly magnitude plot.

.. code-block:: python
   :linenos:

   import kdiagram as kd 
   import matplotlib.pyplot as plt

   # 1. Generate data as DataFrame
   df = kd.datasets.load_uncertainty_data(
       as_frame=True,
       n_samples=200,
       n_periods=1, # Only need first period for this plot
       anomaly_frac=0.2, # Ensure anomalies exist
       prefix="flow",
       start_year=2024,
       seed=99
   )

   # 2. Create the plot using the generated DataFrame
   ax = kd.plot_anomaly_magnitude(
       df=df,
       actual_col='flow_actual',
       q_cols=['flow_2024_q0.1', 'flow_2024_q0.9'],
       title="Anomaly Magnitude on Generated Data",
       cbar=True,
       savefig="../images/dataset_plot_example_anomaly.png"
   )
   plt.close() # Close plot after saving

.. image:: ../images/dataset_plot_example_anomaly.png
   :alt: Example plot generated from dataset function
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This **Anomaly Magnitude Plot** visualizes the errors from the
   synthetic dataset generated by
   :func:`~kdiagram.datasets.load_uncertainty_data`. Only points where
   the 'actual' value falls outside the [Q10, Q90] interval are shown.

   **Analysis and Interpretation:**

   * **Angle (θ):** Represents the index of the generated sample
     (0 to 199), distributed around the circle.
   * **Radius (r):** Shows the **magnitude** of the anomaly – how far
     the ``flow_actual`` value was from the closest bound
     (``flow_2024_q0.1`` or ``flow_2024_q0.9``). Larger radii indicate
     more severe prediction interval failures.
   * **Color:** Distinguishes between **under-predictions** (actual < Q10,
     shown in blues by default and in the legend) and
     **over-predictions** (actual > Q90, shown in reds by default and
     in the legend). The **intensity** of the color, indicated by the
     colorbar, also reflects the anomaly magnitude (radius).

   **🔍 Key Insights from this Example:**

   * The presence of both blue and red points confirms that the
     data generation process successfully created both under- and
     over-prediction anomalies as requested by ``anomaly_frac=0.2``.
   * The points are scattered across various angles, suggesting the
     anomalies were introduced randomly across the samples, without a
     strong angular (index-based) pattern in this synthetic dataset.
   * The radii vary, with some points near the center (small anomaly
     magnitude) and others further out (larger magnitude, up to ~8
     units according to the color bar), indicating a range of error
     severities was generated.

   **💡 Connection to Data Generation:**

   * ``n_samples=200`` created 200 potential points around the circle.
   * ``anomaly_frac=0.2`` aimed to make ~40 points appear as anomalies.
   * ``prefix="flow"`` and ``start_year=2024`` determined the column
     names (`flow_actual`, `flow_2024_q0.1`, `flow_2024_q0.9`)
     required by the plotting function call.
   * The range of radii (anomaly magnitudes) seen reflects the random
     deviations introduced during the synthetic anomaly generation step
     within the ``load_uncertainty_data`` function.
     
.. raw:: html

    <hr>
     
Generating Taylor Data and Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example generates data suitable for Taylor diagrams using
:func:`~kdiagram.datasets.make_taylor_data` and plots it using
:func:`~kdiagram.plot.evaluation.plot_taylor_diagram`. The data is
retrieved as a Bunch object, and relevant attributes are passed to the
plot function.

.. code-block:: python
   :linenos:

   import kdiagram as kd 
   import matplotlib.pyplot as plt

   # 1. Generate data as Bunch object
   taylor_data = kd.datasets.make_taylor_data(
       n_models=4,
       n_samples=150,
       seed=101,
       corr_range=(0.6, 0.98),
       std_range=(0.8, 1.2)
   )

   # 2. Create the plot using data from the Bunch
   # Assuming plot function is kd.plot_taylor_diagram
   ax = kd.plot_taylor_diagram(
       *taylor_data.predictions, # Unpack list of prediction arrays
       reference=taylor_data.reference,
       names=taylor_data.model_names,
       title="Taylor Diagram on Generated Data",
       acov='half_circle',
       # Save the plot
       savefig="../images/dataset_plot_example_taylor.png"
   )
   plt.close() # Close plot after saving

.. image:: ../images/dataset_plot_example_taylor.png
   :alt: Example Taylor Diagram generated from dataset function
   :align: center
   :width: 75%


.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This example first uses
   :func:`~kdiagram.datasets.make_taylor_data` to generate a
   reference dataset and four simulated model prediction datasets
   with varying statistical properties. It then visualizes these
   using :func:`~kdiagram.plot.evaluation.plot_taylor_diagram`.

   **Analysis and Interpretation:**

   * **Axes & Reference:** The plot displays standard deviation as the
     radial distance from the origin and correlation as the angle
     (decreasing clockwise from the left 'W' axis, where Corr=1.0).
     The red arc represents the standard deviation of the reference
     data (which is approximately 1.0).
   * **Model Performance:** Each colored dot represents a model:
   
     * **Model A (Red):** High correlation (~0.9) and standard
       deviation slightly less than the reference (~0.9). It captures
       the pattern well but slightly underestimates variability.
     * **Model B (Purple):** Lower correlation (~0.7) and much higher
       standard deviation (~1.3). It matches the pattern less well
       and overestimates variability.
     * **Model C (Brown):** Good correlation (~0.8) but lower
       standard deviation (~0.8). Captures the pattern reasonably
       but underestimates variability.
     * **Model D (Grey):** Similar correlation to Model B (~0.75) but
       lower standard deviation (~0.85), closer to Model A/C in
       variability.
   * **Overall Skill (RMSD):** The distance from each model point to
     the reference point on the arc (at Corr=1.0, StdDev=1.0)
     indicates the centered RMS difference. Model C appears closest,
     followed perhaps by Model A, suggesting they have the best
     overall balance in this simulation. Model B is clearly the
     furthest (worst RMSD).

   **💡 Connection to Data Generation:**

   * The spread of points reflects the target ranges set in
     `make_taylor_data`: ``corr_range=(0.6, 0.98)`` and
     ``std_range=(0.8, 1.2)``. The function successfully generated
     models whose actual statistics fall within or near these target
     ranges relative to the reference standard deviation of ~1.0.
   * This demonstrates how the generation function can create diverse
     scenarios for testing how different models might appear on a
     Taylor Diagram.
     

Generating Fingerprint Data and Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example uses :func:`~kdiagram.datasets.make_fingerprint_data`
to generate a feature importance matrix (returned directly as a
DataFrame using ``as_frame=True``) and visualizes it with
:func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`.

.. code-block:: python
   :linenos:

   import kdiagram as kd 
   import matplotlib.pyplot as plt

   # 1. Generate data as DataFrame
   fp_df = kd.datasets.make_fingerprint_data(
       n_layers=4,
       n_features=7,
       layer_names=['SVM', 'RF', 'MLP', 'XGB'],
       feature_names=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'],
       seed=303,
       as_frame=True, # Get DataFrame directly
   )

   # 2. Create the plot using the generated DataFrame
   # plot_feature_fingerprint takes the importance matrix (df/array),
   # features (list/df.columns), and labels (list/df.index)
   ax = kd.plot_feature_fingerprint(
       importances=fp_df, # Pass DataFrame directly
       features=fp_df.columns.tolist(), # Get features from columns
       labels=fp_df.index.tolist(),     # Get labels from index
       title="Feature Fingerprint on Generated Data",
       fill=True,
       cmap='Accent',
       # Save the plot
       savefig="../images/dataset_plot_example_fingerprint.png"
   )
   plt.close() # Close plot after saving

.. image:: ../images/dataset_plot_example_fingerprint.png
   :alt: Example Feature Fingerprint plot generated from dataset function
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This **Feature Importance Fingerprint** plot uses a radar chart
   to compare the importance profiles of 7 features (F1-F7) across
   4 simulated models (SVM, RF, MLP, XGB), generated using
   :func:`~kdiagram.datasets.make_fingerprint_data`.

   **Analysis and Interpretation:**

   * **Axes:** Each axis radiating from the center corresponds to one
     of the features (F1 through F7).
   * **Polygons (Layers):** Each colored, filled polygon represents
     one model, as indicated by the legend.
   * **Radius (Normalized Importance):** The distance from the center
     along a feature's axis indicates the *relative importance* of
     that feature *for that specific model*. Since normalization is
     applied per model (the default ``normalize=True`` was used here),
     the radius scales from 0 to 1 (maximum importance *for that model*).
   * **Shape ("Fingerprint"):** The overall shape of each polygon
     provides a distinct "fingerprint", showing which features are
     most influential for each model relative to its own other features.

   **🔍 Key Insights from this Example:**

   * **Distinct Profiles:** Each model clearly relies on different
     features. For instance:
     
     * **SVM (Green):** Primarily driven by F3, with some
       contribution from F1 and F2.
      
     * **RF (Orange):** Shows high relative importance for F1 and
       F6, moderate for F2.
     * **MLP (Blue):** Relies most heavily on F3 and F5.
     * **XGB (Brown):** Dominated by F4, with moderate importance
       for F2, F3, and F5.
   * **Feature Comparison:** We can compare feature relevance *across*
     models. F3 is important for SVM, MLP, and XGB, but not RF. F7
     appears relatively unimportant for all models shown. F1 is crucial
     for RF but less so for others.
   * **Normalization Effect:** Because normalization was used, we are
     comparing the *patterns* of importance. We cannot directly compare
     the absolute importance score of F3 for SVM vs. F3 for MLP from
     this plot alone (use ``normalize=False`` for that).

   **💡 Connection to Data Generation:**

   * The number of axes (7) and polygons (4) match the `n_features`
     and `n_layers` parameters passed to `make_fingerprint_data`.
   * The distinct shapes reflect the `add_structure=True` (default)
     setting in the generator, which aims to make fingerprints differ.
   * The radius scaling to 1.0 for each polygon's maximum point is due
     to `normalize=True` being active.



Generating Cyclical Data and Plotting Relationship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example generates data with cyclical patterns using
:func:`~kdiagram.datasets.make_cyclical_data` (as a DataFrame) and
then plots the relationship between the true values (mapped to angle)
and the normalized predictions (mapped to radius) using
:func:`~kdiagram.plot.relationship.plot_relationship`.

.. code-block:: python
   :linenos:

   import kdiagram as kd # Assuming top-level access or specific imports
   import matplotlib.pyplot as plt
   import numpy as np 

   # 1. Generate cyclical data as DataFrame
   cycle_df = kd.datasets.make_cyclical_data(
       n_samples=365, # Simulate daily data for a year
       n_series=2,
       cycle_period=365,
       pred_bias=[0.5, -0.5],
       pred_phase_shift=[0, np.pi / 12], # Second model lags slightly
       seed=404,
       as_frame=True # Get DataFrame directly
   )

   # 2. Create the plot using the generated DataFrame
   ax = kd.plot_relationship(
       cycle_df['y_true'],
       cycle_df['model_A'], # Access generated prediction columns
       cycle_df['model_B'],
       names=['Model A', 'Model B'], # Use generated names
       title="Relationship Plot on Generated Cyclical Data",
       theta_scale='uniform', # Use uniform angle spacing (like time steps)
       acov='default',      # Full circle
       s=15, alpha=0.6,
       # Save the plot
       savefig="../images/dataset_plot_example_cyclical.png"
   )
   plt.close() # Close plot after saving

.. image:: ../images/dataset_plot_example_cyclical.png
   :alt: Example Relationship plot generated from cyclical dataset function
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot visualizes the relationship between a synthetically
   generated cyclical 'true' signal and predictions from two models
   (Model A, Model B), created using
   :func:`~kdiagram.datasets.make_cyclical_data`. The plot uses
   :func:`~kdiagram.plot.relationship.plot_relationship`.

   **Analysis and Interpretation:**

   * **Angle (θ):** Represents the **time step index** (0 to 364),
     distributed uniformly around the full 360 degrees because
     ``theta_scale='uniform'`` was used. It does *not* directly
     represent the magnitude of `y_true` in this case.
   * **Radius (r):** Represents the **normalized predicted value** for
     each model, scaled independently to the range [0, 1]. Radius=1
     corresponds to the maximum prediction *for that specific model*,
     and Radius=0 corresponds to its minimum prediction.
   * **Colors:** Distinguish Model A (blue-grey) from Model B
     (brown-orange).

   **🔍 Key Insights from this Example:**

   * **Cyclical Patterns:** Both models clearly exhibit cyclical
     behavior, forming distinct orbital patterns, reflecting the
     underlying sine wave generated by `make_cyclical_data`.
   * **Phase Shift:** Model B's pattern appears slightly rotated
     clockwise relative to Model A's pattern. This visualizes the
     `pred_phase_shift` introduced during data generation, where
     Model B was made to lag Model A.
   * **Normalization Effect:** The radial positions show the relative
     level of each prediction *within its own range*. We can compare
     if Model A is at its peak (radius near 1) at the same time step
     (angle) as Model B is at its peak.
   * **Bias Effect:** The slight difference in the average radial
     position between the two models might reflect the different
     `pred_bias` values applied during generation.

   **💡 When to Use:**

   * **Visualize Cyclical Relationships:** Ideal when `y_true` (or the
     variable mapped to angle) represents a cyclical process like
     time of day, day of year, or phase angle.
   * **Compare Normalized Model Responses:** Useful for comparing the
     *relative* pattern or timing of different model predictions over
     a cycle or sequence, even if their absolute scales differ, thanks
     to the independent radial normalization.
   * **Identify Lags/Leads:** Phase differences between prediction
     series become visually apparent as angular offsets.


Loading Uncertainty Data for Model Drift Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example generates synthetic multi-period data using
:func:`~kdiagram.datasets.load_uncertainty_data` (returned as a Bunch
object) and visualizes the uncertainty drift across horizons using
:func:`~kdiagram.plot.uncertainty.plot_model_drift`. The Bunch object
makes accessing the required column lists straightforward.

.. code-block:: python
   :linenos:

   import kdiagram as kd 
   import matplotlib.pyplot as plt

   # 1. Generate data as Bunch object
   # Generate 5 periods for a clearer drift visual
   data = kd.datasets.load_uncertainty_data(
       as_frame=False, # Get Bunch object
       n_samples=100,
       n_periods=5,
       prefix='drift_val',
       start_year=2020,
       interval_width_trend=0.8, # Make width increase over time
       seed=50
   )

   # 2. Prepare arguments for the plot function from Bunch attributes
   # Ensure horizon labels match the generated periods
   horizons = [str(data.start_year + i) for i in range(data.n_periods)]

   # 3. Create the plot using the generated data and extracted info
   ax = kd.plot_model_drift(
       df=data.frame,          # The DataFrame within the Bunch
       q10_cols=data.q10_cols, # List of Q10 columns from Bunch
       q90_cols=data.q90_cols, # List of Q90 columns from Bunch
       horizons=horizons,      # Generated horizon labels
       title="Model Drift on Generated Data",
       acov='quarter_circle',
       # Save the plot
       savefig="../images/dataset_plot_example_drift.png"
   )
   plt.close() # Close plot after saving

.. image:: ../images/dataset_plot_example_drift.png
   :alt: Example Model Drift plot generated from dataset function
   :align: center
   :width: 70%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This example uses :func:`~kdiagram.datasets.load_uncertainty_data`
   to generate synthetic data simulating increasing interval widths
   over 5 periods (2020-2024). The resulting DataFrame and column
   lists (extracted from the Bunch object) are then passed to
   :func:`~kdiagram.plot.uncertainty.plot_model_drift` to visualize
   this trend.

   **Analysis and Interpretation:**

   * **Plot Type:** A polar bar chart confined to a 90-degree arc
     (``acov='quarter_circle'``).
   * **Angle (θ):** Each position corresponds to a forecast horizon,
     labeled here with the years 2020 through 2024.
   * **Radius (r):** The length of each bar represents the **average
     prediction interval width** (mean of Q90 - Q10) calculated
     across all samples *for that specific year*.
   * **Color:** Bars are colored using the default `coolwarm` map,
     transitioning from cool (blue) for lower radial values to warm
     (red) for higher values.
   * **Annotations:** The number above each bar shows the calculated
     mean interval width for that horizon.

   **🔍 Key Insights from this Example:**

   * **Increasing Uncertainty:** The bars clearly get taller (larger
     radius) moving clockwise from 2020 to 2024. This visually
     confirms the positive **drift** in average uncertainty.
   * **Quantified Drift:** The annotations show the mean width
     increasing steadily from ~3.97 in 2020 to ~7.12 in 2024.
   * **Color Reinforcement:** The color shift from blue towards red
     also indicates the increasing magnitude of the average interval
     width across the horizons.

   **💡 Connection to Data Generation:**

   * The clear increase in bar height is a direct result of setting
     ``interval_width_trend=0.8`` when calling
     ``load_uncertainty_data``. This parameter caused the synthetic
     interval widths to widen, on average, for each subsequent period.
   * The labels 2020-2024 correspond correctly to ``start_year=2020``
     and ``n_periods=5``.
   * The use of the Bunch object simplified plotting by providing
     pre-parsed lists ``data.q10_cols`` and ``data.q90_cols``.


Zhongshan Data: Interval Consistency Plot (Half Circle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load Zhongshan data (as Bunch) and plot interval consistency (using
coefficient of variation for radius) restricted to a 180-degree view.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import matplotlib.pyplot as plt
   import warnings
   import pandas as pd

   warnings.filterwarnings("ignore", message=".*already exists.*")
   ax = None
   try:
       # 1. Load data as Bunch
       data = kd.datasets.load_zhongshan_subsidence(
           as_frame=False, download_if_missing=True
           )

       # 2. Check data
       if (data is not None and hasattr(data, 'frame')
               and data.q10_cols and data.q50_cols and data.q90_cols):
           print(f"Plotting interval consistency for Zhongshan.")

           # 3. Create the Interval Consistency plot
           ax = kd.plot_interval_consistency(
               df=data.frame,
               qlow_cols=data.q10_cols,
               qup_cols=data.q90_cols,
               q50_cols=data.q50_cols, # Use Q50 for color context
               use_cv=True,           # Use Coefficient of Variation
               acov='half_circle',    # <<< Use 180 degree view
               title="Zhongshan Interval Consistency (CV, 180°)",
               cmap='Purples',
               s=15, alpha=0.7,
               # Save the plot
               savefig="../images/dataset_plot_example_zhongshan_consistency_half.png"
           )
           plt.close()
       else:
           print("Loaded data object missing required attributes.")

   except FileNotFoundError as e:
       print(f"ERROR - Zhongshan data not found: {e}")
   except Exception as e:
       print(f"An unexpected error occurred: {e}")

   if ax is None: print("Plot generation skipped.")

.. image:: ../images/dataset_plot_example_zhongshan_consistency_half.png
   :alt: Example Interval Consistency plot using Zhongshan data (180 deg)
   :align: center
   :width: 75%

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot uses
   :func:`~kdiagram.plot.uncertainty.plot_interval_consistency`
   to show the **stability of prediction interval widths** (Q90-Q10)
   over time (2022-2026) for the Zhongshan sample dataset. The
   angular coverage is set to 180 degrees (``acov='half_circle'``).

   **Analysis and Interpretation:**

   * **Angle (θ):** Represents the sample index (location 0-897),
     mapped linearly onto the top half of the circle (0° to 180°).
   * **Radius (r):** Shows the **Coefficient of Variation (CV)** of
     the interval width across the years for each location. A higher
     radius signifies greater *relative* inconsistency in the
     predicted uncertainty width over time.
   * **Color:** Represents the **average Q50** (median subsidence
     prediction) across all years for each location, using the
     `Purples` colormap (lighter = lower avg Q50, darker = higher
     avg Q50), as shown by the color bar.

   **🔍 Key Insights from this Example:**

   * **Dominant Consistency:** Similar to the previous consistency
     plot (which used a narrower angle), the overwhelming majority
     of locations cluster very close to the origin (radius near 0).
     This indicates **very high consistency** (low CV) in the
     predicted interval widths over the 5-year period for most
     sample points.
   * **Identified Outliers:** A small number of distinct outlier
     points are visible at much larger radii (CVs > 20), indicating
     locations where the model's uncertainty prediction is highly
     variable across the years relative to its average width.
   * **Color Context:** The dense cluster near the center mostly shows
     lighter purple shades, suggesting that the highly consistent
     predictions often correspond to areas with lower average Q50
     subsidence values. The few high-CV outliers show a mix of colors.
   * **Effect of `acov`:** Compared to an `eighth_circle`, the
     `half_circle` view displays roughly four times as many locations,
     confirming the pattern holds across a larger sample subset.

   **💡 Use Case Connection:**

   * This reinforces the finding that while the uncertainty estimate
     is stable for most locations in the sample, specific outlier
     locations exist where the model's uncertainty predictions are
     erratic over time and require scrutiny.
   * Decision-makers might trust the uncertainty bounds more in the
     low-CV cluster, especially where average predicted subsidence
     (color) is also low.

.. raw:: html

    <hr>

.. rubric:: References

.. footbibliography::