.. _gallery:

==============================================
Gallery of Uncertainty Visualizations
==============================================

This gallery showcases examples of the various polar diagnostic plots
available in the `k-diagram.plot.uncertainty` module. Each example
includes runnable code to generate the plot.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``../images/gallery_actual_vs_predicted.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots.


----------------------
Actual vs. Predicted
----------------------

Compares actual observed values against point predictions (e.g., Q50)
sample-by-sample. Useful for assessing basic accuracy and bias.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(66)
   n_points = 120
   df = pd.DataFrame({'sample': range(n_points)})
   signal = 20 + 15 * np.cos(np.linspace(0, 6 * np.pi, n_points))
   df['actual'] = signal + np.random.randn(n_points) * 3
   df['predicted'] = signal * 0.9 + np.random.randn(n_points) * 2 + 2

   # --- Plotting ---
   kd.plot_actual_vs_predicted(
       df=df,
       actual_col='actual',
       pred_col='predicted',
       title='Gallery: Actual vs. Predicted (Dots)',
       line=False, # Use dots instead of lines
       r_label="Value",
       actual_props={'s': 25, 'alpha': 0.7},
       pred_props={'s': 25, 'marker': 'x', 'alpha': 0.7},
       savefig="../images/gallery_actual_vs_predicted.png" # Save the plot
   )
   plt.close() # Close the plot window after saving

.. image:: ../images/gallery_actual_vs_predicted.png
   :alt: Actual vs. Predicted Plot Example
   :align: center
   :width: 75%

--------------------
Anomaly Magnitude
--------------------

Highlights instances where the actual value falls outside the
prediction interval [Qlow, Qup]. Shows the location (angle), type
(color), and severity (radius) of anomalies.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   n_points = 180
   df = pd.DataFrame({'sample_id': range(n_points)})
   df['actual'] = np.random.normal(loc=20, scale=5, size=n_points)
   df['q10'] = df['actual'] - np.random.uniform(2, 6, size=n_points)
   df['q90'] = df['actual'] + np.random.uniform(2, 6, size=n_points)
   # Add anomalies
   under_indices = np.random.choice(n_points, 20, replace=False)
   df.loc[under_indices, 'actual'] = df.loc[under_indices, 'q10'] - \
                                      np.random.uniform(1, 5, size=20)
   available = list(set(range(n_points)) - set(under_indices))
   over_indices = np.random.choice(available, 20, replace=False)
   df.loc[over_indices, 'actual'] = df.loc[over_indices, 'q90'] + \
                                     np.random.uniform(1, 5, size=20)

   # --- Plotting ---
   kd.plot_anomaly_magnitude(
       df=df,
       actual_col='actual',
       q_cols=['q10', 'q90'],
       title="Gallery: Prediction Anomaly Magnitude",
       cbar=True,
       s=30,
       verbose=0, # Keep output clean for gallery
       savefig="../images/gallery_anomaly_magnitude.png"
   )
   plt.close()

.. image:: ../images/gallery_anomaly_magnitude.png
   :alt: Anomaly Magnitude Plot Example
   :align: center
   :width: 75%

--------------------
Overall Coverage
--------------------

Calculates and displays the overall empirical coverage rate(s) compared
to the nominal rate. Useful for comparing average interval calibration
across models. Shown here with a radar plot for two simulated models.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   y_true = np.random.rand(100) * 10
   # Model 1 (e.g., ~80% coverage)
   y_pred_q1 = np.sort(np.random.normal(
       loc=y_true[:, np.newaxis], scale=1.5, size=(100, 2)), axis=1)
   # Model 2 (e.g., ~60% coverage - narrower intervals)
   y_pred_q2 = np.sort(np.random.normal(
       loc=y_true[:, np.newaxis], scale=0.8, size=(100, 2)), axis=1)
   q_levels = [0.1, 0.9] # Nominal 80% interval

   # --- Plotting ---
   kd.plot_coverage(
       y_true,
       y_pred_q1,
       y_pred_q2,
       names=['Model A (Wider)', 'Model B (Narrower)'],
       q=q_levels,
       kind='radar', # Use radar chart for profile comparison
       title='Gallery: Overall Coverage Comparison (Radar)',
       cov_fill=True,
       verbose=0,
       savefig="../images/gallery_coverage_radar.png"
   )
   plt.close()

.. image:: ../images/gallery_coverage_radar.png
   :alt: Overall Coverage Radar Plot Example
   :align: center
   :width: 70%

----------------------
Coverage Diagnostic
----------------------

Visualizes coverage success (radius 1) or failure (radius 0) for
each individual data point. Helps diagnose *where* intervals fail.
The solid line shows the overall average coverage rate. Shown here
using bars.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(88)
   n_points = 200
   df = pd.DataFrame({'point_id': range(n_points)})
   df['actual_val'] = np.random.normal(loc=5, scale=1.5, size=n_points)
   df['q_lower'] = 5 - np.random.uniform(1, 3, n_points)
   df['q_upper'] = 5 + np.random.uniform(1, 3, n_points)
   # Some points deliberately outside
   df.loc[::15, 'actual_val'] = df.loc[::15, 'q_upper'] + 1

   # --- Plotting ---
   kd.plot_coverage_diagnostic(
       df=df,
       actual_col='actual_val',
       q_cols=['q_lower', 'q_upper'],
       title='Gallery: Point-wise Coverage Diagnostic (Bars)',
       as_bars=True, # Display as bars instead of scatter
       fill_gradient=True, # Show background gradient
       verbose=0,
       savefig="../images/gallery_coverage_diagnostic_bars.png"
   )
   plt.close()

.. image:: ../images/gallery_coverage_diagnostic_bars.png
   :alt: Coverage Diagnostic Plot Example (Bars)
   :align: center
   :width: 75%

-------------------------
Interval Consistency
-------------------------

Analyzes the stability of the prediction interval width (Qup - Qlow)
for each location over multiple time steps. Radius shows variability
(CV or Std Dev); color often shows average Q50. High radius means
inconsistent width.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(42)
   n_points = 100
   n_years = 4
   years = list(range(2021, 2021 + n_years))
   df = pd.DataFrame({'id': range(n_points)})
   qlow_cols, qup_cols, q50_cols = [], [], []
   for i, year in enumerate(years):
       ql, qu, q50 = f'val_{year}_q10', f'val_{year}_q90', f'val_{year}_q50'
       qlow_cols.append(ql); qup_cols.append(qu); q50_cols.append(q50)
       base_low = np.random.rand(n_points)*5 + i*0.2
       width = np.random.rand(n_points)*3 + 1 + np.sin(
           np.linspace(0, np.pi, n_points))*i # Vary width
       df[ql] = base_low; df[qu] = base_low + width
       df[q50] = base_low + width/2 + np.random.randn(n_points)*0.5

   # --- Plotting ---
   kd.plot_interval_consistency(
       df=df,
       qlow_cols=qlow_cols,
       qup_cols=qup_cols,
       q50_cols=q50_cols, # Color by average Q50
       use_cv=True,       # Radius = Coefficient of Variation of width
       title='Gallery: Interval Width Consistency (CV)',
       acov='half_circle',
       cmap='viridis',
       savefig="../images/gallery_interval_consistency_cv.png"
   )
   plt.close()

.. image:: ../images/gallery_interval_consistency_cv.png
   :alt: Interval Consistency Plot Example
   :align: center
   :width: 75%

-------------------
Interval Width
-------------------

Visualizes the magnitude of the prediction interval width (Qup - Qlow)
for each sample at a single time point. Radius directly represents the
width. Color can represent width or an optional third variable (`z_col`).

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(77)
   n_points = 150
   df = pd.DataFrame({'location': range(n_points)})
   df['elevation'] = np.linspace(100, 500, n_points)
   df['q10_val'] = np.random.rand(n_points) * 20
   width = 5 + (df['elevation'] / 100) * np.random.uniform(0.5, 2, n_points)
   df['q90_val'] = df['q10_val'] + width
   df['q50_val'] = df['q10_val'] + width / 2 # Use as z_col

   # --- Plotting ---
   kd.plot_interval_width(
       df=df,
       q_cols=['q10_val', 'q90_val'],
       z_col='q50_val', # Color points by Q50 value
       title='Gallery: Interval Width (Colored by Q50)',
       cmap='plasma',
       cbar=True,
       s=30,
       savefig="../images/gallery_interval_width_z.png"
   )
   plt.close()

.. image:: ../images/gallery_interval_width_z.png
   :alt: Interval Width Plot Example
   :align: center
   :width: 75%

----------------
Model Drift
----------------

Shows how *average* uncertainty (mean interval width) evolves across
different forecast horizons using a polar bar chart. Helps diagnose
model degradation over lead time.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(0)
   years = [2023, 2024, 2025, 2026, 2027]
   n_samples = 50
   df = pd.DataFrame()
   q10_cols, q90_cols = [], []
   for i, year in enumerate(years):
       ql, qu = f'val_{year}_q10', f'val_{year}_q90'
       q10_cols.append(ql); q90_cols.append(qu)
       q10 = np.random.rand(n_samples)*5 + i*0.5 # Width tends to increase
       q90 = q10 + np.random.rand(n_samples)*2 + 1 + i*0.8
       df[ql]=q10; df[qu]=q90

   # --- Plotting ---
   kd.plot_model_drift(
       df=df,
       q10_cols=q10_cols,
       q90_cols=q90_cols,
       horizons=years, # Label bars with years
       acov='quarter_circle', # Use 90 degree span
       title='Gallery: Model Drift Across Horizons',
       savefig="../images/gallery_model_drift.png"
   )
   plt.close()

.. image:: ../images/gallery_model_drift.png
   :alt: Model Drift Plot Example
   :align: center
   :width: 70%

-------------------------
Temporal Uncertainty
-------------------------

A general polar scatter plot for visualizing multiple data series. Often
used to show different quantiles (e.g., Q10, Q50, Q90) for a *single*
time step to illustrate the uncertainty spread across samples.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(99)
   n_points = 80
   df = pd.DataFrame({'id': range(n_points)})
   base = 10 + 5*np.sin(np.linspace(0, 2*np.pi, n_points))
   df['val_q10'] = base - np.random.rand(n_points)*2 - 1
   df['val_q50'] = base + np.random.randn(n_points)*0.5
   df['val_q90'] = base + np.random.rand(n_points)*2 + 1

   # --- Plotting ---
   kd.plot_temporal_uncertainty(
       df=df,
       q_cols=['val_q10', 'val_q50', 'val_q90'],
       names=['Q10', 'Q50', 'Q90'],
       title='Gallery: Uncertainty Spread (Q10, Q50, Q90)',
       normalize=False, # Show raw values
       cmap='coolwarm', # Use diverging map for bounds
       s=20,
       mask_angle=True,
       savefig="../images/gallery_temporal_uncertainty_quantiles.png"
   )
   plt.close()

.. image:: ../images/gallery_temporal_uncertainty_quantiles.png
   :alt: Temporal Uncertainty Plot Example (Quantiles)
   :align: center
   :width: 75%

--------------------
Uncertainty Drift
--------------------

Visualizes how the interval width pattern evolves across multiple time
steps using concentric rings. Each ring represents a time step, showing
the relative uncertainty width at each angle (location).

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(55)
   n_points = 90; n_years = 4; years = range(2020, 2020 + n_years)
   df = pd.DataFrame({'id': range(n_points)})
   qlow_cols, qup_cols = [], []
   for i, year in enumerate(years):
       ql, qu = f'value_{year}_q10', f'value_{year}_q90'
       qlow_cols.append(ql); qup_cols.append(qu)
       base_low = np.random.rand(n_points)*3 + i*0.1
       width = (np.random.rand(n_points)+0.5)*(1.5+i*0.3 + np.cos(
           np.linspace(0, 2*np.pi, n_points)))
       df[ql] = base_low; df[qu] = base_low + width
       df[qu] = np.maximum(df[qu], df[ql]) # Ensure non-negative width

   # --- Plotting ---
   kd.plot_uncertainty_drift(
       df=df,
       qlow_cols=qlow_cols,
       qup_cols=qup_cols,
       dt_labels=[str(y) for y in years],
       title='Gallery: Uncertainty Drift (Rings)',
       cmap='magma',
       base_radius=0.1, band_height=0.1,
       savefig="../images/gallery_uncertainty_drift_rings.png"
   )
   plt.close()

.. image:: ../images/gallery_uncertainty_drift_rings.png
   :alt: Uncertainty Drift Rings Plot Example
   :align: center
   :width: 75%

--------------------
Prediction Velocity
--------------------

Visualizes the average rate of change (velocity) of the median (Q50)
prediction over consecutive time periods for each location. Radius
indicates velocity magnitude; color can indicate velocity or average Q50.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(123)
   n_points = 100; years = range(2020, 2024)
   df = pd.DataFrame({'location_id': range(n_points)})
   q50_cols = []
   base_val = np.random.rand(n_points)*10
   trend = np.linspace(0, 5, n_points)
   for i, year in enumerate(years):
       q50_col = f'val_{year}_q50'
       q50_cols.append(q50_col)
       noise = np.random.randn(n_points)*0.5
       df[q50_col] = base_val + trend*i + noise

   # --- Plotting ---
   kd.plot_velocity(
       df=df,
       q50_cols=q50_cols,
       title='Gallery: Prediction Velocity (Colored by Avg Q50)',
       use_abs_color=True, # Color by magnitude of Q50
       normalize=True,     # Normalize radius (velocity)
       cmap='cividis',
       cbar=True,
       s=25,
       savefig="../images/gallery_velocity_abs_color.png"
   )
   plt.close()

.. image:: ../images/gallery_velocity_abs_color.png
   :alt: Prediction Velocity Plot Example
   :align: center
   :width: 75%
   

.. raw:: html

   <hr>
   <h2 style="text-align: center;">Model Evaluation (Taylor Diagrams)</h2>
   <hr>

-----------------------------------------------
Taylor Diagram (Flexible Input & Background)
-----------------------------------------------

Uses :func:`~kdiagram.plot.evaluation.taylor_diagram`. This example
shows its flexibility by accepting raw data arrays and adding a
background colormap based on the 'rwf' (Radial Weighting Function)
strategy, emphasizing points with good correlation and reference-like
standard deviation.

.. code-block:: python
   :linenos:

   import kdiagram.plot.evaluation as kde 
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(101)
   n_points = 150
   reference = np.random.normal(0, 1.0, n_points) # Reference std dev = 1.0

   # Model A: High correlation, slightly lower std dev
   pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
   # Model B: Lower correlation, higher std dev
   pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
   # Model C: Good correlation, similar std dev
   pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)

   y_preds = [pred_a, pred_b, pred_c]
   names = ["Model A", "Model B", "Model C"]

   # --- Plotting ---
   kde.taylor_diagram(
       y_preds=y_preds,
       reference=reference,
       names=names,
       cmap='Blues',             # Add background shading
       radial_strategy='rwf',    # Use RWF strategy for background
       norm_c=True,              # Normalize background colors
       title='Gallery: Taylor Diagram (RWF Background)',
       # savefig="../images/gallery_taylor_diagram_rwf.png"
   )
   # Use savefig in practice:
   plt.savefig("../images/gallery_taylor_diagram_rwf.png", bbox_inches='tight')
   plt.close()

.. image:: ../images/gallery_taylor_diagram_rwf.png
   :alt: Taylor Diagram with RWF Background Example
   :align: center
   :width: 80%

-------------------------------------------
Taylor Diagram (Background Shading Focus)
-------------------------------------------

Uses :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`. This
example highlights the background colormap feature, here using the
'convergence' strategy where color intensity relates directly to the
correlation coefficient. It also demonstrates changing the plot
orientation.

.. code-block:: python
   :linenos:

   import kdiagram.plot.evaluation as kde 
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (can reuse from previous example) ---
   np.random.seed(101)
   n_points = 150
   reference = np.random.normal(0, 1.0, n_points)
   pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
   pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
   pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)
   y_preds = [pred_a, pred_b, pred_c]
   names = ["Model A", "Model B", "Model C"]

   # --- Plotting ---
   kde.plot_taylor_diagram_in(
       *y_preds,                     # Pass predictions as separate args
       reference=reference,
       names=names,
       radial_strategy='convergence',# Background color shows correlation
       cmap='viridis',
       zero_location='N',            # Place Corr=1 at the Top (North)
       direction=1,                  # Counter-clockwise angles
       cbar=True,                    # Show colorbar for correlation
       title='Gallery: Taylor Diagram (Correlation Background, N-oriented)',
       # savefig="../images/gallery_taylor_diagram_in_conv.png"
   )
   # Use savefig in practice:
   plt.savefig("../images/gallery_taylor_diagram_in_conv.png", bbox_inches='tight')
   plt.close()

.. image:: ../images/gallery_taylor_diagram_in_conv.png
   :alt: Taylor Diagram with Correlation Background Example
   :align: center
   :width: 80%


-----------------------------
Taylor Diagram (Basic Plot)
-----------------------------

Uses :func:`~kdiagram.plot.evaluation.plot_taylor_diagram`. This example
shows a more standard Taylor Diagram layout without background shading,
focusing purely on the positions of the model points relative to the
reference. Uses a half-circle layout.

.. code-block:: python
   :linenos:

   import kdiagram.plot.evaluation as kde # Assuming this is the module path
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation (can reuse from previous example) ---
   np.random.seed(101)
   n_points = 150
   reference = np.random.normal(0, 1.0, n_points)
   pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
   pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
   pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)
   y_preds = [pred_a, pred_b, pred_c]
   names = ["Model A", "Model B", "Model C"]

   # --- Plotting ---
   # Note: Assuming plot_taylor_diagram has similar core functionality
   # Adjust parameters based on its actual final signature if different
   kde.plot_taylor_diagram(
       *y_preds,
       reference=reference,
       names=names,
       acov='half_circle',      # Use 90-degree layout
       zero_location='W',       # Place Corr=1 at the Left (West) - default
       direction=-1,            # Clockwise angles - default
       # draw_ref_arc=True,     # Default likely True
       # angle_to_corr=True,    # Default likely True
       title='Gallery: Basic Taylor Diagram (Half Circle)',
       # savefig="../images/gallery_taylor_diagram_basic.png"
   )
   # Use savefig in practice:
   plt.savefig("../images/gallery_taylor_diagram_basic.png", bbox_inches='tight')
   plt.close()

.. image:: ../images/gallery_taylor_diagram_basic.png
   :alt: Basic Taylor Diagram Example
   :align: center
   :width: 80%


.. raw:: html

   <hr>


.. raw:: html

   <hr>
   <h2 style="text-align: center;">Feature-Based Visualization</h2>
   <hr>

--------------------------------
Feature Importance Fingerprint
--------------------------------

Uses :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`.
This radar chart compares the importance profiles ("fingerprints") of
several features across different groups or layers (e.g., different years
or models). This example shows raw (unnormalized) importance values.

.. code-block:: python
   :linenos:

   import kdiagram.plot.feature_based as kdf # Assuming this module path
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   features = ['Rainfall', 'Temperature', 'Wind Speed',
               'Soil Moisture', 'Solar Radiation', 'Topography']
   n_features = len(features)
   years = ['2022', '2023', '2024']
   n_layers = len(years)

   # Generate importance scores (rows=years, cols=features)
   # Make them slightly different per year
   np.random.seed(123)
   importances = np.random.rand(n_layers, n_features) * 0.5
   importances[0, 0] = 0.8 # Rainfall important in 2022
   importances[1, 3] = 0.9 # Soil Moisture important in 2023
   importances[2, 1] = 0.7 # Temperature important in 2024
   importances[2, 4] = 0.75# Solar Radiation also important in 2024

   # --- Plotting ---
   kdf.plot_feature_fingerprint(
       importances=importances,
       features=features,
       labels=years,
       normalize=False, # Show raw importance scores
       fill=True,
       cmap='Pastel1',
       title="Gallery: Feature Importance Fingerprint (Yearly)",
       # savefig="../images/gallery_feature_fingerprint.png"
   )
   # Use savefig in practice:
   plt.savefig("../images/gallery_feature_fingerprint.png", bbox_inches='tight')
   plt.close()

.. image:: ../images/gallery_feature_fingerprint.png
   :alt: Feature Importance Fingerprint Plot Example
   :align: center
   :width: 75%


.. raw:: html

   <hr>
   <h2 style="text-align: center;">Relationship Visualization</h2>
   <hr>

---------------------------------
Relationship Plot
---------------------------------

Uses :func:`~kdiagram.plot.relationship.plot_relationship`. This plot
maps true values (`y_true`) to the angle and normalized predicted values
(`y_pred`) to the radius. It helps visualize how multiple prediction
series relate to the true values across their range. This example uses
proportional scaling for the angle.

.. code-block:: python
   :linenos:

   import kdiagram.plot.relationship as kdr 
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(200)
   n_points = 150
   # True values with some range
   y_true = np.linspace(0, 20, n_points) + np.random.normal(0, 1, n_points)
   # Prediction 1: Good correlation + noise
   y_pred1 = y_true * 1.1 + np.random.normal(0, 2, n_points)
   # Prediction 2: Weaker correlation, different scale + noise
   y_pred2 = y_true * 0.5 + 5 + np.random.normal(0, 3, n_points)

   # --- Plotting ---
   kdr.plot_relationship(
       y_true, y_pred1, y_pred2, # Pass y_true first, then predictions
       names=["Model Alpha", "Model Beta"],
       theta_scale='proportional', # Angle based on y_true value
       acov='default',           # Use full circle
       title="Gallery: True vs. Predicted Relationship",
       s=40, alpha=0.6,
       # savefig="../images/gallery_relationship.png"
   )
   # Use savefig in practice:
   plt.savefig("../images/gallery_relationship.png", bbox_inches='tight')
   plt.close()

.. image:: ../images/gallery_relationship.png
   :alt: Relationship Plot Example
   :align: center
   :width: 75%

.. raw:: html

   <hr>

# Add this section within your existing docs/source/gallery.rst file

# ... (previous gallery sections) ...

.. raw:: html

   <hr>
   <h2 style="text-align: center;">Utility Function Examples</h2>
   <hr>

