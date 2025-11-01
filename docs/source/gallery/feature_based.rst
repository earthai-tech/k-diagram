.. _gallery_feature_based:

========================================
Feature-Based Visualization Gallery
========================================

This gallery page showcases plots from `k-diagram` focused on
understanding feature influence and importance. Currently, it features
the Feature Importance Fingerprint plot.

.. note::
   You need to run the code snippets locally to generate the plot
   images referenced below (e.g., ``images/gallery_feature_fingerprint.png``).
   Ensure the image paths in the ``.. image::`` directives match where
   you save the plots (likely an ``images`` subdirectory relative to
   this file).

.. _gallery_plot_feature_fingerprint: 

--------------------------------
Feature Importance Fingerprint
--------------------------------

The :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`
function is a  tool for model interpretation. It creates a polar
radar chart to visualize and compare the importance profiles of multiple
features across different contexts (e.g., different models or time
periods). Each context is represented by a unique "fingerprint,"
allowing for an immediate visual comparison of what drives the model's
decisions.

First, let's break down the components of this comparative plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular axis is assigned to a specific **input
     feature** (e.g., 'Rainfall', 'Temperature').
   * **Radius (r):** Corresponds to the **importance score** of that
     feature for a given layer. This can be the raw score or, more
     commonly, a normalized score (``normalize=True``) where 1.0 is the
     most important feature *within that layer*.
   * **Polygon (Layer):** Each colored polygon represents a different
     **layer** or context, such as a different model, a different time
     period, or a different customer segment. The polygon's shape is the
     "fingerprint" of feature influence for that layer.

With this framework, let's apply the plot to a real-world problem,
starting with a classic model comparison and then moving to a more
advanced analysis of concept drift.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Comparing Different Models' "Logic"**

A primary use of this plot is to compare the internal "logic" of two or
more competing models. Do they rely on the same features to make
decisions, or do they have fundamentally different approaches to solving
the problem?

Let's imagine a telecommunications company has trained a simple
`Logistic Regression` model and a complex `Gradient Boosting` model to
predict customer churn. They need to understand what each model has
learned before deploying it.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- 1. Data Generation: Feature Importances for Two Models ---
   features = [
       'Tenure', 'Monthly Charges', 'Total Charges',
       'Data Usage', 'Support Calls', 'Contract Type'
   ]
   labels = ['Logistic Regression', 'Gradient Boosting']

   # A simple model might rely heavily on a few key features
   logreg_importances = [0.9, 0.8, 0.7, 0.1, 0.2, 0.6]
   # A more complex model might learn from a wider array of signals
   boosting_importances = [0.5, 0.6, 0.6, 0.9, 0.8, 0.4]

   importances = np.array([logreg_importances, boosting_importances])

   # --- 2. Plotting ---
   kd.plot_feature_fingerprint(
       importances=importances,
       features=features,
       labels=labels,
       normalize=True, # Focus on the relative pattern of importance
       title="Use Case 1: Churn Model Feature Importance Fingerprints",
       acov="full", # use full circle.
       savefig="gallery/images/gallery_feature_fingerprint_models.png"
   )
   plt.close()

.. figure:: ../images/feature_based/gallery_feature_fingerprint_models.png
   :align: center
   :width: 70%
   :alt: A radar chart comparing the feature importance of two different models.

   The "fingerprints" of two models, showing that the Logistic
   Regression (blue) relies on tenure and charges, while the Gradient
   Boosting model (orange) relies more on usage and support calls.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This plot immediately reveals the different "worldviews" of the two
   models. The **Logistic Regression** model (blue polygon) has a spiky
   fingerprint, extending furthest on the ``Tenure`` and ``Monthly Charges``
   axes. This indicates it has learned a simple, strong relationship based
   primarily on contract length and cost. In contrast, the **Gradient
   Boosting** model (cyan polygon) shows a more distributed profile. Its
   most important features are ``Data Usage`` and ``Support Calls``,
   suggesting it has learned a more nuanced, behavior-based pattern of
   churn. This insight is critical for deciding which model's logic is
   more aligned with the company's business strategy.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Diagnosing Feature Importance Drift Over Time**

A model's logic may not be static. The factors that predict an outcome
one year might be different the next, a phenomenon known as **concept
drift**. This plot is an excellent tool for diagnosing this drift by
comparing a model's feature importance fingerprints calculated from
different time periods.

Let's analyze a model that predicts crop yield. We'll simulate how the
importance of different environmental factors might change over three
consecutive years due to changing climate patterns.

.. code-block:: python
   :linenos:

   # --- 1. Data Generation: Feature Importances for Three Years ---
   features = ['Rainfall', 'Temperature', 'Wind Speed',
               'Soil Moisture', 'Solar Radiation', 'Topography']
   years = ['2022 (Wet Year)', '2023 (Dry Year)', '2024 (Hot Year)']
   
   # Simulate importance scores that change each year
   importances_yearly = np.array([
       # 2022: A wet year, so rainfall and topography are key
       [0.9, 0.3, 0.2, 0.5, 0.4, 0.6],
       # 2023: A dry year, so soil moisture becomes critical
       [0.4, 0.5, 0.1, 0.9, 0.6, 0.3],
       # 2024: A hot year, so temperature and solar radiation dominate
       [0.2, 0.9, 0.3, 0.4, 0.8, 0.1]
   ])

   # --- 2. Plotting ---
   kd.plot_feature_fingerprint(
       importances=importances_yearly,
       features=features,
       labels=years,
       normalize=True,
       title="Use Case 2: Yearly Drift in Crop Yield Feature Importance",
       cmap='Set2',
       savefig="gallery/images/gallery_feature_fingerprint_drift.png"
   )
   plt.close()

.. figure:: ../images/feature_based/gallery_feature_fingerprint_drift.png
   :align: center
   :width: 70%
   :alt: A radar chart showing how feature importances change over three years.

   Three overlapping polygons, each with a different shape, showing that
   the most important feature for the model changes each year.

.. topic:: 🧠 Interpretation
   :class: hint

   This plot clearly visualizes the phenomenon of concept drift. Each
   year has a distinctly shaped "fingerprint," revealing how the model's
   reliance on different features has evolved. In the **2022 (Wet Year)**,
   the model's predictions were overwhelmingly driven by **Rainfall**. In
   the **2023 (Dry Year)**, the most important feature shifted dramatically
   to **Soil Moisture**. Finally, in the **2024 (Hot Year)**, **Temperature**
   and **Solar Radiation** became the dominant factors. This is a critical
   insight, suggesting that a single, static model is not sufficient and
   that the model may need to be retrained or adapted regularly to account
   for these changing environmental drivers.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper understanding of the statistical concepts behind feature
importance and model interpretation, please refer back to the main
:ref:`ug_feature_fingerprint` section.
      

.. _gallery_plot_fingerprint:

------------------------------------
Feature Fingerprint (Dynamic)
------------------------------------

The :func:`~kdiagram.plot.feature_based.plot_fingerprint` function is a
versatile tool for model and data interpretation. As a next-generation
evolution of the feature fingerprint plot, it not only visualizes
pre-computed importance scores but can also **dynamically calculate
them from raw data**. This allows for rapid, code-efficient exploration
of feature significance across different groups or contexts.

It can operate in two primary modes:

1.  **Unsupervised**: To find the most variable or dispersed features
    within different data segments (e.g., using standard deviation).
2.  **Supervised**: To find features most correlated with a target
    variable.

First, let's review the plot's structure.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Each angular axis is assigned to a specific **input
     feature** (e.g., 'Alcohol', 'Flavanoids').
   * **Radius (r):** Corresponds to the **importance score** of that
     feature. When calculated dynamically, this could be a standard
     deviation, variance, or correlation value. Normalizing this score
     (``normalize=True``) is common to compare the relative patterns.
   * **Polygon (Layer):** Each colored polygon represents a different
     **layer** or context. This function can automatically generate
     these layers by splitting the data using a ``group_col``.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 1: Unsupervised Fingerprint for Variability Analysis**

An ideal use of this function is to understand the intrinsic
properties of a dataset. Let's imagine we have a dataset of different
wine cultivars and want to identify which chemical properties are the
most *variable* for each type. This can reveal the defining, or most
inconsistent, characteristics of each group without respect to a target.

Here, we'll use ``method='std'`` to compute the standard deviation for
each feature, grouped by wine type.

.. code-block:: python
   :linenos:

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.datasets import load_wine
   import kdiagram as kd

   # --- 1) Load and tidy
   wine = load_wine()
   df = pd.DataFrame(wine.data, columns=wine.feature_names)
   df["wine_type"] = pd.Series(wine.target).map(
       {0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"}
   )

   # --- 2) Standardize features globally (z-score) to remove scale effects
   X = df.drop(columns=["wine_type"])
   Z = (X - X.mean()) / X.std(ddof=0)

   # --- 3) Per-cultivar variability on standardized features
   std_by_type = (
       pd.concat([Z, df["wine_type"]], axis=1)
         .groupby("wine_type")
         .std(ddof=0)
   )

   # --- 4) Keep a compact, readable set of axes
   # Pick the top-8 features by average variability across cultivars
   features_top8 = (
       std_by_type.mean(axis=0)
       .sort_values(ascending=False)
       .head(8)
       .index
       .tolist()
   )

   # --- 5) Plot: pass precomputed matrix (layers x features)
   kd.plot_fingerprint(
       std_by_type[features_top8],   # precomputed importances (DataFrame)
       precomputed=True,
       labels=std_by_type.index.tolist(),
       features=features_top8,
       normalize=True,               # compare shapes per cultivar
       title="Chemical Variability Fingerprint by Wine Cultivar",
       acov="half_circle",           # cleaner labels
       # savefig="gallery/images/plot_fingerprint_variability.png",
   )
   plt.close()


.. figure:: ../images/feature_based/plot_fingerprint_variability.png
   :align: center
   :width: 80%
   :alt: A semi-circular radar chart showing feature variability for wines.

   The "fingerprints" show that 'Cultivar A' is most variable in its
   'flavanoids', while 'Cultivar C' is most variable in 'proline'.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This unsupervised analysis, laid out on a semi-circle for clarity,
   reveals the unique variability signature of each wine type. The
   fingerprints show a clear divergence in chemical consistency:

   * **Cultivar C (cyan)** has a profile dominated by extreme
     variability in **color_intensity**, which reaches a normalized
     score of 1.0. This suggests color is the least consistent, and
     therefore most defining, trait for this group.
   * **Cultivars A (blue) and B (brown)**, in contrast, are both most
     variable in **magnesium**. However, their overall shapes differ,
     with Cultivar A showing higher relative variability in ash-related
     properties compared to Cultivar B.

   This kind of analysis is invaluable for characterization and
   identifying which features make each group distinct.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 2: Supervised Fingerprint for Correlation Analysis**

Now, let's switch to a supervised problem. We want to understand what
drives the *quality* of a wine. We can use the function to compute the
absolute correlation of each feature with a target variable (`y_col`).

Let's simulate a scenario where the factors driving quality differ
between two vineyards. This is a common real-world problem where the
context (the vineyard) changes the feature importance landscape.

.. code-block:: python
   :linenos:

   # --- 1. Generate Synthetic Quality Data ---
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import kdiagram as kd

   # Reuse df from Use Case 1 (already has features + wine_type)
   # If running standalone, rebuild df with load_wine() as above.

   np.random.seed(42)

   # --- 1) Create vineyard context
   df["vineyard"] = np.random.choice(["Hillside", "Valley"], size=len(df), p=[0.5, 0.5])
   hillside = df["vineyard"] == "Hillside"
   valley   = ~hillside

   # --- 2) Build a full-length, index-aligned quality Series
   quality = pd.Series(0.0, index=df.index)

   # Hillside: alcohol & flavanoids drive quality
   quality.loc[hillside] += (
       1.2 * df.loc[hillside, "alcohol"]
       + 2.0 * df.loc[hillside, "flavanoids"]
   )

   # Valley: proline & color_intensity drive quality
   quality.loc[valley] += (
       0.005 * df.loc[valley, "proline"]      # rescale proline so it isn’t dominating
       + 1.5   * df.loc[valley, "color_intensity"]
   )

   # Add modest noise everywhere
   quality += np.random.normal(0, 0.5, size=len(df))

   df["quality_score"] = quality

   # --- 3) Choose a compact, interpretable feature set
   drivers = ["alcohol", "flavanoids", "proline", "color_intensity"]
   # Add a few supporting axes with high overall variance to improve context
   extra = (
       df.drop(columns=["wine_type", "vineyard", "quality_score"])
         .std()
         .sort_values(ascending=False)
         .index.difference(drivers)
         .tolist()[:4]
   )
   features_to_show = drivers + extra   # 8 axes total

   # --- 4) Plot absolute correlation per vineyard
   kd.plot_fingerprint(
       df,
       precomputed=False,
       y_col="quality_score",
       group_col="vineyard",
       method="abs_corr",                 # |corr(y, x)| per group
       features=features_to_show,
       normalize=True,
       acov="full",                       # full-circle works nicely here
       title="Quality Driver Fingerprints by Vineyard",
       # savefig="gallery/images/plot_fingerprint_correlation.png",
   )
   plt.close()


.. figure:: ../images/feature_based/plot_fingerprint_correlation.png
   :align: center
   :width: 75%
   :alt: A full radar chart comparing feature correlations for two vineyards.

   The plot shows that for the 'Hillside' vineyard, 'flavanoids' and
   'alcohol' are most correlated with quality, while for the 'Valley'
   vineyard, it's 'proline' and 'color_intensity'.

.. topic:: 💡 Interpretation
   :class: hint

   The plot immediately reveals a story about "terroir"—how
   the vineyard's location fundamentally changes the formula for a
   high-quality wine. The two fingerprints are nearly inverted.
   The **Hillside** vineyard's fingerprint (blue) is sharply peaked,
   showing that its quality is overwhelmingly correlated with
   **alcohol** and, to a lesser extent, **flavanoids**. In stark contrast, 
   the **Valley** vineyard (cyan) relies on a completely different set 
   of drivers. Its quality is most strongly correlated with **color_intensity** 
   and **proline**, while alcohol and flavanoids are of minor importance.

   This critical insight shows there is no single path to quality;
   optimal harvesting and blending strategies must be tailored to each
   vineyard's unique fingerprint.

.. admonition:: Best Practice
   :class: hint

   * **Method Selection**: Use unsupervised methods (``'std'``, ``'var'``)
     for data characterization and supervised methods (``'abs_corr'``)
     when you have a clear prediction target.
   * **Normalization**: Keep ``normalize=True`` (the default) when you
     care about the *relative* pattern of importances within each
     group. This answers: "What is the most important feature *for this
     group*?"
   * **Angular Coverage**: The default ``acov="half_circle"`` is often
     excellent for readability, especially with many features, as it
     prevents labels from overlapping at the top and bottom. Use
     ``"full"`` when a circular metaphor is more intuitive.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For more details on the statistical calculations, please see the
main User Guide section on :ref:`ug_plot_fingerprint`.

.. _gallery_plot_feature_interaction:

---------------------------
Polar Feature Interaction
---------------------------

The :func:`~kdiagram.plot.feature_based.plot_feature_interaction`
function is a powerful diagnostic tool for visualizing the joint effect
of two features on a target variable. By mapping these interactions
onto a polar heatmap, it excels at revealing complex, non-linear
relationships and conditional patterns that are often missed by
traditional 1D or 2D Cartesian plots.

First, let's break down the components of this insightful plot.

.. admonition:: Plot Anatomy
   :class: anatomy

   * **Angle (θ):** Represents the first independent feature. This axis
     is ideal for cyclical data (e.g., 'hour of day', 'month of year'),
     where the start and end points connect seamlessly.
   * **Radius (r):** Represents the second independent feature, plotted
     concentrically. The lowest value is at the center, and the
     highest is at the periphery.
   * **Color:** Represents the aggregated value of the dependent
     (target) variable for all data points falling within a specific
     angle-radius bin. The aggregation statistic (e.g., 'mean' or
     'std') can be specified.

With this framework, we can explore how seemingly independent features
can conspire to influence an outcome.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">


**Use Case 1: Comparing Modes — Basic (Heatmap) vs. Annular (Wedges)**

A classic application is modeling solar panel energy output. The output
is not determined by the hour or cloud cover alone, but by their strong
interaction. High output is only possible during daylight hours *and* when 
cloud cover is low. These plots make that relationship immediately obvious.
We can visualize this comparison using two different modes:
the default heatmap (``mode='basic'``) and the discrete wedge view
(``mode='annular'``).

The :func:`~kdiagram.plot.feature_based.plot_feature_interaction`
function integrates directly with Matplotlib, allowing us to pass an
``ax`` object to place them side-by-side on a subplot.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation ---
   np.random.seed(0)
   n_points = 5000
   hour_of_day = np.random.uniform(0, 24, n_points)
   cloud_cover = np.random.rand(n_points)

   # Target depends on the interaction between daylight and cloud cover
   daylight = np.sin(hour_of_day * np.pi / 24) ** 2
   cloud_factor = (1 - cloud_cover ** 0.5)
   output = 100 * daylight * cloud_factor + np.random.rand(n_points) * 5
   output[(hour_of_day < 6) | (hour_of_day > 18)] = 0 # No output at night

   df_solar = pd.DataFrame({
       'hour': hour_of_day,
       'cloud_cover': cloud_cover,
       'panel_output': output,
   })

   # --- Create a 1x2 Subplot Figure ---
   # Note: We must use subplot_kw to create polar axes
   fig, (ax1, ax2) = plt.subplots(
       1, 2,
       figsize=(16, 8),
       subplot_kw={'projection': 'polar'}
   )
   fig.suptitle('Solar Panel Output: Basic vs. Annular Mode', fontsize=18, y=1.05)

   # --- Plot 1: Basic (default heatmap) ---
   kd.plot_feature_interaction(
       df=df_solar,
       theta_col='hour',
       r_col='cloud_cover',
       color_col='panel_output',
       theta_period=24,
       theta_bins=24,
       r_bins=8,
       cmap='inferno',
       title='(a) Basic Mode (Heatmap)',
       ax=ax1  # Pass the first axis
   )

   # --- Plot 2: Annular (wedges) with Custom Ticks ---
   kd.plot_feature_interaction(
       df=df_solar,
       theta_col='hour',
       r_col='cloud_cover',
       color_col='panel_output',
       theta_period=24,
       theta_bins=24,
       r_bins=8,
       cmap='inferno',
       mode="annular",  # Use curved wedges
       title='(b) Annular Mode (Wedges)',
       # --- Custom, human-readable ticks ---
       theta_ticks=[0, 6, 12, 18],
       theta_ticklabels={0: "Midnight", 6: "6 AM", 12: "Noon", 18: "6 PM"},
       r_ticks=[0, 0.5, 1.0],
       r_ticklabels={0: "Clear Sky", 0.5: "Partial", 1.0: "Overcast"},
       ax=ax2  # Pass the second axis
   )

   # --- Save the combined figure ---
   #plt.tight_layout(pad=3.0)
   kd.savefig('gallery/images/plot_feature_interaction_solar_comparison.png')
   plt.close(fig)

.. figure:: ../images/feature_based/plot_feature_interaction_solar_comparison.png
   :align: center
   :width: 95%
   :alt: Side-by-side comparison of basic and annular polar heatmaps.

   A comparison of the (a) basic heatmap and (b) annular wedge plot
   for the same solar panel data.

.. topic:: 🧠 Analysis and Interpretation
   :class: hint

   This side-by-side comparison highlights the strengths of each mode.
   Both plots tell the same core story: a clear day/night divide (no
   output from "6 PM" to "6 AM") and a "hot spot" of peak output
   (bright yellow) centered at "Noon" and "Clear Sky" (the innermost
   ring).

   * **Plot (a) Basic Mode:** This default mode uses a `pcolormesh`,
     which creates a **smooth, interpolated heatmap**. The colors
     blend between bins, which is excellent for visualizing gradual
     transitions and the overall "shape" of the data gradient.
     However, it relies on default angular ticks (0°, 90°, etc.),
     which require mental translation (e.g., 90° is 6 AM).

     The plot presents a striking visual narrative of solar energy
     generation. The most immediate feature is the stark day/night
     divide, with the entire right hemisphere of the plot rendered in
     black, confirming zero output between 6 PM and 6 AM regardless of
     cloud conditions.

     The "hot spot" of peak performance—a bright yellow core—is precisely
     located at an angle of 180° (representing noon) and at the plot's
     center (representing minimal cloud cover). From this peak, the
     power output decays along two clear gradients:
   
     1. **Radially:** Moving outwards along the 180° line shows output
        fading from yellow to purple, illustrating how increasing cloud
        cover diminishes power, even at the sun's zenith.
     2. **Angularly:** Following any concentric circle away from 180°
        shows the color darkening, representing the natural decline in
        solar intensity as the day progresses from noon towards dusk or
        dawn.
      
   * **Plot (b) Annular Mode:** This mode draws each bin as a
     **discrete, hard-edged wedge**. This provides a clearer,
     segmented view that emphasizes the binned nature of the
     aggregation. Its true power is revealed when combined with
     **custom tick labels**. The axes are no longer abstract angles
     and radii but are labeled with intuitive, domain-specific
     terms: "Noon", "6 PM", "Clear Sky", and "Overcast".

   **Conclusion:** Use the **basic mode** for a smooth overview of
   gradients. Use the **annular mode** when you want to emphasize the
   discrete bins or when using custom tick labels to create a
   highly readable, presentation-ready figure for a general audience.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">
   

**Use Case 2: Identifying Market Volatility**

Beyond simple averages, this plot can visualize higher-order moments
like standard deviation to uncover volatility. Consider a financial
dataset where we want to understand stock price volatility based on the
time of day and a real-time market sentiment score. Here, we set
``statistic='std'`` to find combinations of time and sentiment that
lead to the most unpredictable pricing.

.. code-block:: python
   :linenos:

   # --- Data Generation for Market Volatility ---
   np.random.seed(42)
   n_trades = 10000
   trade_hour = np.random.uniform(9.5, 16, n_trades) # Trading hours
   sentiment = np.random.uniform(-1, 1, n_trades)   # Sentiment score

   # Volatility is highest at market open/close and during high sentiment
   time_vol = 1 / ((trade_hour - 12.75)**2 + 0.5)
   senti_vol = (sentiment + 1.1)**2
   price_change = np.random.randn(n_trades) * time_vol * senti_vol

   df_market = pd.DataFrame({
       'hour': trade_hour,
       'sentiment_score': sentiment,
       'price_change_abs': np.abs(price_change)
   })

   # --- Plotting Volatility ---
   kd.plot_feature_interaction(
       df=df_market,
       theta_col='hour',
       r_col='sentiment_score',
       color_col='price_change_abs',
       statistic='std', # Visualize standard deviation
       theta_period=24,
       theta_bins=16,
       r_bins=10,
       cmap='plasma',
       title='Market Price Volatility by Hour and Sentiment',
       savefig='gallery/images/plot_feature_interaction_volatility.png',
   )
   plt.close()

.. figure:: ../images/feature_based/plot_feature_interaction_volatility.png
   :align: center
   :width: 75%
   :alt: Polar plot showing market volatility by hour and sentiment.

   Volatility (bright colors) is highest at market open/close and when
   sentiment is most positive (outermost ring).

.. topic:: 🧠 Interpretation
   :class: hint

   This visualization uncovers the precise conditions that trigger
   market instability. The plot is dominated by a vast, calm sea of
   deep blue in the center, indicating that mid-day trading with
   neutral sentiment is highly predictable.

   However, two distinct "horns" of high volatility, colored bright
   yellow, erupt at the market's open (~140° or 9:30 AM) and close
   (~240° or 4:00 PM). The plot reveals a critical interaction: this
   instability is most extreme at the outer radius, meaning that high
   positive sentiment dramatically amplifies the volatility inherent
   at the start and end of the trading day. This insight allows traders
   to pinpoint the riskiest conditions: not just *when* to be cautious,
   but under *what market sentiment* that caution is most warranted.

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

**Use Case 3: Annular Mode & Custom Domain Ticks**

The "annular" mode renders each bin as a
distinct curved wedge, which can be visually clearer than the default
heatmap. More importantly, we use ``theta_ticks``,
``theta_ticklabels``, ``r_ticks``, and ``r_ticklabels`` to map the raw
data values (like ``hour=9.5`` or ``sentiment=-1.0``) to
human-readable, domain-specific labels (like "Open 9:30" or
"Bearish"). This makes the plot self-explanatory.

.. code-block:: python
   :linenos:

   import kdiagram as kd
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # --- Data Generation for Market Volatility ---
   np.random.seed(42)
   n_trades = 10000
   trade_hour = np.random.uniform(9.5, 16, n_trades) # Trading hours
   sentiment = np.random.uniform(-1, 1, n_trades)   # Sentiment score

   # Volatility is highest at market open/close and during high sentiment
   time_vol = 1 / ((trade_hour - 12.75)**2 + 0.5)
   senti_vol = (sentiment + 1.1)**2
   price_change = np.random.randn(n_trades) * time_vol * senti_vol

   df_market = pd.DataFrame({
       'hour': trade_hour,
       'sentiment_score': sentiment,
       'price_change_abs': np.abs(price_change)
   })

   # --- Plotting Volatility with Annular Mode & Custom Ticks ---
   kd.plot_feature_interaction(
       df=df_market,
       theta_col='hour',
       r_col='sentiment_score',
       color_col='price_change_abs',
       statistic='std', # Visualize standard deviation
       theta_period=24, # Use 24 to scale hours correctly
       theta_bins=16,
       r_bins=10,
       acov='half_circle', # Focus on the trading day
       cmap='plasma',
       title='Market Price Volatility by Hour and Sentiment',
       mode="annular",  # Use curved wedges
       theta_ticks=[9.5, 12.0, 16.0],
       theta_ticklabels={9.5: "Open 9:30", 12.0: "Noon", 16.0: "Close 16:00"},
       theta_tick_step=1.0, # 1 unit in your theta data space
       r_ticks=[-1, -0.5, 0, 0.5, 1],
       r_ticklabels={-1:"Bearish", 0:"Neutral", 1:"Bullish"},
       savefig='gallery/images/plot_feature_interaction_volatility_mode_annular.png',
   )
   plt.close()

.. figure:: ../images/feature_based/plot_feature_interaction_volatility_mode_annular.png
   :align: center
   :width: 75%
   :alt: Annular polar plot with custom labels for market hours and sentiment.

   The plot uses mode="annular" for clear bins and custom tick labels
   like "Open 9:30", "Noon", "Bearish", and "Bullish" for readability.

.. topic:: 🧠 Interpretation
   :class: hint

   This visualization is far more intuitive for a non-technical
   audience. The ``mode="annular"`` renders bins as discrete sectors,
   avoiding the interpolation of the default heatmap.

   The key improvement comes from the custom tick labels. Instead of
   interpreting ``theta=9.5``, the analyst immediately sees "Open 9:30".
   Similarly, the radius is clearly marked "Bearish", "Neutral", and
   "Bullish". The plot confirms the findings from Use Case 2: volatility
   (yellow) peaks at the market "Open" and "Close". It adds a new,
   clearer insight: this volatility is most pronounced when sentiment
   is "Bullish" (the outermost ring).

.. raw:: html

   <hr style="border-top: 1px solid #ccc; margin: 30px 0;">

   
**Use Case 4: Focused Analysis in Manufacturing**

Sometimes, a full 360° view is not necessary, especially when one
feature is not cyclical. We can use the ``acov`` (angular coverage)
parameter to create a sector plot for a more focused analysis.
Imagine a process where product defects are related to machine speed
and lubricant viscosity. We can map the linear viscosity scale to a
180° arc using ``acov='half_circle'``.

.. code-block:: python
   :linenos:

   # --- Data Generation for Manufacturing Defects ---
   np.random.seed(123)
   n_samples = 8000
   speed = np.random.uniform(100, 500, n_samples)      # Speed in RPM
   viscosity = np.random.uniform(20, 80, n_samples) # Viscosity in cSt

   # Defects occur primarily at high speeds with low viscosity
   defect_prob = 1 / (1 + np.exp(
       -0.02 * ((speed - 400) - (viscosity - 50) * 5)
   ))
   defects = np.random.binomial(1, defect_prob)

   df_qc = pd.DataFrame({
       'speed_rpm': speed, 'viscosity_cst': viscosity, 'is_defect': defects
   })

   # --- Plotting with Angular Coverage Control ---
   kd.plot_feature_interaction(
       df=df_qc,
       theta_col='viscosity_cst', # Non-cyclical feature
       r_col='speed_rpm',
       color_col='is_defect',
       statistic='mean',     # Mean of binary = defect rate
       acov='half_circle',   # Use a 180-degree view
       theta_bins=15,
       r_bins=10,
       cmap='cividis',
       title='Product Defect Rate by Speed and Viscosity',
       savefig='gallery/images/plot_feature_interaction_defects.png',
   )
   plt.close()

.. figure:: ../images/feature_based/plot_feature_interaction_defects.png
   :align: center
   :width: 75%
   :alt: Semi-circular plot for manufacturing defect analysis.

   The focused semi-circle plot pinpoints the highest defect rate
   (bright yellow) at high speeds and low-to-mid viscosity.

.. topic:: 🧠 Interpretation
   :class: hint

   The semi-circular plot acts as a diagnostic map, pinpointing a
   critical failure zone with high precision. The defect rate,
   represented by the mean of a binary outcome, escalates to nearly
   100% (bright yellow) in a specific operational window: when machine
   speeds are highest (the outermost rings, >400 RPM) **and** when
   lubricant viscosity is in the low-to-mid range (an angular sector
   between roughly 30° and 60°).

   Conversely, the plot clearly defines "safe zones." The deep blue
   inner rings indicate that low speeds (<200 RPM) are consistently
   safe, irrespective of viscosity. Furthermore, high viscosity (angles
   approaching 180°) appears to mitigate defect risk, even at high
   speeds. This provides an immediate, actionable insight for
   engineers: to eliminate defects, they must either reduce speed or
   significantly increase lubricant viscosity.

.. raw:: html

   <hr style="border-top: 2px solid #ccc; margin: 40px 0;">

For a deeper dive into the underlying mathematics of polar mapping and
binning, please refer to the main User Guide section on
:ref:`ug_plot_feature_interaction`.

.. raw:: html

   <hr>
