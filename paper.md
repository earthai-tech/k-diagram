---
title: 'k-diagram: Rethinking Forecasting Uncertainty via Polar-based Visualization'
tags:
  - Python
  - forecasting
  - uncertainty quantification
  - visualization
  - model evaluation
  - polar plots
  - diagnostics
  - environmental science
authors:
  - name: Kouao Laurent Kouadio
    orcid: 0000-0001-7259-7254 
    affiliation: "1, 2" 

affiliations:
 - name: School of Geosciences and Info-Physics, Central South University, Changsha, Hunan, 410083, China
   index: 1
 - name: Hunan Key Laboratory of Nonferrous Resources and Geological Hazards Exploration, Changsha, Hunan, 410083, China
   index: 2
date: 10 May 2025 
bibliography: paper.bib
---

# Summary

`k-diagram` is an open-source Python package providing specialized
diagnostic plots based on polar coordinates for the comprehensive
evaluation of forecasting models, with a strong emphasis on understanding
and visualizing predictive uncertainty. It offers a suite of functions
('k-diagrams') to assess prediction interval coverage, consistency,
anomaly magnitude, model drift across horizons, feature importance
patterns, and standard evaluation metrics (e.g., via Taylor diagrams).
Designed to complement traditional evaluation methods, `k-diagram` aids
interpretability and facilitates deeper insights into forecast
reliability, particularly for complex spatiotemporal systems relevant
to environmental science and geohazard assessment.

# Statement of Need

Evaluating the performance of forecasting models, especially in high-stakes
domains like environmental hazard prediction (e.g., land subsidence,
flooding), requires moving beyond simple point-forecast accuracy metrics.
Understanding the structure, reliability, and limitations of a model's
predictive uncertainty is critical for risk assessment and informed
decision-making [@Murphy1993What]. However, standard visualization tools
often struggle to present multi-faceted uncertainty information
(e.g., interval width, coverage success/failure, temporal drift)
simultaneously and intuitively across many data points or locations.

Existing libraries provide excellent general plotting (e.g., `matplotlib`
[@Hunter:2007], `seaborn` [@Waskom2021]) and some model evaluation metrics
(e.g., `scikit-learn` [@scikit-learn]), but lack dedicated tools for the
specific diagnostic visualization of uncertainty characteristics in a way
that highlights patterns related to temporal progression or spatial
distribution inherent in many forecasting problems. This gap became evident
during research applying advanced deep learning models to forecast complex
phenomena like urban land subsidence in challenging environments such as
Nansha, China [@Liu2024]. While models could generate
probabilistic forecasts, interpreting the spatiotemporal behavior of the
predicted uncertainty remained difficult.

`k-diagram` addresses this need by introducing a suite of visualizations
primarily based on polar coordinates. This approach allows mapping different
dimensions of forecast performance and uncertainty onto angle and radius,
providing compact overviews and revealing patterns potentially obscured in
Cartesian plots. For example, mapping time or sample index to angle allows
visual inspection of consistency, drift, or anomaly clustering in a circular
layout. `k-diagram` aims to make the analysis of predictive uncertainty
more intuitive and actionable, treating uncertainty itself as a crucial signal
to be diagnosed.

# Functionality

`k-diagram` is implemented in Python (\autoref{fig1:worflow}), leveraging core scientific libraries
including `numpy` [@harris2020array], `pandas` [@mckinney-proc-scipy-2010;
@reback2020pandas], `matplotlib` [@Hunter:2007], `scipy` [@2020SciPy-NMeth],
and `scikit-learn` [@scikit-learn]. Its main features include:

* **Uncertainty Diagnostics:** A collection of polar plots under
    `kdiagram.plot.uncertainty` for visualizing:
    * Prediction interval coverage (point-wise diagnostics and overall scores).
    * Anomaly magnitude (severity and type of interval failures).
    * Interval width (magnitude and consistency over time/samples).
    * Model drift (average uncertainty increase over forecast horizons).
    * Uncertainty drift (evolution of uncertainty patterns using concentric rings).
    * Prediction velocity (rate of change of central forecasts).
* **Model Evaluation:** Functions under `kdiagram.plot.evaluation` for
    generating standard and enhanced Taylor Diagrams [@Taylor2001] to compare
    models based on correlation, standard deviation, and RMSD relative to a
    reference.
* **Feature Importance:** Radar charts under `kdiagram.plot.feature_based`
    for visualizing and comparing feature importance profiles ("fingerprints")
    across different models or contexts.
* **Relationship Visualization:** Polar scatter plots under
    `kdiagram.plot.relationship` mapping true values to angle and
    (normalized) predictions to radius.
* **Data Utilities:** Helper functions under `kdiagram.utils` for detecting,
    validating, and reshaping pandas DataFrames containing quantile-based
    forecast data, facilitating data preparation for the plotting functions.
* **Command-Line Interface (CLI):** A `k-diagram` command allowing users
    to generate key plots directly from CSV files without writing Python scripts.
* **Customization:** Plots offer various parameters for adjusting appearance,
    angular coverage (`acov`), colormaps, normalization, and labeling.

![Structure of the k-diagram, illustrating the identification of global features.\label{fig1:workflow}](docs/source/_static/paper_fig1.png)


### Related Work

The visualization methods implemented in `k-diagram` were developed alongside 
research applying deep learning models to complex environmental forecasting 
challenges. Specifically, these polar diagnostics were utilized to analyze 
and interpret the uncertainty associated with land subsidence predictions 
using an Extreme Temporal Fusion Transformer model in Zhongshan city, 
China. The results and application context of that specific study are detailed 
in a separate manuscript submitted to the International Journal of 
Forecasting [@kouadio_subsidence_2025]. This JOSS paper focuses 
specifically on the `k-diagram` software package itself – its design, 
functionality, and general applicability for visualizing forecast uncertainty. 

### Installation 

The latest stable release of `k-diagram` is available on the Python Package 
Index (PyPI) and can be installed using pip:

```bash
pip install k-diagram
````
Detailed installation instructions are available in the documentation.

### Documentation

Comprehensive documentation, including a User Guide explaining the concepts 
behind the plots, a gallery of examples, and a detailed API reference, 
is hosted on ReadTheDocs: [https://k-diagram.readthedocs.io/](https://k-diagram.readthedocs.io/).

### License

`k-diagram` is made available under the OSI-approved Apache License 2.0. The 
full license text can be found in the `LICENSE` file in the root of the repository.

# Example Usage

The following snippet demonstrates generating an Anomaly Magnitude plot (\autoref{fig2:example}),
identifying where actual values fall outside the 10th-90th percentile
prediction interval.

```python
import kdiagram as kd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Sample Data
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

# 2. Create the Anomaly Magnitude Plot
ax = kd.plot_anomaly_magnitude(
    df=df,
    actual_col='actual',
    q_cols=['q10', 'q90'], # Provide lower and upper bounds as a list
    title="Sample Anomaly Magnitude Plot",
    cbar=True,            # Show color bar indicating magnitude
    s=100,                # marker size adjusted 
    verbose=0             # Suppress console summary for brevity 

)
# In a script, use plt.show() or ax.figure.savefig("plot.png")
# plt.show() # Not needed if saving or in some environments
```
![Example k-diagram plot demonstrating [Specific Feature Shown, e.g., Anomaly Magnitude] for simulated data. Red points indicate overpredictions, blue points under-predictions,relative to the interval bounds. Radius indicates the magnitude of the anomaly.\label{fig2:example}](docs/source/images/paper_fig2.png)

### Contributing

Contributions to `k-diagram` are highly encouraged\! We welcome bug reports, 
feature requests, documentation improvements, and code contributions. 
Please refer to the `CONTRIBUTING.rst` guide in the documentation (or repository) 
for detailed guidelines on how to contribute. Development discussions and 
issue tracking occur on the [GitHub repository](https://github.com/earthai-tech/k-diagram/issues).


### Acknowledgements

We extend our sincere gratitude to the anonymous colleagues who provided 
invaluable feedback during the development of `k-diagram`, as well as those 
who rigorously tested its early iterations. Their insights and dedication 
were instrumental in refining the software.

We also appreciate the constructive feedback from early users and 
reviewers, whose contributions have significantly enhanced the quality 
and usability of the project.


### References

