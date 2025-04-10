.. _release_notes:

===============
Release Notes
===============

This document tracks the changes, new features, and bug fixes for
each release of the `k-diagram` package.

----------------
Version 1.0.0
----------------
*(Released: 2025-04-10)*

Initial Release
~~~~~~~~~~~~~~~~~

This is the first public release of the `k-diagram` package.

**Key Features Included:**

* **Uncertainty Visualization Suite (`kdiagram.plot.uncertainty`):**
    * :func:`~kdiagram.plot.uncertainty.plot_actual_vs_predicted`:
        Compare actual vs. point predictions.
    * :func:`~kdiagram.plot.uncertainty.plot_anomaly_magnitude`:
        Visualize magnitude and type of prediction interval failures.
    * :func:`~kdiagram.plot.uncertainty.plot_coverage`: Calculate
        and plot overall coverage scores (bar, line, pie, radar).
    * :func:`~kdiagram.plot.uncertainty.plot_coverage_diagnostic`:
        Diagnose point-wise interval coverage on a polar plot.
    * :func:`~kdiagram.plot.uncertainty.plot_interval_consistency`:
        Assess stability of interval width over time (Std Dev / CV).
    * :func:`~kdiagram.plot.uncertainty.plot_interval_width`:
        Visualize prediction interval width magnitude across samples.
    * :func:`~kdiagram.plot.uncertainty.plot_model_drift`: Track
        average interval width drift across forecast horizons (polar bars).
    * :func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty`:
        General polar scatter for comparing multiple series (e.g., quantiles).
    * :func:`~kdiagram.plot.uncertainty.plot_uncertainty_drift`:
        Visualize drift of uncertainty patterns using concentric rings.
    * :func:`~kdiagram.plot.uncertainty.plot_velocity`: Visualize
        rate of change (velocity) of median predictions.
* **Model Evaluation (`kdiagram.plot.evaluation`):**
    * Taylor Diagram functions (:func:`~kdiagram.plot.evaluation.taylor_diagram`,
        :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`,
        :func:`~kdiagram.plot.evaluation.plot_taylor_diagram`) for
        summarizing model skill (correlation, standard deviation, RMSD).
* **Feature Importance (`kdiagram.plot.feature_based`):**
    * :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`:
        Radar charts for comparing feature importance profiles.
* **Relationship Visualization (`kdiagram.plot.relationship`):**
    * :func:`~kdiagram.plot.relationship.plot_relationship`: Polar
        scatter mapping true values to angle and predictions to radius.
* **Utility Functions (`kdiagram.utils`):**
    * Helpers for detecting, building names for, and reshaping quantile
        data in DataFrames (:func:`~kdiagram.utils.detect_quantiles_in`,
        :func:`~kdiagram.utils.build_q_column_names`,
        :func:`~kdiagram.utils.reshape_quantile_data`,
        :func:`~kdiagram.utils.melt_q_data`,
        :func:`~kdiagram.utils.pivot_q_data`).
* **Command-Line Interface (CLI):**
    * `k-diagram` command for generating core plots directly from CSV
        files via the terminal.
* **Documentation:**
    * Initial version including Installation Guide, Quick Start, User
        Guide (concepts & interpretation), Plot Gallery, Utility Examples,
        API Reference, Contribution Guidelines, and License.