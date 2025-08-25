.. _api_reference:

===============
API Reference
===============

Welcome to the `k-diagram` API reference. This section provides detailed
information on the functions, classes, and modules included in the
package.

The documentation here is largely auto-generated from the docstrings
within the `k-diagram` source code. Ensure you have installed the
package (see :doc:`installation`) for the documentation build process
to find the modules correctly.

.. _api_plot_modules:

Plotting Functions (`kdiagram.plot`)
---------------------------------------

This is the core module containing the specialized visualization
functions, organized by their diagnostic purpose.

.. _api_uncertainty:

Uncertainty Visualization (`kdiagram.plot.uncertainty`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions focused on visualizing prediction intervals, coverage,
anomalies, drift, and other uncertainty-related diagnostics.

.. autosummary::
   :toctree: _autosummary/uncertainty
   :nosignatures:

   ~kdiagram.plot.uncertainty.plot_actual_vs_predicted
   ~kdiagram.plot.uncertainty.plot_anomaly_magnitude
   ~kdiagram.plot.uncertainty.plot_coverage
   ~kdiagram.plot.uncertainty.plot_coverage_diagnostic
   ~kdiagram.plot.uncertainty.plot_interval_consistency
   ~kdiagram.plot.uncertainty.plot_interval_width
   ~kdiagram.plot.uncertainty.plot_model_drift
   ~kdiagram.plot.uncertainty.plot_temporal_uncertainty
   ~kdiagram.plot.uncertainty.plot_uncertainty_drift
   ~kdiagram.plot.uncertainty.plot_velocity
   ~kdiagram.plot.uncertainty.plot_radial_density_ring
   ~kdiagram.plot.uncertainty.plot_polar_heatmap
   ~kdiagram.plot.uncertainty.plot_polar_quiver

.. _api_errors:

Error Analysis (`kdiagram.plot.errors`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for diagnosing and visualizing model errors, focusing on
systemic vs. random errors, comparing error distributions, and
visualizing 2D uncertainty.

.. autosummary::
   :toctree: _autosummary/errors
   :nosignatures:

   ~kdiagram.plot.errors.plot_error_bands
   ~kdiagram.plot.errors.plot_error_violins
   ~kdiagram.plot.errors.plot_error_ellipses

.. _api_probabilistic:

Probabilistic Diagnostics (`kdiagram.plot.probabilistic`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for the in-depth evaluation of probabilistic forecasts,
assessing calibration, sharpness, and overall performance.

.. autosummary::
   :toctree: _autosummary/probabilistic
   :nosignatures:

   ~kdiagram.plot.probabilistic.plot_pit_histogram
   ~kdiagram.plot.probabilistic.plot_polar_sharpness
   ~kdiagram.plot.probabilistic.plot_crps_comparison
   ~kdiagram.plot.probabilistic.plot_credibility_bands
   ~kdiagram.plot.probabilistic.plot_calibration_sharpness

.. _api_comparison:

Model Comparison (`kdiagram.plot.comparison`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for comparing multi-model performances using radar charts
and reliability diagrams.

.. autosummary::
   :toctree: _autosummary/comparison
   :nosignatures:

   ~kdiagram.plot.comparison.plot_model_comparison
   ~kdiagram.plot.comparison.plot_reliability_diagram
   ~kdiagram.plot.comparison.plot_polar_reliability
   ~kdiagram.plot.comparison.plot_horizon_metrics

.. _api_relationship:

Relationship Visualization (`kdiagram.plot.relationship`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for visualizing relationships between true values,
predictions, and errors.

.. autosummary::
   :toctree: _autosummary/relationship
   :nosignatures:

   ~kdiagram.plot.relationship.plot_relationship
   ~kdiagram.plot.relationship.plot_conditional_quantiles
   ~kdiagram.plot.relationship.plot_error_relationship
   ~kdiagram.plot.relationship.plot_residual_relationship

.. _api_feature_based:

Feature-Based Visualization (`kdiagram.plot.feature_based`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for visualizing feature importance and influence patterns.

.. autosummary::
   :toctree: _autosummary/feature_based
   :nosignatures:

   ~kdiagram.plot.feature_based.plot_feature_fingerprint
   ~kdiagram.plot.feature_based.plot_feature_interaction

.. _api_context:

Contextual Diagnostics (`kdiagram.plot.context`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard Cartesian plots that provide essential context for the
main polar diagrams, covering time series, correlation, and error
distribution analysis.

.. autosummary::
   :toctree: _autosummary/context
   :nosignatures:

   ~kdiagram.plot.context.plot_time_series
   ~kdiagram.plot.context.plot_scatter_correlation
   ~kdiagram.plot.context.plot_error_distribution
   ~kdiagram.plot.context.plot_qq
   ~kdiagram.plot.context.plot_error_autocorrelation
   ~kdiagram.plot.context.plot_error_pacf

.. _api_evaluation:

Model Evaluation (`kdiagram.plot.evaluation`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for evaluating model performance, primarily using Taylor
Diagrams.

.. autosummary::
   :toctree: _autosummary/evaluation
   :nosignatures:

   ~kdiagram.plot.evaluation.taylor_diagram
   ~kdiagram.plot.evaluation.plot_taylor_diagram_in
   ~kdiagram.plot.evaluation.plot_taylor_diagram


.. _api_utils:

Utility Functions (`kdiagram.utils`)
--------------------------------------

Helper functions for data preparation, mathematical computations, and
validations.

.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   ~kdiagram.utils.bin_by_feature
   ~kdiagram.utils.build_cdf_interpolator
   ~kdiagram.utils.build_q_column_names
   ~kdiagram.utils.calculate_calibration_error
   ~kdiagram.utils.compute_coverage_score
   ~kdiagram.utils.compute_crps
   ~kdiagram.utils.compute_forecast_errors
   ~kdiagram.utils.compute_interval_width
   ~kdiagram.utils.compute_pinball_loss
   ~kdiagram.utils.compute_pit
   ~kdiagram.utils.compute_winkler_score
   ~kdiagram.utils.detect_quantiles_in
   ~kdiagram.utils.get_forecast_arrays
   ~kdiagram.utils.melt_q_data
   ~kdiagram.utils.pivot_forecasts_long
   ~kdiagram.utils.pivot_q_data
   ~kdiagram.utils.plot_hist_kde
   ~kdiagram.utils.reshape_quantile_data

.. _api_datasets:

Datasets (`kdiagram.datasets`)
--------------------------------

Functions for loading sample datasets and generating synthetic data
for examples and testing.

.. autosummary::
   :toctree: _autosummary/datasets
   :nosignatures:

   ~kdiagram.datasets.load_uncertainty_data
   ~kdiagram.datasets.load_zhongshan_subsidence
   ~kdiagram.datasets.make_cyclical_data
   ~kdiagram.datasets.make_fingerprint_data
   ~kdiagram.datasets.make_multi_model_quantile_data
   ~kdiagram.datasets.make_taylor_data
   ~kdiagram.datasets.make_uncertainty_data