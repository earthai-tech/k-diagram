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

.. _api_plot_uncertainty: 

Plotting Functions (`kdiagram.plot`)
---------------------------------------

This is the core module containing the specialized visualization
functions.

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

.. _api_feature_based: 

Feature-Based Visualization (`kdiagram.plot.feature_based`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for visualizing feature importance and influence patterns.

.. autosummary::
   :toctree: _autosummary/feature_based
   :nosignatures:

   ~kdiagram.plot.feature_based.plot_feature_fingerprint

.. _api_relationship: 

Relationship Visualization (`kdiagram.plot.relationship`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for visualizing the relationship between true and predicted
values using polar coordinates.

.. autosummary::
   :toctree: _autosummary/relationship
   :nosignatures:

   ~kdiagram.plot.relationship.plot_relationship


.. _api_utils:

Utility Functions (`kdiagram.utils`)
--------------------------------------

Helper functions primarily focused on detecting, validating, and
manipulating quantile-related data within pandas DataFrames, often
used for preparing data for visualization functions.

.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   ~kdiagram.utils.build_q_column_names
   ~kdiagram.utils.detect_quantiles_in
   ~kdiagram.utils.melt_q_data
   ~kdiagram.utils.pivot_q_data   
   ~kdiagram.utils.reshape_quantile_data



