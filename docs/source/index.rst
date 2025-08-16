.. k-diagram documentation master file, created by
   sphinx-quickstart on Thu Apr 10 12:44:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. k-diagram documentation master file

#############################################
k-diagram: Polar Insights for Forecasting
#############################################

.. card:: **Navigate the complexities of forecast uncertainty and model behavior with specialized polar visualizations.**
    :margin: 0 0 1 0

    Welcome to the official documentation for `k-diagram`. This package
    provides a unique perspective on evaluating forecasting models,
    especially when uncertainty quantification is crucial. Dive in to
    discover how polar plots can reveal deeper insights into your
    model's performance, stability, and potential weaknesses.

.. container:: text-center

    .. button-ref:: installation
        :color: primary
        :expand:
        :outline:

        Install k-diagram

    .. button-ref:: quickstart
        :color: secondary
        :expand:

        Quick Start Guide

    .. button-ref:: gallery/index     
        :color: secondary
        :expand:
        
        Plot Gallery
        
.. card:: Key Features
    :class-header: text-center sd-font-weight-bold
    :margin: auto  

    * **Intuitive Polar Perspective:** Visualize multi-dimensional
      aspects like uncertainty spread, temporal drift, and spatial
      patterns in a compact circular layout.
    * **Advanced Error Analysis:** A dedicated suite of plots to
      diagnose systemic bias vs. random error, compare error
      distributions, and visualize 2D uncertainty.
    * **Targeted Diagnostics:** Functions specifically designed to
      assess interval coverage, consistency, anomaly magnitude, model
      velocity, and drift.
    * **Uncertainty-Aware Evaluation:** Move beyond point-forecast
      accuracy and evaluate the reliability of your model's
      uncertainty estimates.
    * **Identify Model Weaknesses:** Pinpoint where and when your
      forecasts are less reliable or exhibit significant anomalies.
    * **Clear Communication:** Generate publication-ready plots to
      effectively communicate model performance and uncertainty
      characteristics.

.. # Table of Contents Tree (often hidden and rendered by the theme's sidebar)
.. # The 'hidden' option prevents it from being displayed directly here.
.. # The theme (like Furo or PyData) uses this to build the sidebar navigation.

.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents:
   :hidden:

   installation
   quickstart
   motivation
   user_guide/index      
   cli_usage
   gallery/index         
   api                   
   contributing
   code_of_conduct
   citing
   release_notes/index       
   glossary
   license


.. rubric:: See Also

Quick links to the main sections of the API Reference:

* :ref:`Uncertainty Visualization <api_uncertainty>`: Functions for
  analyzing prediction intervals, coverage, anomalies, and drift.
* :ref:`Error Visualization <api_errors>`: Functions for diagnosing
  systemic bias, variance, and comparing error distributions.
* :ref:`Model Evaluation <api_evaluation>`: Functions for generating
  Taylor Diagrams to compare model performance.
* :ref:`Feature Importance <api_feature_based>`: Functions for
  visualizing feature influence patterns (fingerprints).
* :ref:`Relationship Visualization <api_relationship>`: Functions for
  plotting true vs. predicted values in polar coordinates.
* Full :doc:`API Reference <api>`: Browse the complete API documentation.

