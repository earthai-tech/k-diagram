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
    :margin: 1 0 2 0

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
        :text: Plot Gallery      <-- Updated button target (optional but good)
        :color: secondary
        :expand:

.. panels::
    :header: text-center
    :column: col-lg-12 p-0

    ---
    :header: **Overview (from README)**
    :body:

    .. include:: ../../README.md
       :parser: myst_parser.sphinx_
       :start-after: ### ✨ Why k-diagram?
       :end-before: ---

       *(Note: This includes the main description from your README.md. Ensure
       your README is up-to-date and located correctly relative to this
       file, typically two levels up if index.rst is in docs/source/)*

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
   citing
   release_notes       
   glossary
   license


.. rubric:: See Also

Quick links to the main sections of the API Reference:

* :ref:`Uncertainty Visualization <api_uncertainty>`: Functions for
    analyzing prediction intervals, coverage, anomalies, and drift.
* :ref:`Model Evaluation <api_evaluation>`: Functions for generating
    Taylor Diagrams to compare model performance.
* :ref:`Feature Importance <api_feature_based>`: Functions for
    visualizing feature influence patterns (fingerprints).
* :ref:`Relationship Visualization <api_relationship>`: Functions for
    plotting true vs. predicted values in polar coordinates.
* Full :doc:`API Reference <api>`: Browse the complete API documentation.

