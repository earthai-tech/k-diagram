.. k-diagram documentation master file

#########################################
k-diagram: Polar Insights for Forecasting
#########################################

.. card::
   :class-card: sd-border-0 sd-shadow-none sd-p-0 hero-image-card
   :margin: 0

   .. image:: /_static/hero_image.png
      :alt: A collage of beautiful polar plots from k-diagram
      :align: center
      :width: 100%

.. card:: **Navigate the complexities of forecast uncertainty and model behavior with specialized polar visualizations.**
   :class-card: sd-border-0 sd-shadow-none
   :text-align: center
   :margin: 4 0 1 0

   Welcome to `k-diagram`! This package provides a unique perspective
   on evaluating forecasting models by leveraging the power of polar
   coordinates. Move beyond standard metrics and discover how circular
   plots can reveal deeper insights into your model's performance,
   stability, and hidden weaknesses.

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

.. raw:: html

   <p align="center" style="margin-top: 2em; margin-bottom: 1.5em;">
     <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/k-diagram">
     <a href="https://github.com/earthai-tech/k-diagram/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/earthai-tech/k-diagram?style=flat-square&logo=apache&color=purple"></a>
     <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/k-diagram">
     <a href="https://github.com/earthai-tech/k-diagram/actions/workflows/python-package-conda.yml"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/earthai-tech/k-diagram/python-package-conda.yml?branch=main&style=flat-square"></a>
   </p>

.. raw:: html

   <hr class="hr-spacer">


.. card:: Who is this for?
   :class-header: text-center sd-font-weight-bold
   :margin: 4 0 1 0

   **Ideal for:** Data scientists, machine learning engineers,
   meteorologists, climate scientists, and researchers who need to
   diagnose and communicate the performance of complex forecasting
   models, especially when uncertainty is a key factor.

.. raw:: html

   <hr class="hr-spacer">
    
.. container:: cta-tiles

   .. grid:: 1 2 2 2
      :gutter: 3

      .. grid-item-card:: 📘 User Guide
         :link: /user_guide/index
         :link-type: doc
         :class-card: sd-shadow-sm sd-rounded-lg

         
         Start here to learn the core concepts and mathematical
         foundations behind the plots.

      .. grid-item-card:: 🖼️ Plot Gallery
         :link: /gallery/index
         :link-type: doc
         :class-card: sd-shadow-sm sd-rounded-lg

         
         Browse every plot type with runnable code
         and interpretation guides.

      .. grid-item-card:: 💻 CLI Reference
         :link: /cli/index
         :link-type: doc
         :class-card: sd-shadow-sm sd-rounded-lg

         
         Generate plots from your terminal. See the full
         list of commands and options.

      .. grid-item-card:: 📝 Release Notes
         :link: /release_notes/index
         :link-type: doc
         :class-card: sd-shadow-sm sd-rounded-lg sparkle-link

         
         .. raw:: html

            <span class="new-badge-card" data-release-date="2025-08-29">NEW</span>

         Check out the latest features and fixes.


.. topic:: Key Features
   :class: sd-rounded-lg sd-shadow-sm

   - 🔭 **Intuitive Polar Perspective:** Visualize multi-dimensional aspects 
     like uncertainty spread, temporal drift, and spatial patterns in a 
     compact circular layout.
   - 🧮 **Advanced Error Analysis:** A dedicated suite of plots to diagnose 
     systemic bias vs. random error, compare error distributions, and 
     visualize 2D uncertainty.
   - 🧭 **Targeted Diagnostics:** Functions specifically designed to assess 
     interval coverage, consistency, anomaly magnitude, model velocity, and drift.
   - 🎲 **Uncertainty-Aware Evaluation:** Move beyond point-forecast 
     accuracy and evaluate the reliability of your model’s uncertainty estimates.
   - 🧩 **Identify Model Weaknesses:** Pinpoint where and when your 
     forecasts are less reliable or exhibit significant anomalies.
   - 📣 **Clear Communication:** Generate publication-ready plots to 
     effectively communicate model performance and uncertainty characteristics.


      
.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents:
   :hidden:

   installation
   quickstart
   motivation
   user_guide/index
   cli/index
   gallery/index
   api
   contributing
   code_of_conduct
   citing
   release_notes/index
   development
   license
   glossary
   references
   

.. raw:: html

   <hr class="hr-spacer">

.. container:: see-also-tiles

   .. grid:: 1 2 2 2
      :gutter: 3

      .. grid-item-card:: 🔍 Uncertainty Visualization
         :link-type: ref
         :link: api_uncertainty
         :class-card: sd-rounded-lg sd-shadow-sm seealso-card card--uncertainty

         Analyze prediction intervals, coverage, anomalies, and drift.

      .. grid-item-card:: 📉 Error Visualization
         :link-type: ref
         :link: api_errors
         :class-card: sd-rounded-lg sd-shadow-sm seealso-card card--errors

         Diagnose bias vs. variance, compare error distributions,
         visualize 2D uncertainty.

      .. grid-item-card:: 📊 Model Evaluation
         :link-type: ref
         :link: api_evaluation
         :class-card: sd-rounded-lg sd-shadow-sm seealso-card card--evaluation

         Generate Taylor diagrams and other evaluation views.

      .. grid-item-card:: 🧠 Feature Interaction
         :link-type: ref
         :link: api_feature_based
         :class-card: sd-rounded-lg sd-shadow-sm seealso-card card--importance

         Visualize feature influence patterns (fingerprints).

      .. grid-item-card:: 🔗 Relationship Visualization
         :link-type: ref
         :link: api_relationship
         :class-card: sd-rounded-lg sd-shadow-sm seealso-card card--relationship

         Plot true vs. predicted values in polar coordinates.


      .. grid-item-card:: 📚 Full API Reference
         :link: api
         :link-type: doc
         :class-card: sd-shadow-sm see-also-card see-also-accent

         Browse every function, class, and parameter.
