.. _motivation:

============================
Motivation and Background
============================

This page outlines the scientific context and practical motivations
that led to the development of the ``k-diagram`` package.

The Challenge of Modern Forecasting
--------------------------------------

In an era of increasingly complex systems and high-dimensional data,
the demands on forecasting have grown immensely. Fields ranging from
energy and finance to climatology and supply chain management now rely
on sophisticated models, including deep learning approaches like
Temporal Fusion Transformers :footcite:p:`Lim2021, Kouadio2025`, 
to make predictions. However, the power of these models brings a new challenge:
understanding and evaluating their performance in a way that is both
comprehensive and actionable.

A critical limitation of traditional evaluation is its reliance on
aggregate metrics. A single score, like mean absolute error or a
proper scoring rule, can conceal crucial details about a model's
behavior. It often fails to reveal *when* or *why* a model performs
poorly, or how the reliability of its predictions changes across
different conditions. This overlooks the fundamental principle that
the "goodness" of a forecast is multifaceted, encompassing qualities
like reliability and sharpness, not just accuracy alone
:footcite:p:`Gneiting2007b`. This challenge is particularly acute in
spatiotemporal forecasting, where performance can vary dramatically
across locations and time horizons :footcite:p:`Hong2025`. A prime example of 
this challenge can be found in the domain of urban geohazards. 


The Challenge: Forecasting Complex Urban Geohazards
-----------------------------------------------------

Urban environments worldwide face increasing pressure from geohazards,
often exacerbated by rapid urbanization and climate stress. **Land
subsidence**, the gradual sinking of the ground surface, is a prime
example, posing significant threats to infrastructure stability,
groundwater resources, and the resilience of coastal and low-lying
cities :footcite:p:`Liu2024`.

Forecasting the evolution of such phenomena is notoriously challenging.
It involves understanding the complex, often non-linear interplay
between diverse drivers acting across space and time—including
hydrological factors (groundwater levels, rainfall), geological
conditions (soil types, seismic activity), and anthropogenic pressures
(urban development, resource extraction).

While advanced models can improve predictive
accuracy, a critical gap remains: the **adequate assessment and
communication of forecast uncertainty**. Standard
evaluation often focuses on point forecast accuracy, neglecting the
inherent variability and potential unreliability of predictions. This
overlooks the fundamental principle that the "goodness" of a forecast
is multifaceted, encompassing qualities like reliability and sharpness,
not just accuracy alone.


The Need for Uncertainty-Aware Diagnostics
--------------------------------------------

Effective decision-making in urban planning, infrastructure management,
groundwater regulation, and hazard mitigation hinges not just on knowing
the most likely future state, but also on understanding the
**confidence** in that prediction and the **range of plausible
outcomes**. Standard metrics and plots often fail to provide intuitive
insights into the structure, consistency, and potential failures of
predictive uncertainty. Even established visualizations like fan charts
are primarily designed for single time series and do not scale well to
high-dimensional problems :footcite:p:`Sokol2025`.

During research focused on forecasting land subsidence in rapidly
developing areas like Nansha and particularly the complex urban setting
of **Zhongshan, China** :footcite:p:`kouadiob2025`, this challenge became
acutely apparent. For instance advanced models like the Extreme Temporal Fusion
Transformer :footcite:p:`Kouadio2025` could generate multi-horizon quantile
forecasts, however, interpreting the reliability and spatiotemporal patterns of
the predicted uncertainty bounds proved difficult with conventional
tools. How could we effectively diagnose if intervals were
well-calibrated? Where were the most significant prediction anomalies
occurring? How did uncertainty propagate across different forecast lead
times and geographical zones?


The Genesis of k-diagram
--------------------------

``k-diagram`` (where "k" acknowledges the author, Kouadio) was born
directly from the need to address these challenges. It stemmed from the
realization that **predictive uncertainty should be treated not merely
as a residual error metric, but as a first-class signal** demanding
dedicated tools for its exploration and interpretation.

The core idea was to leverage the **polar coordinate system** to create
novel visualizations ("k-diagrams") offering different perspectives on
model behavior and uncertainty:

* Visualizing coverage success/failure point-by-point (``Coverage Diagnostic``).
* Quantifying the severity and type of interval failures (``Anomaly Magnitude``).
* Assessing the stability of uncertainty estimates over time (``Interval Consistency``).
* Tracking how uncertainty magnitude changes across samples or evolves
  over forecast horizons (``Interval Width``, ``Uncertainty Drift``).
* Comparing overall model skill using established metrics, like the
  Taylor Diagram (:footcite:t:`Taylor2001`), in a polar layout.

These visualization methods, developed during the course of land
subsidence research, aim to provide more intuitive,
spatially explicit (when angle represents location or index), and
diagnostically rich insights than standard Cartesian plots alone.

Our Vision
------------

The ultimate goal of ``k-diagram`` is to help catalyze a shift
towards a more **interpretable and uncertainty-aware forecasting
paradigm**. By providing tools to move beyond opaque, single-score
metrics, we aim to empower researchers and practitioners to
deeply analyze and visualize predictive uncertainty. We hope this
enables more robust model evaluation, facilitates clearer
communication of forecast reliability, and ultimately supports
more informed, risk-aware decision-making in environmental
science, geohazard management, and other fields grappling with
complex forecasting challenges.

Contribution to the Community
-------------------------------

Beyond its specific application, ``k-diagram`` is a contribution
to the open-source ecosystem for scientific computing. By
providing a specialized, well-documented, and extensible toolkit,
we aim to lower the barrier for sophisticated forecast
diagnostics. We hope this package will not only serve as a
practical tool but also as an educational resource that fosters
collaboration and promotes reproducible best practices in forecast
verification and uncertainty quantification.


.. raw:: html

   <hr>
   
.. rubric:: References

.. footbibliography::