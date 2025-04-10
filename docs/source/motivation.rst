.. _motivation:

============================
Motivation and Background
============================

This page outlines the scientific context and practical motivations that
led to the development of the `k-diagram` package.

The Challenge: Forecasting Complex Urban Geohazards
------------------------------------------------------

Urban environments worldwide face increasing pressure from geohazards,
often exacerbated by rapid urbanization and climate stress. **Land
subsidence**, the gradual sinking of the ground surface, is a prime
example, posing significant threats to infrastructure stability,
groundwater resources, and the resilience of coastal and low-lying
cities.

Forecasting the evolution of such phenomena is notoriously challenging.
It involves understanding the complex, often non-linear interplay
between diverse drivers acting across space and time – including
hydrological factors (groundwater levels, rainfall), geological
conditions (soil types, seismic activity), and anthropogenic pressures
(urban development, resource extraction).

While advancements in modeling, including deep learning approaches like
Temporal Fusion Transformers, offer potential for improved predictive
accuracy, a critical gap often remains: the **adequate assessment and
communication of forecast uncertainty**. Standard evaluation often
focuses on point forecast accuracy, neglecting the inherent variability
and potential unreliability of predictions, especially when projecting
further into the future or across heterogeneous landscapes.

The Need for Uncertainty-Aware Diagnostics
--------------------------------------------

Effective decision-making in urban planning, infrastructure management,
groundwater regulation, and hazard mitigation hinges not just on knowing
the most likely future state, but also on understanding the **confidence**
in that prediction and the **range of plausible outcomes**. Standard
metrics and plots often fail to provide intuitive insights into the
structure, consistency, and potential failures of predictive uncertainty.

During research focused on forecasting land subsidence in rapidly
developing areas like Nansha and particularly the complex urban setting
of **Zhongshan, China**, this challenge became acutely apparent. While
advanced models could generate multi-horizon quantile forecasts,
interpreting the reliability and spatial-temporal patterns of the predicted
uncertainty bounds proved difficult with conventional tools. How could we
effectively diagnose if intervals were well-calibrated? Where were the
most significant prediction anomalies occurring? How did uncertainty
propagate across different forecast lead times and geographical zones?

The Genesis of k-diagram
---------------------------

`k-diagram` (where 'k' acknowledges the author, Kouadio) was born directly
from the need to address these challenges. It stemmed from the realization
that **predictive uncertainty should be treated not merely as a residual
error metric, but as a first-class signal** demanding dedicated tools for
its exploration and interpretation.

The core idea was to leverage the **polar coordinate system** to create
novel visualizations ("k-diagrams") offering different perspectives on
model behavior and uncertainty:

* Visualizing coverage success/failure point-by-point (`Coverage Diagnostic`).
* Quantifying the severity and type of interval failures (`Anomaly Magnitude`).
* Assessing the stability of uncertainty estimates over time (`Interval Consistency`).
* Tracking how uncertainty magnitude changes across samples or evolves
    over forecast horizons (`Interval Width`, `Uncertainty Drift`).
* Comparing overall model skill using established metrics in a polar
    layout (`Taylor Diagram`).

These visualization methods, developed during the course of the land
subsidence research (aspects of which are detailed in a paper submitted
to *Nature Sustainability* co-authored with Jianxi Liu, and Liu Rong), 
aim to provide more intuitive, spatially
explicit (when angle represents location or index), and diagnostically
rich insights than standard Cartesian plots alone.

Our Vision
------------

The ultimate goal of `k-diagram` is to contribute towards a more
**interpretable and uncertainty-aware forecasting paradigm**. By providing
tools to deeply analyze and visualize predictive uncertainty, we hope to
enable more robust model evaluation, facilitate better communication of
forecast reliability, and ultimately support more informed, risk-aware
decision-making in environmental science, geohazard management, and
other fields grappling with complex forecasting challenges.