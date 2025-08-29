.. _glossary:

========
Glossary
========

This glossary defines key terms and concepts used throughout the
`k-diagram` documentation and package.

.. glossary::
   :sorted:

   ACF (Autocorrelation Function)
     A function that measures the correlation between a time series
     and its own past values at different time lags. Used by
     :func:`~kdiagram.plot.context.plot_error_autocorrelation` to
     check for remaining patterns in forecast errors.
     
   ACov (Angular Coverage)
     A parameter (`acov`) controlling the angular span of the polar
     plots, such as 'default' (360°), 'half_circle' (180°),
     'quarter_circle' (90°), or 'eighth_circle' (45°).

   Anomaly (Prediction)
     An instance where the observed (actual) value falls outside the
     predicted uncertainty interval (i.e., below Qlow or above Qup).
     Visualized by :func:`~kdiagram.plot.uncertainty.plot_anomaly_magnitude`.

   Anomaly Magnitude
     The severity of a prediction anomaly, measured as the absolute
     distance between the actual value and the nearest violated
     prediction interval bound.

   AUC (Area Under the Curve)
     A summary metric for the ROC curve, representing a model's
     overall ability to discriminate between positive and negative
     classes. A higher AUC is better.
     
   Calibration (Interval)
     The degree to which the empirical coverage rate of prediction
     intervals matches their nominal coverage rate. A well-calibrated
     90% interval should cover approximately 90% of the actual values.
     is "honest" about its own uncertainty. Assessed by
     plots like the :func:`~kdiagram.plot.probabilistic.plot_pit_histogram`
     and :func:`~kdiagram.plot.comparison.plot_polar_reliability`.

   Consistency (Interval Width)
     The stability or variability of the prediction interval width
     (Qup - Qlow) for a specific location or sample across multiple
     time steps or forecast horizons. Assessed by
     :func:`~kdiagram.plot.uncertainty.plot_interval_consistency`.

   Conditional Bias
     A systemic error where a model's bias (the tendency to
     over- or under-predict) changes depending on the value of
     another variable, such as the true value or the predicted
     value itself. Diagnosed by
     :func:`~kdiagram.plot.relationship.plot_error_relationship`.
     
   Confusion Matrix
     A table used to evaluate the performance of a classification
     model. It summarizes the counts of True Positives (TP),
     False Positives (FP), True Negatives (TN), and False
     Negatives (FN). Visualized by
     :func:`~kdiagram.plot.evaluation.plot_polar_confusion_matrix`.
          
   Coverage (Empirical)
     The actual fraction or percentage of observed (true) values that
     fall within their corresponding prediction intervals in a given
     dataset. Calculated by :func:`~kdiagram.plot.uncertainty.plot_coverage`
     and visualized point-wise by
     :func:`~kdiagram.plot.uncertainty.plot_coverage_diagnostic`.

   Coverage (Nominal)
     The theoretical or intended coverage rate of a prediction interval,
     determined by the quantile levels used. For example, the interval
     between the 10th (Q10) and 90th (Q90) percentiles has a nominal
     coverage of 80%.

   Drift (Model / Concept)
     The degradation or change in a model's performance or underlying
     data relationships over time or changing conditions.

   Drift (Uncertainty)
     The change in the magnitude or pattern of predicted uncertainty
     (typically interval width) over time or forecast horizons.
     Visualized by :func:`~kdiagram.plot.uncertainty.plot_model_drift`
     (average drift) and
     :func:`~kdiagram.plot.uncertainty.plot_uncertainty_drift` (pattern drift).

   F1-Score
     A classification metric that calculates the harmonic mean of
     Precision and Recall. It provides a single score that
     balances both metrics, and is particularly useful for
     imbalanced datasets.
     
   Fingerprint (Feature)
     A characteristic profile of feature importance values for a specific
     model, time period, or group, often visualized using a radar chart via
     :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`.

   Forecast Horizon
     The number of time steps into the future for which a forecast
     is made. A "multi-horizon" forecast provides predictions for
     multiple future time steps simultaneously.

   Forecasting
     The process of making predictions about future events based on
     past and present data. This typically involves analyzing time
     series data to identify patterns and project them forward.
     
   Heteroscedasticity
     A condition where the variance of a model's errors is not
     constant. Diagnosed by plots like
     :func:`~kdiagram.plot.relationship.plot_conditional_quantiles` and
     :func:`~kdiagram.plot.relationship.plot_residual_relationship`.
     
   Interval Width
     The difference between the upper quantile (Qup) and lower quantile
     (Qlow) of a prediction interval, representing the magnitude of
     predicted uncertainty. Visualized by
     :func:`~kdiagram.plot.uncertainty.plot_interval_width`.

   K-Diagram
     The term used for the specialized polar diagnostic plots generated
     by this package, named after the author (Kouadio).
     
   Lag
     In time series analysis, the time difference or number of time
     steps between an observation and a previous observation. Used
     in ACF and PACF plots to measure autocorrelation at different
     past intervals.
     
   PACF (Partial Autocorrelation Function)
     A function that measures the direct correlation between a time
     series and a lagged version of itself, after removing the
     influence of shorter lags. Used by
     :func:`~kdiagram.plot.context.plot_error_pacf`.

   Pinball Loss
     A metric used to evaluate the accuracy of a single quantile
     forecast. The CRPS is the average of the Pinball Loss over all
     quantiles. Visualized by
     :func:`~kdiagram.plot.evaluation.plot_pinball_loss`.

   PIT (Probability Integral Transform)
     A method for evaluating the calibration of a probabilistic
     forecast. For a well-calibrated model, the PIT values of the
     observations should be uniformly distributed. Visualized by
     :func:`~kdiagram.plot.probabilistic.plot_pit_histogram`.

   Point Forecast
     A single-value prediction of a future outcome, typically the
     mean or median of the forecast distribution. Contrasts with a
     probabilistic forecast.
     
   Polar Plot / Coordinates
     A graphical system where points are located by an angle (theta, θ)
     and a distance from a central point (radius, r). Used extensively
     in `k-diagram`.

   Precision
     A classification metric that measures the accuracy of positive
     predictions. Defined as :math:`TP / (TP + FP)`.
     
   Prediction Interval (PI)
     A range [Qlow, Qup] derived from quantile forecasts, intended to
     contain the actual observed value with a certain probability
     (nominal coverage).

   Probabilistic Forecast
     A forecast that provides a full probability distribution for a
     future outcome, rather than just a single point value. This is
     often represented by a set of quantiles.
     
   Proper Scoring Rule
     A metric used to evaluate the quality of a probabilistic
     forecast that simultaneously assesses both calibration and
     sharpness. A key property is that the forecaster is incentivized
     to report their true belief to get the best score. The CRPS and
     Winkler Score are examples.
         
   Q-Q Plot (Quantile-Quantile Plot)
     A plot that compares the quantiles of a sample distribution
     (e.g., forecast errors) against the quantiles of a theoretical
     distribution (e.g., normal) to check for similarity.
     
   Quantile
     A value below which a certain proportion of the data or probability
     distribution falls. Common examples used in forecasting are Q10 (10th
     percentile), Q50 (50th percentile or median), and Q90 (90th
     percentile).

   Radar Chart
     A type of polar plot where multiple quantitative variables
     (represented by axes radiating from the center) are shown for one
     or more observations (represented by polygons or lines). Used by
     :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint` and
     optionally by :func:`~kdiagram.plot.uncertainty.plot_coverage`.

   Recall (Sensitivity)
     A classification metric that measures the ability of a model to
     find all the actual positive samples. Defined as TP / (TP + FN).

   Reliability Diagram
     A plot that compares predicted probabilities to observed
     frequencies to assess a classifier's calibration.

   Residual
     Another term for forecast **error**, calculated as the
     difference between the actual observed value and the model's
     predicted value (:math:`e_i = y_{true,i} - y_{pred,i}`).
     
   ROC Curve (Receiver Operating Characteristic Curve)
     A plot that shows the performance of a binary classifier by
     plotting the True Positive Rate against the False Positive Rate.

   Sharpness
     A measure of the concentration or narrowness of a probabilistic
     forecast's distribution, typically quantified by the average
     prediction interval width. A sharper forecast is more precise.
     Visualized by :func:`~kdiagram.plot.probabilistic.plot_polar_sharpness`.
     
   RMSD (Centered Root Mean Square Difference)
     A metric implicitly represented on a Taylor Diagram as the distance
     between a model point and the reference point. It measures the overall
     difference considering both standard deviation and correlation.

   Taylor Diagram
     A polar-style diagram summarizing model skill by plotting standard
     deviation (radius), correlation (angle), and RMSD (distance from
     reference) relative to observed data. Generated by functions in
     :mod:`kdiagram.plot.evaluation`.

   Tidy Data
     A standard for structuring datasets where each row is an
     observation, each column is a variable, and each table
     represents a single observational unit. The reshaping
     utilities in :mod:`kdiagram.utils.q_utils` are designed to
     help create tidy data.
     
   Uncertainty Quantification (UQ)
     The process of estimating and characterizing the uncertainty
     associated with model predictions, simulations, or measurements.

     A rigorous framework for identifying, characterizing, and
     managing the uncertainty inherent in computational models and
     predictions. UQ moves beyond single point forecasts to provide a
     probabilistic view of all possible outcomes. It aims to
     distinguish between two primary sources of uncertainty:

     - **Aleatoric uncertainty**: The inherent randomness or
       variability in a system that cannot be reduced with more
       data (e.g., the roll of a die).
     - **Epistemic uncertainty**: Uncertainty due to a lack of
       knowledge, such as imperfect model parameters or structure.
       This type of uncertainty can potentially be reduced with
       more data or better models.

     The ultimate goal of UQ is to produce a **probabilistic
     forecast** that is both **calibrated** (reliable) and
     **sharp** (precise). The tools in `k-diagram` are designed to
     visually diagnose the quality of these UQ efforts.
     
   Velocity (Prediction)
     The average rate of change of the central prediction estimate (e.g.,
     Q50) over consecutive time steps for a given location or sample.
     Visualized by :func:`~kdiagram.plot.uncertainty.plot_velocity`.

   Winkler Score
     A proper scoring rule for evaluating a prediction interval that
     rewards sharpness (narrow intervals) while penalizing for a lack
     of coverage.
     