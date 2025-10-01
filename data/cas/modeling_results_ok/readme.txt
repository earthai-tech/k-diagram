README for modeling_results_ok

This directory contains the pre-computed modeling artifacts needed to reproduce the figures and tables from the paper, "CAS: Cluster-Aware Scoring for Probabilistic Forecasts".

Running the results_R*.py scripts from the examples/cas/scripts/ directory will use these files as direct inputs.

---
File Contents

metrics_all_domains.csv: A summary file containing aggregated scores (CRPS, Winkler, coverage, CAS) for all models across all domains and forecast horizons.

predictions_subsidence.csv: Contains the model predictions (q10, q50, q90) and the observed outcomes (y) for each time step in the subsidence dataset.

predictions_wind.csv: Contains the model predictions (q10, q50, q90) and the observed outcomes (y) for each time step in the wind dataset.

---
Important Note on Hydrology Data

The predictions_hydro.csv file is not included in this directory due to its large file size, which can exceed repository limits.

To generate this missing file, you must run the modeling pipeline for the "hydro" domain as described in the main project notebook (CAS_end_to_end.ipynb). This will create the predictions_hydro.csv file in this folder, allowing you to fully reproduce all results.