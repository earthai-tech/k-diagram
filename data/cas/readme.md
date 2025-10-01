# Cluster-Aware Severity (CAS) Score and Datasets

This document describes the data and artifacts for the paper:
**"CAS: Cluster-Aware Scoring for Probabilistic Forecasts."**

The paper introduces Cluster-Aware Severity (CAS) 
`kdiagram.metrics.cluster_aware_severity_score`, a distribution-aware
metric that penalizes forecast errors based on their local density
and run length. It is designed to identify "bursts," which are
contiguous sequences of misses that pose significant operational risks
but are often missed by traditional, order-agnostic scores like CRPS.

This folder provides the data necessary to reproduce the paper's findings.


# CAS data folder

This folder contains the data used in our paper:

> **CAS: Cluster-Aware Scoring for Probabilistic Forecasts**  
> ( paper in submission in Expert Systems with Applications Journal)

We evaluate probabilistic interval forecasts across three public 
domains—hourly wind, daily hydrology, and irregularly sampled land 
subsidence—and publish both the **raw inputs** and the **prepared 
modeling artifacts** needed to reproduce all results (`results_R*.py`).

---

## Datasets (sources)

1. **Wind (GEFCom2014 – hourly)**  
   IEA Wind Task 51 benchmarks: <https://iea-wind.org/task51/task51-information-portal/benchmarks>
2. **Hydrology (CAMELS-US – daily)**  
   RAL/NCAR CAMELS portal: <https://ral.ucar.edu/solutions/products/camels>
3. **Land subsidence (EGMS points – irregular)**  
   EEA SDI Catalogue record (UUID):  
   <https://sdi.eea.europa.eu/catalogue/srv/api/records/7eb207d6-0a62-4280-b1ca-f4ad1d9f91c3>

Please consult each source for licensing and terms of use. We request
that you cite the original datasets in addition to our paper.

---

## Folder layout

```
data/
└─ cas/
├─ raw/
│  ├─ camels\_timeseries.csv
│  ├─ egms\_point.csv
│  └─ gefcom\_hourly.csv
├─ preprocessed/
│  └─ .gitkeep
├─ modeling\_results\_ok/
│  ├─ metrics\_all\_domains.csv
│  ├─ metrics\_hydro.csv
│  ├─ metrics\_subsidence.csv
│  ├─ metrics\_wind.csv
│  ├─ predictions\_hydro.csv
│  ├─ predictions\_subsidence.csv
│  └─ predictions\_wind.csv
└─ outputs/        # figures & tables written by results\_R\*.py
└─ .gitkeep


```
-   `raw/` – Contains a sample of the minimally cleaned data from the
    sources. Due to GitHub's file size limits, the full raw datasets
    are not included. Users should download the full data from the
    original sources and structure it to match the schemas below.
-   `preprocessed/` – Initially empty, this folder will be populated
    with harmonized, long-format panels by the preprocessing scripts.
-   `modeling_results_ok/` – **Key inputs for the results scripts.**
    Contains pre-computed model predictions and metrics for each domain.
-   `outputs/` – Initially empty, this is the target directory where
    the `results_R*.py` scripts will save all generated figures and
    tables.
    
---

## File schemas

### Raw (illustrative; columns may include additional metadata)

-   **`gefcom_hourly.csv`** (Wind)
    -   `zone_id`, `t` (UTC hourly), `y` (target), and optional
        covariates like temperature (`T`).
-   **`camels_timeseries.csv`** (Hydro)
    -   `gauge_id`, `t` (daily), `y` (mm/day streamflow), and optional
        forcings (`pmm_day`, `tc`).
-   **`egms_point.csv`** (Subsidence)
    -   `point_id`, `t` (acquisition datetime), `y` (LOS displacement),
        and geo-metadata.

### Modeling artifacts (used by `results_R*.py`)

- **`predictions_*.csv`** (one file per domain)
  - Required columns:
    - `domain` \in {`wind`,`hydro`,`subsidence`}
    - `model` \in {`qar`,`qgbm`,`xtft`}
    - `horizon` (integer steps ahead; hours for wind, days for hydro,
      acquisition steps for subsidence)
    - `series_id` (zone/gauge/point)
    - `t` (timestamp; numeric step is acceptable for regular panels)
    - `y` (realization)
    - `q10`, `q50`, `q90` (forecast quantiles; **non-crossing is enforced
                           in the scripts**)
- **`metrics_*.csv`** and **`metrics_all_domains.csv`**
  - Per (domain, model, horizon[, series]) aggregates such as `coverage`, 
    `crps`, `winkler`, and (optionally) CAS summaries used by R9.
  - `metrics_all_domains.csv` is a union table referenced by `results_R9.py` 
    (see column autodetection in that script).

> **Nominal Interval**: Unless stated otherwise, all results use the
> central 80% prediction interval `[q10, q90]` (nominal coverage 0.80)
> for scoring and diagnostics.


## Reproduce the paper figures/tables

From the project root:

1. Ensure `data/cas/modeling_results_ok/` contains the three `predictions_*.csv`
   files (and metrics (`metrics_*.csv`) if you want to run R9 immediately).
2. Run the result scripts (they will write to `data/cas/outputs/`):

   - R5 robustness & sensitivity -> heatmaps and Kendall’s τ stability  
   - R6 calibration fixes -> reliability overlays and delta tables  
   - R7 statistical testing -> DM tests (CRPS proxy & Winkler) and MCS for |CAS|  
   - R8 operational impact -> burst-ROC (ex-ante risk index) & summary table  
   - R9 error taxonomy -> Figure 9 & correlations table
   
3. Verify the generated files under `data/cas/outputs/`.

> If you start from `raw/`, adapt your own preprocessing to produce the 
> long-format prediction files described above, or mirror our `preprocessed/` 
> layout and then generate `modeling_results_ok/`.

## Notes on the subsidence panel

- Sampling is **irregular**; horizons are indexed by acquisition **steps**, 
  not calendar gaps.
- CAS density uses a kernel in **days** (`H_IRREG_DAYS` in the scripts) 
  whenever timestamps are datetime-typed.


## Citations

Please cite both the datasets and our paper when using this repository. The 
reference metadata should be made available once the paper is published. 


**Paper**  

> "Kouadio K. L & R. Liu (2025). *CAS: Cluster-Aware Scoring for Probabilistic Forecasts.* (
> paper submitted in `Expert Systems with Applications`): DOI not available yet.

**Datasets**

-   **GEFCom2014** (IEA Wind Task 51): Hong, T., Pinson, P., Fan, S., et al. (2016).
    Probabilistic energy forecasting: Global energy forecasting
    competition 2014 and beyond. *International Journal of Forecasting*,
    32(3), 896-910. <https://iea-wind.org/task51/task51-information-portal/benchmarks> 
-   **CAMELS-US**: Newman, A. J., Clark, M. P., et al. (2014). The large-sample
    watershed-scale hydrometeorological dataset for the contiguous USA
    (CAMELS). *UCAR/NCAR*. <https://doi.org/10.5065/D6MW2F4D>
-   **EGMS**: European Environment Agency (2024). European ground motion
    service: Basic 2019-2023. (EEA SDI record UUID `7eb207d6-0a62-4280-b1ca-f4ad1d9f91c3`):
    <https://doi.org/10.2909/7eb207d6-0a62-4280-b1ca-f4ad1d9f91c3>


## Contact

For questions about the prepared artifacts or CAS evaluation code, open an 
issue or contact the authors of the paper.
