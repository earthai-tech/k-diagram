
# CAS examples

This folder provides the code to reproduce the figures and tables from 
the paper on Cluster-Aware Scoring (CAS). It contains an end-to-end Jupyter 
notebook that runs the entire pipeline, from data processing to generating 
final results. For users who prefer a modular approach, the folder also 
includes standalone Python scripts, each designed to create a specific 
figure or table from the paper


---

## Layout

```
examples/
└─ cas/
├─ notebooks/
│  └─ CAS\_end\_to\_end.ipynb
└─ scripts/
├─ └─ cas\_modeling.py
├─ └─ prepare\_cas\_datasets.py
├─ └─ preprocessing\_cas\_data.py
├─ └─ results\_config.py
├─ └─ results\_R1.py  ... results\_R9.py
└─ (outputs are written next to the scripts or to DATA\_ROOT)

```

Data used by the examples lives outside this folder. The expected
location is:

```

data/cas/
├─ modeling\_results\_ok/
│  ├─ predictions\_{wind,hydro,subsidence}.csv
│  └─ metrics\_all\_domains.csv (+ per-domain metrics)
└─ outputs/   # created by the result scripts

```

If your data is elsewhere, adjust paths in `results_config.py`.

---

## Requirements

Python 3.9+ and the following packages:

- numpy, pandas, matplotlib
- scipy, scikit-learn
- (optional) statsmodels for extras

Install with:

```

pip install kdiagram 

```
or if you use your own CAS implementation from the paper: 

```

pip install numpy pandas matplotlib scipy

```
---

## Configure paths

Open `examples/cas/scripts/results_config.py` and set:

- `DATA_ROOT` to the folder that contains `modeling_results_ok/`
- `OUTDIR` if you want outputs somewhere else

Most `results_R*.py` import these values. If a script defines its
own `BASE_DIR`, change it to use `results_config.DATA_ROOT`, or
keep a copy of the `modeling_results_ok/` folder next to that
script.

---

## Notebook

`notebooks/CAS_end_to_end.ipynb` runs the full pipeline:
preprocess (optional), evaluate, and render figures/tables. The
notebook assumes the same `results_config.py` settings.

If the notebook is not downloadable in your environment, you can
still reproduce all results by running the scripts below.

---

## Result scripts (what each produces)

R1  Reliability and PIT (Fig 1) + Table 1  
R2  Cluster-Aware Severity (CAS) trade-offs (Fig 2–3) + Table 2  
R3  Per-horizon winners and |CAS| vs horizon (Fig 4) + Table 3  
R4  Case studies with fan charts and local CAS stems (Fig 5a-5c)  
R5  Robustness heatmaps and rank stability (Fig 6) + Table 4  
R6  Isotonic calibration, reliability before/after (Fig 7) + T5  
R7  Statistical tests: DM (CRPS_proxy, Winkler) and MCS (Table 6–7)  
R8  Operational burst ROC (ex-ante risk index) (Fig 8) + table  
R9  Error taxonomy: log(1+|CAS|) vs CRPS (Fig 9) + correlations

Run any script from the `scripts/` folder, e.g.:

```

python results\_R1.py
python results\_R2.py
...
python results\_R9.py

```

Each script writes `.png` and `.pdf` figures and `.csv`/`.tex`
tables to the configured `OUTDIR`.

---

## Inputs expected by the result scripts

From `data/cas/modeling_results_ok/`:

- `predictions_wind.csv`
- `predictions_hydro.csv`
- `predictions_subsidence.csv`
- `metrics_all_domains.csv` (used by R3, R7, R9)

Each `predictions_*.csv` must contain at least:
`domain, model, horizon, series_id, t, y, q10, q50, q90`.

Quantiles may cross in the raw files; the scripts enforce
non-crossing before scoring.

---

## Outputs

By default, outputs go to `data/cas/outputs/` if you set
`OUTDIR = DATA_ROOT / "outputs"` in `results_config.py`. Otherwise
they are placed next to the script file.

---

## Troubleshooting

- **File not found**  
  Check `results_config.py` paths and that the four CSVs above are
  present.

- **Missing packages**  
  Re-install requirements from the list in this README.

- **Notebook cannot be saved or downloaded**  
  Use the scripts (`results_R*.py`) to reproduce all figures and
  tables without the notebook.

---

## Citation

Please cite both the datasets and our CAS paper when using these
examples. See `data/cas/readme.md` for dataset links and the
paper reference in the project root.

