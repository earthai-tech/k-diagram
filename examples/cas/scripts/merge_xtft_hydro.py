"""
Merge XTFT hydro predictions back into predictions_hydro.csv and
recompute metrics_hydro.csv + metrics_all_domains.csv.

Run this after run_xtft_hydro.py has produced:
  data/cas/modeling_results_ok/predictions_hydro_xtft.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_DATA = _REPO / "data" / "cas"
os.environ.setdefault("KDIAGRAM_DATA_DIR", str(_DATA))
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kdiagram.metrics import (  # noqa: E402
    cluster_aware_severity_score as cas_score,
)

OUT_DIR = _DATA / "modeling_results_ok"
ALPHA = 0.10
CAS_WINDOW = 21


# ── helpers from cas_modeling.py ─────────────────────────────────────────────
def _pinball(u, tau):
    return (tau - (u < 0).astype(float)) * u


def crps_from_quantiles(y, qmap, *, taus):
    y = np.asarray(y, float)
    taus = sorted(list(taus))
    taus_ext = [0.0] + taus + [1.0]
    w = [
        0.5 * (taus_ext[i + 1] - taus_ext[i - 1])
        for i in range(1, len(taus_ext) - 1)
    ]
    w = np.asarray(w, float)
    losses = [
        _pinball(
            y - np.asarray(qmap.get(t, np.full_like(y, np.nan)), float), t
        )
        for t in taus
    ]
    L = np.vstack(losses).T
    return 2.0 * np.nansum(L * w, axis=1)


def winkler(y, ql, qu, alpha=ALPHA):
    y, ql, qu = (
        np.asarray(y, float),
        np.asarray(ql, float),
        np.asarray(qu, float),
    )
    w = qu - ql
    return (
        w
        + (2 / alpha) * (ql - y) * (y < ql)
        + (2 / alpha) * (y - qu) * (y > qu)
    )


def covg(y, ql, qu):
    return float(
        np.nanmean(
            (np.asarray(y, float) >= ql) & (np.asarray(y, float) <= qu)
        )
    )


def compute_cas(y, ql, qu, sort_by):
    y_pred = np.c_[ql, qu]
    return float(
        cas_score(
            y,
            y_pred,
            window_size=CAS_WINDOW,
            sort_by=np.asarray(sort_by),
            nan_policy="omit",
        )
    )


def metrics_for(df_model):
    """Compute scalar metrics for one (model, horizon) block."""
    rows = []
    for (domain, model, horizon), dd in df_model.groupby(
        ["domain", "model", "horizon"]
    ):
        cv = covg(dd["y"], dd["q10"], dd["q90"])
        d_cv = float(cv - (1 - ALPHA))
        wk = float(np.nanmean(winkler(dd["y"], dd["q10"], dd["q90"])))
        crps = float(
            np.nanmean(
                crps_from_quantiles(
                    dd["y"].values,
                    {
                        0.10: dd["q10"].values,
                        0.50: dd["q50"].values,
                        0.90: dd["q90"].values,
                    },
                    taus=[0.10, 0.50, 0.90],
                )
            )
        )
        cas_vals = []
        for _, g in dd.groupby("series_id", sort=False):
            mask = (
                np.isfinite(g["y"])
                & np.isfinite(g["q10"])
                & np.isfinite(g["q90"])
            )
            if mask.sum() == 0:
                continue
            sv = (
                g.loc[mask, "t"]
                .values.astype("datetime64[ns]")
                .astype("int64")
            )
            c = compute_cas(
                g.loc[mask, "y"].values,
                g.loc[mask, "q10"].values,
                g.loc[mask, "q90"].values,
                sv,
            )
            if np.isfinite(c):
                cas_vals.append(c)
        rows.append(
            dict(
                domain=domain,
                model=model,
                horizon=horizon,
                n=int(len(dd)),
                coverage=cv,
                delta_cov=d_cv,
                winkler=wk,
                crps=crps,
                cas=float(np.mean(cas_vals)) if cas_vals else np.nan,
            )
        )
    return pd.DataFrame(rows)


# ── load existing predictions ─────────────────────────────────────────────────
pred_path = OUT_DIR / "predictions_hydro.csv"
xtft_path = OUT_DIR / "predictions_hydro_xtft.csv"

if not xtft_path.exists():
    raise FileNotFoundError(f"Run run_xtft_hydro.py first: {xtft_path}")

pred = pd.read_csv(pred_path)
xtft = pd.read_csv(xtft_path)

print(f"Existing predictions: {len(pred)} rows")
print(f"XTFT predictions:     {len(xtft)} rows")
n_filled = int(np.isfinite(pd.to_numeric(xtft["q10"], errors="coerce")).sum())
print(f"  Valid XTFT rows (q10 finite): {n_filled}/{len(xtft)}")

# Replace existing xtft rows (all NaN) with new predictions
pred_no_xtft = pred[pred["model"] != "xtft"].copy()
merged = pd.concat([pred_no_xtft, xtft], ignore_index=True)
merged = merged.sort_values(
    ["model", "horizon", "series_id", "t"]
).reset_index(drop=True)
merged.to_csv(pred_path, index=False)
print(f"\n[Saved] {pred_path} ({len(merged)} rows)")

# ── recompute hydro metrics ───────────────────────────────────────────────────
merged["q10"] = pd.to_numeric(merged["q10"], errors="coerce")
merged["q50"] = pd.to_numeric(merged["q50"], errors="coerce")
merged["q90"] = pd.to_numeric(merged["q90"], errors="coerce")
merged["y"] = pd.to_numeric(merged["y"], errors="coerce")

met_hydro = metrics_for(merged)
met_hydro.to_csv(OUT_DIR / "metrics_hydro.csv", index=False)
print(f"[Saved] {OUT_DIR / 'metrics_hydro.csv'}")
print(met_hydro[["model", "horizon", "cas"]].to_string())

# ── rebuild metrics_all_domains.csv ──────────────────────────────────────────
domain_files = {
    "hydro": OUT_DIR / "predictions_hydro.csv",
    "wind": OUT_DIR / "predictions_wind.csv",
    "subsidence": OUT_DIR / "predictions_subsidence.csv",
}
all_mets = []
for _dom, fp in domain_files.items():
    df_d = pd.read_csv(fp)
    for col in ["q10", "q50", "q90", "y"]:
        df_d[col] = pd.to_numeric(df_d[col], errors="coerce")
    m = metrics_for(df_d)
    all_mets.append(m)

combo = pd.concat(all_mets, ignore_index=True)
combo.to_csv(OUT_DIR / "metrics_all_domains.csv", index=False)
print(f"\n[Saved] {OUT_DIR / 'metrics_all_domains.csv'} ({len(combo)} rows)")
print(combo[["domain", "model", "horizon", "cas"]].to_string())

print("\n[Done] merge_xtft_hydro complete.")
