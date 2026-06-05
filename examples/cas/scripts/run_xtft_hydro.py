"""
Run XTFT on the hydro domain and write predictions to a temp CSV.

Usage (from k-diagram repo root, inside xtft-env):
  conda run -n xtft-env python examples/cas/scripts/run_xtft_hydro.py

Produces: data/cas/modeling_results_ok/predictions_hydro_xtft.csv
  columns: domain, model, horizon, series_id, t, y, q10, q50, q90

This file is then merged into predictions_hydro.csv by merge_xtft_hydro.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent
_REPO  = _HERE.parents[2]
_DATA  = _REPO / "data" / "cas"
os.environ.setdefault("KDIAGRAM_DATA_DIR", str(_DATA))

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

# TF config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from fusionlab.nn.hybrid import XTFT
from fusionlab.nn.losses import combined_quantile_loss
from sklearn.preprocessing import LabelEncoder

print(f"TF:         {tf.__version__}")
print(f"Python:     {sys.version.split()[0]}")

# ── config ────────────────────────────────────────────────────────────────────
DOMAIN     = "hydro"
QUANTILES  = [0.10, 0.50, 0.90]
LOOKBACK   = 12   # reduced from 60 — hydro has ~160 rows/horizon
SEED       = 42
EPOCHS     = 60
PATIENCE   = 10
BATCH      = 32   # smaller batches for small dataset

PQT_DIR = _DATA / "preprocessed"
OUT_DIR = _DATA / "modeling_results_ok"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "predictions_hydro_xtft.csv"

# hydro sampling (same as cas_modeling.py)
HYDRO_MAX_ROWS = 500_000
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── feature columns (only those present in supervised_long_hydro.parquet) ────
# Missing: y_lag14, y_lag28, y_roll14_mean/std, y_roll30_mean/std
DYN_COLS_CANDIDATES = [
    "p_mm_day", "t_c",
    "sin_doy", "cos_doy", "sin_dow", "cos_dow",
    "y_lag1", "y_lag2", "y_lag3", "y_lag7", "y_lag14", "y_lag28",
    "y_roll3_mean", "y_roll3_std",
    "y_roll7_mean", "y_roll7_std",
    "y_roll14_mean", "y_roll14_std",
    "y_roll30_mean", "y_roll30_std",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _sample_by_series(df, target_rows, seed=SEED):
    counts = (
        df.groupby("series_id", observed=True)
        .size()
        .sample(frac=1.0, random_state=seed)
    )
    keep, total = [], 0
    for sid, cnt in counts.items():
        keep.append(sid)
        total += cnt
        if total >= target_rows:
            break
    return df[df["series_id"].isin(keep)].copy()


def _xtft_windows(df, lookback, dyn_cols, le):
    xs, xd, xf, ys, idx_out = [], [], [], [], []
    known = set(le.classes_.tolist())
    unk   = le.transform(["__UNK__"])[0]
    df    = df.reset_index(drop=True)

    for sid, g in df.groupby("series_id", sort=False):
        g    = g.sort_values("t").reset_index(drop=True)
        s    = le.transform([str(sid)])[0] if str(sid) in known else unk
        Xd_g = g[dyn_cols].values.astype("float32")
        y_g  = g["y_future"].values.astype("float32")

        for i in range(lookback, len(g)):
            xs.append([s])
            xd.append(Xd_g[i - lookback : i])
            xf.append(np.zeros((lookback, 1), dtype="float32"))
            ys.append([y_g[i]])
            idx_out.append(g.index[i])

    Xs = np.asarray(xs,  dtype="float32")
    Xd = np.asarray(xd,  dtype="float32")
    Xf = np.asarray(xf,  dtype="float32")
    Y  = np.asarray(ys,  dtype="float32")[:, np.newaxis, :]
    return Xs, Xd, Xf, Y, np.asarray(idx_out)


# ── load data ─────────────────────────────────────────────────────────────────
name = f"supervised_long_{DOMAIN}"
pqt  = PQT_DIR / f"{name}.parquet"
csv  = PQT_DIR / f"{name}.csv"
if csv.exists():
    df = pd.read_csv(csv)
    print(f"Loaded CSV: {csv}")
elif pqt.exists():
    try:
        df = pd.read_parquet(pqt)
        print(f"Loaded parquet: {pqt}")
    except Exception as e:
        raise RuntimeError(
            f"Parquet read failed ({e}). "
            f"Convert to CSV first: python -c \"import pandas as pd; "
            f"pd.read_parquet('{pqt}').to_csv('{csv}', index=False)\""
        )
else:
    raise FileNotFoundError(f"No {name}.parquet/csv in {PQT_DIR}")

df["t"]         = pd.to_datetime(df["t"], errors="coerce")
df["series_id"] = df["series_id"].astype(str)
df = df.dropna(subset=["t"])

# Keep only needed columns
keep = ["series_id", "t", "y", "split", "horizon", "y_future"] + DYN_COLS_CANDIDATES
df = df[[c for c in keep if c in df.columns]].copy()

# Downsample hydro for memory
print(f"Sampling hydro to ~{HYDRO_MAX_ROWS:,} rows …")
df = _sample_by_series(df, HYDRO_MAX_ROWS)
print(f"  After sampling: {len(df):,} rows")

# Fill any residual NaN in features with column medians
for col in DYN_COLS_CANDIDATES:
    if col in df.columns:
        med = df[col].median()
        df[col] = df[col].fillna(med)

# Filter to only columns present in the data
dyn_cols = [c for c in DYN_COLS_CANDIDATES if c in df.columns]
print(f"Dynamic features ({len(dyn_cols)}): {dyn_cols}")

dtr = df[df["split"] == "train"].copy()
dva = df[df["split"] == "val"].copy()
dte = df[df["split"] == "test"].copy()
print(f"Splits — train: {len(dtr):,}  val: {len(dva):,}  test: {len(dte):,}")

# Label encoder over all series
all_tr = pd.concat([dtr, dva], ignore_index=True)
le = LabelEncoder()
all_ids = np.concatenate([
    all_tr["series_id"].unique(),
    np.array(["__UNK__"])
])
le.fit(all_ids)

# ── per-horizon XTFT loop ─────────────────────────────────────────────────────
horizons = sorted(df["horizon"].unique())
print(f"\nHorizons: {horizons}")

all_rows = []

for h in horizons:
    dtr_h = dtr[dtr["horizon"] == h].copy()
    dva_h = dva[dva["horizon"] == h].copy()
    dte_h = dte[dte["horizon"] == h].copy()

    if dte_h.empty or dtr_h.empty:
        print(f"  h={h}: skipped (empty split)")
        continue

    print(f"\n  h={h}: train={len(dtr_h)}  val={len(dva_h)}  test={len(dte_h)}")

    try:
        Xs_tr, Xd_tr, Xf_tr, Y_tr, _    = _xtft_windows(dtr_h, LOOKBACK, dyn_cols, le)
        Xs_va, Xd_va, Xf_va, Y_va, _    = _xtft_windows(dva_h, LOOKBACK, dyn_cols, le)
        Xs_te, Xd_te, Xf_te, _,   te_idx = _xtft_windows(dte_h, LOOKBACK, dyn_cols, le)

        if len(Xs_tr) == 0:
            raise ValueError("No training windows")

        print(f"    Windows — train: {len(Xs_tr)}  val: {len(Xs_va)}  test: {len(Xs_te)}")

        # Build model fresh each horizon
        mdl = XTFT(
            static_input_dim=1,
            dynamic_input_dim=len(dyn_cols),
            future_input_dim=1,
            embed_dim=32,
            forecast_horizon=1,
            quantiles=QUANTILES,
            max_window_size=LOOKBACK,
            memory_size=128,
            num_heads=2,
            dropout_rate=0.1,
            attention_units=32,
            hidden_units=64,
            lstm_units=64,
            scales=None,
            use_residuals=True,
            use_batch_norm=False,
            final_agg="last",
        )
        _ = mdl([Xs_tr[:2], Xd_tr[:2], Xf_tr[:2]])   # build weights
        loss_fn = combined_quantile_loss(mdl.quantiles)
        mdl.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)

        cb = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, verbose=0)]

        mdl.fit(
            [Xs_tr, Xd_tr, Xf_tr], Y_tr,
            validation_data=([Xs_va, Xd_va, Xf_va], Y_va),
            epochs=EPOCHS, batch_size=BATCH,
            verbose=0, callbacks=cb,
        )

        raw = mdl.predict([Xs_te, Xd_te, Xf_te], verbose=0)
        raw = np.asarray(raw)
        if   raw.ndim == 4: raw = raw[:, 0, :, 0]
        elif raw.ndim == 3: raw = raw[:, 0, :]

        # Align predictions to dte_h rows
        dte_h = dte_h.reset_index(drop=True)
        q10 = np.full(len(dte_h), np.nan)
        q50 = np.full(len(dte_h), np.nan)
        q90 = np.full(len(dte_h), np.nan)

        k = min(len(te_idx), len(raw))
        te_idx_k = te_idx[:k]
        # te_idx is positional in dte_h (post reset_index)
        q10[te_idx_k] = raw[:k, 0]
        q50[te_idx_k] = raw[:k, 1]
        q90[te_idx_k] = raw[:k, 2]

        part = pd.DataFrame({
            "domain":    DOMAIN,
            "model":     "xtft",
            "horizon":   h,
            "series_id": dte_h["series_id"].values,
            "t":         dte_h["t"].values,
            "y":         dte_h["y_future"].values,
            "q10":       q10,
            "q50":       q50,
            "q90":       q90,
        })
        all_rows.append(part)

        n_filled = int(np.isfinite(q10).sum())
        print(f"    Done — {n_filled}/{len(dte_h)} rows filled")

        # Free memory
        del mdl
        tf.keras.backend.clear_session()

    except Exception as exc:
        import traceback
        print(f"    [WARN] h={h} failed: {exc}")
        traceback.print_exc()
        # Still append NaN rows so downstream merge stays aligned
        dte_h = dte_h.reset_index(drop=True)
        part = pd.DataFrame({
            "domain":    DOMAIN,
            "model":     "xtft",
            "horizon":   h,
            "series_id": dte_h["series_id"].values,
            "t":         dte_h["t"].values,
            "y":         dte_h["y_future"].values,
            "q10":       np.nan,
            "q50":       np.nan,
            "q90":       np.nan,
        })
        all_rows.append(part)

# ── save ──────────────────────────────────────────────────────────────────────
if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[Saved] {OUT_CSV}  ({len(out)} rows)")
    n_valid = int(np.isfinite(out["q10"]).sum())
    print(f"  Rows with valid q10: {n_valid}/{len(out)}")
else:
    print("[WARN] No results to save.")

print("\n[Done] run_xtft_hydro complete.")
