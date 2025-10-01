# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
CAS modeling pipeline: train/evaluate probabilistic models per domain.

This script reads supervised datasets (wide features + y_future) and
trains three model families at multiple horizons:

  * QGBM  – LightGBM with quantile objective
  * QAR   – per-series Quantile AutoRegression (statsmodels)
  * XTFT  – Transformer-based sequence model (FusionLab XTFT)
            (requires `fusionlab-learn` and `tensorflow`)

For each horizon and model, it exports per-row quantile predictions
(q10/q50/q90) and aggregates metrics:

  - coverage (and delta to nominal :math:`1-\alpha`),
  - Winkler interval score,
  - CRPS (via quantile approximation),
  - Cluster-Aware Severity (CAS).

Outputs:
  - data:  OUT_DIR / f"predictions_{domain}.csv"
  - mets:  OUT_DIR / f"metrics_{domain}.csv"
  - all-domain summary (in __main__): OUT_DIR / "metrics_all_domains.csv"

Tip: to enable XTFT, install:
  pip install fusionlab-learn tensorflow==2.15( preferably)
See: https://fusion-lab.readthedocs.io/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

from kdiagram.metrics import (
    clustered_aware_severity_score as cas_score,
)

XTFT_AVAILABLE = True
try:
    # For XTFT support, install:
    #   pip install fusionlab-learn tensorflow==2.15
    # Docs: https://fusion-lab.readthedocs.io/
    import tensorflow as tf
    from fusionlab.nn.hybrid import XTFT
    from fusionlab.nn.losses import combined_quantile_loss
except Exception:
    XTFT_AVAILABLE = False

# ----------------------------- CONFIG ---------------------------------

# BASE_DIR = Path(__file__).resolve().parent
# PQT_DIR = BASE_DIR / "out" / "preprocessed_parquet"
# CSV_DIR = BASE_DIR / "out" / "preprocessed_csv"
# OUT_DIR = BASE_DIR / "out" / "modeling_results"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------- PATHS (repo-aware defaults) ------------------


def _find_repo_root(start: Path) -> Path:
    markers = ("pyproject.toml", ".git", "README.md")
    p = start.resolve()
    for _ in range(6):
        if any((p / m).exists() for m in markers) and (p / "data").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


# 1) Allow environment override (most explicit)
DATA_ROOT = os.getenv("KDIAGRAM_DATA_DIR")

# 2) Otherwise infer repo root and use data/cas/
if DATA_ROOT:
    DATA_ROOT = Path(DATA_ROOT)
else:
    REPO_ROOT = _find_repo_root(Path(__file__).parent)
    DATA_ROOT = REPO_ROOT / "data" / "cas"

# 3) Final directories (align to your new structure)
PQT_DIR = DATA_ROOT / "preprocessed"  # parquet/csv live together
CSV_DIR = DATA_ROOT / "preprocessed"
OUT_DIR = DATA_ROOT / "modeling_results_ok"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ALPHA = 0.10
QUANTILES = [0.10, 0.50, 0.90]
CAS_WINDOW = 21
SEED = 42
# ----------------------------- CONFIG ---------------------------------
SAVE_JSON_PER_H = True
# sampling config (only used for hydro)
HYDRO_MAX_ROWS = 500_000  # e.g. 200_000 or None for full
HYDRO_SAMPLING_STRATEGY = "series"  # "series" | "row"

np.random.seed(SEED)

# XTFT lookbacks (in rows; your supervised set is per-step)
XTFT_LOOKBACK = {"wind": 48, "hydro": 60, "subsidence": 12}

# Feature picks per domain (must exist in files)
BASE_LAGS = [
    "y_lag1",
    "y_lag2",
    "y_lag3",
    "y_lag7",
    "y_lag14",
    "y_lag28",
]
ROLLS = [
    "y_roll7_mean",
    "y_roll7_std",
    "y_roll14_mean",
    "y_roll14_std",
    "y_roll30_mean",
    "y_roll30_std",
]
TIME_FEATS = ["sin_doy", "cos_doy", "sin_dow", "cos_dow"]

COVARS = {
    "wind": ["T"],
    "hydro": ["p_mm_day", "t_c"],
    "subsidence": ["lat", "lon"],
}

pd.options.mode.copy_on_write = True

# ----------------------------- IO / helpers ----------------------------


def _pinball(u: np.ndarray, tau: float) -> np.ndarray:
    return (tau - (u < 0).astype(float)) * u


def crps_from_quantiles(
    y: np.ndarray, qmap: dict[float, np.ndarray], *, taus: list[float]
) -> np.ndarray:
    """
    Approximate CRPS via trapezoid rule on provided quantiles.
    """
    y = np.asarray(y, float)
    taus = sorted(list(taus))
    taus_ext = [0.0] + taus + [1.0]
    w = []
    for i in range(1, len(taus_ext) - 1):
        w.append(0.5 * (taus_ext[i + 1] - taus_ext[i - 1]))
    w = np.asarray(w, float)
    losses = []
    for tau in taus:
        q = np.asarray(qmap.get(tau, np.full_like(y, np.nan)), float)
        losses.append(_pinball(y - q, tau))
    L = np.vstack(losses).T
    crps = 2.0 * np.nansum(L * w, axis=1)
    return crps


def _sample_by_series(
    df: pd.DataFrame,
    target_rows: int,
    *,
    seed: int = SEED,
) -> pd.DataFrame:
    s_counts = (
        df.groupby("series_id", observed=True)
        .size()
        .sample(frac=1.0, random_state=seed)
    )
    keep, total = [], 0
    for sid, cnt in s_counts.items():
        keep.append(sid)
        total += cnt
        if total >= target_rows:
            break
    return df[df["series_id"].isin(keep)].copy()


def _sample_by_split_horizon(
    df: pd.DataFrame,
    target_rows: int,
    *,
    seed: int = SEED,
) -> pd.DataFrame:
    g = df.groupby(["split", "horizon"], observed=True)
    sizes = g.size()
    tot = int(sizes.sum())
    alloc = (
        (sizes / max(1, tot) * target_rows).round().astype(int).clip(lower=1)
    )
    parts = []
    for key, n in alloc.items():
        grp = g.get_group(key)
        n = min(n, len(grp))
        parts.append(grp.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _shrink_mem(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].nunique(dropna=False) / max(1, len(df)) < 0.5:
            df[c] = df[c].astype("category")
    return df


def _read_supervised(domain: str) -> pd.DataFrame:
    name = f"supervised_long_{domain}"
    pqt = PQT_DIR / f"{name}.parquet"
    csv = CSV_DIR / f"{name}.csv"
    if pqt.exists():
        df = pd.read_parquet(pqt)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"Missing {name}.parquet/csv in {PQT_DIR} or {CSV_DIR}"
        )
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df["series_id"] = df["series_id"].astype(str)
    return df.dropna(subset=["t"])


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------- METRICS -----------------------------------


def winkler(y, ql, qu, alpha=ALPHA) -> np.ndarray:
    y = np.asarray(y, float)
    ql = np.asarray(ql, float)
    qu = np.asarray(qu, float)
    w = qu - ql
    under = (y < ql).astype(float)
    over = (y > qu).astype(float)
    return (
        w + (2.0 / alpha) * (ql - y) * under + (2.0 / alpha) * (y - qu) * over
    )


def covg(y, ql, qu) -> float:
    y = np.asarray(y, float)
    return float(np.nanmean((y >= ql) & (y <= qu)))


def compute_cas(y, ql, qu, sort_by, window=CAS_WINDOW) -> float:
    y = np.asarray(y, float)
    y_pred = np.c_[ql, qu]
    return float(
        cas_score(
            y,
            y_pred,
            window_size=window,
            sort_by=np.asarray(sort_by),
            nan_policy="omit",
        )
    )


# -------------------------- MODELS (MAIN) ------------------------------


def fit_qgbm(
    dtr: pd.DataFrame,
    dte: pd.DataFrame,
    feats: list[str],
    qtiles: list[float],
) -> dict[float, np.ndarray]:
    """
    Train pooled LightGBM quantile models and predict on test.

    Parameters
    ----------
    dtr : DataFrame
        Train (optionally concatenated with validation).
    dte : DataFrame
        Test split to score.
    feats : list of str
        Feature columns (may include a series label).
    qtiles : list of float
        Quantiles to fit (e.g., [0.1, 0.5, 0.9]).

    Returns
    -------
    dict[float, ndarray]
        Mapping {quantile -> predictions} aligned to `dte`.
    """
    Xtr = dtr[feats].values
    ytr = dtr["y_future"].values
    Xte = dte[feats].values
    out = {}
    for q in qtiles:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
        )
        m.fit(Xtr, ytr)
        out[q] = m.predict(Xte)
    return out


def _degenerate(y, x, tol=1e-12):
    return (np.nanstd(y) < tol) or (np.nanstd(x) < tol) or (len(y) < 3)


def fit_qar_per_series(
    dtr: pd.DataFrame,
    dte: pd.DataFrame,
    qtiles: list[float],
    lag_col: str = "y_lag1",
) -> dict[float, np.ndarray]:
    """
    Per-series Quantile AutoRegression using statsmodels.QuantReg.

    Fits a simple AR(1)-style quantile regression `y_future ~ const + y_lag1`
    per `series_id`, then predicts on that series' test rows.

    Parameters
    ----------
    dtr : DataFrame
        Train (optionally concatenated with validation).
    dte : DataFrame
        Test split to score.
    qtiles : list of float
        Quantiles to fit.
    lag_col : str, default="y_lag1"
        Name of the lag feature to use as regressor.

    Returns
    -------
    dict[float, ndarray]
        Mapping {quantile -> predictions} aligned to `dte`.
    """
    preds = {q: np.full(len(dte), np.nan) for q in qtiles}
    dtr = dtr.sort_values(["series_id", "t"])
    dte = dte.sort_values(["series_id", "t"])
    for sid, gtr in dtr.groupby("series_id", sort=False):
        gte = dte[dte["series_id"] == sid]
        if gte.empty:
            continue
        gtr = gtr.dropna(subset=[lag_col, "y_future"])
        if len(gtr) < 8:
            continue

        X = gtr[[lag_col]].values.ravel()
        y = gtr["y_future"].values

        if _degenerate(y, X):
            qvals = (
                np.quantile(y, qtiles) if len(y) else [np.nan] * len(qtiles)
            )
            for qi, q in enumerate(qtiles):
                preds[q][dte["series_id"].values == sid] = qvals[qi]
            continue

        Xtr = sm.add_constant(gtr[[lag_col]])
        ytr = gtr["y_future"].values
        Xte = sm.add_constant(gte[[lag_col]], has_constant="add")
        for q in qtiles:
            try:
                mdl = sm.QuantReg(ytr, Xtr)
                res = mdl.fit(q=q)
                yhat = res.predict(Xte).values
            except Exception:
                yhat = np.full(len(Xte), np.nan)
            mask = dte["series_id"].values == sid
            preds[q][mask] = yhat
    return preds


# -------------------------- XTFT (MAIN) --------------------------------


def _labeler(*dfs) -> LabelEncoder:
    le = LabelEncoder()
    all_ids = (
        pd.concat([d["series_id"] for d in dfs], ignore_index=True)
        .astype(str)
        .unique()
    )
    all_ids = np.concatenate([all_ids, np.array(["__UNK__"])])
    le.fit(all_ids)
    return le


def _xtft_windows(df, lookback, dyn_cols, le):
    xs, xd, xf, ys = [], [], [], []
    known = set(le.classes_.tolist())
    unk = le.transform(["__UNK__"])[0]
    for sid, g in df.groupby("series_id", sort=False):
        g = g.sort_values("t")
        s_str = str(sid)
        s = le.transform([s_str])[0] if s_str in known else unk
        Xd = g[dyn_cols].values
        y = g["y_future"].values
        for i in range(lookback, len(g)):
            win = Xd[i - lookback : i]
            xs.append([s])
            xd.append(win)
            xf.append(np.zeros((lookback, 1)))
            ys.append([y[i]])
    Xs = np.asarray(xs)
    Xd = np.asarray(xd)
    Xf = np.asarray(xf)
    Y = np.asarray(ys)[:, np.newaxis, :]
    return Xs, Xd, Xf, Y


def fit_xtft_one_h(
    dtr: pd.DataFrame,
    dva: pd.DataFrame,
    dte: pd.DataFrame,
    domain: str,
    qtiles: list[float],
    lookback: int,
    dyn_cols: list[str],
) -> dict[float, np.ndarray]:
    """
    Train FusionLab XTFT for a single horizon and predict quantiles.

    Requires `fusionlab-learn` and `tensorflow`.
    Install:
        pip install fusionlab-learn tensorflow
    Docs: https://fusion-lab.readthedocs.io/

    Parameters
    ----------
    dtr, dva, dte : DataFrame
        Train, validation, and test splits (same horizon).
    domain : str
        Domain name (e.g., "wind", "hydro", "subsidence").
    qtiles : list of float
        Quantiles to predict.
    lookback : int
        Number of past steps in each sequence window.
    dyn_cols : list of str
        Dynamic (time-varying) feature columns.

    Returns
    -------
    dict[float, ndarray]
        Mapping {quantile -> predictions} aligned to `dte`.
        If XTFT is unavailable, returns NaNs.
    """
    if not XTFT_AVAILABLE:
        return {q: np.full(len(dte), np.nan) for q in qtiles}

    all_tr = pd.concat([dtr, dva], axis=0, ignore_index=True)
    le = _labeler(all_tr)
    Xs_tr, Xd_tr, Xf_tr, Y_tr = _xtft_windows(dtr, lookback, dyn_cols, le)
    Xs_va, Xd_va, Xf_va, Y_va = _xtft_windows(dva, lookback, dyn_cols, le)

    mdl = XTFT(
        static_input_dim=1,
        dynamic_input_dim=len(dyn_cols),
        future_input_dim=1,
        embed_dim=32,
        forecast_horizon=1,
        quantiles=qtiles,
        max_window_size=lookback,
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
    _ = mdl([Xs_tr, Xd_tr, Xf_tr])
    loss_fn = combined_quantile_loss(mdl.quantiles)
    mdl.compile(optimizer="adam", loss=loss_fn)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    mdl.fit(
        x=[Xs_tr, Xd_tr, Xf_tr],
        y=Y_tr,
        validation_data=([Xs_va, Xd_va, Xf_va], Y_va),
        epochs=40,
        batch_size=64,
        verbose=1,
        callbacks=cb,
    )

    dte_pos = dte.reset_index(drop=True)
    Xs_te, Xd_te, Xf_te, _ = _xtft_windows(dte_pos, lookback, dyn_cols, le)
    raw = mdl.predict([Xs_te, Xd_te, Xf_te], verbose=0)

    raw = np.asarray(raw)
    if raw.ndim == 4:
        raw = raw[:, 0, :, 0]
    elif raw.ndim == 3:
        raw = raw[:, 0, :]
    elif raw.ndim == 2:
        pass
    else:
        raise ValueError(f"XTFT out shape {raw.shape}")

    out = {q: np.full(len(dte_pos), np.nan) for q in qtiles}
    fill_idx = []
    for _, g in dte_pos.groupby("series_id", sort=False):
        idx = g.index.values
        if len(idx) > lookback:
            fill_idx.extend(idx[lookback:])
    fill_idx = np.asarray(fill_idx)

    if len(fill_idx) != len(raw):
        k = min(len(fill_idx), len(raw))
        fill_idx = fill_idx[:k]
        raw = raw[:k, :]

    for qi, q in enumerate(qtiles):
        out[q][fill_idx] = raw[:, qi]
    return out


# -------------------------- EVAL / PACK (MAIN) -------------------------


def eval_and_pack(
    dte_h: pd.DataFrame,
    pred_map: dict[str, dict[float, np.ndarray]],
    domain: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble per-row predictions and compute metrics for one horizon.

    Parameters
    ----------
    dte_h : DataFrame
        Test rows at a single horizon with columns
        ['series_id','t','y_future', ...].
    pred_map : dict
        {'model_name': {quantile: ndarray predictions}}.
    domain : str
        Domain label for outputs.
    horizon : int
        Forecast horizon being evaluated.

    Returns
    -------
    (pred_df, met_df) : (DataFrame, DataFrame)
        pred_df columns: domain, model, horizon, series_id, t, y, q10, q50, q90
        met_df columns: domain, model, horizon, n, coverage, delta_cov,
                        winkler, crps, cas
    """
    rows = []
    for m, qm in pred_map.items():
        q10 = qm.get(0.10, np.full(len(dte_h), np.nan))
        q50 = qm.get(0.50, np.full(len(dte_h), np.nan))
        q90 = qm.get(0.90, np.full(len(dte_h), np.nan))
        part = pd.DataFrame(
            {
                "domain": domain,
                "model": m,
                "horizon": horizon,
                "series_id": dte_h["series_id"].values,
                "t": dte_h["t"].values,
                "y": dte_h["y_future"].values,
                "q10": q10,
                "q50": q50,
                "q90": q90,
            }
        )
        rows.append(part)
    pred_df = pd.concat(rows, ignore_index=True)

    mets = []
    for m in pred_df["model"].unique():
        dd = pred_df[pred_df["model"] == m].copy()
        cv = covg(dd["y"], dd["q10"], dd["q90"])
        d_cov = float(cv - (1.0 - ALPHA))
        wk = float(np.nanmean(winkler(dd["y"], dd["q10"], dd["q90"], ALPHA)))

        crps_vec = crps_from_quantiles(
            dd["y"].values,
            {
                0.10: dd["q10"].values,
                0.50: dd["q50"].values,
                0.90: dd["q90"].values,
            },
            taus=[0.10, 0.50, 0.90],
        )
        crps = float(np.nanmean(crps_vec))

        cas_vals = []
        for _, g in dd.groupby("series_id", sort=False):
            mask = (
                np.isfinite(g["y"])
                & np.isfinite(g["q10"])
                & np.isfinite(g["q90"])
            )
            if mask.sum() == 0:
                continue
            yv = g.loc[mask, "y"].values
            ql = g.loc[mask, "q10"].values
            qu = g.loc[mask, "q90"].values
            sv = (
                g.loc[mask, "t"]
                .values.astype("datetime64[ns]")
                .astype("int64")
            )
            c = compute_cas(yv, ql, qu, sv)
            if np.isfinite(c):
                cas_vals.append(c)
        cas_metric = float(np.mean(cas_vals)) if cas_vals else np.nan

        mets.append(
            {
                "domain": domain,
                "model": m,
                "horizon": horizon,
                "n": int(len(dd)),
                "coverage": cv,
                "delta_cov": d_cov,
                "winkler": wk,
                "crps": crps,
                "cas": cas_metric,
            }
        )
    met_df = pd.DataFrame(mets)

    if SAVE_JSON_PER_H:
        js_path = OUT_DIR / f"metrics_{domain}_h{horizon}.json"
        js_path.parent.mkdir(parents=True, exist_ok=True)
        with open(js_path, "w", encoding="utf-8") as f:
            recs = met_df.to_dict(orient="records")
            json.dump(recs, f, ensure_ascii=False, indent=2)

    return pred_df, met_df


# ------------------------- DRIVER (MAIN) -------------------------------


def _feature_list(domain: str, cols: list[str]) -> list[str]:
    feats = TIME_FEATS + BASE_LAGS + ROLLS
    feats += COVARS.get(domain, [])
    return [c for c in feats if c in cols]


def run_domain(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline for one domain: load → split → fit → predict → metrics.

    Reads `supervised_long_{domain}` from PQT_DIR/CSV_DIR, filters to
    required columns, applies optional sampling for hydro, computes
    features list, loops over horizons and trains QGBM, QAR, XTFT,
    aggregates predictions and metrics, and writes outputs under OUT_DIR.

    Parameters
    ----------
    domain : str
        One of {"hydro", "subsidence", "wind"}.

    Returns
    -------
    (pred_df, met_df) : (DataFrame, DataFrame)
        Concatenated predictions and metrics over all horizons.
    """
    print(f"\n=== {domain.upper()} ===")
    df = _read_supervised(domain)

    keep = ["series_id", "t", "y", "split", "horizon", "y_future"]
    if domain == "wind":
        keep += [
            "T",
            "sin_doy",
            "cos_doy",
            "sin_dow",
            "cos_dow",
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_lag7",
            "y_lag14",
            "y_lag28",
            "y_roll7_mean",
            "y_roll7_std",
            "y_roll14_mean",
            "y_roll14_std",
            "y_roll30_mean",
            "y_roll30_std",
        ]
    elif domain == "hydro":
        keep += [
            "p_mm_day",
            "t_c",
            "sin_doy",
            "cos_doy",
            "sin_dow",
            "cos_dow",
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_lag7",
            "y_lag14",
            "y_lag28",
            "y_roll7_mean",
            "y_roll7_std",
            "y_roll14_mean",
            "y_roll14_std",
            "y_roll30_mean",
            "y_roll30_std",
        ]
    elif domain == "subsidence":
        keep += [
            "lat",
            "lon",
            "sin_doy",
            "cos_doy",
            "sin_dow",
            "cos_dow",
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_lag7",
            "y_lag14",
            "y_lag28",
            "y_roll7_mean",
            "y_roll7_std",
            "y_roll14_mean",
            "y_roll14_std",
            "y_roll30_mean",
            "y_roll30_std",
        ]

    df = df.loc[:, [c for c in keep if c in df.columns]].copy()
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    df = _shrink_mem(df)

    if domain == "hydro" and HYDRO_MAX_ROWS:
        print(
            f"Sampling HYDRO to ~{HYDRO_MAX_ROWS:,} rows "
            f"via {HYDRO_SAMPLING_STRATEGY} strategy..."
        )
        if HYDRO_SAMPLING_STRATEGY == "series":
            df = _sample_by_series(df, HYDRO_MAX_ROWS, seed=SEED)
        else:
            df = _sample_by_split_horizon(df, HYDRO_MAX_ROWS, seed=SEED)
        missing = {"train", "val", "test"} - set(df["split"].unique())
        if missing:
            print(f"[warn] missing splits after sampling: {missing}")

    if domain != "hydro":
        try:
            df = df.sort_values(
                ["series_id", "t"], kind="mergesort", ignore_index=True
            )
        except TypeError:
            pass

    feats_all = _feature_list(domain, df.columns)
    if not feats_all:
        raise ValueError(f"No features for {domain}.")

    dtr = df[df["split"] == "train"].copy()
    dva = df[df["split"] == "val"].copy()
    dte = df[df["split"] == "test"].copy()

    preds_all, mets_all = [], []
    for h in sorted(df["horizon"].unique()):
        dtr_h = dtr[dtr["horizon"] == h].copy()
        dva_h = dva[dva["horizon"] == h].copy()
        dte_h = dte[dte["horizon"] == h].copy()
        if dte_h.empty or dtr_h.empty:
            continue

        le = LabelEncoder().fit(df["series_id"].astype(str))
        for dd in (dtr_h, dva_h, dte_h):
            dd["series_lab"] = le.transform(dd["series_id"].astype(str))

        feats = ["series_lab"] + feats_all
        print(
            f"  horizon={h}: train={len(dtr_h)} val={len(dva_h)} test={len(dte_h)}"
        )

        qgbm = fit_qgbm(pd.concat([dtr_h, dva_h]), dte_h, feats, QUANTILES)
        qar = fit_qar_per_series(
            pd.concat([dtr_h, dva_h]), dte_h, QUANTILES, lag_col="y_lag1"
        )
        dyn_cols = [c for c in feats_all if c != "series_lab"]
        xtft = fit_xtft_one_h(
            dtr_h,
            dva_h,
            dte_h,
            domain,
            QUANTILES,
            lookback=XTFT_LOOKBACK[domain],
            dyn_cols=dyn_cols,
        )

        pred_map = {
            "qgbm": qgbm,
            "qar": qar,
            "xtft" if XTFT_AVAILABLE else "xtft_unavailable": xtft,
        }
        p_df, m_df = eval_and_pack(dte_h, pred_map, domain, h)
        preds_all.append(p_df)
        mets_all.append(m_df)

    pred_df = pd.concat(preds_all, ignore_index=True)
    met_df = pd.concat(mets_all, ignore_index=True)

    if SAVE_JSON_PER_H:
        all_js = OUT_DIR / f"metrics_{domain}_all.json"
        with open(all_js, "w", encoding="utf-8") as f:
            recs = met_df.to_dict(orient="records")
            json.dump(recs, f, ensure_ascii=False, indent=2)

    _write_csv(pred_df, OUT_DIR / f"predictions_{domain}.csv")
    _write_csv(met_df, OUT_DIR / f"metrics_{domain}.csv")
    return pred_df, met_df


# --------------------------- MAIN ------------------------------------

if __name__ == "__main__":
    if not XTFT_AVAILABLE:
        print(
            "[info] XTFT unavailable. To enable it, install:\n"
            "       pip install fusionlab-learn tensorflow\n"
            "       Docs: https://fusion-lab.readthedocs.io/\n"
        )

    all_mets = []
    for dom in ["hydro", "subsidence", "wind"]:
        # XXX consider try/except per domain if running ad-hoc
        # try:
        _, m = run_domain(dom)
        all_mets.append(m)
        # except Exception as e:
        #     print(f"[WARN] {dom} failed: {e}")

    if all_mets:
        combo = pd.concat(all_mets, ignore_index=True)
        _write_csv(combo, OUT_DIR / "metrics_all_domains.csv")
        print("Saved:", OUT_DIR / "metrics_all_domains.csv")

    print("Done.")
