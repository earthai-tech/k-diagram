# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Build supervised CAS datasets from cleaned domain tables.

This module converts domain-cleaned CSVs (or in-memory DataFrames)
into time-indexed, feature-engineered tables ready for modeling.
It:
  * enforces a target sampling frequency (optional),
  * adds calendar/time features and lag/rolling statistics,
  * creates multi-horizon supervised targets (long format),
  * performs time-ordered splits per series (train/val/test),
  * writes outputs (Parquet preferred, CSV.GZ fallback),
  * and emits a manifest JSON describing the run.

Inputs are cleaned per-domain tables with minimal schema:
``domain``, ``series_id``, ``t`` (datetime-parseable), ``y`` (numeric).
Typical CAS layout:

    data/cas/
      raw/                 # raw inputs (from data sources)
      preprocessed/        # cleaned + supervised outputs (this module)
      modeling_results_ok/ # model predictions + metrics
      outputs/             # paper figures/tables

Main entry points
-----------------
- preprocess_dataframe(df, ...): core transform on a DataFrame.
- preprocess_cas_file(path, outdir, ...): CSV → outputs + manifest.
- preprocess_{wind,hydro,subsidence}(...): thin wrappers.

See the CLI at the bottom for reproducible runs from the shell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------- defaults ---------------------------------

DEFAULT_FREQ = "D"
DEFAULT_HORIZONS = [1, 3, 7, 14, 28]
DEFAULT_LAGS = [1, 2, 3, 7, 14, 28]
DEFAULT_ROLLS = [7, 14, 30]
DEFAULT_VAL = 0.15
DEFAULT_TEST = 0.15
DEFAULT_SEED = 13


# ---------------------------- utilities --------------------------------


def _safe_to_datetime(
    s: pd.Series,
    dayfirst: bool | None = None,
) -> pd.Series:
    # strip whitespace + common invisible marks
    s = s.astype(str)
    s = s.str.replace("\u200f", "", regex=False)
    s = s.str.strip()

    tries: list[dict[str, Any]] = []
    if dayfirst is not None:
        tries = [dict(dayfirst=dayfirst, errors="coerce")]
    else:
        tries = [
            dict(dayfirst=False, errors="coerce"),
            dict(dayfirst=True, errors="coerce"),
        ]
    out = None
    for kw in tries:
        out = pd.to_datetime(s, **kw)  # type: ignore
        if out.notna().mean() > 0.8:
            break
    if out is None:
        out = pd.to_datetime(s, errors="coerce")  # type: ignore
    return out


def _enforce_freq(
    df: pd.DataFrame,
    freq: str | None,
) -> pd.DataFrame:
    # Resample per (domain, series_id) if freq is set
    if not freq:
        return df
    parts: list[pd.DataFrame] = []
    for (d, sid), g in df.groupby(["domain", "series_id"], sort=False):
        g = g.set_index("t").sort_index()
        full = g.resample(freq).asfreq()
        for c in g.columns:
            full[c] = full[c].astype(g[c].dtype) if c in full else np.nan
        nums = g.select_dtypes(include=[np.number]).columns
        covars = [c for c in nums if c != "y"]
        full[covars] = full[covars].ffill()
        full["domain"] = d
        full["series_id"] = sid
        full = full.reset_index()
        tmin, tmax = g.index.min(), g.index.max()
        ok = (full["t"] >= tmin) & (full["t"] <= tmax)
        parts.append(full.loc[ok])
    return pd.concat(parts, ignore_index=True)


def _sin_cos_cycle(
    x: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    rad = 2.0 * np.pi * (x % period) / period
    return np.sin(rad), np.cos(rad)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["t"].dt.year
    df["month"] = df["t"].dt.month
    df["weekofyear"] = df["t"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["t"].dt.dayofyear
    df["dayofweek"] = df["t"].dt.dayofweek
    df["sin_doy"], df["cos_doy"] = _sin_cos_cycle(df["dayofyear"], 366)
    df["sin_dow"], df["cos_dow"] = _sin_cos_cycle(df["dayofweek"], 7)
    return df


def _add_lags_rolls(
    df: pd.DataFrame,
    lags: list[int],
    rolls: list[int],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.sort_values(["domain", "series_id", "t"]).copy()
    parts: list[pd.DataFrame] = []
    for (_d, _sid), g in df.groupby(["domain", "series_id"], sort=False):
        g = g.sort_values("t").copy()
        for L in lags:
            g[f"y_lag{L}"] = g["y"].shift(L)
        for W in rolls:
            g[f"y_roll{W}_mean"] = (
                g["y"].shift(1).rolling(W, min_periods=max(1, W // 3)).mean()
            )
            g[f"y_roll{W}_std"] = (
                g["y"].shift(1).rolling(W, min_periods=max(1, W // 3)).std()
            )
        parts.append(g)

    if not parts:
        # Keep schema consistent
        return df.copy()

    return pd.concat(parts, ignore_index=True)


def _make_supervised_long(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    df = df.sort_values(["domain", "series_id", "t"]).copy()
    for (_d, _sid), g in df.groupby(["domain", "series_id"], sort=False):
        g = g.sort_values("t").copy()
        for h in horizons:
            gh = g.copy()
            gh["horizon"] = h
            gh["y_future"] = g["y"].shift(-h).values
            pieces.append(gh)
    long = pd.concat(pieces, ignore_index=True)
    return long.dropna(subset=["y_future"])


def _time_splits_per_series(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (_d, _sid), g in df.groupby(["domain", "series_id"], sort=False):
        g = g.sort_values("t").copy()
        n = len(g)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = max(1, n - max(1, n_test))
            n_val = max(0, n - n_train - n_test)
        labels = np.array(
            (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test),
            dtype=object,
        )
        if len(labels) < n:
            pad = n - len(labels)
            labels = np.concatenate([labels, np.array(["test"] * pad)])
        g["split"] = labels
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _save_df(
    df: pd.DataFrame,
    base: Path,
    prefer_parquet: bool = True,
) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet:
        try:
            path = base.with_suffix(".parquet")
            df.to_parquet(path, index=False)
            return path
        except Exception:
            pass
    path = base.with_suffix(".csv.gz")
    df.to_csv(path, index=False, compression="gzip")
    return path


# ---------------------------- core pipeline ------------------------------


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    freq: str | None = DEFAULT_FREQ,
    horizons: list[int] = DEFAULT_HORIZONS,
    lags: list[int] = DEFAULT_LAGS,
    rolls: list[int] = DEFAULT_ROLLS,
    val_size: float = DEFAULT_VAL,
    test_size: float = DEFAULT_TEST,
    seed: int = DEFAULT_SEED,
    suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """
    Transform a cleaned domain table into supervised CAS datasets.

    Steps:
      1) Parse/validate timestamps and sort (domain, series_id, t).
      2) Optionally resample per series to ``freq`` within native span.
      3) Add calendar features and lag/rolling statistics (past-only).
      4) Create multi-horizon targets in long format.
      5) Assign time-ordered train/val/test splits per series.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned table with columns ``domain``, ``series_id``,
        ``t`` (datetime-like), and ``y`` (numeric). Additional
        numeric covariates are preserved and forward-filled when
        resampling.
    freq : str or None, default="D"
        Target sampling frequency (e.g., ``"D"``, ``"H"``). Use
        ``None`` to skip resampling.
    horizons : list of int, default=[1, 3, 7, 14, 28]
        Forecast horizons (steps ahead) for ``y_future`` targets.
    lags : list of int, default=[1, 2, 3, 7, 14, 28]
        Past lags (in time steps) for ``y_lag*`` features.
    rolls : list of int, default=[7, 14, 30]
        Window sizes for past-only rolling mean/std features.
    val_size : float, default=0.15
        Fraction for validation split per series.
    test_size : float, default=0.15
        Fraction for test split per series.
    seed : int, default=13
        Random seed placeholder (kept for API stability).
    suffix : str, default=""
        Suffix appended to dict keys to distinguish domains.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Keys:
          - ``f"combined_clean{suffix}"``: resampled/clean base.
          - ``f"features_wide{suffix}"``: feature-engineered wide.
          - ``f"supervised_long{suffix}"``: targets in long format.

    Raises
    ------
    ValueError
        If required columns are missing, timestamps are invalid,
        or transformations yield empty outputs.
    """
    needed = {"domain", "series_id", "t", "y"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["t"] = _safe_to_datetime(df["t"])
    if df["t"].isna().all():
        raise ValueError("All timestamps failed to parse; check 't' values.")

    df = df.dropna(subset=["t"]).copy()
    if df.empty:
        raise ValueError("No rows remain after dropping NaT in 't'.")

    df["domain"] = df["domain"].astype(str).str.strip().str.lower()
    df["series_id"] = df["series_id"].astype(str)

    df = df.sort_values(["domain", "series_id", "t"]).drop_duplicates(
        ["domain", "series_id", "t"]
    )

    combined_clean = _enforce_freq(df, freq)
    if combined_clean.empty:
        raise ValueError(
            "Empty after resampling. Try freq=None or a larger "
            "interval like '6D' for subsidence."
        )

    combined_clean = combined_clean.sort_values(
        ["domain", "series_id", "t"]
    ).reset_index(drop=True)

    feats = _add_time_features(combined_clean)
    feats = _add_lags_rolls(feats, lags, rolls)
    if feats.empty:
        raise ValueError(
            "Empty after lag/rolling. Reduce lags/rolls or ensure "
            "enough history per series."
        )

    lag_cols = [f"y_lag{L}" for L in lags]
    roll_mean = [f"y_roll{W}_mean" for W in rolls]
    roll_std = [f"y_roll{W}_std" for W in rolls]
    roll_cols = roll_mean + roll_std
    must_have = ["y"] + lag_cols + roll_cols
    need_cols = [c for c in must_have if c in feats.columns]
    features_wide = feats.dropna(subset=need_cols).copy()

    supervised_long = _make_supervised_long(features_wide, horizons)

    features_wide = _time_splits_per_series(
        features_wide, val_size, test_size, seed
    )
    supervised_long = _time_splits_per_series(
        supervised_long, val_size, test_size, seed
    )

    return {
        f"combined_clean{suffix}": combined_clean,
        f"features_wide{suffix}": features_wide,
        f"supervised_long{suffix}": supervised_long,
    }


def preprocess_cas_file(
    input_path: str | Path,
    outdir: str | Path,
    *,
    domain_hint: str | None = None,
    freq: str | None = DEFAULT_FREQ,
    horizons: list[int] = DEFAULT_HORIZONS,
    lags: list[int] = DEFAULT_LAGS,
    rolls: list[int] = DEFAULT_ROLLS,
    val_size: float = DEFAULT_VAL,
    test_size: float = DEFAULT_TEST,
    seed: int = DEFAULT_SEED,
    prefer_parquet: bool = True,
    suffix: str = "",
) -> dict[str, Any]:
    """
    CSV → supervised CAS datasets on disk + manifest.

    Reads a cleaned domain CSV, optionally injects a ``domain``
    column if missing (via ``domain_hint``), calls
    :func:`preprocess_dataframe`, writes outputs to ``outdir``,
    and saves a manifest JSON describing parameters/paths.

    Parameters
    ----------
    input_path : str or Path
        Path to a cleaned domain CSV.
    outdir : str or Path
        Directory to write outputs.
    domain_hint : str, optional
        Domain to inject if ``domain`` is absent
        (e.g., ``"wind"``, ``"hydro"``, ``"subsidence"``).
    freq, horizons, lags, rolls, val_size, test_size, seed :
        See :func:`preprocess_dataframe`.
    prefer_parquet : bool, default=True
        Write Parquet when possible, else CSV.GZ.
    suffix : str, default=""
        Suffix for file stems and manifest key names.

    Returns
    -------
    dict
        Manifest with script name, inputs, parameters,
        output paths, and schema summary.
    """
    input_path = Path(input_path)
    outdir = Path(outdir)
    df = pd.read_csv(input_path)

    if "domain" not in df.columns and domain_hint:
        df["domain"] = str(domain_hint).lower().strip()

    outputs = preprocess_dataframe(
        df,
        freq=freq,
        horizons=horizons,
        lags=lags,
        rolls=rolls,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        suffix=suffix,
    )

    comb_path = _save_df(
        outputs[f"combined_clean{suffix}"],
        outdir / f"combined_clean{suffix}",
        prefer_parquet,
    )
    wide_path = _save_df(
        outputs[f"features_wide{suffix}"],
        outdir / f"features_wide{suffix}",
        prefer_parquet,
    )
    long_path = _save_df(
        outputs[f"supervised_long{suffix}"],
        outdir / f"supervised_long{suffix}",
        prefer_parquet,
    )

    manifest = dict(
        script="preprocessing_cas_data.py",
        input=str(input_path),
        outdir=str(outdir),
        freq=freq,
        horizons=horizons,
        lags=lags,
        rolls=rolls,
        splits=dict(val_size=val_size, test_size=test_size),
        seed=seed,
        outputs=dict(
            combined=str(comb_path),
            features_wide=str(wide_path),
            supervised_long=str(long_path),
        ),
        schema=dict(
            minimal=["domain", "series_id", "t", "y"],
            generated_time_features=[
                "year",
                "month",
                "weekofyear",
                "dayofyear",
                "dayofweek",
                "sin_doy",
                "cos_doy",
                "sin_dow",
                "cos_dow",
            ],
            targets=dict(supervised_long="y_future"),
            split_column="split",
        ),
        notes=(
            "Rolling windows are past-only (shifted). "
            "Splits are time-ordered per series. "
            "Parquet preferred, CSV.GZ fallback."
        ),
    )
    with open(outdir / f"manifest.json{suffix}", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------- domain helpers ----------------------------


def preprocess_wind(
    input_path: str | Path,
    outdir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Wind wrapper for :func:`preprocess_cas_file`.

    Parameters
    ----------
    input_path : str or Path
        Path to ``wind_clean.csv``.
    outdir : str or Path
        Destination directory (e.g., ``data/cas/preprocessed``).
    **kwargs : Any
        Forwarded to :func:`preprocess_cas_file`.

    Returns
    -------
    dict
        Manifest describing outputs and parameters.
    """
    return preprocess_cas_file(
        input_path, outdir, domain_hint="wind", **kwargs
    )


def preprocess_hydro(
    input_path: str | Path,
    outdir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Hydro wrapper for :func:`preprocess_cas_file`.

    Parameters
    ----------
    input_path : str or Path
        Path to ``hydro_clean.csv``.
    outdir : str or Path
        Destination directory (e.g., ``data/cas/preprocessed``).
    **kwargs : Any
        Forwarded to :func:`preprocess_cas_file`.

    Returns
    -------
    dict
        Manifest describing outputs and parameters.
    """
    return preprocess_cas_file(
        input_path, outdir, domain_hint="hydro", **kwargs
    )


def preprocess_subsidence(
    input_path: str | Path,
    outdir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Subsidence wrapper for :func:`preprocess_cas_file`.

    Parameters
    ----------
    input_path : str or Path
        Path to ``subsidence_clean.csv``.
    outdir : str or Path
        Destination directory (e.g., ``data/cas/preprocessed``).
    **kwargs : Any
        Forwarded to :func:`preprocess_cas_file`.

    Returns
    -------
    dict
        Manifest describing outputs and parameters.
    """
    return preprocess_cas_file(
        input_path, outdir, domain_hint="subsidence", **kwargs
    )


# ---------------------------- CLI runner --------------------------------

if __name__ == "__main__":
    # Example absolute path usage (as in earlier drafts):
    # base = r'F:\repositories\k-diagram\_drafts\out\cas_data\prepared'

    import argparse

    def _csv_list(s: str) -> list[int]:
        return [int(x) for x in s.split(",") if x.strip()]

    parser = argparse.ArgumentParser(
        description=(
            "Build supervised CAS datasets from cleaned domain CSVs. "
            "Writes combined_clean, features_wide, supervised_long "
            "and a manifest per domain."
        )
    )
    parser.add_argument(
        "--in",
        dest="indir",
        type=Path,
        default=Path("data/cas/preprocessed"),
        help="Directory containing *clean.csv inputs.",
    )
    parser.add_argument(
        "--out",
        dest="outdir",
        type=Path,
        default=Path("data/cas/preprocessed"),
        help="Directory to write outputs (can be same as --in).",
    )
    parser.add_argument(
        "--wind",
        type=str,
        default="wind_clean.csv",
        help="Wind cleaned CSV filename.",
    )
    parser.add_argument(
        "--hydro",
        type=str,
        default="hydro_clean.csv",
        help="Hydro cleaned CSV filename.",
    )
    parser.add_argument(
        "--subs",
        type=str,
        default="subsidence_clean.csv",
        help="Subsidence cleaned CSV filename.",
    )
    parser.add_argument(
        "--wind-suffix",
        type=str,
        default="_wind",
        help="Suffix for wind outputs (stems & manifest keys).",
    )
    parser.add_argument(
        "--hydro-suffix",
        type=str,
        default="_hydro",
        help="Suffix for hydro outputs (stems & manifest keys).",
    )
    parser.add_argument(
        "--subs-suffix",
        type=str,
        default="_subsidence",
        help="Suffix for subsidence outputs (stems & keys).",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=DEFAULT_FREQ,
        help="Target sampling frequency (e.g., D, H). Use 'None' "
        "to skip resampling.",
    )
    parser.add_argument(
        "--horizons",
        type=_csv_list,
        default=DEFAULT_HORIZONS,
        help="Comma-separated horizons, e.g., 1,3,7,14,28",
    )
    parser.add_argument(
        "--lags",
        type=_csv_list,
        default=DEFAULT_LAGS,
        help="Comma-separated lags, e.g., 1,2,3,7,14,28",
    )
    parser.add_argument(
        "--rolls",
        type=_csv_list,
        default=DEFAULT_ROLLS,
        help="Comma-separated rolling windows, e.g., 7,14,30",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=DEFAULT_VAL,
        help="Validation fraction per series.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=DEFAULT_TEST,
        help="Test fraction per series.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed placeholder (kept for API stability).",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Disable Parquet outputs (write CSV.GZ only).",
    )
    args = parser.parse_args()

    freq = None if str(args.freq).lower() == "none" else args.freq
    prefer_parquet = not args.no_parquet

    args.outdir.mkdir(parents=True, exist_ok=True)

    wind_manifest = preprocess_wind(
        args.indir / args.wind,
        outdir=args.outdir,
        freq=freq,
        horizons=args.horizons,
        lags=args.lags,
        rolls=args.rolls,
        val_size=args.val,
        test_size=args.test,
        seed=args.seed,
        prefer_parquet=prefer_parquet,
        suffix=args.wind_suffix,
    )
    hydro_manifest = preprocess_hydro(
        args.indir / args.hydro,
        outdir=args.outdir,
        freq=freq,
        horizons=args.horizons,
        lags=args.lags,
        rolls=args.rolls,
        val_size=args.val,
        test_size=args.test,
        seed=args.seed,
        prefer_parquet=prefer_parquet,
        suffix=args.hydro_suffix,
    )
    subs_manifest = preprocess_subsidence(
        args.indir / args.subs,
        outdir=args.outdir,
        freq=freq,  # pass None here if desired for sparse data
        horizons=args.horizons,
        lags=args.lags,
        rolls=args.rolls,
        val_size=args.val,
        test_size=args.test,
        seed=args.seed,
        prefer_parquet=prefer_parquet,
        suffix=args.subs_suffix,
    )

    print("Manifests written:")
    for name, man in [
        ("wind", wind_manifest),
        ("hydro", hydro_manifest),
        ("subsidence", subs_manifest),
    ]:
        print(f"- {name}: {man['outputs']}")
