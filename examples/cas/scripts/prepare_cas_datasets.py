# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Prepare CAS raw datasets into normalized, time-indexed tables.

This script ingests domain-specific raw CSV files for the CAS study
(wind, hydrology, subsidence), standardizes key columns, and writes
clean outputs (CSV and optional Parquet) for downstream modeling.

Key responsibilities
--------------------
* Parse and validate minimal schema per domain:
  - columns: ``domain``, ``series_id``, ``t`` (datetime), ``y`` (value)
* Robust datetime parsing (supports mixed formats and yyyymmdd ints)
* Numeric coercion of target and optional covariates
* Sorted output by (``series_id``, ``t``) with stable indexing
* Writes to ``out_dir`` with predictable filenames:
  - ``wind_clean.csv/.parquet``
  - ``hydro_clean.csv/.parquet``
  - ``subsidence_clean.csv/.parquet``

Outputs are designed to feed the next step that builds supervised
frames (lags, horizons) before model training.

Quick usage
-----------
Command line::

    python examples/cas/scripts/prepare_cas_datasets.py \
      --raw data/cas/raw \
      --out data/cas/preprocessed

Programmatic::

    from pathlib import Path
    from prepare_cas_datasets import prepare_all
    out = prepare_all(
        Path("data/cas/raw/gefcom_hourly.csv"),
        Path("data/cas/raw/camels_timeseries.csv"),
        Path("data/cas/raw/egms_point.csv"),
        out_dir=Path("data/cas/preprocessed"),
    )
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_dt_from_tokens(
    y: pd.Series,
    m: pd.Series,
    d: pd.Series,
    h: pd.Series | None = None,
) -> pd.Series:
    if h is None:
        return pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
    return pd.to_datetime(
        dict(year=y, month=m, day=d, hour=h), errors="coerce"
    )


def _parse_dt(s: pd.Series, fmt: str | None = None) -> pd.Series:
    if fmt is not None:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    # fast paths then fallback
    out = pd.to_datetime(
        s, errors="coerce"
    )  #  infer_datetime_format=True deprecated
    # handle bare integers like 20190105
    mask_int = out.isna() & s.astype(str).str.fullmatch(r"\d{8}").fillna(
        False
    )
    if mask_int.any():
        out.loc[mask_int] = pd.to_datetime(
            s[mask_int].astype(str), format="%Y%m%d", errors="coerce"
        )
    return out


# ---------- WIND (GEFCom) -------------------------------------


def prepare_wind(
    in_csv: str | Path,
    out_csv: str | Path,
    out_parquet: str | Path | None = None,
) -> pd.DataFrame:
    """
    Clean and standardize the wind (GEFCom-style) dataset.

    Expects a tidy table with at least the columns:
    ``domain``, ``series_id``, ``t``, ``y``. Optionally accepts
    the covariate ``T`` if present. Produces a sorted DataFrame
    with coerced types and writes CSV/Parquet.

    Parameters
    ----------
    in_csv : str or Path
        Path to the raw wind CSV file.
    out_csv : str or Path
        Output path for the cleaned CSV.
    out_parquet : str or Path, optional
        Output path for the cleaned Parquet. If ``None``,
        Parquet is skipped.

    Returns
    -------
    pandas.DataFrame
        Cleaned, time-sorted wind frame with columns:
        ``domain``, ``series_id``, ``t``, ``y`` (and optional
        ``T``).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(in_csv)
    need = {"domain", "series_id", "t", "y"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Missing cols: {sorted(miss)}")

    df["t"] = _parse_dt(df["t"])
    df["y"] = _to_num(df["y"])
    if "T" in df.columns:
        df["T"] = _to_num(df["T"])

    df = df.dropna(subset=["t", "y"]).sort_values(["series_id", "t"])
    keep = ["domain", "series_id", "t", "y"]
    for c in ["T"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].reset_index(drop=True)

    out_csv = Path(out_csv)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)

    if out_parquet is not None:
        pq = Path(out_parquet)
        _ensure_dir(pq)
        df.to_parquet(pq, index=False)

    return df


# ---------- HYDRO (CAMELS) ------------------------------------


def prepare_hydro(
    in_csv: str | Path,
    out_csv: str | Path,
    out_parquet: str | Path | None = None,
) -> pd.DataFrame:
    """
    Clean and standardize the hydrology (CAMELS-style) dataset.

    Expects a tidy table with at least the columns:
    ``domain``, ``series_id``, ``t``, ``y``. Optionally accepts
    covariates ``p_mm_day`` and ``t_c`` if present. Produces a
    sorted DataFrame with coerced types and writes CSV/Parquet.

    Parameters
    ----------
    in_csv : str or Path
        Path to the raw hydrology CSV file.
    out_csv : str or Path
        Output path for the cleaned CSV.
    out_parquet : str or Path, optional
        Output path for the cleaned Parquet. If ``None``,
        Parquet is skipped.

    Returns
    -------
    pandas.DataFrame
        Cleaned, time-sorted hydro frame with columns:
        ``domain``, ``series_id``, ``t``, ``y`` (and optional
        ``p_mm_day``, ``t_c``).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(in_csv)
    need = {"domain", "series_id", "t", "y"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Missing cols: {sorted(miss)}")

    df["t"] = _parse_dt(df["t"])
    df["y"] = _to_num(df["y"])

    for c in ["p_mm_day", "t_c"]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    df = df.dropna(subset=["t", "y"]).sort_values(["series_id", "t"])
    keep = ["domain", "series_id", "t", "y"]
    for c in ["p_mm_day", "t_c"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].reset_index(drop=True)

    out_csv = Path(out_csv)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)

    if out_parquet is not None:
        pq = Path(out_parquet)
        _ensure_dir(pq)
        df.to_parquet(pq, index=False)

    return df


# ---------- SUBSIDENCE (EGMS points) --------------------------


def prepare_subsidence(
    in_csv: str | Path,
    out_csv: str | Path,
    *,
    date_col: str | None = None,
    time_col: str | None = "t",
    out_parquet: str | Path | None = None,
) -> pd.DataFrame:
    """
    Clean and standardize the subsidence (EGMS-style) dataset.

    Performs robust column mapping for ``series_id``, ``y``, and
    time fields. If the file separates date/time, pass
    ``date_col`` and ``time_col`` to build a timestamp. If
    ``domain`` is absent, injects ``'subsidence'``. Keeps optional
    metadata columns (``product``, ``lat``, ``lon``) if present.

    Parameters
    ----------
    in_csv : str or Path
        Path to the raw subsidence CSV file.
    out_csv : str or Path
        Output path for the cleaned CSV.
    date_col : str, optional
        Name of a date-like column (e.g., yyyymmdd) to combine
        with time when needed.
    time_col : str, optional
        Name of a time-like column to combine with date. Default
        is ``'t'``.
    out_parquet : str or Path, optional
        Output path for the cleaned Parquet. If ``None``,
        Parquet is skipped.

    Returns
    -------
    pandas.DataFrame
        Cleaned, time-sorted subsidence frame with columns:
        ``domain``, ``series_id``, ``t``, ``y`` (and optional
        ``product``, ``lat``, ``lon``).

    Raises
    ------
    ValueError
        If minimal columns cannot be inferred.
    """
    df = pd.read_csv(in_csv)

    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"series_id", "point_id", "id"}:
            colmap["series_id"] = c
        elif cl in {"y", "disp", "displacement", "value"}:
            colmap["y"] = c
        elif cl in {"domain"}:
            colmap["domain"] = c
        elif cl in {"t", "time"}:
            colmap["t"] = c

    for need in ["series_id", "y"]:
        if need not in colmap:
            raise ValueError(f"Cannot find '{need}' in {list(df.columns)}")

    if date_col is not None and time_col is not None:
        date_raw = df[date_col]
        time_raw = df[time_col] if time_col in df.columns else "00:00:00"
        d = _parse_dt(date_raw)
        t = pd.to_datetime(time_raw, errors="coerce").dt.time
        t = t.fillna(pd.to_datetime("00:00").time())
        dt = pd.to_datetime(
            d.dt.date.astype(str) + " " + t.astype(str), errors="coerce"
        )
        df["t"] = dt
    else:
        tcol = colmap.get("t")
        if tcol is None:
            for cand in ["date", "Date", "DATE", "yyyymmdd"]:
                if cand in df.columns:
                    df["t"] = _parse_dt(df[cand])
                    break
        else:
            df["t"] = _parse_dt(df[tcol])

    df["y"] = _to_num(df[colmap["y"]])
    df["series_id"] = df[colmap["series_id"]].astype(str)

    keep_extra = []
    for c in ["product", "lat", "lon"]:
        if c in df.columns:
            keep_extra.append(c)

    if "domain" not in colmap and "domain" not in df.columns:
        df["domain"] = "subsidence"

    df = df.dropna(subset=["t", "y"]).sort_values(["series_id", "t"])

    keep = ["domain", "series_id", "t", "y"] + keep_extra
    df = df[keep].reset_index(drop=True)

    out_csv = Path(out_csv)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)

    if out_parquet is not None:
        pq = Path(out_parquet)
        _ensure_dir(pq)
        df.to_parquet(pq, index=False)

    return df


# ---------- ONE-SHOT DRIVER -----------------------------------


def prepare_all(
    wind_csv: str | Path,
    hydro_csv: str | Path,
    subs_csv: str | Path,
    out_dir: str | Path = "prepared",
) -> dict[str, pd.DataFrame]:
    """
    One-shot driver to prepare all three domains.

    Runs ``prepare_wind``, ``prepare_hydro``, and
    ``prepare_subsidence`` with consistent output naming under
    ``out_dir``.

    Parameters
    ----------
    wind_csv : str or Path
        Raw wind CSV (GEFCom-style).
    hydro_csv : str or Path
        Raw hydrology CSV (CAMELS-style).
    subs_csv : str or Path
        Raw subsidence CSV (EGMS-style).
    out_dir : str or Path, default="prepared"
        Output directory where the cleaned files will be written.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping with keys ``'wind'``, ``'hydro'``, ``'subsidence'``
        and cleaned DataFrames as values.
    """
    out_dir = Path(out_dir)
    out = {}

    out["wind"] = prepare_wind(
        wind_csv,
        out_dir / "wind_clean.csv",
        out_dir / "wind_clean.parquet",
    )
    out["hydro"] = prepare_hydro(
        hydro_csv,
        out_dir / "hydro_clean.csv",
        out_dir / "hydro_clean.parquet",
    )
    out["subsidence"] = prepare_subsidence(
        subs_csv,
        out_dir / "subsidence_clean.csv",
        out_parquet=out_dir / "subsidence_clean.parquet",
    )
    return out


if __name__ == "__main__":
    # Example absolute path usage (as in your draft):
    # base = r'F:\repositories\k-diagram\_drafts\out\cas_data'

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare CAS raw datasets (wind/hydro/subsidence) into "
            "normalized CSV/Parquet tables."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/cas/raw"),
        help="Directory containing the raw CSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/cas/preprocessed"),
        help="Directory to write cleaned outputs.",
    )
    parser.add_argument(
        "--wind",
        type=str,
        default="gefcom_hourly.csv",
        help="Filename for the wind raw CSV.",
    )
    parser.add_argument(
        "--hydro",
        type=str,
        default="camels_timeseries.csv",
        help="Filename for the hydrology raw CSV.",
    )
    parser.add_argument(
        "--subs",
        type=str,
        default="egms_point.csv",
        help="Filename for the subsidence raw CSV.",
    )
    args = parser.parse_args()

    wind_csv = args.raw / args.wind
    hydro_csv = args.raw / args.hydro
    subs_csv = args.raw / args.subs

    if (
        not wind_csv.exists()
        or not hydro_csv.exists()
        or not subs_csv.exists()
    ):
        raise SystemExit(
            f"Missing inputs. Expected:\n"
            f"  {wind_csv}\n  {hydro_csv}\n  {subs_csv}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    out = prepare_all(wind_csv, hydro_csv, subs_csv, out_dir=args.out)

    print(
        "Prepared datasets:\n"
        f"- wind:        {len(out['wind']):>7} rows\n"
        f"- hydro:       {len(out['hydro']):>7} rows\n"
        f"- subsidence:  {len(out['subsidence']):>7} rows\n"
        f"Written under: {args.out}"
    )
