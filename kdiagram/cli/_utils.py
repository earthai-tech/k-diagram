# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

from __future__ import annotations

import argparse
import io
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np  # noqa
import pandas as pd

from ..core._io_utils import _normalize_ext
from ..core.io import read_data, write_data

PathLike = Union[str, Path]

FileLike = io.IOBase


__all__ = [
    "load_df",
    "save_df",
    "ensure_columns",
    "ensure_numeric",
    "parse_list",
    "parse_pairs",
    "parse_quantiles",
    "normalize_acov",
    "expand_prefix_cols",
    "detect_quantile_columns",
    "natural_sort_key",
]


# ---------- IO helpers -------------------------------------------------


def load_df(
    src: PathLike | FileLike,
    *,
    format: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    index_col: str | Iterable[str] | None = None,
    sort_index: bool = False,
    drop_na: str | Iterable[str] | None = None,
    fillna: Any | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Thin wrapper around core.read_data."""
    df = read_data(
        src,
        format=format,
        storage_options=storage_options,
        errors=errors,
        verbose=verbose,
        index_col=index_col,
        sort_index=sort_index,
        drop_na=drop_na,
        fillna=fillna,
        **kwargs,
    )
    if isinstance(df, pd.DataFrame):
        return df
    # If chunked/HTML-list slipped through, force a DataFrame
    return (
        pd.concat(list(df), ignore_index=True)
        if isinstance(df, Iterable)
        else pd.DataFrame()
    )


def save_df(
    df: pd.DataFrame,
    dest: str | Path | io.IOBase,
    *,
    format: str | None = None,
    errors: str = "raise",
    overwrite: bool = False,
    mkdirs: bool = True,
    index: bool | None = None,
    storage_options: Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    ext = _normalize_ext(dest, explicit=format)

    # Default CSV writes to *no* index unless user specifies one
    if index is None and ext == ".csv":
        index = False

    return write_data(
        df,
        dest,
        format=ext,
        errors=errors,
        overwrite=overwrite,
        mkdirs=mkdirs,
        index=index,
        storage_options=storage_options,
        **kwargs,
    )


# ---------- Schema / validation ---------------------------------------


def ensure_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    error: str = "raise",
    name: str | None = None,
) -> None:
    """Validate required columns exist."""
    need = list(cols)
    miss = [c for c in need if c not in df.columns]
    if miss and error == "raise":
        lbl = f" for {name}" if name else ""
        raise ValueError(f"Missing columns{lbl}: {miss}")
    if miss and error == "warn":
        import warnings

        warnings.warn(f"Missing columns: {miss}", stacklevel=2)


def ensure_numeric(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...],
    *,
    copy: bool = False,
    errors: str = "raise",
) -> pd.DataFrame:
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("errors must be one of {'raise','warn','ignore'}")

    out = df.copy() if copy else df
    cols = list(cols)

    introduced = False
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"Missing column: {c}")

        if errors == "raise":
            # raises on first non-numeric
            try:
                out[c] = pd.to_numeric(out[c], errors="raise")
            except Exception as e:
                raise TypeError(
                    f"Column '{c}' cannot be converted to numeric: {e}"
                ) from e
        else:
            # coerce + track new NaNs
            before = out[c].isna().to_numpy()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            after = out[c].isna().to_numpy()
            introduced = introduced or bool(np.any(~before & after))

    if introduced and errors == "warn":
        warnings.warn(
            "Non-numeric values were coerced to NaN.",
            UserWarning,
            stacklevel=2,
        )

    return out


# ---------- CLI parsing helpers ---------------------------------------


def parse_list(
    x: str | Sequence[str] | None,
    *,
    sep: str = ",",
    strip: bool = True,
    empty_as_none: bool = True,
) -> list[str] | None:
    """Parse comma lists or pass through lists."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        vals = list(map(str, x))
    else:
        vals = [t for t in str(x).split(sep)]
    vals = [v.strip() if strip else v for v in vals]
    vals = [v for v in vals if v != ""]
    if empty_as_none and not vals:
        return None
    return vals


def parse_pairs(
    x: str | Sequence[str | Sequence[str]] | None,
    *,
    out_len: int = 2,
    sep_outer: str = ";",
    sep_inner: str = ",",
) -> list[tuple[str, ...]] | None:
    """
    Parse "a,b;c,d" → [("a","b"),("c","d")].
    If a list is given, each item can be a tuple/list or "a,b".
    """
    if x is None:
        return None
    items: list[str | Sequence[str]] = (
        list(x) if isinstance(x, (list, tuple)) else str(x).split(sep_outer)
    )
    out: list[tuple[str, ...]] = []
    for it in items:
        if isinstance(it, (list, tuple)):
            t = tuple(map(str, it))
        else:
            t = tuple(s.strip() for s in str(it).split(sep_inner))
        if len(t) != out_len:
            raise ValueError(
                f"Pair requires {out_len} items, got {len(t)}: {t}"
            )
        out.append(t)
    return out


def parse_quantiles(
    q: str | Sequence[str | float] | None,
) -> list[float] | None:
    """Parse quantiles like '0.1,0.5,0.9' or '10%,90%'."""
    vals = parse_list(q)
    if vals is None:
        return None
    out: list[float] = []
    for v in vals:
        s = str(v).strip()
        s = s[:-1] if s.endswith("%") else s
        f = float(s)
        f = f / 100.0 if isinstance(v, str) and v.endswith("%") else f
        if not (0.0 < f < 1.0):
            raise ValueError(f"Quantile out of (0,1): {v}")
        out.append(f)
    return out


def normalize_acov(acov: str) -> str:
    """Return a validated angular coverage label."""
    allowed = {
        "default",
        "half_circle",
        "quarter_circle",
        "eighth_circle",
    }
    a = (acov or "default").lower()
    return a if a in allowed else "default"


def _flatten_cols(val: Any) -> list[str]:
    """
    Normalize CLI column inputs into a flat list of column names.

    Accepts:
      - None
      - "a,b" (CSV string)
      - ["a", "b"]
      - [["a", "b"], "c,d"]
    """
    if val is None:
        return []

    # Single string token
    if isinstance(val, str):
        return split_csv(val)

    flat: list[str] = []
    # List/iterable of tokens (strings or lists)
    for item in val:
        if isinstance(item, (list, tuple)):
            # Already tokenized
            for sub in item:
                if isinstance(sub, str):
                    flat.extend(split_csv(sub))
                else:
                    flat.append(str(sub))
        elif isinstance(item, str):
            flat.extend(split_csv(item))
        else:
            flat.append(str(item))
    return flat


# ---------- Column discovery / expansion -------------------------------


def natural_sort_key(s: str) -> tuple:
    """Split digits to help '2023', '2024' sort numerically."""
    rgx = re.compile(r"(\d+)")
    parts = rgx.split(str(s))
    key: list[Any] = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return tuple(key)


def expand_prefix_cols(
    prefix: str,
    horizons: Sequence[str | int],
    *,
    q: str,
) -> list[str]:
    """Build '{prefix}_{h}_q{q}' column names."""
    cols: list[str] = []
    for h in horizons:
        cols.append(f"{prefix}_{h}_q{q}")
    return cols


def detect_quantile_columns(
    df: pd.DataFrame,
    *,
    value_prefix: str | None = None,
    horizons: Sequence[str | int] | None = None,
) -> dict[str, list[str]]:
    """
    Detect q10/q50/q90 wide columns by pattern:
    '{prefix}_{h}_q{10|50|90}'.
    If prefix+horizons given, build and filter by presence.
    """
    q10: list[str] = []
    q50: list[str] = []
    q90: list[str] = []
    hz: list[str] = []

    if value_prefix is not None and horizons is not None:
        for h in horizons:
            c10 = f"{value_prefix}_{h}_q10"
            c50 = f"{value_prefix}_{h}_q50"
            c90 = f"{value_prefix}_{h}_q90"
            if c10 in df.columns:
                q10.append(c10)
            if c50 in df.columns:
                q50.append(c50)
            if c90 in df.columns:
                q90.append(c90)
        # derive horizons that actually exist
        hz = []
        for cols in (q10, q50, q90):
            for c in cols:
                tok = c.removeprefix(f"{value_prefix}_")
                tok = tok.removesuffix("_q10")
                tok = tok.removesuffix("_q50")
                tok = tok.removesuffix("_q90")
                hz.append(tok)
        hz = sorted(set(hz), key=natural_sort_key)
        return {"q10": q10, "q50": q50, "q90": q90, "horizons": hz}

    # Fallback: scan columns by regex
    rgx = re.compile(r"^(?P<p>.+)_(?P<h>[^_]+)_q(?P<q>10|50|90)$")
    found: dict[str, dict[str, list[str]]] = {}
    for c in df.columns:
        m = rgx.match(str(c))
        if not m:
            continue
        p = m.group("p")
        h = m.group("h")
        q = m.group("q")
        d = found.setdefault(p, {"10": [], "50": [], "90": [], "hz": []})
        d[q].append(c)
        d["hz"].append(h)

    # choose the prefix with most hits
    best_p = None
    best_n = -1
    for p, d in found.items():
        n = len(d["10"]) + len(d["50"]) + len(d["90"])
        if n > best_n:
            best_p, best_n = p, n

    if best_p is None:
        return {"q10": [], "q50": [], "q90": [], "horizons": []}

    d = found[best_p]
    hz = sorted(set(d["hz"]), key=natural_sort_key)
    return {
        "q10": sorted(d["10"], key=natural_sort_key),
        "q50": sorted(d["50"], key=natural_sort_key),
        "q90": sorted(d["90"], key=natural_sort_key),
        "horizons": hz,
    }


# --- parsing helpers (CLI-wide) ---------------------------------


@dataclass
class ModelSpec:
    name: str
    cols: list[str]


# ---------- small alias + parsing helpers ----------


def ns_get(ns: argparse.Namespace, *names: str, default=None):
    for n in names:
        if hasattr(ns, n) and getattr(ns, n) is not None:
            return getattr(ns, n)
    return default


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def parse_model_token(tok: str) -> ModelSpec:
    # accepts "name:col1,col2" or just "col1,col2"
    if ":" in tok:
        name, rest = tok.split(":", 1)
        cols = _split_csv(rest)
        return ModelSpec(name=name.strip(), cols=cols)
    cols = _split_csv(tok)
    nm = cols[0] if cols else "model"
    return ModelSpec(name=nm, cols=cols)


# ---------- auto detection from df ----------

_DEF_YTRUE = ["y_true", "actual", "y", "target", "truth", "label"]
_DEF_YPRED = ["y_pred", "pred", "prediction", "q50", "median"]
_Q_RE = re.compile(r"(.+?)_q(\d{1,3})$", flags=re.I)


def detect_y_true(df: pd.DataFrame) -> str | None:
    for c in _DEF_YTRUE:
        if c in df.columns:
            return c
    return None


def detect_single_pred(df: pd.DataFrame) -> str | None:
    for c in _DEF_YPRED:
        if c in df.columns:
            return c
    return None


def detect_quantile_groups(
    df: pd.DataFrame, levels: Iterable[int] = (10, 50, 90)
) -> list[list[str]]:
    # group by base using *_q{level}
    found: dict[str, dict[int, str]] = {}
    for col in df.columns:
        m = _Q_RE.match(col)
        if not m:
            continue
        base, lvl = m.group(1), int(m.group(2))
        found.setdefault(base, {})[lvl] = col
    groups: list[list[str]] = []
    lvls = list(levels)
    for _, mp in found.items():
        if all(lo in mp for lo in lvls):
            groups.append([mp[lu] for lu in lvls])
    return groups


# ---------- public resolvers used by CLIs ----------


def resolve_ytrue_preds(
    ns: argparse.Namespace, df: pd.DataFrame
) -> tuple[str, list[ModelSpec]]:
    """
    Supports:
      --actual-col / --y-true
      --pred COLS   (repeatable)
      --y-pred COL  (repeatable)
      --model NAME:COLS (repeatable)
    Falls back to auto-detect.
    """
    y_true = ns_get(ns, "actual_col", "y_true")
    if y_true is None:
        y_true = detect_y_true(df)
    if not y_true:
        raise SystemExit(
            "Could not resolve y_true. Use --actual-col "
            "or --y-true, or include a known column name."
        )

    specs: list[ModelSpec] = []

    # --model NAME:COLS (repeat)
    for tok in ns_get(ns, "model", default=[]) or []:
        specs.append(parse_model_token(tok))

    # --pred COLS (repeat, each may be a CSV)
    for tok in ns_get(ns, "pred", default=[]) or []:
        cols = _split_csv(tok)
        if cols:
            specs.append(ModelSpec(name=cols[0], cols=cols))

    # --y-pred COL (repeat)
    for col in ns_get(ns, "y_pred", default=[]) or []:
        specs.append(ModelSpec(name=str(col), cols=[str(col)]))

    # if nothing passed, try auto
    if not specs:
        qgroups = detect_quantile_groups(df)
        if qgroups:
            for g in qgroups:
                specs.append(ModelSpec(name=g[0], cols=g))
        else:
            sp = detect_single_pred(df)
            if sp:
                specs.append(ModelSpec(name=sp, cols=[sp]))

    if not specs:
        raise SystemExit(
            "Could not resolve predictions. Supply --model, "
            "--pred, or --y-pred; or make sure df has q* "
            "columns or a common pred column."
        )
    return y_true, specs


# ----------------------------- helpers ------------------------------
def split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_q_levels(
    s: str | None,
) -> list[float] | None:
    if not s:
        return None
    try:
        return [float(x) for x in split_csv(s)]
    except Exception as e:
        raise ValueError(f"Invalid --q-levels '{s}': {e}") from e


def _parse_float_list(
    s: object,
) -> list[float] | None:
    """
    Parse a list of floats from a CLI value that may be:
      - None
      - a CSV string: "1.1,0.8"
      - a list of tokens: ["1.1", "0.8"]
      - a mixed list with CSV tokens
    """
    if s is None:
        return None

    tokens: list[str] = []
    try:
        if isinstance(s, str):
            tokens = split_csv(s)
        elif isinstance(s, (list, tuple)):
            for item in s:
                if isinstance(item, str):
                    tokens.extend(split_csv(item))
                else:
                    tokens.append(str(item))
        else:
            # single scalar
            return [float(s)]
        return [float(x) for x in tokens]
    except Exception as e:
        raise ValueError(f"Invalid float list {s!r}: {e}") from e


def _infer_figsize(
    fs: list[float] | None,
) -> tuple[float, float] | None:
    if not fs:
        return None
    if len(fs) != 2:
        raise ValueError("--figsize expects two numbers: W H")
    return float(fs[0]), float(fs[1])


def _parse_models(
    specs: list[str],
) -> tuple[list[str], list[list[str]]]:
    names: list[str] = []
    groups: list[list[str]] = []
    for i, spec in enumerate(specs, start=1):
        if ":" in spec:
            name, cols = spec.split(":", 1)
            name = name.strip() or f"Model_{i}"
        else:
            name, cols = f"Model_{i}", spec
        col_list = split_csv(cols)
        if not col_list:
            raise ValueError(f"Empty columns in --model '{spec}'")
        names.append(name)
        groups.append(col_list)
    return names, groups


def _coerce_q_levels(val: Any) -> list[float]:
    """
    Accept either a CSV string (e.g. '0.1,0.5,0.9')
    or a list of tokens ['0.1','0.5','0.9'].
    Returns a list[float].
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    # fall back to coverage's parser for CSV strings
    return [float(x) for x in _parse_q_levels(val)]


def normalize_specs_as_quantiles(
    specs: list[ModelSpec],
    need_levels: Iterable[int] = (10, 50, 90),
) -> list[list[str]]:
    """
    Map ModelSpec -> ordered quantile cols if possible.
    If an item has exactly those *_q{lvl} cols, sort them by lvl.
    """
    lvls = list(need_levels)
    out: list[list[str]] = []
    for ms in specs:
        mp: dict[int, str] = {}
        for c in ms.cols:
            m = _Q_RE.match(c)
            if m:
                lvl = int(m.group(2))
                mp[lvl] = c
        if all(lo in mp for lo in lvls):
            out.append([mp[lu] for lu in lvls])
        else:
            out.append(ms.cols[:])
    return out


def parse_cols_pair(arg: str) -> list[str]:
    """
    Parse a pair: 'low,up'. Returns ['low', 'up'].
    """
    s = (arg or "").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 'lower,upper'.")
    return parts


def parse_figsize(arg: str | None) -> tuple[float, float] | None:
    """
    Parse 'W,H' or 'WxH' -> (float, float). None passes through.
    """
    if arg is None:
        return None
    s = str(arg).lower().replace("x", ",")
    pts = [p.strip() for p in s.split(",") if p.strip()]
    if len(pts) != 2:
        raise argparse.ArgumentTypeError("Use 'W,H' or 'WxH'.")
    try:
        return float(pts[0]), float(pts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid figsize.") from exc


def _collect_point_preds(
    df,
    ns: argparse.Namespace,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Collect 1D prediction arrays from flexible CLI specs.
    Enforces exactly one column per model/group.
    """
    specs = _collect_pred_specs(ns)
    if not specs:
        raise SystemExit("provide predictions via --model/--pred/--pred-cols")

    # expand any multi-col group to 1-col groups when not using --model
    used_model = bool(getattr(ns, "model", None))
    if not used_model and any(len(cols) > 1 for _, cols in specs):
        expanded = []
        k = 1
        for _, cols in specs:
            for col in cols:
                expanded.append((f"M{k}", [col]))
                k += 1
        specs = expanded

    # ensure each spec has exactly one column
    for name, cols in specs:
        if len(cols) != 1:
            raise SystemExit(
                f"group {name!r} must map to exactly one column; "
                "use one col per model for these plots."
            )

    # sets the private attribute ns._pred_groups
    ns._pred_groups = [(name, cols[0]) for name, cols in specs]

    need = [cols[0] for _, cols in specs]
    ensure_columns(df, need)
    ensure_numeric(df, need, copy=True, errors="raise")

    yps = [df[c].to_numpy(dtype=float) for c in need]
    names = ns.names if ns.names else _names_from_specs(specs)
    return yps, list(names)


def _collect_pred_specs(
    ns: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    """
    Merge prediction specs from --model, --pred/--pred-cols,
    and --q-cols. Returns [(name, [cols...]), ...].
    If --pred is flattened, chunk by len(--q-levels).
    """
    specs: list[tuple[str, list[str]]] = []

    # 1) --model NAME:col1[,col2,...]
    m = getattr(ns, "model", None)
    if m:
        m_names, m_groups = _parse_models(m)
        for name, grp in zip(m_names, m_groups):
            specs.append((name, list(grp)))

    def _groups_from(val) -> list[list[str]]:
        if not val:
            return []
        # already grouped: list-of-lists
        if (
            isinstance(val, list)
            and val
            and all(isinstance(x, list) for x in val)
        ):
            return [list(g) for g in val]
        # flat list or single CSV string -> one group
        if isinstance(val, list):
            return [list(val)]
        s = str(val)
        cols = [c.strip() for c in s.split(",") if c.strip()]
        return [cols]

    # 2) collect groups from --pred / --pred-cols / --q-cols
    groups: list[list[str]] = []
    groups += _groups_from(getattr(ns, "pred", None))
    groups += _groups_from(getattr(ns, "pred_cols", None))
    groups += _groups_from(getattr(ns, "q_cols", None))

    # 3) If we got exactly one flat group but it really
    #    represents multiple groups (flattened), split it by
    #    the number of quantiles if available and divisible.
    if len(groups) == 1:
        g0 = groups[0]
        q_raw = getattr(ns, "q_levels", None)
        if q_raw:
            # parse once, tolerate whitespace
            q_vals = _parse_q_levels(q_raw)
            q_len = len(q_vals)
            if q_len > 0 and len(g0) % q_len == 0 and len(g0) != q_len:
                # split into chunks of size q_len
                groups = [g0[i : i + q_len] for i in range(0, len(g0), q_len)]

    # 4) add unnamed groups with auto names
    for cols in groups:
        specs.append((f"M{len(specs) + 1}", list(cols)))

    return specs


def _names_from_specs(specs: list[tuple[str, list[str]]]) -> list[str]:
    names = []
    auto_idx = 1
    for nm, _ in specs:
        if nm:
            names.append(nm)
        else:
            names.append(f"Model {auto_idx}")
            auto_idx += 1
    return names


def add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_on: str,
    help_off: str,
) -> None:
    """
    Add --name / --no-name to toggle a boolean.
    """
    dest = name.replace("-", "_")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        f"--{name}",
        dest=dest,
        action="store_true",
        help=help_on,
    )
    grp.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=help_off,
    )
    parser.set_defaults(**{dest: default})


# parse a comma list: "a,b,c" -> ["a","b","c"]
def parse_cols_list(arg: str) -> list[str]:
    s = (arg or "").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "Expected a comma list, e.g. 'a,b,c'."
        )
    return parts


def _split_tokens(tokens: Iterable[str]) -> list[str]:
    """
    Expand a list of tokens that may contain commas into a flat
    list of trimmed strings. E.g. ["a,b", "c"] -> ["a", "b", "c"]
    """
    out: list[str] = []
    for t in tokens:
        part = str(t).strip()
        if not part:
            continue
        if "," in part:
            out.extend([p.strip() for p in part.split(",") if p.strip()])
        else:
            out.append(part)
    return out


def _coerce_val(s: str) -> Any:
    ls = s.strip().lower()
    if ls in {"true", "false"}:
        return ls == "true"
    try:
        if "." in ls:
            return float(ls)
        return int(ls)
    except ValueError:
        return s


def _parse_kv_list(tokens: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tok in _split_tokens(tokens or []):
        if "=" not in tok:
            raise argparse.ArgumentTypeError("Style item must be key=value.")
        k, v = tok.split("=", 1)
        k = k.strip()
        v = _coerce_val(v)
        if not k:
            raise argparse.ArgumentTypeError("Empty key in style.")
        out[k] = v
    return out


def _parse_norm_range(
    val: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if val is None:
        return None
    lo, hi = float(val[0]), float(val[1])
    return (lo, hi)


# ------------------------ test helpers  --------------------------------
def _expect_file(path: Path) -> None:
    assert path.exists(), f"missing: {path}"
    assert path.stat().st_size > 0, f"empty: {path}"


def _try_parse_and_run(variants: Iterable[list[str]]) -> None:
    """
    Try argv variants in order. If one parses and runs, stop.
    Raise the last error if all fail.
    """
    from . import build_parser

    last_err: BaseException | None = None
    for argv in variants:
        parser = build_parser()
        try:
            ns = parser.parse_args(argv)
            if not hasattr(ns, "func"):
                raise SystemExit("no func bound")
            ns.func(ns)
            return
        except SystemExit as e:
            last_err = e
        except Exception as e:  # pragma: no cover
            last_err = e
    if last_err:
        raise last_err


# --- flexible argparse actions -------------------------------------------


class ColumnsPairAction(argparse.Action):
    """
    Accept either:
      --q-cols low up
      --q-cols low,up
    and set dest -> ['low', 'up'].
    """

    def __call__(self, parser, ns, values, option_string=None):
        vals = values
        if isinstance(vals, str):
            parts = [p.strip() for p in vals.split(",") if p.strip()]
        else:
            parts = []
            for v in vals:
                if isinstance(v, str) and "," in v:
                    parts += [p.strip() for p in v.split(",") if p.strip()]
                else:
                    parts.append(str(v).strip())
        if len(parts) != 2:
            raise argparse.ArgumentError(
                self, "expected two columns: 'low,up' or 'low up'"
            )
        setattr(ns, self.dest, parts)


class ColumnsListAction(argparse.Action):
    """
    Accept columns as:
      --cols a b c
      --cols a,b,c
      --cols a b --cols c,d
    Always store a list[str] in the namespace.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        ns: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        parts: list[str] = []
        # values can be a str (no nargs) or a list[str] (with nargs="+")
        if isinstance(values, str):
            tokens = values.split(",")
            parts = [t.strip() for t in tokens if t.strip()]
        else:
            # list[str]; allow comma-separated inside each token
            for v in values:
                if isinstance(v, str):
                    parts.extend(
                        [t.strip() for t in v.split(",") if t.strip()]
                    )
                else:
                    parts.append(str(v).strip())

        # Merge with an already-set value (aliases / repeats)
        prev = getattr(ns, self.dest, None)
        if prev is None:
            out = parts
        elif isinstance(prev, (list, tuple)):
            out = list(prev) + parts
        else:
            # defensive: previous value was a single string
            out = [str(prev)] + parts

        setattr(ns, self.dest, out)


class FlexibleListAction(argparse.Action):
    """
    Accept lists via repeated tokens and/or CSV:
      --qlow-cols a b c
      --qlow-cols a,b,c
      --qlow-cols a b,c
    dest -> ['a','b','c']
    """

    def __call__(self, parser, ns, values, option_string=None):
        vals = values
        out: list[str] = []
        if isinstance(vals, str):
            chunks = [p.strip() for p in vals.split(",") if p.strip()]
            out.extend(chunks)
        else:
            for v in vals:
                s = str(v).strip()
                if "," in s:
                    out += [p.strip() for p in s.split(",") if p.strip()]
                elif s:
                    out.append(s)
        if not out:
            raise argparse.ArgumentError(self, "expects one or more names")
        setattr(ns, self.dest, out)


__all__ += [
    "ColumnsPairAction",
    "FlexibleListAction",
]
