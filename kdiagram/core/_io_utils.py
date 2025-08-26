# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)

from __future__ import annotations

import inspect
import io
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

# Typed overload support (PEP 692 / Unpack)
try:  # py>=3.12
    from typing import TypedDict, Unpack
except Exception:  # py<=3.11
    from typing_extensions import (  # type: ignore
        TypedDict,
        Unpack,
    )

__all__ = [
    "_BaseKwargs",
    "_ChunkKwargs",
    "_NoChunkKwargs",
    "Unpack",
    "_normalize_ext",
    "_handle_error",
    "_post_process",
    "_get_valid_kwargs",
]


class _BaseKwargs(TypedDict, total=False):
    """Arbitrary kwargs; used only for typing via Unpack."""


class _ChunkKwargs(_BaseKwargs, total=False):
    """Kwargs that include chunked reading (csv/json/sql)."""

    # Readers that support chunksize: read_csv/read_json/read_sql
    chunksize: int


class _NoChunkKwargs(_BaseKwargs, total=False):
    """Kwargs that do NOT include chunked reading."""

    # Absence of 'chunksize' helps type pick non-iterator branch
    pass


def _post_process(
    df: pd.DataFrame,
    index_col: str | Iterable[str] | None,
    sort_index: bool,
    drop_na: str | Iterable[str] | None,
    fillna: Any | Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Apply optional light cleaning steps to a DataFrame."""
    out = df

    # Fill missing values
    if fillna is not None:
        try:
            out = out.fillna(fillna)
        except Exception as e:
            warnings.warn(
                f"[read_data] fillna failed: {e}",
                stacklevel=3,
            )

    # Drop NA
    if drop_na is not None:
        try:
            if isinstance(drop_na, str):
                how = drop_na.lower()
                if how not in {"any", "all"}:
                    raise ValueError(
                        "drop_na must be 'any', 'all', or an "
                        "iterable of columns"
                    )
                out = out.dropna(how=how)
            else:
                # default to keeping rows if at least one of the
                # given columns is present -> drop only if all are NA
                out = out.dropna(
                    subset=list(drop_na),
                    how="all",
                )
        except Exception as e:
            warnings.warn(
                f"[read_data] dropna failed: {e}",
                stacklevel=3,
            )

    # Set index
    if index_col is not None:
        try:
            out = out.set_index(
                list(index_col)
                if isinstance(index_col, (list, tuple, set))
                else index_col
            )
        except Exception as e:
            warnings.warn(
                f"[read_data] set_index failed: {e}",
                stacklevel=3,
            )

    # Sort
    if sort_index:
        try:
            out = out.sort_index()
        except Exception as e:
            warnings.warn(
                f"[read_data] sort_index failed: {e}",
                stacklevel=3,
            )

    return out


def _normalize_ext(
    pathlike: Any,
    explicit: str | None = None,
) -> str | None:
    """Infer a normalized extension key for handlers."""
    if explicit:
        e = explicit.lower()
        if not e.startswith("."):
            e = "." + e
        return e

    # File-like: cannot infer without explicit format
    if hasattr(pathlike, "read") and not isinstance(
        pathlike, (str, bytes, Path)
    ):
        return None

    # BytesIO/StringIO
    if isinstance(pathlike, (io.BytesIO, io.StringIO)):
        return None

    p = Path(str(pathlike))

    # Handle compressed suffixes like .csv.gz -> use .csv
    comp = {".gz", ".bz2", ".xz", ".zip"}
    if len(p.suffixes) >= 2 and p.suffixes[-1].lower() in comp:
        return p.suffixes[-2].lower()

    return p.suffix.lower() if p.suffix else None


def _handle_error(msg: str, mode: str, stacklevel=3) -> None:
    if mode == "raise":
        raise ValueError(msg)
    elif mode == "warn":
        warnings.warn(msg, stacklevel=stacklevel)


def _get_valid_kwargs(
    callable_obj: Any,
    kwargs: dict[str, Any],
    error="ignore",
) -> dict[str, Any]:
    r"""
    Filter kwargs to what the callable accepts. If the callable
    accepts **kwargs (VAR_KEYWORD), return kwargs unchanged.
    """
    # If the callable_obj is an instance, use its class for sig
    if not inspect.isclass(callable_obj) and not callable(callable_obj):
        callable_obj = callable_obj.__class__

    try:
        sig = inspect.signature(callable_obj)
    except ValueError:
        msg = "Unable to inspect callable signature; passing no kwargs."
        _handle_error(msg, error, stacklevel=2)
        return {}

    # If function accepts **kwargs, don't filter
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ):
        return kwargs

    valid_params = set(sig.parameters.keys())
    valid_kwargs: dict[str, Any] = {}
    invalid: list[str] = []

    for k, v in kwargs.items():
        if k in valid_params:
            valid_kwargs[k] = v
        else:
            invalid.append(k)

    if invalid and error == "warn":
        warnings.warn(
            "Ignoring invalid keyword(s): " + ", ".join(invalid),
            stacklevel=2,
        )

    return valid_kwargs
