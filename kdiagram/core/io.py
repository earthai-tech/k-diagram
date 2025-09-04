# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import (
    Any,
    Literal,
    Union,
    overload,
)

import pandas as pd

from ._io_utils import (
    Unpack,
    _ChunkKwargs,
    _get_valid_kwargs,
    _handle_error,
    _NoChunkKwargs,
    _normalize_ext,
    _post_process,
)
from .property import PandasDataHandlers

PathLike = Union[str, Path]
FileLike = io.IOBase

# ------------------ Overloads ------------------


@overload
def read_data(
    source: PathLike | FileLike,
    *,
    format: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    html: Literal["list"] = "list",
    sql_con: Any | None = None,
    sql_params: Mapping[str, Any] | None = None,
    index_col: str | Iterable[str] | None = None,
    sort_index: bool = False,
    drop_na: str | Iterable[str] | None = None,
    fillna: Any | Mapping[str, Any] | None = None,
    **kwargs: Unpack[_NoChunkKwargs],
) -> list[pd.DataFrame]: ...


@overload
def read_data(
    source: PathLike | FileLike,
    *,
    format: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    html: Literal["first", "concat"] = "first",
    sql_con: Any | None = None,
    sql_params: Mapping[str, Any] | None = None,
    index_col: str | Iterable[str] | None = None,
    sort_index: bool = False,
    drop_na: str | Iterable[str] | None = None,
    fillna: Any | Mapping[str, Any] | None = None,
    **kwargs: Unpack[_NoChunkKwargs],
) -> pd.DataFrame: ...


@overload
def read_data(
    source: PathLike | FileLike,
    *,
    format: (
        Literal[
            "csv",
            ".csv",
            "json",
            ".json",
            "sql",
            ".sql",
        ]
        | None
    ) = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    html: Literal[
        "first",
        "concat",
        "list",
    ] = "first",
    sql_con: Any | None = None,
    sql_params: Mapping[str, Any] | None = None,
    index_col: str | Iterable[str] | None = None,
    sort_index: bool = False,
    drop_na: str | Iterable[str] | None = None,
    fillna: Any | Mapping[str, Any] | None = None,
    **kwargs: Unpack[_ChunkKwargs],
) -> Iterator[pd.DataFrame]: ...


@overload
def read_data(
    source: PathLike | FileLike,
    *,
    format: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    html: Literal[
        "first",
        "concat",
        "list",
    ] = "first",
    sql_con: Any | None = None,
    sql_params: Mapping[str, Any] | None = None,
    index_col: str | Iterable[str] | None = None,
    sort_index: bool = False,
    drop_na: str | Iterable[str] | None = None,
    fillna: Any | Mapping[str, Any] | None = None,
    **kwargs: Unpack[_NoChunkKwargs],
) -> pd.DataFrame: ...


def read_data(
    source: str | Path | io.IOBase,
    *,
    format: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    html: str = "first",
    sql_con: Any | None = None,
    sql_params: Mapping[str, Any] | None = None,
    index_col: str | Iterable[str] | None = None,
    sort_index: bool = False,
    drop_na: str | Iterable[str] | None = None,
    fillna: Any | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame | list[pd.DataFrame] | Iterable[pd.DataFrame]:
    # validate error policy early
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("errors must be one of {'raise','warn','ignore'}")

    # pick a parser from handlers based on extension
    handlers = PandasDataHandlers()
    ext = _normalize_ext(source, explicit=format)

    # if no extension can be inferred, bail or warn
    if ext is None:
        msg = (
            "Unable to infer format. Provide 'format' "
            "explicitly for file-like inputs or "
            "extension-less paths."
        )
        if errors == "raise":
            raise ValueError(msg)
        _handle_error(msg, errors)
        return pd.DataFrame()

    # normalize to dot-prefixed extension
    if not ext.startswith("."):
        ext = "." + ext

    # unsupported formats return empty or raise
    if ext not in handlers.parsers:
        avail = sorted(handlers.parsers.keys())
        msg = f"Unsupported format '{ext}'. Available: {avail}"
        if errors == "raise":
            raise ValueError(msg)
        _handle_error(msg, errors)
        return pd.DataFrame()

    # bind the pandas reader
    reader = handlers.parsers[ext]

    # plumb fsspec options when supported
    if storage_options is not None and "storage_options" not in kwargs:
        kwargs["storage_options"] = storage_options

    # main dispatch with html/sql specials
    try:
        # html returns a list; choose a strategy
        if ext == ".html":
            kwargs = _get_valid_kwargs(pd.read_html, kwargs)
            tables = pd.read_html(source, **kwargs)
            if verbose >= 2:
                print(f"[read_data] html: found {len(tables)} tables")
            if html == "first":
                out = tables[0] if tables else pd.DataFrame()
            elif html == "concat":
                out = (
                    pd.concat(tables, ignore_index=True)
                    if tables
                    else pd.DataFrame()
                )
            elif html == "list":
                out = tables
            else:
                raise ValueError(
                    "html must be one of {'first','concat','list'}"
                )
            # post-process when a single frame
            if isinstance(out, pd.DataFrame):
                out = _post_process(
                    out, index_col, sort_index, drop_na, fillna
                )
            return out

        # sql: read query text then execute via read_sql
        if ext == ".sql":
            if sql_con is None:
                msg = (
                    "Reading '.sql' requires 'sql_con' "
                    "(DB connection/engine)."
                )
                if errors == "raise":
                    raise ValueError(msg)
                _handle_error(msg, errors)
                return pd.DataFrame()

            if hasattr(source, "read") and not isinstance(
                source, (str, bytes, Path)
            ):
                sql_text = source.read()
            else:
                with open(
                    source,
                    encoding=kwargs.pop("encoding", "utf-8"),
                ) as f:
                    sql_text = f.read()

            out_df = pd.read_sql(
                sql_text, con=sql_con, params=sql_params, **kwargs
            )
            return _post_process(
                out_df, index_col, sort_index, drop_na, fillna
            )

        # generic path or file-like dispatch
        kwargs = _get_valid_kwargs(reader, kwargs)
        out_obj = reader(source, **kwargs)

        # readers may return an iterator (e.g., chunks)
        if not isinstance(out_obj, pd.DataFrame):
            # assume iterable of DataFrames
            return out_obj

        # normal post-process path
        return _post_process(out_obj, index_col, sort_index, drop_na, fillna)

    except Exception as exc:
        # consistent failure behavior
        msg = f"Failed to read '{source}': {exc}"
        if errors == "raise":
            raise
        _handle_error(msg, errors)
        return pd.DataFrame()


read_data.__doc__ = r"""
Read tabular data from a path/URL/file-like using the parsers
provided by :class:`PandasDataHandlers`.

The function infers the format from the file extension (or uses
``format``), dispatches to the corresponding pandas reader, and
applies optional post-processing (``fillna``, ``drop_na``,
``index_col``, sorting). Special care is taken for HTML (which
returns multiple tables) and for SQL files, which require a DB
connection.

Parameters
----------
source : str or Path or IOBase
    Path/URL to the file or an open file-like object. For file-like
    objects, you must specify ``format``.
format : str, optional
    Explicit format key (e.g., ``'csv'``, ``'.csv'``, ``'xlsx'``).
    If omitted, the extension is inferred from ``source``.
storage_options : mapping, optional
    Options for remote storage (fsspec-compatible), forwarded to
    the underlying pandas reader when supported.
errors : {'raise','warn','ignore'}, default='raise'
    Behavior on unsupported/undetected format or read errors.
    
    - ``'raise'`` : raise an exception
    - ``'warn'``  : emit a warning and return an empty DataFrame
    - ``'ignore'``: silently return an empty DataFrame
    
verbose : int, default=0
    Verbosity level. ``0`` silent, ``1`` basic, ``>=2`` chatty.

html : {'first','concat','list'}, default='first'
    Strategy when reading HTML (``pandas.read_html`` returns a list
    of DataFrames):
    - ``'first'``  : return the first table
    - ``'concat'`` : concatenate all tables (row-wise)
    - ``'list'``   : return the full list of tables

sql_con : any, optional
    SQLAlchemy engine or DBAPI connection. Required when
    reading a ``.sql`` file; the SQL query is read from the file
    and executed via ``pandas.read_sql``.
sql_params : mapping, optional
    Parameters forwarded to ``pandas.read_sql`` (e.g., params for
    parametrized queries).

index_col : str or iterable of str, optional
    Column(s) to set as index after reading.
sort_index : bool, default=False
    Sort the DataFrame by index after reading (if ``index_col`` is
    set) or by row index otherwise.
drop_na : {'any','all'} or iterable of str, optional
    Drop-NA policy:
    - ``'any'`` / ``'all'`` applies to **rows** after reading.
    - Iterable of column names applies ``dropna`` with ``subset``.
fillna : scalar or mapping, optional
    Value or column-wise mapping to fill missing values.

**kwargs
    Additional keyword arguments forwarded to the underlying pandas
    reader (e.g., ``dtype=``, ``parse_dates=``, ``encoding=``,
    ``sep=``, ``engine=`` for CSV, ``sheet_name=`` for Excel, etc.).

Returns
-------
pandas.DataFrame or list[pandas.DataFrame] or iterable of DataFrames

    - Most formats return a single ``DataFrame``.
    - HTML returns per ``html`` strategy.
    - If you pass a chunking parameter supported by the reader
      (e.g., ``chunksize`` for CSV), the underlying reader may
      return an **iterator** of ``DataFrame`` chunks.

Notes
-----
- **Format detection.** The format is inferred from the file
  extension. For compressed files like ``.csv.gz``, detection
  falls back to the base extension (``.csv``).
- **File-like inputs.** For file-like objects (e.g., ``BytesIO``),
  a ``format`` **must** be provided since there is no extension.
- **HTML.** ``pandas.read_html`` returns a list of tables. Use
  ``html='concat'`` to vertically concatenate them when they
  share columns.
- **SQL.** For ``.sql`` files, the file content is read and passed
  as a query to ``pandas.read_sql``; ``sql_con`` is required.
- **Post-processing.** Simple cleaning hooks are provided
  (``fillna``, ``drop_na``, ``index_col``, ``sort_index``) so that
  you can normalize shapes downstream.

See Also
--------
pandas.read_csv
pandas.read_excel
pandas.read_html
pandas.read_sql
pandas.DataFrame.fillna
pandas.DataFrame.dropna
pandas.DataFrame.set_index

References
----------
.. [1] McKinney, W.  *Python for Data Analysis*.  O’Reilly.
.. [2] pandas documentation.  IO tools.  https://pandas.pydata.org

Examples
--------
Read CSV with inferred delimiter::

    >>> df = read_data("data/events.csv")

Read Excel sheet by name::

    >>> df = read_data("data/book.xlsx", sheet_name="Sheet2")

Read HTML and concatenate all tables::

    >>> df = read_data("tables.html", html="concat")

Read a SQL file with an SQLAlchemy engine::

    >>> engine = create_engine("sqlite:///db.sqlite")
    >>> df = read_data("query.sql", sql_con=engine)

Read from an in-memory buffer by specifying format::

    >>> import io
    >>> buf = io.StringIO("a,b\\n1,2\\n3,4\\n")
    >>> df = read_data(buf, format="csv")
"""


@overload
def write_data(
    df: pd.DataFrame,
    dest: None,
    *,
    format: Literal["str", ".str"],
    **kwargs: object,
) -> str: ...
@overload
def write_data(
    df: pd.DataFrame,
    dest: object,
    *,
    format: Literal["dict", ".dict"],
    **kwargs: object,
) -> dict[str, object]: ...
@overload
def write_data(
    df: pd.DataFrame,
    dest: object,
    *,
    format: Literal["clip", ".clip"],
    **kwargs: object,
) -> None: ...
@overload
def write_data(
    df: pd.DataFrame,
    dest: object,
    *,
    format: Literal["gbq", ".gbq"],
    **kwargs: object,
) -> object: ...
@overload
def write_data(
    df: pd.DataFrame,
    dest: object,
    *,
    format: str | None = None,
    **kwargs: object,
) -> Path | None: ...


def write_data(
    df: pd.DataFrame,
    dest: str | Path | io.IOBase | None,
    *,
    format: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    errors: str = "raise",
    verbose: int = 0,
    overwrite: bool = False,
    mkdirs: bool = True,
    index: bool | None = None,
    # SQL specifics
    sql_con: Any | None = None,
    sql_table: str | None = None,
    # passthrough to writers
    **kwargs: Any,
) -> Path | None | Any:
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("errors must be one of {'raise','warn','ignore'}")

    ext = _normalize_ext(dest, explicit=format)
    if ext is None:
        msg = (
            "Unable to infer format. Provide 'format' explicitly "
            "for file-like or when 'dest' is None."
        )
        if errors == "raise":
            raise ValueError(msg)
        _handle_error(msg, errors)
        return None

    if not ext.startswith("."):
        ext = "." + ext

    handlers = PandasDataHandlers()
    writers = handlers.writers(df)

    if ext not in writers:
        avail = sorted(writers.keys())
        msg = f"Unsupported format '{ext}'. Available: {avail}"
        if errors == "raise":
            raise ValueError(msg)
        _handle_error(msg, errors)
        return None

    writer = writers[ext]

    call_kw = dict(kwargs)
    if storage_options is not None and "storage_options" not in call_kw:
        call_kw["storage_options"] = storage_options

    if index is not None:
        call_kw.setdefault("index", index)

    def _prepare_path(p: Path) -> None:
        if p.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {p}")
        if mkdirs:
            p.parent.mkdir(parents=True, exist_ok=True)

    try:
        # --- SQL ---
        if ext == ".sql":
            if sql_con is None or not sql_table:
                msg = "Writing '.sql' requires 'sql_con' and 'sql_table'."
                if errors == "raise":
                    raise ValueError(msg)
                _handle_error(msg, errors)
                return None
            call_kw = _get_valid_kwargs(writer, call_kw)
            return writer(name=sql_table, con=sql_con, **call_kw)

        # --- dict export: default to orient='list' ---
        if ext == ".dict":
            call_kw.setdefault("orient", "list")
            call_kw = _get_valid_kwargs(writer, call_kw)
            obj = writer(**call_kw)
            if dest is not None:
                warnings.warn(
                    "[write_data] '.dict' ignores path; returning "
                    "Python object.",
                    stacklevel=2,
                )
            return obj

        # --- records export ---
        if ext == ".rec":
            call_kw = _get_valid_kwargs(writer, call_kw)
            obj = writer(**call_kw)
            if isinstance(dest, (str, Path, io.IOBase)):
                warnings.warn(
                    "[write_data] '.rec' ignores path; returning recarray.",
                    stacklevel=2,
                )
            return obj

        # --- string export ---
        if ext == ".str":
            call_kw = _get_valid_kwargs(writer, call_kw)
            text = writer(**call_kw)
            if dest is None:
                return text
            if isinstance(dest, io.IOBase):
                dest.write(text)
                return None
            p = Path(dest)
            _prepare_path(p)
            p.write_text(text, encoding=call_kw.get("encoding", "utf-8"))
            return p

        # --- clipboard ---
        if ext == ".clip":
            call_kw = _get_valid_kwargs(writer, call_kw)
            writer(**call_kw)
            return None

        # --- BigQuery ---
        if ext == ".gbq":
            call_kw = _get_valid_kwargs(writer, call_kw)
            return writer(**call_kw)

        # --- generic file-like or path-based writers ---
        if dest is None:
            msg = f"Destination is required for '{ext}' export."
            if errors == "raise":
                raise ValueError(msg)
            _handle_error(msg, errors)
            return None

        if isinstance(dest, io.IOBase):
            call_kw = _get_valid_kwargs(writer, call_kw)
            writer(dest, **call_kw)
            return None

        p = Path(dest)
        _prepare_path(p)

        # JSON path write: call then return Path
        if ext == ".json":
            call_kw = _get_valid_kwargs(writer, call_kw)
            writer(p, **call_kw)
            return p

        # All other path-based writers: call then return Path
        call_kw = _get_valid_kwargs(writer, call_kw)
        writer(p, **call_kw)
        return p

    except FileExistsError:
        if errors == "raise":
            raise
        _handle_error("Destination exists and overwrite=False.", errors)
        return None
    except Exception as exc:
        msg = f"Failed to write '{dest}': {exc}"
        if errors == "raise":
            raise
        _handle_error(msg, errors)
        return None


write_data.__doc__ = r"""
Write a ``DataFrame`` to disk, a buffer, or an external
service using writers advertised by
:class:`PandasDataHandlers`.  The function infers the
format from the destination suffix, or uses ``format``
when provided.  It manages directory creation, safe
overwrites, storage options, and offers special handling
for formats with non-path semantics.

Depending on the format and destination, the return value
may be a path, ``None``, or an object produced by the
underlying writer (e.g., a Python ``dict`` or a GBQ job).

Parameters
----------
df : pandas.DataFrame
    The frame to export.
dest : str or pathlib.Path or IOBase or None
    Destination path, file-like buffer, or ``None``.  Some
    formats return a value when ``dest`` is ``None`` (e.g.
    ``'.str'`` returns a string).
format : str or None, optional
    Explicit format key, e.g. ``'csv'``, ``'.parq'``,
    ``'xlsx'``.  If ``None``, inferred from ``dest``.
storage_options : mapping, optional
    fsspec compatible options for remote filesystems.
errors : {'raise','warn','ignore'}, default='raise'
    Policy when the format is unsupported or writing fails.
    ``'raise'`` raises, ``'warn'`` warns and returns ``None``,
    ``'ignore'`` is silent and returns ``None``.
verbose : int, default=0
    Verbosity level.  ``0`` silent, ``1`` basic, ``>=2`` more
    details about writer choices.
overwrite : bool, default=False
    If ``False`` and the destination exists, a
    ``FileExistsError`` is raised (or downgraded per policy).
mkdirs : bool, default=True
    Create parent directories if they do not exist.
index : bool or None, optional
    Forwarded as ``index=`` when supported by the writer.
sql_con : any, optional
    SQLAlchemy engine or DBAPI connection.  Required for
    ``'.sql'`` writes via ``DataFrame.to_sql``.
sql_table : str or None, optional
    Target table name for SQL writes.
**kwargs
    Extra writer options forwarded to pandas writers, e.g.
    ``sep``, ``encoding``, ``mode``, ``if_exists``,
    ``compression``, ``engine``, or format-specific flags.

Returns
-------
pathlib.Path or None or object
    - ``Path`` for path-based writers on success.
    - ``None`` for sink-like writers (clipboard) or when the
      policy returns no value.
    - A writer-specific object for some formats, e.g. a
      Python ``dict`` for ``'.dict'``, a string for
      ``'.str'``, or a GBQ job for ``'.gbq'``.

Raises
------
ValueError
    If policy is ``'raise'`` and arguments are inconsistent,
    the format is unsupported, or an error occurs.
FileExistsError
    When the destination exists and ``overwrite`` is ``False``
    (unless downgraded by policy).

Notes
-----
- Writers are selected from ``PandasDataHandlers.writers(df)``.
- For formats that ignore paths (e.g. ``'.dict'``, ``'.rec'``,
  ``'.clip'``, ``'.gbq'``), the function returns the writer’s
  object or ``None`` as appropriate.
- ``'.sql'`` writes require both ``sql_con`` and ``sql_table``,
  and delegate to ``DataFrame.to_sql``.
- Parent directories are created when ``mkdirs=True``.
- ``storage_options`` is forwarded when supported by pandas.

Examples
--------
Write CSV with UTF-8 encoding and index included::

    >>> p = write_data(df, "out/data.csv",
    ...                encoding="utf-8", index=True)

Write to Excel, creating parent folders as needed::

    >>> p = write_data(df, "out/report.xlsx", overwrite=True)

Export to JSON with indentation::

    >>> p = write_data(df, "out/data.json", indent=2)

Write SQL to a SQLite database::

    >>> from sqlalchemy import create_engine
    >>> eng = create_engine("sqlite:///db.sqlite")
    >>> _ = write_data(df, "table.sql", format="sql",
    ...                sql_con=eng, sql_table="my_table",
    ...                if_exists="replace")

Get a string rendering without touching the FS::

    >>> s = write_data(df, None, format="str")
    >>> assert isinstance(s, str)

See Also
--------
pandas.DataFrame.to_csv
pandas.DataFrame.to_excel
pandas.DataFrame.to_sql
pandas.DataFrame.to_parquet
pandas.DataFrame.to_json

References
----------
.. [1] McKinney, W.  *Python for Data Analysis*.  O’Reilly.
.. [2] pandas documentation.  IO tools.  https://pandas.pydata.org
"""
