#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utility functions for quantile extraction and validation.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

from ..decorators import SaveFile, check_non_emptiness
from .diagnose_q import validate_quantiles
from .generic_utils import error_policy
from .handlers import columns_manager
from .validator import check_spatial_columns, exist_features, is_frame

__all__ = ["reshape_quantile_data", "melt_q_data", "pivot_q_data"]


@SaveFile
@check_non_emptiness
def reshape_quantile_data(
    df: pd.DataFrame,
    value_prefix: str,
    spatial_cols: list[str] | None = None,
    dt_col: str = "year",
    error: str = "warn",
    savefile: str | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    is_frame(df, df_only=True, objname="Data 'df'")

    if spatial_cols:
        missing_spatial = set(spatial_cols) - set(df.columns)
        if missing_spatial:
            msg = f"Missing spatial columns: {missing_spatial}"
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                warnings.warn(msg, stacklevel=2)
            spatial_cols = list(set(spatial_cols) & set(df.columns))

    # Find quantile columns
    quant_cols = [col for col in df.columns if col.startswith(value_prefix)]

    if not quant_cols:
        msg = f"No columns found with prefix '{value_prefix}'"
        if error == "raise":
            raise ValueError(msg)
        elif error == "warn":
            warnings.warn(msg, stacklevel=2)
        return pd.DataFrame()

    # Extract metadata from column names
    pattern = re.compile(rf"{re.escape(value_prefix)}_(\d{{4}})_q([0-9.]+)$")

    meta = []
    valid_cols = []
    for col in quant_cols:
        match = pattern.match(col)
        if match:
            year, quantile = match.groups()
            meta.append((col, int(year), float(quantile)))
            valid_cols.append(col)

    if verbose >= 1:
        print(f"Found {len(valid_cols)} valid quantile columns")

    if not valid_cols:
        return pd.DataFrame()

    # Melt dataframe
    id_vars = spatial_cols if spatial_cols else []
    melt_df = df.melt(
        id_vars=id_vars,
        value_vars=valid_cols,
        var_name="column",
        value_name="value",
    )

    # Add metadata columns
    meta_df = pd.DataFrame(meta, columns=["column", dt_col, "quantile"])
    melt_df = melt_df.merge(meta_df, on="column")

    # Pivot to wide format
    pivot_df = melt_df.pivot_table(
        index=id_vars + [dt_col],
        columns="quantile",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Clean column names
    pivot_df.columns = [
        f"{value_prefix}_q{col}" if isinstance(col, float) else col
        for col in pivot_df.columns
    ]

    return pivot_df.sort_values(by=dt_col, ascending=True).reset_index(
        drop=True
    )


reshape_quantile_data.__doc__ = r"""
Reshape a wide-format DataFrame with quantile columns into a
DataFrame where the quantiles are separated into distinct
columns for each quantile value.

This method transforms columns that follow the naming pattern
``{value_prefix}_{dt_value}_q{quantile}`` into a structured format,
preserving spatial coordinates and adding the temporal dimension
based on extracted datetime values [1]_.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing quantile columns. The columns should
    follow the pattern ``{value_prefix}_{dt_val}_q{quantile}``, where:

    - `value_prefix` is the base name for the quantile measurement
      (e.g., ``'predicted_subsidence'``)
    - `dt_val` is the datetime value (e.g., year or month)
    - `quantile` is the quantile value (e.g., 0.1, 0.5, 0.9)
value_prefix : str
    Base name for quantile measurement columns (e.g.,
    ``'predicted_subsidence'``). This is used to identify the
    quantile columns in the DataFrame.
spatial_cols : list of str, optional
    List of spatial column names (e.g., ``['longitude', 'latitude']``).
    These columns will be preserved through the reshaping operations.
    If `None`, the default columns (e.g., ``['longitude', 'latitude']``)
    will be used.
dt_col : str, default='year'
    Name of the column that will contain the extracted temporal
    information (e.g., 'year'). This will be used as a column in the
    output DataFrame for temporal dimension tracking.
error : {'raise', 'warn', 'ignore'}, default='warn'
    Specifies how to handle errors when certain columns or data
    patterns are not found. Options include:
    - ``'raise'``: Raises a ValueError with a message if columns are missing.
    - ``'warn'``: Issues a warning with a message if columns are missing.
    - ``'ignore'``: Silently returns an empty DataFrame when issues are found.
savefile : str, optional
    Path to save the reshaped DataFrame. If provided, the DataFrame
    will be saved to this location.
verbose : int, default=0
    Level of verbosity for progress messages. Higher values
    correspond to more detailed output during processing:
    - 0: Silent
    - 1: Basic progress
    - 2: Column parsing details
    - 3: Metadata extraction
    - 4: Reshaping steps
    - 5: Full debug

Returns
-------
pd.DataFrame
    A reshaped DataFrame with quantiles as separate columns for each
    quantile value. The DataFrame will have the following columns:

    - Spatial columns (if any)
    - Temporal column (specified by ``dt_col``)
    - ``{value_prefix}_q{quantile}`` value columns for each quantile

Examples
--------
>>> from kdiagram.utils.q_utils import reshape_quantile_data
>>> import pandas as pd
>>> wide_df = pd.DataFrame({
...     'lon': [-118.25, -118.30],
...     'lat': [34.05, 34.10],
...     'subs_2022_q0.1': [1.2, 1.3],
...     'subs_2022_q0.5': [1.5, 1.6],
...     'subs_2023_q0.1': [1.7, 1.8]
... })
>>> reshaped_df = reshape_quantile_data(wide_df, 'subs')
>>> reshaped_df.columns
Index(['lon', 'lat', 'year', 'subs_q0.1', 'subs_q0.5'], dtype='object')

Notes
-----

- The column names must follow the pattern
  ``{value_prefix}_{dt_value}_q{quantile}`` for proper extraction.
- The temporal dimension is determined by the ``dt_col`` argument.
- Spatial columns are automatically detected or can be passed explicitly.
- The quantiles are pivoted and separated into distinct columns
  based on the unique quantile values found in the DataFrame [2]_.

.. math::

    \mathbf{W}_{m \times n} \rightarrow \mathbf{L}_{p \times k}

where:

- :math:`m` = Original row count
- :math:`n` = Original columns (quantile + spatial + temporal)
- :math:`p` = :math:`m \times t` (t = unique temporal values)
- :math:`k` = Spatial cols + 1 temporal + q quantile cols


See Also
--------
pandas.melt : For reshaping DataFrames from wide to long format.
kdiagram.utils.q_utils.melt_q_data : Alternative method for reshaping quantile data.


References
----------
.. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
       in Python". Proceedings of the 9th Python in Science Conference.
.. [2] Wickham, H. (2014). "Tidy Data". Journal of Statistical Software,
       59(10), 1-23.
"""


@SaveFile
@check_non_emptiness
def melt_q_data(
    df: pd.DataFrame,
    value_prefix: str | None = None,
    dt_name: str = "dt_col",
    q: list[float | str] | None = None,
    error: str = "raise",
    sort_values: str | None = None,
    spatial_cols: tuple[str, str] | None = None,
    savefile: str | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    # Validate error handling
    error = error_policy(
        error,
        base="warn",
        msg="error must be one of 'raise','warn', or 'ignore'",
    )

    is_frame(df, df_only=True, objname="Data 'df'")

    # Compile regex to match columns like: {value_prefix}_{dt_val}_q{quantile}
    pattern = re.compile(rf"^{re.escape(value_prefix)}_(\d+)_q([0-9.]+)$")

    # Collect matching columns & metadata
    meta = []
    quant_cols = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            dt_val, q_val = match.groups()
            meta.append((col, dt_val, float(q_val)))
            quant_cols.append(col)

    if verbose >= 2:
        print(
            f"[INFO] Found {len(quant_cols)} quantile columns "
            f"for prefix '{value_prefix}'."
        )

    # Handle case: no matched columns
    if not quant_cols:
        msg = (
            f"No columns found with prefix '{value_prefix}' "
            "following the pattern {prefix}_{dt_val}_q{quant}"
        )
        handle_error(msg, error)
        return pd.DataFrame()

    # Filter by requested quantiles if needed
    if q is not None:
        # skip doc; assume validate_quantiles is imported
        valid_q = validate_quantiles(q, mode="soft", dtype="float64")
        # Convert all to float for comparison
        q_floats = [float(x) for x in valid_q]
        new_meta = [(c, d, v) for (c, d, v) in meta if v in q_floats]
        if not new_meta:
            msg = f"No columns match requested quantiles {q}"
            handle_error(msg, error)
            return pd.DataFrame()
        meta = new_meta
        quant_cols = [m[0] for m in meta]

    # Detect or validate spatial columns
    # skip doc; assume columns_manager & check_spatial_columns are imported
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    if spatial_cols:
        check_spatial_columns(df, spatial_cols)
        if verbose >= 2:
            print(f"[INFO] Spatial columns detected: {spatial_cols}")

    # Prepare for melting
    id_vars = list(spatial_cols) if spatial_cols else []
    # Melt only the quantile columns
    melt_df = df.melt(
        id_vars=id_vars,
        value_vars=quant_cols,
        var_name="column",
        value_name=value_prefix,
    )
    if verbose >= 4:
        print(f"[DEBUG] After melt, shape: {melt_df.shape}")

    # Merge with metadata (columns -> dt & quantile)
    meta_df = pd.DataFrame(meta, columns=["column", dt_name, "quantile"])
    merged_df = melt_df.merge(meta_df, on="column", how="left")

    # Pivot with (spatial + dt_name) as index, 'quantile' as columns
    pivot_index = id_vars + [dt_name] if id_vars else [dt_name]
    pivot_df = merged_df.pivot_table(
        index=pivot_index,
        columns="quantile",
        values=value_prefix,
        aggfunc="first",
    ).reset_index()

    # Rename pivoted columns -> e.g. subs_q0.1, subs_q0.9
    new_cols = []
    for col in pivot_df.columns:
        if isinstance(col, float):
            new_cols.append(
                f"{value_prefix}_q{col:.2f}".rstrip("0").rstrip(".")
            )
        else:
            new_cols.append(str(col))
    pivot_df.columns = new_cols

    # Sort final columns for consistency
    sort_cols = list(spatial_cols) + [dt_name] if spatial_cols else [dt_name]
    pivot_df = pivot_df.sort_values(sort_cols).reset_index(drop=True)

    if verbose >= 4:
        print(f"[DEBUG] After pivot, shape: {pivot_df.shape}")

    if verbose >= 1:
        print(f"[INFO] melt_q_data complete. Final shape: {pivot_df.shape}")

    # Sort if requested
    if sort_values is not None:
        try:
            # Verify that `sort_values` columns exist
            exist_features(pivot_df, features=sort_values)
        except Exception as e:
            if verbose >= 2:
                print(
                    f"[WARN] Unable to sort by '{sort_values}'. "
                    f"{str(e)} Fallback to no sorting."
                )
            sort_values = None

        if sort_values is not None:
            try:
                pivot_df = pivot_df.sort_values(by=sort_values)
            except Exception as e:
                if verbose >= 2:
                    print(
                        f"[WARN] Sorting failed: {str(e)}. No sort applied."
                    )
    return pivot_df


def handle_error(msg: str, error: str) -> None:
    """Centralized error handling."""
    if error == "raise":
        raise ValueError(msg)
    elif error == "warn":
        warnings.warn(msg, stacklevel=2)


melt_q_data.__doc__ = r"""
Reshape a wide DataFrame with time-embedded quantile columns into a
tidy wide table with explicit temporal and quantile dimensions.

This function looks for columns named like
``{value_prefix}_{dt_value}_q{quantile}`` (e.g., ``subs_2022_q0.1``)
and returns a table with one row per temporal value (and optional
spatial coordinates), and one **column per quantile**
(e.g., ``subs_q0.1``, ``subs_q0.5``, ...). Internally it melts,
extracts metadata (time & quantile), then pivots so quantiles become
separate columns.


.. math::

   \mathcal{S}=\text{spatial indices},\quad
   \mathcal{T}=\text{times},\quad
   \mathcal{Q}=\text{quantiles}

.. math::

   \mathbf{W}\in\mathbb{R}^{m\times n}
   \ \xrightarrow{\ \text{melt+pivot}\ }\ 
   \mathbf{L}\in\mathbb{R}^{p\times k}

with

- :math:`p=\lvert\{(s,t): s\in\mathcal{S},\,t\in\mathcal{T}\}\rvert`
- :math:`k=\lvert\mathcal{S}\rvert + 1 + \lvert\mathcal{Q}\rvert`

The source columns are named

.. math::

   \mathrm{col}(t,\alpha)=
   \texttt{f"{value\_prefix}\_\{t\}\_q\{\alpha\}"}

and hold values :math:`y_{s,t,\alpha}`. The output table contains,
for each :math:`\alpha\in\mathcal{Q}`, the column

.. math::

   \texttt{f"{value\_prefix}\_q\{\alpha\}"}
   \quad\text{with entries}\quad
   \left[\mathbf{L}\right]_{(s,t),\,\alpha}
   = y_{s,t,\alpha}.
   
Parameters
----------
df : pandas.DataFrame
    Input DataFrame containing quantile columns named with the pattern
    ``{value_prefix}_{dt_value}_q{quantile}``. Here ``dt_value`` is a
    time token (e.g., year) and ``quantile`` is a numeric label (e.g.,
    ``0.1``, ``0.5``, ``0.9``).
value_prefix : str
    Base measurement name used to identify quantile columns
    (e.g., ``'subs'`` or ``'predicted_subsidence'``). **Required**.
dt_name : str, default='dt_col'
    Name of the output column holding the extracted temporal value
    (e.g., ``'year'``).
q : list of {float, str}, optional
    Which quantiles to keep. Floats like ``0.1`` or strings like
    ``"10%"`` are accepted. If ``None``, all detected quantiles are used.
error : {'raise', 'warn', 'ignore'}, default='raise'
    Behavior when no matching columns are found or a filter removes all:
    - ``'raise'`` : raise ``ValueError`` with details
    - ``'warn'``  : warn and return an empty DataFrame
    - ``'ignore'``: silently return an empty DataFrame
sort_values : str, optional
    If provided, sort the **final** DataFrame by this column. If the
    column is missing, a warning is printed when ``verbose >= 1`` and no
    sort is applied.
spatial_cols : tuple[str, ...] or list[str], optional
    Names of columns that identify spatial coordinates (e.g.,
    ``('lon', 'lat')``). If provided, they are retained in the index
    during aggregation and preserved in the output. If omitted, spatial
    columns are **not** retained (unless your environment-specific
    helper auto-detects them).
savefile : str, optional
    Path to save the reshaped DataFrame (handled by ``@SaveFile``).
verbose : int, default=0
    Verbosity level: 0=silent, 1=progress, 2=column parsing,
    3=metadata extraction, 4=reshaping steps, 5=full debug.

Returns
-------
pandas.DataFrame
    A tidy-wide DataFrame with columns:
    - Spatial columns (if provided via ``spatial_cols``)
    - The temporal column named ``dt_name``
    - One column per quantile: ``{value_prefix}_q{quantile}``

    Quantile column names are normalized to compact fixed-point strings,
    e.g. ``subs_q0.1``, ``subs_q0.25``, ``subs_q0.9`` (trailing zeros
    are trimmed).

Notes
-----

- Expected input column pattern:
  ``{value_prefix}_{dt_value}_q{quantile}``. The time token is captured
  literally (e.g., ``2022``) and emitted into the ``dt_name`` column.
- Quantile labels are parsed as floats. They are re-emitted with stable
  string formatting (e.g., ``q0.1``, ``q0.25``).
- If ``spatial_cols`` is not provided, spatial coordinates are typically
  **not** preserved (unless a custom ``columns_manager`` performs
  automatic detection in your codebase).
- The function sorts rows by ``spatial_cols + [dt_name]`` (when
  ``spatial_cols`` are present) or by ``[dt_name]`` for consistency.
- If ``sort_values`` is given, a secondary sort is attempted; failures
  are downgraded to warnings when ``verbose >= 2``.

Examples
--------
Basic reshape without spatial coordinates::
    
    >>> from kdiagram.utils.q_utils import melt_q_data
    >>> wide_df = pd.DataFrame({
    ...     'lon': [-118.25, -118.30],
    ...     'lat': [34.05, 34.10],
    ...     'subs_2022_q0.1': [1.2, 1.3],
    ...     'subs_2022_q0.5': [1.5, 1.6],
    ...     'subs_2023_q0.9': [1.7, 1.8]
    ... })
    >>> out = melt_q_data(wide_df, 'subs', dt_name='year')
    >>> out.columns.tolist()
    ['year', 'subs_q0.1', 'subs_q0.5', 'subs_q0.9']

Preserving spatial coordinates::

    >>> out2 = melt_q_data(
    ...     wide_df, 'subs', dt_name='year', spatial_cols=('lon', 'lat')
    ... )
    >>> out2.columns[:3].tolist()
    ['lon', 'lat', 'year']

Filtering to a subset of quantiles::

    >>> out3 = melt_q_data(wide_df, 'subs', q=[0.1, '50%'])
    >>> [c for c in out3.columns if c.startswith('subs_q')]
    ['subs_q0.1', 'subs_q0.5']

See Also
--------
pandas.melt : Reshape from wide to long.
pandas.DataFrame.pivot_table : Pivot long to wide.

References
----------
.. [1] Wickham, H. (2014). *Tidy Data*. J. Stat. Software, 59(10).
.. [2] McKinney, W. (2010). *Data Structures for Statistical
       Computing in Python*. Proc. SciPy.
"""


@SaveFile
@check_non_emptiness
def pivot_q_data(
    df: pd.DataFrame,
    value_prefix: str,
    dt_col: str = "dt_col",
    q: list[float | str] | None = None,
    spatial_cols: tuple[str, str] | None = None,
    error: str = "raise",
    verbose: int = 0,
) -> pd.DataFrame:
    def handle_error(
        msg: str, error: str, default: pd.DataFrame
    ) -> pd.DataFrame:
        """Centralized error handling."""
        if error == "raise":
            raise ValueError(msg)
        elif error == "warn":
            warnings.warn(msg, stacklevel=2)
        return default

    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if error not in ["raise", "warn", "ignore"]:
        raise ValueError("error must be 'raise', 'warn', or 'ignore'")

    # Create working copy and validate structure
    df = df.copy()
    required_cols = {dt_col}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        msg = f"Missing required columns: {missing}"
        return handle_error(msg, error, pd.DataFrame())

    # Detect quantile columns
    quant_pattern = re.compile(rf"^{re.escape(value_prefix)}_q([0-9.]+)$")
    quant_columns = [col for col in df.columns if quant_pattern.match(col)]

    if not quant_columns:
        msg = f"No quantile columns found with prefix '{value_prefix}'"
        return handle_error(msg, error, pd.DataFrame())

    # Extract and validate quantile values
    quantiles = sorted(
        [float(quant_pattern.match(col).group(1)) for col in quant_columns],
        key=lambda x: float(x),
    )

    if verbose >= 1:
        print(f"Found quantiles: {quantiles}")

    # Filter requested quantiles
    if q is not None:
        valid_q = validate_quantiles(q, mode="soft", dtype="float64")
        quant_columns = [
            col
            for col in quant_columns
            if float(quant_pattern.match(col).group(1)) in valid_q
        ]
        if not quant_columns:
            msg = f"No columns match filtered quantiles {q}"
            return handle_error(msg, error, pd.DataFrame())

    # Identify spatial columns (non-temporal, non-quantile)
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    if spatial_cols:
        check_spatial_columns(df, spatial_cols)

    # spatial_cols = [
    #     col for col in df.columns
    #     if col not in quant_columns + [dt_col]
    # ]
    # Melt quantile columns to long format
    id_vars = spatial_cols + [dt_col]
    melt_df = df.melt(
        id_vars=id_vars,
        value_vars=quant_columns,
        var_name="quantile",
        value_name="value",
    )

    # Extract numeric quantile values
    melt_df["quantile"] = (
        melt_df["quantile"].str.extract(r"q([0-9.]+)$").astype(float)
    )

    # Pivot to wide format
    try:
        wide_df = melt_df.pivot_table(
            index=spatial_cols,
            columns=[dt_col, "quantile"],
            values="value",
            aggfunc="first",  # Handle potential duplicates
        )
    except ValueError as e:
        msg = f"Pivoting failed: {str(e)}"
        return handle_error(msg, error, pd.DataFrame())

    # Flatten multi-index columns
    wide_df.columns = [
        f"{value_prefix}_{dt}_q{quantile:.2f}".rstrip("0").rstrip(".")
        for (dt, quantile) in wide_df.columns
    ]

    return wide_df.reset_index()


pivot_q_data.__doc__ = r"""
Convert a long/tidy-wide quantile table back to a time-embedded
wide format with columns named like ``{value_prefix}_{t}_q{α}``.

This function expects one column per quantile in the input
(e.g., ``subs_q0.1``, ``subs_q0.5``) plus a temporal column
``dt_col`` and optional spatial columns. It reconstructs the
original wide layout by creating a column for each pair
``(t,α)`` and moving values into
``{value_prefix}_{t}_q{α}``.

.. math::

   \mathcal{S}=\text{spatial indices},\quad
   \mathcal{T}=\text{times},\quad
   \mathcal{Q}=\text{quantiles}

Input frame :math:`\mathbf{L}` has columns

.. math::

   \{\text{spatial}\}\ \cup\ \{\texttt{dt\_col}\}\
   \cup\ \{\ \texttt{f"{value\_prefix}\_q\{\alpha\}"}\
   :\ \alpha\in\mathcal{Q}\ \}.

The output frame :math:`\mathbf{W}` has columns

.. math::

   \{\text{spatial}\}\ \cup\
   \{\ \texttt{f"{value\_prefix}\_\{t\}\_q\{\alpha\}"}\
   :\ t\in\mathcal{T},\,\alpha\in\mathcal{Q}\ \}.

For each location :math:`s\in\mathcal{S}`, time :math:`t`, and
quantile :math:`\alpha`,

.. math::

   \mathbf{W}\big[s,\ \texttt{f"{value\_prefix}\_\{t\}\_q\{\alpha\}"}\big]
   \ =\ 
   \mathbf{L}\big[(s,t),\ \texttt{f"{value\_prefix}\_q\{\alpha\}"}\big].

Parameters
----------
df : pandas.DataFrame
    Input table with:
        
    - Spatial columns (optional)
    - Temporal column ``dt_col``
    - One column per quantile:
      ``{value_prefix}_q{α}``
      
value_prefix : str
    Base measurement name (e.g., ``'subs'``). Used to reconstruct
    wide column names.
dt_col : str, default='dt_col'
    Temporal column from which tokens ``t`` are taken.
q : list of {float, str}, optional
    Quantiles to include in the output. If ``None``, all detected
    quantiles are used.
spatial_cols : tuple[str, ...] or list[str], optional
    Names of spatial coordinate columns. If provided, used as the
    index during pivot and preserved in the output.
error : {'raise', 'warn', 'ignore'}, default='raise'
    Handling for missing components:
        
    - ``'raise'`` : raise ``ValueError`` with details
    - ``'warn'``  : warn and return an empty frame or partial
    - ``'ignore'``: silently return partial results
    
verbose : {0,1,2,3,4,5}, default=0
    Detail level for progress messages.

Returns
-------
pandas.DataFrame
    Wide DataFrame with columns:
        
    - Spatial columns (if provided)
    - For each :math:`t\in\mathcal{T}` and :math:`\alpha\in\mathcal{Q}`,
      a column named ``{value_prefix}_{t}_q{α}``.

Notes
-----

- Quantile columns in the input must follow the pattern
  ``{value_prefix}_q{α}``. The temporal values are taken from
  ``dt_col``.
- Column names in the output are normalized with compact fixed-point
  quantile formatting (e.g., ``q0.1``, ``q0.25``).
- Duplicate entries for the same ``(spatial, t, α)`` are aggregated
  with ``aggfunc='first'`` by default. Adjust upstream data to avoid
  ambiguity if necessary.
- The transformation is the (approximate) inverse of
  :func:`melt_q_data`:

  .. math::

     \mathbf{L}\ \xrightarrow{\ \text{pivot}\ }\ \mathbf{W}
     \quad\text{and}\quad
     \mathbf{W}\ \xrightarrow{\ \text{melt+pivot}\ }\ \mathbf{L}.

Examples
--------
From long/tidy-wide to time-embedded wide::

    >>> long_df = pd.DataFrame({
    ...   'lon': [-118.25, -118.25, -118.30],
    ...   'lat': [34.05, 34.05, 34.10],
    ...   'year': [2022, 2023, 2022],
    ...   'subs_q0.1': [1.2, 1.7, 1.3],
    ...   'subs_q0.5': [1.5, 1.9, 1.6]
    ... })
    >>> wide_df = pivot_q_data(long_df, 'subs', dt_col='year')
    >>> sorted(c for c in wide_df.columns if c.startswith('subs_'))
    ['subs_2022_q0.1', 'subs_2022_q0.5',
     'subs_2023_q0.1', 'subs_2023_q0.5']

Filtering quantiles::

    >>> wide_df2 = pivot_q_data(long_df, 'subs', dt_col='year',
    ...                         q=[0.1])
    >>> sorted(c for c in wide_df2.columns if c.startswith('subs_'))
    ['subs_2022_q0.1', 'subs_2023_q0.1']

See Also
--------
melt_q_data
    Forward reshape: parse time-embedded quantile columns and
    produce one column per quantile.
pandas.DataFrame.pivot_table
    Core reshaping used in the wide reconstruction.

References
----------
.. [1] Wickham, H. (2014). *Tidy Data*. J. Stat. Software, 59(10).
.. [2] McKinney, W. (2013). *Python for Data Analysis*. O’Reilly.

"""
