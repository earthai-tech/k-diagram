# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)

"""
Dataset Loading Utilities (:mod:`kdiagram.datasets.load`)
=========================================================

This module provides functions to load sample datasets or generate
synthetic datasets suitable for demonstrating and testing `k-diagram`
visualizations.

Functions typically return data either as a standard pandas DataFrame
or as a Bunch object containing structured information like data arrays,
feature names, target names, and descriptions.
"""
from __future__ import annotations 
import re
import textwrap
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Tuple, Any


from ..api.bunch import Bunch

# Import the generator function
from .make import make_uncertainty_data

__all__ = ["load_uncertainty_data"]


def load_uncertainty_data(
    as_frame: bool = False,
    n_samples: int = 150,
    n_periods: int = 4,
    anomaly_frac: float = 0.15,
    start_year: int = 2022,
    prefix: str = "value",
    base_value: float = 10.0,
    trend_strength: float = 1.5,
    noise_level: float = 2.0,
    interval_width_base: float = 4.0,
    interval_width_noise: float = 1.5,
    interval_width_trend: float = 0.5,
    seed: Optional[int] = 42,
) -> Union[Bunch, pd.DataFrame]:
    """Load or generate the synthetic uncertainty dataset.

    This function generates a synthetic dataset using
    :func:`~kdiagram.datasets.make.make_uncertainty_data` and returns
    it either as a pandas DataFrame or a scikit-learn-style Bunch
    object containing the data and metadata.

    The generated data includes spatial coordinates, an auxiliary
    feature ('elevation'), an 'actual' value column for the first
    period, and multiple quantile columns (Q10, Q50, Q90) spanning
    several time periods, suitable for various uncertainty plots.

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the function returns the data as a pandas DataFrame.
        If False, the function returns a :class:`~kdiagram.bunch.Bunch`
        object containing structured data and metadata (similar to
        scikit-learn dataset loaders).

    n_samples : int, default=150
        Number of data points (rows/locations) to generate.

    n_periods : int, default=4
        Number of consecutive time periods (e.g., years) for which
        to generate quantile data.

    anomaly_frac : float, default=0.15
        Approximate fraction (0.0 to 1.0) of samples where the
        'actual' value is deliberately placed outside the Q10-Q90
        interval of the first period.

    start_year : int, default=2022
        Starting year used for naming time-dependent columns.

    prefix : str, default="value"
        Base prefix for naming value and quantile columns.

    base_value : float, default=10.0
        Approximate mean value for the signal in the first period.

    trend_strength : float, default=1.5
        Strength of the linear trend applied over periods.

    noise_level : float, default=2.0
        Standard deviation of random noise added.

    interval_width_base : float, default=4.0
        Approximate base width of the Q10-Q90 interval initially.

    interval_width_noise : float, default=1.5
        Random noise added to interval width.

    interval_width_trend : float, default=0.5
        Trend applied to the interval width over periods.

    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        Returns a Bunch object with the following attributes:
            - ``frame``: The generated pandas DataFrame.
            - ``data``: NumPy array of the numeric data columns.
            - ``feature_names``: List of 'feature' column names
              (e.g., 'longitude', 'latitude', 'elevation').
            - ``target_names``: List containing the name of the 'actual'
              column.
            - ``target``: NumPy array of the 'actual' column data.
            - ``quantile_cols``: Dictionary mapping quantile levels
              (e.g., 0.1, 0.5, 0.9) to lists of corresponding column
              names across periods. E.g., {'q0.1': ['value_2022_q0.1', ...]}
            - ``n_periods``: Number of time periods generated for quantiles.
            - ``prefix``: The prefix used for value columns.
            - ``DESCR``: A detailed description of the synthetic dataset.

        If ``as_frame=True``:
        Returns the generated data as a pandas DataFrame.

    Examples
    --------
    >>> from kdiagram.datasets import load_synthetic_uncertainty_data
    >>> # Load as Bunch object (default)
    >>> data_bunch = load_synthetic_uncertainty_data(n_samples=10, n_periods=2, seed=0)
    >>> print(data_bunch.DESCR)
    >>> print(list(data_bunch.keys()))
    >>> print(data_bunch.frame.shape)
    >>> print(data_bunch.quantile_cols.keys())

    >>> # Load as DataFrame
    >>> df = load_synthetic_uncertainty_data(as_frame=True, n_samples=20)
    >>> print(df.columns)

    """
    # Generate the base dataframe
    df = make_uncertainty_data(
        n_samples=n_samples,
        n_periods=n_periods,
        anomaly_frac=anomaly_frac,
        start_year=start_year,
        prefix=prefix,
        base_value=base_value,
        trend_strength=trend_strength,
        noise_level=noise_level,
        interval_width_base=interval_width_base,
        interval_width_noise=interval_width_noise,
        interval_width_trend=interval_width_trend,
        seed=seed,
    )

    if as_frame:
        return df

    # --- Process for Bunch output ---

    # Identify column types based on naming convention
    feature_names = ['longitude', 'latitude', 'elevation']
    actual_col = f'{prefix}_actual'
    target_names = [actual_col]
    target = df[actual_col].values if actual_col in df else None

    quantile_cols = {'q0.1': [], 'q0.5': [], 'q0.9': []}
    numeric_cols = list(feature_names) # Start with features
    if actual_col in df:
        numeric_cols.append(actual_col)

    q_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_q(0\.1|0\.5|0\.9)$")
    for col in df.columns:
        match = q_pattern.match(col)
        if match:
            year, q_level = match.groups()
            key = f'q{q_level}'
            if key in quantile_cols:
                quantile_cols[key].append(col)
                numeric_cols.append(col) # Add to numeric cols list

    # Extract numeric data as numpy array
    data_array = df[numeric_cols].values

    # Create description string
    descr = textwrap.dedent(f"""\
    Synthetic Uncertainty Dataset for k-diagram

    **Generated Parameters:**
    - n_samples       : {n_samples}
    - n_periods       : {n_periods}
    - start_year      : {start_year}
    - prefix          : {prefix}
    - anomaly_frac    : {anomaly_frac:.2f}
    - base_value      : {base_value}
    - trend_strength  : {trend_strength}
    - noise_level     : {noise_level}
    - interval_width* : base={interval_width_base}, noise={interval_width_noise}, trend={interval_width_trend}
    - seed            : {seed}

    **Data Structure:**
    - frame           : Complete pandas DataFrame.
    - data            : NumPy array of numeric columns only.
    - feature_names   : Identified spatial/auxiliary features.
    - target_names    : Name of the 'actual' value column.
    - target          : NumPy array of 'actual' values.
    - quantile_cols   : Dict mapping quantiles ('q0.1', 'q0.5', 'q0.9')
                      to lists of corresponding column names across periods.
    - n_periods       : Number of periods for quantile columns.

    This dataset is designed for testing and demonstrating k-diagram's
    uncertainty visualization functions. The 'actual' column values
    correspond to the *first* time period ({start_year}) and include
    artificially generated anomalies if anomaly_frac > 0.
    Quantile columns span {n_periods} periods starting from {start_year}.
    """)

    # Create and return Bunch object
    return Bunch(
        frame=df,
        data=data_array,
        feature_names=feature_names,
        target_names=target_names,
        target=target,
        quantile_cols=quantile_cols,
        n_periods=n_periods,
        prefix=prefix,
        DESCR=descr
    )