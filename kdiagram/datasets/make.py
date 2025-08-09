# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)

"""
Dataset Generation Utilities (:mod:`kdiagram.datasets.make`)
============================================================

This module provides functions to create synthetic datasets tailored
for demonstrating and testing the various plotting functions within
the `k-diagram` package, particularly those focused on uncertainty.
"""
from __future__ import annotations

import textwrap
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..api.bunch import Bunch

__all__ = [ 
    "make_uncertainty_data",
    "make_taylor_data",
    "make_multi_model_quantile_data",
    "make_cyclical_data" 
    ]

def make_cyclical_data(
    n_samples: int = 365,
    n_series: int = 2,
    cycle_period: float = 365,
    noise_level: float = 0.5,
    amplitude_true: float = 10.0,
    offset_true: float = 20.0,
    pred_bias: Union[float, List[float]] = [0, 1.5], 
    pred_noise_factor: Union[float, List[float]] = [1.0, 1.5], 
    pred_amplitude_factor: Union[float, List[float]] = [1.0, 0.8], 
    pred_phase_shift: Union[float, List[float]] = [0, np.pi / 6], 
    prefix: str = "model",
    series_names: Optional[List[str]] = None,
    seed: Optional[int] = 404,
    as_frame: bool = False,
) -> Union[Bunch, pd.DataFrame]:
    r"""Generate synthetic data with cyclical patterns.

    Creates a dataset containing a 'true' cyclical signal (e.g.,
    seasonal temperature) and one or more 'prediction' series that
    may have different amplitudes, phases, biases, and noise levels
    relative to the true signal.

    This data is useful for demonstrating and testing functions like
    :func:`~kdiagram.plot.relationship.plot_relationship` or
    :func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty` where
    visualizing behavior over a cycle is important.

    Parameters
    ----------
    n_samples : int, default=365
        Number of data points (rows) to generate, representing steps
        within cycles.

    n_series : int, default=2
        Number of simulated prediction series (e.g., models) to generate.

    cycle_period : float, default=365
        The number of samples that constitute one full cycle (e.g.,
        365 for daily data over one year). Used for generating the
        underlying sinusoidal pattern.

    noise_level : float, default=0.5
        Standard deviation of the Gaussian noise added to the true
        signal and scaled for predictions.

    amplitude_true : float, default=10.0
        Amplitude of the underlying sinusoidal 'true' signal.

    offset_true : float, default=20.0
        Vertical offset (mean) of the 'true' signal.

    pred_bias : float or list of float, default=[0, 1.5]
        Systematic bias added to each prediction series relative to the
        true signal. If a float, the same bias is applied to all. If
        a list, its length must match `n_series`.

    pred_noise_factor : float or list of float, default=[1.0, 1.5]
        Factor by which `noise_level` is multiplied for each
        prediction series. Allows models to have different noise
        levels. If a float, applied to all. If list, length must
        match `n_series`.

    pred_amplitude_factor : float or list of float, default=[1.0, 0.8]
        Factor by which `amplitude_true` is multiplied for each
        prediction series. Allows models to under/over-estimate
        cyclical amplitude. If float, applied to all. If list,
        length must match `n_series`.

    pred_phase_shift : float or list of float, default=[0, np.pi / 6]
        Phase shift (in radians) applied to the sinusoidal component
        of each prediction series relative to the true signal.
        Positive values create a lag. If float, applied to all. If
        list, length must match `n_series`.

    prefix : str, default="model"
        Base prefix used for naming the prediction columns
        (e.g., ``model_A``, ``model_B``).

    series_names : list of str, optional
        Optional list of names for the prediction series. If ``None``,
        names like "Model_A", "Model_B" are generated. Must match
        `n_series` if provided. Default is ``None``.

    seed : int, optional
        Seed for NumPy's random number generator for reproducibility.
        Default is 404.

    as_frame : bool, default=False
        Determines the return type:
        - If ``False`` (default): Returns a Bunch object.
        - If ``True``: Returns only the pandas DataFrame.

    Returns
    -------
    data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with the following attributes:
            - ``frame`` : pandas.DataFrame
                DataFrame containing 'time_step', 'y_true', and
                prediction columns (e.g., 'model_A', 'model_B').
            - ``feature_names`` : list of str
                Name of the time step column (``['time_step']``).
            - ``target_names`` : list of str
                Name of the target column (``['y_true']``).
            - ``target`` : ndarray of shape (n_samples,)
                NumPy array of 'y_true' values.
            - ``series_names`` : list of str
                Names assigned to the prediction series/models.
            - ``prediction_columns`` : list of str
                Names of the prediction columns in the frame.
            - ``DESCR`` : str
                Description of the generated cyclical dataset.

        If ``as_frame=True``:
        pandas.DataFrame
            The generated data solely as a pandas DataFrame.

    Raises
    ------
    ValueError
        If list lengths for prediction parameters do not match `n_series`.
    TypeError
        If inputs cannot be processed numerically.

    Examples
    --------
    >>> from kdiagram.datasets import make_cyclical_data

    >>> # Generate data as Bunch
    >>> cycle_bunch = make_cyclical_data(n_samples=12*2, n_series=1,
    ...                                  cycle_period=12, seed=5)
    >>> print(cycle_bunch.DESCR)
    >>> print(cycle_bunch.frame.head())

    >>> # Generate data as DataFrame
    >>> cycle_df = make_cyclical_data(n_samples=50, n_series=2,
    ...                               as_frame=True, seed=6)
    >>> print(cycle_df.columns)
    """
    # --- Input Validation & Setup ---
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Ensure prediction parameters are lists of correct length
    params_to_check = {
        "pred_bias": pred_bias,
        "pred_noise_factor": pred_noise_factor,
        "pred_amplitude_factor": pred_amplitude_factor,
        "pred_phase_shift": pred_phase_shift,
    }
    processed_params = {}
    for name, param in params_to_check.items():
        if isinstance(param, (int, float)):
            processed_params[name] = [param] * n_series
        elif isinstance(param, list):
            if len(param) != n_series:
                raise ValueError(
                    f"Length of '{name}' ({len(param)}) must match "
                    f"n_series ({n_series})."
                )
            processed_params[name] = param
        else:
            raise TypeError(f"'{name}' must be float or list of floats.")

    # --- Generate Time Step and True Signal ---
    time_step = np.arange(n_samples)
    # Angular frequency based on cycle period
    omega = 2 * np.pi / cycle_period
    theta = omega * time_step

    # True signal (e.g., sine wave + offset + noise)
    y_true = offset_true + amplitude_true * np.sin(theta) + \
             rng.normal(0, noise_level, n_samples)

    data_dict = {
        'time_step': time_step,
        'y_true': y_true
    }

    # --- Generate Model Names & Prediction Columns ---
    if series_names is None:
        series_names_list = [f"{prefix}_{chr(65+i)}" for i in range(n_series)]
    elif len(series_names) != n_series:
        raise ValueError(
            f"Length of series_names ({len(series_names)}) must "
            f"match n_series ({n_series})."
        )
    else:
        series_names_list = list(series_names)

    prediction_cols_list = []

    for i, series_name in enumerate(series_names_list):
        col_name = series_name # Use provided or generated name
        prediction_cols_list.append(col_name)

        # Get parameters for this series
        amp = amplitude_true * processed_params["pred_amplitude_factor"][i]
        bias = processed_params["pred_bias"][i]
        noise = noise_level * processed_params["pred_noise_factor"][i]
        phase = processed_params["pred_phase_shift"][i]

        # Generate prediction series
        y_pred = offset_true + bias + amp * np.sin(theta + phase) + \
                 rng.normal(0, noise, n_samples)

        data_dict[col_name] = y_pred

    # --- Create DataFrame ---
    df = pd.DataFrame(data_dict)

    # Define column categories for Bunch
    feature_names = ['time_step']
    target_name = ['y_true']

    # --- Return based on as_frame ---
    if as_frame:
        # Order columns logically
        ordered_cols = target_name + feature_names + prediction_cols_list
        return df[ordered_cols]
    else:
        # Create Bunch description
        descr = textwrap.dedent(f"""\
        Synthetic Cyclical Pattern Data for k-diagram

        **Description:**
        Simulates a dataset with a primary 'true' cyclical signal and
        {n_series} related prediction series over {n_samples} time steps.
        The true signal is a sine wave with added noise. Prediction
        series are generated based on the true signal but may include
        systematic bias, different amplitude scaling, phase shifts (lag/lead),
        and varying noise levels, according to the specified parameters.

        **Generation Parameters:**
        - n_samples             : {n_samples}
        - n_series              : {n_series}
        - cycle_period          : {cycle_period}
        - noise_level           : {noise_level:.2f} (base for y_true)
        - amplitude_true        : {amplitude_true:.2f}
        - offset_true           : {offset_true:.2f}
        - pred_bias             : {processed_params['pred_bias']}
        - pred_noise_factor     : {processed_params['pred_noise_factor']}
        - pred_amplitude_factor : {processed_params['pred_amplitude_factor']}
        - pred_phase_shift      : {processed_params['pred_phase_shift']} (radians)
        - prefix                : '{prefix}'
        - seed                  : {seed}

        **Data Structure (Bunch object):**
        - frame           : Complete pandas DataFrame.
        - feature_names   : List of feature column names (['time_step']).
        - target_names    : List containing the target column name (['y_true']).
        - target          : NumPy array of 'y_true' values.
        - series_names    : List of prediction series names.
        - prediction_columns: List of prediction column names in the frame.
        - DESCR           : This description.

        This dataset is suitable for visualizing relationships or temporal
        patterns in a polar context using functions like plot_relationship
        or plot_temporal_uncertainty.
        """)

        # Build arrays with a uniform dtype to avoid pandas -> np.find_common_type
        num_cols = feature_names + prediction_cols_list
    
        target_array = df[target_name[0]].to_numpy(dtype=np.float64, copy=True)
        data_array   = df[num_cols].to_numpy(dtype=np.float64, copy=True)

        return Bunch(
            frame=df[target_name + feature_names + prediction_cols_list], 
            data=data_array, 
            feature_names=feature_names,
            target_names=target_name,
            target=target_array,
            series_names=series_names_list,
            prediction_columns=prediction_cols_list,
            DESCR=descr
        )
    
    
def make_fingerprint_data(
    n_layers: int = 3,
    n_features: int = 8,
    layer_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
    sparsity: float = 0.1,
    add_structure: bool = True,
    seed: Optional[int] = 303,
    as_frame: bool = False,
) -> Union[Bunch, pd.DataFrame]:
    r"""Generate synthetic feature importance data for fingerprint plots.

    Creates a dataset representing feature importance scores across
    multiple layers (e.g., different models, time periods, or
    experimental groups). The output is suitable for visualization
    with :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`.

    Parameters
    ----------
    n_layers : int, default=3
        Number of layers (rows) in the importance matrix, representing
        different models, groups, or time steps.

    n_features : int, default=8
        Number of features (columns) in the importance matrix.

    layer_names : list of str, optional
        Optional list of names for the layers (rows). If ``None``,
        generic names like "Layer A", "Layer B" are generated. Must
        match `n_layers` if provided. Default is ``None``.

    feature_names : list of str, optional
        Optional list of names for the features (columns). If ``None``,
        generic names like "Feature 1", "Feature 2" are generated.
        Must match `n_features` if provided. Default is ``None``.

    value_range : tuple of (float, float), default=(0.0, 1.0)
        The approximate range ``(min_val, max_val)`` from which the raw
        importance scores are initially sampled (uniformly).

    sparsity : float, default=0.1
        The approximate fraction (0.0 to 1.0) of importance values
        that will be randomly set to zero, simulating unimportant
        features for certain layers.

    add_structure : bool, default=True
        If ``True``, adds some simple patterns to the importance
        values to make the "fingerprints" more distinct between
        layers (e.g., first layer emphasizes early features, last
        layer emphasizes later features). If ``False``, values are purely
        random within the range (plus sparsity).

    seed : int, optional
        Seed for NumPy's random number generator for reproducibility.
        Default is 303.

    as_frame : bool, default=False
        Determines the return type:
        - If ``False`` (default): Returns a Bunch object containing
          the importance matrix, names, and metadata.
        - If ``True``: Returns a pandas DataFrame containing the
          importance matrix, indexed by layer names and with feature
          names as columns.

    Returns
    -------
    data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with the following attributes:
            - ``importances`` : ndarray of shape (n_layers, n_features)
                The generated feature importance matrix.
            - ``frame`` : pandas.DataFrame
                The importance matrix presented as a DataFrame with
                layer names as index and feature names as columns.
            - ``layer_names`` : list of str
                Names associated with the layers (rows).
            - ``feature_names`` : list of str
                Names associated with the features (columns).
            - ``DESCR`` : str
                A description of the generated synthetic data.

        If ``as_frame=True``:
        pandas.DataFrame
            The generated importance matrix as a pandas DataFrame, with
            layer names as index and feature names as columns.

    Raises
    ------
    ValueError
        If `layer_names` or `feature_names` lengths mismatch dimensions,
        or if `sparsity` or `value_range` are invalid.

    Examples
    --------
    >>> from kdiagram.datasets import make_fingerprint_data

    >>> # Get data as Bunch (default)
    >>> fp_bunch = make_fingerprint_data(n_layers=4, n_features=10, seed=1)
    >>> print(fp_bunch.DESCR)
    >>> print(fp_bunch.importances.shape)
    >>> print(fp_bunch.frame.head(2))

    >>> # Get data as DataFrame
    >>> fp_df = make_fingerprint_data(as_frame=True, seed=2)
    >>> print(fp_df)

    >>> # Use with plotting function
    >>> # import kdiagram.plot.feature_based as kdf
    >>> # data = make_fingerprint_data()
    >>> # kdf.plot_feature_fingerprint(
    ... #    importances=data.importances, # or data.frame
    ... #    features=data.feature_names,
    ... #    labels=data.layer_names
    ... # )
    """
    # --- Input Validation & Setup ---
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be between 0.0 and 1.0")
    if not (isinstance(value_range, tuple) and len(value_range) == 2
            and value_range[0] <= value_range[1]):
        raise ValueError("value_range must be a tuple (min, max)"
                         " with min <= max.")

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate names if needed
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(f"Length of feature_names ({len(feature_names)}) "
                         f"must match n_features ({n_features}).")

    if layer_names is None:
        layer_names = [f"Layer_{chr(65+i)}" for i in range(n_layers)]
    elif len(layer_names) != n_layers:
        raise ValueError(f"Length of layer_names ({len(layer_names)}) "
                         f"must match n_layers ({n_layers}).")

    # --- Generate Importance Matrix ---
    min_val, max_val = value_range
    importances = rng.uniform(min_val, max_val, size=(n_layers, n_features))

    # Add optional structure
    if add_structure and n_layers > 1 and n_features > 1:
        for i in range(n_layers):
            # Example structure: layer 'i' emphasizes feature 'i' (cycling)
            emphasized_feature = i % n_features
            importances[i, emphasized_feature] = rng.uniform(
                (min_val + max_val) / 1.5, # Emphasize higher values
                max_val * 1.1 # Allow slightly exceeding max
                )
            # Maybe deemphasize another feature
            deemphasized_feature = (i + n_features // 2) % n_features
            if deemphasized_feature != emphasized_feature:
                 importances[i, deemphasized_feature] = rng.uniform(
                     min_val * 0.9, # Allow slightly below min
                     (min_val + max_val) / 2.5 # Emphasize lower values
                 )
        # Ensure values stay within reasonable bounds if needed
        importances = np.clip(importances, min_val * 0.8 , max_val * 1.2)


    # Introduce sparsity
    if sparsity > 0:
        mask = rng.choice(
            [0, 1],
            size=importances.shape,
            p=[sparsity, 1 - sparsity]
            )
        importances *= mask

    # --- Assemble DataFrame ---
    df = pd.DataFrame(
        importances,
        index=layer_names,
        columns=feature_names
    )

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        # Create Bunch description
        descr = textwrap.dedent(f"""\
        Synthetic Feature Fingerprint Data

        **Description:**
        Simulated feature importance matrix for {n_layers} layers/groups
        and {n_features} features. Values were sampled uniformly from
        the range {value_range} and approximately {sparsity*100:.0f}% were
        randomly set to zero (sparsity).{' Some basic structure was added.'
        if add_structure else ''} This dataset is suitable for use with
        plot_feature_fingerprint.

        **Generation Parameters:**
        - n_layers       : {n_layers}
        - n_features     : {n_features}
        - value_range    : {value_range}
        - sparsity       : {sparsity:.2f}
        - add_structure  : {add_structure}
        - seed           : {seed}

        **Contents (Bunch object):**
        - importances    : NumPy array ({n_layers}, {n_features}) with scores.
        - frame          : Pandas DataFrame view of importances matrix.
        - layer_names    : List of {n_layers} layer names (index).
        - feature_names  : List of {n_features} feature names (columns).
        - DESCR          : This description.
        """)

        return Bunch(
            importances=importances,
            frame=df,
            layer_names=list(layer_names),
            feature_names=list(feature_names),
            DESCR=descr
        )
    
def make_uncertainty_data(
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
    as_frame: bool = False, 
) -> Union[Bunch, pd.DataFrame]: # Updated return type
    r"""Generate synthetic dataset for uncertainty visualization.

    Creates a dataset with features commonly used for testing and
    demonstrating `k-diagram` uncertainty plotting functions.
    The dataset includes simulated actual values, quantile predictions
    (Q10, Q50, Q90) evolving across multiple time periods,
    deliberately introduced anomalies relative to the first period's
    interval, and basic spatial/feature columns.

    This function allows generating data with controlled trends and noise
    in both the central tendency (Q50) and the uncertainty width
    (Q90-Q10), making it suitable for testing drift and consistency
    visualizations.

    Parameters
    ----------
    n_samples : int, default=150
        Number of data points (rows) to generate, representing
        different locations or independent samples.

    n_periods : int, default=4
        Number of consecutive time periods (e.g., years) for which
        to generate quantile data (Q10, Q50, Q90).

    anomaly_frac : float, default=0.15
        Approximate fraction (0.0 to 1.0) of samples where the
        'actual' value (representing the first period) is
        deliberately placed outside the generated Q10-Q90 interval
        of that *first* time period. Useful for testing anomaly plots.

    start_year : int, default=2022
        The starting year used for naming time-dependent columns
        following the pattern ``{prefix}_{year}_q{quantile}``.

    prefix : str, default="value"
        The base prefix used for naming the generated value and
        quantile columns (e.g., ``value_2022_q0.1``).

    base_value : float, default=10.0
        Approximate mean value for the underlying signal and Q50
        prediction in the first time period.

    trend_strength : float, default=1.5
        Controls the strength of the linear trend added to the Q50
        predictions over consecutive time periods. A positive value
        means the Q50 tends to increase over time.

    noise_level : float, default=2.0
        Standard deviation of the Gaussian noise added to the base
        signal and quantile predictions, controlling random spread.

    interval_width_base : float, default=4.0
        Approximate base width of the Q10-Q90 prediction interval
        in the first time period.

    interval_width_noise : float, default=1.5
        Amount of random noise (sampled uniformly) added to the
        interval width for each sample and period, introducing
        variability in uncertainty estimates.

    interval_width_trend : float, default=0.5
        Controls the linear trend applied to the average interval
        width over time periods. A positive value simulates
        increasing uncertainty (drift).

    seed : int, optional
        Seed for NumPy's random number generator to ensure
        reproducible results. Default is 42.

    as_frame : bool, default=False
        Determines the return type:
        - If ``False`` (default): Returns a Bunch object containing
          the DataFrame and associated metadata (see Returns section).
        - If ``True``: Returns only the generated pandas DataFrame.

    Returns
    -------
    data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with the following attributes:
            - ``frame`` : pandas.DataFrame
                The complete generated data. Columns include spatial info,
                features, the 'actual' value for the first period, and
                quantile columns (Q10, Q50, Q90) for all periods.
            - ``feature_names`` : list of str
                Names of spatial/auxiliary feature columns
                (``['location_id', 'longitude', 'latitude', 'elevation']``).
            - ``target_names`` : list of str
                Name of the target column (``['{prefix}_actual']``).
            - ``target`` : ndarray of shape (n_samples,)
                NumPy array of the 'actual' target values.
            - ``quantile_cols`` : dict
                Dictionary mapping quantile levels ('q0.1', 'q0.5', 'q.09')
                to lists of corresponding column names across periods.
            - ``q10_cols`` : list of str
                Convenience list of all Q10 column names.
            - ``q50_cols`` : list of str
                Convenience list of all Q50 column names.
            - ``q90_cols`` : list of str
                Convenience list of all Q90 column names.
            - ``n_periods`` : int
                Number of periods for which quantile data was generated.
            - ``prefix`` : str
                The prefix used for value/quantile columns.
            - ``DESCR`` : str
                A detailed description of the synthetic dataset and the
                parameters used for its generation.

        If ``as_frame=True``:
        pandas.DataFrame
            The generated data solely as a pandas DataFrame, ordered
            logically.

    Raises
    ------
    TypeError
        If inputs cannot be processed numerically.

    Examples
    --------
    >>> from kdiagram.datasets import make_uncertainty_data

    >>> # Generate data as a Bunch object (default)
    >>> data_bunch = make_uncertainty_data(n_samples=20, n_periods=3, seed=10)
    >>> print(data_bunch.DESCR)
    >>> print(data_bunch.frame.shape)
    >>> print(f"Q10 Columns: {data_bunch.q10_cols}")

    >>> # Generate data directly as a DataFrame
    >>> df_direct = make_uncertainty_data(as_frame=True, n_samples=30)
    >>> print(df_direct.columns)
    """
    # --- Generation Logic (same as before) ---
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    location_id = np.arange(n_samples)
    longitude = rng.uniform(-120, -115, n_samples)
    latitude = rng.uniform(33, 36, n_samples)
    elevation = rng.uniform(50, 500, n_samples) + latitude * 5
    base_signal = base_value + \
                  np.sin(np.linspace(0, 3 * np.pi, n_samples)) * 5 + \
                  rng.normal(0, noise_level / 2, n_samples)
    actual_first_period = base_signal + rng.normal(
        0, noise_level / 2, n_samples
        )

    data_dict = {
        'location_id': location_id,
        'longitude': longitude,
        'latitude': latitude,
        'elevation': elevation,
        # Store actual only once, representing T=0 or reference time
        f'{prefix}_actual': actual_first_period.copy()
    }

    all_q10_cols, all_q50_cols, all_q90_cols = [], [], []
    quantile_cols_dict = {'q0.1': [], 'q0.5': [], 'q0.9': []}

    for i in range(n_periods):
        year = start_year + i
        q10_col = f'{prefix}_{year}_q0.1'
        q50_col = f'{prefix}_{year}_q0.5'
        q90_col = f'{prefix}_{year}_q0.9'

        all_q10_cols.append(q10_col)
        all_q50_cols.append(q50_col)
        all_q90_cols.append(q90_col)
        quantile_cols_dict['q0.1'].append(q10_col)
        quantile_cols_dict['q0.5'].append(q50_col)
        quantile_cols_dict['q0.9'].append(q90_col)

        current_trend = trend_strength * i
        q50 = base_signal + current_trend + rng.normal(
            0, noise_level / 3, n_samples)

        current_interval_width = (
            interval_width_base
            + interval_width_trend * i
            + rng.uniform(
                -interval_width_noise / 2,
                 interval_width_noise / 2,
                 n_samples)
        )
        current_interval_width = np.maximum(0.1, current_interval_width)

        q10 = q50 - current_interval_width / 2
        q90 = q50 + current_interval_width / 2

        data_dict[q10_col] = q10
        data_dict[q50_col] = q50
        data_dict[q90_col] = q90

    df = pd.DataFrame(data_dict)

    actual_col_name = f'{prefix}_actual'
    if anomaly_frac > 0 and n_samples > 0:
        n_anomalies = int(anomaly_frac * n_samples)
        if n_anomalies > 0 and all_q10_cols and all_q90_cols:
            anomaly_indices = rng.choice(
                n_samples, size=n_anomalies, replace=False
            )
            n_under = n_anomalies // 2
            under_indices = anomaly_indices[:n_under]
            over_indices = anomaly_indices[n_under:]

            q10_first = df[all_q10_cols[0]].iloc[under_indices]
            q90_first = df[all_q90_cols[0]].iloc[over_indices]

            df.loc[under_indices, actual_col_name] = q10_first - \
                rng.uniform(0.5, 3.0, size=len(under_indices)) * \
                (interval_width_base / 2 + 1)

            df.loc[over_indices, actual_col_name] = q90_first + \
                rng.uniform(0.5, 3.0, size=len(over_indices)) * \
                (interval_width_base / 2 + 1)

    # Define final column order
    feature_names = ['location_id', 'longitude', 'latitude', 'elevation']
    target_names = [actual_col_name]
    pred_cols_sorted = [
        col for pair in zip(all_q10_cols, all_q50_cols, all_q90_cols)
        for col in pair
        ]
    ordered_cols = feature_names + target_names + pred_cols_sorted
    df = df[ordered_cols]

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        # Create Bunch object
        numeric_cols = feature_names + target_names + pred_cols_sorted
        # data_array = df[numeric_cols].values # Data array (optional)
        # target_array = df[target_names[0]].values
        target_array = df[target_names[0]].to_numpy(dtype=np.float64, copy=True)
        data_array   = df[numeric_cols].to_numpy(dtype=np.float64, copy=True)
        

        # Create detailed description string
        descr = textwrap.dedent(f"""\
        Synthetic Multi-Period Uncertainty Dataset for k-diagram

        **Description:**
        This dataset simulates quantile forecasts (Q10, Q50, Q90) for a
        single variable ('{prefix}') over {n_periods} consecutive time periods
        (starting from {start_year}) across {n_samples} independent samples or
        locations. It includes simulated spatial coordinates and an
        auxiliary feature ('elevation'). An 'actual' value column
        (``{actual_col_name}``) corresponding to the *first* time
        period is provided, with ~{anomaly_frac*100:.0f}% of these values
        artificially placed outside the first period's Q10-Q90 interval
        to simulate prediction anomalies.

        The Q50 predictions follow a base signal with added noise and a
        linear trend controlled by `trend_strength`. The prediction
        interval width (Q90-Q10) also includes baseline width, noise,
        and a linear trend controlled by `interval_width_trend`.

        **Generation Parameters:**
        - n_samples             : {n_samples}
        - n_periods             : {n_periods}
        - start_year            : {start_year}
        - prefix                : '{prefix}'
        - anomaly_frac          : {anomaly_frac:.2f}
        - base_value            : {base_value:.2f}
        - trend_strength        : {trend_strength:.2f} (for Q50)
        - noise_level           : {noise_level:.2f} (added to Q50/actual)
        - interval_width_base   : {interval_width_base:.2f}
        - interval_width_noise  : {interval_width_noise:.2f}
        - interval_width_trend  : {interval_width_trend:.2f}
        - seed                  : {seed}

        **Data Structure (Bunch object):**
        - frame           : Complete pandas DataFrame.
        - feature_names   : List of spatial/auxiliary feature column names.
        - target_names    : List containing the target column name.
        - target          : NumPy array of target ('actual') values.
        - quantile_cols   : Dict mapping quantiles ('q0.1', 'q0.5', 'q0.9')
                          to lists of column names across periods.
        - q10_cols        : Convenience list of Q10 column names.
        - q50_cols        : Convenience list of Q50 column names.
        - q90_cols        : Convenience list of Q90 column names.
        - n_periods       : Number of periods with quantile data.
        - prefix          : Prefix used for value/quantile columns.
        - DESCR           : This description.

        This dataset is ideal for testing functions like plot_model_drift,
        plot_uncertainty_drift, plot_interval_consistency,
        plot_anomaly_magnitude, plot_coverage_diagnostic, etc.
        """)

        # Create and return Bunch object
        return Bunch(
            frame=df,
            data=data_array, 
            feature_names=feature_names,
            target_names=target_names,
            target=target_array,
            quantile_cols=quantile_cols_dict,
            q10_cols=all_q10_cols, 
            q50_cols=all_q50_cols,
            q90_cols=all_q90_cols,
            n_periods=n_periods,
            prefix=prefix,
            DESCR=descr
        )

def make_taylor_data(
    n_samples: int = 100,
    n_models: int = 3,
    ref_std: float = 1.0,
    corr_range: Tuple[float, float] = (0.5, 0.99),
    std_range: Tuple[float, float] = (0.7, 1.3),
    noise_level: float = 0.3,
    bias_level: float = 0.1,
    seed: Optional[int] = 101,
    as_frame: bool = False, # Added parameter
) -> Union[Bunch, pd.DataFrame]: # Updated return type
    r"""Generate synthetic data suitable for Taylor Diagrams.

    Creates a reference dataset and multiple simulated "prediction"
    datasets designed to exhibit a range of standard deviations and
    correlations relative to the reference. This is useful for
    demonstrating and testing Taylor Diagram functions like
    :func:`~kdiagram.plot.evaluation.taylor_diagram`.

    The function generates predictions :math:`p` based on the reference
    :math:`r` and independent noise :math:`\epsilon` using the
    relationship :math:`p = a \cdot r + b \cdot \epsilon + \text{bias}`,
    where coefficients `a` and `b` are calculated to approximate the
    target correlation and standard deviation for each model.

    Parameters
    ----------
    n_samples : int, default=100
        Number of data points in each generated series.

    n_models : int, default=3
        Number of simulated prediction series ('models') to generate.

    ref_std : float, default=1.0
        The target standard deviation for the generated reference
        series (centered at mean 0).

    corr_range : tuple of (float, float), default=(0.5, 0.99)
        The approximate range ``(min_corr, max_corr)`` from which
        target correlation coefficients (:math:`\rho`) for the models
        will be randomly sampled. Values should be between 0 and 1
        for standard Taylor Diagram usage.

    std_range : tuple of (float, float), default=(0.7, 1.3)
        The approximate range ``(min_std_factor, max_std_factor)``
        used to scale the standard deviation of the predictions
        relative to the actual reference std dev (`ref_std`). Factors
        should be non-negative.

    noise_level : float, default=0.3
        Standard deviation of the random noise component (`epsilon`)
        added to generate predictions. Must be > 0 if target
        correlation is < 1.

    bias_level : float, default=0.1
        Maximum absolute bias randomly added to each model prediction
        series (sampled uniformly from ``[-bias_level, bias_level]``).
        Note: Taylor Diagrams are insensitive to overall bias.

    seed : int, optional
        Random seed for NumPy's random number generator to ensure
        reproducible results. Default is 101.

    as_frame : bool, default=False
        Determines the return type:
        - If ``False`` (default): Returns a Bunch object containing
          data arrays, stats, names, and metadata.
        - If ``True``: Returns a pandas DataFrame containing the
          reference and all prediction series as columns.

    Returns
    -------
    data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with the following attributes:
            - ``frame`` : pandas.DataFrame
                DataFrame containing 'reference' and prediction columns
                (e.g., 'Model_A', 'Model_B').
            - ``reference`` : ndarray of shape (n_samples,)
                The generated reference data array.
            - ``predictions`` : list of ndarray
                List containing the generated prediction arrays
                (n_models arrays, each shape (n_samples,)).
            - ``model_names`` : list of str
                Generated names for models (e.g., "Model_A").
            - ``stats`` : pandas.DataFrame
                DataFrame with calculated 'stddev' and 'corrcoef'
                for each model vs reference, indexed by model name.
            - ``ref_std`` : float
                The actual calculated standard deviation of the
                generated reference array.
            - ``DESCR`` : str
                Description of the generated dataset and parameters.

        If ``as_frame=True``:
        pandas.DataFrame
            A DataFrame where the first column is 'reference' and
            subsequent columns contain the predictions for each model,
            named according to `model_names`.

    Raises
    ------
    ValueError
        If `noise_level` is zero or negative when target correlation
        is less than 1, or if range values are invalid.

    Examples
    --------
    >>> from kdiagram.datasets import make_taylor_data

    >>> # Get data as Bunch (default)
    >>> taylor_bunch = make_taylor_data(n_models=2, seed=0)
    >>> print(taylor_bunch.DESCR)
    >>> print(taylor_bunch.stats)
    >>> print(taylor_bunch.frame.head(3))

    >>> # Get data as DataFrame
    >>> taylor_df = make_taylor_data(n_models=2, seed=0, as_frame=True)
    >>> print(taylor_df.head(3))
    >>> print(taylor_df.columns)

    >>> # Use with plotting function
    >>> # import kdiagram.plot.evaluation as kde
    >>> # data = make_taylor_data()
    >>> # kde.taylor_diagram(
    ... #    stddev=data.stats['stddev'],
    ... #    corrcoef=data.stats['corrcoef'],
    ... #    ref_std=data.ref_std,
    ... #    names=data.model_names
    ... # )
    >>> # kde.plot_taylor_diagram(
    ... #    *data.predictions, reference=data.reference, names=data.model_names
    ... # )
    """
    # --- Input Validation & Setup ---
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Basic validation for ranges
    if not (0 <= corr_range[0] <= corr_range[1] <= 1.0):
         warnings.warn(
             "corr_range limits should ideally be between 0 and 1 for "
             "standard Taylor Diagrams. Adjusting..."
         )
         corr_range = (max(0, corr_range[0]), min(1.0, corr_range[1]))
         if corr_range[0] > corr_range[1]: corr_range = (0.5, 0.99)

    if not (0 <= std_range[0] <= std_range[1]):
         warnings.warn(
             "std_range factors should be non-negative and min <= max."
             " Using defaults."
         )
         std_range = (0.7, 1.3)

    if noise_level <= 1e-9 and corr_range[1] < 1.0 - 1e-9:
         raise ValueError(
             "noise_level cannot be zero if target correlation < 1 is possible."
         )

    # --- Generate Reference Data ---
    reference_raw = rng.normal(0, ref_std, n_samples)
    # Center mean at 0
    reference = (reference_raw - np.mean(reference_raw))
    # Scale to desired std dev
    current_std = np.std(reference)
    if current_std > 1e-9:
        reference = reference * (ref_std / current_std)
    # Store actual std dev
    actual_ref_std = np.std(reference)

    # --- Generate Model Predictions ---
    predictions = []
    model_names = []
    calculated_stds = []
    calculated_corrs = []

    for i in range(n_models):
        model_name = f"Model_{chr(65+i)}" # Model A, B, C...
        model_names.append(model_name)

        # Sample target stats for this model
        target_rho = rng.uniform(corr_range[0], corr_range[1])
        target_std_factor = rng.uniform(std_range[0], std_range[1])
        target_std = target_std_factor * actual_ref_std

        # Calculate coefficients a and b for p = a*r + b*noise + bias
        a = target_rho * target_std_factor
        b_squared_term = target_std**2 - (a * actual_ref_std)**2

        if b_squared_term < -1e-9:
            warnings.warn(
                f"Model {model_name}: Cannot achieve target std "
                f"({target_std:.2f}) with target correlation "
                f"({target_rho:.2f}) and noise "
                f"({noise_level:.2f}). Setting b=0.", UserWarning
            )
            b = 0
        else:
            # Ensure noise_level isn't zero if b_squared_term > 0
            if noise_level <= 1e-9 and b_squared_term > 1e-9:
                 raise ValueError(
                     "noise_level cannot be zero if needed to reach target std"
                 )
            b = np.sqrt(max(0, b_squared_term)) / max(noise_level, 1e-9)

        # Generate noise and bias
        noise = rng.normal(0, noise_level, n_samples)
        bias = rng.uniform(-bias_level, bias_level)

        # Create prediction
        pred = a * reference + b * noise + bias
        predictions.append(pred)

        # Calculate actual stats
        calculated_stds.append(np.std(pred))
        # Clip correlation calculation for safety
        corr_val = np.corrcoef(pred, reference)[0, 1]
        calculated_corrs.append(np.clip(corr_val, -1.0, 1.0))

    # --- Assemble DataFrame (used for both frame and Bunch) ---
    df_dict = {'reference': reference}
    for name, pred_array in zip(model_names, predictions):
        df_dict[name] = pred_array
    df = pd.DataFrame(df_dict)

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        # Assemble stats DataFrame
        stats_df = pd.DataFrame({
            'stddev': calculated_stds,
            'corrcoef': calculated_corrs
        }, index=model_names)

        # Assemble description
        descr = textwrap.dedent(f"""\
        Synthetic Taylor Diagram Data

        **Generated Parameters:**
        - n_samples    : {n_samples}
        - n_models     : {n_models}
        - ref_std      : {ref_std:.2f} (target), {actual_ref_std:.2f} (actual)
        - corr_range   : ({corr_range[0]:.2f}, {corr_range[1]:.2f}) (target)
        - std_range    : ({std_range[0]:.2f}, {std_range[1]:.2f}) (target factor)
        - noise_level  : {noise_level:.2f}
        - bias_level   : {bias_level:.2f}
        - seed         : {seed}

        **Contents (Bunch object):**
        - frame        : DataFrame with reference and prediction columns.
        - reference    : NumPy array (n_samples,) - Reference data.
        - predictions  : List of {n_models} NumPy arrays (n_samples,) - Model data.
        - model_names  : List of {n_models} strings - Model labels.
        - stats        : DataFrame with actual calculated 'stddev' and
                         'corrcoef' for each model vs reference.
        - ref_std      : Actual standard deviation of the reference data.
        - DESCR        : This description.
        """)

        # Create and return Bunch object
        return Bunch(
            frame=df, 
            reference=reference,
            predictions=predictions,
            model_names=model_names,
            stats=stats_df,
            ref_std=actual_ref_std,
            DESCR=descr
        )
    
def make_multi_model_quantile_data(
    n_samples: int = 100,
    n_models: int = 3,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    prefix: str = "pred",
    model_names: Optional[List[str]] = None,
    true_mean: float = 50.0,
    true_std: float = 10.0,
    bias_range: Tuple[float, float] = (-2.0, 2.0),
    width_range: Tuple[float, float] = (5.0, 15.0),
    noise_level: float = 1.0,
    seed: Optional[int] = 202,
    as_frame: bool = False, 
) -> Union[Bunch, pd.DataFrame]: 
    r"""Generate synthetic data with multiple models' quantile predictions.

    Creates a dataset simulating a scenario where multiple models
    provide quantile forecasts (e.g., Q10, Q50, Q90) for the same
    target variable (`y_true`) at a single time point or forecast
    horizon. Each simulated model can have different systematic biases
    and prediction interval widths.

    This function is primarily used for demonstrating and testing other
    `k-diagram` functions that compare model performance or visualize
    quantile spreads across different models, such as
    :func:`~kdiagram.plot.uncertainty.plot_coverage` or
    :func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty`. It
    helps create reproducible examples and test cases.

    Parameters
    ----------
    n_samples : int, default=100
        Number of data points (rows) to generate in the dataset.
        Represents the number of independent samples or locations.

    n_models : int, default=3
        Number of simulated models for which to generate quantile
        predictions.

    quantiles : list of float, default=[0.1, 0.5, 0.9]
        List of quantile levels (values between 0 and 1) to generate
        for *each* simulated model. The list must include 0.5, as
        other quantiles are generated relative to the Q50 (median)
        prediction. The provided list will be sorted internally.

    prefix : str, default="pred"
        Base prefix used for naming the generated prediction columns.
        The final names follow the pattern
        ``{prefix}_{model_name}_q{quantile}``.

    model_names : list of str, optional
        Optional list providing custom names for the simulated models.
        If provided, its length must equal `n_models`. If ``None``,
        default names like "Model_A", "Model_B", etc., are generated.
        Default is ``None``.

    true_mean : float, default=50.0
        The mean value used for generating the underlying 'true'
        target variable `y_true` from a normal distribution.

    true_std : float, default=10.0
        The standard deviation used for generating the underlying
        'true' target variable `y_true` from a normal distribution.

    bias_range : tuple of (float, float), default=(-2.0, 2.0)
        Specifies the range ``(min_bias, max_bias)`` from which a
        systematic bias is uniformly sampled for each model. This
        bias is added to the model's Q50 prediction relative to
        `y_true`.

    width_range : tuple of (float, float), default=(5.0, 15.0)
        Specifies the range ``(min_width, max_width)`` from which the
        target average width between the lowest and highest specified
        quantiles (e.g., Q90-Q10) is uniformly sampled for each model.

    noise_level : float, default=1.0
        Standard deviation of the Gaussian noise added independently
        to each generated quantile prediction value, introducing
        random variability.

    seed : int, optional
        Seed for the random number generator (NumPy's default_rng)
        to ensure reproducible dataset generation. Default is 202.

    as_frame : bool, default=False
        Determines the return type:
        - If ``False`` (default): Returns a Bunch object containing
          the DataFrame and associated metadata (see Returns section).
        - If ``True``: Returns only the generated pandas DataFrame.

    Returns
    -------
    data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with the following attributes:
            - ``frame`` : pandas.DataFrame of shape (n_samples, \
3 + n_models * n_quantiles)
                The complete generated data. Columns include 'y_true',
                'feature_1', 'feature_2', and prediction columns like
                ``{prefix}_{model_name}_q{quantile}``.
            - ``data`` : ndarray of shape (n_samples, \
2 + n_models * n_quantiles)
                NumPy array containing only the numeric feature and
                prediction columns (excludes 'y_true').
            - ``feature_names`` : list of str
                Names of the generated auxiliary feature columns
                (``['feature_1', 'feature_2']``).
            - ``target_names`` : list of str
                Name of the target column (``['y_true']``).
            - ``target`` : ndarray of shape (n_samples,)
                NumPy array containing the 'y_true' target values.
            - ``model_names`` : list of str
                Names assigned to the simulated models.
            - ``quantile_levels``: list of float
                Sorted list of the quantile levels generated.
            - ``prediction_columns`` : dict
                Dictionary mapping each model name to a list of its
                corresponding quantile column names in the frame.
            - ``prefix`` : str
                The prefix used for prediction columns.
            - ``DESCR`` : str
                A description of the generated synthetic dataset.

        If ``as_frame=True``:
        pandas.DataFrame
            The generated data solely as a pandas DataFrame.

    Raises
    ------
    ValueError
        If `quantiles` list does not contain 0.5, if `model_names`
        length mismatches `n_models`, or if range parameters
        (`bias_range`, `width_range`) are invalid.
    TypeError
        If internal calculations fail due to non-numeric types
        (should not happen with default generation).

    See Also
    --------
    make_uncertainty_data : Generate data with temporal drift.
    make_taylor_data : Generate data suitable for Taylor diagrams.
    load_synthetic_uncertainty_data : API to load synthetic data
        with temporal drift.

    Notes
    -----
    - The function generates `y_true` from a normal distribution.
    - Each model's Q50 prediction is simulated as
      `y_true + sampled_bias + noise`.
    - Other quantiles (Qlow, Qup, etc.) for a model are generated
      symmetrically around its Q50 based on a sampled interval
      `model_width`, then noise is added, and finally, the values
      are sorted per sample to ensure quantile ordering (e.g.,
      Q10 <= Q50 <= Q90).
    - The generated `feature_1` and `feature_2` columns are simple
      random numbers provided for context or potential use as axes,
      but are not directly used in generating the target or quantile
      predictions in this function.

    Examples
    --------
    >>> from kdiagram.datasets import make_multi_model_quantile_data

    >>> # Generate data as a Bunch object (default)
    >>> data_bunch = make_multi_model_quantile_data(
    ...     n_samples=50, n_models=2, seed=1
    ... )
    >>> print(data_bunch.DESCR) # Display dataset description
    >>> print(data_bunch.frame.head(3)) # Show first few rows of DataFrame
    >>> print(f"Quantile levels generated: {data_bunch.quantile_levels}")
    >>> print(f"Model A columns: {data_bunch.prediction_columns['Model_A']}")

    >>> # Generate data directly as a DataFrame
    >>> df_direct = make_multi_model_quantile_data(
    ...     n_samples=20, as_frame=True, seed=2
    ... )
    >>> print(df_direct.columns)
    """

    # --- Input Validation ---
    if 0.5 not in quantiles:
        # Current logic relies on 0.5 being present for centering
        raise ValueError("The `quantiles` list must contain 0.5 (median).")

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if not width_range[0] <= width_range[1] or width_range[0] < 0:
        raise ValueError(
            "width_range must be (min, max) with min >= 0 and min <= max."
        )
    if not bias_range[0] <= bias_range[1]:
        raise ValueError(
            "bias_range must be (min, max) with min <= max."
        )

    # --- Setup ---
    # Ensure unique and sorted quantiles
    quantiles_sorted = sorted(list(set(quantiles)))
    if len(quantiles_sorted) < 2:
        q_min, q_max = quantiles_sorted[0], quantiles_sorted[0]
    else:
        q_min = quantiles_sorted[0]
        q_max = quantiles_sorted[-1]
    q_median = 0.5

    # Factor to scale half-width based on min/max quantile range vs Q10-Q90
    # Avoid division by zero if only one quantile provided
    width_denominator = (0.9 - 0.1)
    width_numerator = (q_max - q_min)
    if len(quantiles_sorted) > 1 and abs(width_numerator) > 1e-9:
        width_scale_factor = width_numerator / width_denominator
    else:
        width_scale_factor = 1.0 # No scaling needed if range is zero/single q

    # --- Data Generation ---
    y_true = rng.normal(true_mean, true_std, n_samples)
    feature_1 = rng.uniform(0, 1, n_samples)
    feature_2 = rng.normal(5, 2, n_samples)

    data_dict = { # Use dict to build data before DataFrame
        'y_true': y_true,
        'feature_1': feature_1,
        'feature_2': feature_2,
    }

    # Generate Model Names
    if model_names is None:
        model_names_list = [f"Model_{chr(65+i)}" for i in range(n_models)]
    elif len(model_names) != n_models:
        raise ValueError(
            f"Length of model_names ({len(model_names)}) must "
            f"match n_models ({n_models})."
        )
    else:
        model_names_list = list(model_names)

    prediction_columns_dict = {name: [] for name in model_names_list}

    # --- Generate predictions for each model ---
    for i, model_name in enumerate(model_names_list):
        # Sample model-specific parameters
        model_bias = rng.uniform(bias_range[0], bias_range[1])
        model_width = rng.uniform(width_range[0], width_range[1])

        # Store generated quantiles temporarily before sorting
        temp_model_quantiles = {}

        # Generate Q50 (median) prediction first
        q50_pred = y_true + model_bias + rng.normal(
            0, noise_level, n_samples
        )
        q50_col_name = f'{prefix}_{model_name}_q0.5'
        temp_model_quantiles[0.5] = q50_pred
        # Add name to tracking dict immediately
        prediction_columns_dict[model_name].append(q50_col_name)

        # Generate other quantiles based on Q50 and target width
        for q in quantiles_sorted:
            if q == q_median: continue # Skip if median

            # Calculate offset using proportional distance from median
            # Avoid division by zero if q_max == q_min
            q_range = q_max - q_min
            # from scipy.stats import norm
            # z_score = norm.ppf(q) # Z-score for the quantile
            # Use standard deviation implied by width (e.g. q90-q10 ~ 2.56*std)
            # implied_std = model_width / (norm.ppf(q_max) - norm.ppf(q_min)) if (q_max != q_min) else 1.0
            # quantile_offset = z_score * implied_std
            
            if abs(q_range) > 1e-9 and abs(width_scale_factor) > 1e-9:
                quantile_offset = (
                    (model_width / width_scale_factor) *
                    (q - q_median) / q_range * 2
                )
            else: # Handle single quantile or zero range
                quantile_offset = 0

            q_pred = q50_pred + quantile_offset + rng.normal(
                0, noise_level / 2, n_samples # Slightly less noise for bounds
            )
            temp_model_quantiles[q] = q_pred

        # Ensure quantile order and add to main data dict
        # Create temporary DF for sorting this model's quantiles
        model_data_temp = pd.DataFrame(temp_model_quantiles)
        # Sort values row-wise
        sorted_data = np.sort(model_data_temp.values, axis=1)
        # Assign sorted values back, creating final column names
        for k, q in enumerate(quantiles_sorted):
            col_name = f'{prefix}_{model_name}_q{q:.2f}'.rstrip('0').rstrip('.')
            data_dict[col_name] = sorted_data[:, k]
            # Add to tracking dict if not already added (handles Q50 case)
            if col_name not in prediction_columns_dict[model_name]:
                prediction_columns_dict[model_name].append(col_name)

    # Create the final DataFrame
    df = pd.DataFrame(data_dict)

    # Order columns somewhat logically
    feature_names = ['feature_1', 'feature_2']
    target_name = ['y_true']
    pred_cols_sorted = sorted([
        col for col in df.columns if col.startswith(prefix)
        ])
    ordered_cols = target_name + feature_names + pred_cols_sorted
    df = df[ordered_cols]

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        # Create Bunch object
        data_numeric_cols = feature_names + pred_cols_sorted
        data_array = df[data_numeric_cols].values
        target_array = df[target_name[0]].values

        descr = textwrap.dedent(f"""\
        Synthetic Multi-Model Quantile Dataset for k-diagram

        **Generated Parameters:**
        - n_samples    : {n_samples}
        - n_models     : {n_models}
        - quantiles    : {quantiles_sorted}
        - prefix       : {prefix}
        - true_mean    : {true_mean:.2f}
        - true_std     : {true_std:.2f}
        - bias_range   : {bias_range}
        - width_range  : {width_range}
        - noise_level  : {noise_level:.2f}
        - seed         : {seed}

        **Data Structure (Bunch object):**
        - frame           : Complete pandas DataFrame.
        - data            : NumPy array of numeric feature & prediction columns.
        - feature_names   : List of auxiliary feature column names.
        - target_names    : List containing the target column name ('y_true').
        - target          : NumPy array of 'y_true' values.
        - model_names     : List of simulated model names.
        - quantile_levels : Sorted list of quantile levels generated.
        - prediction_columns : Dict mapping model names to their column names.
        - prefix          : Prefix used for prediction columns.
        - DESCR           : This description.

        This dataset simulates quantile predictions from {n_models} models
        for a single time point, allowing comparison of their
        uncertainty characteristics.
        """)

        return Bunch(
            frame=df,
            data=data_array,
            feature_names=feature_names,
            target_names=target_name,
            target=target_array,
            model_names=model_names_list,
            quantile_levels=quantiles_sorted,
            prediction_columns=prediction_columns_dict,
            prefix=prefix,
            DESCR=descr
        )
    
