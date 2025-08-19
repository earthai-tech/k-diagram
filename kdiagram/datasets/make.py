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

import numpy as np
import pandas as pd

from ..api.bunch import Bunch

__all__ = [
    "make_uncertainty_data",
    "make_taylor_data",
    "make_multi_model_quantile_data",
    "make_cyclical_data",
]


def make_cyclical_data(
    n_samples: int = 365,
    n_series: int = 2,
    cycle_period: float = 365,
    noise_level: float = 0.5,
    amplitude_true: float = 10.0,
    offset_true: float = 20.0,
    pred_bias: float | list[float] = None,
    pred_noise_factor: float | list[float] = None,
    pred_amplitude_factor: float | list[float] = None,
    pred_phase_shift: float | list[float] = None,
    prefix: str = "model",
    series_names: list[str] | None = None,
    seed: int | None = 404,
    as_frame: bool = False,
) -> Bunch | pd.DataFrame:
    # --- Input Validation & Setup ---
    if pred_phase_shift is None:
        pred_phase_shift = [0, np.pi / 6]
    if pred_amplitude_factor is None:
        pred_amplitude_factor = [1.0, 0.8]
    if pred_noise_factor is None:
        pred_noise_factor = [1.0, 1.5]
    if pred_bias is None:
        pred_bias = [0, 1.5]
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
    y_true = (
        offset_true
        + amplitude_true * np.sin(theta)
        + rng.normal(0, noise_level, n_samples)
    )

    data_dict = {"time_step": time_step, "y_true": y_true}

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
        col_name = series_name  # Use provided or generated name
        prediction_cols_list.append(col_name)

        # Get parameters for this series
        amp = amplitude_true * processed_params["pred_amplitude_factor"][i]
        bias = processed_params["pred_bias"][i]
        noise = noise_level * processed_params["pred_noise_factor"][i]
        phase = processed_params["pred_phase_shift"][i]

        # Generate prediction series
        y_pred = (
            offset_true
            + bias
            + amp * np.sin(theta + phase)
            + rng.normal(0, noise, n_samples)
        )

        data_dict[col_name] = y_pred

    # --- Create DataFrame ---
    df = pd.DataFrame(data_dict)

    # Define column categories for Bunch
    feature_names = ["time_step"]
    target_name = ["y_true"]

    # --- Return based on as_frame ---
    if as_frame:
        # Order columns logically
        ordered_cols = target_name + feature_names + prediction_cols_list
        return df[ordered_cols]
    else:
        # Create Bunch description
        descr = textwrap.dedent(
            f"""\
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
        """
        )

        # Build arrays with a uniform dtype to avoid pandas -> np.find_common_type
        num_cols = feature_names + prediction_cols_list

        target_array = df[target_name[0]].to_numpy(
            dtype=np.float64, copy=True
        )
        data_array = df[num_cols].to_numpy(dtype=np.float64, copy=True)

        return Bunch(
            frame=df[target_name + feature_names + prediction_cols_list],
            data=data_array,
            feature_names=feature_names,
            target_names=target_name,
            target=target_array,
            series_names=series_names_list,
            prediction_columns=prediction_cols_list,
            DESCR=descr,
        )


make_cyclical_data.__doc__ = r"""
Generate synthetic cyclical data for relationship and temporal plots.

Creates a dataset with a single **true** cyclical signal and one or
more **prediction** series that can differ in amplitude, phase, bias,
and noise relative to the truth. This is useful for demos of
polar relationship and temporal-uncertainty plots in `k-diagram`
:footcite:p:`harris2020array, 2020SciPy-NMeth, Hunter:2007`.

This data is useful for demonstrating and testing functions like
:func:`~kdiagram.plot.relationship.plot_relationship` or
:func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty` where
visualizing behavior over a cycle is important.

Parameters
----------
n_samples : int, default=365
    Number of time steps to generate. Interpreted as evenly
    spaced samples over one or more cycles.

n_series : int, default=2
    Number of simulated prediction series (e.g., models).

cycle_period : float, default=365
    Samples per full cycle :math:`P`. The angular frequency is
    :math:`\omega = 2\pi / P`. Use ``365`` for daily data over
    one year, ``12`` for monthly data over one year, etc.

noise_level : float, default=0.5
    Standard deviation of Gaussian noise added to the **true**
    signal. Prediction series scale this by ``pred_noise_factor``.

amplitude_true : float, default=10.0
    Amplitude of the sinusoidal **true** signal.

offset_true : float, default=20.0
    Vertical offset (mean level) of the **true** signal.

pred_bias : float or list of float, optional
    Additive bias for each prediction series. If a scalar is
    provided it is broadcast to all ``n_series``. If a list is
    provided, its length must equal ``n_series``. Defaults to
    ``[0.0, 1.5]`` when ``None``.

pred_noise_factor : float or list of float, optional
    Multiplier for ``noise_level`` per series. Scalar values are
    broadcast; lists must match ``n_series`` in length. Defaults
    to ``[1.0, 1.5]`` when ``None``.

pred_amplitude_factor : float or list of float, optional
    Multiplier of ``amplitude_true`` per series (allows under/
    over-estimation of the cycle amplitude). Scalar broadcast is
    supported. Defaults to ``[1.0, 0.8]`` when ``None``.

pred_phase_shift : float or list of float, optional
    Phase shift (radians) added to each series. Positive values
    produce a lag relative to the truth. Scalar broadcast is
    supported. Defaults to ``[0.0, np.pi / 6]`` when ``None``.

prefix : str, default='model'
    Prefix used to generate prediction column names, e.g.,
    ``model_A``, ``model_B``, …

series_names : list of str, optional
    Explicit names for prediction columns. If omitted, names are
    generated from ``prefix`` as ``prefix_A``, ``prefix_B``, …
    Must have length ``n_series`` if provided.

seed : int or None, default=404
    Seed for NumPy’s random generator. If ``None``, a fresh RNG
    is used.

as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` with
    metadata and arrays. If ``True``, return only the pandas
    ``DataFrame``.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default), a Bunch with:

    - ``frame`` : pandas ``DataFrame`` containing ``'time_step'``,
      ``'y_true'``, and prediction columns.
    - ``feature_names`` : ``['time_step']``.
    - ``target_names`` : ``['y_true']``.
    - ``target`` : ``ndarray`` of shape ``(n_samples,)`` with the
      true signal.
    - ``series_names`` : list of prediction series names.
    - ``prediction_columns`` : list of prediction column names.
    - ``DESCR`` : human-readable description.

    If ``as_frame=True``, only the pandas ``DataFrame`` is
    returned.

Raises
------
ValueError
    If a provided list for prediction parameters does not match
    ``n_series`` in length.

TypeError
    If prediction parameters are not float or list of float.

Notes
-----
**Signal model.** Let :math:`P` be the cycle period and
:math:`\omega = 2\pi/P`. The **true** signal at time step
:math:`t \in \{0,\dots,n\_samples-1\}` is

.. math::

   y_{\text{true}}(t)
   \;=\;
   \texttt{offset\_true}
   \;+\;
   \texttt{amplitude\_true}\,\sin(\omega t)
   \;+\;
   \varepsilon_t,
   \qquad
   \varepsilon_t \sim \mathcal{N}(0,\sigma^2),
   \;\; \sigma=\texttt{noise\_level}.

For series :math:`k=1,\dots,n\_{\text{series}}`, the prediction is

.. math::

   y_{\text{pred}}^{(k)}(t)
   \;=\;
   \texttt{offset\_true}
   \;+\;
   b_k
   \;+\;
   \big(\texttt{amplitude\_true}\,\alpha_k\big)
   \sin(\omega t + \phi_k)
   \;+\;
   \eta^{(k)}_t,

with :math:`\eta^{(k)}_t \sim \mathcal{N}\!\big(0,\,
(\sigma\,\gamma_k)^2\big)`.
Here :math:`b_k` is the bias (``pred_bias``),
:math:`\alpha_k` the amplitude factor (``pred_amplitude_factor``),
:math:`\phi_k` the phase shift (``pred_phase_shift``), and
:math:`\gamma_k` the noise factor (``pred_noise_factor``).
Numerical generation and plotting typically rely on array/scientific
and graphics stacks :footcite:p:`harris2020array, 2020SciPy-NMeth, Hunter:2007`.

See Also
--------
kdiagram.plot.relationship.plot_relationship
    Polar relationship scatter for true vs. predictions.

kdiagram.plot.uncertainty.plot_temporal_uncertainty
    General-purpose polar series plot; useful for Q10/Q50/Q90 and
    cyclical visualizations.

Examples
--------
>>> Generate a small cyclical dataset as a Bunch:
>>> 
>>> from kdiagram.datasets import make_cyclical_data
>>> ds = make_cyclical_data(
...     n_samples=24, n_series=2, cycle_period=12, seed=7
... )
>>> ds.frame.head().columns.tolist()[:3]
['time_step', 'y_true', ds.prediction_columns[0]]
>>> 
>>> Return only a DataFrame and supply custom names:
>>> 
>>> df = make_cyclical_data(
...     n_samples=50,
...     n_series=3,
...     series_names=['A','B','C'],
...     as_frame=True,
...     seed=1
... )
>>> set(['time_step','y_true']).issubset(df.columns)
True

References 
------------

.. footbibliography::
    
"""


def make_fingerprint_data(
    n_layers: int = 3,
    n_features: int = 8,
    layer_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
    sparsity: float = 0.1,
    add_structure: bool = True,
    seed: int | None = 303,
    as_frame: bool = False,
) -> Bunch | pd.DataFrame:

    # --- Input Validation & Setup ---
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be between 0.0 and 1.0")
    if not (
        isinstance(value_range, tuple)
        and len(value_range) == 2
        and value_range[0] <= value_range[1]
    ):
        raise ValueError(
            "value_range must be a tuple (min, max)" " with min <= max."
        )

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate names if needed
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) "
            f"must match n_features ({n_features})."
        )

    if layer_names is None:
        layer_names = [f"Layer_{chr(65+i)}" for i in range(n_layers)]
    elif len(layer_names) != n_layers:
        raise ValueError(
            f"Length of layer_names ({len(layer_names)}) "
            f"must match n_layers ({n_layers})."
        )

    # --- Generate Importance Matrix ---
    min_val, max_val = value_range
    importances = rng.uniform(min_val, max_val, size=(n_layers, n_features))

    # Add optional structure
    if add_structure and n_layers > 1 and n_features > 1:
        for i in range(n_layers):
            # Example structure: layer 'i' emphasizes feature 'i' (cycling)
            emphasized_feature = i % n_features
            importances[i, emphasized_feature] = rng.uniform(
                (min_val + max_val) / 1.5,  # Emphasize higher values
                max_val * 1.1,  # Allow slightly exceeding max
            )
            # Maybe deemphasize another feature
            deemphasized_feature = (i + n_features // 2) % n_features
            if deemphasized_feature != emphasized_feature:
                importances[i, deemphasized_feature] = rng.uniform(
                    min_val * 0.9,  # Allow slightly below min
                    (min_val + max_val) / 2.5,  # Emphasize lower values
                )
        # Ensure values stay within reasonable bounds if needed
        importances = np.clip(importances, min_val * 0.8, max_val * 1.2)

    # Introduce sparsity
    if sparsity > 0:
        mask = rng.choice(
            [0, 1], size=importances.shape, p=[sparsity, 1 - sparsity]
        )
        importances *= mask

    # --- Assemble DataFrame ---
    df = pd.DataFrame(importances, index=layer_names, columns=feature_names)

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        # Create Bunch description
        descr = textwrap.dedent(
            f"""\
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
        """
        )

        return Bunch(
            importances=importances,
            frame=df,
            layer_names=list(layer_names),
            feature_names=list(feature_names),
            DESCR=descr,
        )


make_fingerprint_data.__doc__ = r"""
Generate synthetic feature-importance data for fingerprint plots.

Creates a matrix of feature-importance scores across multiple
**layers** (e.g., models, periods, experimental groups) suitable
for visualization with
:func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`.
This is handy for comparing profiles in a compact polar radar
view and for testing feature-comparison workflows in forecasting
and ML :footcite:p:`scikit-learn, Lim2021, kouadiob2025`.

Parameters
----------
n_layers : int, default=3
    Number of rows (layers) in the importance matrix. Each row
    represents a group such as a model or time period.

n_features : int, default=8
    Number of columns (features) in the importance matrix.

layer_names : list of str, optional
    Names for the layers. If ``None``, generic names like
    ``'Layer_A'``, ``'Layer_B'`` are generated. Must have length
    ``n_layers`` if provided.

feature_names : list of str, optional
    Names for the features. If ``None``, generic names like
    ``'Feature_1'``, ``'Feature_2'`` are generated. Must have
    length ``n_features`` if provided.

value_range : tuple of (float, float), default=(0.0, 1.0)
    Approximate sampling range ``(min_val, max_val)`` for raw
    importance scores. Values are drawn from a uniform
    distribution before structure/sparsity are applied.

sparsity : float, default=0.1
    Fraction in ``[0, 1]`` of entries that are set to zero
    at random, simulating unimportant features for some layers.

add_structure : bool, default=True
    If ``True``, inject simple patterns to make fingerprints
    distinct, e.g., emphasizing one feature per layer and
    de-emphasizing another. If ``False``, the matrix is fully
    random apart from sparsity.

seed : int or None, default=303
    Seed for NumPy’s random generator. If ``None``, a fresh RNG
    is used.

as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` with
    metadata and arrays. If ``True``, return only the pandas
    ``DataFrame`` indexed by layers with feature columns.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default), a Bunch with:

    - ``importances`` : ``ndarray`` of shape
      ``(n_layers, n_features)``.
    - ``frame`` : pandas ``DataFrame`` view of the matrix with
      layers as index and features as columns.
    - ``layer_names`` : list of layer names.
    - ``feature_names`` : list of feature names.
    - ``DESCR`` : human-readable description.

    If ``as_frame=True``, only the pandas ``DataFrame`` is
    returned.

Raises
------
ValueError
    If ``layer_names`` or ``feature_names`` lengths do not match
    the specified dimensions, if ``sparsity`` is outside
    ``[0, 1]``, or if ``value_range`` does not satisfy
    ``min_val <= max_val``.

Notes
-----
**Generation model.** Let :math:`I \in \mathbb{R}^{L \times F}`
denote the importance matrix with :math:`L = \texttt{n\_layers}`
and :math:`F = \texttt{n\_features}`. Raw scores are sampled as

.. math::

   I_{k,j}^{(0)} \sim \mathcal{U}(m, M),
   \qquad m = \texttt{value\_range[0]},\; M = \texttt{value\_range[1]}.

If structure is enabled, a layer-specific emphasis and
de-emphasis may be applied, producing :math:`I^{(1)}`. Finally,
a sparsity mask :math:`\;M_{k,j} \sim \text{Bernoulli}(1-s)\;`
with :math:`s=\texttt{sparsity}` is applied:

.. math::

   I_{k,j} \;=\; I_{k,j}^{(1)} \cdot M_{k,j}.

Scores are left in their original scale; you may normalize
per-layer or per-feature downstream if desired. For practical
feature-importance workflows and attribution in forecasting,
see :footcite:t:`scikit-learn` and :footcite:t:`Lim2021`. The
fingerprint visualization concept is part of our polar analytics
framework :footcite:t:`kouadiob2025`.

See Also
--------
kdiagram.plot.feature_based.plot_feature_fingerprint
    Radar-style comparison of multi-feature profiles across layers.

Examples
--------
>>> Return a Bunch with arrays and a DataFrame view:
>>> 
>>> from kdiagram.datasets import make_fingerprint_data
>>> fp = make_fingerprint_data(n_layers=4, n_features=10, seed=1)
>>> fp.importances.shape
(4, 10)
>>> list(fp.frame.index)[:2], list(fp.frame.columns)[:3]
(['Layer_A', 'Layer_B'], ['Feature_1', 'Feature_2', 'Feature_3'])
>>> 
>>> Return only a DataFrame with custom names:
>>> 
>>> df = make_fingerprint_data(
...     n_layers=3,
...     n_features=5,
...     layer_names=['L1','L2','L3'],
...     feature_names=['f1','f2','f3','f4','f5'],
...     as_frame=True,
...     seed=2,
... )
>>> df.shape
(3, 5)

References
----------
.. footbibliography::
"""


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
    seed: int | None = 42,
    as_frame: bool = False,
) -> Bunch | pd.DataFrame:

    # --- Generation Logic (same as before) ---
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    location_id = np.arange(n_samples)
    longitude = rng.uniform(-120, -115, n_samples)
    latitude = rng.uniform(33, 36, n_samples)
    elevation = rng.uniform(50, 500, n_samples) + latitude * 5
    base_signal = (
        base_value
        + np.sin(np.linspace(0, 3 * np.pi, n_samples)) * 5
        + rng.normal(0, noise_level / 2, n_samples)
    )
    actual_first_period = base_signal + rng.normal(
        0, noise_level / 2, n_samples
    )

    data_dict = {
        "location_id": location_id,
        "longitude": longitude,
        "latitude": latitude,
        "elevation": elevation,
        # Store actual only once, representing T=0 or reference time
        f"{prefix}_actual": actual_first_period.copy(),
    }

    all_q10_cols, all_q50_cols, all_q90_cols = [], [], []
    quantile_cols_dict = {"q0.1": [], "q0.5": [], "q0.9": []}

    for i in range(n_periods):
        year = start_year + i
        q10_col = f"{prefix}_{year}_q0.1"
        q50_col = f"{prefix}_{year}_q0.5"
        q90_col = f"{prefix}_{year}_q0.9"

        all_q10_cols.append(q10_col)
        all_q50_cols.append(q50_col)
        all_q90_cols.append(q90_col)
        quantile_cols_dict["q0.1"].append(q10_col)
        quantile_cols_dict["q0.5"].append(q50_col)
        quantile_cols_dict["q0.9"].append(q90_col)

        current_trend = trend_strength * i
        q50 = (
            base_signal
            + current_trend
            + rng.normal(0, noise_level / 3, n_samples)
        )

        current_interval_width = (
            interval_width_base
            + interval_width_trend * i
            + rng.uniform(
                -interval_width_noise / 2, interval_width_noise / 2, n_samples
            )
        )
        current_interval_width = np.maximum(0.1, current_interval_width)

        q10 = q50 - current_interval_width / 2
        q90 = q50 + current_interval_width / 2

        data_dict[q10_col] = q10
        data_dict[q50_col] = q50
        data_dict[q90_col] = q90

    df = pd.DataFrame(data_dict)

    actual_col_name = f"{prefix}_actual"
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

            df.loc[under_indices, actual_col_name] = q10_first - rng.uniform(
                0.5, 3.0, size=len(under_indices)
            ) * (interval_width_base / 2 + 1)

            df.loc[over_indices, actual_col_name] = q90_first + rng.uniform(
                0.5, 3.0, size=len(over_indices)
            ) * (interval_width_base / 2 + 1)

    # Define final column order
    feature_names = ["location_id", "longitude", "latitude", "elevation"]
    target_names = [actual_col_name]
    pred_cols_sorted = [
        col
        for pair in zip(all_q10_cols, all_q50_cols, all_q90_cols)
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
        target_array = df[target_names[0]].to_numpy(
            dtype=np.float64, copy=True
        )
        data_array = df[numeric_cols].to_numpy(dtype=np.float64, copy=True)

        # Create detailed description string
        descr = textwrap.dedent(
            f"""\
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
        """
        )

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
            DESCR=descr,
        )


make_uncertainty_data.__doc__ = r"""
Generate a synthetic multi-period uncertainty dataset.

Creates a compact dataset for testing `k-diagram` uncertainty
visualizations: simulated **actuals** (for the first period),
quantile predictions **Q10/Q50/Q90** over multiple periods,
controllable trends and noise, injected interval-coverage
failures (anomalies), and simple spatial features. This is
useful for coverage, calibration, drift, and consistency
diagnostics :footcite:p:`Jolliffe2012, Gneiting2007b, kouadiob2025`.

Parameters
----------
n_samples : int, default=150
    Number of rows (locations) to generate.

n_periods : int, default=4
    Number of consecutive periods (e.g., years) for which to
    generate quantiles.

anomaly_frac : float, default=0.15
    Fraction in ``[0, 1]`` of rows whose first-period actual is
    forced **outside** the Q10–Q90 interval (half under-, half
    over-prediction, up to rounding).

start_year : int, default=2022
    First period’s year used in column names.

prefix : str, default='value'
    Base prefix for generated value/quantile columns.

base_value : float, default=10.0
    Mean level for the latent signal that drives Q50.

trend_strength : float, default=1.5
    Linear trend added to Q50 by period index (lead time).

noise_level : float, default=2.0
    Standard deviation for Gaussian noise added to the latent
    signal (for Q50 and actuals).

interval_width_base : float, default=4.0
    Baseline width of the Q10–Q90 interval in the first period.

interval_width_noise : float, default=1.5
    Uniform jitter magnitude applied per row/period to the
    interval width.

interval_width_trend : float, default=0.5
    Linear trend added to interval width across periods.

seed : int or None, default=42
    NumPy RNG seed for reproducibility. If ``None``, a fresh RNG
    is used.

as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` with
    arrays and metadata. If ``True``, return only the pandas
    ``DataFrame``.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default), a Bunch with:

    - ``frame`` : pandas ``DataFrame`` with spatial features,
      first-period actual, and Q10/Q50/Q90 columns by period.
    - ``feature_names`` : ``['location_id','longitude','latitude',
      'elevation']``.
    - ``target_names`` : ``[f'{prefix}_actual']``.
    - ``target`` : ``ndarray`` of actual values.
    - ``quantile_cols`` : dict mapping ``'q0.1'``, ``'q0.5'``,
      ``'q0.9'`` to lists of column names across periods.
    - ``q10_cols``, ``q50_cols``, ``q90_cols`` : convenience lists.
    - ``n_periods`` : number of generated periods.
    - ``prefix`` : the column name prefix.
    - ``DESCR`` : human-readable description.

    If ``as_frame=True``, only the pandas ``DataFrame`` is
    returned.

Raises
------
TypeError
    If numeric inputs cannot be processed.

Notes
-----
**Column naming.** Quantile columns encode the year :math:`y`
and quantile level :math:`q`:

.. math::

   \text{quantile name}
   \;\equiv\;
   \texttt{<prefix>}\_{y}\_\texttt{q}q,
   \qquad
   y \in \{\texttt{start\_year},\dots\},
   \;\; q \in \{0.1,0.5,0.9\}.

The first-period actual is stored once as
``f"{prefix}_actual"``.

**Signal and interval model.** Let period index be
:math:`t \in \{0,\dots,n\_\text{periods}-1\}` and row index
:math:`i`. Define latent base signal :math:`s_i` and Q50:

.. math::

   s_i \;=\; \texttt{base\_value}
          \;+\; \varepsilon_i,
   \qquad
   \varepsilon_i \sim \mathcal{N}(0, \sigma^2),\;
   \sigma=\texttt{noise\_level}/2,

.. math::

   Q50_{i,t} \;=\; s_i \;+\; t\cdot\texttt{trend\_strength}
                   \;+\; \eta_{i,t},
   \quad
   \eta_{i,t} \sim \mathcal{N}\!\big(0,
   (\texttt{noise\_level}/3)^2\big).

Interval width :math:`w_{i,t}` has baseline, trend, and jitter:

.. math::

   w_{i,t}
   \;=\;
   \max\!\Bigl(
     0.1,\,
     \texttt{interval\_width\_base}
     + t\cdot\texttt{interval\_width\_trend}
     + u_{i,t}
   \Bigr),
   \quad
   u_{i,t} \sim \mathcal{U}\!\Bigl(-\tfrac{
   \texttt{interval\_width\_noise}}{2},\,
   \tfrac{\texttt{interval\_width\_noise}}{2}\Bigr),

and

.. math::

   Q10_{i,t} \;=\; Q50_{i,t} - \tfrac{1}{2}w_{i,t},\qquad
   Q90_{i,t} \;=\; Q50_{i,t} + \tfrac{1}{2}w_{i,t}.

**Anomaly injection (first period).** For a fraction
``anomaly_frac`` of rows we enforce a coverage failure:

.. math::

   y^{\text{actual}}_{i}
   \notin
   [\,Q10_{i,0},\,Q90_{i,0}\,],

splitting under/over cases approximately evenly to aid tests of
coverage diagnostics and anomaly magnitude plots. Use this data to
study calibration vs. sharpness trade-offs
:footcite:p:`Gneiting2007b` and operational verification practice
:footcite:p:`Jolliffe2012`.

See Also
--------
kdiagram.plot.uncertainty.plot_coverage
    Aggregate empirical coverage vs. nominal levels.

kdiagram.plot.uncertainty.plot_coverage_diagnostic
    Point-wise success/failure on a polar layout.

kdiagram.plot.uncertainty.plot_interval_consistency
    Temporal stability of interval widths per location.

kdiagram.plot.uncertainty.plot_model_drift
    Lead-time trend of mean interval width.

kdiagram.plot.uncertainty.plot_anomaly_magnitude
    Where and how severely intervals fail.

Examples
--------
>>> # Return a Bunch and inspect quantile columns:
>>> 
>>> from kdiagram.datasets import make_uncertainty_data
>>> ds = make_uncertainty_data(n_samples=12, n_periods=3, seed=7)
>>> sorted(ds.quantile_cols.keys())
['q0.1', 'q0.5', 'q0.9']
>>> 
>>> # Return only a DataFrame and check column order:
>>> 
>>> df = make_uncertainty_data(as_frame=True, n_samples=5, seed=0)
>>> df.columns[:6].tolist()  # features + actual then Q10/Q50/Q90
['location_id', 'longitude', 'latitude', 'elevation',
 f'{ 'value'}_actual', 'value_2022_q0.1']  # doctest: +ELLIPSIS

References
----------
.. footbibliography::
"""


def make_taylor_data(
    n_samples: int = 100,
    n_models: int = 3,
    ref_std: float = 1.0,
    corr_range: tuple[float, float] = (0.5, 0.99),
    std_range: tuple[float, float] = (0.7, 1.3),
    noise_level: float = 0.3,
    bias_level: float = 0.1,
    seed: int | None = 101,
    as_frame: bool = False,
) -> Bunch | pd.DataFrame:
    # --- Input Validation & Setup ---
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Basic validation for ranges
    if not (0 <= corr_range[0] <= corr_range[1] <= 1.0):
        warnings.warn(
            "corr_range limits should ideally be between 0 and 1 for "
            "standard Taylor Diagrams. Adjusting...",
            stacklevel=2,
        )
        corr_range = (max(0, corr_range[0]), min(1.0, corr_range[1]))
        if corr_range[0] > corr_range[1]:
            corr_range = (0.5, 0.99)

    if not (0 <= std_range[0] <= std_range[1]):
        warnings.warn(
            "std_range factors should be non-negative and min <= max."
            " Using defaults.",
            stacklevel=2,
        )
        std_range = (0.7, 1.3)

    if noise_level <= 1e-9 and corr_range[1] < 1.0 - 1e-9:
        raise ValueError(
            "noise_level cannot be zero if target correlation < 1 is possible."
        )

    # --- Generate Reference Data ---
    reference_raw = rng.normal(0, ref_std, n_samples)
    # Center mean at 0
    reference = reference_raw - np.mean(reference_raw)
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
        model_name = f"Model_{chr(65+i)}"  # Model A, B, C...
        model_names.append(model_name)

        # Sample target stats for this model
        target_rho = rng.uniform(corr_range[0], corr_range[1])
        target_std_factor = rng.uniform(std_range[0], std_range[1])
        target_std = target_std_factor * actual_ref_std

        # Calculate coefficients a and b for p = a*r + b*noise + bias
        a = target_rho * target_std_factor
        b_squared_term = target_std**2 - (a * actual_ref_std) ** 2

        if b_squared_term < -1e-9:
            warnings.warn(
                f"Model {model_name}: Cannot achieve target std "
                f"({target_std:.2f}) with target correlation "
                f"({target_rho:.2f}) and noise "
                f"({noise_level:.2f}). Setting b=0.",
                UserWarning,
                stacklevel=2,
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
    df_dict = {"reference": reference}
    for name, pred_array in zip(model_names, predictions):
        df_dict[name] = pred_array
    df = pd.DataFrame(df_dict)

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        # Assemble stats DataFrame
        stats_df = pd.DataFrame(
            {"stddev": calculated_stds, "corrcoef": calculated_corrs},
            index=model_names,
        )

        # Assemble description
        descr = textwrap.dedent(
            f"""\
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
        """
        )

        return Bunch(
            frame=df,
            reference=reference,
            predictions=predictions,
            model_names=model_names,
            stats=stats_df,
            ref_std=actual_ref_std,
            DESCR=descr,
        )


make_taylor_data.__doc__ = r"""
Generate synthetic data for Taylor diagrams.

Taylor diagrams, introduced by :footcite:t:`Taylor2001`, summarize
correlation, standard deviation, and centered RMS difference between
model outputs and a reference. This routine creates one reference
series and several model-like series with controllable correlation
and spread, suitable for exercising plotting functions such as
:func:`~kdiagram.plot.evaluation.taylor_diagram`. Practical guidance
on verification appears in :footcite:p:`Jolliffe2012`.

Parameters
----------
n_samples : int, default=100
    Number of observations in each generated series.

n_models : int, default=3
    Number of model (prediction) series to simulate.

ref_std : float, default=1.0
    Target standard deviation for the reference series
    (mean is centered to 0).

corr_range : tuple of (float, float), default=(0.5, 0.99)
    Closed interval from which target correlations :math:`\rho`
    for models are sampled uniformly. Values should be in
    :math:`[0,1]` for standard Taylor use.

std_range : tuple of (float, float), default=(0.7, 1.3)
    Closed interval for multiplicative factors applied to the
    reference standard deviation to obtain each model’s target
    spread.

noise_level : float, default=0.3
    Standard deviation of the independent noise used to reach
    the requested spread and correlation. Must be positive if
    any target correlation is less than 1.

bias_level : float, default=0.1
    Maximum absolute bias added to each model series (uniform
    in ``[-bias_level, bias_level]``). Note that Taylor diagrams
    are insensitive to overall bias.

seed : int or None, default=101
    NumPy random seed. If ``None``, a fresh RNG is used.

as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` with
    arrays, names, and summary stats. If ``True``, return only
    a pandas ``DataFrame`` with columns for the reference and
    each model series.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default), a Bunch with:

    - ``frame`` : pandas ``DataFrame`` with ``'reference'`` and
      model columns.
    - ``reference`` : ``ndarray`` of shape ``(n_samples,)``.
    - ``predictions`` : list of ``ndarray`` predictions.
    - ``model_names`` : list of model labels.
    - ``stats`` : pandas ``DataFrame`` with columns
      ``'stddev'`` and ``'corrcoef'`` vs the reference.
    - ``ref_std`` : actual standard deviation of the reference.
    - ``DESCR`` : human-readable description.

    If ``as_frame=True``, only the pandas ``DataFrame`` is
    returned.

Raises
------
ValueError
    If ranges are invalid, or ``noise_level`` is non-positive
    while a sub-perfect target correlation is requested.

Notes
-----
**Construction.** Let the reference be :math:`r` with
:math:`\mathrm{E}[r]=0` and :math:`\mathrm{sd}(r)=\sigma_r`
(we target :math:`\sigma_r=\texttt{ref\_std}`). For model
:math:`k`, we synthesize

.. math::

   p^{(k)} \;=\; a^{(k)} r \;+\; b^{(k)} \epsilon^{(k)} \;+\; \text{bias}^{(k)},

with :math:`\epsilon^{(k)} \sim \mathcal{N}(0,\sigma_\epsilon^2)`
independent of :math:`r`, where
:math:`\sigma_\epsilon=\texttt{noise\_level}`. Ignoring bias
(centered statistics), the model spread and correlation satisfy

.. math::

   \sigma_{p}^{(k)} \;=\; \sqrt{(a^{(k)} \sigma_r)^2 + (b^{(k)} \sigma_\epsilon)^2},
   \qquad
   \rho^{(k)} \;=\; \frac{a^{(k)} \sigma_r}{\sigma_{p}^{(k)}}.

We sample a target
:math:`\rho^{(k)} \in \texttt{corr\_range}` and a target spread
factor :math:`\alpha^{(k)} \in \texttt{std\_range}`, set
:math:`\sigma_p^{(k)} = \alpha^{(k)} \sigma_r`, choose

.. math::

   a^{(k)} \;=\; \rho^{(k)} \alpha^{(k)}, \qquad
   b^{(k)} \;=\; \frac{\sqrt{\left(\sigma_p^{(k)}\right)^2 -
                           \left(a^{(k)} \sigma_r\right)^2}}
                        {\sigma_\epsilon},

and draw a small constant :math:`\text{bias}^{(k)} \in
[-\texttt{bias\_level},\texttt{bias\_level}]`. Centered Taylor
statistics are unaffected by bias. See :footcite:t:`Taylor2001`
for interpretation; broader verification context is covered in
:footcite:p:`Jolliffe2012`.

See Also
--------
kdiagram.plot.evaluation.taylor_diagram
    Flexible Taylor diagram from raw arrays or pre-computed stats.

kdiagram.plot.evaluation.plot_taylor_diagram
    Standard Taylor diagram from raw arrays.

kdiagram.plot.evaluation.plot_taylor_diagram_in
    Taylor diagram with background shading.

Examples
--------
>>>  # Get arrays and stats as a Bunch:
>>> 
>>> from kdiagram.datasets import make_taylor_data
>>> ds = make_taylor_data(n_models=2, seed=0)
>>> list(ds.frame.columns)
['reference', 'Model_A', 'Model_B']
>>> set(ds.stats.columns) == {'stddev', 'corrcoef'}
True
>>> 
>>> # Return only a DataFrame:
>>> 
>>> df = make_taylor_data(as_frame=True, seed=1)
>>> 'reference' in df.columns
True

References
----------
.. footbibliography::
"""


def make_multi_model_quantile_data(
    n_samples: int = 100,
    n_models: int = 3,
    quantiles: list[float] = None,
    prefix: str = "pred",
    model_names: list[str] | None = None,
    true_mean: float = 50.0,
    true_std: float = 10.0,
    bias_range: tuple[float, float] = (-2.0, 2.0),
    width_range: tuple[float, float] = (5.0, 15.0),
    noise_level: float = 1.0,
    seed: int | None = 202,
    as_frame: bool = False,
) -> Bunch | pd.DataFrame:

    # --- Input Validation ---
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
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
        raise ValueError("bias_range must be (min, max) with min <= max.")

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
    width_denominator = 0.9 - 0.1
    width_numerator = q_max - q_min
    if len(quantiles_sorted) > 1 and abs(width_numerator) > 1e-9:
        width_scale_factor = width_numerator / width_denominator
    else:
        width_scale_factor = (
            1.0  # No scaling needed if range is zero/single q
        )

    # --- Data Generation ---
    y_true = rng.normal(true_mean, true_std, n_samples)
    feature_1 = rng.uniform(0, 1, n_samples)
    feature_2 = rng.normal(5, 2, n_samples)

    data_dict = {  # Use dict to build data before DataFrame
        "y_true": y_true,
        "feature_1": feature_1,
        "feature_2": feature_2,
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
    for _i, model_name in enumerate(model_names_list):
        # Sample model-specific parameters
        model_bias = rng.uniform(bias_range[0], bias_range[1])
        model_width = rng.uniform(width_range[0], width_range[1])

        # Store generated quantiles temporarily before sorting
        temp_model_quantiles = {}

        # Generate Q50 (median) prediction first
        q50_pred = y_true + model_bias + rng.normal(0, noise_level, n_samples)
        q50_col_name = f"{prefix}_{model_name}_q0.5"
        temp_model_quantiles[0.5] = q50_pred
        # Add name to tracking dict immediately
        prediction_columns_dict[model_name].append(q50_col_name)

        # Generate other quantiles based on Q50 and target width
        for q in quantiles_sorted:
            if q == q_median:
                continue  # Skip if median

            # Calculate offset using proportional distance from median
            # Avoid division by zero if q_max == q_min
            q_range = q_max - q_min
            # from scipy.stats import norm
            # z_score = norm.ppf(q) # Z-score for the quantile
            # Use standard deviation implied by width (e.g. q90-q10 ~ 2.56*std)
            # implied_std = model_width / (norm.ppf(q_max) - norm.ppf(q_min))
            #  if (q_max != q_min) else 1.0
            # quantile_offset = z_score * implied_std

            if abs(q_range) > 1e-9 and abs(width_scale_factor) > 1e-9:
                quantile_offset = (
                    (model_width / width_scale_factor)
                    * (q - q_median)
                    / q_range
                    * 2
                )
            else:  # Handle single quantile or zero range
                quantile_offset = 0

            q_pred = (
                q50_pred
                + quantile_offset
                + rng.normal(
                    0,
                    noise_level / 2,
                    n_samples,  # Slightly less noise for bounds
                )
            )
            temp_model_quantiles[q] = q_pred

        # Ensure quantile order and add to main data dict
        # Create temporary DF for sorting this model's quantiles
        model_data_temp = pd.DataFrame(temp_model_quantiles)
        # Sort values row-wise
        sorted_data = np.sort(model_data_temp.values, axis=1)
        # Assign sorted values back, creating final column names
        for k, q in enumerate(quantiles_sorted):
            col_name = f"{prefix}_{model_name}_q{q:.2f}".rstrip("0").rstrip(
                "."
            )
            data_dict[col_name] = sorted_data[:, k]
            # Add to tracking dict if not already added (handles Q50 case)
            if col_name not in prediction_columns_dict[model_name]:
                prediction_columns_dict[model_name].append(col_name)

    # Create the final DataFrame
    df = pd.DataFrame(data_dict)

    # Order columns somewhat logically
    feature_names = ["feature_1", "feature_2"]
    target_name = ["y_true"]
    pred_cols_sorted = sorted(
        [col for col in df.columns if col.startswith(prefix)]
    )
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

        descr = textwrap.dedent(
            f"""\
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
        """
        )

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
            DESCR=descr,
        )


make_multi_model_quantile_data.__doc__ = r"""
Generate multi-model quantile forecast data for a single horizon.

Simulates a target variable :math:`y_{\text{true}}` and
quantile predictions (e.g., Q10/Q50/Q90) from several models
for the **same** forecast time. Each model can have its own
systematic bias and characteristic interval width, enabling
reproducible examples for coverage/calibration and cross-model
comparisons :footcite:p:`Gneiting2007b, Jolliffe2012`.

Parameters
----------
n_samples : int, default=100
    Number of rows (independent samples/locations).

n_models : int, default=3
    Number of simulated models providing quantile forecasts.

quantiles : list of float, default=[0.1, 0.5, 0.9]
    Quantile levels in ``(0, 1)`` to generate for **each** model.
    Must include ``0.5`` (the median). The list is de-duplicated
    and sorted internally.

prefix : str, default='pred'
    Base prefix for prediction columns. Final names follow
    ``{prefix}_{model_name}_q{quantile}``.

model_names : list of str, optional
    Custom model names of length ``n_models``. If ``None``,
    ``'Model_A'``, ``'Model_B'``, … are generated.

true_mean : float, default=50.0
    Mean of the Normal distribution used to draw ``y_true``.

true_std : float, default=10.0
    Standard deviation of the Normal distribution for ``y_true``.

bias_range : tuple of (float, float), default=(-2.0, 2.0)
    Uniform range from which a model-specific bias for Q50 is
    sampled and added to ``y_true``.

width_range : tuple of (float, float), default=(5.0, 15.0)
    Uniform range for the target **overall** interval width
    (e.g., Q90–Q10) of each model.

noise_level : float, default=1.0
    Standard deviation of independent Gaussian noise added to
    each generated quantile series.

seed : int or None, default=202
    NumPy RNG seed (``default_rng``). If ``None``, a fresh RNG is used.

as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` with
    arrays/metadata; if ``True``, return only the pandas ``DataFrame``.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default), a Bunch with:

    - ``frame`` : pandas ``DataFrame`` of shape
      ``(n_samples, 3 + n_models * n_quantiles)`` containing
      ``'y_true'``, two auxiliary features, and all quantile columns.
    - ``data`` : ``ndarray`` with numeric feature + prediction columns.
    - ``feature_names`` : ``['feature_1', 'feature_2']``.
    - ``target_names`` : ``['y_true']``.
    - ``target`` : ``ndarray`` of ``y_true`` values.
    - ``model_names`` : list of model labels.
    - ``quantile_levels`` : sorted list of unique quantiles.
    - ``prediction_columns`` : dict mapping each model name to its
      list of quantile column names.
    - ``prefix`` : the column prefix.
    - ``DESCR`` : human-readable description.

    If ``as_frame=True``, only the pandas ``DataFrame`` is returned.

Raises
------
ValueError
    If ``0.5`` is not in ``quantiles``, if name/range lengths are
    inconsistent, or if ranges are invalid.

TypeError
    If non-numeric inputs prevent computation.

Notes
-----
**Generation model.** Draw the truth as
:math:`y_{\text{true}} \sim \mathcal{N}(\mu, \sigma^2)` with
``mu=true_mean`` and ``sigma=true_std``. For model :math:`m`, let
:math:`b^{(m)}` be a sampled bias
and :math:`W^{(m)}` a sampled overall width (e.g., Q90–Q10). The
median prediction (Q50) is

.. math::

   q_{0.5}^{(m)} \;=\; y_{\text{true}} \;+\; b^{(m)} \;+\;
   \varepsilon^{(m)}, \qquad
   \varepsilon^{(m)} \sim \mathcal{N}(0, \sigma_\varepsilon^2),

with ``sigma_ε = noise_level``. Other quantiles are created by
adding offsets proportional to their distance from the median and
scaled so that the extreme quantiles span approximately
:math:`W^{(m)}`; small independent noise is then added. Finally, for
each row we sort the model’s quantile values to enforce
:math:`q_{\alpha} \le q_{0.5} \le q_{\beta}` (e.g., Q10 ≤ Q50 ≤ Q90),
which is useful for coverage and calibration diagnostics
:footcite:p:`Gneiting2007b, Jolliffe2012`.

Two auxiliary columns (``feature_1``, ``feature_2``) are included
for convenience in examples; they do not influence the simulated
target or quantiles.

See Also
--------
make_uncertainty_data
    Temporal multi-period quantiles with drift/consistency controls.
make_taylor_data
    Synthetic data tailored for Taylor diagram evaluation.
kdiagram.plot.uncertainty.plot_coverage
    Aggregate empirical coverage vs nominal.
kdiagram.plot.uncertainty.plot_temporal_uncertainty
    General polar visualization for multiple series.

Examples
--------
>>> # As a Bunch with metadata:
>>> 
>>> from kdiagram.datasets import make_multi_model_quantile_data
>>> ds = make_multi_model_quantile_data(n_samples=50, n_models=2, seed=1)
>>> ds.model_names
['Model_A', 'Model_B']
>>> sorted(ds.quantile_levels)
[0.1, 0.5, 0.9]
>>> ds.prediction_columns['Model_A'][:3]  # doctest: +ELLIPSIS
['pred_Model_A_q0.1', 'pred_Model_A_q0.5', 'pred_Model_A_q0.9']
>>> 
>>> # As a DataFrame:
>>> 
>>> df = make_multi_model_quantile_data(as_frame=True, seed=2)
>>> set(['y_true','feature_1','feature_2']).issubset(df.columns)
True

References
----------
.. footbibliography::
"""
