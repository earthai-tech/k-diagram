#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""Forecast utilites"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..decorators import isdf
from .handlers import columns_manager
from .validator import exist_features, validate_yy

__all__ = [
    "calculate_probabilistic_scores",
    "pivot_forecasts_long",
    "compute_interval_width",
    "bin_by_feature",
    "compute_forecast_errors",
]


def calculate_probabilistic_scores(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> pd.DataFrame:

    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )

    # --- PIT Calculation ---
    sort_idx = np.argsort(quantiles)
    sorted_preds = y_preds_quantiles[:, sort_idx]
    pit_values = np.mean(sorted_preds <= y_true[:, np.newaxis], axis=1)

    # --- Sharpness Calculation ---
    lower_bound = y_preds_quantiles[:, np.argmin(quantiles)]
    upper_bound = y_preds_quantiles[:, np.argmax(quantiles)]
    sharpness = upper_bound - lower_bound

    # --- CRPS Calculation (approximated via pinball loss) ---
    pinball_loss = np.where(
        y_true[:, np.newaxis] >= y_preds_quantiles,
        (y_true[:, np.newaxis] - y_preds_quantiles) * quantiles,
        (y_preds_quantiles - y_true[:, np.newaxis]) * (1 - quantiles),
    )
    crps = np.mean(pinball_loss, axis=1)

    return pd.DataFrame(
        {"pit_value": pit_values, "sharpness": sharpness, "crps": crps}
    )


calculate_probabilistic_scores.__doc__ = r"""
Calculates probabilistic scores for each observation.

Computes the Probability Integral Transform (PIT), sharpness
(interval width), and Continuous Ranked Probability Score (CRPS)
for each forecast-observation pair. This utility provides a
per-observation breakdown of key probabilistic metrics.

Parameters
----------
y_true : np.ndarray
    1D array of observed (true) values.
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts. Each row corresponds to an
    observation in ``y_true``, and each column is a specific
    quantile forecast.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of ``y_preds_quantiles``.

Returns
-------
pd.DataFrame
    A DataFrame with columns 'pit_value', 'sharpness', and 'crps',
    where each row corresponds to an observation.

See Also
--------
compute_pit : Calculate only the PIT values.
compute_crps : Calculate only the average CRPS score.
compute_winkler_score : Score a single prediction interval.

Notes
-----
This function calculates three fundamental scores for assessing
the quality of a probabilistic forecast, which is judged by the
joint properties of calibration and sharpness
:footcite:p:`Gneiting2007b`.

1.  **Probability Integral Transform (PIT)**: This score assesses
    **calibration**. For each observation :math:`y_i`, the PIT is
    approximated as the fraction of forecast quantiles less than
    or equal to the observation.

    .. math::
       :label: eq:pit_score

       \text{PIT}_i = \frac{1}{M} \sum_{j=1}^{M}
       \mathbf{1}\{q_{i,j} \le y_i\}

2.  **Sharpness**: This score assesses **precision**. It is the
    width of the prediction interval between the lowest
    (:math:`q_{min}`) and highest (:math:`q_{max}`) provided
    quantiles for each observation :math:`i`.

    .. math::
       :label: eq:sharpness_score_ind

       \text{Sharpness}_i = y_{i, q_{max}} - y_{i, q_{min}}

3.  **Continuous Ranked Probability Score (CRPS)**: This is an
    overall score that rewards both calibration and sharpness.
    It is approximated as the average of the **Pinball Loss**
    across all :math:`M` quantiles for each observation :math:`i`.

    .. math::
       :label: eq:crps_score_ind

       \text{CRPS}_i \approx \frac{1}{M} \sum_{j=1}^{M}
       2 \mathcal{L}_{\tau_j}(q_{i,j}, y_i)

Examples
--------
>>> import numpy as np
>>> from scipy.stats import norm
>>> from kdiagram.utils.forecast_utils import calculate_probabilistic_scores
>>>
>>> # Generate synthetic data
>>> np.random.seed(42)
>>> n_samples = 5
>>> y_true = np.random.normal(loc=10, scale=2, size=n_samples)
>>> quantiles = np.array([0.1, 0.5, 0.9])
>>> y_preds = norm.ppf(
...     quantiles, loc=y_true[:, np.newaxis], scale=1.5
... )
>>>
>>> # Calculate the scores
>>> scores_df = calculate_probabilistic_scores(
...     y_true, y_preds, quantiles
... )
>>> print(scores_df)
   pit_value  sharpness      crps
0   0.666667   3.844655  0.865381
1   0.333333   3.844655  0.892013
2   0.666667   3.844655  1.269438
3   0.666667   3.844655  0.472782
4   0.333333   3.844655  1.171358

References
----------
.. footbibliography::
"""


@isdf
def pivot_forecasts_long(
    df: pd.DataFrame,
    qlow_cols: list[str],
    q50_cols: list[str],
    qup_cols: list[str],
    horizon_labels: list[str] | None = None,
    id_vars: str | list[str] | None = None,
) -> pd.DataFrame:
    if not (len(qlow_cols) == len(q50_cols) == len(qup_cols)):
        raise ValueError("Quantile column lists must have the same length.")

    if not horizon_labels:
        horizon_labels = [f"H{i+1}" for i in range(len(qlow_cols))]

    if len(horizon_labels) != len(qlow_cols):
        raise ValueError(
            "Length of horizon_labels must match"
            " the number of quantile columns."
        )

    id_vars = columns_manager(id_vars) or []

    # Create temporary mapping dataframes for melting
    df_long_list = []
    for i, label in enumerate(horizon_labels):
        temp_df = df[
            id_vars + [qlow_cols[i], q50_cols[i], qup_cols[i]]
        ].copy()
        temp_df["horizon"] = label
        temp_df.rename(
            columns={
                qlow_cols[i]: "q_low",
                q50_cols[i]: "q_median",
                qup_cols[i]: "q_high",
            },
            inplace=True,
        )
        df_long_list.append(temp_df)

    return pd.concat(df_long_list, ignore_index=True)


pivot_forecasts_long.__doc__ = r"""
Reshapes multi-horizon forecast data from wide to long format.

This is a powerful data wrangling utility that transforms a
DataFrame with separate columns for each horizon's quantiles
(e.g., 'q10_2023', 'q50_2023', 'q10_2024', 'q50_2024') into a
"long" format DataFrame. The resulting table has dedicated
columns for 'horizon', 'q_low', 'q_median', and 'q_high', which
is often a more convenient structure for plotting and analysis.

Parameters
----------
df : pd.DataFrame
    The input DataFrame in wide format.
qlow_cols : list of str
    List of column names for the lower quantile, one for each
    forecast horizon, in order.
q50_cols : list of str
    List of column names for the median quantile, in the same
    horizon order.
qup_cols : list of str
    List of column names for the upper quantile, in the same
    horizon order.
horizon_labels : list of str, optional
    Custom labels for each forecast horizon. If not provided,
    generic labels like 'H1', 'H2' will be generated. The
    length must match the number of quantile columns.
id_vars : str or list of str, optional
    Identifier column(s) to keep in the long-format DataFrame
    (e.g., a location or sample ID). These columns will be
    repeated for each horizon.

Returns
-------
pd.DataFrame
    The reshaped DataFrame in long format.

Raises
------
ValueError
    If the lengths of the quantile column lists or the
    ``horizon_labels`` are inconsistent.

See Also
--------
pandas.melt : The underlying pandas function for unpivoting.
plot_horizon_metrics : A plot that benefits from this data format.

Examples
--------
>>> import pandas as pd
>>> from kdiagram.utils.forecast_utils import pivot_forecasts_long
>>>
>>> # Create a sample wide-format DataFrame
>>> df_wide = pd.DataFrame({
...     'location_id': ['A', 'B'],
...     'q10_2023': [10, 12], 'q50_2023': [15, 18], 'q90_2023': [20, 24],
...     'q10_2024': [12, 14], 'q50_2024': [18, 21], 'q90_2024': [24, 28],
... })
>>>
>>> # Reshape the data
>>> df_long = pivot_forecasts_long(
...     df_wide,
...     qlow_cols=['q10_2023', 'q10_2024'],
...     q50_cols=['q50_2023', 'q50_2024'],
...     qup_cols=['q90_2023', 'q90_2024'],
...     horizon_labels=['Year 2023', 'Year 2024'],
...     id_vars='location_id'
... )
>>> print(df_long)
  location_id    horizon  q_low  q_median  q_high
0           A  Year 2023     10        15      20
1           B  Year 2023     12        18      24
2           A  Year 2024     12        18      24
3           B  Year 2024     14        21      28
"""


@isdf
def compute_interval_width(
    df: pd.DataFrame,
    *quantile_pairs: list[str | float],
    prefix: str = "width_",
    inplace: bool = False,
) -> pd.DataFrame:

    if not quantile_pairs:
        raise ValueError(
            "At least one pair of quantile" " columns must be provided."
        )

    output_df = df if inplace else df.copy()

    for pair in quantile_pairs:
        if len(pair) != 2:
            raise ValueError(
                "Each quantile pair must contain exactly"
                f" two columns, but got {pair}."
            )

        lower_col, upper_col = pair
        exist_features(df, features=[lower_col, upper_col])

        width = output_df[upper_col] - output_df[lower_col]
        new_col_name = f"{prefix}{upper_col}"
        output_df[new_col_name] = width

    return output_df


compute_interval_width.__doc__ = r"""
Computes the width of one or more prediction intervals.

This is a fundamental data preparation utility that calculates the
difference between upper and lower quantile columns for one or
more forecast intervals. The resulting interval width is a key
measure of a forecast's **sharpness**.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the quantile forecast columns.
quantile_pairs : list of (str or float)
    One or more lists or tuples, each containing two elements in
    the order: ``[lower_quantile_col, upper_quantile_col]``.
prefix : str, default='width\_'
    The prefix for the new interval width column names. The new
    name will be f"{prefix}{upper_col_name}".
inplace : bool, default=False
    If ``True``, modifies the original DataFrame by adding the new
    columns. If ``False`` (default), returns a new DataFrame.

Returns
-------
pd.DataFrame
    The DataFrame with the new interval width column(s) added.

Raises
------
ValueError
    If a provided pair does not contain exactly two column names.

See Also
--------
plot_polar_sharpness : A plot that directly uses this metric.
compute_winkler_score : A score that uses interval width as a component.

Notes
-----
The width of a prediction interval is the most direct measure of
a forecast's **sharpness**, a key property of probabilistic
forecasts :footcite:p:`Gneiting2007b`. A smaller width
indicates a more precise, or sharper, forecast.

For a given observation :math:`i`, the interval width :math:`w_i`
is the simple difference between the upper and lower quantile
forecasts:

.. math::

   w_i = q_{upper, i} - q_{lower, i}

Examples
--------
>>> import pandas as pd
>>> from kdiagram.utils.forecast_utils import compute_interval_width
>>>
>>> df = pd.DataFrame({
...     'q10_model_A': [1, 2], 'q90_model_A': [10, 12],
...     'q05_model_A': [0, 1], 'q95_model_A': [11, 13]
... })
>>>
>>> # Calculate the 80% and 90% interval widths
>>> widths_df = compute_interval_width(
...     df, ['q10_model_A', 'q90_model_A'], ['q05_model_A', 'q95_model_A']
... )
>>> print(widths_df)
   q10_model_A  q90_model_A  q05_model_A  q95_model_A  width_q90_model_A  width_q95_model_A
0            1           10            0           11                  9                 11
1            2           12            1           13                 10                 12

References
----------
.. footbibliography::
"""


@isdf
def bin_by_feature(
    df: pd.DataFrame,
    bin_on_col: str,
    target_cols: str | list[str],
    n_bins: int = 10,
    agg_funcs: str | list[str] | dict = "mean",
) -> pd.DataFrame:
    target_cols = columns_manager(target_cols)
    required_cols = [bin_on_col] + target_cols
    exist_features(df, features=required_cols)

    # Create bins using pandas.cut
    bin_labels = f"{bin_on_col}_bin"
    df_binned = df.copy()
    df_binned[bin_labels] = pd.cut(df_binned[bin_on_col], bins=n_bins)

    # Group by the new bins and aggregate
    stats = df_binned.groupby(bin_labels, observed=False)[target_cols].agg(
        agg_funcs
    )

    return stats.reset_index()


bin_by_feature.__doc__ = r"""
Bins data by a feature and computes aggregate statistics.

This is a powerful data wrangling utility that groups a DataFrame
into bins based on the values in a specified column
(``bin_on_col``). It then calculates aggregate statistics (like
mean, std, etc.) for one or more target columns within each bin.
This is the core logic behind plots like ``plot_error_bands``.

Parameters
----------
df : pd.DataFrame
    The input DataFrame.
bin_on_col : str
    The name of the column whose values will be used for binning.
    This column must contain numeric data.
target_cols : str or list of str
    The name(s) of the column(s) for which to compute statistics.
n_bins : int, default=10
    The number of equal-width bins to create.
agg_funcs : str, list of str, or dict, default='mean'
    The aggregation function(s) to apply. Can be any function
    accepted by pandas' ``.agg()`` method (e.g., 'mean', 'std',
    ['mean', 'std'], or {'col_A': 'sum'}).

Returns
-------
pd.DataFrame
    A DataFrame containing the aggregate statistics for each bin.

See Also
--------
pandas.cut : The underlying pandas function used for binning.
pandas.DataFrame.groupby : The underlying pandas function for aggregation.
plot_error_bands : A plot that uses this binning logic.

Notes
-----
This function first uses ``pandas.cut`` to partition the values
in ``bin_on_col`` into ``n_bins`` discrete, equal-width intervals.
It then uses ``pandas.DataFrame.groupby`` to group the DataFrame
by these new bins and applies the specified aggregation
function(s) to the ``target_cols`` for each group.

Examples
--------
>>> import pandas as pd
>>> from kdiagram.utils.forecast_utils import bin_by_feature
>>>
>>> df = pd.DataFrame({
...     'forecast_value': [10, 12, 20, 22, 30, 32],
...     'error': [-1, 1.5, -2, 2.5, -3, 3.5]
... })
>>>
>>> # Calculate the mean and standard deviation of the error,
>>> # binned by the forecast value.
>>> binned_stats = bin_by_feature(
...     df,
...     bin_on_col='forecast_value',
...     target_cols='error',
...     n_bins=3,
...     agg_funcs=['mean', 'std']
... )
>>> print(binned_stats)
  forecast_value_bin  mean       std
0      (9.978, 17.333]  0.25  1.767767
1   (17.333, 24.667]  0.25  3.181981
2     (24.667, 32.0]  0.25  4.596194
"""


@isdf
def compute_forecast_errors(
    df: pd.DataFrame,
    actual_col: str,
    *pred_cols: str,
    error_type: Literal["raw", "absolute", "squared", "percentage"] = "raw",
    prefix: str = "error_",
    inplace: bool = False,
) -> pd.DataFrame:

    if not pred_cols:
        raise ValueError("At least one prediction column must be provided.")

    required_cols = [actual_col] + list(pred_cols)
    exist_features(df, features=required_cols)

    output_df = df if inplace else df.copy()

    actual_vals = output_df[actual_col]

    for pred_col in pred_cols:
        pred_vals = output_df[pred_col]
        new_col_name = f"{prefix}{pred_col}"

        if error_type == "raw":
            errors = actual_vals - pred_vals
        elif error_type == "absolute":
            errors = (actual_vals - pred_vals).abs()
        elif error_type == "squared":
            errors = (actual_vals - pred_vals) ** 2
        elif error_type == "percentage":
            # Avoid division by zero
            errors = (
                100
                * (actual_vals - pred_vals)
                / actual_vals.replace(0, np.nan)
            )
        else:
            raise ValueError(f"Unknown error_type: '{error_type}'")

        output_df[new_col_name] = errors

    return output_df


compute_forecast_errors.__doc__ = r"""
Computes forecast errors for one or more models.

This is a core data preparation utility that calculates the
difference between true and predicted values. It supports several
common error types and can operate on multiple prediction columns
at once, making it easy to prepare data for the diagnostic plots
in the :mod:`kdiagram.plot.errors` module.

Parameters
----------
df : pd.DataFrame
    The input DataFrame containing the actual and predicted values.
actual_col : str
    The name of the column containing the true observed values.
*pred_cols : str
    One or more column names containing the predicted values from
    different models.
error_type : {'raw', 'absolute', 'squared', 'percentage'}, default='raw'
    The type of error to calculate:
        
    - 'raw': :math:`y_{true} - y_{pred}`
    - 'absolute': :math:`|y_{true} - y_{pred}|`
    - 'squared': :math:`(y_{true} - y_{pred})^2`
    - 'percentage': :math:`100 \cdot (y_{true} - y_{pred}) / y_{true}`
    
prefix : str, default='error\_'
    The prefix to add to the new error column names. For example,
    a prediction column 'Model_A' will become 'error_Model_A'.
inplace : bool, default=False
    If ``True``, modifies the original DataFrame by adding the new
    columns. If ``False`` (default), returns a new DataFrame.

Returns
-------
pd.DataFrame
    The DataFrame with the new error column(s) added.

Raises
------
ValueError
    If no prediction columns are provided or if the specified
    ``error_type`` is invalid.

See Also
--------
plot_error_violins : A plot that directly uses these error columns.
plot_error_bands : A plot that uses these errors for aggregation.

Notes
-----
The forecast error (or residual), :math:`e_i`, for an
observation :math:`i` is the fundamental quantity for diagnosing
model performance. This function calculates it in several forms:

1.  **Raw Error**: The simple difference, which preserves the
    direction of the error (positive for under-prediction,
    negative for over-prediction).

    .. math::

       e_i = y_{true,i} - y_{pred,i}

2.  **Absolute Error**: The magnitude of the error, which is
    always non-negative.

    .. math::

       e_{abs,i} = |y_{true,i} - y_{pred,i}|

3.  **Squared Error**: Penalizes larger errors more heavily.

    .. math::

       e_{sq,i} = (y_{true,i} - y_{pred,i})^2

4.  **Percentage Error**: Expresses the error as a percentage
    of the true value. Note that this can be unstable if
    :math:`y_{true,i}` is close to zero.

    .. math::

       e_{\%,i} = 100 \cdot \frac{y_{true,i} - y_{pred,i}}{y_{true,i}}

Examples
--------
>>> import pandas as pd
>>> from kdiagram.utils.forecast_utils import compute_forecast_errors
>>>
>>> df = pd.DataFrame({
...     'actual': [10, 20, 30],
...     'model_A_preds': [12, 18, 33],
...     'model_B_preds': [10, 25, 28],
... })
>>>
>>> # Calculate raw and absolute errors for both models
>>> df_errors_raw = compute_forecast_errors(
...     df, 'actual', 'model_A_preds', 'model_B_preds'
... )
>>> df_errors_abs = compute_forecast_errors(
...     df, 'actual', 'model_A_preds', 'model_B_preds',
...     error_type='absolute', prefix='abs_error_'
... )
>>> print(df_errors_raw)
   actual  model_A_preds  model_B_preds  error_model_A_preds  error_model_B_preds
0      10             12             10                   -2                    0
1      20             18             25                    2                   -5
2      30             33             28                   -3                    2
"""
