#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Callable, Literal, Union, overload

import numpy as np
import pandas as pd
from scipy.stats import kstest, uniform

from .handlers import columns_manager
from .validator import exist_features, validate_length_range, validate_yy

FrameLike = Union[np.ndarray, pd.DataFrame, pd.Series]

__all__ = [
    "minmax_scaler",
    "compute_coverage_score",
    "compute_winkler_score",
    "build_cdf_interpolator",
    "calculate_calibration_error",
    "compute_pinball_loss",
    "compute_pit",
    "compute_crps",
    "get_forecast_arrays",
]


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: None = None,
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["numpy"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> np.ndarray: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: None,
    pred_cols: str,
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["numpy"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> np.ndarray: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: None,
    pred_cols: list[str],
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["numpy"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> np.ndarray: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: str | list[str],
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["numpy"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: None = None,
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["pandas"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> pd.Series: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: None,
    pred_cols: str,
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["pandas"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> pd.Series: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: None,
    pred_cols: list[str],
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["pandas"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> pd.DataFrame: ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: str | list[str],
    *,
    drop_na: bool = ...,
    na_policy: Literal["any", "all", "none"] = ...,
    fillna: object | None = ...,
    return_as: Literal["pandas"] = ...,
    squeeze: bool = ...,
    with_index: bool = ...,
    sort_index: bool = ...,
    dtype: object | None = ...,
    ensure_numeric: bool = ...,
    coerce_numeric: bool = ...,
    copy: bool = ...,
) -> tuple[pd.Series, pd.Series | pd.DataFrame]: ...


def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str | None = None,
    pred_cols: str | list[str] | None = None,
    *,
    drop_na: bool = True,
    na_policy: Literal["any", "all", "none"] = "any",
    fillna: object | None = None,
    return_as: Literal["numpy", "pandas"] = "numpy",
    squeeze: bool = True,
    with_index: bool = False,
    sort_index: bool = False,
    dtype: object | None = None,
    ensure_numeric: bool = False,
    coerce_numeric: bool = False,
    copy: bool = True,
):
    if actual_col is None and pred_cols is None:
        raise ValueError(
            "Provide at least one of 'actual_col' or 'pred_cols'."
        )

    # collect required columns
    cols: list[str] = []
    if actual_col:
        cols.append(actual_col)
    pcols = columns_manager(pred_cols) or []
    cols.extend(pcols)

    # validate presence
    exist_features(df, features=cols)

    # subset and optional copy/sort
    sub = df.loc[:, cols].copy() if copy else df.loc[:, cols]
    if sort_index:
        sub = sub.sort_index()

    # optional fill
    if fillna is not None:
        if fillna == "ffill":
            sub = sub.ffill()
        elif fillna == "bfill":
            sub = sub.bfill()
        else:
            sub = sub.fillna(fillna)

        # if fillna in ("ffill", "bfill"):
        #     sub = sub.fillna(method=str(fillna))
        # else:
        #     sub = sub.fillna(fillna)

    # drop NA per policy
    if drop_na and na_policy != "none":
        how = "any" if na_policy == "any" else "all"
        sub = sub.dropna(how=how)

    # optional numeric enforcement
    if ensure_numeric:
        errors = "coerce" if coerce_numeric else "raise"
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors=errors)

    # extract pieces
    y_true = sub[actual_col] if actual_col else None
    y_pred = None
    if pcols:
        y_pred = sub[pcols]

    # cast pandas dtypes if requested
    if return_as == "pandas" and dtype is not None:
        if y_true is not None:
            y_true = y_true.astype(dtype)
        if y_pred is not None:
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.astype(dtype)
            else:
                y_pred = y_pred.astype(dtype)

    # prepare NumPy outputs
    if return_as == "numpy":
        if y_true is not None:
            y_true = y_true.to_numpy(dtype=dtype)
        if y_pred is not None:
            arr = y_pred.to_numpy(dtype=dtype)
            if squeeze and arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()
            y_pred = arr

    # squeeze pandas single-column preds to Series
    if (
        return_as == "pandas"
        and isinstance(pred_cols, str)
        and y_pred is not None
        and isinstance(y_pred, pd.DataFrame)
    ):
        y_pred = y_pred.iloc[:, 0]

    # index handling
    if with_index:
        idx = sub.index.to_numpy() if return_as == "numpy" else sub.index
        if y_true is not None and y_pred is not None:
            return idx, y_true, y_pred
        if y_true is not None:
            return idx, y_true
        return idx, y_pred  # type: ignore[return-value]

    # standard returns
    if y_true is not None and y_pred is not None:
        return y_true, y_pred
    if y_true is not None:
        return y_true
    return y_pred  # type: ignore[return-value]


get_forecast_arrays.__doc__ = r"""
Extract true and/or predicted values from a DataFrame.

This is a flexible bridge between a DataFrame-centric workflow
and NumPy-based utilities. It supports dropping or filling NAs,
numeric coercion, and optional index return, providing a robust
way to prepare data for analysis.

Parameters
----------
df : pd.DataFrame
    The source DataFrame.
actual_col : str, optional
    The name of the column holding the ground-truth values.
pred_cols : str or list of str, optional
    The name(s) of the prediction column(s). A string implies a
    single point forecast; a list implies multiple columns, such
    as for quantile forecasts.
drop_na : bool, default=True
    If ``True``, drop rows with missing data according to the
    ``na_policy``.
na_policy : {"any", "all", "none"}, default="any"
    The policy for dropping rows with NA values:
        
    - "any": Drop rows if any selected column has an NA.
    - "all": Drop rows only if all selected columns are NA.
    - "none": Do not drop rows based on NAs.
    
fillna : scalar, dict or {"ffill", "bfill"}, optional
    A value or method to use for filling NA values before any
    dropping occurs.
return_as : {"numpy", "pandas"}, default="numpy"
    The desired container type for the output.
squeeze : bool, default=True
    If ``True`` and a single prediction column is requested, the
    output will be squeezed to a 1D array or Series.
with_index : bool, default=False
    If ``True``, the DataFrame index is returned as the first
    item in the output tuple.
sort_index : bool, default=False
    If ``True``, the DataFrame is sorted by its index before
    extracting the data.
dtype : object, optional
    The target data type for the output arrays or Series.
ensure_numeric : bool, default=False
    If ``True``, raises an error if any selected column is not
    of a numeric data type.
coerce_numeric : bool, default=False
    If ``True`` and ``ensure_numeric=True``, attempts to convert
    non-numeric columns to a numeric type, with invalid
    parsing resulting in NaN.
copy : bool, default=True
    If ``True``, operates on a copy of the data, ensuring the
    original DataFrame is not modified.

Returns
-------
np.ndarray, pd.Series, pd.DataFrame, or tuple
    The return type depends on the input parameters:
        
    - If only ``actual_col`` is provided -> y_true
    - If only ``pred_cols`` is provided -> y_pred(s)
    - If both are provided -> (y_true, y_pred(s))
    
    If ``with_index=True``, the index is prepended to the
    return value(s).

See Also
--------
compute_forecast_errors : A utility that uses this function's output.
compute_pit : Another utility that benefits from this data extraction.

Notes
-----
This function is designed to be the primary entry point for
extracting data before passing it to the mathematical or plotting
functions in `k-diagram`. It provides a single, consistent
interface for handling various data cleaning and formatting tasks.

For ``return_as="numpy"`` with a single prediction column, the
output is a 1D array by default. To preserve the 2D column
vector shape ``(n, 1)``, set ``squeeze=False``.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from kdiagram.utils.mathext import get_forecast_arrays
>>>
>>> df = pd.DataFrame({
...     'actual': [10, 20, 30, 40, np.nan],
...     'pred_point': [12, 18, 33, 42, 48],
...     'q10': [8, 15, 25, 35, 45],
...     'q90': [12, 25, 35, 45, 55],
... })
>>>
>>> # Example 1: Get both true and quantile predictions as NumPy arrays
>>> y_true, y_preds_q = get_forecast_arrays(
...     df, actual_col='actual', pred_cols=['q10', 'q90']
... )
>>> print("--- True Values (NumPy) ---")
>>> print(y_true)
>>> print("\\n--- Quantile Predictions (NumPy) ---")
>>> print(y_preds_q)

.. code-block:: text
   :caption: Expected Output for Example 1

   --- True Values (NumPy) ---
   [10. 20. 30. 40.]

   --- Quantile Predictions (NumPy) ---
   [[ 8 12]
    [15 25]
    [25 35]
    [35 45]]

>>> # Example 2: Get a single prediction as a pandas Series
>>> y_preds_series = get_forecast_arrays(
...     df, pred_cols='pred_point', return_as='pandas', drop_na=False
... )
>>> print("\\n--- Point Predictions (pandas Series) ---")
>>> print(y_preds_series)

.. code-block:: text
   :caption: Expected Output for Example 2

   --- Point Predictions (pandas Series) ---
   0    12
   1    18
   2    33
   3    42
   4    48
   Name: pred_point, dtype: int64
"""


def compute_pit(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """
    Computes the Probability Integral Transform (PIT) for each observation.

    Parameters
    ----------
    y_true : np.ndarray
        1D array of the true observed values.
    y_preds_quantiles : np.ndarray
        2D array of quantile forecasts, with shape (n_samples, n_quantiles).
    quantiles : np.ndarray
        1D array of the quantile levels.

    Returns
    -------
    np.ndarray
        A 1D array of PIT values, one for each observation.
    """
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )
    # Sort quantiles and predictions to ensure correct calculation
    sort_idx = np.argsort(quantiles)
    sorted_preds = y_preds_quantiles[:, sort_idx]

    # PIT is the fraction of forecast quantiles <= the true value
    pit_values = np.mean(sorted_preds <= y_true[:, np.newaxis], axis=1)

    return pit_values


compute_pit.__doc__ = r"""
Computes the Probability Integral Transform (PIT) for each observation.

Parameters
----------
y_true : np.ndarray
    1D array of the true observed values.
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts, with shape
    ``(n_samples, n_quantiles)``.
quantiles : np.ndarray
    1D array of the quantile levels.

Returns
-------
np.ndarray
    A 1D array of PIT values, one for each observation.

See Also
--------
plot_pit_histogram : A visualization of these PIT values.
calculate_calibration_error : A summary score based on PIT values.

Notes
-----
The Probability Integral Transform (PIT) is a fundamental tool
for evaluating the calibration of probabilistic forecasts
:footcite:p:`Gneiting2007b`.

When the predictive distribution is represented by a finite set
of :math:`M` quantiles, the PIT value for each observation
:math:`y_i` is approximated as the fraction of forecast
quantiles that are less than or equal to the observation:

.. math::

   \text{PIT}_i = \frac{1}{M} \sum_{j=1}^{M}
   \mathbf{1}\{q_{i,j} \le y_i\}

where :math:`q_{i,j}` is the :math:`j`-th quantile forecast for
observation :math:`i`, and :math:`\mathbf{1}` is the indicator
function. A uniform distribution of PIT values indicates perfect
calibration.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import compute_pit
>>>
>>> # Define true values and quantile forecasts for 3 observations
>>> y_true = np.array([10, 1, 5.5])
>>> quantiles = np.array([0.1, 0.5, 0.9])
>>> y_preds = np.array([
...     [8, 11, 13],  # Forecast for y_true = 10
...     [0, 0.5, 2],  # Forecast for y_true = 1
...     [4, 5, 6]     # Forecast for y_true = 5.5
... ])
>>>
>>> # Calculate the PIT value for each observation
>>> pit_values = compute_pit(y_true, y_preds, quantiles)
>>> print(pit_values)

.. code-block:: text
   :caption: Expected Output

   [0.33333333 0.66666667 0.66666667]
   
References
----------
.. footbibliography::
"""


def compute_crps(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )

    # Reshape y_true for broadcasting
    y_true_reshaped = y_true[:, np.newaxis]

    # Calculate Pinball Loss for all quantiles at once
    pinball_losses = np.where(
        y_true_reshaped >= y_preds_quantiles,
        (y_true_reshaped - y_preds_quantiles) * quantiles,
        (y_preds_quantiles - y_true_reshaped) * (1 - quantiles),
    )

    # Average over quantiles for each observation, then over all observations
    return np.mean(np.mean(pinball_losses, axis=1))


compute_crps.__doc__ = r"""
Approximates the Continuous Ranked Probability Score (CRPS).

The CRPS is calculated as the average of the Pinball Loss across
all provided quantiles. It is a proper scoring rule that assesses
both calibration and sharpness simultaneously. A lower score is
better.

Parameters
----------
y_true : np.ndarray
    1D array of the true observed values.
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts.
quantiles : np.ndarray
    1D array of the quantile levels.

Returns
-------
float
    The average CRPS over all observations.

See Also
--------
compute_pinball_loss : The underlying metric for a single quantile.
plot_crps_comparison : A visualization of this score.

Notes
-----
The Continuous Ranked Probability Score (CRPS) is a widely
used metric for evaluating probabilistic forecasts
:footcite:p:`Gneiting2007b`. It is approximated here by
averaging the Pinball Loss :math:`\mathcal{L}_{\tau}` over all
:math:`M` provided quantile levels :math:`\tau`.

The Pinball Loss for a single quantile forecast :math:`q` at
level :math:`\tau` is:

.. math::
   :label: eq:pinball_loss_util

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

The final score is the average over all observations and all
quantiles.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import compute_crps
>>>
>>> # Define true values and quantile forecasts for 2 observations
>>> y_true = np.array([10, 25])
>>> quantiles = np.array([0.1, 0.5, 0.9])
>>> y_preds = np.array([
...     [8, 11, 13],  # Forecast for y_true = 10
...     [20, 22, 26]   # Forecast for y_true = 25
... ])
>>>
>>> # Calculate the average CRPS
>>> crps_score = compute_crps(y_true, y_preds, quantiles)
>>> print(f"Average CRPS: {crps_score:.3f}")

.. code-block:: text
   :caption: Expected Output

   Average CRPS: 1.467
   
References
----------
.. footbibliography::
"""


def calculate_calibration_error(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    # Validate inputs
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )

    # Calculate PIT values
    sort_idx = np.argsort(quantiles)
    sorted_preds = y_preds_quantiles[:, sort_idx]
    pit_values = np.mean(sorted_preds <= y_true[:, np.newaxis], axis=1)

    if len(pit_values) < 2:
        return 1.0  # Max penalty for insufficient data to test

    # Compare the empirical distribution
    # of PIT values to a uniform distribution
    ks_statistic, _ = kstest(pit_values, uniform.cdf)

    return ks_statistic


def build_cdf_interpolator(
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    # Sort quantiles and predictions to ensure correct interpolation
    sort_idx = np.argsort(quantiles)
    sorted_quantiles = quantiles[sort_idx]
    sorted_preds = np.asarray(y_preds_quantiles)[:, sort_idx]
    n_samples = sorted_preds.shape[0]

    def _interpolator(y_true: np.ndarray) -> np.ndarray:
        """The returned CDF interpolator function."""
        y_true = np.asarray(y_true)

        if len(y_true) != n_samples:
            raise ValueError(
                f"The number of true values ({len(y_true)}) must match the "
                f"number of forecast distributions the interpolator was "
                f"built with ({n_samples})."
            )
        pit_values = np.zeros_like(y_true, dtype=float)
        for i in range(len(y_true)):
            # Use np.interp for robust linear interpolation
            pit_values[i] = np.interp(
                y_true[i],
                sorted_preds[i, :],
                sorted_quantiles,
                left=0.0,  # Values below the lowest quantile get p=0
                right=1.0,  # Values above the highest quantile get p=1
            )
        return pit_values

    return _interpolator


build_cdf_interpolator.__doc__ = r"""
Builds an interpolator to act as a Cumulative Distribution Function.

This function takes a set of quantile forecasts and returns a
callable function that linearly interpolates between them. This
effectively creates an empirical, continuous Cumulative
Distribution Function (CDF) for each individual forecast, which
is a foundational tool for probabilistic analysis.

Parameters
----------
y_preds_quantiles : np.ndarray
    2D array of quantile forecasts, with shape
    ``(n_samples, n_quantiles)``. Each row represents a complete
    probabilistic forecast for a single observation.
quantiles : np.ndarray
    1D array of the quantile levels corresponding to the columns
    of the prediction array (e.g., ``[0.05, 0.1, ..., 0.95]``).

Returns
-------
Callable[[np.ndarray], np.ndarray]
    A function that takes a 1D array of observed values
    (``y_true``) and returns the corresponding PIT values, which
    are the CDF evaluated at each of those points.

Raises
------
ValueError
    If the number of `y_true` values passed to the returned
    interpolator does not match the number of forecast
    distributions it was built with.

See Also
--------
compute_pit : A simplified utility that uses this logic directly.
scipy.interpolate.interp1d : The underlying concept for interpolation.

Notes
-----
The Probability Integral Transform (PIT) is a key concept in
probabilistic forecast evaluation :footcite:p:`Gneiting2007b`.
For a continuous predictive CDF :math:`F`, the PIT of an
observation :math:`y` is :math:`F(y)`. This utility constructs
an empirical approximation of :math:`F` for each forecast.

The function works by creating a closure: the returned
``_interpolator`` function "remembers" the quantile forecasts it
was built with. For each observation :math:`y_i`, it performs a
linear interpolation using the corresponding forecast quantiles
:math:`\mathbf{q}_i = (q_{i,1}, ..., q_{i,M})` as the x-coordinates
and the quantile levels :math:`\mathbf{\tau} = (\tau_1, ..., \tau_M)`
as the y-coordinates. This allows you to estimate the cumulative
probability for any value of :math:`y_i`.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import build_cdf_interpolator
>>>
>>> # Forecasts for 3 observations at 3 quantiles (0.1, 0.5, 0.9)
>>> preds_quantiles = np.array([
...     [8, 10, 12],
...     [0, 1, 2],
...     [4, 5, 6]
... ])
>>> quantiles = np.array([0.1, 0.5, 0.9])
>>>
>>> # Build the interpolator
>>> cdf_func = build_cdf_interpolator(preds_quantiles, quantiles)
>>>
>>> # Now, use the interpolator to find the PIT for 3 observations
>>> y_true = np.array([10.0, 0.5, 5.5])
>>> pit_values = cdf_func(y_true)
>>> print(pit_values)
[0.5 0.3 0.7]

References
----------
.. footbibliography::
"""


def compute_coverage_score(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    *,
    method: Literal["within", "above", "below"] = "within",
    return_counts: bool = False,
) -> float | int:
    # Validate and convert inputs
    y_true, y_pred_lower = validate_yy(y_true, y_pred_lower)
    _, y_pred_upper = validate_yy(y_true, y_pred_upper)

    # Handle NaNs by creating a mask of valid (non-NaN) entries
    valid_mask = (
        ~np.isnan(y_true) & ~np.isnan(y_pred_lower) & ~np.isnan(y_pred_upper)
    )

    y_true_valid = y_true[valid_mask]
    lower_valid = y_pred_lower[valid_mask]
    upper_valid = y_pred_upper[valid_mask]

    n_valid = len(y_true_valid)
    if n_valid == 0:
        return 0.0 if not return_counts else 0

    if method == "within":
        count = np.sum(
            (y_true_valid >= lower_valid) & (y_true_valid <= upper_valid)
        )
    elif method == "above":
        count = np.sum(y_true_valid > upper_valid)
    elif method == "below":
        count = np.sum(y_true_valid < lower_valid)
    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose from"
            " 'within', 'above', or 'below'."
        )

    if return_counts:
        return int(count)

    return float(count / n_valid)


compute_coverage_score.__doc__ = r"""
Computes the coverage score for a given prediction interval.

This utility calculates the fraction (or count) of true values
that fall within, above, or below the specified prediction
interval. It is a fundamental metric for assessing the
calibration of a forecast's uncertainty bounds.

Parameters
----------
y_true : np.ndarray
    1D array of the true observed values.
y_pred_lower : np.ndarray
    1D array of the lower bound of the prediction interval.
y_pred_upper : np.ndarray
    1D array of the upper bound of the prediction interval.
method : {'within', 'above', 'below'}, default='within'
    The type of coverage to calculate:

    - 'within': The standard coverage score. Calculates the
      proportion of true values such that
      `lower <= true <= upper`.
    - 'above': Calculates the proportion of true values that
      are strictly *above* the upper bound (`true > upper`).
    - 'below': Calculates the proportion of true values that
      are strictly *below* the lower bound (`true < lower`).
      
return_counts : bool, default=False
    If ``True``, returns the raw count of observations matching
    the condition instead of the proportion (a float between
    0 and 1).

Returns
-------
float or int
    The coverage score as a proportion or a raw count.

See Also
--------
plot_coverage : A visualization of this score.
compute_winkler_score : A score that penalizes for lack of coverage.

Notes
-----
The empirical coverage is a key diagnostic for checking if a
model's prediction intervals are well-calibrated. For a given
:math:`(1-\alpha) \cdot 100\%` prediction interval, the
empirical coverage should be close to :math:`1-\alpha`.

For the standard 'within' method, the coverage for a set of
:math:`N` observations is calculated as:

.. math::
   :label: eq:coverage_score

   \text{Coverage} = \frac{1}{N} \sum_{i=1}^{N}
   \mathbf{1}\{y_{lower,i} \le y_{true,i} \le y_{upper,i}\}

where :math:`\mathbf{1}` is the indicator function. The 'above'
and 'below' methods are useful for diagnosing the direction of
miscalibration.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import compute_coverage_score
>>>
>>> y_true = np.array([1, 2, 3, 4, 5, 6])
>>> y_lower = np.array([0, 3, 2, 5, 4, 7])
>>> y_upper = np.array([2, 4, 4, 6, 6, 8])
>>>
>>> # Calculate the standard coverage (4 out of 6 are within)
>>> coverage = compute_coverage_score(y_true, y_lower, y_upper)
>>> print(f"Coverage Score: {coverage:.2f}")

.. code-block:: text
   :caption: Expected Output

   Coverage Score: 0.67

>>> # Calculate the number of points below the interval
>>> count_below = compute_coverage_score(
...     y_true, y_lower, y_upper, method='below', return_counts=True
... )
>>> print(f"Count below interval: {count_below}")

.. code-block:: text
   :caption: Expected Output

   Count below interval: 2
"""


def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    quantile: float,
) -> float:
    # Validate and handle NaNs
    y_true, y_pred_quantile = validate_yy(y_true, y_pred_quantile)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred_quantile)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred_quantile[valid_mask]

    if len(y_true_valid) == 0:
        return np.nan

    if not (0 < quantile < 1):
        raise ValueError("Quantile level must be between 0 and 1.")

    # Calculate Pinball Loss
    loss = np.where(
        y_true_valid >= y_pred_valid,
        (y_true_valid - y_pred_valid) * quantile,
        (y_pred_valid - y_true_valid) * (1 - quantile),
    )

    return np.mean(loss)


compute_pinball_loss.__doc__ = r"""
Computes the Pinball Loss for a single quantile forecast.

The Pinball Loss is a metric used to evaluate the accuracy of a
specific quantile forecast. It is the foundational component of
the Continuous Ranked Probability Score (CRPS). A lower score is
better.

Parameters
----------
y_true : np.ndarray
    1D array of the true observed values.
y_pred_quantile : np.ndarray
    1D array of the predicted values for a single quantile.
quantile : float
    The quantile level (must be between 0 and 1) for which the
    predictions were made.

Returns
-------
float
    The average Pinball Loss over all observations.

See Also
--------
compute_crps : A score calculated by averaging the pinball loss.
plot_crps_comparison : A visualization of the CRPS.

Notes
-----
The Pinball Loss, :math:`\mathcal{L}_{\tau}`, is a proper scoring
rule for evaluating a single quantile forecast :math:`q` at level
:math:`\tau` against an observation :math:`y`. It asymmetrically
penalizes errors, giving a weight of :math:`\tau` to
under-predictions and :math:`(1 - \tau)` to over-predictions.

.. math::

   \mathcal{L}_{\tau}(q, y) =
   \begin{cases}
     (y - q) \tau & \text{if } y \ge q \\
     (q - y) (1 - \tau) & \text{if } y < q
   \end{cases}

This function calculates the average of this loss over all
provided observations.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import compute_pinball_loss
>>>
>>> y_true = np.array([10, 10, 5])
>>> y_pred_q90 = np.array([8, 12, 5]) # Under-predict, over-predict, exact
>>> quantile = 0.9
>>>
>>> # Loss for y=10, q=8: (10-8) * 0.9 = 1.8
>>> # Loss for y=10, q=12: (12-10) * (1-0.9) = 0.2
>>> # Loss for y=5, q=5: (5-5) * 0.9 = 0.0
>>> # Average = (1.8 + 0.2 + 0.0) / 3 = 0.667
>>>
>>> loss = compute_pinball_loss(y_true, y_pred_q90, quantile)
>>> print(f"Average Pinball Loss for Q90: {loss:.3f}")

.. code-block:: text
   :caption: Expected Output

   Average Pinball Loss for Q90: 0.667

References
----------
.. footbibliography::
"""


def compute_winkler_score(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    # Validate and handle NaNs
    y_true, y_pred_lower = validate_yy(y_true, y_pred_lower)
    _, y_pred_upper = validate_yy(y_true, y_pred_upper)

    valid_mask = (
        ~np.isnan(y_true) & ~np.isnan(y_pred_lower) & ~np.isnan(y_pred_upper)
    )

    y_true_valid = y_true[valid_mask]
    lower_valid = y_pred_lower[valid_mask]
    upper_valid = y_pred_upper[valid_mask]

    if len(y_true_valid) == 0:
        return np.nan

    # Calculate interval width (sharpness)
    interval_width = upper_valid - lower_valid

    # Calculate penalties for observations outside the interval
    penalty_lower = (2 / alpha) * (lower_valid - y_true_valid)
    penalty_upper = (2 / alpha) * (y_true_valid - upper_valid)

    # The score is the width plus any applicable penalty
    scores = (
        interval_width
        + np.where(y_true_valid < lower_valid, penalty_lower, 0)
        + np.where(y_true_valid > upper_valid, penalty_upper, 0)
    )

    return np.mean(scores)


compute_winkler_score.__doc__ = r"""
Computes the Winkler score for a given prediction interval.

The Winkler score is a proper scoring rule that evaluates a
prediction interval by combining its width (sharpness) with a
penalty for observations that fall outside the interval. A lower
score indicates a better forecast.

Parameters
----------
y_true : np.ndarray
    1D array of the true observed values.
y_pred_lower : np.ndarray
    1D array of the lower bound of the prediction interval.
y_pred_upper : np.ndarray
    1D array of the upper bound of the prediction interval.
alpha : float, default=0.1
    The significance level for the prediction interval. For
    example, alpha=0.1 corresponds to a (1-0.1)*100 = 90%
    prediction interval.

Returns
-------
float
    The average Winkler score over all observations.

See Also
--------
compute_coverage_score : A metric that only assesses coverage.
compute_interval_width : A metric that only assesses sharpness.

Notes
-----
The Winkler score :footcite:p:`Gneiting2007b` is designed to
evaluate both the **sharpness** and **calibration** of a
prediction interval simultaneously. The score for a single
observation :math:`y` and a :math:`(1-\alpha)` prediction
interval :math:`[l, u]` is defined as:

.. math::

   S_{\alpha}(l, u, y) = (u - l) +
   \begin{cases}
     \frac{2}{\alpha}(l - y) & \text{if } y < l \\
     0 & \text{if } l \le y \le u \\
     \frac{2}{\alpha}(y - u) & \text{if } y > u
   \end{cases}

The first term, :math:`(u - l)`, is the interval width, which
rewards sharpness (narrower intervals). The second term is a
penalty that is applied only if the observation falls outside
the interval. The penalty increases as the observation gets
further from the violated bound. This function returns the
average of this score over all observations.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import compute_winkler_score
>>>
>>> y_true = np.array([1, 5, 12])
>>> y_lower = np.array([2, 4, 8])
>>> y_upper = np.array([8, 6, 10])
>>>
>>> # For a 90% interval (alpha=0.1)
>>> # Obs 1 (y=1): outside. Width=6. Penalty=(2/0.1)*(2-1)=20. Score=26.
>>> # Obs 2 (y=5): inside. Width=2. Penalty=0. Score=2.
>>> # Obs 3 (y=12): outside. Width=2. Penalty=(2/0.1)*(12-10)=40. Score=42.
>>> # Average = (26 + 2 + 42) / 3 = 23.33
>>>
>>> score = compute_winkler_score(
...     y_true, y_lower, y_upper, alpha=0.1
... )
>>> print(f"Average Winkler Score: {score:.2f}")

.. code-block:: text
   :caption: Expected Output

   Average Winkler Score: 23.33

References
----------
.. footbibliography::
"""


@overload
def minmax_scaler(
    X: FrameLike,
    y: None = ...,
    feature_range: tuple[float, float] = ...,
    eps: float = ...,
) -> np.ndarray: ...


@overload
def minmax_scaler(
    X: FrameLike,
    y: FrameLike,
    feature_range: tuple[float, float] = ...,
    eps: float = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


def minmax_scaler(
    X: FrameLike,
    y: FrameLike | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-8,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    def _to_array(obj: FrameLike) -> np.ndarray:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_numpy()
        return np.asarray(obj)

    X_arr = _to_array(X)
    X_shape = X_arr.shape
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    # validate range
    feature_range = validate_length_range(  # type: ignore
        feature_range, param_name="Feature range"
    )
    min_val, max_val = feature_range
    if min_val >= max_val:
        raise ValueError("feature_range must be (min, max) with min < max.")

    # min-max per column
    X_min = X_arr.min(axis=0, keepdims=True)
    X_max = X_arr.max(axis=0, keepdims=True)

    num = X_arr - X_min
    denom = (X_max - X_min) + eps
    X_scaled = min_val + (max_val - min_val) * (num / denom)

    # restore 1D if original was 1D/column vector
    if (
        (len(X_shape) == 1)
        or (X_arr.ndim == 1)
        or (X_arr.ndim > 1 and X_shape[1] == 1)
    ):
        X_scaled = X_scaled.ravel()

    if y is not None:
        y_arr = _to_array(y).astype(float)
        y_min = np.min(y_arr)
        y_max = np.max(y_arr)
        y_num = y_arr - y_min
        y_denom = (y_max - y_min) + eps
        y_scaled = min_val + (max_val - min_val) * (y_num / y_denom)
        return X_scaled, y_scaled

    return X_scaled


minmax_scaler.__doc__ = r"""
Scale features to a specified range using a Min-Max approach.

This function transforms features by scaling each feature to a
given range, typically [0, 1]. This method is robust to
features with zero variance by adding a small epsilon to the
denominator to prevent division-by-zero errors.

Parameters
----------
X : {numpy.ndarray, pandas.DataFrame, pandas.Series}
    The input data to scale. Can be a 1D array or a 2D matrix
    of features.
y : {numpy.ndarray, pandas.DataFrame, pandas.Series}, optional
    Optional target values to scale using the same approach.
    If provided, it is scaled independently of ``X``.
feature_range : tuple of (float, float), default=(0.0, 1.0)
    The desired range of the transformed data.
eps : float, default=1e-8
    A small constant added to the denominator to ensure
    numerical stability when a feature has zero variance.

Returns
-------
X_scaled : numpy.ndarray
    The transformed version of ``X``, with each feature scaled
    to the specified ``feature_range``.
y_scaled : numpy.ndarray, optional
    The scaled version of ``y``, returned only if ``y`` is
    provided.

See Also
--------
sklearn.preprocessing.MinMaxScaler : The scikit-learn equivalent.

Notes
-----
The Min-Max scaling is a common preprocessing step for many
machine learning algorithms that are sensitive to the magnitude
of features.

For each feature (column) in the input data :math:`\mathbf{X}`,
the transformation is calculated as:

.. math::

   X_{\text{scaled}} = \text{min}_{\text{range}} +
   (\text{max}_{\text{range}} - \text{min}_{\text{range}})
   \cdot \frac{\mathbf{X} - \min(\mathbf{X})}
   {(\max(\mathbf{X}) - \min(\mathbf{X})) + \varepsilon}

where :math:`\text{min}_{\text{range}}` and
:math:`\text{max}_{\text{range}}` are the bounds of the
``feature_range``, and :math:`\varepsilon` is a small epsilon
to prevent division by zero.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils.mathext import minmax_scaler
>>>
>>> # Scale a 2D array
>>> X = np.array([[1, 10], [2, 20], [3, 30]])
>>> X_scaled = minmax_scaler(X)
>>> print(X_scaled)
[[0.  0. ]
 [0.5 0.5]
 [1.  1. ]]

>>> # Scale to a different range
>>> X_scaled_custom = minmax_scaler(X, feature_range=(-1, 1))
>>> print(X_scaled_custom)
[[-1. -1.]
 [ 0.  0.]
 [ 1.  1.]]
"""
