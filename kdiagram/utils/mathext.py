#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 

from typing import (
    overload, Literal, List, Tuple, Union, Optional, 
    Callable
)
from scipy.stats import kstest, uniform
import numpy as np
import pandas as pd

from .handlers import columns_manager 
from .validator import validate_length_range
from .validator import validate_yy, exist_features


__all__ = [
    "minmax_scaler", 
    "compute_coverage_score", 
    "compute_winkler_score", 
    "build_cdf_interpolator", 
    "calculate_calibration_error",
    "compute_pinball_loss", 
    "compute_pit", 
    "compute_crps", 
    "get_forecast_arrays"
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
) -> np.ndarray:
    ...


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
) -> np.ndarray:
    ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: None,
    pred_cols: List[str],
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
) -> np.ndarray:
    ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: Union[str, List[str]],
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
) -> Tuple[np.ndarray, np.ndarray]:
    ...


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
) -> pd.Series:
    ...


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
) -> pd.Series:
    ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: None,
    pred_cols: List[str],
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
) -> pd.DataFrame:
    ...


@overload
def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str,
    pred_cols: Union[str, List[str]],
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
) -> Tuple[pd.Series, Union[pd.Series, pd.DataFrame]]:
    ...

def get_forecast_arrays(
    df: pd.DataFrame,
    actual_col: str | None = None,
    pred_cols: str | List[str] | None = None,
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
    r"""
    Extract true and/or predicted values from a DataFrame.

    Flexible bridge between a DataFrame-centric workflow and
    NumPy-based utilities. Supports dropping or filling NAs,
    numeric coercion, and optional index return.

    Parameters
    ----------
    df : DataFrame
        Source table.
    actual_col : str, optional
        Column holding ground-truth values.
    pred_cols : str or list of str, optional
        Prediction column(s). A string implies a single
        series; a list implies multiple columns.
    drop_na : bool, default True
        Drop rows with missing data as per ``na_policy``.
    na_policy : {"any","all","none"}, default "any"
        NA row filter:
        - "any": drop rows with any NA among selected cols.
        - "all": drop rows where all selected cols are NA.
        - "none": do not drop; may still ``fillna``.
    fillna : scalar, dict or {"ffill","bfill"}, optional
        Fill strategy before NA dropping (if any).
    return_as : {"numpy","pandas"}, default "numpy"
        Output container type.
    squeeze : bool, default True
        For a single prediction column, reduce to 1-D.
    with_index : bool, default False
        If ``True``, return index as the first item.
    sort_index : bool, default False
        Sort frame by index before extraction.
    dtype : object, optional
        Target dtype for NumPy arrays / Series.
    ensure_numeric : bool, default False
        Enforce numeric dtype on selected columns.
    coerce_numeric : bool, default False
        If ``True``, invalid parses become NaN.
    copy : bool, default True
        Work on a copy of the selected subset.

    Returns
    -------
    array/Series/DataFrame or tuple
        - Only ``actual_col`` → y_true
        - Only ``pred_cols``  → y_pred(s)
        - Both               → (y_true, y_pred(s))
        If ``with_index=True``, the index is prepended to
        the return value.

    Notes
    -----
    For "numpy" output and a single prediction column with
    ``squeeze=True``, the predictions are 1-D. Set
    ``squeeze=False`` to keep shape ``(n, 1)``.
    """
    if actual_col is None and pred_cols is None:
        raise ValueError(
            "Provide at least one of 'actual_col' or "
            "'pred_cols'."
        )

    # collect required columns
    cols: List[str] = []
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
        if fillna in ("ffill", "bfill"):
            sub = sub.fillna(method=str(fillna))
        else:
            sub = sub.fillna(fillna)

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

def compute_crps(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """
    Approximates the Continuous Ranked Probability Score (CRPS).

    The CRPS is calculated as the average of the Pinball Loss across
    all provided quantiles. A lower score is better.

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
    """
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )
    
    # Reshape y_true for broadcasting
    y_true_reshaped = y_true[:, np.newaxis]
    
    # Calculate Pinball Loss for all quantiles at once
    pinball_losses = np.where(
        y_true_reshaped >= y_preds_quantiles,
        (y_true_reshaped - y_preds_quantiles) * quantiles,
        (y_preds_quantiles - y_true_reshaped) * (1 - quantiles)
    )
    
    # Average over quantiles for each observation, then over all observations
    return np.mean(np.mean(pinball_losses, axis=1))

def calculate_calibration_error(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """
    Calculates the calibration error of a probabilistic forecast.

    This function quantifies the deviation of a forecast from perfect
    calibration by using the Kolmogorov-Smirnov (KS) statistic on the
    Probability Integral Transform (PIT) values. A lower score indicates
    better calibration.

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
    float
        The Kolmogorov-Smirnov statistic, a value between 0 and 1 where
        0 represents perfect calibration.
    """
    # Validate inputs
    y_true, y_preds_quantiles = validate_yy(
        y_true, y_preds_quantiles, allow_2d_pred=True
    )
    
    # Calculate PIT values
    sort_idx = np.argsort(quantiles)
    sorted_preds = y_preds_quantiles[:, sort_idx]
    pit_values = np.mean(sorted_preds <= y_true[:, np.newaxis], axis=1)
    
    if len(pit_values) < 2:
        return 1.0 # Max penalty for insufficient data to test

    # Compare the empirical distribution of PIT values to a uniform distribution
    ks_statistic, _ = kstest(pit_values, uniform.cdf)
    
    return ks_statistic

def build_cdf_interpolator(
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Builds an interpolator to act as a Cumulative Distribution Function (CDF).

    This function takes a set of quantile forecasts and returns a callable
    function that linearly interpolates between them, effectively creating
    an empirical CDF for each forecast.

    Parameters
    ----------
    y_preds_quantiles : np.ndarray
        2D array of quantile forecasts, with shape (n_samples, n_quantiles).
    quantiles : np.ndarray
        1D array of the quantile levels corresponding to the columns of
        the prediction array.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that takes an array of observed values (`y_true`) and
        returns the corresponding PIT values (the CDF evaluated at each point).
    """
    # Sort quantiles and predictions to ensure correct interpolation
    sort_idx = np.argsort(quantiles)
    sorted_quantiles = quantiles[sort_idx]
    sorted_preds = np.asarray(y_preds_quantiles)[:, sort_idx]

    def _interpolator(y_true: np.ndarray) -> np.ndarray:
        """The returned CDF interpolator function."""
        y_true = np.asarray(y_true)
        pit_values = np.zeros_like(y_true, dtype=float)

        for i in range(len(y_true)):
            # Use np.interp for robust linear interpolation
            pit_values[i] = np.interp(
                y_true[i],
                sorted_preds[i, :],
                sorted_quantiles,
                left=0.0,  # Values below the lowest quantile get p=0
                right=1.0  # Values above the highest quantile get p=1
            )
        return pit_values

    return _interpolator


def compute_coverage_score(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    *,
    method: Literal['within', 'above', 'below'] = 'within',
    return_counts: bool = False,
) -> Union[float, int]:
    """
    Computes the coverage score for a given prediction interval.

    This utility calculates the fraction (or count) of true values
    that fall within, above, or below the specified prediction interval.

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
        If True, returns the raw count of observations matching the
        condition instead of the proportion (a float between 0 and 1).

    Returns
    -------
    float or int
        The coverage score as a proportion or a raw count.
    """
    # Validate and convert inputs
    y_true, y_pred_lower = validate_yy(y_true, y_pred_lower)
    _, y_pred_upper = validate_yy(y_true, y_pred_upper)
    
    # Handle NaNs by creating a mask of valid (non-NaN) entries
    valid_mask = ~np.isnan(y_true) & ~np.isnan(
        y_pred_lower) & ~np.isnan(y_pred_upper)
    
    y_true_valid = y_true[valid_mask]
    lower_valid = y_pred_lower[valid_mask]
    upper_valid = y_pred_upper[valid_mask]
    
    n_valid = len(y_true_valid)
    if n_valid == 0:
        return 0.0 if not return_counts else 0

    if method == 'within':
        count = np.sum((y_true_valid >= lower_valid
                        ) & (y_true_valid <= upper_valid))
    elif method == 'above':
        count = np.sum(y_true_valid > upper_valid)
    elif method == 'below':
        count = np.sum(y_true_valid < lower_valid)
    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose from"
            " 'within', 'above', or 'below'."
        )
        
    if return_counts:
        return int(count)
    
    return float(count / n_valid)

def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    quantile: float,
) -> float:
    """
    Computes the Pinball Loss for a single quantile forecast.

    The Pinball Loss is a metric used to evaluate the accuracy of a
    specific quantile forecast. It is the foundation for the CRPS.
    A lower score is better.

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
    """
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
        (y_pred_valid - y_true_valid) * (1 - quantile)
    )
    
    return np.mean(loss)

def compute_winkler_score(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
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
        The significance level for the prediction interval. For example,
        alpha=0.1 corresponds to an 80% prediction interval.

    Returns
    -------
    float
        The average Winkler score over all observations.
    """
    # Validate and handle NaNs
    y_true, y_pred_lower = validate_yy(y_true, y_pred_lower)
    _, y_pred_upper = validate_yy(y_true, y_pred_upper)
    
    valid_mask = ~np.isnan(y_true) & ~np.isnan(
        y_pred_lower) & ~np.isnan(y_pred_upper)
    
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
    scores = interval_width + np.where(
        y_true_valid < lower_valid, penalty_lower, 0
    ) + np.where(
        y_true_valid > upper_valid, penalty_upper, 0
    )
    
    return np.mean(scores)


def minmax_scaler(
    X: Union[np.ndarray, pd.DataFrame, pd.Series],
    y: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-8,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    r"""
    Scale features (and optionally target) to a specified
    range (default [0, 1]) using a Min-Max approach.
    This method is robust to zero denominators via an
    epsilon offset.

    .. math::
       X_{\text{scaled}} = \text{range}_{\min}
       + (\text{range}_{\max} - \text{range}_{\min})
         \cdot \frac{X - X_{\min}}
         {(X_{\max} - X_{\min}) + \varepsilon}

    Parameters
    ----------
    X : {numpy.ndarray, pandas.DataFrame, pandas.Series}
        Feature matrix or vector. If array-like, shape
        is (n_samples, n_features) or (n_samples, ).
    y : {numpy.ndarray, pandas.DataFrame, pandas.Series}, optional
        Optional target values to scale with the same
        approach. If provided, must be 1D or a single
        column.
    feature_range : (float, float), optional
        Desired range for the scaled values. Default
        is (0.0, 1.0).
    eps : float, optional
        A small offset to avoid division-by-zero when
        ``X_max - X_min = 0``. Default is 1e-8.

    Returns
    -------
    X_scaled : numpy.ndarray
        Transformed version of X within the desired
        range.
    y_scaled : numpy.ndarray, optional
        Scaled version of y, if provided.

    Notes
    -----
    - This scaler is commonly used for neural networks
      and other methods sensitive to the absolute
      magnitude of features.
    - Passing an epsilon helps prevent NaN or inf
      results for constant vectors or features.

    Examples
    --------
    >>> import numpy as np
    >>> from kdiagram.utils.mathext import minmax_scaler
    >>> X = np.array([[1, 2],[3, 4],[5, 6]])
    >>> X_scaled = minmax_scaler(X)
    >>> # X_scaled now lies in [0,1] per feature.
    """

    # Convert inputs to arrays
    def _to_array(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.values
        return np.asarray(obj)

    X_arr = _to_array(X)
    X_shape = X_arr.shape
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    # range min & max
    feature_range = validate_length_range(
        feature_range, param_name="Feature range"
    )
    min_val, max_val = feature_range
    if min_val >= max_val:
        raise ValueError("feature_range must be (min, max) with min < max.")

    # compute min & max
    X_min = X_arr.min(axis=0, keepdims=True)
    X_max = X_arr.max(axis=0, keepdims=True)

    # scaling
    num = X_arr - X_min
    denom = (X_max - X_min) + eps
    X_scaled = min_val + (max_val - min_val) * (num / denom)

    # reshape back if 1D
    if (
        (len(X_shape) == 1)
        or (X_arr.ndim == 1)
        or (X_arr.ndim > 1 and X_shape[1] == 1)
    ):
        X_scaled = X_scaled.ravel()

    # if y is provided
    if y is not None:
        y_arr = _to_array(y).astype(float)
        y_min = y_arr.min()
        y_max = y_arr.max()
        y_num = y_arr - y_min
        y_denom = (y_max - y_min) + eps
        y_scaled = min_val + (max_val - min_val) * (y_num / y_denom)
        return X_scaled, y_scaled
    return X_scaled
