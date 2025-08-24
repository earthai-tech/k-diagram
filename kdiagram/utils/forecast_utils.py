
from typing import List, Optional, Literal, Union, Dict
import numpy as np
import pandas as pd


from ..decorators import isdf 
from .validator import exist_features, validate_yy
from .utils.handlers import columns_manager

__all__= [ 
    "calculate_probabilistic_scores", "pivot_forecasts_long", 
    "compute_interval_width", 
    "bin_by_feature", "compute_forecast_errors" 
    ]

def calculate_probabilistic_scores(
    y_true: np.ndarray,
    y_preds_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> pd.DataFrame:
    """
    Calculates probabilistic scores for each observation.

    Computes the Probability Integral Transform (PIT), sharpness
    (interval width), and Continuous Ranked Probability Score (CRPS)
    for each forecast-observation pair.

    Parameters
    ----------
    y_true : np.ndarray
        1D array of observed (true) values.
    y_preds_quantiles : np.ndarray
        2D array of quantile forecasts. Each row corresponds to an
        observation in y_true.
    quantiles : np.ndarray
        1D array of the quantile levels corresponding to the columns
        of y_preds_quantiles.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'pit_value', 'sharpness', and 'crps'.
    """
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
        (y_preds_quantiles - y_true[:, np.newaxis]) * (1 - quantiles)
    )
    crps = np.mean(pinball_loss, axis=1)

    return pd.DataFrame({
        'pit_value': pit_values,
        'sharpness': sharpness,
        'crps': crps
    })

@isdf
def pivot_forecasts_long(
    df: pd.DataFrame,
    qlow_cols: List[str],
    q50_cols: List[str],
    qup_cols: List[str],
    horizon_labels: Optional[List[str]] = None,
    id_vars: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Reshapes multi-horizon forecast data from wide to long format.

    Transforms a DataFrame with separate columns for each horizon's
    quantiles (e.g., 'q10_2023', 'q50_2023', 'q10_2024', 'q50_2024')
    into a long-format DataFrame with columns like 'horizon', 'q_low',
    'q_median', and 'q_high'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame in wide format.
    qlow_cols : list of str
        List of column names for the lower quantile, one for each horizon.
    q50_cols : list of str
        List of column names for the median quantile.
    qup_cols : list of str
        List of column names for the upper quantile.
    horizon_labels : list of str, optional
        Labels for each forecast horizon. If not provided, generic
        labels like 'H1', 'H2' will be generated.
    id_vars : str or list of str, optional
        Identifier columns to keep in the long-format DataFrame
        (e.g., location ID, sample ID).

    Returns
    -------
    pd.DataFrame
        The reshaped DataFrame in long format.
    """
    if not (len(qlow_cols) == len(q50_cols) == len(qup_cols)):
        raise ValueError("Quantile column lists must have the same length.")
    
    if not horizon_labels:
        horizon_labels = [f"H{i+1}" for i in range(len(qlow_cols))]
    
    if len(horizon_labels) != len(qlow_cols):
        raise ValueError("Length of horizon_labels must match"
                         " the number of quantile columns.")

    id_vars = columns_manager(id_vars) or []
    
    # Create temporary mapping dataframes for melting
    df_long_list = []
    for i, label in enumerate(horizon_labels):
        temp_df = df[id_vars + [qlow_cols[i], q50_cols[i], qup_cols[i]]].copy()
        temp_df['horizon'] = label
        temp_df.rename(columns={
            qlow_cols[i]: 'q_low',
            q50_cols[i]: 'q_median',
            qup_cols[i]: 'q_high'
        }, inplace=True)
        df_long_list.append(temp_df)
        
    return pd.concat(df_long_list, ignore_index=True)

@isdf
def compute_interval_width(
    df: pd.DataFrame,
    *quantile_pairs: List[Union[str, float]],
    prefix: str = 'width_',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Computes the width of one or more prediction intervals.

    For each pair of column names provided, this function calculates
    the difference (upper - lower) and adds it as a new column to the
    DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the quantile forecast columns.
    *quantile_pairs : list of (str or float)
        One or more lists/tuples, each containing two elements: the
        column name for the lower quantile and the column name for the
        upper quantile.
    prefix : str, default='width_'
        The prefix for the new interval width column names. The new
        name will be f"{prefix}{upper_col_name}".
    inplace : bool, default=False
        If True, modifies the original DataFrame. If False (default),
        returns a new DataFrame with the added columns.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new interval width columns.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'q10': [1, 2], 'q90': [10, 12],
    ...     'q05': [0, 1], 'q95': [11, 13]
    ... })
    >>> widths_df = compute_interval_width(
    ...     df, ['q10', 'q90'], ['q05', 'q95']
    ... )
    >>> print(widths_df)
       q10  q90  q05  q95  width_q90  width_q95
    0    1   10    0   11          9         11
    1    2   12    1   13         10         12
    """
    if not quantile_pairs:
        raise ValueError("At least one pair of quantile columns must be provided.")

    output_df = df if inplace else df.copy()

    for pair in quantile_pairs:
        if len(pair) != 2:
            raise ValueError(f"Each quantile pair must contain exactly two columns, but got {pair}.")
        
        lower_col, upper_col = pair
        exist_features(df, features=[lower_col, upper_col])
        
        width = output_df[upper_col] - output_df[lower_col]
        new_col_name = f"{prefix}{upper_col}"
        output_df[new_col_name] = width
        
    return output_df

@isdf
def bin_by_feature(
    df: pd.DataFrame,
    bin_on_col: str,
    target_cols: Union[str, List[str]],
    n_bins: int = 10,
    agg_funcs: Union[str, List[str], Dict] = 'mean',
) -> pd.DataFrame:
    """
    Bins data by a feature and computes aggregate statistics.

    This function groups the DataFrame into bins based on the values in
    `bin_on_col` and then calculates aggregate statistics (like mean,
    std, etc.) for the `target_cols` within each bin.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    bin_on_col : str
        The name of the column whose values will be used for binning.
    target_cols : str or list of str
        The name(s) of the column(s) for which to compute statistics.
    n_bins : int, default=10
        The number of bins to create.
    agg_funcs : str, list of str, or dict, default='mean'
        The aggregation function(s) to apply. Can be any function
        accepted by pandas' .agg() method (e.g., 'mean', 'std',
        ['mean', 'std'], {'col_A': 'sum'}).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregate statistics for each bin.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'forecast_value': [10, 12, 20, 22, 30, 32],
    ...     'error': [-1, 1.5, -2, 2.5, -3, 3.5]
    ... })
    >>> binned_stats = bin_by_feature(
    ...     df,
    ...     bin_on_col='forecast_value',
    ...     target_cols='error',
    ...     n_bins=3,
    ...     agg_funcs=['mean', 'std']
    ... )
    >>> print(binned_stats)
       forecast_value_bin  mean       std
    0         (9.978, 17.4]  0.25  1.767767
    1         (17.4, 24.8]  0.25  3.181981
    2         (24.8, 32.0]  0.25  4.60
    """
    target_cols = columns_manager(target_cols)
    required_cols = [bin_on_col] + target_cols
    exist_features(df, features=required_cols)

    # Create bins using pandas.cut
    bin_labels = f"{bin_on_col}_bin"
    df_binned = df.copy()
    df_binned[bin_labels] = pd.cut(df_binned[bin_on_col], bins=n_bins)

    # Group by the new bins and aggregate
    stats = df_binned.groupby(
        bin_labels, observed=False)[target_cols].agg(agg_funcs)
    
    return stats.reset_index()

@isdf
def compute_forecast_errors(
    df: pd.DataFrame,
    actual_col: str,
    *pred_cols: str,
    error_type: Literal['raw', 'absolute', 'squared', 'percentage'] = 'raw',
    prefix: str = 'error_',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Computes forecast errors for one or more models.

    This utility takes a DataFrame with actual and predicted values
    and adds new columns containing the calculated errors, making it
    easy to prepare data for error analysis plots.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    actual_col : str
        The name of the column containing the true observed values.
    *pred_cols : str
        One or more column names containing the predicted values from
        different models.
    error_type : {'raw', 'absolute', 'squared', 'percentage'}, default='raw'
        The type of error to calculate:
        - 'raw': actual - predicted
        - 'absolute': |actual - predicted|
        - 'squared': (actual - predicted)^2
        - 'percentage': 100 * (actual - predicted) / actual
    prefix : str, default='error_'
        The prefix to add to the new error column names. For example,
        a prediction column 'Model_A' will become 'error_Model_A'.
    inplace : bool, default=False
        If True, modifies the original DataFrame. If False (default),
        returns a new DataFrame with the added error columns.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new error columns.
    """
    if not pred_cols:
        raise ValueError("At least one prediction column must be provided.")
    
    required_cols = [actual_col] + list(pred_cols)
    exist_features(df, features=required_cols)

    output_df = df if inplace else df.copy()
    
    actual_vals = output_df[actual_col]

    for pred_col in pred_cols:
        pred_vals = output_df[pred_col]
        new_col_name = f"{prefix}{pred_col}"
        
        if error_type == 'raw':
            errors = actual_vals - pred_vals
        elif error_type == 'absolute':
            errors = (actual_vals - pred_vals).abs()
        elif error_type == 'squared':
            errors = (actual_vals - pred_vals) ** 2
        elif error_type == 'percentage':
            # Avoid division by zero
            errors = 100 * (actual_vals - pred_vals) / actual_vals.replace(0, np.nan)
        else:
            raise ValueError(f"Unknown error_type: '{error_type}'")
            
        output_df[new_col_name] = errors
        
    return output_df

