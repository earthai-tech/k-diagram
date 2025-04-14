# -*- coding: utf-8 -*-
#   License: Apache 2.0 
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np
import pandas as pd
from typing import ( 
    Optional, 
    Tuple, 
    Union, 
)
from .validator import validate_length_range 

__all__=["minmax_scaler"]

def minmax_scaler(
    X: Union[np.ndarray, pd.DataFrame, pd.Series],
    y: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-8
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    feature_range = validate_length_range (
        feature_range, param_name="Feature range")
    min_val, max_val = feature_range
    if min_val >= max_val:
        raise ValueError("feature_range must be (min, max) with min < max.")

    # compute min & max
    X_min = X_arr.min(axis=0, keepdims=True)
    X_max = X_arr.max(axis=0, keepdims=True)

    # scaling
    num = X_arr - X_min
    denom = (X_max - X_min) + eps
    X_scaled = min_val + (max_val - min_val)*(num/denom)

    # reshape back if 1D
    if (len(X_shape)==1) or (X_arr.ndim == 1) or (
            X_arr.ndim > 1 and X_shape[1] == 1):
        X_scaled = X_scaled.ravel()

    # if y is provided
    if y is not None:
        y_arr = _to_array(y).astype(float)
        y_min = y_arr.min()
        y_max = y_arr.max()
        y_num = y_arr - y_min
        y_denom = (y_max - y_min) + eps
        y_scaled = (min_val
                    + (max_val - min_val)
                    * (y_num / y_denom))
        return X_scaled, y_scaled
    return X_scaled
