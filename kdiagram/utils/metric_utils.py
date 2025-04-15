# -*- coding: utf-8 -*-
# File: kdiagram/utils/metric_utils.py
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0
# -------------------------------------------------------------------
# Provides utilities for retrieving metric scoring functions.
# Adapted  from concepts in scikit-learn and gofast <https://github.com/earthai-tech>
# -------------------------------------------------------------------

"""
Metric Utilities (:mod:`kdiagram.utils.metric_utils`)
=======================================================

Provides helper functions related to calculating and retrieving
evaluation metrics, primarily adapting scikit-learn metrics to a
simple ``(y_true, y_pred) -> score`` interface.
"""
from __future__ import annotations

import functools
import warnings
from typing import Callable, Any, Dict

# Attempt to import metrics, warn if sklearn
#  is missing (should be dependency)
try:
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        mean_absolute_percentage_error,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn(
        "scikit-learn not found. Metric availability will be limited."
    )
    _SKLEARN_AVAILABLE = False
    # Define dummy functions if sklearn is missing to avoid NameErrors later
    # Users will get the ValueError from get_scorer anyway if they try to use them.
    def r2_score(*args, **kwargs): raise ImportError("scikit-learn missing")
    def mean_absolute_error(*args, **kwargs): raise ImportError("scikit-learn missing")
    def mean_squared_error(*args, **kwargs): raise ImportError("scikit-learn missing")
    def mean_absolute_percentage_error(*args, **kwargs): raise ImportError("scikit-learn missing")
    def accuracy_score(*args, **kwargs): raise ImportError("scikit-learn missing")
    def precision_score(*args, **kwargs): raise ImportError("scikit-learn missing")
    def recall_score(*args, **kwargs): raise ImportError("scikit-learn missing")
    def f1_score(*args, **kwargs): raise ImportError("scikit-learn missing")


__all__ = ['get_scorer']

# --- Define Wrapper Functions for Consistency & Default Args ---

# Wrapper for RMSE
@functools.wraps(mean_squared_error)
def _rmse_score(y_true, y_pred, **kwargs):
    """Calculates Root Mean Squared Error."""
    # squared=False returns RMSE
    return mean_squared_error(y_true, y_pred, squared=False, **kwargs)

# Wrapper for MSE (explicitly squared=True)
@functools.wraps(mean_squared_error)
def _mse_score(y_true, y_pred, **kwargs):
    """Calculates Mean Squared Error."""
    return mean_squared_error(y_true, y_pred, squared=True, **kwargs)

# Wrappers for classification metrics with default average='weighted'
# and zero_division handling (common requirement for sklearn >= 1.1)
@functools.wraps(precision_score)
def _precision_weighted(y_true, y_pred, **kwargs):
    """Calculates Precision Score with default weighted average."""
    kwargs.setdefault('average', 'weighted')
    kwargs.setdefault('zero_division', 0)
    return precision_score(y_true, y_pred, **kwargs)

@functools.wraps(recall_score)
def _recall_weighted(y_true, y_pred, **kwargs):
    """Calculates Recall Score with default weighted average."""
    kwargs.setdefault('average', 'weighted')
    kwargs.setdefault('zero_division', 0)
    return recall_score(y_true, y_pred, **kwargs)

@functools.wraps(f1_score)
def _f1_weighted(y_true, y_pred, **kwargs):
    """Calculates F1 Score with default weighted average."""
    kwargs.setdefault('average', 'weighted')
    kwargs.setdefault('zero_division', 0)
    return f1_score(y_true, y_pred, **kwargs)

# --- Scorer Dictionary ---
# Maps lowercase string names to metric functions (y_true, y_pred) -> score
# Uses wrappers to ensure consistent signature and defaults
_SCORERS: Dict[str, Callable[..., float]] = {
    # Regression
    "r2": r2_score,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "mse": _mse_score,
    "mean_squared_error": _mse_score,
    "rmse": _rmse_score,
    "root_mean_squared_error": _rmse_score,
    "mape": mean_absolute_percentage_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    # Classification
    "accuracy": accuracy_score,
    "accuracy_score": accuracy_score,
    "precision": _precision_weighted, # Default weighted average
    "precision_weighted": _precision_weighted,
    "recall": _recall_weighted, # Default weighted average
    "recall_weighted": _recall_weighted,
    "f1": _f1_weighted, # Default weighted average
    "f1_weighted": _f1_weighted,
    # Add other specific averages if needed, e.g.:
    "precision_macro": functools.partial(precision_score, average='macro', zero_division=0),
    "recall_micro": functools.partial(recall_score, average='micro', zero_division=0),
    "f1_binary": functools.partial(f1_score, average='binary', zero_division=0), # Requires pos_label for binary
}

def get_scorer(scoring: str) -> Callable[[Any, Any], float]:
    # Docstring is omitted here as requested
    if not _SKLEARN_AVAILABLE:
         raise ImportError(
             "scikit-learn is required for most metrics. "
             "Please install it to use get_scorer."
             )
    if not isinstance(scoring, str):
        raise TypeError(f"Expected string scoring name, got {type(scoring)}")

    scorer = _SCORERS.get(scoring.lower()) # Case-insensitive lookup
    if scorer is None:
        # If not found, maybe sklearn has it directly?
        try:
            # This attempts to get sklearn's scorer object, which might
            # have a different signature, adapt if necessary or just error out.
            from sklearn.metrics import get_scorer as sklearn_get_scorer
            scorer_obj = sklearn_get_scorer(scoring)
            # Need to wrap scorer_obj to match (y_true, y_pred) signature?
            scorer = lambda y_true, y_pred: scorer_obj._score_func(y_true, y_pred, **scorer_obj._kwargs)
            # For simplicity here, let's just raise the ValueError.
            # raise ValueError(f"Metric '{scoring}' not found in internal registry.")

        except (ImportError, ValueError): # Catch sklearn errors too
             raise ValueError(
                f"Unknown scoring metric '{scoring}'. Check spelling. "
                f"Available metrics include: {sorted(_SCORERS.keys())}"
            ) from None
             
    return scorer