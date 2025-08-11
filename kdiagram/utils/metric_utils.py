# File: kdiagram/utils/metric_utils.py
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

from __future__ import annotations

from typing import Any, Callable

from ..compat.sklearn import (
    mean_squared_error as compat_mse,
)
from ..compat.sklearn import (
    root_mean_squared_error as compat_rmse,
)

__all__ = ["get_scorer", "available_scorers"]

# Cache for the internal scorer registry so we only build it once.
_SCORERS_CACHE: dict[str, Callable[..., float]] | None = None


def _load_sklearn_metrics():
    """Lazy import sklearn.metrics; raise a clear error if missing."""
    try:
        import sklearn.metrics as sm  # type: ignore

        return sm
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required for metric utilities. "
            "Install it via `pip install scikit-learn`."
        ) from exc


def _build_registry() -> dict[str, Callable[..., float]]:
    r"""
    Build the internal mapping of common metric aliases to callables
    with signature (y_true, y_pred, **kwargs) -> score.
    """
    sm = _load_sklearn_metrics()

    # --- Wrappers to normalize behavior/aliases ---
    def _rmse(y_true, y_pred, **kw):
        # Use the compatibility function for RMSE
        return compat_rmse(y_true, y_pred, **kw)

    def _mse(y_true, y_pred, **kw):
        # Use the compatibility function for MSE
        return compat_mse(y_true, y_pred, squared=True, **kw)

    def _precision_weighted(y_true, y_pred, **kw):
        kw.setdefault("average", "weighted")
        kw.setdefault("zero_division", 0)
        return sm.precision_score(y_true, y_pred, **kw)

    def _recall_weighted(y_true, y_pred, **kw):
        kw.setdefault("average", "weighted")
        kw.setdefault("zero_division", 0)
        return sm.recall_score(y_true, y_pred, **kw)

    def _f1_weighted(y_true, y_pred, **kw):
        kw.setdefault("average", "weighted")
        kw.setdefault("zero_division", 0)
        return sm.f1_score(y_true, y_pred, **kw)

    # --- Core registry (lowercase keys) ---
    registry: dict[str, Callable[..., float]] = {
        # Regression
        "r2": sm.r2_score,
        "r2_score": sm.r2_score,
        "mae": sm.mean_absolute_error,
        "mean_absolute_error": sm.mean_absolute_error,
        "mse": _mse,
        "mean_squared_error": _mse,
        "rmse": _rmse,
        "root_mean_squared_error": _rmse,
        "mape": sm.mean_absolute_percentage_error,
        "mean_absolute_percentage_error": sm.mean_absolute_percentage_error,
        # Classification
        "accuracy": sm.accuracy_score,
        "accuracy_score": sm.accuracy_score,
        "precision": _precision_weighted,
        "precision_weighted": _precision_weighted,
        "recall": _recall_weighted,
        "recall_weighted": _recall_weighted,
        "f1": _f1_weighted,
        "f1_weighted": _f1_weighted,
        # Useful fixed-average shortcuts
        "precision_macro": lambda yt, yp, **k: sm.precision_score(
            yt, yp, average="macro", zero_division=0, **k
        ),
        "recall_micro": lambda yt, yp, **k: sm.recall_score(
            yt, yp, average="micro", zero_division=0, **k
        ),
        "f1_binary": lambda yt, yp, **k: sm.f1_score(
            yt, yp, average="binary", zero_division=0, **k
        ),
    }
    return registry


def _get_registry() -> dict[str, Callable[..., float]]:
    global _SCORERS_CACHE
    if _SCORERS_CACHE is None:
        _SCORERS_CACHE = _build_registry()
    return _SCORERS_CACHE


def get_scorer(scoring: str) -> Callable[[Any, Any], float]:
    r"""
    Return a metric function matching `scoring` with signature:
        (y_true, y_pred, **kwargs) -> score

    Resolution order (case-insensitive):
      1) Internal registry of common aliases (e.g., 'rmse', 'mse', 'f1').
      2) Any callable in `sklearn.metrics` with the exact name provided
         (e.g., 'explained_variance_score', 'median_absolute_error').

    Parameters
    ----------
    scoring : str
        Name or alias of the metric.

    Returns
    -------
    Callable
        A function that accepts (y_true, y_pred, **kwargs) and returns a float.

    Raises
    ------
    TypeError
        If `scoring` is not a string.
    ImportError
        If scikit-learn is not installed.
    ValueError
        If no metric can be resolved for the given name.
    """
    if not isinstance(scoring, str):
        raise TypeError(f"Expected a string metric name, got {type(scoring)}")

    key = scoring.lower()
    reg = _get_registry()
    fn = reg.get(key)
    if fn is not None:
        return fn

    # Fallback: try to grab a function directly from sklearn.metrics by name
    sm = _load_sklearn_metrics()
    attr = getattr(sm, scoring, None)
    if callable(attr):
        return attr  # already (y_true, y_pred, **kwargs) style in sklearn

    # Final failure: guide the user
    known = sorted(reg.keys())
    raise ValueError(
        f"Unknown scoring metric '{scoring}'.\n"
        f"Known aliases: {known}\n"
        f"Tip: you can also use any function name from sklearn.metrics, "
        f"e.g., 'explained_variance_score'."
    )


def available_scorers() -> list[str]:
    r"""
    List the metric aliases available in the internal registry.
    (This does not enumerate every function in sklearn.metrics.)

    Returns
    -------
    List[str]
        Sorted list of available alias names.
    """
    return sorted(_get_registry().keys())
