# -*- coding: utf-8 -*-
# License: Apache 2.0 Licence 
# Author: L. Kouadio <etanoyau@gmail.com>

"""
K-Diagram: Rethinking Forecasting Uncertainty via Polar-Based Visualization
============================================================================
`k-diagram` is a Python package designed to provide specialized diagnostic polar plots,
called "k-diagrams," for comprehensive model evaluation and forecast analysis.
"""
import logging
import warnings
import importlib

# Configure basic logging and suppress certain third-party library warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

# Dynamic import function
def _lazy_import(module_name, alias=None):
    """Lazily import a module to reduce initial package load time."""
    def _lazy_loader():
        return importlib.import_module(module_name)
    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader

# Define the version
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.3"

# Dependency check
_required_dependencies = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("sklearn", "scikit-learn"),
]

_missing_dependencies = []
for package, import_name in _required_dependencies:
    try:
        if import_name:
            _lazy_import(import_name, package)
        else:
            _lazy_import(package)
    except ImportError as e:
        _missing_dependencies.append(f"{package}: {str(e)}")

if _missing_dependencies:
    warnings.warn("Some dependencies are missing. K-Diagram may not function correctly:\n" +
                  "\n".join(_missing_dependencies), ImportWarning)

# Suppress FutureWarnings or SyntaxWarning if desired, but allow users
# to re-enable them
# Define the warning categories and their corresponding actions
_WARNING_CATEGORIES = {
    "FutureWarning": FutureWarning,
    "SyntaxWarning": SyntaxWarning
}

# Default actions for each warning category
_WARNINGS_STATE = {
    "SyntaxWarning": "ignore"
}

def suppress_warnings(suppress: bool = True):
    """
    Suppress or re-enable FutureWarnings and SyntaxWarnings.

    Function allows users to suppress specific warnings globally within
    the package. By default, it suppresses both `FutureWarning` and 
    `SyntaxWarning`. Users can re-enable these warnings by setting 
    `suppress=False`.

    Parameters
    ----------
    suppress : bool, default=True
        - If `True`, suppresses `FutureWarning` and `SyntaxWarning` by setting 
          their filter to the action specified in `_WARNINGS_STATE`.
        - If `False`, re-enables the warnings by resetting their filter to 
          the default behavior.
    """
    for warning_name, action in _WARNINGS_STATE.items():
        category = _WARNING_CATEGORIES.get(warning_name, Warning)
        if suppress:
            # Suppress the warning by applying the specified action
            warnings.filterwarnings(action, category=category)
        else:
            # Re-enable the warning by resetting to default behavior
            warnings.filterwarnings("default", category=category)

# Suppress warnings by default when the package is initialized
suppress_warnings()
# from . import datasets 
from .plot import (
    plot_actual_vs_predicted,
    plot_anomaly_magnitude,
    plot_coverage_diagnostic,
    plot_interval_consistency,
    plot_interval_width,
    plot_model_drift,
    plot_temporal_uncertainty,
    plot_uncertainty_drift,
    plot_velocity,
    plot_coverage,
    plot_taylor_diagram,
    plot_taylor_diagram_in,
    taylor_diagram,
    plot_feature_fingerprint,
    plot_relationship,
    plot_model_comparison
)

__all__ = [
    "__version__",
    "plot_actual_vs_predicted",
    "plot_anomaly_magnitude",
    "plot_coverage_diagnostic",
    "plot_interval_consistency",
    "plot_interval_width",
    "plot_model_drift",
    "plot_temporal_uncertainty",
    "plot_uncertainty_drift",
    "plot_velocity",
    "plot_coverage",
    "plot_taylor_diagram",
    "plot_taylor_diagram_in",
    "taylor_diagram",
    "plot_feature_fingerprint",
    "plot_relationship",
    "plot_model_comparison"
]
