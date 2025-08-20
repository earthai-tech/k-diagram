# License: Apache 2.0 Licence
# Author: L. Kouadio <etanoyau@gmail.com>

"""
K-Diagram: Polar Diagnostics for Forecast Uncertainty
=======================================================
`k-diagram` is a Python package designed to provide
specialized diagnostic polar plots, called "k-diagrams,
for comprehensive model evaluation and forecast analysis.
"""
import importlib
import logging
import warnings

# Configure basic logging and suppress certain third-party library warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True


# Dynamic import function
def _lazy_import(module_name, alias=None):
    """Lazily import a module to reduce initial package load time."""

    def _lazy_loader():
        return importlib.import_module(module_name)

    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader


try:
    from ._scm_version import version as __version__
except Exception:
    # Fallback version 
    __version__ = "1.2.0"

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
    warnings.warn(
        "Some dependencies are missing. K-Diagram may not function correctly:\n"
        + "\n".join(_missing_dependencies),
        ImportWarning,
        stacklevel=2,
    )

# Re-export config helpers
from .config import configure_warnings, warnings_config  # noqa: F401, E402
# from . import datasets # noqa: E402
from .plot import (  # noqa: E402
    plot_actual_vs_predicted,
    plot_anomaly_magnitude,
    plot_coverage,
    plot_coverage_diagnostic,
    plot_error_bands,
    plot_error_ellipses,
    plot_error_violins,
    plot_feature_fingerprint,
    plot_horizon_metrics,
    plot_interval_consistency,
    plot_interval_width,
    plot_model_comparison,
    plot_model_drift,
    plot_polar_heatmap,
    plot_polar_quiver,
    plot_radial_density_ring,
    plot_relationship,
    plot_reliability_diagram,
    plot_taylor_diagram,
    plot_taylor_diagram_in,
    plot_temporal_uncertainty,
    plot_uncertainty_drift,
    plot_velocity,
    taylor_diagram,
)

__all__ = [
    "__version__",
    "configure_warnings",
    "warnings_config",
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
    "plot_model_comparison",
    "plot_radial_density_ring",
    "plot_reliability_diagram",
    "plot_horizon_metrics",
    "plot_horizon_metrics",
    "plot_polar_heatmap",
    "plot_polar_quiver",
    "plot_error_bands",
    "plot_error_ellipses",
    "plot_error_violins",
]
