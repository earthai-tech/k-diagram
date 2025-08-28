# conftest.py
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import kdiagram.utils._deps as deps

# Force a non-GUI backend before importing pyplot
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # Matplotlib might be optional in some envs
    plt = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def clear_requirements_cache():
    deps._REQUIREMENT_CACHE.clear()
    yield
    deps._REQUIREMENT_CACHE.clear()


@pytest.fixture(scope="session", autouse=True)
def _isolate_mpl_config(tmp_path_factory: pytest.TempPathFactory) -> None:
    """
    Isolate Matplotlib config so user-level settings don't affect tests.
    """
    temp_dir = tmp_path_factory.mktemp("mplconfig")
    prev = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(temp_dir)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = prev


@pytest.fixture(autouse=True)
def _prevent_show(monkeypatch: pytest.MonkeyPatch):
    """
    Prevent GUI popups during tests even if code calls plt.show().
    """
    if plt is None:
        yield
        return

    def _noop(*args, **kwargs):
        return

    monkeypatch.setattr(plt, "show", _noop, raising=False)
    yield


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Ensure figures are closed after each test to keep Agg quiet and
    avoid resource leaks across tests.
    """
    yield
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass


def pytest_configure(config: pytest.Config) -> None:
    """
    Warning policy:
      - Treat deprecations originating from our package as errors.
      - Keep third-party warnings visible; avoid blanket ignores here.
        (Narrow, justified ignores belong in pytest.ini if needed.)
    """
    # Our package: fail on deprecations
    warnings.filterwarnings(
        "error",
        category=DeprecationWarning,
        module=r"^kdiagram(\.|$)",
    )
    warnings.filterwarnings(
        "error",
        category=FutureWarning,
        module=r"^kdiagram(\.|$)",
    )
    warnings.filterwarnings(
        "error",
        category=PendingDeprecationWarning,
        module=r"^kdiagram(\.|$)",
    )

    # Example of a narrowly-scoped, opt-in ignore (keep commented
    # unless you have a documented rationale in pytest.ini):
    #
    # warnings.filterwarnings(
    #     "ignore",
    #     message=r".*tight_layout.*",
    #     category=UserWarning,
    #     module=r"^matplotlib\.",
    # )


@pytest.fixture
def demo_csv_context(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    time = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 50 + np.linspace(0, 10, n) + rng.normal(0, 2, n)
    m1 = y + rng.normal(0, 2.5, n)
    m2 = y - 3 + rng.normal(0, 3.0, n)
    df = pd.DataFrame({"time": time, "actual": y, "m1": m1, "m2": m2})
    p = tmp_path / "context.csv"
    df.to_csv(p, index=False)
    return p
