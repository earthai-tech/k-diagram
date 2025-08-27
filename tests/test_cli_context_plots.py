from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


# ----------------------------- fixtures -----------------------------
@pytest.fixture
def demo_csv_context(tmp_path: Path) -> Path:
    """
    Synthetic data for context plots:

    Columns:
      - actual: ground truth
      - m1, m2: predictions
    """
    n = 180
    rng = np.random.default_rng(123)

    x = np.linspace(0.0, 10.0, n)
    season = 2.0 * np.sin(2 * np.pi * x / 5.0)
    trend = 0.5 * x
    noise = rng.normal(0.0, 1.0, size=n)

    actual = 20 + trend + season + noise
    m1 = actual + rng.normal(0.0, 1.0, size=n)  # good-ish
    m2 = 0.85 * actual + 2.0 + rng.normal(0.0, 2.0, size=n)  # biased

    df = pd.DataFrame({"actual": actual, "m1": m1, "m2": m2})
    path = tmp_path / "context.csv"
    df.to_csv(path, index=False)
    return path


# ------------------- plot-scatter-correlation ----------------------


def test_plot_scatter_correlation_cli(
    demo_csv_context: Path, tmp_path: Path
) -> None:
    out = tmp_path / "scatter_corr.png"

    v1 = [
        "plot-scatter-correlation",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-cols",
        "m1,m2",
        "--names",
        "A",
        "B",
        "--cmap",
        "plasma",
        "--s",
        "35",
        "--alpha",
        "0.6",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-scatter-correlation",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ----------------- plot-error-autocorrelation ----------------------


def test_plot_error_autocorrelation_cli(
    demo_csv_context: Path, tmp_path: Path
) -> None:
    out = tmp_path / "acf_errors.png"

    v1 = [
        "plot-error-autocorrelation",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-col",
        "m1",
        "--lags",
        "60",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-error-autocorrelation",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-col",
        "m2",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ------------------- plot-error-distribution -----------------------


def test_plot_error_distribution_cli(
    demo_csv_context: Path, tmp_path: Path
) -> None:
    out = tmp_path / "err_dist.png"

    v1 = [
        "plot-error-distribution",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-col",
        "m1",
        "--bins",
        "40",
        "--hist-color",
        "#888888",
        "--kde-color",
        "#1f77b4",
        "--alpha",
        "0.8",
        "--figsize",
        "8,6",
        "--savefig",
        str(out),
    ]
    # Second variant with the other model
    v2 = [
        "plot-error-distribution",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-col",
        "m2",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ------------------------ plot-error-pacf --------------------------


@pytest.mark.skipif(
    pytest.importorskip("statsmodels", reason="statsmodels required") is None,
    reason="statsmodels not available",
)
def test_plot_error_pacf_cli(demo_csv_context: Path, tmp_path: Path) -> None:
    out = tmp_path / "pacf_errors.png"

    v1 = [
        "plot-error-pacf",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-col",
        "m1",
        "--lags",
        "30",
        "--savefig",
        str(out),
    ]
    v2 = [
        "plot-error-pacf",
        str(demo_csv_context),
        "--actual-col",
        "actual",
        "--pred-col",
        "m2",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1, v2])
    _expect_file(out)
