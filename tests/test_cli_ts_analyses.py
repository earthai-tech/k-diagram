from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


@pytest.fixture
def demo_csv_ts(tmp_path: Path) -> Path:
    """Time-series demo CSV for CLI tests."""
    n = 120
    rng = np.random.default_rng(42)

    time = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = np.linspace(0.0, 8.0, n)
    season = 4.5 * np.sin(np.arange(n) * 2 * np.pi / 30.0)
    noise = rng.normal(0.0, 1.2, n)

    actual = 50.0 + trend + season + noise
    m1 = actual + rng.normal(0.0, 1.0, n)
    m2 = actual + rng.normal(0.0, 2.0, n)

    q10 = m1 - 2.5
    q90 = m1 + 2.5

    df = pd.DataFrame(
        {
            "time": time,
            "actual": actual,
            "m1": m1,
            "m2": m2,
            "q10": q10,
            "q90": q90,
        }
    )
    path = tmp_path / "ts.csv"
    df.to_csv(path, index=False)
    return path


# -------------------------- plot-time-series ------------------------------
def test_plot_time_series_cli(demo_csv_ts: Path, tmp_path: Path) -> None:
    out = tmp_path / "ts_plot.png"

    # Variant 1: x_col + two preds + bands + names + cmap
    v1 = [
        "plot-time-series",
        str(demo_csv_ts),
        "--x-col",
        "time",
        "--actual-col",
        "actual",
        "--pred-cols",
        "m1,m2",
        "--names",
        "Model-1",
        "Model-2",
        "--q-lower-col",
        "q10",
        "--q-upper-col",
        "q90",
        "--cmap",
        "plasma",
        "--title",
        "Forecast vs Actuals",
        "--savefig",
        str(out),
    ]

    # Variant 2: no x_col (uses index), single pred via --pred
    v2 = [
        "plot-time-series",
        str(demo_csv_ts),
        "--actual-col",
        "actual",
        "--pred",
        "m1",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v1, v2])
    _expect_file(out)


# ------------------------------- plot-qq -----------------------------------
def test_plot_qq_cli(demo_csv_ts: Path, tmp_path: Path) -> None:
    out = tmp_path / "qq_plot.png"

    # Basic Q–Q of errors: actual vs a chosen prediction
    v = [
        "plot-qq",
        str(demo_csv_ts),
        "--actual-col",
        "actual",
        "--pred-col",
        "m1",
        "--title",
        "Q-Q of Forecast Errors",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v])
    _expect_file(out)
