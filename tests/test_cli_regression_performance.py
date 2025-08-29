# tests/test_cli_regression_performance.py

from __future__ import annotations

from pathlib import Path

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


def test_cli_plot_regression_performance_data_mode(
    demo_csv_context: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "reg_perf_data.png"

    v = [
        "plot-regression-performance",
        str(demo_csv_context),
        "--y-true",
        "actual",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "A",
        "B",
        "--metrics",
        "r2",
        "neg_mean_absolute_error",
        "--metric-label",
        "r2:R²",
        "neg_mean_absolute_error:MAE",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v])
    _expect_file(out)


def test_cli_plot_regression_performance_values_mode(
    tmp_path: Path,
) -> None:
    out = tmp_path / "reg_perf_values.png"

    v = [
        "plot-regression-performance",
        "--metric-values",
        "r2:0.82,0.74",
        "neg_mean_absolute_error:-3.2,-3.6",
        "neg_root_mean_squared_error:-4.1,-4.8",
        "--names",
        "A",
        "B",
        "--metric-label",
        "r2:R²",
        "neg_mean_absolute_error:MAE",
        "neg_root_mean_squared_error:RMSE",
        "--savefig",
        str(out),
    ]

    _try_parse_and_run([v])
    _expect_file(out)
