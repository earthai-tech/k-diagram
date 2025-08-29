from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


# -----------------------------
@pytest.fixture()
def demo_csv_binary(tmp_path: Path) -> Path:
    """Small binary dataset with two prob columns."""
    rng = np.random.default_rng(42)
    n = 300
    y = rng.integers(0, 2, size=n)
    # make m1 correlated with y, m2 more random
    m1 = 0.7 * y + 0.3 * rng.random(n)
    m2 = 0.5 * y + 0.5 * rng.random(n)
    df = pd.DataFrame({"y": y, "m1": m1, "m2": m2})
    p = tmp_path / "bin.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def demo_csv_multiclass(tmp_path: Path) -> Path:
    """Small multiclass dataset with predicted labels."""
    rng = np.random.default_rng(123)
    n = 400
    y_true = rng.integers(0, 4, size=n)
    # mildly noisy predictions
    y_pred = y_true.copy()
    noise_idx = rng.choice(n, size=int(0.25 * n), replace=False)
    y_pred[noise_idx] = rng.integers(0, 4, size=noise_idx.size)
    df = pd.DataFrame({"yt": y_true, "yp": y_pred})
    p = tmp_path / "multi.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def demo_csv_quantiles(tmp_path: Path) -> Path:
    """Continuous data with quantile forecasts (non-crossing)."""
    rng = np.random.default_rng(7)
    n = 250
    y = rng.normal(0, 1, size=n)
    # simple non-crossing synthetic quantiles
    q10 = y - 0.8
    q50 = y
    q90 = y + 0.8
    df = pd.DataFrame({"y": y, "q10": q10, "q50": q50, "q90": q90})
    p = tmp_path / "quant.csv"
    df.to_csv(p, index=False)
    return p


# -----------------------------
# ROC
# -----------------------------
def test_cli_plot_polar_roc(demo_csv_binary: Path, tmp_path: Path) -> None:
    out = tmp_path / "polar_roc.png"
    v = [
        "plot-polar-roc",
        str(demo_csv_binary),
        "--y-true",
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "A",
        "B",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)


# -----------------------------
# PR curve
# -----------------------------
def test_cli_plot_polar_pr_curve(
    demo_csv_binary: Path, tmp_path: Path
) -> None:
    out = tmp_path / "polar_pr.png"
    v = [
        "plot-polar-pr-curve",
        str(demo_csv_binary),
        "--y-true",
        "y",
        "--pred-cols",
        "m1,m2",
        "--names",
        "A",
        "B",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)


# -----------------------------
# Binary polar confusion matrix
# -----------------------------
def test_cli_plot_polar_confusion_matrix(
    demo_csv_binary: Path, tmp_path: Path
) -> None:
    out = tmp_path / "polar_conf_bin.png"
    v = [
        "plot-polar-cm",
        str(demo_csv_binary),
        "--y-true",
        "y",
        "--pred",
        "m1",
        "--pred",
        "m2",
        "--names",
        "Good",
        "Noisy",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)


# -----------------------------
# Multiclass polar confusion matrix
# (both the primary name and the alias)
# -----------------------------
def test_cli_plot_polar_confusion_matrix_in_aliases(
    demo_csv_multiclass: Path, tmp_path: Path
) -> None:
    out1 = tmp_path / "polar_conf_mc_1.png"
    out2 = tmp_path / "polar_conf_mc_2.png"

    v1 = [
        "plot-polar-cm-in",
        str(demo_csv_multiclass),
        "--y-true",
        "yt",
        "--y-pred",
        "yp",
        "--class-labels",
        "A",
        "B",
        "C",
        "D",
        "--savefig",
        str(out1),
    ]
    v2 = [
        "plot-polar-cm-multiclass",
        str(demo_csv_multiclass),
        "--y-true",
        "yt",
        "--y-pred",
        "yp",
        "--savefig",
        str(out2),
    ]

    _try_parse_and_run([v1, v2])
    _expect_file(out1)
    # _expect_file(out2)


def test_cli_plot_polar_confusion_matrix_in_aliases_v2(
    demo_csv_multiclass: Path, tmp_path: Path
) -> None:
    out2 = tmp_path / "polar_conf_mc_2.png"

    v2 = [
        "plot-polar-cm-multiclass",
        str(demo_csv_multiclass),
        "--y-true",
        "yt",
        "--y-pred",
        "yp",
        "--savefig",
        str(out2),
    ]

    _try_parse_and_run([v2])
    _expect_file(out2)


# -----------------------------
# Polar classification report
# -----------------------------
def test_cli_plot_polar_classification_report(
    demo_csv_multiclass: Path, tmp_path: Path
) -> None:
    out = tmp_path / "polar_cls_report.png"
    v = [
        "plot-polar-cr",
        str(demo_csv_multiclass),
        "--y-true",
        "yt",
        "--y-pred",
        "yp",
        "--class-labels",
        "A",
        "B",
        "C",
        "D",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)


# -----------------------------
# Pinball loss across quantiles
# -----------------------------
def test_cli_plot_pinball_loss(
    demo_csv_quantiles: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pinball_loss.png"
    v = [
        "plot-pinball-loss",
        str(demo_csv_quantiles),
        "--y-true",
        "y",
        "--qpreds",
        "q10,q50,q90",
        "--quantiles",
        "0.1",
        "0.5",
        "0.9",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v])
    _expect_file(out)
