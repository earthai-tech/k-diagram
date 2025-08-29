from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kdiagram.cli._utils import _expect_file, _try_parse_and_run


@pytest.fixture
def demo_csv_interaction(tmp_path: Path) -> Path:
    """
    CSV for polar heatmap interaction:
      - theta_col: hour   (0..24)
      - r_col:     cloud  (0..1)
      - color_col: output (>=0)
    """
    rng = np.random.default_rng(123)
    n = 1200

    hour = rng.uniform(0.0, 24.0, size=n)
    cloud = rng.uniform(0.0, 1.0, size=n)

    # Output depends on hour (daylight) and cloud factor, plus noise
    daylight = np.sin(hour * np.pi / 24.0) ** 2
    cloud_factor = 1.0 - np.sqrt(cloud)
    output = 100.0 * daylight * cloud_factor + rng.normal(0.0, 3.0, size=n)
    output[(hour < 6) | (hour > 18)] = 0.0  # night

    df = pd.DataFrame({"hour": hour, "cloud": cloud, "output": output})
    path = tmp_path / "interaction.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def demo_csv_fingerprint_layers(tmp_path: Path) -> Path:
    """
    CSV for radar fingerprint (rows are layers).
      - cols: f1..f6
      - labels_col: layer (optional)
    """
    rng = np.random.default_rng(7)
    n_layers = 3
    n_feats = 6
    imp = rng.random((n_layers, n_feats))
    df = pd.DataFrame(imp, columns=[f"f{i+1}" for i in range(n_feats)])
    df.insert(0, "layer", [f"M{i+1}" for i in range(n_layers)])
    path = tmp_path / "fingerprint_layers.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def demo_csv_fingerprint_features(tmp_path: Path) -> Path:
    """
    CSV for radar fingerprint with --transpose (rows are features).
      - cols: L1..L3 (layer columns)
      - labels_col: feature (names for features)
    """
    rng = np.random.default_rng(17)
    n_feats = 5
    n_layers = 3
    imp = rng.random((n_feats, n_layers))
    df = pd.DataFrame(imp, columns=[f"L{i+1}" for i in range(n_layers)])
    df.insert(0, "feature", [f"Feat{i+1}" for i in range(n_feats)])
    path = tmp_path / "fingerprint_features.csv"
    df.to_csv(path, index=False)
    return path


# ------------------- plot-feature-interaction ----------------------
@pytest.mark.parametrize("use_period", [True, False])
@pytest.mark.parametrize("stat", ["mean", "median"])
def test_plot_feature_interaction_cli(
    demo_csv_interaction: Path,
    tmp_path: Path,
    use_period: bool,
    stat: str,
) -> None:
    out = tmp_path / f"interaction_{int(use_period)}_{stat}.png"

    v1 = [
        "plot-feature-interaction",
        str(demo_csv_interaction),
        "--theta-col",
        "hour",
        "--r-col",
        "cloud",
        "--color-col",
        "output",
        "--statistic",
        stat,
        "--theta-bins",
        "24",
        "--r-bins",
        "8",
        "--cmap",
        "inferno",
        "--savefig",
        str(out),
    ]
    if use_period:
        # insert --theta-period after args but before savefig
        v1[10:10] = ["--theta-period", "24"]

    _try_parse_and_run([v1])
    _expect_file(out)


# ----------------- plot-feature-fingerprint -----------------------
def test_plot_feature_fingerprint_cli_layers(
    demo_csv_fingerprint_layers: Path,
    tmp_path: Path,
) -> None:
    """
    Rows are layers (default). Provide labels via --labels-col.
    """
    out = tmp_path / "fingerprint_layers.png"

    v1 = [
        "plot-feature-fingerprint",
        str(demo_csv_fingerprint_layers),
        "--cols",
        "f1,f2,f3,f4,f5,f6",
        "--labels-col",
        "layer",
        "--title",
        "Model Importance Fingerprints",
        "--cmap",
        "tab10",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v1])
    _expect_file(out)


def test_plot_feature_fingerprint_cli_labels_explicit(
    demo_csv_fingerprint_layers: Path,
    tmp_path: Path,
) -> None:
    """
    Rows are layers (default). Provide explicit --labels.
    """
    out = tmp_path / "fingerprint_labels.png"

    v2 = [
        "plot-feature-fingerprint",
        str(demo_csv_fingerprint_layers),
        "--cols",
        "f1,f2,f3,f4,f5,f6",
        "--labels",
        "A",
        "B",
        "C",
        "--features",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "--normalize",
        "--fill",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v2])
    _expect_file(out)


def test_plot_feature_fingerprint_cli_transpose(
    demo_csv_fingerprint_features: Path,
    tmp_path: Path,
) -> None:
    """
    Rows are features -> use --transpose. Provide labels via cols,
    features via --labels-col.
    """
    out = tmp_path / "fingerprint_transpose.png"

    v3 = [
        "plot-feature-fingerprint",
        str(demo_csv_fingerprint_features),
        "--cols",
        "L1,L2,L3",
        "--labels-col",
        "feature",
        "--transpose",
        "--cmap",
        "Set3",
        "--title",
        "Transposed Fingerprint",
        "--savefig",
        str(out),
    ]
    _try_parse_and_run([v3])
    _expect_file(out)
