import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Rectangle

from kdiagram.plot.evaluation import (
    plot_pinball_loss,
    plot_polar_classification_report,
    plot_polar_confusion_matrix,
    plot_polar_confusion_matrix_in,
    plot_polar_pr_curve,
    plot_polar_roc,
    plot_regression_performance,
)

# ---------------------------
# Helpers / fixtures
# ---------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def binary_data(rng):
    n = 300
    y_true = (rng.random(n) > 0.4).astype(int)
    y_pred1 = y_true * 0.6 + rng.random(n) * 0.4
    y_pred2 = rng.random(n)
    return y_true, y_pred1, y_pred2


@pytest.fixture
def multiclass_labels(rng):
    n = 400
    y_true = rng.integers(0, 4, size=n)
    # add some structured mistakes
    y_pred = y_true.copy()
    mask = (y_true == 2) & (rng.random(n) < 0.35)
    y_pred[mask] = 3
    return y_true, y_pred


# ---------------------------
# ROC (cartesian)
# ---------------------------


def test_roc_cartesian_axes_and_labels(binary_data):
    y_true, y1, y2 = binary_data
    ax = plot_polar_roc(y_true, y1, y2, names=["A", "B"], kind="cartesian")
    try:
        # Rectilinear, not polar
        assert ax.name != "polar"
        # Labels and limits
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"
        x0, x1 = ax.get_xlim()
        y0, y1_ = ax.get_ylim()
        assert 0.0 <= x0 <= 0.0 and 1.0 <= x1 <= 1.0  # ~ [0,1]
        assert 0.0 <= y0 <= 0.0 and 1.0 <= y1_ <= 1.0
    finally:
        plt.close(ax.figure)


# ---------------------------
# PR (cartesian)
# ---------------------------


def test_pr_cartesian_axes_and_grid(binary_data):
    y_true, y1, y2 = binary_data
    ax = plot_polar_pr_curve(
        y_true, y1, y2, names=["A", "B"], kind="cartesian"
    )
    try:
        assert ax.name != "polar"
        assert ax.get_xlabel() == "Recall"
        assert ax.get_ylabel() == "Precision"
        x0, x1 = ax.get_xlim()
        y0, y1_ = ax.get_ylim()
        assert 0.0 <= x0 <= 0.0 and 1.0 <= x1 <= 1.0
        assert 0.0 <= y0 <= 0.0 and 1.0 <= y1_ <= 1.0
    finally:
        plt.close(ax.figure)


# ---------------------------
# Binary Confusion Matrix (cartesian)
# ---------------------------


def test_confmat_binary_cartesian_barcount(binary_data):
    y_true, y1, y2 = binary_data
    ax = plot_polar_confusion_matrix(
        y_true, y1, y2, names=["A", "B"], kind="cartesian"
    )
    try:
        assert ax.name != "polar"
        # 4 categories, bars split among 2 models
        n_bars = sum(isinstance(p, Rectangle) for p in ax.patches)
        assert n_bars == 4 * 2
        # Default category labels
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert ticklabels == [
            "True Positive",
            "False Positive",
            "True Negative",
            "False Negative",
        ]
    finally:
        plt.close(ax.figure)


def test_confmat_binary_cartesian_rejects_multiclass(multiclass_labels):
    y_true, y_pred = multiclass_labels
    with pytest.raises(ValueError) as ei:
        plot_polar_confusion_matrix(y_true, y_pred, kind="cartesian")
    # message mirrors polar path
    msg = str(ei.value)
    assert "currently supports only binary" in msg
    assert "plot_polar_confusion_multiclass" in msg


# ---------------------------
# Multiclass Confusion Matrix (cartesian)
# ---------------------------


def test_confmat_multiclass_cartesian_grouping(multiclass_labels):
    y_true, y_pred = multiclass_labels
    ax = plot_polar_confusion_matrix_in(
        y_true, y_pred, class_labels=["A", "B", "C", "D"], kind="cartesian"
    )
    try:
        assert ax.name != "polar"
        # n_classes bars per true class -> n_classes^2 bars
        n_classes = len(np.unique(np.concatenate((y_true, y_pred))))
        n_bars = sum(isinstance(p, Rectangle) for p in ax.patches)
        assert n_bars == n_classes * n_classes
        # Tick labels show "True\n<Label>"
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert all(lbl.startswith("True\n") for lbl in ticklabels)
    finally:
        plt.close(ax.figure)


# ---------------------------
# Classification Report (cartesian)
# ---------------------------


def test_classification_report_cartesian_shapes(multiclass_labels):
    y_true, y_pred = multiclass_labels
    ax = plot_polar_classification_report(
        y_true, y_pred, class_labels=["A", "B", "C", "D"], kind="cartesian"
    )
    try:
        assert ax.name != "polar"
        # 3 metrics per class
        n_classes = len(np.unique(np.concatenate((y_true, y_pred))))
        n_bars = sum(isinstance(p, Rectangle) for p in ax.patches)
        assert n_bars == 3 * n_classes
        assert ax.get_ylabel() == "Score"
        # Upper limit around 1.05
        _, ymax = ax.get_ylim()
        assert 1.03 <= ymax <= 1.07
    finally:
        plt.close(ax.figure)


# ---------------------------
# Pinball Loss (cartesian)
# ---------------------------


def test_pinball_loss_cartesian_axes(rng):
    n = 250
    y_true = rng.normal(loc=0.0, scale=1.0, size=n)
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    # simple “forecast”: constant quantiles = empirical quantiles of y_true
    qvals = np.quantile(y_true, qs)
    yq = np.tile(qvals, (n, 1))

    ax = plot_pinball_loss(
        y_true, yq, qs, kind="cartesian", title="Pinball Loss"
    )
    try:
        assert ax.name != "polar"
        assert ax.get_xlabel() == "Quantile Level"
        assert "Pinball Loss" in ax.get_title()
        x0, x1 = ax.get_xlim()
        assert 0.0 <= x0 <= 0.0 and 1.0 <= x1 <= 1.0
    finally:
        plt.close(ax.figure)


# ---------------------------
# Regression Performance (cartesian)
# ---------------------------


def test_regression_performance_cartesian_groupbars():
    names = ["M1", "M2", "M3"]
    metric_values = {
        "r2": [0.5, 0.7, 0.2],
        "neg_mean_absolute_error": [
            -1.2,
            -0.9,
            -1.8,
        ],  # lower is better (negated)
        "neg_root_mean_squared_error": [-1.5, -1.1, -1.9],
    }
    ax = plot_regression_performance(
        metric_values=metric_values,
        names=names,
        kind="cartesian",
        title="Perf",
    )
    try:
        assert ax.name != "polar"
        # bars = n_metrics * n_models
        n_metrics = len(metric_values)
        n_models = len(names)
        n_bars = sum(isinstance(p, Rectangle) for p in ax.patches)
        assert n_bars == n_metrics * n_models
        assert ax.get_ylabel() == "Score"
        # ytick labels should be the normalized scale labels ('Worst', ..., 'Best')
        yticks = [t.get_text() for t in ax.get_yticklabels()]
        assert (
            "Worst" in yticks[0] or "0" in yticks[0]
        )  # allow numeric fallback
    finally:
        plt.close(ax.figure)


# ---------------------------
# Validate kind gate
# ---------------------------


def test_kind_validation_raises(binary_data):
    y_true, y1, _ = binary_data
    with pytest.raises(ValueError):
        plot_polar_roc(y_true, y1, kind="not-a-mode")
