import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from scipy.stats import norm
from sklearn.datasets import make_classification

from kdiagram.plot.evaluation import (
    plot_pinball_loss,
    plot_polar_classification_report,
    plot_polar_confusion_matrix,
    plot_polar_confusion_matrix_in,
    plot_polar_pr_curve,
    plot_polar_roc,
)

# Use a non-interactive backend for testing
plt.switch_backend("Agg")

# --- Fixtures for generating test data ---


@pytest.fixture(scope="module")
def binary_class_data():
    """Fixture for binary classification plots."""
    np.random.seed(42)
    X, y_true = make_classification(
        n_samples=200,
        n_classes=2,
        weights=[0.8, 0.2],
        flip_y=0.1,
        random_state=42,
    )
    # Simulate probabilities
    y_pred1_proba = y_true * 0.7 + np.random.rand(200) * 0.3
    y_pred2_proba = np.random.rand(200)
    return {
        "y_true": y_true,
        "preds": [y_pred1_proba, y_pred2_proba],
        "names": ["Good Model", "Random Model"],
    }


@pytest.fixture(scope="module")
def multiclass_data():
    """Fixture for multiclass classification plots."""
    np.random.seed(0)
    y_true = np.random.randint(0, 4, 300)
    y_pred = y_true.copy()
    # Introduce some errors
    error_indices = np.random.choice(300, 50, replace=False)
    y_pred[error_indices] = (y_pred[error_indices] + 1) % 4
    return {"y_true": y_true, "y_pred": y_pred}


@pytest.fixture(scope="module")
def probabilistic_data():
    """Fixture for pinball loss plot."""
    np.random.seed(1)
    n_samples = 100
    y_true = np.random.normal(loc=50, scale=10, size=n_samples)
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    scales = np.array([12, 10, 8, 10, 12])
    preds = norm.ppf(quantiles, loc=y_true[:, np.newaxis], scale=scales)
    return {"y_true": y_true, "preds": preds, "quantiles": quantiles}


# --- Tests for plot_polar_roc ---


def test_plot_polar_roc_runs(binary_class_data):
    ax = plot_polar_roc(
        binary_class_data["y_true"], *binary_class_data["preds"]
    )
    assert isinstance(ax, Axes)
    plt.close()


# --- Tests for plot_polar_pr_curve ---


def test_plot_polar_pr_curve_runs(binary_class_data):
    ax = plot_polar_pr_curve(
        binary_class_data["y_true"], *binary_class_data["preds"]
    )
    assert isinstance(ax, Axes)
    plt.close()


# --- Tests for plot_polar_confusion_matrix (binary) ---


def test_plot_polar_confusion_matrix_runs(binary_class_data):
    ax = plot_polar_confusion_matrix(
        binary_class_data["y_true"],
        *binary_class_data["preds"],
        normalize=False,
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_polar_confusion_matrix_raises_on_multiclass(multiclass_data):
    with pytest.raises(NotImplementedError, match="only supports binary"):
        plot_polar_confusion_matrix(
            multiclass_data["y_true"], multiclass_data["y_pred"]
        )


# --- Tests for plot_polar_confusion_multiclass ---


def test_plot_polar_confusion_multiclass_runs(multiclass_data):
    ax = plot_polar_confusion_matrix_in(
        multiclass_data["y_true"], multiclass_data["y_pred"]
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_plot_polar_confusion_multiclass_warns_on_mismatched_labels(
    multiclass_data,
):
    with pytest.warns(UserWarning, match="does not match number of classes"):
        plot_polar_confusion_matrix_in(
            multiclass_data["y_true"],
            multiclass_data["y_pred"],
            class_labels=["A", "B"],  # Mismatched length
        )
    plt.close()


# --- Tests for plot_polar_classification_report ---


def test_plot_polar_classification_report_runs(multiclass_data):
    ax = plot_polar_classification_report(
        multiclass_data["y_true"], multiclass_data["y_pred"]
    )
    assert isinstance(ax, Axes)
    # Check for correct number of bars: n_classes * n_metrics
    assert len(ax.patches) == 4 * 3
    plt.close()


# --- Tests for plot_pinball_loss ---


def test_plot_pinball_loss_runs(probabilistic_data):
    ax = plot_pinball_loss(
        probabilistic_data["y_true"],
        probabilistic_data["preds"],
        probabilistic_data["quantiles"],
    )
    assert isinstance(ax, Axes)
    # Check that it plotted the correct number of points
    assert len(ax.lines[0].get_xdata()) == len(
        probabilistic_data["quantiles"]
    )
    plt.close()


# --- General Error Handling Tests ---


def test_empty_input_raises_error(binary_class_data):
    """Test that functions raise error on empty input via decorator."""
    with pytest.raises(ValueError, match=r"At least one prediction"):
        plot_polar_roc(binary_class_data["y_true"])  # No predictions

    with pytest.raises(ValueError, match=r"At least one prediction"):
        plot_polar_confusion_matrix(binary_class_data["y_true"])


def test_mismatched_names_warns(binary_class_data):
    """Test that a warning is issued for mismatched names."""
    with pytest.warns(UserWarning, match="Number of names does not match"):
        plot_polar_roc(
            binary_class_data["y_true"],
            *binary_class_data["preds"],
            names=["Only one name"],
        )
    plt.close()
