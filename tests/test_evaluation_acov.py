import matplotlib.pyplot as plt
import numpy as np
import pytest

from kdiagram.plot.evaluation import (
    plot_pinball_loss,
    plot_polar_classification_report,
    plot_polar_confusion_matrix,
    plot_polar_confusion_matrix_in,
    plot_polar_pr_curve,
    plot_polar_roc,
    plot_regression_performance,
)


@pytest.mark.parametrize(
    "acov, expected_span",
    [
        ("full", 0.5 * np.pi),  # 90 degrees for all issue
        ("half", 0.5 * np.pi),
        ("quarter", 0.5 * np.pi),
        ("eighth", 0.5 * np.pi),
    ],
)
def test_acov_span(acov, expected_span):
    # Test the acov functionality for different span values

    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])

    # Test plot_polar_roc with the given acov value
    # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    if acov != "quarter":
        with pytest.warns(UserWarning):
            ax = plot_polar_roc(y_true, y_pred, acov=acov)
    else:
        ax = plot_polar_roc(
            y_true,
            y_pred,
            acov=acov,
        )

    # Check if the angular span is correct for the given acov
    assert ax.get_thetamax() - ax.get_thetamin() == pytest.approx(
        np.degrees(expected_span), rel=1e-2
    )


@pytest.mark.parametrize(
    "acov, expected_span",
    [
        # pr_curve using 90 whatever value of acov
        ("full", 0.5 * np.pi),
        ("half", 0.5 * np.pi),
        ("quarter", 0.5 * np.pi),
        ("eighth", 0.5 * np.pi),
    ],
)
def test_pr_curve(acov, expected_span):
    # Test the acov functionality for different span values

    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])

    # Test plot_polar_roc with the given acov value
    # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    if acov != "quarter":
        with pytest.warns(UserWarning):
            ax = plot_polar_pr_curve(y_true, y_pred, acov=acov)
    else:
        ax = plot_polar_pr_curve(y_true, y_pred, acov=acov)

    # Check if the angular span is correct for the given acov
    assert ax.get_thetamax() - ax.get_thetamin() == pytest.approx(
        np.degrees(expected_span), rel=1e-2
    )


@pytest.mark.parametrize(
    "acov, expected_warning",
    [
        ("invalid_acov", "currently renders best as a quarter circle"),
        (None, "acov='quarter_circle'"),
    ],
)
def test_invalid_acov(acov, expected_warning):
    # Test for invalid or None acov values to check warning functionality

    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])

    with pytest.warns(UserWarning, match=expected_warning):
        plot_polar_roc(y_true, y_pred, acov=acov)


@pytest.mark.parametrize(
    "acov",
    ["full", "half", "quarter", "eighth"],
)
def test_plot_polar_functions_with_acov(acov):
    # Check that different functions properly handle acov parameter
    y_true_bin = np.array([1, 0, 0, 1, 0])
    y_pred_bin = np.array([0, 0, 0, 1, 0])

    y_true = np.array([1, 2, 3, 4, 4])
    y_pred = np.array([1.1, 1.9, 2.8, 3.9, 4.1])
    y_pred_class = np.array([1, 1, 2, 3, 4])

    # Test multiple plotting functions with the acov parameter
    if acov != "full":
        with pytest.warns(UserWarning):
            plot_polar_confusion_matrix(y_true_bin, y_pred_bin, acov=acov)
            plot_polar_confusion_matrix_in(y_true, y_pred_class, acov=acov)
            plot_polar_classification_report(y_true, y_pred_class, acov=acov)
            plot_pinball_loss(
                y_true,
                np.random.rand(5, 3),
                quantiles=np.array([0.1, 0.5, 0.9]),
                acov=acov,
            )
            plot_regression_performance(y_true, y_pred, acov=acov)
    else:
        plot_polar_confusion_matrix(y_true_bin, y_pred_bin, acov=acov)
        plot_polar_confusion_matrix_in(y_true, y_pred_class, acov=acov)
        plot_polar_classification_report(y_true, y_pred_class, acov=acov)
        plot_pinball_loss(
            y_true,
            np.random.rand(5, 3),
            quantiles=np.array([0.1, 0.5, 0.9]),
            acov=acov,
        )
        plot_regression_performance(y_true, y_pred, acov=acov)

    assert True


def test_plot_with_acov_title():
    # Check if the title reflects acov correctly
    acov = "quarter"
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_polar_roc(y_true, y_pred, acov=acov, ax=ax)

    # Ensure the title is set correctly (i.e., contains acov)
    assert "Polar ROC Curve" in ax.get_title()
