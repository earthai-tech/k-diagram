import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.metrics import (
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    # mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from kdiagram.compat.sklearn import mean_squared_error, root_mean_squared_error
from kdiagram.utils.metric_utils import available_scorers, get_scorer


def test_available_scorers_contains_core_aliases():
    names = set(available_scorers())
    # A few representative aliases we expect
    for alias in ["r2", "mse", "rmse", "mae", "precision", "recall", "f1"]:
        assert alias in names


def test_get_scorer_type_error():
    with pytest.raises(TypeError):
        get_scorer(123)  # not a string


def test_unknown_metric_raises_value_error():
    with pytest.raises(ValueError) as exc:
        get_scorer("definitely_not_a_metric")  # nonsense
    # Helpful message should list known aliases
    assert "Known aliases:" in str(exc.value)


def test_regression_rmse_and_mse():
    rng = np.random.RandomState(0)
    y_true = rng.randn(100)
    y_pred = y_true + rng.randn(100) * 0.1

    rmse = get_scorer("rmse")
    mse_alias = get_scorer("mse")

    # Compare against sklearn implementations
    exp_rmse = root_mean_squared_error(y_true, y_pred)
    exp_mse = mean_squared_error(y_true, y_pred, squared=True)

    assert rmse(y_true, y_pred) == pytest.approx(exp_rmse)
    assert mse_alias(y_true, y_pred) == pytest.approx(exp_mse)


def test_regression_mae_and_r2():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.9, 2.1, 2.9, 4.2])

    mae = get_scorer("mae")
    r2 = get_scorer("r2")

    assert mae(y_true, y_pred) == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert r2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


def test_classification_weighted_defaults_precision_recall_f1():
    # Imbalanced prediction that misses a class completely to test zero_division handling
    y_true = np.array([0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0])  # predicts only class 0

    prec = get_scorer(
        "precision"
    )  # should default to average="weighted", zero_division=0
    rec = get_scorer("recall")
    f1w = get_scorer("f1")

    exp_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    exp_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    exp_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    assert prec(y_true, y_pred) == pytest.approx(exp_prec)
    assert rec(y_true, y_pred) == pytest.approx(exp_rec)
    assert f1w(y_true, y_pred) == pytest.approx(exp_f1)


def test_f1_binary_convenience():
    # Clean binary case
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    f1b = get_scorer("f1_binary")  # average="binary", zero_division=0
    exp_f1b = f1_score(y_true, y_pred, average="binary", zero_division=0)

    assert f1b(y_true, y_pred) == pytest.approx(exp_f1b)


def test_fallback_to_sklearn_metric_by_name():
    # Not in our alias registry; should fall back to sklearn.metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.8, 2.2, 3.1, 3.9])

    expl_var = get_scorer("explained_variance_score")
    exp_val = explained_variance_score(y_true, y_pred)

    assert expl_var(y_true, y_pred) == pytest.approx(exp_val)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
