import numpy as np
import pandas as pd
import pytest

from kdiagram.utils.mathext import minmax_scaler


def test_2d_numpy_basic_default_range():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Xs = minmax_scaler(X)
    # Each column should be scaled to [0, 1]
    assert Xs.shape == X.shape
    assert np.isclose(Xs.min(axis=0), [0.0, 0.0]).all()
    assert np.isclose(Xs.max(axis=0), [1.0, 1.0]).all()


def test_2d_numpy_custom_range():
    X = np.array([[0.0, -1.0], [5.0, 0.0], [10.0, 1.0]])
    Xs = minmax_scaler(X, feature_range=(-1.0, 1.0))
    # Check range bounds per feature
    assert np.isclose(Xs.min(axis=0), [-1.0, -1.0]).all()
    assert np.isclose(Xs.max(axis=0), [1.0, 1.0]).all()


def test_1d_numpy_input_returns_1d_and_scales_01():
    x = np.array([5.0, 15.0, 25.0])
    xs = minmax_scaler(x)
    # stays 1D (ravel path); min==0, max==1
    assert xs.ndim == 1 and xs.shape == (3,)
    assert np.isclose(xs.min(), 0.0)
    assert np.isclose(xs.max(), 1.0)


def test_pandas_inputs_dataframe_and_series():
    X_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    y_sr = pd.Series([100, 200, 300])

    Xs, ys = minmax_scaler(X_df, y_sr)
    # Outputs are numpy arrays scaled to [0,1]
    assert isinstance(Xs, np.ndarray) and Xs.shape == (3, 2)
    assert isinstance(ys, np.ndarray) and ys.shape == (3,)

    assert np.isclose(Xs.min(axis=0), [0.0, 0.0]).all()
    assert np.isclose(Xs.max(axis=0), [1.0, 1.0]).all()
    assert np.isclose(ys.min(), 0.0)
    assert np.isclose(ys.max(), 1.0)


def test_constant_input_uses_eps_and_collapses_to_min_of_range():
    # Constant features -> numerator 0 -> should map to min_val of range
    X = np.array([[7.0, 42.0], [7.0, 42.0], [7.0, 42.0]])
    # Use a non-default range to verify exact output
    Xs = minmax_scaler(X, feature_range=(5.0, 10.0))
    assert np.allclose(Xs, 5.0)  # all values equal min of range

    # Constant y also maps to min of range
    y = np.array([9.0, 9.0, 9.0])
    _, ys = minmax_scaler(X[:, 0], y, feature_range=(2.0, 3.0))
    assert np.allclose(ys, 2.0)


def test_column_vector_ravel_behavior():
    # Shape (n,1) should be raveled back to 1D
    X = np.array([[1.0], [2.0], [3.0]])
    Xs = minmax_scaler(X)
    assert Xs.ndim == 1 and Xs.shape == (3,)
    assert np.isclose(Xs.min(), 0.0) and np.isclose(Xs.max(), 1.0)


def test_invalid_feature_range_raises():
    X = np.array([0.0, 1.0, 2.0])
    with pytest.raises(
        ValueError,
        match="The first element in Feature range must be less than the second.",
    ):
        _ = minmax_scaler(X, feature_range=(1.0, 1.0))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
