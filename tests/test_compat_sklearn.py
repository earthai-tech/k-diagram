# tests/test_compat_sklearn.py
from __future__ import annotations

import inspect
import sys
import types
from numbers import Integral
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.version import parse
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import NotFittedError

from kdiagram.compat import sklearn as skl_compat


# ------------------------------
# Interval shim (signature-based)
# ------------------------------
def test_interval_inclusive_removed_branch(monkeypatch):
    """
    Simulate older sklearn where Interval.__init__ has no 'inclusive' param.
    """
    real_sig = inspect.signature(skl_compat.sklearn_Interval.__init__)

    def fake_signature(obj):
        if obj is skl_compat.sklearn_Interval.__init__:
            # build a signature that does NOT include 'inclusive'
            return inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "self", inspect.Parameter.POSITIONAL_ONLY
                    ),
                    inspect.Parameter(
                        "types", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    inspect.Parameter(
                        "left", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    inspect.Parameter(
                        "right", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    inspect.Parameter(
                        "closed", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                ]
            )
        return real_sig

    monkeypatch.setattr(skl_compat.inspect, "signature", fake_signature)
    # Should NOT error even if we pass inclusive; shim must drop it.
    obj = skl_compat.Interval(Integral, 0, 1, closed="left", inclusive=True)
    assert isinstance(obj, skl_compat.sklearn_Interval)


def test_interval_inclusive_supported_branch():
    # On modern sklearn, passing inclusive should be accepted
    obj = skl_compat.Interval(Integral, 0, 1, closed="left", inclusive=True)
    assert isinstance(obj, skl_compat.sklearn_Interval)


# ------------------------------
# type_of_target + fallback impl
# ------------------------------
def test__type_of_target_core_paths():
    # continuous (floats)
    assert skl_compat._type_of_target([0.5, 1.2, 3.3]) == "continuous"
    # binary (ints)
    assert skl_compat._type_of_target([0, 1, 0, 1]) == "binary"
    # multiclass
    assert skl_compat._type_of_target([0, 1, 2, 1]) == "multiclass"
    # # multilabel-indicator
    # assert skl_compat._type_of_target(
    #     [[1, 0], [0, 1], [1, 1]]) == "multilabel-indicator"


def test__type_of_target_numeric_validation_error():
    with pytest.raises(ValueError, match="numeric"):
        skl_compat._type_of_target(["a", "b", "c"])


def test_type_of_target_delegates_to_sklearn():
    # Basic smoke test: just ensure it returns a valid sklearn value
    assert skl_compat.type_of_target([0, 1, 0, 1]) in {
        "binary",
        "multiclass",
        "continuous",
        "multilabel-indicator",
        "unknown",
        "continuous-multioutput",
        "multiclass-multioutput",
        "multilabel-sequences",
    }


# ------------------------------
# validate_params wrapper
# ------------------------------
def test_validate_params_with_and_without_flag(monkeypatch):
    params = {"x": [int], "y": [str]}

    # Case 1: Signature contains prefer_skip_nested_validation (real modern sklearn)
    dec = skl_compat.validate_params(
        params, prefer_skip_nested_validation=False
    )

    @dec
    def fn(x, y):
        return f"{x}-{y}"

    assert fn(3, "ok") == "3-ok"
    with pytest.raises(skl_compat.InvalidParameterError):
        fn("bad", "ok")

    # Case 2: Pretend signature lacks the flag
    real_sig = inspect.signature(skl_compat.sklearn_validate_params)

    def fake_sig(_):
        # Drop the flag from the signature
        pars = [
            p
            for p in real_sig.parameters.values()
            if p.name != "prefer_skip_nested_validation"
        ]
        return real_sig.replace(parameters=pars)

    monkeypatch.setattr(skl_compat.inspect, "signature", fake_sig)
    dec2 = skl_compat.validate_params(
        params, prefer_skip_nested_validation=True
    )

    @dec2
    def fn2(x, y):
        return f"{x}:{y}"

    assert fn2(7, "ok") == "7:ok"


# ---------------------------------------------------------
# ColumnTransformer feature-name helpers (two implementations)
# ---------------------------------------------------------
class _TxOut:
    def __init__(self, names):
        self._names = np.array(names)
        self.feature_names_in_ = None

    def get_feature_names_out(self):
        return self._names


class _TxNames:
    def __init__(self, names):
        self._names = names

    def get_feature_names(self):
        return self._names


class _TxNoNames:
    def transform(self, column):
        # pretend it produces 2 columns
        n_rows = len(column) if isinstance(column, (list, tuple)) else 1
        return np.zeros((n_rows, 2))


def test_get_column_transformer_feature_names_all_paths():
    ct = SimpleNamespace(
        _n_features=5,
        transformers_=[
            ("keep1", _TxOut(["a", "b"]), [0, 1]),
            ("keep2", _TxNames(["c1", "c2", "c3"]), 2),
            ("dropme", "drop", [3, 4]),
            ("fallback", _TxNoNames(), [1, 2, 3]),
        ],
    )
    out = skl_compat.get_column_transformer_feature_names(
        ct, input_features=["f0", "f1", "f2", "f3", "f4"]
    )
    assert out == ["a", "b", "c1", "c2", "c3", "fallback__0", "fallback__1"]


def test_get_column_transformer_feature_names2_all_paths():
    ct = SimpleNamespace(
        transformers_=[
            ("t1", _TxOut(["x", "y"]), [0, 1]),
            ("t2", _TxNames(["z"]), 2),
            ("t3", _TxNoNames(), [0, 2]),
        ]
    )
    out = skl_compat.get_column_transformer_feature_names2(
        ct, input_features=["A", "B", "C"]
    )
    assert out == ["x", "y", "z", "t3__0", "t3__1"]


# --------------------------------------
# get_feature_names / get_feature_names_*
# --------------------------------------
def test_get_feature_names_variants_and_error():
    class EstOut:
        def get_feature_names_out(self, *a, **k):
            return ["o1", "o2"]

    class EstNames:
        def get_feature_names(self, *a, **k):
            return ["n1"]

    assert skl_compat.get_feature_names(EstOut()) == ["o1", "o2"]
    assert skl_compat.get_feature_names_out(EstOut()) == ["o1", "o2"]
    assert skl_compat.get_feature_names(EstNames()) == ["n1"]

    class NoNames: ...

    with pytest.raises(AttributeError):
        skl_compat.get_feature_names(NoNames())


# --------------------------
# ColumnTransformer utilities
# --------------------------
def test_get_transformers_from_column_transformer_and_error():
    ct = SimpleNamespace(transformers_=[("n", object(), [0])])
    assert (
        skl_compat.get_transformers_from_column_transformer(ct)
        == ct.transformers_
    )

    with pytest.raises(AttributeError):
        skl_compat.get_transformers_from_column_transformer(object())


# -----------------
# check_is_fitted()
# -----------------
def test_check_is_fitted_success_and_failure():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = LinearRegression().fit(X, y)
    assert (
        skl_compat.check_is_fitted(
            model, attributes=None, msg=None, all_or_any=all
        )
        is None
    )

    with pytest.raises(NotFittedError):
        skl_compat.check_is_fitted(
            LinearRegression(), attributes=None, msg=None, all_or_any=all
        )


# --------------
# fetch_openml()
# --------------
def test_fetch_openml_injects_as_frame_by_default(monkeypatch):
    calls = {}

    def fake_fetch_openml(*args, **kwargs):
        calls["kwargs"] = kwargs.copy()
        return "SENTINEL"

    # Patch the function sklearn.datasets.fetch_openml that the wrapper imports
    import sklearn.datasets as skd

    monkeypatch.setattr(skd, "fetch_openml", fake_fetch_openml, raising=True)

    # Modern path (not LT 0.24): inject as_frame=True if not provided
    monkeypatch.setattr(skl_compat, "SKLEARN_LT_0_24", False)
    out = skl_compat.fetch_openml(data_id=61)
    assert out == "SENTINEL"
    assert calls["kwargs"].get("as_frame") is True

    # Old path simulation: do not add as_frame
    monkeypatch.setattr(skl_compat, "SKLEARN_LT_0_24", True)
    out2 = skl_compat.fetch_openml(name="iris")
    assert out2 == "SENTINEL"
    assert "as_frame" not in (
        calls["kwargs"] or calls["kwargs"]["as_frame"] is True
    )  # previous call retained; not critical


def test_plot_confusion_matrix_not_available(monkeypatch):
    # Create a dummy module 'sklearn.metrics' without the symbol so
    # "from sklearn.metrics import plot_confusion_matrix" raises ImportError.
    dummy = types.ModuleType("sklearn.metrics")
    sys.modules["sklearn.metrics"] = dummy
    with pytest.raises(NotImplementedError):
        skl_compat.plot_confusion_matrix(None, None, None)


# --------------------
# train_test_split shim
# --------------------
def test_train_test_split_shuffle_default_and_override():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)

    # Default insert shuffle=True
    Xtr, Xte, ytr, yte = skl_compat.train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    assert len(Xte) == 2 and len(Xtr) == 8

    # Explicit shuffle=False preserves ordering of the tail split
    Xtr2, Xte2, ytr2, yte2 = skl_compat.train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    # Last 3 elements should be in test set in order
    assert np.all(yte2 == np.array([7, 8, 9]))


# -----------------------------------
# get_transformer_feature_names helper
# -----------------------------------
def test_get_transformer_feature_names_all_paths():
    class TxOut:
        def get_feature_names_out(self, input_features=None):
            # Return some derived names
            return [f"{s}_out" for s in (input_features or ["x"])]

    class TxNames:
        def get_feature_names(self, input_features=None):
            return [f"{s}_name" for s in (input_features or ["x"])]

    class TxNone:
        pass

    assert skl_compat.get_transformer_feature_names(TxOut(), ["a", "b"]) == [
        "a_out",
        "b_out",
    ]
    assert skl_compat.get_transformer_feature_names(TxNames(), ["a"]) == [
        "a_name"
    ]

    with pytest.raises(AttributeError):
        skl_compat.get_transformer_feature_names(TxNone())


# ------------------------------
# get_pipeline_feature_names shim
# ------------------------------
def test_get_pipeline_feature_names_paths():
    class TxOut:
        def get_feature_names_out(self, feats):
            return [f"{f}_o" for f in feats]

    class TxNames:
        def get_feature_names(self, feats):
            return [f"{f}_n" for f in feats]

    class TxOHE:
        # Simulate OneHotEncoder with categories_
        def __init__(self):
            self.categories_ = [np.array(["A", "B"]), np.array(["C"])]

    pipe = SimpleNamespace(
        steps=[("a", TxOut()), ("b", TxNames()), ("c", TxOHE())]
    )
    feats = skl_compat.get_pipeline_feature_names(
        pipe, input_features=["f1", "f2"]
    )
    # After TxOut -> ["f1_o","f2_o"]; after TxNames
    # -> ["f1_o_n","f2_o_n"]; after TxOHE -> categories concatenated
    assert feats == ["A", "B", "C"]


# -------------------------
# MSE / RMSE compatibility
# -------------------------
def test_mean_squared_error_new_path_and_rmse(monkeypatch):
    y_true = np.array([0.0, 2.0, 4.0])
    y_pred = np.array([0.0, 1.0, 7.0])

    # Force "new" path (>= 1.4): squared arg removed
    monkeypatch.setattr(skl_compat, "SKLEARN_VERSION", parse("1.4"))
    mse = skl_compat.mean_squared_error(y_true, y_pred, squared=True)
    rmse = skl_compat.mean_squared_error(y_true, y_pred, squared=False)
    # Compare with sklearn base function directly
    base = skl_compat.sklearn_mse(y_true, y_pred)
    assert mse == pytest.approx(base)
    assert rmse == pytest.approx(np.sqrt(base))

    # root_mean_squared_error helper
    rmse2 = skl_compat.root_mean_squared_error(y_true, y_pred)
    assert rmse2 == pytest.approx(rmse)
