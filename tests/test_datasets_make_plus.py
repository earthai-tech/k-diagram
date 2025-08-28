from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kdiagram.datasets.make import (
    make_classification_data,
    make_regression_data,
)

# default profiles now include "High Variance"


def test_make_regression_data_frame_and_bunch() -> None:
    n = 120
    df = make_regression_data(n_samples=n, as_frame=True)
    # basic shape & required cols
    assert len(df) == n
    assert "y_true" in df.columns

    # default models -> columns are prefixed "pred_"
    exp = {
        "pred_Good_Model",
        "pred_Biased_Model",
        "pred_High_Variance",  # <- updated expectation
    }
    assert exp.issubset(set(df.columns))

    # bunch path parity
    b = make_regression_data(n_samples=n, as_frame=False)
    assert hasattr(b, "frame") and isinstance(b.frame, pd.DataFrame)
    assert b.frame.equals(df)
    # data matches the prediction columns the bunch advertises
    pred_cols = list(b.prediction_columns)
    assert set(pred_cols) == exp
    assert b.data.shape == (n, len(pred_cols))
    # target matches
    assert np.allclose(b.target, df["y_true"].to_numpy())


# partial renaming (rename first k models only)
def test_make_regression_data_partial_rename() -> None:
    # Rename only the first two defaults; the third keeps its default
    df = make_regression_data(
        n_samples=50,
        as_frame=True,
        model_names=["A", "B"],  # only first two are renamed
    )
    exp = {"A", "B", "pred_High_Variance"}
    assert exp.issubset(set(df.columns)), f"columns: {set(df.columns)}"

    # Bunch should advertise the same prediction columns
    b = make_regression_data(
        n_samples=50,
        as_frame=False,
        model_names=["A", "B"],
    )
    assert set(b.prediction_columns) == exp


# custom model profiles + clipping
def test_make_regression_data_custom_profiles_and_clip() -> None:
    profiles = {
        "Goodish": {
            "bias": 0.0,
            "noise_std": 1.0,
            "error_type": "additive",
        },
        "Multi": {
            "bias": 0.1,
            "noise_std": 0.2,
            "error_type": "multiplicative",
        },
    }
    df = make_regression_data(
        n_samples=40,
        model_profiles=profiles,
        clip_negative=True,  # ensure no negatives after clipping
        shuffle=False,  # deterministic ordering in the frame
        seed=42,
        as_frame=True,
    )

    exp = {"pred_Goodish", "pred_Multi"}
    assert exp.issubset(set(df.columns)), f"columns: {set(df.columns)}"

    # clipping should ensure non-negative predictions
    assert float(df[list(exp)].min().min()) >= 0.0


def test_make_regression_data_custom_profiles_repro() -> None:
    prof = {
        "Good": {"bias": 0.0, "noise_std": 3.0, "error_type": "additive"},
        "Biased": {"bias": -5.0, "noise_std": 1.0, "error_type": "additive"},
    }
    df1 = make_regression_data(
        n_samples=64,
        n_features=3,
        model_profiles=prof,
        seed=123,
        as_frame=True,
    )
    df2 = make_regression_data(
        n_samples=64,
        n_features=3,
        model_profiles=prof,
        seed=123,
        as_frame=True,
    )
    pdt.assert_frame_equal(df1, df2)  # reproducible with same seed
    assert {"pred_Good", "pred_Biased"}.issubset(df1.columns)


def test_make_classification_data_binary_probs_and_labels() -> None:
    n = 300
    df = make_classification_data(
        n_samples=n,
        n_features=6,
        n_classes=2,
        n_models=2,
        model_names=["m1", "m2"],
        include_binary_pred_cols=True,
        seed=7,
        as_frame=True,
    )
    # columns present
    assert "y" in df.columns
    assert {"m1", "m2", "pred_m1", "pred_m2"}.issubset(df.columns)
    # probability range
    assert float(df["m1"].min()) >= 0.0
    assert float(df["m1"].max()) <= 1.0
    # labels are {0,1}
    assert set(df["pred_m1"].unique()).issubset({0, 1})
    # reproducibility
    df2 = make_classification_data(
        n_samples=n,
        n_features=6,
        n_classes=2,
        n_models=2,
        model_names=["m1", "m2"],
        include_binary_pred_cols=True,
        seed=7,
        as_frame=True,
    )
    pdt.assert_frame_equal(df, df2)


def test_make_classification_data_multiclass_probs_sum_to_one() -> None:
    n, c, m = 200, 4, 2
    df = make_classification_data(
        n_samples=n,
        n_features=5,
        n_classes=c,
        n_models=m,
        model_names=["m1", "m2"],
        add_compat_cols=True,  # adds yt/yp for CLI convenience
        seed=11,
        as_frame=True,
    )

    # true & alias labels exist
    assert {"y", "yt"}.issubset(df.columns)
    # per-model proba columns exist and sum to 1 across classes
    for name in ["m1", "m2"]:
        cols = [f"proba_{name}_{k}" for k in range(c)]
        for col in cols:
            assert col in df.columns
            assert (df[col].between(0.0, 1.0)).all()
        s = df[cols].sum(axis=1).to_numpy()
        assert np.allclose(s, 1.0, atol=1e-6)
        # predicted labels exist
        assert f"pred_{name}" in df.columns

    # yp alias (to first model) present
    assert "yp" in df.columns


def test_make_classification_data_invalid_weights_raises() -> None:
    with pytest.raises(ValueError):
        _ = make_classification_data(
            n_samples=50,
            n_features=3,
            n_classes=3,
            weights=[0.7, 0.3],  # wrong length
            as_frame=True,
        )
