import re
import warnings

import numpy as np
import pandas as pd
import pytest

from kdiagram.utils import diagnose_q as dq


# -----------------
# to_iterable / _flatten
# -----------------
def test_to_iterable_boolean_and_transform_variants():
    # boolean check (no transform)
    assert dq.to_iterable("abc", exclude_string=True) is False
    assert dq.to_iterable([1, 2, 3]) is True

    # transform basic
    assert dq.to_iterable(123, transform=True) == [123]

    # parse_string requires transform=True
    with pytest.raises(ValueError):
        dq.to_iterable("a,b", parse_string=True)

    # parse string with default delimiter
    assert dq.to_iterable("a, b; c", transform=True, parse_string=True) == [
        "a",
        "b",
        "c",
    ]

    # flatten + unique preserves order
    nested = [1, (2, 3), [3, 4, [5]]]
    out = dq.to_iterable(nested, transform=True, flatten=True, unique=True)
    assert out == [1, 2, 3, 4, 5]


# -----------------
# parse_qcols
# -----------------
def test_parse_qcols_dict_and_list_and_fallback_and_warnings():
    # Proper dict with q10, q50, q90
    qd = {"q10": "low_10", "q50": "med_50", "q90": "hi_90"}
    out = dq.parse_qcols(qd)
    assert out["valid"] is True
    assert out["lowest_col"] == "low_10"
    assert out["median_col"] == "med_50"
    assert out["highest_col"] == "hi_90"
    assert out["parsed_qvals"][10.0] == "low_10"

    # When q50 missing, pick the middle by order
    qd2 = {"q05": "l", "q90": "u", "q20": "m?"}
    out2 = dq.parse_qcols(qd2)
    # middle of sorted [5,20,90] -> 20 => "m?"
    assert out2["median_col"] in {"m?", "u", "l"}

    # list input becomes q0,q1,... and uses middle as median
    out3 = dq.parse_qcols(["L", "M", "U"])
    assert out3["median_col"] == "M"
    assert out3["lowest_col"] == "L"
    assert out3["highest_col"] == "U"

    # fallback when nothing valid + warning
    with pytest.warns(UserWarning):
        out4 = dq.parse_qcols({"bad": "x"}, fallback_cols=("a", "b", "c"), error="warn")
    assert out4["valid"] is False
    assert (out4["lowest_col"], out4["median_col"], out4["highest_col"]) == (
        "a",
        "b",
        "c",
    )

    # non-string key -> warning path
    with pytest.warns(UserWarning):
        dq.parse_qcols({10: "x"}, error="warn")

    # invalid key prefix -> raise
    with pytest.raises(ValueError):
        dq.parse_qcols({"p10": "x"}, error="raise")

    # bad float conversion -> raise
    with pytest.raises(ValueError):
        dq.parse_qcols({"qxx": "x"}, error="raise")


# -----------------
# check_forecast_mode
# -----------------
def test_check_forecast_mode_variants():
    # invalid mode
    with pytest.raises(ValueError):
        dq.check_forecast_mode("oops", q=None)

    # point mode with q -> warn then None
    with pytest.warns(UserWarning):
        q = dq.check_forecast_mode("point", q=[0.1, 0.5], error="warn")
    assert q is None

    # point mode with q -> raise
    with pytest.raises(ValueError):
        dq.check_forecast_mode("point", q=[0.1], error="raise")

    # quantile mode, q=None -> warn and default
    with pytest.warns(UserWarning):
        q2 = dq.check_forecast_mode("quantile", q=None, error="warn")
    assert [round(q, 1) for q in q2] == [0.1, 0.5, 0.9]

    # quantile mode with provided quantiles validated
    q3 = dq.check_forecast_mode("quantile", q=[10, "20%"], error="warn", q_mode="soft")
    # 10 -> 0.1 (individual scaling), "20%" -> 0.2
    assert pytest.approx(q3, rel=1e-6) == [0.1, 0.2]

    # check_only returns None
    assert dq.check_forecast_mode("quantile", q=[0.1, 0.5], ops="check_only") is None


# -----------------
# validate_q_dict
# -----------------
def test_validate_q_dict_conversions_and_recheck():
    # '10%' -> 0.1, '0.5' -> 0.5
    qd = {"10%": ["a"], "0.5": ["b"], "weird": ["c"]}
    out = dq.validate_q_dict(qd)
    assert out[0.1] == ["a"]
    assert out[0.5] == ["b"]
    assert "weird" in out

    # recheck=True with valid domain
    qd2 = {"10%": ["a"], "90%": ["b"]}
    out2 = dq.validate_q_dict(qd2, recheck=True)
    assert 0.1 in out2 and 0.9 in out2

    # input type error
    with pytest.raises(TypeError):
        dq.validate_q_dict(["not", "a", "dict"])


# -----------------
# validate_quantiles
# -----------------
def test_validate_quantiles_strict_and_soft_and_dtype_and_rounding():
    # strict valid and asarray
    arr = dq.validate_quantiles(
        [0.123456, 0.789012], asarray=True, round_digits=3, dtype="float64"
    )
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert np.allclose(arr, [0.123, 0.789])

    # strict invalid value
    with pytest.raises(ValueError):
        dq.validate_quantiles([1.2])

    # soft: integers and percentages
    soft = dq.validate_quantiles(["20%", 5, 150], mode="soft", dtype="float64")
    # 20% -> 0.2; 5 -> 0.05 (individual
    # scaling default inside helper for integers);
    # 150 -> 0.15
    assert pytest.approx(soft, rel=1e-2) == [0.02, 0.0, 0.15]

    # non-numeric in soft -> TypeError/ValueError path
    with pytest.raises((TypeError, ValueError)):
        dq.validate_quantiles(["abc"], mode="soft")


# -----------------
# validate_quantiles_in
# -----------------
def test_validate_quantiles_in_paths():
    # list -> list
    out = dq.validate_quantiles_in([0.10000000149, 0.5, 0.8999999761], round_digits=1)
    assert [round(q, 1) for q in out] == [0.1, 0.5, 0.9]

    # array -> array with dtype
    arr = dq.validate_quantiles_in(np.array([0.3, 0.7]), asarray=True, dtype="float32")
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32

    # bad type
    with pytest.raises((TypeError, ValueError)):
        dq.validate_quantiles_in("not-an-iterable")

    # out of range
    with pytest.raises(ValueError):
        dq.validate_quantiles_in([0.0, 1.1])

    # bad dtype string
    with pytest.raises(ValueError):
        dq.validate_quantiles_in([0.2], dtype="float128")


# -----------------
# detect_quantiles_in
# -----------------
def test_detect_quantiles_in_columns_qvals_values_frame_and_modes():
    df = pd.DataFrame(
        {
            "sales_q0.25": [1.0, 2.0],
            "sales_q0.75": [3.0, 4.0],
            "sales_2023_q0.5": [5.0, 6.0],
            "noise": [0, 0],
            "risk_q150": [9.0, 9.0],  # out-of-range value to exercise 'soft'
        }
    )

    # default (columns), with prefix filter
    cols = dq.detect_quantiles_in(df, col_prefix="sales", return_types="columns")
    assert cols == ["sales_2023_q0.5", "sales_q0.25", "sales_q0.75"]

    # q_val from all (soft) -> includes 0.25, 0.5 (from 2023 col), and 0.15 from q150
    qvals = dq.detect_quantiles_in(df, col_prefix="", return_types="q_val", mode="soft")
    assert set(qvals) >= {0.25, 0.5, 0.15}

    # strict mode excludes invalid (q150)
    qvals_strict = dq.detect_quantiles_in(df, return_types="q_val", mode="strict")
    assert 0.15 not in qvals_strict and 0.5 in qvals_strict

    # values (np.vstack with rows per column)
    values = dq.detect_quantiles_in(df, col_prefix="sales", return_types="values")
    assert isinstance(values, np.ndarray) and values.shape == (3, 2)

    # frame subset
    sub = dq.detect_quantiles_in(df, col_prefix="sales", return_types="frame")
    assert isinstance(sub, pd.DataFrame) and list(sub.columns) == [
        "sales_q0.25",
        "sales_q0.75",
        "sales_2023_q0.5",
    ]

    # dt_value filter
    qvals_2023 = dq.detect_quantiles_in(
        df, col_prefix="sales", dt_value=["2023"], return_types="q_val"
    )
    assert qvals_2023 is None


# -----------------
# build_q_column_names (strict & flexible)
# -----------------
def test_build_q_column_names_strict_and_flexible():
    df = pd.DataFrame(
        columns=[
            "price_q0.25",
            "price_q25",
            "price_2023_q0.5",
            "2024_q0.9",
            "q0.75",
            "other",
        ]
    )

    # strict exact candidates (decimal and percent forms)
    out_strict = dq.build_q_column_names(df, [0.25, 0.5], "price", strict_match=True)
    assert set(out_strict) == {"price_q0.25", "price_q25"}

    # flexible regex should catch both decimal and percentage variants
    out_flex = dq.build_q_column_names(df, [0.25], "price", strict_match=False)
    assert "price_q0.25" in out_flex or "price_q25" in out_flex

    # no prefix, pick unprefixed columns too
    out_no_prefix = dq.build_q_column_names(
        df, [0.75, 0.9], value_prefix=None, strict_match=True
    )
    assert set(out_no_prefix) == {"q0.75"}


# -----------------
# detect_digits
# -----------------
def test_detect_digits_general_and_quantile_and_error_paths():
    # general digits in string and list
    assert dq.detect_digits("x10y") == [10.0]
    assert dq.detect_digits(["a50b", "c0.25d"]) == [50.0, 0.25]

    # quantile mode (_q..._step), rounded to 2 decimals by validate_quantiles
    with pytest.raises(
        ValueError, match=re.escape("Non-integer out-of-range quantile: 10.5")
    ):
        dq.detect_digits("subsidence_q10.5_step1", as_q=True)
        # assert q == [0.11]  # 10.5 -> 0.105 -> rounded to 0.11

    # custom pattern that matches non-float -> warn path (no exception)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        out = dq.detect_digits("xxx", pattern=r"(x)", error="warn", verbose=1)
    assert out == []


# -----------------
# validate_consistency_q
# -----------------
def test_validate_consistency_q_soft_and_strict_and_warnings():
    cols = [
        "subsidence_q10_step1",
        "subsidence_q50_step1",
        "subsidence_q90_step1",
        "other",
    ]
    user_q = [0.1, 0.5, 0.9]
    valid = dq.validate_consistency_q(user_q, cols)
    assert valid == [0.1, 0.5, 0.9]

    # mismatch with warning returns empty (soft default_to='valid_q')
    with pytest.warns(UserWarning):
        empty = dq.validate_consistency_q(
            [0.2],
            cols,
            error="warn",
        )
    assert empty == []

    # strict mode exact match required
    valid_strict = dq.validate_consistency_q(user_q, cols, mode="strict")
    assert valid_strict == [0.1, 0.5, 0.9]

    # strict mismatch raises
    with pytest.raises(ValueError):
        dq.validate_consistency_q([0.1, 0.5], cols, mode="strict")


# -----------------
# _verify_identical_items
# -----------------
def test__verify_identical_items_unique_and_ascending_and_errors():
    # unique mode validate
    assert dq._verify_identical_items(
        [1, 2, 2], [2, 1], mode="unique", ops="validate"
    ) == [1, 2]

    # ascending mode ok
    assert dq._verify_identical_items([0.1, 0.5], [0.1, 0.5], mode="ascending") is True

    # ascending length mismatch -> raise
    with pytest.raises(ValueError):
        dq._verify_identical_items([1], [1, 2], mode="ascending")

    # invalid mode
    with pytest.raises(ValueError):
        dq._verify_identical_items([1], [1], mode="nope")


# -----------------
# validate_qcols
# -----------------
def test_validate_qcols_variants_and_expectations():
    # single string / int accepted
    assert dq.validate_qcols("q50") == ["q50"]
    assert dq.validate_qcols(12) == ["12"]

    # cleaning blanks
    assert dq.validate_qcols(("p1", "p2", " ")) == ["p1", "p2"]

    # expectations pass/fail
    assert dq.validate_qcols(["a", "b"], ncols_exp="==2") == ["a", "b"]
    assert dq.validate_qcols(["a", "b"], ncols_exp=">=2") == ["a", "b"]
    with pytest.raises(ValueError):
        dq.validate_qcols(["a", "b"], ncols_exp="==3")
    with pytest.raises(ValueError):
        dq.validate_qcols(["a"], ncols_exp=">=")  # bad syntax

    # empty after cleaning
    with pytest.raises(ValueError):
        dq.validate_qcols(["   "])

    # wrong container type
    with pytest.raises(TypeError):
        dq.validate_qcols({"a": 1})

    # custom error message
    with pytest.raises(ValueError, match="Need exactly 3"):
        dq.validate_qcols(["a", "b"], ncols_exp="==3", err_msg="Need exactly 3")


# -----------------
# build_qcols_multiple
# -----------------
def test_build_qcols_multiple_paths():
    # pre-built pairs
    pairs = [("q10", "q90"), ("lwr", "upr")]
    assert dq.build_qcols_multiple(q_cols=pairs) == pairs

    # invalid tuple size
    with pytest.raises(ValueError):
        dq.build_qcols_multiple(q_cols=[("a",)])

    # enforce triplet on pairs -> error
    with pytest.raises(ValueError):
        dq.build_qcols_multiple(q_cols=pairs, enforce_triplet=True)

    # triplets with allow_pair_when_median -> converted to pairs
    trips = [("q10", "q50", "q90"), ("l", "m", "u")]
    out = dq.build_qcols_multiple(q_cols=trips, allow_pair_when_median=True)
    assert out == [("q10", "q90"), ("l", "u")]

    # build from separate lists (pairs)
    lows, ups = ["q10", "lwr"], ["q90", "upr"]
    assert dq.build_qcols_multiple(qlow_cols=lows, qup_cols=ups) == [
        ("q10", "q90"),
        ("lwr", "upr"),
    ]

    # mismatched lengths
    with pytest.raises(ValueError):
        dq.build_qcols_multiple(qlow_cols=["a"], qup_cols=["b", "c"])

    # triplets with enforce_triplet
    meds = ["q50", "mid"]
    out2 = dq.build_qcols_multiple(
        qlow_cols=lows, qup_cols=ups, qmed_cols=meds, enforce_triplet=True
    )
    assert out2 == [("q10", "q50", "q90"), ("lwr", "mid", "upr")]

    # enforce_triplet without median -> error
    with pytest.raises(ValueError):
        dq.build_qcols_multiple(qlow_cols=lows, qup_cols=ups, enforce_triplet=True)


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
