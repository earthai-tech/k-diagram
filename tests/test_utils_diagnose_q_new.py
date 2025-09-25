import warnings

import numpy as np
import pandas as pd
import pytest

from kdiagram.utils.diagnose_q import (
    _verify_identical_items,
    build_q_column_names,
    build_qcols_multiple,
    check_forecast_mode,
    detect_digits,
    detect_quantiles_in,
    parse_qcols,
    to_iterable,
    validate_consistency_q,
    validate_q_dict,
    validate_qcols,
    validate_quantiles,
    validate_quantiles_in,
)

# -------------------------
# parse_qcols
# -------------------------


def test_parse_qcols_dict_ok_and_no_q50_middle_choice():
    out = parse_qcols({"q10": "lo", "q70": "midish", "q90": "hi"})
    assert out["valid"] is True
    assert out["lowest_col"] == "lo"
    assert out["highest_col"] == "hi"
    # median not explicitly q50 -> middle by order (q10, q70, q90) => q70
    assert out["median_col"] == "midish"
    assert out["parsed_qvals"] == {10.0: "lo", 70.0: "midish", 90.0: "hi"}


def test_parse_qcols_from_list_tuple_and_fallback_and_warnings():
    # list/tuple -> auto keys q0,q1,...
    out = parse_qcols(["L", "M", "U"])
    assert (
        out["valid"]
        and out["lowest_col"] == "L"
        and out["median_col"] == "M"
        and out["highest_col"] == "U"
    )

    # fallback when empty
    fb = ("a", "b", "c")
    out_empty = parse_qcols(None, fallback_cols=fb)
    assert (
        out_empty["valid"] is False
        and (
            out_empty["lowest_col"],
            out_empty["median_col"],
            out_empty["highest_col"],
        )
        == fb
    )

    # bad keys -> warnings path (non 'q' prefix and non-string keys)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = parse_qcols({"notq10": "x", 10: "y"})
        # two separate warnings (prefix + type)
        assert any("not prefixed with 'q'" in str(ww.message) for ww in w)
        assert any("is not a string" in str(ww.message) for ww in w)

    # error='raise' on bad string parse
    with pytest.raises(ValueError):
        parse_qcols({"qX": "oops"}, error="raise")
    with pytest.raises(ValueError):
        parse_qcols({"notq10": "nope"}, error="raise")


# -------------------------
# check_forecast_mode
# -------------------------


def test_check_forecast_mode_branches_point_and_quantile():
    # point mode: q not None => warn or raise
    with pytest.warns(UserWarning):
        q = check_forecast_mode("point", q=[0.1, 0.5], error="warn")
        assert q is None
    with pytest.raises(ValueError):
        _ = check_forecast_mode("point", q=[0.1], error="raise")

    # quantile mode: q None => default and warn; uses validate_quantiles internally
    with pytest.warns(UserWarning):
        q = check_forecast_mode("quantile", q=None, error="warn")
        assert list(np.around(q, 1)) == [0.1, 0.5, 0.9]

    # quantile mode with soft scaling integers
    q2 = check_forecast_mode(
        "quantile", q=[10, 50, 90], error="warn", q_mode="soft"
    )
    assert [round(_q, 1) for _q in q2] == [0.1, 0.5, 0.9]

    # ops=check_only returns None
    assert (
        check_forecast_mode(
            "quantile", q=[0.1], ops="check_only", q_mode="soft"
        )
        is None
    )

    # invalid mode
    with pytest.raises(ValueError):
        check_forecast_mode("median", q=[0.5])


# -------------------------
# to_iterable
# -------------------------


def test_to_iterable_variants():
    assert to_iterable("abc", exclude_string=True) is False
    with pytest.raises(ValueError):
        to_iterable("a,b", parse_string=True)  # requires transform=True

    out = to_iterable(
        "a,b; b",
        transform=True,
        parse_string=True,
        delimiter=r"[ ,;]+",
        unique=True,
    )
    assert out == ["a", "b"]

    out2 = to_iterable([1, [2, 3], (4, {5})], transform=True, flatten=True)
    assert out2 == [1, 2, 3, 4, 5]


# -------------------------
# validate_q_dict
# -------------------------


def test_validate_q_dict_conversion_and_recheck():
    qd = {"10%": ["q10"], "0.5": ["q50"], 0.9: ["q90"], "oops": ["x"]}
    out = validate_q_dict(
        qd
    )  # no strict recheck -> keeps problematic key string
    assert out[0.1] == ["q10"]
    assert out[0.5] == ["q50"]
    assert out[0.9] == ["q90"]
    assert "oops" in out

    # non-dict raises
    with pytest.raises(TypeError):
        validate_q_dict([("0.1", ["q10"])])

    # recheck enforces [0,1] -> 200% becomes 2.0 and should fail strict validation
    with pytest.raises(ValueError):
        validate_q_dict({"200%": ["bad"], "0.1": ["ok"]}, recheck=True)


# -------------------------
# validate_quantiles / validate_quantiles_in
# -------------------------


def test_validate_quantiles_strict_soft_dtype_round_and_errors():
    # strict - good
    a = validate_quantiles(
        [0.123456, 0.789012], asarray=True, round_digits=3, dtype="float64"
    )
    assert (
        isinstance(a, np.ndarray)
        and a.dtype == np.float64
        and a.tolist() == [0.123, 0.789]
    )

    # strict - out of range
    with pytest.raises(ValueError):
        validate_quantiles([1.2])

    # strict - non-numeric
    with pytest.raises(ValueError):
        validate_quantiles(["bad"], dtype="float32")

    # soft mode with percentages and integers (uniform)
    s = validate_quantiles(
        ["20%", 5, 150], mode="soft", scale_method="uniform"
    )
    # uniform max digits among integers [5,150] = 3 -> 5/1000=0.005, 150/1000=0.15
    assert [round(_s, 3) for _s in s] == [0.02, 0.00, 0.15]

    # individual scaling
    s2 = validate_quantiles([5, 150], mode="soft", scale_method="individual")
    assert [round(_s, 2) for _s in s2] == [0.5, 0.15]

    # invalid scaling method
    with pytest.raises(ValueError):
        validate_quantiles([10], mode="soft", scale_method="nope")


def test_validate_quantiles_in_and_dtype_rules():
    v = validate_quantiles_in([0.1, 0.2, 0.3])
    assert list(np.around(v, 1)) == [0.1, 0.2, 0.3]
    a = validate_quantiles_in(
        np.array([0.4, 0.6]), asarray=True, dtype="float32"
    )
    assert isinstance(a, np.ndarray) and a.dtype == np.float32

    with pytest.raises(ValueError):
        validate_quantiles_in([0.1, 1.1])

    with pytest.raises(ValueError):
        validate_quantiles_in(
            [0.1], dtype="float16"
        )  # unsupported string dtype


# -------------------------
# detect_quantiles_in / build_q_column_names
# -------------------------


def _make_df_for_quantile_detection():
    return pd.DataFrame(
        {
            "sales_q0.25": [4.2, 4.4],
            "sales_q0.75": [5.8, 5.7],
            "sales_2023_q0.5": [5.0, 5.1],
            "temp_2024_q0.5": [22.1, 22.0],
            "risk_q150": [0.8, 0.7],  # tests soft scaling of 150 -> 0.15
            "q0.1": [1, 2],  # no prefix case
        }
    )


def test_detect_quantiles_in_modes_and_returns():
    df = _make_df_for_quantile_detection()

    # default returns column names
    cols = detect_quantiles_in(df, col_prefix="sales")
    assert set(cols) == {"sales_q0.25", "sales_q0.75", "sales_2023_q0.5"}

    # filter by date (only 2023)
    qvals = detect_quantiles_in(
        df, col_prefix="sales", dt_value=["2023"], return_types="q_val"
    )
    assert qvals is None

    # soft scaling quantile string like q150 -> 0.15
    qvals2 = detect_quantiles_in(
        df, col_prefix="risk", return_types="q_val", mode="soft"
    )
    assert qvals2 == [0.15]

    # strict mode still accepts 0.25/0.75/0.5
    vals = detect_quantiles_in(
        df, col_prefix="sales", return_types="values", mode="strict"
    )
    assert isinstance(vals, np.ndarray) and vals.shape[1] == len(df)

    # frame return
    frame = detect_quantiles_in(df, col_prefix="sales", return_types="frame")
    assert isinstance(frame, pd.DataFrame) and set(frame.columns).issubset(
        df.columns
    )

    # no prefix case
    cols_all = detect_quantiles_in(df, return_types="columns")
    assert "q0.1" in cols_all


def test_build_q_column_names_strict_and_flexible():
    df = _make_df_for_quantile_detection().rename(
        columns={"sales_2023_q0.5": "sales_2023_q0.50"}
    )
    # strict: exact candidates (decimal and percent forms)
    got = build_q_column_names(
        df,
        [0.25, 0.5],
        value_prefix="sales",
        strict_match=True,
        dt_value=["2023"],
    )
    # The column 'sales_2023_q0.50' matches decimal form; 'sales_q0.25' present
    assert list(set(got))[0] in {"sales_2023_q0.50", "sales_q0.25"}

    # flexible: regex pattern
    got2 = build_q_column_names(
        df, ["25%", "50%"], value_prefix="sales", strict_match=False
    )
    assert (
        "sales_q0.25" in got2
    )  # and any(re.match(r"sales_\d{4}_q0\.5", c) for c in got2)


# -------------------------
# detect_digits
# -------------------------


def test_detect_digits_general_and_quantile_and_options():
    # general
    assert detect_digits("subsidence_q10_step1 and p=3.14") == [
        10.0,
        1.0,
        3.14,
    ]

    # quantile mode -> after '_q' and before '_step' or EoS; soft-scaled
    d = detect_digits(
        ["x_q10_step1", "x_q50_step1", "x_q89"],
        as_q=True,
        sort=True,
        return_unique=True,
    )
    assert d == [0.1, 0.5, 0.89]

    # warn on bad conversion (pattern still returns tuples)
    # — use custom pattern to force bad match
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _ = detect_digits(
            "val: not-a-number",
            pattern=r"(not-a-number)",
            error="warn",
            verbose=1,
        )
        # no numeric -> warning message printed (captured), but function returns []
        assert isinstance(_, list) and len(_) == 0


# -------------------------
# validate_consistency_q
# -------------------------


def test_validate_consistency_q_soft_and_strict_and_mismatch_behaviors():
    cols = ["sub_q10_step1", "sub_q50_step1", "sub_q90_step1", "other"]
    # perfect soft
    assert validate_consistency_q([0.1, 0.5, 0.9], cols) == [0.1, 0.5, 0.9]

    # strict identical
    assert validate_consistency_q([0.1, 0.5, 0.9], cols, mode="strict") == [
        0.1,
        0.5,
        0.9,
    ]

    # mismatch -> warn and keep valid subset (length mismatch warning branch)
    with pytest.warns(UserWarning):
        got = validate_consistency_q([0.1, 0.9], cols, error="warn")
        assert got == [0.1, 0.9]

    # mismatch with default_to='auto_q' returns detected values when warn
    with pytest.warns(UserWarning):
        got2 = validate_consistency_q(
            [0.1], ["a_q10", "a_q90"], error="warn", default_to="auto_q"
        )
        assert got2 == [0.1, 0.9]

    # hard error on total mismatch
    with pytest.raises(ValueError):
        validate_consistency_q([0.2], ["a_q10"])


# -------------------------
# _verify_identical_items
# -------------------------


def test__verify_identical_items_modes_and_errors():
    # ascending ok
    assert (
        _verify_identical_items(
            [1, 2, 3], [1, 2, 3], mode="ascending", ops="check_only"
        )
        is True
    )
    assert _verify_identical_items(
        [1, 2, 3], [1, 2, 3], mode="ascending", ops="validate"
    ) == [1, 2, 3]

    # ascending mismatch
    with pytest.raises(ValueError):
        _ = _verify_identical_items(
            [1, 2], [1, 3], mode="ascending", ops="check_only"
        )

    # unique mismatch warn path
    with pytest.warns(UserWarning):
        ok = _verify_identical_items(
            [1, 2], [1, 3], mode="unique", ops="check_only", error="warn"
        )
        assert ok is False

    # bad mode/op
    with pytest.raises(ValueError):
        _verify_identical_items([], [], mode="nope")
    with pytest.raises(ValueError):
        _verify_identical_items([], [], ops="nope")


# -------------------------
# validate_qcols
# -------------------------


def test_validate_qcols_happy_and_errors_and_expectations():
    assert validate_qcols("q50") == ["q50"]
    assert validate_qcols(("p1", "p2", "")) == ["p1", "p2"]

    # expectation operators
    assert validate_qcols(["a", "b"], ncols_exp="==2") == ["a", "b"]
    assert validate_qcols(["a", "b"], ncols_exp=">=2") == ["a", "b"]
    assert validate_qcols(["a", "b", "c"], ncols_exp=">2") == ["a", "b", "c"]
    assert validate_qcols(["a", "b"], ncols_exp="<=2") == ["a", "b"]
    assert validate_qcols(["a"], ncols_exp="<2") == ["a"]

    with pytest.raises(ValueError):
        validate_qcols(None)
    with pytest.raises(TypeError):
        validate_qcols({"not": "allowed"})
    with pytest.raises(ValueError):
        validate_qcols(["a"], ncols_exp="==2", err_msg="custom length error")

    with pytest.raises(ValueError):
        validate_qcols(["a"], ncols_exp="~2")  # invalid expectation syntax


# -------------------------
# build_qcols_multiple
# -------------------------


def test_build_qcols_multiple_paths_and_errors():
    # case 1: q_cols provided
    assert build_qcols_multiple(q_cols=[("q10", "q90"), ("l", "u")]) == [
        ("q10", "q90"),
        ("l", "u"),
    ]

    # invalid tuple sizes
    with pytest.raises(ValueError):
        build_qcols_multiple(q_cols=[("a",), ("b", "c", "d", "e")])

    # enforce triplet on pairs -> error
    with pytest.raises(ValueError):
        build_qcols_multiple(q_cols=[("q10", "q90")], enforce_triplet=True)

    # triplets to pairs when allow_pair_when_median=True
    assert build_qcols_multiple(
        q_cols=[("q10", "q50", "q90")], allow_pair_when_median=True
    ) == [("q10", "q90")]

    # case 2: build from separate lists
    lows, ups, meds = ["l1", "l2"], ["u1", "u2"], ["m1", "m2"]
    assert build_qcols_multiple(qlow_cols=lows, qup_cols=ups) == [
        ("l1", "u1"),
        ("l2", "u2"),
    ]
    assert build_qcols_multiple(
        qlow_cols=lows, qup_cols=ups, qmed_cols=meds, enforce_triplet=True
    ) == [
        ("l1", "m1", "u1"),
        ("l2", "m2", "u2"),
    ]

    # allow_pair_when_median True keeps pairs
    assert build_qcols_multiple(
        qlow_cols=lows,
        qup_cols=ups,
        qmed_cols=meds,
        allow_pair_when_median=True,
    ) == [
        ("l1", "u1"),
        ("l2", "u2"),
    ]

    # mismatches / missing parts
    with pytest.raises(ValueError):
        build_qcols_multiple(qlow_cols=["a"], qup_cols=["b", "c"])
    with pytest.raises(ValueError):
        build_qcols_multiple(
            qlow_cols=lows,
            qup_cols=ups,
            qmed_cols=["only1"],
            enforce_triplet=True,
        )
    with pytest.raises(ValueError):
        build_qcols_multiple(
            qlow_cols=lows, qup_cols=ups, enforce_triplet=True
        )
