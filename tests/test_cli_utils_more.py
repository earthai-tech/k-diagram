from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

import kdiagram.cli._utils as cu

# ----------- ensure_columns / ensure_numeric -------------------------------


def test_ensure_columns_and_numeric_warn_and_ignore():
    df = pd.DataFrame({"a": ["1", "x", "3"], "b": [1, 2, 3]})

    # ensure_columns warn path
    with pytest.warns(UserWarning):
        cu.ensure_columns(df, ["a", "c"], error="warn")

    # raise on missing column
    with pytest.raises(ValueError):
        cu.ensure_columns(df, ["c"])

    # errors='warn' introduces NaN and emits one warning
    with pytest.warns(UserWarning, match="coerced to NaN"):
        out = cu.ensure_numeric(df.copy(), ["a"], copy=False, errors="warn")
        assert out["a"].isna().sum() == 1

    # errors='ignore' → silently coerces with NaN introduced
    out2 = cu.ensure_numeric(df.copy(), ["a"], copy=False, errors="ignore")
    assert out2["a"].isna().sum() == 1

    # bad errors value
    with pytest.raises(ValueError):
        cu.ensure_numeric(df, ["a"], errors="boom")


# ----------- parsing: lists, pairs, quantiles, figsize ---------------------


def test_parse_list_and_pairs_and_quantiles():
    assert cu.parse_list("a,b,,c") == ["a", "b", "c"]
    assert cu.parse_list("", empty_as_none=True) is None
    assert cu.parse_list("", empty_as_none=False) == []

    assert cu.parse_pairs("a,b;c,d") == [("a", "b"), ("c", "d")]
    with pytest.raises(ValueError):
        cu.parse_pairs("a,b,c")

    assert cu.parse_quantiles(["10%", "0.5", "90%"]) == [0.1, 0.5, 0.9]
    with pytest.raises(ValueError):
        cu.parse_quantiles(["0", "1"])


def test_parse_cols_pair_and_figsize_and_float_list():
    assert cu.parse_cols_pair("lo,up") == ["lo", "up"]
    with pytest.raises(argparse.ArgumentTypeError):
        cu.parse_cols_pair("onlyone")

    assert cu.parse_figsize("6x4") == (6.0, 4.0)
    assert cu.parse_figsize(" 8 , 5 ") == (8.0, 5.0)
    with pytest.raises(argparse.ArgumentTypeError):
        cu.parse_figsize("bad")

    # _parse_float_list
    assert cu._parse_float_list(1.2) == [1.2]
    assert cu._parse_float_list(["1", "2.5"]) == [1.0, 2.5]
    assert cu._parse_float_list(["1,2", "3"]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        cu._parse_float_list(["a"])


def test__infer_figsize_and__parse_q_levels_and__coerce_q_levels():
    assert cu._infer_figsize([7, 3]) == (7.0, 3.0)
    with pytest.raises(ValueError):
        cu._infer_figsize([1])

    assert cu._parse_q_levels("0.1,0.9") == [0.1, 0.9]
    assert cu._coerce_q_levels(["0.1", "0.9"]) == [0.1, 0.9]
    assert cu._coerce_q_levels(None) == []
    with pytest.raises(ValueError):
        cu._parse_q_levels("bad")


# ----------- column discovery / expansion ----------------------------------


def test_detect_quantile_columns_modes_and_sorting():
    df = pd.DataFrame(
        {
            "val_2_q10": [0],
            "val_10_q50": [0],
            "val_1_q10": [0],
            "val_1_q50": [0],
            "val_1_q90": [0],
        }
    )
    # Regex scan chooses prefix='val' and collects all seen horizons, naturally sorted
    d = cu.detect_quantile_columns(df)
    assert d["horizons"] == ["1", "2", "10"]  # <-- include "10"

    # explicit prefix+horizons path filters by presence
    d2 = cu.detect_quantile_columns(
        df, value_prefix="val", horizons=[1, 2, 3]
    )
    assert d2["horizons"] == ["1", "2"]
    assert "val_1_q10" in d2["q10"]


def test_expand_prefix_and_normalize_acov_and_natural_sort():
    cols = cu.expand_prefix_cols("y", [3, 1, 2], q="50")
    assert cols == ["y_3_q50", "y_1_q50", "y_2_q50"]
    assert cu.normalize_acov("Half_Circle") == "half_circle"
    assert cu.normalize_acov("weird") == "default"
    # natural sort: 'z2' comes before 'z10'
    assert cu.natural_sort_key("z2") < cu.natural_sort_key("z10")


# ----------- detect/resolve ytrue & preds ----------------------------------


def test_detect_and_resolve_ytrue_preds_from_auto_and_flags():
    # (a) Auto-detect single pred: use a recognized default name (e.g. 'pred')
    df_point_auto = pd.DataFrame({"actual": [1, 2, 3], "pred": [1, 2, 2]})
    ns_point = SimpleNamespace(
        actual_col=None,
        y_true=None,
        model=None,
        pred=None,
        y_pred=None,
        names=None,
        q_levels=None,
        pred_cols=None,
        q_cols=None,
    )
    y, specs = cu.resolve_ytrue_preds(ns_point, df_point_auto)
    assert y == "actual" and specs[0].cols == ["pred"]

    # (b) Explicit model token still works with arbitrary column names
    df_point_model = pd.DataFrame({"actual": [1, 2, 3], "m": [1, 2, 2]})
    ns2 = SimpleNamespace(
        actual_col="actual",
        y_true=None,
        model=["M1:m"],
        pred=None,
        y_pred=None,
        names=None,
        q_levels=None,
        pred_cols=None,
        q_cols=None,
    )
    y2, specs2 = cu.resolve_ytrue_preds(ns2, df_point_model)
    assert (
        y2 == "actual" and specs2[0].name == "M1" and specs2[0].cols == ["m"]
    )

    # (c) Auto from q-groups: present → preferred over single pred
    df_q = pd.DataFrame(
        {
            "actual": [1, 2, 3],
            "p_q10": [0, 1, 2],
            "p_q50": [1, 2, 3],
            "p_q90": [2, 3, 4],
        }
    )
    ns_q = SimpleNamespace(
        actual_col="actual",
        y_true=None,
        model=None,
        pred=None,
        y_pred=None,
        names=None,
        q_levels=None,
        pred_cols=None,
        q_cols=None,
    )
    y3, specs3 = cu.resolve_ytrue_preds(ns_q, df_q)
    assert y3 == "actual"
    assert specs3 and specs3[0].cols == ["p_q10", "p_q50", "p_q90"]


# ----------- _collect_pred_specs / _collect_point_preds ---------------------


def test_collect_pred_specs_group_splitting_by_q_levels():
    ns = SimpleNamespace(
        model=None,
        pred=None,
        pred_cols=None,
        q_cols=None,
        q_levels="10,50,90",
    )
    # flattened group of 6 cols should be split into two groups of 3 by q-levels
    cols = "a_q10,a_q50,a_q90,b_q10,b_q50,b_q90"
    ns.pred = cols
    groups = cu._collect_pred_specs(ns)
    assert len(groups) == 2 and all(len(g[1]) == 3 for g in groups)


def test_collect_point_preds_expands_when_no_model_and_enforces_one_col():
    df = pd.DataFrame({"actual": [1, 2, 3], "m1": [1, 2, 3], "m2": [0, 1, 2]})
    # Without --model, passing two cols in one spec expands to separate groups
    SimpleNamespace(model=None, pred="m1,m2", names=None)
    yps, names = cu._collect_point_preds(
        df.assign(actual=[1, 2, 3]),
        SimpleNamespace(model=None, pred="m1,m2", names=None),
    )
    assert len(yps) == 2 and names == ["M1", "M2"]

    # With --model, a group with >1 col is invalid for point preds
    ns_bad = SimpleNamespace(model=["M: m1,m2"], pred=None, names=None)
    with pytest.raises(SystemExit):
        cu._collect_point_preds(df, ns_bad)


# ----------- argparse Actions & helpers -------------------------------------


def test_columns_actions_and_ns_get_and_add_bool_flag():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pair", action=cu.ColumnsPairAction)
    p.add_argument("--cols", action=cu.ColumnsListAction, nargs="+")
    p.add_argument("--flex", action=cu.FlexibleListAction, nargs="+")
    cu.add_bool_flag(p, "show-grid", True, "on", "off")

    ns = p.parse_args(
        [
            "--pair",
            "lo,up",
            "--cols",
            "a,b",
            "c",
            "--flex",
            "x",
            "y,z",
            "--no-show-grid",
        ]
    )
    assert ns.pair == ["lo", "up"]
    assert ns.cols == ["a", "b", "c"]
    assert ns.flex == ["x", "y", "z"]
    assert ns.show_grid is False

    # ns_get picks the first existing attribute
    res = cu.ns_get(ns, "missing", "flex")
    assert res == ["x", "y", "z"]


def test_kv_and_tokens_and_maps_and_metrics_and_labels():
    # token splitting
    assert cu._split_tokens(["a,b", "c"]) == ["a", "b", "c"]

    # kv list
    st = cu._parse_kv_list(["alpha=0.1", "bold=true", "pad=2"])
    assert st == {"alpha": 0.1, "bold": True, "pad": 2}
    with pytest.raises(argparse.ArgumentTypeError):
        cu._parse_kv_list(["bad"])

    # name:bool map
    nb = cu._parse_name_bool_map(["r2:true", "rmse:false", "oops", "x:maybe"])
    assert nb == {"r2": True, "rmse": False}

    # metric values (merging across split tokens)
    mv = cu._parse_metric_values(["r2:0.8,0.9", "rmse:1,2"])
    assert mv == {"r2": [0.8, 0.9], "rmse": [1.0, 2.0]}
    with pytest.raises(SystemExit):
        cu._parse_metric_values(["r2:0.8", "rmse:1,2,3"])  # unequal lengths

    # resolve metric labels
    ns = argparse.Namespace(
        no_metric_labels=False, metric_label=["r2:R²", "rmse:Root RMSE"]
    )
    assert cu._resolve_metric_labels(ns) == {"r2": "R²", "rmse": "Root RMSE"}
    ns2 = argparse.Namespace(no_metric_labels=True, metric_label=None)
    assert cu._resolve_metric_labels(ns2) is False
