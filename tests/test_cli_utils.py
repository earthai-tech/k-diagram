# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import kdiagram.cli._utils as utils


def test_parse_list_and_pairs_and_quantiles():
    assert utils.parse_list("a,b, c") == ["a", "b", "c"]
    assert utils.parse_list(["x", "y"]) == ["x", "y"]
    assert utils.parse_list("", empty_as_none=True) is None

    pairs = utils.parse_pairs("a,b;c,d")
    assert pairs == [("a", "b"), ("c", "d")]

    pairs2 = utils.parse_pairs([("u", "v"), "w,z"])
    assert pairs2 == [("u", "v"), ("w", "z")]

    q = utils.parse_quantiles("10%, 50%,0.9")
    assert q == [0.1, 0.5, 0.9]

    with pytest.raises(ValueError):
        _ = utils.parse_quantiles(["-0.1", "1.2"])


def test_normalize_acov_and_natural_sort_key():
    assert utils.normalize_acov("Quarter_Circle") == "quarter_circle"
    assert utils.normalize_acov("weird") == "default"

    items = ["h2", "h10", "h1", "h3"]
    sorted_items = sorted(items, key=utils.natural_sort_key)
    assert sorted_items == ["h1", "h2", "h3", "h10"]


def test_expand_and_detect_quantile_columns():
    df = pd.DataFrame(
        {
            "subs_2023_q10": [1, 2],
            "subs_2023_q50": [2, 3],
            "subs_2023_q90": [4, 5],
            "subs_2024_q10": [2, 2],
            "subs_2024_q90": [5, 6],
            "other": [0, 0],
        }
    )
    cols = utils.expand_prefix_cols("subs", [2023, 2024], q="10")
    assert cols == ["subs_2023_q10", "subs_2024_q10"]

    det = utils.detect_quantile_columns(
        df,
        value_prefix="subs",
        horizons=[2023, 2024],
    )
    # 2023 has all three; 2024 has q10 and q90 only
    assert "subs_2023_q10" in det["q10"]
    assert "subs_2023_q50" in det["q50"]
    assert "subs_2023_q90" in det["q90"]
    assert "subs_2024_q10" in det["q10"]
    assert "subs_2024_q90" in det["q90"]
    # horizons reflect those present
    assert set(det["horizons"]) == {"2023", "2024"}

    # Regex discovery without hints
    det2 = utils.detect_quantile_columns(df)
    assert set(det2["horizons"]) == {"2023", "2024"}
    assert "subs_2023_q50" in det2["q50"]


def test_ensure_columns_and_numeric_coercion():
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["4", "x", "6"], "c": [0.1, 0.2, 0.3]}
    )
    utils.ensure_columns(df, ["a", "b"])
    with pytest.raises(ValueError):
        utils.ensure_columns(df, ["z"], error="raise")

    with pytest.warns(UserWarning):
        utils.ensure_columns(df, ["z"], error="warn")

    with pytest.warns(UserWarning):
        out = utils.ensure_numeric(df, ["a", "b"], copy=True, errors="warn")
    assert pd.api.types.is_numeric_dtype(out["a"])
    # "x" becomes NaN
    assert np.isnan(out.loc[1, "b"])

    with pytest.raises(TypeError):
        _ = utils.ensure_numeric(
            df[["b"]],
            ["b"],
            copy=True,
            errors="raise",
        )


def test_load_df_and_save_df_roundtrip_csv(tmp_path: Path):
    df = pd.DataFrame({"x": [1, 2, None], "y": [3.0, 4.0, 5.0]})
    dst = tmp_path / "data.csv"

    # write
    utils.save_df(
        df,
        dst,
        format=None,
        overwrite=True,
    )
    # write_data may return path or None; file must exist
    assert Path(dst).exists()

    # read with fillna post-process
    got = utils.load_df(
        dst,
        format=None,
        fillna={"x": 0},
    )
    exp = df.copy()
    exp["x"] = exp["x"].fillna(0)
    pdt.assert_frame_equal(
        got.reset_index(drop=True),
        exp.reset_index(drop=True),
        check_dtype=False,
    )


def test_load_df_from_buffer(tmp_path: Path):
    # test file-like source
    csv = "a,b\n1,2\n3,4\n"
    buf = io.StringIO(csv)
    got = utils.load_df(buf, format="csv")
    assert list(got.columns) == ["a", "b"]
    assert len(got) == 2
