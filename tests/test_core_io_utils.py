from __future__ import annotations

import io
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest

from kdiagram.core._io_utils import (
    _get_valid_kwargs,
    _handle_error,
    _normalize_ext,
    _post_process,
)


def test_normalize_ext_explicit_and_compressed(tmp_path: Path):
    p = tmp_path / "d.csv.gz"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    # explicit wins
    assert _normalize_ext(p, explicit="json") == ".json"
    # compressed suffix handled → base extension
    assert _normalize_ext(p) == ".csv"

    # bytes/string IO → cannot infer
    assert _normalize_ext(io.BytesIO(b"a,b\n1,2\n")) is None
    assert _normalize_ext(io.StringIO("a,b\n1,2\n")) is None

    # no suffix → None
    assert _normalize_ext(tmp_path / "noext") is None


def test_get_valid_kwargs_filters_and_kwargs_passthrough():
    def f(a, b=1, *, sep=","):
        return a, b, sep

    # unknown keys are dropped
    kw = {"a": 1, "b": 2, "sep": "|", "bogus": 123}
    out = _get_valid_kwargs(f, kw)
    assert out == {"a": 1, "b": 2, "sep": "|"}

    # a callable with **kwargs passes everything through
    def g(a, **kwargs):
        return a, kwargs

    out2 = _get_valid_kwargs(g, kw)
    assert out2 is kw  # identical dict back

    # non-callable → treat as instance; if inspect fails, returns {}
    bad = types.SimpleNamespace()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = _get_valid_kwargs(bad, {"x": 1}, error="warn")
        assert res == {}
        assert any(
            "Unable to inspect callable signature" in str(w.message)
            for w in rec
        )


def test_post_process_fill_drop_index_sort():
    df = pd.DataFrame(
        {"k": [2, 1, 3], "x": [1, None, None], "y": [None, 3, 0]}
    )

    out = _post_process(
        df,
        index_col="k",
        sort_index=True,
        drop_na=["x", "y"],  # drop rows where x and y are both NA
        fillna={"y": 0},  # fill remaining NA in y
    )
    assert list(out.index) == [1, 2, 3]
    assert out.loc[2, "y"] == 0
    assert pd.isna(out.loc[3, "x"]) and out.loc[3, "y"] == 0

    # 'any' drops any row that has at least one NA → here all 3 rows have an NA
    out2 = _post_process(
        df, index_col=None, sort_index=False, drop_na="any", fillna=None
    )
    assert out2.shape[0] == 0

    # sanity: 'all' would drop only rows where *all columns* are NA → none here
    out3 = _post_process(
        df, index_col=None, sort_index=False, drop_na="all", fillna=None
    )
    assert out3.shape[0] == 3

    # invalid string → warn & keep data shape
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out4 = _post_process(
            df,
            index_col=None,
            sort_index=False,
            drop_na="invalid",
            fillna=None,
        )
        assert out4.shape == df.shape
        assert any("dropna failed" in str(w.message) for w in rec)


def test_handle_error_warn_and_raise():
    with pytest.warns(UserWarning, match="something odd"):
        _handle_error("something odd", mode="warn", stacklevel=1)

    with pytest.raises(ValueError, match="bad"):
        _handle_error("bad", mode="raise")

    # ignore → no exception/warn
    _ = _handle_error("silent", mode="ignore")
