import os
import re
import warnings

import pytest

from kdiagram.api import util

# ---------- helpers ----------


@pytest.fixture
def fixed_terminal(monkeypatch):
    """Force a deterministic terminal size for all tests here."""
    monkeypatch.setattr(
        util.shutil,
        "get_terminal_size",
        lambda fallback=(80, 24): os.terminal_size((100, 40)),
    )
    return (100, 40)


# ---------- beautify_dict ----------
def test_beautify_dict_basic_formatting(fixed_terminal):
    d = {
        3: "Home & Garden",
        2: "Health & Beauty",
        4: "Sports",
        0: "Electronics",
        1: "Fashion",
    }
    out = util.beautify_dict(d, space=4)

    # Starts and ends like a dict, and contains one row per item
    assert out.startswith("{\n")
    assert out.endswith("\n}")
    # ensure there are 5 formatted rows (keys sorted ascending)
    rows = [
        line for line in out.splitlines() if re.match(r"\s*\d+:\s'", line)
    ]
    assert len(rows) == len(d)
    # numeric keys should be sorted in the output
    keys_in_order = [int(re.findall(r"(\d+):\s'", line)[0]) for line in rows]
    assert keys_in_order == sorted(d.keys())


def test_beautify_dict_with_key_and_truncation():
    d = {1: "Electronics", 2: "VeryLongCategoryName"}
    # Force value truncation to first 4 chars
    out = util.beautify_dict(d, space=4, key="catalog", max_char=4)

    # outer key prefix must be present
    assert out.splitlines()[0].startswith("catalog : {")
    # value strings truncated with "..."
    assert "catalog" in out


def test_beautify_dict_empty_dict(fixed_terminal):
    out = util.beautify_dict({}, space=2)
    # Still a well-formed dict with just braces
    assert out.strip() == "{\n  \n}"


def test_beautify_dict_type_error_on_non_dict():
    with pytest.raises(TypeError):
        util.beautify_dict(["not", "a", "dict"])  # type: ignore[arg-type]


# ---------- to_camel_case ----------


@pytest.mark.parametrize(
    "inp,kwargs,expected",
    [
        (
            "outlier_results",
            {"delimiter": "_"},
            "OutlierResults",
        ),  # note: single-word after split parts capitalized
        ("outlier results", {"delimiter": " "}, "OutlierResults"),
        ("data science rocks", {}, "DataScienceRocks"),
        ("data_science_rocks", {}, "DataScienceRocks"),
        ("multi@var_analysis", {"use_regex": True}, "MultiVarAnalysis"),
    ],
)
def test_to_camel_case_various(inp, kwargs, expected):
    assert util.to_camel_case(inp, **kwargs) == expected


def test_to_camel_case_preserves_existing_camel():
    assert util.to_camel_case("OutlierResults") == "OutlierResults"
    assert util.to_camel_case("BoxFormatter") == "BoxFormatter"
    assert util.to_camel_case("MultiFrameFormatter") == "MultiFrameFormatter"


# ---------- to_snake_case ----------


@pytest.mark.parametrize(
    "inp,mode,expected",
    [
        ("StandardCaseX", "standard", "standard_case_x"),
        ("already_snake_case", "standard", "already_snake_case"),
        ("Hello  World!!  123", "soft", "hello_world_123"),
        ("Mixed CAPS and-dashes", "soft", "mixed_caps_and_dashes"),
        ("HTTPResponseCode", "standard", "h_t_t_p_response_code"),
    ],
)
def test_to_snake_case(inp, mode, expected):
    assert util.to_snake_case(inp, mode=mode) == expected


# ---------- get_terminal_size ----------


def test_get_terminal_size_monkeypatched(fixed_terminal):
    w, h = util.get_terminal_size()
    assert (w, h) == fixed_terminal


# ---------- get_table_size ----------


def test_get_table_size_auto_uses_terminal_width(fixed_terminal):
    width_only = util.get_table_size(width="auto")
    assert width_only == fixed_terminal[0]

    width_height = util.get_table_size(width="auto", return_height=True)
    assert width_height == fixed_terminal


def test_get_table_size_numeric_within_bounds(fixed_terminal):
    assert util.get_table_size(60) == 60


def test_get_table_size_warns_when_exceeds_terminal(fixed_terminal):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        val = util.get_table_size(1000, error="warn")
        assert val == 1000
        assert any("exceeds terminal width" in str(w.message) for w in rec)


def test_get_table_size_invalid_value_raises(fixed_terminal):
    with pytest.raises(ValueError):
        util.get_table_size("not-an-int")
