import re

import pytest

import kdiagram.api.summary as summary_mod
from kdiagram.api.summary import ResultSummary


def test_resultsummary_add_results_creates_snake_case_attrs_and_stores_copy():
    rs = ResultSummary(name="Data Check", max_char=200)
    data = {
        "Missing Data": {"A": 2},
        "Out Of Range": 7,
    }
    rs.add_results(data)
    # attributes are snake_case
    assert rs.missing_data == {"A": 2}
    assert rs.out_of_range == 7
    # internal storage is a (deep) copy: mutate original and ensure rs.results unchanged
    data["Missing Data"]["A"] = 999
    assert rs.missing_data == {"A": 2}  # unchanged


def test_resultsummary_str_truncates_and_adds_note_when_needed():
    rs = ResultSummary(name="data check", max_char=10)  # very small to force '...'
    rs.add_results({"very_long_value": "abcdefghijklmno"})  # > 10
    s = str(rs)
    # Name should be CamelCase in header
    assert s.splitlines()[0].startswith("DataCheck(")
    # truncation and note must appear
    assert "..." in s
    assert "Note: Output may be truncated" in s
    # shows entries count
    assert "[ 1 entries ]" in s


def test_resultsummary_str_no_note_when_muted():
    rs = ResultSummary(name="demo", max_char=5, mute_note=True)
    rs.add_results({"x": "abcdefghijklmnopqrstuvwxyz"})
    s = str(rs)
    assert "..." in s
    assert "Note: Output may be truncated" not in s


def test_resultsummary_repr_empty_and_populated():
    empty = ResultSummary(name="Cool Name")
    r = repr(empty)
    assert r.startswith("<Empty Cool Name>")

    rs = ResultSummary(name="Cool Name")
    rs.add_results({"a": 1, "b": 2})
    r2 = repr(rs)
    assert "<Cool Name with 2 entries." in r2


def test_resultsummary_pad_keys_auto_alignment_spaces():
    # longest key is much longer to ensure padding visible
    rs = ResultSummary(name="pad", pad_keys="auto", max_char=200)
    rs.add_results({"short": 1, "a_very_long_key": 2})
    out = str(rs)

    # Find the lines with our keys
    lines = [ln for ln in out.splitlines() if "short" in ln or "a_very_long_key" in ln]
    # Expect "short" line to have extra spaces before colon because of ljust
    short_line = next(ln for ln in lines if "short" in ln)
    # Match "short<spaces> :"
    assert re.search(r"short\s+ :", short_line) is not None


def test_resultsummary_flatten_nested_false_uses_beautify(monkeypatch):
    # Monkeypatch beautify_dict inside module to make assertion easy
    monkeypatch.setattr(summary_mod, "beautify_dict", lambda d, **k: "<<pretty>>")

    rs = ResultSummary(name="X", pad_keys="auto", max_char=200, flatten_nested_dicts=False)
    rs.add_results({"nested": {"k": "v"}, "other": 1})

    s = str(rs)
    # our beautify stub should be present
    assert "<<pretty>>" in s
    # the simple value also appears
    assert "other" in s and "1" in s


def test_resultsummary_add_results_type_error():
    rs = ResultSummary()
    with pytest.raises(TypeError):
        rs.add_results(["not", "a", "dict"])  # type: ignore[arg-type]

if __name__=="__main__": # pragma: no-cover
    pytest.main( [__file__])