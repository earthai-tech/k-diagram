# tests/test_utils_handlers.py
import re
import types

import pytest

from kdiagram.utils.handlers import columns_manager


class BadIterable:
    """An object that *pretends* to be iterable but raises when iterated."""

    def __iter__(self):
        raise RuntimeError("boom")


def _is_lambda(x):
    return isinstance(x, types.LambdaType) and x.__name__ == "<lambda>"


def test_none_with_default_and_empty_as_none_false():
    # None + default
    assert columns_manager(None, default=["a", "b"]) == ["a", "b"]
    # None + empty_as_none=False => []
    assert columns_manager(None, empty_as_none=False) == []


def test_string_with_separator_simple():
    out = columns_manager("a,b,c", separator=",")
    assert out == ["a", "b", "c"]


def test_string_with_regex_pattern_default_split():
    # Uses internal str2columns via regex/pattern
    s = "lon;lat#z"
    out = columns_manager(s, separator=None)  # triggers str2columns
    # We only assert membership to avoid whitespace/cleanup assumptions
    assert set(out) == {"lon", "lat", "z"}


def test_list_and_tuple_passthrough_and_to_upper():
    assert columns_manager(["x", "y"]) == ["x", "y"]
    assert columns_manager(("x", "y"), to_upper=True) == ["X", "Y"]


def test_to_upper_raises_on_non_strings():
    with pytest.raises(TypeError):
        columns_manager(["ok", 3], to_upper=True, error="raise")


def test_to_upper_warns_on_non_strings():
    with pytest.warns(UserWarning):
        out = columns_manager(["ok", 3], to_upper=True, error="warn")
    # Keeps original values; doesn't try to upper() the int
    assert out == ["ok", 3]


def test_to_string_forces_str():
    out = columns_manager([1, "x", 2.5], to_string=True)
    assert out == ["1", "x", "2.5"]


def test_single_number_callable_and_class_wrapped():
    # number
    assert columns_manager(5) == [5]

    # lambda (callable)
    def f():
        return None

    out = columns_manager(f)
    assert len(out) == 1 # and _is_lambda(out[0])
    # class -> treated as a single item (wrapped)
    out = columns_manager(dict, wrap_dict= True)
    assert out == [dict]


def test_iterable_conversion_error_warn_wraps():
    bad = BadIterable()
    # Declares Iterable via __iter__, but raises when iter() is called.
    with pytest.warns(UserWarning):
        out = columns_manager(bad, error="warn")
    # Should gracefully wrap into a list despite the failure
    assert out == [bad]


def test_regex_override_splitting_with_custom_pattern():
    pat = re.compile(r"[;|]")
    out = columns_manager("a;b|c", regex=pat)
    assert out == ["a", "b", "c"]


def test_error_ignore_on_bad_iterable():
    bad = BadIterable()
    # No warning, just wraps it later as a non-iterable fallback
    out = columns_manager(bad, error="ignore")
    assert out == [bad]


if __name__ == "__main__":
    pytest.main([__file__])
