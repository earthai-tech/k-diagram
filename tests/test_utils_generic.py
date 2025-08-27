import re

import numpy as np
import pytest

from kdiagram.utils import generic_utils as gu


# -----------------
# str2columns
# -----------------
def test_str2columns_defaults_and_custom_patterns():
    text = "this.is an-example,too|and;again"
    # default behavior should split on punctuation, spaces, pipes, etc.
    out = gu.str2columns(text)
    assert out == ["this", "is", "an", "example", "too", "and", "again"]

    # explicit pattern string
    out2 = gu.str2columns("a,b; c", pattern=r"[,\s;]+")
    assert out2 == ["a", "b", "c"]

    # compiled regex
    splitter = re.compile(r"[;|]+")
    out3 = gu.str2columns("x;y|z", regex=splitter)
    assert out3 == ["x", "y", "z"]


# -----------------
# smart_format
# -----------------
def test_smart_format_various_inputs():
    assert (
        gu.smart_format(["apple", "banana", "cherry"])
        == "'apple','banana' and 'cherry'"
    )
    assert gu.smart_format(["one"]) == "'one'"
    assert gu.smart_format([]) == ""
    # Non-iterable falls back to str()
    assert gu.smart_format(123) == "123"
    # custom connector
    assert gu.smart_format(["a", "b"], choice="or") == "'a' or 'b'"


# -----------------
# count_functions
# -----------------
def test_count_functions_lists_and_counts():
    mod = "kdiagram.utils.generic_utils"

    # counts (should be positive)
    cnt = gu.count_functions(mod, include_class=True, return_counts=True)
    assert isinstance(cnt, int) and cnt > 0

    # listing includes known functions and is sorted
    names = gu.count_functions(mod, include_class=True, return_counts=False)
    assert "str2columns" in names
    assert "smart_format" in names
    assert names == sorted(names)

    # exclude classes flag shouldn't error (no classes in this module)
    names_no_cls = gu.count_functions(
        mod, include_class=False, return_counts=False
    )
    assert "str2columns" in names_no_cls


# -----------------
# drop_nan_in
# -----------------
def test_drop_nan_in_filters_all_arrays_and_warns():
    y_true = np.array([1.0, np.nan, 3.0, 4.0], dtype=float)
    y_pred1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    y_pred2 = np.array([1.1, 2.2, 3.3, 4.4], dtype=float)

    with pytest.warns(UserWarning):
        yt, yp1, yp2 = gu.drop_nan_in(y_true, y_pred1, y_pred2, error="warn")

    assert yt.tolist() == [1.0, 3.0, 4.0]
    assert yp1.tolist() == [10.0, 30.0, 40.0]
    assert yp2.tolist() == [1.1, 3.3, 4.4]


def test_drop_nan_in_raise_and_nan_policy_omit_and_clean():
    y_true = np.array([1.0, np.nan, 2.0], dtype=float)
    y_pred = np.array([9.0, 8.0, 7.0], dtype=float)

    # error='raise' path
    with pytest.raises(ValueError):
        gu.drop_nan_in(y_true, y_pred, error="raise")

    # nan_policy='omit' should filter
    yt, yp = gu.drop_nan_in(y_true, y_pred, nan_policy="omit", error="ignore")
    assert yt.tolist() == [1.0, 2.0]
    assert yp.tolist() == [9.0, 7.0]

    # no NaN case returns unchanged content
    yt2, yp2 = gu.drop_nan_in(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
    assert yt2.tolist() == [1.0, 2.0]
    assert yp2.tolist() == [5.0, 6.0]


# -----------------
# get_valid_kwargs
# -----------------
def test_get_valid_kwargs_with_function_and_invalid_keys_warns():
    def f(a, b=2, *, c=None):  # simple callable
        return a + b

    valid = gu.get_valid_kwargs(f, {"a": 1, "c": 3, "x": 99})
    assert valid == {"a": 1, "c": 3}

    # class: signature taken from __init__
    class C:
        def __init__(self, foo, bar=0):
            self.foo = foo
            self.bar = bar

    with pytest.warns(UserWarning):
        valid2 = gu.get_valid_kwargs(C, {"foo": 10, "nope": 1}, error="warn")
    assert valid2 == {"foo": 10}

    # instance -> uses its class
    c = C(1)

    valid3 = gu.get_valid_kwargs(c, {"foo": 2, "zzz": 42}, error="ignore")
    assert valid3 == {"foo": 2}


# -----------------
# error_policy
# -----------------
def test_error_policy_basic_and_auto_and_strict():
    assert gu.error_policy("warn") == "warn"

    # auto: None resolves to base
    assert gu.error_policy(None, policy="auto", base="ignore") == "ignore"
    assert gu.error_policy(None, policy="auto", base="warn") == "warn"

    # strict: None not allowed
    with pytest.raises(ValueError):
        gu.error_policy(None, policy="strict")

    # policy=None uses base, but validates base
    with pytest.raises(ValueError):
        gu.error_policy(None, policy=None, base="not-a-policy")


def test_error_policy_invalid_error_raises_custom_exception():
    with pytest.raises(KeyError):
        gu.error_policy("totally-wrong", exception=KeyError)
