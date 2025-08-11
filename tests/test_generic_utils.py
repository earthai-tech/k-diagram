import re

import numpy as np
import pytest

from kdiagram.utils.generic_utils import (
    count_functions,
    drop_nan_in,
    error_policy,
    get_valid_kwargs,
    smart_format,
    str2columns,
)

# ---------------------------
# str2columns
# ---------------------------


def test_str2columns_default_returns_raw():
    assert str2columns("no-split-here") == ["no", "split", "here"]


def test_str2columns_with_pattern():
    # split on commas or whitespace
    parts = str2columns("a, b  c,d", pattern=r"[,\s]+")
    assert parts == ["a", "b", "c", "d"]


def test_str2columns_with_compiled_regex():
    regex = re.compile(r"[#&.*@!_,;\s-]+\s*")
    parts = str2columns("this.is an-example;now", regex=regex)
    assert parts == ["this", "is", "an", "example", "now"]


# ---------------------------
# smart_format
# ---------------------------


def test_smart_format_non_iterable():
    assert smart_format("banana") == "'b','a','n','a','n' and 'a'"


def test_smart_format_singleton_iterable():
    # Quotes are from repr: '"apple"'
    assert smart_format(["apple"]) == "'apple'"


def test_smart_format_multiple_items_and_connector():
    out = smart_format(["apple", "banana", "cherry"], choice="and")
    assert out == "'apple','banana' and 'cherry'"


# ---------------------------
# count_functions
# ---------------------------
@pytest.mark.skip("tests done on the root of the project")
def test_count_functions_returns_int_for_counts():
    # Don’t assert exact number (file may change). Just basic sanity.
    n = count_functions(
        "kdiagram.utils.generic_utils", include_class=True, return_counts=True
    )
    assert isinstance(n, int)
    assert n > 0


# ---------------------------
# drop_nan_in
# ---------------------------


def test_drop_nan_in_raises_on_nan_with_error_raise():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.1, 2.2, 3.3])
    with pytest.raises(ValueError):
        drop_nan_in(y_true, y_pred, error="raise")


def test_drop_nan_in_warns_and_filters():
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred1 = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred2 = np.array([100.0, 200.0, 300.0, 400.0])

    with pytest.warns(UserWarning):
        y_f, p1, p2 = drop_nan_in(y_true, y_pred1, y_pred2, error="warn")

    # NaN row dropped -> length 3
    assert y_f.shape == (3,)
    assert p1.shape == (3,)
    assert p2.shape == (3,)
    # Check alignment kept (indices 0,2,3)
    np.testing.assert_allclose(y_f, [1.0, 3.0, 4.0])
    np.testing.assert_allclose(p1, [10.0, 30.0, 40.0])
    np.testing.assert_allclose(p2, [100.0, 300.0, 400.0])


def test_drop_nan_in_nan_policy_omit():
    y_true = np.array([np.nan, 2.0, np.nan, 4.0])
    y_pred = np.array([9.0, 8.0, 7.0, 6.0])
    y_f, p = drop_nan_in(y_true, y_pred, error="ignore", nan_policy="omit")
    np.testing.assert_allclose(y_f, [2.0, 4.0])
    np.testing.assert_allclose(p, [8.0, 6.0])


# ---------------------------
# get_valid_kwargs
# ---------------------------


def _dummy(a, b=2, *, c=None, d=4):
    return a + b + (c or 0) + d


def test_get_valid_kwargs_filters_and_warns():
    kwargs = {"a": 1, "b": 2, "c": 3, "nope": 99}
    with pytest.warns(UserWarning) as rec:
        valid = get_valid_kwargs(_dummy, kwargs)
    assert valid == {"a": 1, "b": 2, "c": 3}
    # Ensure warning mentions the bad key
    assert any("nope" in str(w.message) for w in rec)


# ---------------------------
# error_policy
# ---------------------------


def test_error_policy_passthrough_valid_value():
    assert error_policy("warn") == "warn"


def test_error_policy_auto_base_used_when_none():
    # none -> policy auto -> base 'warn'
    assert error_policy(None, policy="auto", base="warn") == "warn"


def test_error_policy_strict_disallows_none():
    with pytest.raises(ValueError):
        error_policy(None, policy="strict")


def test_error_policy_policy_none_uses_base_and_validates():
    # Valid base passes through
    assert error_policy(None, policy=None, base="ignore") == "ignore"
    # Invalid base should raise
    with pytest.raises(ValueError):
        error_policy(None, policy=None, base="whatever")


if __name__ == "__main__":
    pytest.main([__file__])
