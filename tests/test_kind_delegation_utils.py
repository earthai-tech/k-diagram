import matplotlib

matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from kdiagram.utils.plot import maybe_delegate_cartesian, validate_kind


def test_validate_kind_defaults_and_casefold():
    # default path when None
    assert validate_kind(None) == "polar"
    # case-insensitive normalization
    assert validate_kind("Polar") == "polar"
    assert validate_kind("CARTESIAN") == "cartesian"


def test_validate_kind_rejects_invalid_value_and_type():
    with pytest.raises(ValueError) as ei:
        validate_kind("weird")
    assert str(ei.value) == "kind must be 'polar' or 'cartesian'."

    with pytest.raises(ValueError) as ei2:
        validate_kind(123)  # type: ignore[arg-type]
    assert str(ei2.value) == "kind must be 'polar' or 'cartesian'."


def test_maybe_delegate_cartesian_calls_renderer_and_propagates_args_kwargs():
    called = {"flag": False, "args": None, "kwargs": None}

    def cartesian_fn(*args, **kwargs):
        called["flag"] = True
        called["args"] = args
        called["kwargs"] = kwargs
        fig, ax = plt.subplots()
        return ax

    ax = maybe_delegate_cartesian(
        "cartesian",
        cartesian_fn,
        1,
        2,
        "x",
        foo="bar",
        spam=42,
    )

    try:
        assert isinstance(ax, Axes)
        assert called["flag"] is True
        assert called["args"] == (1, 2, "x")
        assert called["kwargs"] == {"foo": "bar", "spam": 42}
    finally:
        plt.close(ax.figure)


def test_maybe_delegate_cartesian_returns_none_when_polar():
    hit = {"called": False}

    def cartesian_fn(*_args, **_kwargs):
        hit["called"] = True
        fig, ax = plt.subplots()
        return ax

    out = maybe_delegate_cartesian("polar", cartesian_fn)
    assert out is None
    assert hit["called"] is False
