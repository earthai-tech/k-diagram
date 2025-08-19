import copy

import pytest

from kdiagram.api.bunch import Bunch, FlexDict

# ---------- Bunch ----------


def test_bunch_attribute_get_set_delete_roundtrip():
    b = Bunch(a=1)
    # get existing via attr and key
    assert b.a == 1
    assert b["a"] == 1

    # set via attr, read via key
    b.b = "hello"
    assert b["b"] == "hello"

    # set via key, read via attr
    b["c"] = [1, 2, 3]
    assert b.c == [1, 2, 3]

    # delete via attr
    del b.b
    assert "b" not in b

    # missing attr raises AttributeError (not KeyError)
    with pytest.raises(AttributeError):
        _ = b.missing


def test_bunch_dir_includes_keys():
    b = Bunch(a=1, zebra=2)
    d = dir(b)
    assert "a" in d and "zebra" in d
    # also contains normal dict attrs (sampling one)
    assert "__len__" in d


def test_bunch_copy_and_deepcopy_behavior():
    b = Bunch(n=1, nested={"x": 10})
    # shallow copy should share nested object
    s = b.copy()
    assert isinstance(s, Bunch)
    assert s is not b
    assert s["nested"] is b["nested"]
    s["nested"]["x"] = 99
    assert b["nested"]["x"] == 99  # shared in shallow copy

    # copy.copy uses __copy__
    s2 = copy.copy(b)
    assert isinstance(s2, Bunch)
    assert s2 is not b

    # deep copy should not share nested object
    d = copy.deepcopy(b)
    assert isinstance(d, Bunch)
    assert d is not b
    assert d["nested"] is not b["nested"]
    d["nested"]["x"] = 123
    assert b["nested"]["x"] == 99


def test_bunch_repr():
    b = Bunch(a=1)
    r = repr(b)
    assert r.startswith("Bunch({")
    assert "'a': 1" in r


# ---------- FlexDict ----------


def test_flexdict_attribute_access_and_repr_and_dir():
    fd = FlexDict(pkg="kdiagram", version="1.0")
    # attr + item access
    assert fd.pkg == "kdiagram"
    assert fd["version"] == "1.0"

    # setting new attribute
    fd.goal = "simplify"
    assert fd["goal"] == "simplify"

    # __dir__ returns only keys
    d = fd.__dir__()
    assert set(d) == {"__dict__", "goal", "pkg", "version"}

    # repr contains keys
    r = repr(fd)
    assert "<FlexDict with keys:" in r
    assert "pkg" in r and "version" in r and "goal" in r


def test_flexdict_getattr_missing_raises_attributeerror():
    fd = FlexDict()
    with pytest.raises(AttributeError):
        _ = fd.nope


@pytest.mark.parametrize(
    "key_in, key_expected",
    [
        ("column%%stat", "column"),
        ("name**meta", "name"),
        ("a&&b", "a"),
        ("pipe||seg", "pipe"),
        ("money$$val", "money"),
    ],
)
def test_flexdict_setattr_special_symbol_truncates_key(key_in, key_expected):
    fd = FlexDict()
    setattr(fd, key_in, 42)
    assert key_expected in fd
    assert fd[key_expected] == 42
    # original key must not exist
    assert key_in not in fd


def test_flexdict_setstate_restores_and_binds_dict():
    fd = FlexDict()
    fd.__setstate__({"a": 1, "b": 2})
    assert fd.a == 1 and fd.b == 2
    # __dict__ should be the mapping itself for attr passthrough
    fd.new_key = 3
    assert fd["new_key"] == 3
