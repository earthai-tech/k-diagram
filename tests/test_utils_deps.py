from __future__ import annotations

import types
import warnings

import pytest

import kdiagram.utils._deps as _deps
from kdiagram.utils.deps import ensure_pkg


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    """Reset requirement cache and common mocks between tests."""
    # Fresh cache for each test
    monkeypatch.setattr(_deps, "_REQUIREMENT_CACHE", {}, raising=True)
    # Default: subprocess.run should not be called unless explicitly expected
    calls = {"run": 0, "last_cmd": None}

    def _no_run(*args, **kwargs):
        calls["run"] += 1
        calls["last_cmd"] = args[0] if args else None
        raise AssertionError("subprocess.run was called unexpectedly")

    monkeypatch.setattr(_deps.subprocess, "run", _no_run, raising=True)
    yield


def make_dummy_module(version: str | None = None):
    """Create a simple dummy module-like object."""
    m = types.SimpleNamespace()
    if version is not None:
        m.__version__ = version
    return m


def test_function_runs_when_dependency_available(monkeypatch):
    calls = {"import": 0, "ver": 0}

    def fake_import(name):
        calls["import"] += 1
        assert name == "numpy"
        return make_dummy_module("1.24.0")

    def fake_version(dist):
        calls["ver"] += 1
        assert dist == "numpy"
        return "1.24.0"

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )
    monkeypatch.setattr(_deps.ilmd, "version", fake_version, raising=True)

    @ensure_pkg("numpy", min_version="1.23", errors="raise", verbose=0)
    def f():
        return "ok"

    assert f() == "ok"
    assert calls["import"] == 1
    # version may be queried 1x (ok if 0 if __version__ path used)
    assert calls["ver"] in (0, 1)


def test_missing_dependency_raises_without_auto_install(monkeypatch):
    def fake_import(name):
        raise ImportError("nope")

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )

    @ensure_pkg("pandas", errors="raise", verbose=0)
    def f():
        return 42

    with pytest.raises(ImportError, match="Cannot import 'pandas'"):
        _ = f()


def test_missing_dependency_warns_and_continues_with_errors_warn(monkeypatch):
    def fake_import(name):
        raise ImportError("nope")

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )

    @ensure_pkg("pandas", errors="warn", verbose=0)
    def f():
        # Body does not import pandas; decorator only warns
        return 7

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = f()
        assert out == 7
        assert any("Cannot import 'pandas'" in str(ww.message) for ww in w)


def test_auto_install_attempts_pip_then_still_fails(monkeypatch):
    # Always fail import
    def fake_import(name):
        raise ImportError("still missing")

    # Pretend pip succeeds but import still fails
    class DummyProc:
        returncode = 0

    recorded = {"run_calls": 0, "last_cmd": None}

    def fake_run(cmd, check=True, **kwargs):
        recorded["run_calls"] += 1
        recorded["last_cmd"] = cmd
        return DummyProc()

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )
    monkeypatch.setattr(_deps.subprocess, "run", fake_run, raising=True)

    @ensure_pkg("scikit-learn", auto_install=True, errors="raise", verbose=0)
    def f():
        return "won't reach"

    with pytest.raises(ImportError):
        _ = f()

    assert recorded["run_calls"] == 1

    # Expect: [python, -m, pip, install, --upgrade, <spec>]
    assert recorded["last_cmd"][:5] == [
        _deps.sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
    ]
    assert recorded["last_cmd"][-1] == "scikit-learn"


def test_auto_install_on_version_too_low_then_succeeds(monkeypatch):
    # First import returns old version; after "upgrade" return new version
    state = {"stage": "old"}

    def fake_import(name):
        if state["stage"] == "old":
            return make_dummy_module("1.0.0")
        return make_dummy_module("2.1.0")

    def fake_version(dist):
        # Track with dist_name's first-segment ("scipy")
        return "1.0.0" if state["stage"] == "old" else "2.1.0"

    class DummyProc:
        returncode = 0

    def fake_run(cmd, check=True, **kwargs):
        # Simulate install success and flip stage
        state["stage"] = "new"
        return DummyProc()

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )
    monkeypatch.setattr(_deps.ilmd, "version", fake_version, raising=True)
    monkeypatch.setattr(_deps.subprocess, "run", fake_run, raising=True)

    @ensure_pkg(
        "scipy", min_version="2.0.0", auto_install=True, errors="raise"
    )
    def f():
        return "ok-after-upgrade"

    assert f() == "ok-after-upgrade"


def test_version_too_low_warns_when_errors_warn(monkeypatch):
    def fake_import(name):
        return make_dummy_module("1.0.0")

    def fake_version(dist):
        return "1.0.0"

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )
    monkeypatch.setattr(_deps.ilmd, "version", fake_version, raising=True)

    @ensure_pkg("xpkg", min_version="3.0.0", errors="warn", verbose=0)
    def f():
        return "continued"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert f() == "continued"
        assert any(
            "version 1.0.0 is below required 3.0.0" in str(ww.message)
            for ww in w
        )


def test_class_gate_prevents_init_on_failure(monkeypatch):
    def fake_import(name):
        raise ImportError("no class deps")

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )

    side_effect = {"init_called": False}

    @ensure_pkg("matplotlib", errors="raise", verbose=0)
    class C:
        def __init__(self):
            side_effect["init_called"] = True

    with pytest.raises(ImportError):
        _ = C()

    assert side_effect["init_called"] is False


def test_method_decorator_warns_and_continues(monkeypatch):
    def fake_import(name):
        raise ImportError("missing")

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )

    class K:
        @ensure_pkg("pandas", errors="warn", verbose=0)
        def m(self):
            return "ran"

    k = K()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert k.m() == "ran"
        assert any("Cannot import 'pandas'" in str(ww.message) for ww in w)


def test_caching_avoids_repeated_import(monkeypatch):
    calls = {"import": 0}

    def fake_import(name):
        calls["import"] += 1
        return make_dummy_module("1.2.3")

    def fake_version(dist):
        return "1.2.3"

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )
    monkeypatch.setattr(_deps.ilmd, "version", fake_version, raising=True)

    @ensure_pkg("numpy", min_version="1.0.0", errors="raise", verbose=0)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f(2) == 3
    # Only one import due to cache
    assert calls["import"] == 1


def test_dist_name_is_used_for_metadata_lookup(monkeypatch):
    # import target 'skimage', dist_name 'scikit-image'
    def fake_import(name):
        assert name == "skimage"
        return make_dummy_module("0.21.0")

    seen = {"dist_queried": None}

    def fake_version(dist):
        seen["dist_queried"] = dist
        # Return version associated with dist_name, not import name
        assert dist == "scikit-image"
        return "0.21.0"

    monkeypatch.setattr(
        _deps.importlib, "import_module", fake_import, raising=True
    )
    monkeypatch.setattr(_deps.ilmd, "version", fake_version, raising=True)

    @ensure_pkg(
        "skimage",
        dist_name="scikit-image",
        min_version="0.20.0",
        errors="raise",
    )
    def f():
        return "ok"

    assert f() == "ok"
    assert seen["dist_queried"] == "scikit-image"
