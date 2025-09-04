#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, cast

# Python 3.8+: importlib.metadata is in the stdlib; provide robust fallback
try:  # Python 3.8+
    from importlib import metadata as ilmd
except Exception:  # pragma: no cover
    import importlib_metadata as ilmd  # type: ignore

# Optional but preferred for PEP 440 version parsing
try:
    from packaging.version import InvalidVersion, Version
    from packaging.version import parse as parse_version
except Exception:  # pragma: no cover
    Version = None  # type: ignore
    InvalidVersion = Exception  # type: ignore

T = TypeVar("T")

# Cache checks across decorated call sites so we don't re-import or re-pip
# for the same requirement repeatedly in one process.
_REQUIREMENT_CACHE: dict[
    tuple[str, str | None, str | None], tuple[bool, str]
] = {}


def _vprint(level: int, verbose: int, *msg: Any) -> None:
    """Internal verbosity-controlled print."""
    if verbose >= level:
        print(*msg, file=sys.stderr)


def _best_base_name(import_target: str) -> str:
    """Resolve a reasonable base distribution name from an import target."""
    # Heuristic: first segment of dotted import
    # path (e.g., "matplotlib.pyplot" -> "matplotlib")
    return import_target.split(".", 1)[0]


def _get_installed_version(
    module: Any,
    import_target: str,
    dist_name: str | None,
    verbose: int,
) -> str | None:
    """
    Try to discover the installed version string.

    Priority:
    1) importlib.metadata.version(dist_name or base)
    2) module.__version__ (common pattern)
    """
    base = dist_name or _best_base_name(import_target)
    try:
        ver = ilmd.version(base)  # authoritative if distribution name matches
        if ver:
            return ver
    except ilmd.PackageNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover
        _vprint(2, verbose, f"[ensure_pkg] metadata.version error: {exc!r}")

    ver = getattr(module, "__version__", None)
    if isinstance(ver, str):
        return ver
    return None


def _version_satisfies(
    installed: str,
    minimum: str,
) -> bool:
    """
    Compare versions. Use packaging if available; otherwise fall back to
    a very lenient string comparison (not fully PEP 440 compliant).
    """
    if Version is not None:
        try:
            return parse_version(installed) >= parse_version(minimum)
        except InvalidVersion:
            # If odd formats are hit, fall back to string compare
            pass
    # Fallback: naive compare (best effort, not PEP 440)
    return installed >= minimum


def _pip_install(
    spec: str,
    verbose: int,
) -> tuple[bool, str]:
    """Attempt to 'pip install --upgrade {spec}'. Returns (ok, message)."""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", spec]
    _vprint(1, verbose, f"[ensure_pkg] Attempting: {' '.join(cmd)}")
    try:
        if verbose == 0:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        else:
            proc = subprocess.run(cmd, check=True)
        return proc.returncode == 0, "pip install succeeded"
    except subprocess.CalledProcessError as exc:
        return False, f"pip install failed: {exc}"
    except Exception as exc:  # pragma: no cover
        return False, f"pip install error: {exc}"


def is_pkg_installed(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except Exception:
        return False


def ensure_pkg(  # noqa: D401
    name: str,
    *,
    extra: str = "",
    dist_name: str | None = None,
    min_version: str | None = None,
    auto_install: bool = False,
    use_conda: bool = False,  # placeholder, not used
    errors: str = "raise",
    verbose: int = 1,
) -> Callable[[T], T]:
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("errors must be one of {'raise','warn','ignore'}")

    base = dist_name or _best_base_name(name)
    cache_key = (name, dist_name, min_version)

    def _ensure_once() -> tuple[bool, str]:
        # Use cached result if available
        if cache_key in _REQUIREMENT_CACHE:
            ok, msg = _REQUIREMENT_CACHE[cache_key]
            return ok, msg

        # 1) Try to import the target
        try:
            mod = importlib.import_module(name)
            _vprint(2, verbose, f"[ensure_pkg] Imported '{name}' OK")
        except Exception as exc:
            _vprint(
                1, verbose, f"[ensure_pkg] Import failed for '{name}': {exc}"
            )
            if auto_install:
                spec = f"{base}>={min_version}" if min_version else base
                ok, msg = _pip_install(spec, verbose)
                if ok:
                    try:
                        mod = importlib.import_module(name)
                        _vprint(
                            1,
                            verbose,
                            f"[ensure_pkg] Re-imported '{name}' OK",
                        )
                    except Exception as exc2:
                        msg = f"Re-import after install failed: {exc2}"
                        _REQUIREMENT_CACHE[cache_key] = (False, msg)
                        return False, msg
                else:
                    _REQUIREMENT_CACHE[cache_key] = (False, msg)
                    return False, msg
            else:
                msg = f"Cannot import '{name}'. {extra}".strip()
                _REQUIREMENT_CACHE[cache_key] = (False, msg)
                return False, msg

        # 2) Version check if requested
        if min_version:
            installed_ver = _get_installed_version(
                mod, name, dist_name, verbose
            )
            if not installed_ver:
                _vprint(
                    1,
                    verbose,
                    f"[ensure_pkg] Could not determine version for '{base}'.",
                )
            else:
                if not _version_satisfies(installed_ver, min_version):
                    _vprint(
                        1,
                        verbose,
                        f"[ensure_pkg] '{base}' version {installed_ver} "
                        f"< required {min_version}",
                    )
                    if auto_install:
                        spec = f"{base}>={min_version}"
                        ok, msg = _pip_install(spec, verbose)
                        if ok:
                            try:
                                # Re-import and re-validate version
                                mod = importlib.import_module(name)
                                installed_ver = _get_installed_version(
                                    mod, name, dist_name, verbose
                                )
                                if installed_ver and _version_satisfies(
                                    installed_ver, min_version
                                ):
                                    _REQUIREMENT_CACHE[cache_key] = (
                                        True,
                                        f"{base} upgraded to {installed_ver}",
                                    )
                                    return (
                                        True,
                                        f"{base} upgraded to {installed_ver}",
                                    )
                                else:
                                    msg = (
                                        f"After upgrade, '{base}' version still "
                                        f"insufficient (found {installed_ver!r})."
                                    )
                                    _REQUIREMENT_CACHE[cache_key] = (
                                        False,
                                        msg,
                                    )
                                    return False, msg
                            except Exception as exc2:
                                msg = (
                                    f"Re-import after upgrade failed: {exc2}"
                                )
                                _REQUIREMENT_CACHE[cache_key] = (False, msg)
                                return False, msg
                        else:
                            _REQUIREMENT_CACHE[cache_key] = (False, msg)
                            return False, msg
                    msg = (
                        f"'{base}' version {installed_ver} is below required "
                        f"{min_version}. {extra}"
                    ).strip()
                    _REQUIREMENT_CACHE[cache_key] = (False, msg)
                    return False, msg

        # Success
        _REQUIREMENT_CACHE[cache_key] = (True, f"Requirement OK: {base}")
        return True, f"Requirement OK: {base}"

    def _handle_failure(ok: bool, msg: str) -> None:
        if ok:
            return
        if errors == "raise":
            raise ImportError(msg)
        elif errors == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        # errors == "ignore": do nothing

    def _wrap_function(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ok, msg = _ensure_once()
            _handle_failure(ok, msg)
            try:
                return func(*args, **kwargs)
            except ModuleNotFoundError as e:
                root = (e.name or "").split(".")[0]
                if root == (dist_name or _best_base_name(name)):
                    _handle_failure(
                        False, f"Cannot import '{name}'. {extra}".strip()
                    )
                raise

        return wrapper

    def _wrap_class(cls: type) -> type:
        # Create a thin subclass that performs the check in __new__
        # (earlier than __init__, so heavy __init__ work won't run
        # if requirement is missing).
        class _Wrapped(cls):  # type: ignore[misc, valid-type]
            def __new__(inner_cls, *args: Any, **kwargs: Any):  # type: ignore[override]
                ok, msg = _ensure_once()
                _handle_failure(ok, msg)
                return cast(type, super()).__new__(inner_cls)

        # Preserve identity
        _Wrapped.__name__ = cls.__name__
        _Wrapped.__qualname__ = cls.__qualname__
        _Wrapped.__module__ = cls.__module__
        _Wrapped.__doc__ = cls.__doc__
        return _Wrapped

    def decorator(obj: T) -> T:
        # Class or function/method?
        if isinstance(obj, type):
            return cast(T, _wrap_class(obj))
        elif callable(obj):
            return cast(T, _wrap_function(cast(Callable[..., Any], obj)))
        else:  # pragma: no cover
            raise TypeError(
                "@ensure_pkg can decorate classes or callables only."
            )

    return decorator


ensure_pkg.__doc__ = r"""
Robust decorator that ensures a dependency is importable before
executing a function, method, or constructing a class.  It can
optionally enforce a minimum version and attempt a one-shot
installation via ``pip`` when asked.

Parameters
----------
name : str
    Canonical import target.  For example, ``"scipy"`` or
    ``"matplotlib.pyplot"``.  The first segment of the dotted
    path is used as a fallback distribution name when
    ``dist_name`` is not provided.
extra : str, optional
    Additional text appended to error or warning messages.
    Useful to hint users about alternative installs, platform
    wheels, or GPU variants.  Default is ``""``.
dist_name : str or None, optional
    The distribution name as known to the package manager and
    importlib metadata (often the PyPI name).  When ``None``,
    the first segment of ``name`` is used.  This matters for
    projects whose import path differs from the distribution
    name (e.g., ``import skimage`` vs. package ``scikit-image``).
min_version : str or None, optional
    PEP 440 compliant minimum version string.  When provided,
    the installed version must satisfy ``>= min_version``.
auto_install : bool, optional
    If ``True``, perform a single attempt to install or upgrade
    the requirement using ``pip`` in the current Python
    interpreter.  When ``min_version`` is given, the spec
    ``"{dist_name}>={min_version}"`` is used; otherwise
    ``"{dist_name}"``.  Default is ``False``.
use_conda : bool, optional
    Reserved for a future conda-forge backend.  Currently
    ignored.  Default is ``False``.
errors : {'raise', 'warn', 'ignore'}, optional
    Behavior when the requirement cannot be satisfied after any
    attempted install:
    - ``'raise'`` : raise ``ImportError`` (or runtime check error).
    - ``'warn'``  : emit a warning and proceed anyway.
    - ``'ignore'``: proceed silently.
    Default is ``'raise'``.
verbose : int, optional
    Verbosity level.  ``0`` → silent, ``1`` → brief log,
    ``>=2`` → chatty.  Logging is written to ``stderr``.
    Default is ``1``.

Returns
-------
Callable
    A decorator that can wrap a function, method, or class.  For
    functions and methods the check runs on call.  For classes,
    the check runs before instantiation, so heavy ``__init__``
    logic is not executed when requirements are missing.

Raises
------
ImportError
    If ``errors='raise'`` and the package is missing or does not
    meet the required version, after any attempted installation.
RuntimeError
    Implementations may raise runtime errors on unexpected
    installer failures or metadata resolution problems.

Warns
-----
RuntimeWarning
    If ``errors='warn'`` and the requirement is not satisfied,
    a warning is emitted and execution continues.

Notes
-----
- **Scope and caching.**  Requirement checks may be cached per
  ``(name, dist_name, min_version)`` tuple to avoid repeated
  imports and installer calls within the same process.
- **Import vs distribution names.**  Some projects have an
  import path that differs from the distribution name on PyPI
  or conda (e.g., ``skimage`` vs ``scikit-image``).  Use
  ``dist_name`` to disambiguate.
- **Version parsing.**  When available, ``packaging.version``
  is used for PEP 440 compliant comparison [1]_.  Otherwise a
  best-effort fallback may be used and could be less strict.
- **Auto-install caveats.**  Installing at runtime changes the
  current environment and may require network access, proper
  permissions, and an isolated virtual environment.  Consider
  pinning dependencies in your project config rather than
  relying on runtime installation for production systems.
- **Security.**  Enabling ``auto_install=True`` executes the
  package manager in the current process.  Audit inputs and
  prefer vetted indices and constraints files.  Use with care
  in multi-tenant or sandboxed contexts.
- **Conda environments.**  ``use_conda`` is accepted for API
  stability but ignored.  Prefer managing conda packages with
  environment files and dedicated tooling until a conda backend
  is provided.

Examples
--------
Basic usage to gate a function on ``numpy`` 1.23 or newer::
    
    >>> @ensure_pkg("numpy", min_version="1.23")
    ... def as_array(x):
    ...     import numpy as np
    ...     return np.asarray(x)

Warn but continue when ``pandas`` is missing or too old::

    >>> @ensure_pkg("pandas", min_version="2.0", errors="warn")
    ... def summarize(df):
    ...     import pandas as pd
    ...     return df.describe()

Attempt a one-shot install or upgrade via ``pip``::

    >>> @ensure_pkg("scipy", min_version="1.11",
    ...             auto_install=True, errors="raise")
    ... def needs_scipy():
    ...     from scipy import stats
    ...     return stats.norm.cdf(0.0)

Gate a class before heavy initialization runs::

    >>> @ensure_pkg("matplotlib", errors="raise")
    ... class Plotter:
    ...     def __init__(self):
    ...         import matplotlib.pyplot as plt
    ...         self._plt = plt
    ...     def show_point(self, x, y):
    ...         self._plt.plot([x], [y], "o")
    ...         self._plt.show()

Disambiguate import target vs distribution name::

    >>> @ensure_pkg("skimage", dist_name="scikit-image",
    ...             min_version="0.22")
    ... def do_vision(img):
    ...     from skimage import filters
    ...     return filters.sobel(img)

See Also
--------
importlib.import_module : Programmatic import by module path.
importlib.metadata.version : Query installed distribution version.
packaging.version.Version : PEP 440 compatible version objects.

References
----------
.. [1] PEP 440 — Version Identification and Dependency
       Specification.  Python Enhancement Proposals.
.. [2] importlib.metadata — Accessing package metadata.  Python
       Standard Library.
.. [3] Packaging library — Version parsing and comparisons,
       https://packaging.pypa.io/
.. [4] pip User Guide — Installing packages,
       https://pip.pypa.io/

"""
