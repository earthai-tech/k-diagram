#   License: Apache 2.0
#   Author: LKouadio <etanoyau@gmail.com>

"""k-diagram configs."""

import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from re import Pattern
from typing import Optional, Union

# Map string names to warning classes for convenience
_WARNING_NAME_MAP = {
    "Warning": Warning,
    "UserWarning": UserWarning,
    "DeprecationWarning": DeprecationWarning,
    "FutureWarning": FutureWarning,
    "SyntaxWarning": SyntaxWarning,
    "RuntimeWarning": RuntimeWarning,
    "ImportWarning": ImportWarning,
    "PendingDeprecationWarning": PendingDeprecationWarning,
    "ResourceWarning": ResourceWarning,
}


def _resolve_category(cat: Union[str, type]) -> type:
    if isinstance(cat, str):
        try:
            return _WARNING_NAME_MAP[cat]
        except KeyError as err:
            raise ValueError(f"Unknown warning category name: {cat!r}") from err
    if isinstance(cat, type) and issubclass(cat, Warning):
        return cat
    raise TypeError(f"Category must be a Warning subclass or name, got {cat!r}")


def configure_warnings(
    action: str = "default",
    *,
    categories: Optional[Iterable[Union[str, type]]] = None,
    modules: Optional[Iterable[Union[str, Pattern[str]]]] = None,
    clear: bool = False,
) -> None:
    """
    Configure warning filters for callers (tests/docs/apps). No effect unless called.

    Parameters
    ----------
    action : {"default","ignore","error","always","module","once"}, default "default"
        The warnings action to apply.
    categories : iterable of {Warning subclass or name}, optional
        Which categories to target (e.g., ["SyntaxWarning", FutureWarning]).
        If omitted, applies to the base Warning class (all warnings).
    modules : iterable of str or compiled regex, optional
        Restrict to warnings emitted by modules matching these regex patterns.
        Example: modules=[r"^numpy\\.", r"^matplotlib\\."]
    clear : bool, default False
        If True, first reset all existing filters (warnings.resetwarnings()).

    Examples
    --------
    >>> # Silence syntax warnings only (e.g., noisy on Linux CI)
    >>> from kdiagram import configure_warnings
    >>> configure_warnings("ignore", categories=["SyntaxWarning"])

    >>> # Treat deprecations as errors in tests
    >>> configure_warnings("error", categories=[DeprecationWarning])

    >>> # Ignore runtime warnings coming from numpy.*
    >>> configure_warnings("ignore", categories=["RuntimeWarning"],
    ...                    modules=[r"^numpy\\."])
    """
    if clear:
        warnings.resetwarnings()

    cats = list(categories) if categories is not None else [Warning]
    mods = list(modules) if modules is not None else [None]

    for cat in cats:
        resolved = _resolve_category(cat)
        for mod in mods:
            # warnings.filterwarnings accepts module as a regex string (or None)
            if mod is None:
                warnings.filterwarnings(action, category=resolved)
            else:
                warnings.filterwarnings(action, category=resolved, module=mod)


@contextmanager
def warnings_config(*args, **kwargs):
    """
    Context manager that temporarily applies `configure_warnings(...)`.

    Example
    -------
    >>> from kdiagram import warnings_config
    >>> with warnings_config("ignore", categories=["SyntaxWarning"]):
    ...     # code that would emit SyntaxWarning
    ...     pass
    """
    with warnings.catch_warnings():
        warnings.simplefilter("default")  # start from a known state
        configure_warnings(*args, **kwargs)
        yield


# Backward-compatible shim for the old API (now opt-in & deprecated)
def suppress_warnings(suppress: bool = True) -> None:
    """
    DEPRECATED: Use `configure_warnings(...)` instead.

    - suppress=True  -> ignore SyntaxWarning (was the previous default)
    - suppress=False -> restore default handling for SyntaxWarning
    """
    warnings.warn(
        "suppress_warnings() is deprecated; use configure_warnings(action='ignore', "
        "categories=['SyntaxWarning']) or the warnings_config(...) context manager.",
        DeprecationWarning,
        stacklevel=2,
    )
    if suppress:
        configure_warnings("ignore", categories=["SyntaxWarning"])
    else:
        configure_warnings("default", categories=["SyntaxWarning"])
