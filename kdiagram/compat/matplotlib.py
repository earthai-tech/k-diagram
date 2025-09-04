# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

"""
A compatibility module to handle API changes across different
versions of Matplotlib.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import matplotlib
import numpy as np
from packaging.version import parse

# Get the installed Matplotlib version
_MPL_VERSION = parse(matplotlib.__version__)


__all__ = ["get_cmap", "is_valid_cmap", "get_colors"]


def is_valid_cmap(cmap, allow_none=False, **kw):  # **for future extension
    r"""Check if a colormap identifier is valid.

    This function purely validates whether a given identifier can be
    resolved to a Matplotlib colormap. It does not retrieve the
    colormap object itself.

    Parameters
    ----------
    cmap : any
        The colormap identifier to validate, typically a string.
    allow_none : bool, optional
        If True, `None` is considered a valid input and the
        function will return True. If False (default), `None` is
        considered invalid.

    Returns
    -------
    bool
        True if the `cmap` is a valid, retrievable colormap name
        or if `cmap` is None and `allow_none` is True. Otherwise,
        returns False.
    """
    is_valid = False
    if cmap is None:
        return allow_none

    if not isinstance(cmap, str):
        return False

    try:
        # The most reliable way to check for existence is to try
        # getting it, using the modern API first.
        if _MPL_VERSION >= parse("3.6"):
            is_valid = matplotlib.colormaps.get(cmap)
        else:
            is_valid = matplotlib.cm.get_cmap(cmap)

        if is_valid is None:
            return False

        return True
    except (ValueError, KeyError):
        # ValueError is for old MPL, KeyError for new MPL.
        return False


def get_cmap(
    name,
    default="viridis",
    allow_none=False,
    error=None,
    failsafe="continuous",
    **kw,
):
    r"""Robustly retrieve a Matplotlib colormap with fallbacks.

    This function ensures a valid colormap object is always returned,
    preventing runtime errors from invalid names. It uses a
    cascading fallback system.

    Parameters
    ----------
    name : str or None
        The desired colormap name.
    default : str, optional
        The fallback colormap if `name` is invalid.
        Defaults to 'viridis'.
    allow_none : bool, optional
        If True, a `name` of `None` will return `None` without
        any warnings or errors. Defaults to False.
    failsafe : {'continuous', 'discrete'}, optional
        Specifies the type of ultimate fallback colormap to use if
        both `name` and `default` are invalid.
        - 'continuous': Use 'viridis' (default).
        - 'discrete': Use 'tab10'.

    Returns
    -------
    matplotlib.colors.Colormap or None
        A valid colormap instance, or `None` if `allow_none` is
        True and the input `name` is `None`.
    """
    result = None
    # For API consistency, acknowledge the old 'error' parameter
    # but inform the user that it's no longer used.
    if error is not None:
        warnings.warn(
            "The 'error' parameter is deprecated for get_cmap and is ignored. "
            "This function now always returns a valid colormap by using "
            "fallbacks.",
            FutureWarning,
            stacklevel=2,
        )

        # but does nothing

    # Private helper to prevent repeating the retrieval code
    def _retrieve(cmap_name):
        """Retrieves the colormap object using the correct API."""
        if _MPL_VERSION >= parse("3.6"):
            return matplotlib.colormaps.get(cmap_name)
        else:
            return matplotlib.cm.get_cmap(cmap_name)

    # 1. Handle explicit `None` input first.
    if name is None:
        if allow_none:
            return None
        # If None is not allowed, treat it as an invalid name
        # and proceed to the fallback logic below.
    # 2. Try to validate and retrieve the primary name.
    elif is_valid_cmap(name):
        result = _retrieve(name)

    if result is not None:
        return result
    # 3. If we are here, 'name' was invalid. Warn and fall back to default.
    warnings.warn(
        f"Colormap '{name}' not found. Falling back to default '{default}'.",
        UserWarning,
        stacklevel=2,
    )
    if is_valid_cmap(default):
        result = _retrieve(default)

    if result is not None:
        return result

    # apply failure safe here
    # 4. If the default is also invalid, warn and use the ultimate failsafe.
    # 4. If default is also invalid, determine and use the ultimate failsafe.
    if failsafe == "discrete":
        failsafe_cmap = "tab10"
    else:
        failsafe_cmap = "viridis"
        if failsafe != "continuous":
            warnings.warn(
                f"Invalid `failsafe` value '{failsafe}'. Defaulting to "
                f"'continuous' type ('{failsafe_cmap}').",
                UserWarning,
                stacklevel=2,
            )

    warnings.warn(
        f"Default colormap '{default}' also not found. "
        f"Falling back to failsafe '{failsafe_cmap}'.",
        UserWarning,
        stacklevel=2,
    )
    return _retrieve("viridis")


def get_colors(
    n: int,
    colors: str | Sequence | None = None,
    cmap: str | None = "viridis",
    *,
    default: str = "viridis",
    allow_none: bool = False,
    error: str | None = None,
    failsafe: str = "continuous",
    **kw: Any,
) -> list | None:
    """
    Return a list of exactly *n* colors with sensible fallbacks.

    Priority:
      1) If *colors* is a colormap name → sample that colormap.
      2) If *colors* is a single color → use it for the first item and
         pad remaining from a colormap.
      3) If *colors* is a sequence → use as many as provided, then pad
         from a colormap if shorter, or truncate if longer.
      4) Else → sample from *cmap* (if given) or *default*.

    Parameters
    ----------
    n : int
        The number of colors required.
    colors : str, list of str, optional
        A list of user-specified colors. Can also be a single color
        or a colormap name.
    cmap : str, optional
        A colormap name to sample from if no colors are provided.
    default : str, default="viridis"
        The fallback colormap name if `cmap` is invalid.
    allow_none : bool, default=False
        Whether to allow `None` as a valid return value if no valid
        colors are provided or found.
    error : str, optional
        Deprecated. This argument is no longer used.
    failsafe : str, default="continuous"
        If both `cmap` and `default` are invalid, this defines the
        fallback colormap type. Options are:
        - "continuous": Uses a continuous colormap (default).
        - "discrete": Uses a discrete colormap like `tab10`.

    Returns
    -------
    list
        A list of exactly `n` colors. The list may come from a colormap
        or from the user-specified colors, padded/truncated as needed.
    """
    if n <= 0:
        return [] if not allow_none else None

    # Retrieve colormap (either from cmap or default)
    cmap_obj = get_cmap(
        cmap,
        default=default,
        allow_none=True,
        error=error,
        failsafe=failsafe,
        **kw,
    )

    # Generate colors from the colormap
    _COLORS = cmap_obj(np.linspace(0, 1, n))

    # Case when colors is a sequence (either a list of colors or a single color)
    if colors is not None:
        # If colors is a single color, treat it as a list
        if isinstance(colors, str):
            colors = [colors]

        # Ensure that we have a valid sequence of colors
        user_colors = list(colors)
        # If the provided colors are fewer than `n`, pad from the colormap
        if len(user_colors) < n:
            pad = _COLORS[: n - len(user_colors)]
            return user_colors + list(pad)

        # If the provided colors are more than `n`, truncate them
        if len(user_colors) > n:
            return list(user_colors[:n])

        # If we have exactly `n` colors, return them
        return list(user_colors)

    # If no colors provided, sample from the colormap
    if allow_none and cmap_obj is None:
        return None
    return _COLORS.tolist()
