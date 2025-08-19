# File: kdiagram/compat/sklearn.py
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

"""
A compatibility module to handle API changes across different
versions of Matplotlib.
"""

import warnings

import matplotlib
from packaging.version import parse

# Get the installed Matplotlib version
_MPL_VERSION = parse(matplotlib.__version__)


__all__ = ["get_cmap", "is_valid_cmap"]


def _get_cmap(name="viridis", lut=None, allow_none=False):
    """Get a Matplotlib colormap with version compatibility.

    This function acts as a robust wrapper to retrieve a colormap
    object, ensuring consistent behavior across different
    Matplotlib versions. It handles the API change from
    `matplotlib.cm.get_cmap` to `matplotlib.colormaps.get`.

    Parameters
    ----------
    name : str or None, optional
        The name of the colormap to retrieve. Defaults to "viridis".
    lut : int, optional
        The number of colors in the lookup table. This parameter
        is only used for Matplotlib versions older than 3.6 and
        is ignored in newer versions. Defaults to None.
    allow_none : bool, optional
        If True, passing `name=None` will return `None` without
        raising an error. If False (default), passing `name=None`
        will raise a `ValueError`. This is the primary control for
        preventing unexpected `TypeError` exceptions downstream.

    Returns
    -------
    matplotlib.colors.Colormap or None
        The requested colormap instance. Returns `None` only if
        `name` is None and `allow_none` is True.

    Raises
    ------
    ValueError
        If `name` is `None` and `allow_none` is False, if `name` is
        not a string (and not None), or if the named colormap
        does not exist.
    """
    if name is None:
        if allow_none:
            return None
        else:
            raise ValueError(
                "Colormap `name` cannot be None when `allow_none` is False."
            )

    if not isinstance(name, str):
        raise ValueError(f"Colormap name must be a string, not {type(name)}.")

    if _MPL_VERSION >= parse("3.6"):
        # The new API raises KeyError for invalid names. We catch it and
        # re-raise as ValueError to match the old API's behavior. This
        # ensures that try/except ValueError blocks in the main code
        # continue to work as expected.

        try:
            result = matplotlib.colormaps.get(name)
            if result is None and not allow_none:
                raise KeyError
            else:
                return result

        except (TypeError, KeyError, ValueError) as err:
            raise ValueError(
                f"'{name}' is not a valid colormap name."
            ) from err
    else:
        # The old API raises ValueError for invalid names.

        return matplotlib.cm.get_cmap(name, lut)


def _is_valid_cmap(cmap, default="viridis", error="raise"):
    """Legacy check if a colormap name is valid by attempting to retrieve it.

    This utility leverages the robust `get_cmap` wrapper to handle
    different Matplotlib versions and input types seamlessly.

    Parameters
    ----------
    cmap : str
        The name of the colormap to validate.
    default : str, optional
        The colormap name to return if `cmap` is found to be
        invalid. Defaults to "viridis".
    error : {'raise', 'warn', 'ignore'}, optional
        Defines how to handle an invalid `cmap` name.
        - 'raise': Raise a `ValueError` (default).
        - 'warn': Issue a `UserWarning` and return the `default`.
        - 'ignore': Silently return the `default` value.

    Returns
    -------
    str
        A valid colormap name, which is either the original `cmap`
        or the `default` value if `cmap` was invalid.

    Raises
    ------
    ValueError
        If `cmap` is invalid and `error` is 'raise', or if the
        `error` parameter itself is given an invalid value.
    """
    try:
        # We call get_cmap with its safe default (allow_none=False)
        # because this function's purpose is to validate names.
        # It will raise a ValueError for any invalid input.
        cmap = _get_cmap(cmap, allow_none=False)
        return cmap
    except ValueError as err:
        # The name is invalid, so handle it based on the 'error' flag.
        if error == "raise":
            raise ValueError(
                f"'{cmap}' is not a valid colormap name."
            ) from err
        elif error == "warn":
            warnings.warn(
                f"Invalid `cmap` name '{cmap}'. Falling back to '{default}'.",
                UserWarning,
                stacklevel=2,
            )
            return default
        elif error == "ignore":
            return default
        else:
            raise ValueError(
                "Invalid value for 'error' parameter. Must be 'raise', "
                "'warn', or 'ignore'."
            ) from err


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
