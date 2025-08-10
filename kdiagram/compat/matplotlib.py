
"""
A compatibility module to handle API changes across different
versions of Matplotlib.
"""
import warnings
import matplotlib
from packaging.version import parse

# Get the installed Matplotlib version
_MPL_VERSION = parse(matplotlib.__version__)


__all__= ["get_cmap", "is_valid_cmap"] 

def _get_cmap(name="viridis", lut=None):
    """
    A compatibility wrapper for getting a colormap.

    Handles the deprecation of `matplotlib.cm.get_cmap` in favor of
    `matplotlib.colormaps.get()` in Matplotlib v3.7+.

    Args:
        name (str, optional): The name of the colormap. Defaults to "viridis".
        lut (int, optional): The number of colors in the lookup table.

    Returns:
        A Matplotlib colormap instance.
    """
    # The new API was introduced in 3.6 but get_cmap was deprecated in 3.7.
    # We check for >= 3.6 for safety.
    if _MPL_VERSION >= parse("3.6"):
        # Use the new, recommended API
        return matplotlib.colormaps.get(name, None)
    else:
        # Use the old, deprecated API for older versions
        return matplotlib.cm.get_cmap(name, lut)


def get_cmap(name="viridis", lut=None):
    """
    A compatibility wrapper for getting a colormap that consistently
    raises a ValueError for invalid names across Matplotlib versions.

    Handles the deprecation of `matplotlib.cm.get_cmap` in favor of
    `matplotlib.colormaps.get()` in Matplotlib v3.7+.

    Args:
        name (str, optional): The name of the colormap. Defaults to "viridis".
        lut (int, optional): The number of colors in the lookup table.
                             (Note: This is ignored in Matplotlib >= 3.6)

    Returns:
        A Matplotlib colormap instance.

    Raises:
        ValueError: If the colormap name is not valid.
    """
    if _MPL_VERSION >= parse("3.6"):
        # The new API raises KeyError for invalid names. We catch it and
        # re-raise as ValueError to match the old API's behavior. This
        # ensures that try/except ValueError blocks in the main code
        # continue to work as expected.
        try:
            return matplotlib.colormaps.get(name)
        except ( TypeError, KeyError):
            # Re-raise as ValueError for consistent exception handling.
            raise ValueError(f"'{name}' is not a valid colormap name or alias")
    else:
        # The old API raises ValueError directly, so no change is needed here.
        return matplotlib.cm.get_cmap(name, lut)



def is_valid_cmap(cmap, default="viridis", error="raise"):
    """
    Checks if a colormap name is valid and handles invalid names gracefully.

    This is the recommended utility for validating user-provided colormap names
    before they are used in a plot.

    Args:
        cmap (str): The name of the colormap to validate.
        default (str, optional): The colormap name to return if `cmap` is invalid
            and `error` is not 'raise'. Defaults to "viridis".
        error (str, optional): How to handle an invalid `cmap` name.
            - 'raise': Raise a ValueError.
            - 'warn': Issue a UserWarning and return the `default` value (default).
            - 'ignore': Silently return the `default` value.

    Returns:
        str: A valid colormap name (either the original `cmap` or the `default`).

    Raises:
        ValueError: If `cmap` is invalid and `error` is 'raise'.
        ValueError: If `error` has an invalid value.
    """
    valid_cmaps = set()
    if _MPL_VERSION >= parse("3.6"):
        # New API uses matplotlib.colormaps registry (a dict-like object)
        valid_cmaps = set(matplotlib.colormaps)
    else:
        # Old API used matplotlib.cm.cmap_d dictionary
        valid_cmaps = set(matplotlib.cm.cmap_d)

    if cmap in valid_cmaps:
        return cmap

    # Handle invalid cmap based on the 'error' parameter
    if error == "raise":
        raise ValueError(f"'{cmap}' is not a valid colormap name or alias.")
    elif error == "warn":
        warnings.warn(
            f"Invalid `cmap` name '{cmap}'. Falling back to '{default}'.",
            UserWarning,
            stacklevel=2,  # Point to the function that called this one
        )
        return default
    elif error == "ignore":
        return default
    else:
        raise ValueError(
            f"Invalid value for 'error' parameter: '{error}'. "
            "Must be one of ['raise', 'warn', 'ignore']."
        )
