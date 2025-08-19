# File: kdiagram/datasets/_property.py
# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)
# -------------------------------------------------------------------
# Provides base I/O functions for dataset management (cache dir, download).
# Adapted or inspired by the 'gofast.datasets.io' module from the
# 'gofast' package: https://github.com/earthai-tech/gofast
# Original 'gofast' code licensed under BSD-3-Clause.
# Modifications and 'k-diagram' are under Apache License 2.0.
# -------------------------------------------------------------------
"""
Internal Dataset Storage and Retrieval Utilities
(:mod:`kdiagram.datasets._property`)
=================================================

This internal module provides base functions for managing the local
storage location (cache directory) for datasets used or downloaded by
`k-diagram`. It includes utilities to determine the data directory
path, remove cached data, and potentially download remote dataset
files based on predefined metadata.

These functions are typically intended for internal use by dataset
loading functions within the :mod:`kdiagram.datasets` subpackage and
are not guaranteed to have a stable API for end-users.
"""

from __future__ import annotations

import os
import shutil
import warnings
from collections import namedtuple
from importlib import resources  # Used in download_file_if # noqa
from urllib.parse import urljoin

# Assuming io utils are now one level up relative to datasets/_property.py
# Adjust if kdiagram.utils doesn't exist or io is elsewhere
try:
    from ..utils.io import check_file_exists, fancier_downloader
except ImportError:
    # Handle case where utils might not be structured like this yet
    # Or raise a more specific error if these are essential internal deps
    warnings.warn(
        "Could not import IO utilities from kdiagram.utils.io", stacklevel=2
    )

    # Define dummy functions if needed for static analysis, but runtime will fail
    def check_file_exists(*args, **kwargs):
        return False

    def fancier_downloader(*args, **kwargs):
        raise NotImplementedError


# TODO: Update if k-diagram will host data/descriptions
KD_DMODULE = "kdiagram.datasets.data"  # Path for potential packaged data
KD_DESCR = (
    "kdiagram.datasets.descr"  # Path for potential packaged descriptions
)
KD_REMOTE_DATA_URL = (  # Example URL if k-diagram hosts data samples
    "https://raw.githubusercontent.com/earthai-tech/k-diagram/main/"
    "kdiagram/datasets/data/"
)

# Define structure for remote dataset metadata
RemoteMetadata = namedtuple(
    "RemoteMetadata",
    ["file", "url", "checksum", "descr_module", "data_module"],
)


__all__ = [
    "KD_DMODULE",
    "KD_REMOTE_DATA_URL",
    "get_data",
    "remove_data",
]


# --- Function Definitions ---
def get_data(data_home: str | None = None) -> str:
    r"""Get the path to the k-diagram data cache directory.

    Determines the local directory path used for caching downloaded
    datasets or storing user-provided data relevant to k-diagram.
    The directory is created if it doesn't exist.

    The location defaults to ``~/kdiagram_data`` but can be overridden
    by setting the ``KDIAGRAM_DATA`` environment variable or by
    providing an explicit path to the `data_home` argument.

    Parameters
    ----------
    data_home : str, optional
        Explicit path to the desired data directory. If ``None``,
        checks the 'KDIAGRAM_DATA' environment variable, then falls
        back to ``~/kdiagram_data``. Tilde ('~') is expanded to the
        user's home directory. Default is ``None``.

    Returns
    -------
    data_dir : str
        The absolute path to the k-diagram data cache directory.

    Examples
    --------
    >>> from kdiagram.datasets._property import get_data # Use actual import
    >>> default_path = get_data()
    >>> print(f"Default data directory: {default_path}")
    >>> custom_path = get_data("/path/to/my/kdata")
    >>> print(f"Custom data directory: {custom_path}")
    """
    if data_home is None:
        # Check environment variable first
        data_home = os.environ.get(
            "KDIAGRAM_DATA", os.path.join("~", "kdiagram_data")
        )
    # Expand user path (~ character)
    data_home = os.path.expanduser(data_home)
    # Create directory if it doesn't exist
    try:
        os.makedirs(data_home, exist_ok=True)
    except OSError as e:
        # Handle potential permission errors, etc.
        warnings.warn(
            f"Could not create data directory {data_home}: {e}", stacklevel=2
        )
        # Optionally raise or return a default path if creation fails
    return data_home


def remove_data(data_home: str | None = None) -> None:
    r"""Delete the k-diagram data cache directory and its contents.

    Removes the entire directory specified by `data_home` (or the
    default k-diagram cache directory if `data_home` is ``None``).
    Use with caution, as this permanently deletes cached data.

    Parameters
    ----------
    data_home : str, optional
        The path to the k-diagram data directory to remove. If ``None``,
        locates the directory using :func:`get_data`.
        Default is ``None``.

    Returns
    -------
    None

    Examples
    --------
    >>> from kdiagram.datasets._property import remove_data, get_data
    >>> # To remove the default cache:
    >>> # remove_data()
    >>> # To remove a custom cache:
    >>> # custom_path = get_data("/path/to/my/kdata")
    >>> # remove_data(custom_path)
    """
    # Get the path to the data directory
    data_dir = get_data(data_home)
    # Remove the directory tree if it exists
    if os.path.exists(data_dir):
        print(f"Removing k-diagram data cache directory: {data_dir}")
        shutil.rmtree(data_dir)
    else:
        print(f"k-diagram data cache directory not found: {data_dir}")


def download_file_if(
    metadata: RemoteMetadata | str,  # Added Union for clarity
    data_home: str | None = None,
    download_if_missing: bool = True,
    force_download: bool = False,  # Added force_download
    error: str = "raise",
    verbose: bool = True,
) -> str | None:
    r"""Find, cache, or download a dataset file.

    Checks for a dataset file in sequence:
    1. Checks the installed package resources.
    2. Checks the local k-diagram data cache directory.
    3. Optionally downloads to the cache if missing from both.

    If the file is found in the package resources but not in the cache,
    it is copied to the cache directory for faster future access.

    Parameters
    ----------
    metadata : RemoteMetadata or str
        Metadata defining the remote file, or just the filename string.
        If RemoteMetadata, must contain at least ``file`` and ``url``
        attributes. It should also contain ``data_module`` (e.g.,
        'kdiagram.datasets.data') specifying the package location
        to check for bundled data. If a string (filename) is
        provided, default URL (`KD_REMOTE_DATA_URL`) and data module
        (`KD_DATA_MODULE`) are used.

    data_home : str, optional
        Path to the k-diagram data cache directory. If ``None``, uses
        the default location determined by :func:`get_data`.
        Default is ``None``.

    download_if_missing : bool, default=True
        If ``True``, attempt to download the dataset to the cache if
        it's not found in package resources or the local cache.

    force_download : bool, default=False
        If ``True``, forces a download attempt to the cache,
        overwriting any existing file in the cache. Checks package
        resources first only if download fails. Default is ``False``.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Determines behavior if the download fails:
        - ``'raise'``: Raises a RuntimeError.
        - ``'warn'``: Issues a warning and returns ``None``.
        - ``'ignore'``: Silently ignores the error and returns ``None``.

    verbose : bool, default=True
        If ``True``, prints status messages about checking locations,
        copying, or downloading the file.

    Returns
    -------
    filepath : str or None
        The absolute path to the validated local file (usually in the
        cache directory). Returns ``None`` if the file cannot be found
        or obtained based on the specified parameters.

    Raises
    ------
    ValueError
        If the `error` parameter is invalid, or if essential metadata
        (filename, URL, data_module) is missing for download/lookup.
    RuntimeError
        If the download fails and `error` is set to `'raise'`.
    TypeError
        If `metadata` is not a string or `RemoteMetadata` instance.
    """
    # 1. Validate inputs and resolve metadata
    if error not in ["warn", "raise", "ignore"]:
        raise ValueError(
            "`error` parameter must be 'raise', 'warn', or 'ignore'."
        )

    if isinstance(metadata, str):
        if not KD_REMOTE_DATA_URL or not KD_DMODULE:
            msg = (
                "Default remote URL or data module path not configured."
                " Cannot process file specified only by name."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                warnings.warn(msg, stacklevel=2)
            return None
        # Create metadata object from string filename
        filename = metadata
        meta = RemoteMetadata(
            file=filename,
            url=KD_REMOTE_DATA_URL,
            checksum=None,
            descr_module=None,
            data_module=KD_DMODULE,
        )
    elif isinstance(metadata, RemoteMetadata):
        meta = metadata
        filename = meta.file
        if not hasattr(meta, "data_module") or not meta.data_module:
            raise ValueError(
                "RemoteMetadata must include 'data_module' path."
            )
        if not hasattr(meta, "url") or not meta.url:
            raise ValueError("RemoteMetadata must include 'url'.")
    else:
        raise TypeError(
            "`metadata` must be a string (filename) or RemoteMetadata."
        )

    # 2. Determine Cache Path
    data_dir = get_data(data_home)  # User cache directory
    cache_filepath = os.path.join(data_dir, filename)

    # 3. Handle Forced Download
    if force_download:
        if download_if_missing:
            if verbose:
                print(
                    f"Forcing download attempt for '{filename}' to {data_dir}..."
                )
            dl_success = False
            try:
                # Construct full URL (ensure trailing slash on base URL)
                base_url = (
                    meta.url if meta.url.endswith("/") else meta.url + "/"
                )
                file_url = urljoin(base_url, filename)
                # Use fancier_downloader to download/move to cache
                fancier_downloader(
                    url=file_url,
                    filename=filename,  # Name to save as temporarily/finally
                    dstpath=data_dir,  # Target directory for move
                    check_size=True,  # Enable size check
                    error=error,  # Propagate error setting
                    verbose=verbose > 0,  # Pass verbosity flag
                )
                # Check if file now exists in cache after download attempt
                if os.path.exists(cache_filepath):
                    dl_success = True  # noqa
                    if verbose:
                        print(
                            f"Forced download successful: '{cache_filepath}'"
                        )
                    return cache_filepath  # Success!
                else:
                    # Should not happen if downloader worked & error!='raise'
                    msg = (
                        "Download function reported success (or ignored "
                        f"error) but file '{filename}' not found in cache."
                    )
                    if error == "raise":
                        raise RuntimeError(msg)
                    elif error == "warn":
                        warnings.warn(msg, stacklevel=2)
                    # Continue to check package resource as
                    # fallback only if download failed

            except Exception as e:
                # Handle exceptions from fancier_downloader
                dl_error_msg = (
                    f"Forced download failed for '{filename}'. Error: {e}"
                )
                if error == "raise":
                    raise RuntimeError(dl_error_msg) from e
                elif error == "warn":
                    warnings.warn(dl_error_msg, stacklevel=2)
                # Continue to check package resource as fallback
                # only if download failed
        else:
            # Cannot force download if download_if_missing is False
            warnings.warn(
                f"Cannot force download for '{filename}', "
                f"download_if_missing is False.",
                stacklevel=2,
            )
            # Fall through to check package/cache normally

    # 4. Check Package Resources First (if download wasn't forced or failed)
    package_filepath = None
    try:
        if resources.is_resource(meta.data_module, filename):
            if verbose:
                print(
                    f"Dataset '{filename}' found in package resource: "
                    f"{meta.data_module}"
                )
            # Get path via context manager
            with resources.path(meta.data_module, filename) as rpath:
                package_filepath = str(rpath)
    except (ModuleNotFoundError, TypeError, Exception) as e:
        # ModuleNotFoundError if data_module path is wrong
        # TypeError if non-string arguments
        # Catch broad Exception for other potential resource issues
        if verbose:
            warnings.warn(
                f"Could not check package resources for "
                f"'{meta.data_module}/{filename}': {e}",
                stacklevel=2,
            )

    # If found in package, copy to cache if not already there (or if download failed)
    if package_filepath and not os.path.exists(cache_filepath):
        if verbose:
            print(f"Copying dataset from package to cache: {cache_filepath}")
        try:
            os.makedirs(data_dir, exist_ok=True)  # Ensure cache dir exists
            shutil.copyfile(package_filepath, cache_filepath)
            return cache_filepath  # Return cache path after copying
        except Exception as copy_err:
            warnings.warn(
                f"Could not copy dataset from package to cache: "
                f"{copy_err}. Using package path directly.",
                stacklevel=2,
            )
            # Fallback to returning the package path if copy fails? Risky.
            # Better to return None or raise? Let's return cache path anyway
            #  if copy fails but file exists in package
            # return package_filepath # Use with caution if cache is preferred
            return (
                cache_filepath
                if os.path.exists(cache_filepath)
                else package_filepath
            )

    # If file was found in package AND already exists in cache, use cache path
    if package_filepath and os.path.exists(cache_filepath):
        if verbose:
            print(
                f"Using cached version (also found in package): {cache_filepath}"
            )
        return cache_filepath

    # 5. Check Cache Directory (if not found in package)
    if not package_filepath and os.path.exists(cache_filepath):
        if verbose:
            print(f"Dataset '{filename}' found in cache: {cache_filepath}")
        return cache_filepath

    # 6. Attempt Download (if not found in package or cache, and allowed)
    if download_if_missing:
        if verbose:
            print(
                f"Dataset '{filename}' not found in package or cache. "
                f"Attempting download to {data_dir}..."
            )
        try:
            # Construct full URL
            base_url = meta.url if meta.url.endswith("/") else meta.url + "/"
            file_url = urljoin(base_url, filename)
            # Download and move to cache
            fancier_downloader(
                url=file_url,
                filename=filename,
                dstpath=data_dir,
                check_size=True,
                error=error,
                verbose=verbose > 0,
            )
            # Check if successful
            if os.path.exists(cache_filepath):
                if verbose >= 2:
                    print(f"Download successful: '{cache_filepath}'")
                return cache_filepath
            else:  # Download failed silently (error='ignore' or 'warn')
                return None  # Return None as download wasn't successful
        except Exception as e:
            download_error_msg = (
                f"Failed to download '{filename}' from "
                f"'{file_url}'. Error: {e}"
            )
            if error == "raise":
                raise RuntimeError(download_error_msg) from e
            elif error == "warn":
                warnings.warn(download_error_msg, stacklevel=2)
            return None  # Download failed

    # 7. File Not Found and Download Disabled/Failed
    if verbose:
        print(f"Dataset '{filename}' could not be located.")
    return None
