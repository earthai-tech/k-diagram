# -*- coding: utf-8 -*-
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
# --- Module content starts here ---
from __future__ import annotations # Keep if present

import os
import shutil
import warnings
from importlib import resources # Used in download_file_if # noqa 
from collections import namedtuple
from typing import Optional
from urllib.parse import urljoin

# Assuming io utils are now one level up relative to datasets/_property.py
# Adjust if kdiagram.utils doesn't exist or io is elsewhere
try:
    from ..utils.io import check_file_exists, fancier_downloader
except ImportError:
    # Handle case where utils might not be structured like this yet
    # Or raise a more specific error if these are essential internal deps
    warnings.warn("Could not import IO utilities from kdiagram.utils.io")
    # Define dummy functions if needed for static analysis, but runtime will fail
    def check_file_exists(*args, **kwargs): return False
    def fancier_downloader(*args, **kwargs): raise NotImplementedError


# TODO: Update if k-diagram will host data/descriptions 
KD_DMODULE = "kdiagram.datasets.data" # Path for potential packaged data
KD_DESCR = "kdiagram.datasets.descr" # Path for potential packaged descriptions
KD_REMOTE_DATA_URL = ( # Example URL if k-diagram hosts data samples
    'https://raw.githubusercontent.com/earthai-tech/k-diagram/main/'
    'kdiagram/datasets/data/'
)

# Define structure for remote dataset metadata (if needed)
RemoteMetadata = namedtuple(
    "RemoteMetadata",
    ["file", "url", "checksum", "descr_module", "data_module"]
)


__all__ = [
    'KD_DMODULE', 
    'KD_REMOTE_DATA_URL', 
    'get_data', 
    'remove_data',
    
]

# --- Function Definitions ---

def get_data(data_home: Optional[str] = None) -> str:
    """Get the path to the k-diagram data cache directory.

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
        warnings.warn(f"Could not create data directory {data_home}: {e}")
        # Optionally raise or return a default path if creation fails
    return data_home

def remove_data(data_home: Optional[str] = None) -> None:
    """Delete the k-diagram data cache directory and its contents.

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

def download_file_if_missing(
    metadata: RemoteMetadata | str,
    data_home: Optional[str] = None,
    download_if_missing: bool = True,
    error: str = 'raise',
    verbose: bool = True
) -> Optional[str]:
    """Download and cache a remote file if not present locally.

    Checks if a file defined by `metadata` exists in the local
    k-diagram data cache directory (determined by `get_data`). If
    the file is missing and `download_if_missing` is True, it
    attempts to download it from the specified URL.

    Parameters
    ----------
    metadata : RemoteMetadata or str
        Metadata defining the remote file. Must contain at least
        `file` (filename) and `url` (base URL) attributes if a
        `RemoteMetadata` object. If a string is provided, it's
        treated as the filename, and the default module URL
        (`KD_REMOTE_DATA_URL`) is used.

    data_home : str, optional
        Path to the k-diagram data cache directory. If ``None``, uses
        the default location determined by :func:`get_data`.
        Default is ``None``.

    download_if_missing : bool, default=True
        If ``True``, attempt to download the file if it's not found
        in the local cache. If ``False``, only checks existence and
        returns the path if found, otherwise returns ``None``.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Determines behavior if the download fails:
        - ``'raise'``: Raises a RuntimeError.
        - ``'warn'``: Issues a warning and returns ``None``.
        - ``'ignore'``: Silently ignores the error and returns ``None``.

    verbose : bool, default=True
        If ``True``, prints status messages about checking, downloading,
        or finding the file.

    Returns
    -------
    filepath : str or None
        The absolute path to the local file if it exists or was
        successfully downloaded. Returns ``None`` if the file is missing
        and `download_if_missing` is ``False``, or if the download
        fails and `error` is not 'raise'.

    Raises
    ------
    ValueError
        If the `error` parameter is invalid.
    RuntimeError
        If the download fails and `error` is set to `'raise'`.
    TypeError
        If `metadata` is not a string or `RemoteMetadata` instance.
    """
    # Validate error parameter
    if error not in ['warn', 'raise', 'ignore']:
        raise ValueError(
            "`error` parameter must be 'raise', 'warn', or 'ignore'."
        )

    # Handle string input for metadata convenience
    if isinstance(metadata, str):
        # Assume string is filename, use default URL
        if not KD_REMOTE_DATA_URL:
             msg = ("Default remote data URL is not configured. Cannot "
                    "download file specified only by name.")
             if error == 'raise': raise ValueError(msg)
             elif error == 'warn': warnings.warn(msg)
             return None
        # Create a minimal metadata object
        metadata = RemoteMetadata(
            file=metadata,
            url=KD_REMOTE_DATA_URL,
            checksum=None, # No checksum provided
            descr_module=None,
            data_module=None
        )
    elif not isinstance(metadata, RemoteMetadata):
        raise TypeError(
            "`metadata` must be a string (filename) or RemoteMetadata."
        )

    # Determine target cache directory
    data_dir = get_data(data_home)
    # Construct the full local path for the file
    local_filepath = os.path.join(data_dir, metadata.file)

    # Check if file exists locally
    file_exists = os.path.exists(local_filepath)

    if file_exists:
        if verbose:
            print(f"Data file '{metadata.file}' found in cache:"
                  f" {data_dir}")
        return local_filepath
    elif not download_if_missing:
        if verbose:
            print(f"Data file '{metadata.file}' not found in cache and "
                  f"download is disabled.")
        return None
    else:
        # File missing and download enabled, proceed with download
        if verbose:
            print(f"Data file '{metadata.file}' not found in cache. "
                  f"Attempting download from {metadata.url}...")

        # Ensure download utility is available
        # if 'fancier_downloader' not in globals() or not callable(fancier_downloader):
        #      msg = "Downloader utility is not available."
        #      if error == 'raise': 
        #          raise RuntimeError(msg)
        #      elif error == 'warn':
        #          warnings.warn(msg)
        #      return None

        # Construct the full URL
        # Ensure base URL ends with / if not already present
        base_url = metadata.url if metadata.url.endswith('/') else metadata.url + '/'
        file_url = urljoin(base_url, metadata.file)

        try:
            # Use fancier_downloader: downloads to CWD then moves to dstpath
            # So, dstpath should be the target directory `data_dir`
            # filename should be just the basename
            fancier_downloader(
                url=file_url,
                filename=metadata.file, # Download as this name locally first
                dstpath=data_dir,       # Move it here after download
                check_size=True,        # Check size against header
                error=error,            # Propagate error handling
                verbose=verbose         # Control downloader verbosity
            )
            # If downloader didn't raise error, file should now be at local_filepath
            if os.path.exists(local_filepath):
                 if verbose >=2: # Add higher verbosity level if needed
                     print(f"Download successful: '{local_filepath}'")
                 return local_filepath
            else:
                 # This case *shouldn't* happen if fancier_downloader worked
                 # without raising an error, but handle defensively.
                 msg=f"Download reported success but file not found at {local_filepath}"
                 if error == 'raise': raise RuntimeError(msg)
                 elif error == 'warn': warnings.warn(msg)
                 return None

        except Exception as e:
            # Handle exceptions raised by fancier_downloader or os calls
            # The error handling logic might be duplicated if fancier_downloader
            # also raises/warns based on 'error', but this catches other issues.
            download_error_msg = (
                f"Failed to download or cache '{metadata.file}' from "
                f"'{file_url}'. Error: {e}"
            )
            if error == 'raise':
                raise RuntimeError(download_error_msg) from e
            elif error == 'warn':
                warnings.warn(download_error_msg)
            # If error is 'ignore' or 'warn', return None
            return None