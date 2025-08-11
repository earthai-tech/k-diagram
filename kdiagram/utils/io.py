# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 (see LICENSE file)

# -------------------------------------------------------------------
# Provides input/output utility functions (e.g., downloading).
# Adapted by the 'kdiagram.utils.io.io' module from
# the 'gofast' package: https://github.com/earthai-tech/gofast
# Original 'gofast' code licensed under BSD-3-Clause.
# Modifications and 'k-diagram' are under Apache License 2.0.
# -------------------------------------------------------------------

"""
Input/Output Utilities (:mod:`kdiagram.utils.io`)
================================================

This module contains helper functions related to file input/output
operations, such as downloading files from URLs (with progress),
moving files, and checking for resource existence within packages.
"""

from __future__ import annotations

import os
import shutil
import warnings

HAS_REQUESTS = True
try:
    import requests  # NOQA
except ImportError:
    HAS_REQUESTS = False

__all__ = ["fancier_downloader", "download_file", "check_file_exists"]


def fancier_downloader(
    url: str,
    filename: str,
    dstpath: str | None = None,
    check_size: bool = False,
    error: str = "raise",
    verbose: bool = True,
) -> str | None:
    r"""
    Download a remote file with a progress bar and optional size verification.

    This function downloads a file from the specified ``url`` and saves it locally
    with the given ``filename``. It provides a visual progress bar during the
    download process and offers an option to verify the downloaded file's size
    against the expected size to ensure data integrity. Additionally, the function
    allows for moving the downloaded file to a specified destination directory.

    .. math::
        |S_{downloaded} - S_{expected}| < \epsilon

    where :math:`S_{downloaded}` is the size of the downloaded file,
    :math:`S_{expected}` is the size specified by the server,
    and :math:`\epsilon` is a small tolerance value.

    Parameters
    ----------
    url : str
        The URL from which to download the remote file.

    filename : str
        The desired name for the local file. This is the name under which the
        file will be saved after downloading.

    dstpath : Optional[str], default=None
        The destination directory path where the downloaded file should be saved.
        If ``None``, the file is saved in the current working directory.

    check_size : bool, default=False
        Whether to verify the size of the downloaded file against the expected
        size obtained from the server. This is useful for ensuring the integrity
        of the downloaded file. When ``True``, the function checks:

        .. math::
            |S_{downloaded} - S_{expected}| < \epsilon

        If the size check fails:

        - If ``error='raise'``, an exception is raised.
        - If ``error='warn'``, a warning is emitted.
        - If ``error='ignore'``, the discrepancy is ignored, and the function
          continues.

    error : str, default='raise'
        Specifies how to handle errors during the size verification process.

        - ``'raise'``: Raises an exception if the file size does not match.
        - ``'warn'``: Emits a warning and continues execution.
        - ``'ignore'``: Silently ignores the size discrepancy and proceeds.

    verbose : bool, default=True
        Controls the verbosity of the function. If ``True``, the function will
        print informative messages about the download status, including progress
        updates and success or failure notifications.

    Returns
    -------
    Optional[str]
        Returns ``None`` if ``dstpath`` is provided and the file is moved to the
        destination. Otherwise, returns the local filename as a string.

    Raises
    ------
    RuntimeError
        If the download fails and ``error`` is set to ``'raise'``.

    ValueError
        If an invalid value is provided for the ``error`` parameter.

    Examples
    --------
    >>> from kdiagram.utils.io import fancier_downloader
    >>> url = 'https://example.com/data/file.h5'
    >>> local_filename = 'file.h5'
    >>> # Download to current directory without size check
    >>> fancier_downloader(url, local_filename)
    >>>
    >>> # Download to a specific directory with size verification
    >>> fancier_downloader(
    ...     url,
    ...     local_filename,
    ...     dstpath='/path/to/save/',
    ...     check_size=True,
    ...     error='warn',
    ...     verbose=True
    ... )
    >>>
    >>> # Handle size mismatch by raising an exception
    >>> fancier_downloader(
    ...     url,
    ...     local_filename,
    ...     check_size=True,
    ...     error='raise'
    ... )

    Notes
    -----
    - **Progress Bar**: The function uses the `tqdm` library to display a
      progress bar during the download. If `tqdm` is not installed, it falls
      back to a basic downloader without a progress bar.
    - **Directory Creation**: If the specified ``dstpath`` does not exist,
      the function will attempt to create it to ensure the file is saved
      correctly.
    - **File Integrity**: Enabling ``check_size`` helps in verifying that the
      downloaded file is complete and uncorrupted. However, it does not perform
      a checksum verification.

    See Also
    --------
    - :func:`requests.get` : Function to perform HTTP GET requests.
    - :func:`tqdm` : A library for creating progress bars.
    - :func:`os.makedirs` : Function to create directories.
    - :func:`kdiagram.utils.io.check_file_exists` : Utility to check file
      existence.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
           & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python.
           *Journal of Machine Learning Research*, 12, 2825-2830.
    .. [2] tqdm documentation. https://tqdm.github.io/
    """

    if not HAS_REQUESTS:
        raise ImportError(
            "Requests is not installed. Please use 'pip' or 'conda'" " to install it!."
        )

    if error not in ["ignore", "warn", "raise"]:
        raise ValueError("`error` parameter must be 'raise', 'warn', or 'ignore'.")

    try:
        from tqdm import tqdm  # Import tqdm for progress bar visualization
    except ImportError:
        # If tqdm is not installed, fallback to the basic download_file function
        if verbose:
            warnings.warn(
                "tqdm is not installed. Falling back"
                " to basic downloader without progress bar.",
                stacklevel=2,
            )
        return download_file(url, filename, dstpath)

    try:
        # Initiate the HTTP GET request with streaming enabled
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes

            # Retrieve the total size of the file from the 'Content-Length' header
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # Define the chunk size (1 Kibibyte)

            # Initialize the progress bar with the total file size
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                ncols=77,
                ascii=True,
                desc=f"Downloading {filename}",
            )

            # Open the target file in binary write mode
            with open(filename, "wb") as file:
                # Iterate over the response stream in chunks
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))  # Update the progress bar
                    file.write(data)  # Write the chunk to the file
            progress_bar.close()  # Close the progress bar once download is complete

        # Optional: Verify the size of the downloaded file
        if check_size:
            # Get the actual size of the downloaded file
            downloaded_size = os.path.getsize(filename)
            expected_size = total_size_in_bytes

            # Define a tolerance level (e.g., 1%) for size discrepancy
            tolerance = expected_size * 0.01
            # for consistency if
            if downloaded_size >= expected_size:
                expected_size = downloaded_size

            # Check if the downloaded file size is within the acceptable range
            if not (
                expected_size - tolerance
                <= downloaded_size
                <= expected_size + tolerance
            ):
                # Prepare an informative message about the size mismatch
                size_mismatch_msg = (
                    f"Downloaded file size for '{filename}' ({downloaded_size} bytes) "
                    f"does not match the expected size ({expected_size} bytes)."
                )

                # Handle the discrepancy based on the 'error' parameter
                if error == "raise":
                    raise RuntimeError(size_mismatch_msg)
                elif error == "warn":
                    warnings.warn(size_mismatch_msg, stacklevel=2)
                elif error == "ignore":
                    pass  # Do nothing and continue

            elif verbose:
                print(f"File size for '{filename}' verified successfully.")

        # Move the file to the destination path if 'dstpath' is provided
        if dstpath:
            try:
                # Ensure the destination directory exists
                os.makedirs(dstpath, exist_ok=True)

                # Define the full destination path
                destination_file = os.path.join(dstpath, filename)

                # Move the downloaded file to the destination directory
                os.replace(filename, destination_file)

                if verbose:
                    print(f"File '{filename}' moved to '{destination_file}'.")
            except Exception as move_error:
                # Handle any errors that occur during the file move
                move_error_msg = (
                    f"Failed to move '{filename}' to '{dstpath}'. Error: {move_error}"
                )
                if error == "raise":
                    raise RuntimeError(move_error_msg) from move_error
                elif error == "warn":
                    warnings.warn(move_error_msg, stacklevel=2)
                elif error == "ignore":
                    pass  # Do nothing and continue

            return None  # Return None since the file has been moved
        else:
            if verbose:
                print(f"File '{filename}' downloaded successfully.")
            return filename  # Return the filename if no destination path is provided

    except Exception as download_error:
        # Handle any exceptions that occur during the download process
        download_error_msg = (
            f"Failed to download '{filename}' from '{url}'. Error: {download_error}"
        )
        if error == "raise":
            raise RuntimeError(download_error_msg) from download_error
        elif error == "warn":
            warnings.warn(download_error_msg, stacklevel=2)
        elif error == "ignore":
            pass  # Do nothing and continue

    return None  # Return None as a fallback


def download_file(url, filename, dstpath=None):
    r"""download a remote file.

    Parameters
    -----------
    url: str,
      Url to where the file is stored.
    loadl_filename: str,
      Name of the local file

    dstpath: Optional
      The destination path to save the downloaded file.

    Return
    --------
    None, local_filename
       None if the `dstpath` is supplied and `local_filename` otherwise.

    Example
    ---------
    >>> from kdiagram.utils.io import download_file
    >>> url = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/gofast/datasets/data/h.h5'
    >>> local_filename = 'h.h5'
    >>> download_file(url, local_filename, test_directory)

    """

    import requests

    print(
        "{:-^70}".format(
            f" Please, Wait while {os.path.basename(filename)}" " is downloading. "
        )
    )
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    filename = os.path.join(os.getcwd(), filename)

    if dstpath:
        move_file(filename, dstpath)

    print("{:-^70}".format(" ok! "))

    return None if dstpath else filename


def check_file_exists(package, resource):
    r"""
    Check if a file exists in a package's directory with
    importlib.resources.
    
    Parameters 
    ----------

    package: str
        The package containing the resource.
    resource:  str
        The resource (file) to check.
    
    Returns 
    --------
    bool, 
      Boolean indicating if the resource exists.

    :example:
        >>> from kdiagram.utils.io import check_file_exists
        >>> package_name = 'gofast.datasets.data'  # Replace with your package name
        >>> file_name = 'h.h5'    # Replace with your file name

        >>> file_exists = check_file_exists(package_name, file_name)
        >>> print(f"File exists: {file_exists}")
    """

    import importlib.resources as pkg_resources

    return pkg_resources.is_resource(package, resource)


def move_file(file_path, directory):
    r"""Move file to a directory.

    Create a directory if not exists.

    Parameters
    -----------
    file_path: str,
       Path to the local file
    directory: str,
       Path to locate the directory.

    Example
    ---------
    >>> from kdiagram.utils.io import move_file
    >>> file_path = 'path/to/your/file.txt'  # Replace with your file's path
    >>> directory = 'path/to/your/directory'  # Replace with your directory's path
    >>> move_file(file_path, directory)
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Move the file to the directory
    shutil.move(file_path, os.path.join(directory, os.path.basename(file_path)))
