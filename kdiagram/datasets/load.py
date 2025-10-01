# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0 

"""
Dataset Loading and Generation Utilities (:mod:`kdiagram.datasets.load`)
==========================================================================

Functions to load sample or included datasets, or generate synthetic
datasets suitable for demonstrating and testing `k-diagram`
visualizations. Datasets can be returned as pandas DataFrames or
structured Bunch objects.
"""

from __future__ import annotations

import os
import re
import shutil
import textwrap
import warnings
from importlib import resources

import numpy as np
import pandas as pd

from ..api.bunch import Bunch
from ._property import (
    KD_DMODULE,
    KD_REMOTE_DATA_URL,
    RemoteMetadata,
    download_file_if,
    get_data,
)

__all__ = ["load_uncertainty_data", "load_zhongshan_subsidence"]

_ZHONGSHAN_METADATA = RemoteMetadata(
    file="min_zhongshan.csv",
    url=KD_REMOTE_DATA_URL,
    checksum=None,  # # TODO: Add SHA256 checksum here if available
    descr_module=None,
    data_module=KD_DMODULE,
)


def load_zhongshan_subsidence(
    *,
    as_frame: bool = False,
    years: list[int] | None = None,
    quantiles: list[float] | None = None,
    include_coords: bool = True,
    include_target: bool = True,
    data_home: str | None = None,
    download_if_missing: bool = True,
    force_download: bool = False,
) -> Bunch | pd.DataFrame:
    # --- Step 1: Determine file location (Cache > Package > Download) ---
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    data_dir = get_data(data_home)
    filename = _ZHONGSHAN_METADATA.file
    if os.path.exists(os.path.join(data_dir, filename)):
        local_filepath = os.path.join(data_dir, filename)
    else:
        try:
            # Construct the full path to the file within the package
            # using importlib.resources
            local_filepath = str(
                resources.files(KD_DMODULE).joinpath(filename)
            )
            data_dir = os.path.dirname(local_filepath)
            # took only the file in data path
        except Exception as e:
            # fallback.
            print(f"An unexpected error occurred: {e}")
            local_filepath = os.path.join(data_dir, filename)

    package_module_path = _ZHONGSHAN_METADATA.data_module

    filepath_to_load = None

    # Force download if requested
    if force_download:
        if download_if_missing:
            print(f"Forcing download of '{filename}'...")
            dl_path = download_file_if(
                _ZHONGSHAN_METADATA,
                data_home=data_dir,
                download_if_missing=True,
                error="warn",
                verbose=1,
            )
            if dl_path and os.path.exists(dl_path):
                filepath_to_load = dl_path
            else:
                # Error handled by download func based on 'error' flag
                # We might still try package resource below if download fails
                warnings.warn(
                    f"Forced download failed for {filename}.", stacklevel=2
                )
                pass  # Continue to check package resource
        else:
            warnings.warn(
                f"Cannot force download for {filename}, "
                f"download_if_missing is False.",
                stacklevel=2,
            )
            # Proceed to check local cache/package only

    # Check cache first (unless download was forced and succeeded)
    if filepath_to_load is None and os.path.exists(local_filepath):
        print(f"Loading dataset from cache: {local_filepath}")
        filepath_to_load = local_filepath

    # Check package resources if not found in cache
    if filepath_to_load is None:
        try:
            # New Traversable API
            pkg_root = resources.files(
                package_module_path
            )  # Traversable root
            candidate = pkg_root.joinpath(filename)  # Traversable file

            if candidate.is_file():
                print(
                    "Loading dataset from installed"
                    f" package: {package_module_path}"
                )
                # Get a real filesystem path even if inside a wheel/zip
                with resources.as_file(candidate) as rpath:
                    filepath_to_load = str(rpath)

                    # Copy to cache for future use if not already there
                    if not os.path.exists(local_filepath):
                        try:
                            os.makedirs(
                                os.path.dirname(local_filepath), exist_ok=True
                            )
                            shutil.copyfile(filepath_to_load, local_filepath)
                            print(
                                f"Copied dataset to cache: {local_filepath}"
                            )
                        except Exception as copy_err:
                            warnings.warn(
                                f"Could not copy dataset to cache: {copy_err}",
                                stacklevel=2,
                            )
            else:
                print(
                    f"Dataset not found in package resources: "
                    f"{package_module_path}/{filename}"
                )
        except ModuleNotFoundError:
            print(f"Package data module not found: {package_module_path}")
        except Exception as res_err:
            warnings.warn(
                f"Error accessing package resources: {res_err}",
                stacklevel=2,
            )

    # Attempt download if still not found and allowed
    if filepath_to_load is None and download_if_missing:
        print(f"Attempting download of '{filename}' to cache: {data_dir}")
        filepath_to_load = download_file_if(
            _ZHONGSHAN_METADATA,
            data_home=data_dir,
            download_if_missing=True,
            error="warn",
            verbose=1,  # Use warn first
        )

    # Final check if we have a path
    if filepath_to_load is None or not os.path.exists(filepath_to_load):
        raise FileNotFoundError(
            f"Zhongshan subsidence dataset ('{filename}') not found in "
            f"cache ('{data_dir}'), package resources, and could not be "
            f"downloaded. Try setting download_if_missing=True or check "
            f"internet connection."
        )

    # --- Step 2: Load data ---
    try:
        df = pd.read_csv(filepath_to_load)
    except Exception as e:
        raise OSError(
            f"Error reading dataset file at {filepath_to_load}: {e}"
        ) from e

    # --- Step 3: Subsetting / Column Selection ---
    cols_to_keep = []
    available_years = set()
    available_quantiles = set()
    q_pattern = re.compile(r"_(\d{4})_q([0-9.]+)$")
    target_pattern = re.compile(r"_(\d{4})$")

    # Identify available years and quantiles from column names
    for col in df.columns:
        q_match = q_pattern.search(col)
        t_match = target_pattern.search(col)
        if q_match:
            available_years.add(int(q_match.group(1)))
            available_quantiles.add(float(q_match.group(2)))
        elif (
            t_match
            and col.endswith(t_match.group(1))
            and col.startswith("subsidence")
        ):  # Be specific for target
            available_years.add(int(t_match.group(1)))

    available_years = sorted(list(available_years))
    available_quantiles = sorted(list(available_quantiles))

    # Validate requested years and quantiles
    requested_years = (
        set(years) if years is not None else set(available_years)
    )
    requested_quantiles = (
        set(quantiles) if quantiles is not None else set(available_quantiles)
    )

    invalid_years = requested_years - set(available_years)
    invalid_quantiles = requested_quantiles - set(available_quantiles)

    if invalid_years:
        warnings.warn(
            f"Requested years not available: {invalid_years}. "
            f"Available: {available_years}",
            UserWarning,
            stacklevel=2,
        )
        requested_years &= set(available_years)  # Keep only valid ones
    if invalid_quantiles:
        warnings.warn(
            f"Requested quantiles not available: {invalid_quantiles}. "
            f"Available: {available_quantiles}",
            UserWarning,
            stacklevel=2,
        )
        requested_quantiles &= set(
            available_quantiles
        )  # Keep only valid ones

    # Select columns based on flags and validated requests
    if include_coords:
        if "longitude" in df.columns:
            cols_to_keep.append("longitude")
        if "latitude" in df.columns:
            cols_to_keep.append("latitude")

    target_cols_found = []
    q_cols_found = {
        "q" + f"{q:.1f}".replace("0.", ""): [] for q in requested_quantiles
    }
    all_q_cols_found = []

    for col in df.columns:
        q_match = q_pattern.search(col)
        t_match = target_pattern.search(col)

        # Check target columns
        if include_target and t_match and col.startswith("subsidence"):
            year = int(t_match.group(1))
            if year in requested_years:
                cols_to_keep.append(col)
                target_cols_found.append(col)
        # Check quantile columns
        elif q_match:
            year = int(q_match.group(1))
            q_val = float(q_match.group(2))
            if year in requested_years and q_val in requested_quantiles:
                cols_to_keep.append(col)
                q_key = "q" + f"{q_val:.1f}".replace("0.", "")
                q_cols_found[q_key].append(col)
                all_q_cols_found.append(col)

    # Ensure order is somewhat logical
    cols_to_keep = sorted(
        list(set(cols_to_keep)),
        key=lambda x: (
            not x.startswith("lon")
            and not x.startswith("lat"),  # Coords first
            not x.startswith("subsidence_")
            or q_pattern.search(x) is None,  # Base target next
            x,  # Then sort alphabetically/numerically
        ),
    )
    df_subset = df[cols_to_keep].copy()

    # --- Step 4: Return DataFrame or Bunch ---
    if as_frame:
        return df_subset
    else:
        # Assemble Bunch
        feature_names = []
        if include_coords:
            if "longitude" in df_subset.columns:
                feature_names.append("longitude")
            if "latitude" in df_subset.columns:
                feature_names.append("latitude")
        target_names = target_cols_found
        target_array = (
            df_subset[target_names].values if target_names else None
        )

        # Initialize dict for quantile columns dynamically
        q_cols_found = {}  # Start empty
        all_q_cols_found = []

        # Re-define patterns just in case
        q_pattern = re.compile(r"_(\d{4})_q([0-9.]+)$")
        target_pattern = re.compile(r"_(\d{4})$")

        # Iterate over the ACTUAL columns present in the SUBSETTED DataFrame
        for col in df_subset.columns:
            q_match = q_pattern.search(col)
            # Skip target columns here, handled above by target_cols_found
            if q_match:
                year = int(
                    q_match.group(1)
                )  # Already filtered by requested_years
                q_val_str = q_match.group(2)
                try:
                    q_val = float(q_val_str)
                    # Check if this quantile was requested (already done by subsetting)
                    # --- FIX: Use consistent key format 'qX.Y' ---
                    q_key = f"q{q_val:.1f}"  # e.g., q0.1, q0.5, q.09
                    # --- End Fix ---
                    # Add key to dict if it's the first time seeing this quantile
                    if q_key not in q_cols_found:
                        q_cols_found[q_key] = []
                    q_cols_found[q_key].append(col)
                    all_q_cols_found.append(
                        col
                    )  # Keep track of all q cols found
                except ValueError:
                    warnings.warn(
                        f"Could not parse quantile value '{q_val_str}'"
                        f" from column '{col}'. Skipping.",
                        stacklevel=2,
                    )

        # Create description
        descr = textwrap.dedent(
            f"""\
        Zhongshan Land Subsidence Prediction Dataset

        **Origin:**
        This dataset contains processed outputs from a land subsidence
        forecasting study focused on Zhongshan, China. It includes
        simulated quantile predictions (Q10, Q50, Q90) for multiple
        future years (2022-2026) and base 'target' subsidence values
        for reference years (2022, 2023) at 898 locations.

        **Data Characteristics:**
        - Samples: {len(df_subset)} (Locations)
        - Features: {len(feature_names)} ({", ".join(feature_names)})
        - Target Columns: {len(target_names)} ({", ".join(target_names)})
        - Quantile Columns: {len(all_q_cols_found)} (Subset based on request)
        - Available Years (in original file): {available_years}
        - Available Quantiles (in original file): {available_quantiles}
        - Loaded Years: {sorted(list(requested_years))}
        - Loaded Quantiles: {sorted(list(requested_quantiles))}

        **Contents (Bunch object):**
        - frame           : Filtered pandas DataFrame based on parameters.
        - feature_names   : List of coordinate column names.
        - target_names    : List of loaded target column names.
        - target          : NumPy array of target values (if loaded).
        - longitude       : NumPy array of longitude values (if loaded).
        - latitude        : NumPy array of latitude values (if loaded).
        - quantile_cols   : Dict mapping requested/loaded quantiles
                          ('q0.1', etc.) to lists of column names.
        - q10_cols        : List of loaded Q10 column names.
        - q50_cols        : List of loaded Q50 column names.
        - q90_cols        : List of loaded Q90 column names.
        - years_available : List of all years detected in original columns.
        - quantiles_available: List of all quantiles detected.
        - n_periods       : Number of periods with quantile data.
        - start_year      : Starting year for period columns.
        - DESCR           : This description.

        This dataset is suitable for demonstrating uncertainty plots like
        plot_model_drift, plot_uncertainty_drift, plot_coverage_diagnostic,
        plot_anomaly_magnitude (using target cols), etc.
        """
        )
        try:
            start_year = list(requested_years)[0]
        except IndexError:
            start_year = ""
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
            start_year = ""

        bunch_dict = {
            "frame": df_subset,
            "feature_names": feature_names,
            "target_names": target_names,
            "target": target_array,
            "quantile_cols": q_cols_found,
            "q10_cols": q_cols_found.get("q0.1", []),
            "q50_cols": q_cols_found.get("q0.5", []),
            "q90_cols": q_cols_found.get("q0.9", []),
            "years_available": available_years,
            "quantiles_available": available_quantiles,
            "start_year": start_year,
            "n_periods": len(requested_years),
            "DESCR": descr,
        }
        # Add coordinates as top-level attributes if included
        if include_coords:
            if "longitude" in df_subset:
                bunch_dict["longitude"] = df_subset["longitude"].values
            if "latitude" in df_subset:
                bunch_dict["latitude"] = df_subset["latitude"].values

        return Bunch(**bunch_dict)


load_zhongshan_subsidence.__doc__ = r"""
Load the Zhongshan land subsidence prediction dataset.

This dataset contains sample multi-period quantile predictions
(Q10, Q50, Q90 for 2022–2026) and simulated actual subsidence for
2022 and 2023, along with geographic coordinates for 898 locations
in Zhongshan, China. It is intended for demonstrating and testing
`k-diagram`'s uncertainty and evaluation plots and for reproducing
examples related to spatiotemporal uncertainty diagnostics
:footcite:p:`Liu2024, kouadiob2025`.

The function searches a local cache directory, bundled package
resources, and optionally a remote repository (in that order).
On success it returns either a pandas ``DataFrame`` or a
:class:`~kdiagram.bunch.Bunch` with convenient attributes.

Parameters
----------
as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` that
    includes the filtered ``DataFrame`` plus metadata and sliced
    arrays (e.g., coordinates, target, and quantile columns).
    If ``True``, return only the filtered ``DataFrame``.

years : list of int, optional
    Subset to these calendar years (e.g., ``[2023, 2025]``) when
    selecting target and quantile columns. If ``None``, load all
    years found in the file (quantiles typically 2022–2026; targets
    typically 2022/2023).

quantiles : list of float, optional
    Subset to these quantile levels in ``[0, 1]`` (e.g.,
    ``[0.1, 0.5, 0.9]``). If ``None``, load all detected
    quantiles for the selected years. Defaults to
    ``[0.1, 0.5, 0.9]``.

include_coords : bool, default=True
    If ``True``, include coordinate columns ``'longitude'`` and
    ``'latitude'`` when present.

include_target : bool, default=True
    If ``True``, include base target columns (e.g.,
    ``'subsidence_2022'``, ``'subsidence_2023'``) when present
    and consistent with the requested ``years``.

data_home : str, optional
    Directory path for caching datasets. If ``None``, the path is
    resolved by :func:`~kdiagram.datasets._property.get_data`. You
    may also configure the root via the ``KDIAGRAM_DATA`` environment
    variable. Example default is ``~/kdiagram_data``.

download_if_missing : bool, default=True
    If ``True``, attempt to download the dataset into the cache when
    it is not found locally nor in package resources.

force_download : bool, default=False
    If ``True``, attempt to fetch a fresh copy even if a local file
    exists. Useful to refresh data during development.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default) a Bunch with:
        
    - ``frame`` : pandas ``DataFrame`` filtered by the request.
    - ``feature_names`` : list of included coordinate column names.
    - ``target_names`` : list of included target column names.
    - ``target`` : NumPy array of target values (or ``None``).
    - ``longitude``, ``latitude`` : NumPy arrays when coordinates
      are included.
    - ``quantile_cols`` : dict mapping keys like ``'q0.1'`` to
      lists of matching column names.
    - ``q10_cols``, ``q50_cols``, ``q90_cols`` : convenience lists.
    - ``years_available``, ``quantiles_available`` : lists detected
      in the original file.
    - ``start_year`` : smallest year in the loaded subset (if any).
    - ``n_periods`` : number of loaded years.
    - ``DESCR`` : human-readable dataset description.

    If ``as_frame=True``, only the filtered pandas ``DataFrame`` is
    returned.

Raises
------
FileNotFoundError
    When the dataset cannot be resolved from cache or package
    resources and either downloading is disabled or the download
    fails.

ValueError
    If requested ``years`` or ``quantiles`` are invalid or not
    present in the data file.

Notes
-----
**Search order.** The loader resolves a file path using the
following order: (1) local cache under ``data_home``;
(2) installed package resources; (3) optional remote download
when ``download_if_missing=True``. You can force step (3) with
``force_download=True``.

**Column detection.** Quantile columns encode a year :math:`y` and a
quantile level :math:`q` in their names.

.. math::

   \text{quantile name}
   \;\equiv\;
   \texttt{<prefix>}\_{y}\_\texttt{q}q,
   \qquad
   y \in \{2022,\dots,2026\},\; q \in (0,1)

Target columns encode only the year :math:`y`:

.. math::

   \text{target name}
   \;\equiv\;
   \texttt{subsidence}\_{y}

In code, the implementation detects these with the following regular
expressions (kept as literals, not math):
``r"_(\d{4})_q([0-9.]+)$"`` (quantile columns) and
``r"_(\d{4})$"`` (target columns).

This design enables flexible subsetting by year and quantile without
hard-coding headers.

**Coordinate handling.** When present and ``include_coords=True``,
the columns ``'longitude'`` and ``'latitude'`` are included and
exposed both in the returned frame and as top-level arrays in the
Bunch for convenience.

**Intended use.** The dataset is a compact sample designed for
tutorials, documentation figures, and regression tests of k-diagram
uncertainty diagnostics :footcite:p:`kouadiob2025`. It is not a
comprehensive research release.

See Also
--------
load_uncertainty_data
    Generate a synthetic dataset with controllable anomalies and
    quantiles for testing visual diagnostics.

kdiagram.plot.uncertainty.plot_model_drift
kdiagram.plot.uncertainty.plot_uncertainty_drift
kdiagram.plot.uncertainty.plot_coverage_diagnostic
kdiagram.plot.uncertainty.plot_anomaly_magnitude
    Example consumers of this dataset in documentation figures.

Examples
--------
Basic usage returning a Bunch with metadata:

>>> from kdiagram.datasets import load_zhongshan_subsidence
>>> ds = load_zhongshan_subsidence()
>>> isinstance(ds.frame, type(__import__('pandas').DataFrame()))
True
>>> list(ds.quantile_cols.keys())[:3]
['q0.1', 'q0.5', 'q0.9']
>>>
>>> # Return only the DataFrame and subset to selected years/quantiles:
>>>
>>> df = load_zhongshan_subsidence(
...     as_frame=True, years=[2023, 2025], quantiles=[0.1, 0.9]
... )
>>> set(c.split('_')[-1] for c in df.columns if '_q' in c) <= {'q0.1','q0.9'}
True
>>>
>>> # Force a fresh download into the cache:
>>> _ = load_zhongshan_subsidence(force_download=True)

References
----------

.. footbibliography::
    
"""


def load_uncertainty_data(
    *,
    as_frame: bool = False,
    n_samples: int = 150,
    n_periods: int = 4,
    anomaly_frac: float = 0.15,
    start_year: int = 2022,
    prefix: str = "value",
    base_value: float = 10.0,
    trend_strength: float = 1.5,
    noise_level: float = 2.0,
    interval_width_base: float = 4.0,
    interval_width_noise: float = 1.5,
    interval_width_trend: float = 0.5,
    seed: int | None = 42,
) -> Bunch | pd.DataFrame:
    # --- Generation Logic (Moved from make_uncertainty_data) ---
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    location_id = np.arange(n_samples)
    longitude = rng.uniform(-120, -115, n_samples)
    latitude = rng.uniform(33, 36, n_samples)
    elevation = rng.uniform(50, 500, n_samples) + latitude * 5
    base_signal = (
        base_value
        + np.sin(np.linspace(0, 3 * np.pi, n_samples)) * 5
        + rng.normal(0, noise_level / 2, n_samples)
    )
    actual_first_period = base_signal + rng.normal(
        0, noise_level / 2, n_samples
    )

    data_dict = {
        "location_id": location_id,
        "longitude": longitude,
        "latitude": latitude,
        "elevation": elevation,
        f"{prefix}_actual": actual_first_period.copy(),
    }

    all_q10_cols, all_q50_cols, all_q90_cols = [], [], []
    quantile_cols_dict = {"q0.1": [], "q0.5": [], "q0.9": []}

    for i in range(n_periods):
        year = start_year + i
        q10_col = f"{prefix}_{year}_q0.1"
        q50_col = f"{prefix}_{year}_q0.5"
        q90_col = f"{prefix}_{year}_q0.9"

        all_q10_cols.append(q10_col)
        all_q50_cols.append(q50_col)
        all_q90_cols.append(q90_col)
        quantile_cols_dict["q0.1"].append(q10_col)
        quantile_cols_dict["q0.5"].append(q50_col)
        quantile_cols_dict["q0.9"].append(q90_col)

        current_trend = trend_strength * i
        q50 = (
            base_signal
            + current_trend
            + rng.normal(0, noise_level / 3, n_samples)
        )

        current_interval_width = (
            interval_width_base
            + interval_width_trend * i
            + rng.uniform(
                -interval_width_noise / 2, interval_width_noise / 2, n_samples
            )
        )
        current_interval_width = np.maximum(0.1, current_interval_width)

        q10 = q50 - current_interval_width / 2
        q90 = q50 + current_interval_width / 2

        data_dict[q10_col] = q10
        data_dict[q50_col] = q50
        data_dict[q90_col] = q90

    df = pd.DataFrame(data_dict)

    actual_col_name = f"{prefix}_actual"
    if anomaly_frac > 0 and n_samples > 0:
        n_anomalies = int(anomaly_frac * n_samples)
        if n_anomalies > 0 and all_q10_cols and all_q90_cols:
            anomaly_indices = rng.choice(
                n_samples, size=n_anomalies, replace=False
            )
            n_under = n_anomalies // 2
            under_indices = anomaly_indices[:n_under]
            over_indices = anomaly_indices[n_under:]

            q10_first = df[all_q10_cols[0]].iloc[under_indices]
            q90_first = df[all_q90_cols[0]].iloc[over_indices]

            df.loc[under_indices, actual_col_name] = q10_first - rng.uniform(
                0.5, 3.0, size=len(under_indices)
            ) * (interval_width_base / 2 + 1)

            df.loc[over_indices, actual_col_name] = q90_first + rng.uniform(
                0.5, 3.0, size=len(over_indices)
            ) * (interval_width_base / 2 + 1)

    feature_names = ["location_id", "longitude", "latitude", "elevation"]
    target_names = [actual_col_name]
    pred_cols_sorted = [
        col
        for pair in zip(all_q10_cols, all_q50_cols, all_q90_cols)
        for col in pair
    ]
    ordered_cols = feature_names + target_names + pred_cols_sorted
    df = df[ordered_cols]

    # --- Return based on as_frame ---
    if as_frame:
        return df
    else:
        target_array = df[target_names[0]].values
        descr = textwrap.dedent(
            f"""\
        Synthetic Multi-Period Uncertainty Dataset for k-diagram

        **Description:**
        Generates synthetic data simulating quantile forecasts (Q10,
        Q50, Q90) for '{prefix}' over {n_periods} periods starting
        from {start_year} across {n_samples} samples/locations. Includes
        spatial coordinates, an 'elevation' feature, and an 'actual'
        value (``{actual_col_name}``) for the first period. Anomalies
        (actual values outside the first period's Q10-Q90 interval)
        are introduced for ~{anomaly_frac * 100:.0f}% of samples. Both the
        median (Q50) and the interval width can exhibit configurable
        trends and noise.

        **Generation Parameters:**
        - n_samples             : {n_samples}
        - n_periods             : {n_periods}
        - start_year            : {start_year}
        - prefix                : '{prefix}'
        - anomaly_frac          : {anomaly_frac:.2f}
        - base_value            : {base_value:.2f}
        - trend_strength        : {trend_strength:.2f}
        - noise_level           : {noise_level:.2f}
        - interval_width_base   : {interval_width_base:.2f}
        - interval_width_noise  : {interval_width_noise:.2f}
        - interval_width_trend  : {interval_width_trend:.2f}
        - seed                  : {seed}

        **Bunch Attributes:**
        - frame           : Complete pandas DataFrame.
        - feature_names   : List of coordinate/feature column names.
        - target_names    : List containing the target column name.
        - target          : NumPy array of target values.
        - quantile_cols   : Dict mapping quantiles to column name lists.
        - q10_cols        : List of Q10 column names.
        - q50_cols        : List of Q50 column names.
        - q90_cols        : List of Q90 column names.
        - n_periods       : Number of periods with quantile data.
        - prefix          : Prefix used for value/quantile columns.
        - start_year      : Starting year for period columns.
        - DESCR           : This description.
        """
        )

        return Bunch(
            frame=df,
            feature_names=feature_names,
            target_names=target_names,
            target=target_array,
            quantile_cols=quantile_cols_dict,
            q10_cols=all_q10_cols,
            q50_cols=all_q50_cols,
            q90_cols=all_q90_cols,
            n_periods=n_periods,
            prefix=prefix,
            start_year=start_year,
            DESCR=descr,
        )


load_uncertainty_data.__doc__ = r"""
Generate a synthetic dataset for uncertainty diagnostics.

Creates a compact, controllable dataset for demonstrating
`k-diagram` plots: one period of actuals, multi-period predicted
quantiles (Q10, Q50, Q90), configurable trends and noise, injected
interval failures (anomalies), and optional coordinates. Useful for
examples, unit tests, and performance checks :footcite:p:`kouadiob2025`.

Parameters
----------
as_frame : bool, default=False
    If ``False``, return a :class:`~kdiagram.bunch.Bunch` with the
    generated frame and metadata; if ``True``, return only the
    pandas ``DataFrame``.

n_samples : int, default=150
    Number of rows (locations) to generate.

n_periods : int, default=4
    Number of consecutive periods (e.g., years) for which quantiles
    are generated.

anomaly_frac : float, default=0.15
    Approximate fraction in ``[0, 1]`` where the actual value lies
    outside the first period’s Q10–Q90 interval.

start_year : int, default=2022
    Starting year used when naming time-dependent columns.

prefix : str, default='value'
    Base prefix for value and quantile column names.

base_value : float, default=10.0
    Approximate mean of the signal in the first period.

trend_strength : float, default=1.5
    Linear trend added to the Q50 trajectory across periods.

noise_level : float, default=2.0
    Standard deviation of base random noise added to values.

interval_width_base : float, default=4.0
    Base width of the Q10–Q90 interval in the first period.

interval_width_noise : float, default=1.5
    Random variability added to the interval width per sample/period.

interval_width_trend : float, default=0.5
    Linear trend added to the interval width across periods.

seed : int or None, default=42
    Random seed for reproducibility. If ``None``, use an
    unconstrained RNG state.

Returns
-------
data : :class:`~kdiagram.bunch.Bunch` or pandas.DataFrame
    If ``as_frame=False`` (default) a Bunch with:
        
    - ``frame`` : pandas ``DataFrame`` of synthesized values.
    - ``feature_names`` : included feature columns (e.g., coords).
    - ``target_names`` : names of actual/target columns.
    - ``target`` : NumPy array of target values (if present).
    - ``quantile_cols`` : dict mapping ``'q0.1'``, ``'q0.5'``,
      ``'q0.9'`` to lists of columns across periods.
    - ``q10_cols``, ``q50_cols``, ``q90_cols`` : convenience lists.
    - ``n_periods``, ``prefix``, ``start_year``, and ``DESCR``.

    If ``as_frame=True``, return only the ``DataFrame``.

Notes
-----
The generator injects a user-controlled fraction of interval
failures to create meaningful examples for coverage and anomaly
diagnostics. Use this dataset to exercise plots such as coverage
rates, point-wise coverage diagnostics, anomaly magnitude,
temporal consistency, and drift views 
:footcite:p:`Jolliffe2012, Gneiting2007b, kouadiob2025`.

See Also
--------
load_zhongshan_subsidence
    Real-world sample for Zhongshan subsidence with quantiles and
    coordinates.

kdiagram.plot.uncertainty.plot_coverage
kdiagram.plot.uncertainty.plot_coverage_diagnostic
kdiagram.plot.uncertainty.plot_anomaly_magnitude
kdiagram.plot.uncertainty.plot_interval_consistency
kdiagram.plot.uncertainty.plot_model_drift
    Visual diagnostics this dataset was designed to support.

Examples
--------
>>> # Create a small dataset and explore quantile columns:
>>> 
>>> from kdiagram.datasets import load_uncertainty_data
>>> ds = load_uncertainty_data(n_samples=10, n_periods=3, seed=0)
>>> sorted(ds.quantile_cols.keys())
['q0.1', 'q0.5', 'q0.9']
>>> 
>>> # Return a ``DataFrame`` only:
>>> 
>>> df = load_uncertainty_data(as_frame=True, n_samples=5, seed=1)
>>> df.shape[0] == 5
True

References 
----------

.. footbibliography::
    
"""
