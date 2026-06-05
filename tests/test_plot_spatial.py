# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Tests for kdiagram/plot/spatial.py

Covers: smoke execution, return types, visual state, edge cases (empty
data, all-NaN, single-row, mismatched names), parameter variations, and
file-save behaviour.  All tests run headless via the Agg backend.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from kdiagram.plot.spatial import (
    plot_paired_spatial_polar,
    plot_polar_from_spatial,
    plot_spatial_comparison,
    plot_spatial_coverage,
    plot_spatial_heatmap,
    plot_spatial_ordering,
    plot_spatial_scatter,
    plot_spatial_uncertainty,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all Matplotlib figures before and after every test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def scatter_df():
    """40-point DataFrame with unique (x, y) locations and two metrics."""
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "lon": rng.uniform(113.0, 113.5, n),
            "lat": rng.uniform(22.3, 22.7, n),
            "cas": rng.uniform(0.0, 1.0, n),
            "width": rng.uniform(0.1, 2.0, n),
            "label": [f"S{i}" for i in range(n)],
        }
    )


@pytest.fixture
def uncertainty_df():
    """20 stations × 10 time steps each — for plot_spatial_uncertainty."""
    rng = np.random.default_rng(42)
    n_stations, n_steps = 20, 10
    lons = np.repeat(rng.uniform(113.0, 113.5, n_stations), n_steps)
    lats = np.repeat(rng.uniform(22.3, 22.7, n_stations), n_steps)
    y = rng.normal(0, 1, n_stations * n_steps)
    q10 = y - rng.uniform(0.5, 2.0, n_stations * n_steps)
    q90 = y + rng.uniform(0.5, 2.0, n_stations * n_steps)
    return pd.DataFrame(
        {"lon": lons, "lat": lats, "y": y, "q10": q10, "q90": q90}
    )


@pytest.fixture
def coverage_df():
    """30-point DataFrame with pre-computed coverage rates."""
    rng = np.random.default_rng(7)
    n = 30
    return pd.DataFrame(
        {
            "lon": rng.uniform(113.0, 113.5, n),
            "lat": rng.uniform(22.3, 22.7, n),
            "coverage": rng.beta(9, 1, n),
        }
    )


@pytest.fixture
def comparison_df():
    """40-point DataFrame with three metric columns (one per model)."""
    rng = np.random.default_rng(3)
    n = 40
    return pd.DataFrame(
        {
            "lon": rng.uniform(113.0, 113.5, n),
            "lat": rng.uniform(22.3, 22.7, n),
            "cas_qar": rng.uniform(0.1, 0.9, n),
            "cas_qgbm": rng.uniform(0.05, 0.7, n),
            "cas_xtft": rng.uniform(0.02, 0.5, n),
        }
    )


# ---------------------------------------------------------------------------
# plot_spatial_scatter
# ---------------------------------------------------------------------------


class TestPlotSpatialScatter:
    def test_returns_axes(self, scatter_df):
        ax = plot_spatial_scatter(scatter_df, "lon", "lat", "cas")
        assert isinstance(ax, Axes)

    def test_title_set(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", title="My Title"
        )
        assert ax.get_title() == "My Title"

    def test_default_title_contains_metric(self, scatter_df):
        ax = plot_spatial_scatter(scatter_df, "lon", "lat", "cas")
        assert "cas" in ax.get_title()

    def test_xlabel_ylabel(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df,
            "lon",
            "lat",
            "cas",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        assert ax.get_xlabel() == "Longitude"
        assert ax.get_ylabel() == "Latitude"

    def test_default_labels_use_col_names(self, scatter_df):
        ax = plot_spatial_scatter(scatter_df, "lon", "lat", "cas")
        assert ax.get_xlabel() == "lon"
        assert ax.get_ylabel() == "lat"

    def test_scatter_collection_created(self, scatter_df):
        ax = plot_spatial_scatter(scatter_df, "lon", "lat", "cas")
        assert len(ax.collections) >= 1

    def test_size_col_bubbles(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df,
            "lon",
            "lat",
            "cas",
            size_col="width",
            size_range=(10, 300),
        )
        assert isinstance(ax, Axes)
        # Collection should still exist
        assert len(ax.collections) >= 1

    def test_no_colorbar(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", colorbar=False
        )
        assert isinstance(ax, Axes)

    def test_annotate_flag(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", annotate=True
        )
        assert isinstance(ax, Axes)
        # Annotations are text artists
        assert len(ax.texts) > 0

    def test_annotation_col(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df,
            "lon",
            "lat",
            "cas",
            annotate=True,
            annotation_col="label",
        )
        assert isinstance(ax, Axes)
        assert len(ax.texts) == len(scatter_df)

    def test_vmin_vmax(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", vmin=0.2, vmax=0.8
        )
        sc = ax.collections[0]
        assert sc.norm.vmin == pytest.approx(0.2)
        assert sc.norm.vmax == pytest.approx(0.8)

    def test_savefig_creates_file(self, scatter_df, tmp_path):
        out = tmp_path / "scatter.png"
        plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", savefig=str(out)
        )
        assert out.exists()

    def test_all_nan_metric_returns_none(self):
        df = pd.DataFrame(
            {
                "lon": [1.0, 2.0, 3.0],
                "lat": [1.0, 2.0, 3.0],
                "cas": [np.nan, np.nan, np.nan],
            }
        )
        with pytest.warns(
            UserWarning, match="no data remains after dropping NaNs"
        ):
            ax = plot_spatial_scatter(df, "lon", "lat", "cas")
        assert ax is None

    def test_missing_column_raises(self, scatter_df):
        with pytest.raises(ValueError):
            plot_spatial_scatter(scatter_df, "lon", "lat", "nonexistent")

    def test_empty_dataframe_raises_or_warns(self):
        df = pd.DataFrame(
            {
                "lon": pd.Series([], dtype=float),
                "lat": pd.Series([], dtype=float),
                "cas": pd.Series([], dtype=float),
            }
        )
        with pytest.raises(ValueError):
            plot_spatial_scatter(df, "lon", "lat", "cas")

    def test_custom_cmap(self, scatter_df):
        ax = plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", cmap="plasma"
        )
        assert isinstance(ax, Axes)

    def test_existing_ax_reused(self, scatter_df):
        fig, existing_ax = plt.subplots()
        returned_ax = plot_spatial_scatter(
            scatter_df, "lon", "lat", "cas", ax=existing_ax
        )
        assert returned_ax is existing_ax


# ---------------------------------------------------------------------------
# plot_spatial_heatmap
# ---------------------------------------------------------------------------


class TestPlotSpatialHeatmap:
    def test_returns_axes(self, scatter_df):
        ax = plot_spatial_heatmap(scatter_df, "lon", "lat", "cas")
        assert isinstance(ax, Axes)

    def test_title_set(self, scatter_df):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", title="Heatmap"
        )
        assert ax.get_title() == "Heatmap"

    def test_default_title_contains_metric(self, scatter_df):
        ax = plot_spatial_heatmap(scatter_df, "lon", "lat", "cas")
        assert "cas" in ax.get_title()

    def test_imshow_image_created(self, scatter_df):
        ax = plot_spatial_heatmap(scatter_df, "lon", "lat", "cas")
        assert len(ax.images) >= 1

    def test_contour_overlay(self, scatter_df):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", contour=True, contour_levels=5
        )
        assert isinstance(ax, Axes)
        # Contour creates LineCollection objects
        assert len(ax.collections) >= 1

    def test_no_scatter_overlay(self, scatter_df):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", scatter_overlay=False
        )
        # Without overlay, no PathCollection (scatter) should exist
        from matplotlib.collections import PathCollection

        path_colls = [
            c for c in ax.collections if isinstance(c, PathCollection)
        ]
        assert len(path_colls) == 0

    def test_scatter_overlay_present(self, scatter_df):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", scatter_overlay=True
        )
        from matplotlib.collections import PathCollection

        path_colls = [
            c for c in ax.collections if isinstance(c, PathCollection)
        ]
        assert len(path_colls) >= 1

    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    def test_interpolation_methods(self, scatter_df, method):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", method=method
        )
        assert isinstance(ax, Axes)

    def test_no_colorbar(self, scatter_df):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", colorbar=False
        )
        assert isinstance(ax, Axes)

    def test_savefig_creates_file(self, scatter_df, tmp_path):
        out = tmp_path / "heatmap.png"
        plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", savefig=str(out)
        )
        assert out.exists()

    def test_all_nan_returns_none(self):
        df = pd.DataFrame(
            {
                "lon": [1.0, 2.0, 3.0],
                "lat": [1.0, 2.0, 3.0],
                "cas": [np.nan, np.nan, np.nan],
            }
        )
        with pytest.warns(
            UserWarning, match="no data remains after dropping NaNs"
        ):
            ax = plot_spatial_heatmap(df, "lon", "lat", "cas")
        assert ax is None

    def test_missing_column_raises(self, scatter_df):
        with pytest.raises(ValueError):
            plot_spatial_heatmap(scatter_df, "lon", "lat", "bad_col")

    def test_resolution_param(self, scatter_df):
        ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", resolution=50
        )
        assert isinstance(ax, Axes)
        # The image should be 50×50
        img_data = ax.images[0].get_array()
        assert img_data.shape[0] == 50
        assert img_data.shape[1] == 50

    def test_existing_ax_reused(self, scatter_df):
        fig, existing_ax = plt.subplots()
        returned_ax = plot_spatial_heatmap(
            scatter_df, "lon", "lat", "cas", ax=existing_ax
        )
        assert returned_ax is existing_ax


# ---------------------------------------------------------------------------
# plot_spatial_uncertainty
# ---------------------------------------------------------------------------


class TestPlotSpatialUncertainty:
    def test_returns_axes(self, uncertainty_df):
        ax = plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90"
        )
        assert isinstance(ax, Axes)

    def test_title_set(self, uncertainty_df):
        ax = plot_spatial_uncertainty(
            uncertainty_df,
            "lon",
            "lat",
            "y",
            "q10",
            "q90",
            title="Uncertainty map",
        )
        assert ax.get_title() == "Uncertainty map"

    def test_scatter_collection_created(self, uncertainty_df):
        ax = plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90"
        )
        assert len(ax.collections) >= 1

    def test_aggregation_per_location(self, uncertainty_df):
        """20 stations × 10 rows → exactly 20 plotted points."""
        ax = plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90", legend=False
        )
        sc = ax.collections[0]
        assert len(sc.get_offsets()) == 20

    def test_unique_locations_no_aggregation(self, scatter_df):
        """Unique (lon, lat) rows → same count as input rows."""
        # scatter_df has 40 unique locations; build a compatible DataFrame
        df = scatter_df.copy()
        df["y"] = df["cas"]
        df["q10"] = df["cas"] - df["width"]
        df["q90"] = df["cas"] + df["width"]
        ax = plot_spatial_uncertainty(
            df, "lon", "lat", "y", "q10", "q90", legend=False
        )
        sc = ax.collections[0]
        assert len(sc.get_offsets()) == len(df)

    def test_no_legend(self, uncertainty_df):
        ax = plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90", legend=False
        )
        assert ax.get_legend() is None

    def test_nominal_param(self, uncertainty_df):
        """Different nominal values should still produce a valid plot."""
        ax = plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90", nominal=0.80
        )
        assert isinstance(ax, Axes)

    def test_all_nan_returns_none(self):
        df = pd.DataFrame(
            {
                "lon": [1.0, 2.0],
                "lat": [1.0, 2.0],
                "y": [np.nan, np.nan],
                "q10": [0.0, 0.0],
                "q90": [1.0, 1.0],
            }
        )
        with pytest.warns(UserWarning, match="no data after dropping NaNs"):
            ax = plot_spatial_uncertainty(df, "lon", "lat", "y", "q10", "q90")
        assert ax is None

    def test_missing_column_raises(self, uncertainty_df):
        with pytest.raises(ValueError):
            plot_spatial_uncertainty(
                uncertainty_df, "lon", "lat", "y", "q10", "missing"
            )

    def test_savefig_creates_file(self, uncertainty_df, tmp_path):
        out = tmp_path / "uncertainty.png"
        plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90", savefig=str(out)
        )
        assert out.exists()

    def test_colorbar_label_shows_nominal(self, uncertainty_df):
        ax = plot_spatial_uncertainty(
            uncertainty_df, "lon", "lat", "y", "q10", "q90", nominal=0.9
        )
        fig = ax.figure
        # Colorbar's axis label should mention "90%"
        [
            cb.ax.get_ylabel()
            for cb in fig.axes
            if hasattr(cb, "ax") and cb is not ax
        ]
        label_str = " ".join(
            ax2.get_ylabel() for ax2 in fig.axes if ax2 is not ax
        )
        assert "90" in label_str

    def test_existing_ax_reused(self, uncertainty_df):
        fig, existing_ax = plt.subplots()
        returned_ax = plot_spatial_uncertainty(
            uncertainty_df,
            "lon",
            "lat",
            "y",
            "q10",
            "q90",
            ax=existing_ax,
            legend=False,
        )
        assert returned_ax is existing_ax


# ---------------------------------------------------------------------------
# plot_spatial_coverage
# ---------------------------------------------------------------------------


class TestPlotSpatialCoverage:
    def test_returns_axes(self, coverage_df):
        ax = plot_spatial_coverage(coverage_df, "lon", "lat", "coverage")
        assert isinstance(ax, Axes)

    def test_title_set(self, coverage_df):
        ax = plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", title="Coverage map"
        )
        assert ax.get_title() == "Coverage map"

    def test_default_title_contains_nominal(self, coverage_df):
        ax = plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", nominal=0.9
        )
        assert "90" in ax.get_title()

    def test_xlabel_ylabel(self, coverage_df):
        ax = plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", xlabel="X", ylabel="Y"
        )
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

    def test_scatter_collection_created(self, coverage_df):
        ax = plot_spatial_coverage(coverage_df, "lon", "lat", "coverage")
        assert len(ax.collections) >= 1

    def test_point_count(self, coverage_df):
        ax = plot_spatial_coverage(coverage_df, "lon", "lat", "coverage")
        sc = ax.collections[0]
        assert len(sc.get_offsets()) == len(coverage_df)

    def test_annotate_creates_text(self, coverage_df):
        ax = plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", annotate=True
        )
        assert len(ax.texts) >= len(coverage_df)

    def test_tol_flag_creates_star_annotations(self):
        """All points should get a star when tol is tiny."""
        rng = np.random.default_rng(0)
        n = 10
        df = pd.DataFrame(
            {
                "lon": rng.uniform(0, 1, n),
                "lat": rng.uniform(0, 1, n),
                "coverage": np.full(n, 0.5),  # deviation = 0.5 - 0.9 = -0.4
            }
        )
        ax = plot_spatial_coverage(
            df, "lon", "lat", "coverage", nominal=0.9, tol=0.1
        )
        assert len(ax.texts) == n  # one star per flagged point

    def test_no_colorbar(self, coverage_df):
        ax = plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", colorbar=False
        )
        assert isinstance(ax, Axes)

    def test_nominal_zero_deviation(self):
        """When all coverage == nominal the diverging norm still works."""
        rng = np.random.default_rng(1)
        n = 15
        df = pd.DataFrame(
            {
                "lon": rng.uniform(0, 1, n),
                "lat": rng.uniform(0, 1, n),
                "coverage": np.full(n, 0.9),
            }
        )
        ax = plot_spatial_coverage(df, "lon", "lat", "coverage", nominal=0.9)
        assert isinstance(ax, Axes)

    def test_all_nan_returns_none(self):
        df = pd.DataFrame(
            {
                "lon": [1.0, 2.0],
                "lat": [1.0, 2.0],
                "coverage": [np.nan, np.nan],
            }
        )
        with pytest.warns(UserWarning, match="no data after dropping NaNs"):
            ax = plot_spatial_coverage(df, "lon", "lat", "coverage")
        assert ax is None

    def test_missing_column_raises(self, coverage_df):
        with pytest.raises(ValueError):
            plot_spatial_coverage(coverage_df, "lon", "lat", "missing_col")

    def test_savefig_creates_file(self, coverage_df, tmp_path):
        out = tmp_path / "coverage.png"
        plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", savefig=str(out)
        )
        assert out.exists()

    def test_existing_ax_reused(self, coverage_df):
        fig, existing_ax = plt.subplots()
        returned_ax = plot_spatial_coverage(
            coverage_df, "lon", "lat", "coverage", ax=existing_ax
        )
        assert returned_ax is existing_ax


# ---------------------------------------------------------------------------
# plot_spatial_comparison
# ---------------------------------------------------------------------------


class TestPlotSpatialComparison:
    def test_returns_list_of_axes(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df, "lon", "lat", ["cas_qar", "cas_qgbm", "cas_xtft"]
        )
        assert isinstance(axs, list)
        assert all(isinstance(a, Axes) for a in axs)

    def test_correct_panel_count_three(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df, "lon", "lat", ["cas_qar", "cas_qgbm", "cas_xtft"]
        )
        assert len(axs) == 3

    def test_correct_panel_count_two(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df, "lon", "lat", ["cas_qar", "cas_qgbm"]
        )
        assert len(axs) == 2

    def test_single_metric_col(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df, "lon", "lat", ["cas_qar"]
        )
        assert len(axs) == 1
        assert isinstance(axs[0], Axes)

    def test_names_used_as_subtitles(self, comparison_df):
        names = ["QAR", "QGBM", "XTFT"]
        axs = plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm", "cas_xtft"],
            names=names,
        )
        for ax, name in zip(axs, names):
            assert ax.get_title() == name

    def test_mismatched_names_warns_and_falls_back(self, comparison_df):
        with pytest.warns(UserWarning, match="does not match"):
            axs = plot_spatial_comparison(
                comparison_df,
                "lon",
                "lat",
                ["cas_qar", "cas_qgbm", "cas_xtft"],
                names=["only_one"],
            )
        # Titles should fall back to column names
        assert axs[0].get_title() == "cas_qar"

    def test_ncols_one_single_column(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm", "cas_xtft"],
            ncols=1,
        )
        assert len(axs) == 3

    def test_shared_scale_true(self, comparison_df):
        """All scatter collections should share the same vmin/vmax."""
        axs = plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm", "cas_xtft"],
            shared_scale=True,
        )
        vmins = [ax.collections[0].norm.vmin for ax in axs]
        vmaxs = [ax.collections[0].norm.vmax for ax in axs]
        assert len(set(vmins)) == 1
        assert len(set(vmaxs)) == 1

    def test_shared_scale_false(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm", "cas_xtft"],
            shared_scale=False,
        )
        assert len(axs) == 3

    def test_custom_vmin_vmax_with_shared_scale(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm", "cas_xtft"],
            shared_scale=True,
            vmin=0.0,
            vmax=1.0,
        )
        for ax in axs:
            assert ax.collections[0].norm.vmin == pytest.approx(0.0)
            assert ax.collections[0].norm.vmax == pytest.approx(1.0)

    def test_no_colorbar(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm"],
            colorbar=False,
        )
        assert isinstance(axs, list)

    def test_five_panels_two_rows(self, comparison_df):
        rng = np.random.default_rng(99)
        df = comparison_df.copy()
        df["cas_d"] = rng.uniform(0, 1, len(df))
        df["cas_e"] = rng.uniform(0, 1, len(df))
        cols = ["cas_qar", "cas_qgbm", "cas_xtft", "cas_d", "cas_e"]
        axs = plot_spatial_comparison(df, "lon", "lat", cols, ncols=2)
        assert len(axs) == 5

    def test_comma_string_metric_cols(self, comparison_df):
        """metric_cols can be passed as a comma-separated string (no spaces)."""
        axs = plot_spatial_comparison(
            comparison_df, "lon", "lat", "cas_qar,cas_qgbm"
        )
        assert len(axs) == 2

    def test_empty_metric_cols_raises(self, comparison_df):
        with pytest.raises(ValueError, match="at least one column"):
            plot_spatial_comparison(comparison_df, "lon", "lat", [])

    def test_missing_metric_col_raises(self, comparison_df):
        with pytest.raises(ValueError):
            plot_spatial_comparison(
                comparison_df, "lon", "lat", ["cas_qar", "nonexistent"]
            )

    def test_all_nan_returns_none(self):
        df = pd.DataFrame(
            {
                "lon": [1.0, 2.0, 3.0],
                "lat": [1.0, 2.0, 3.0],
                "m1": [np.nan, np.nan, np.nan],
            }
        )
        with pytest.warns(UserWarning, match="no data after dropping NaNs"):
            result = plot_spatial_comparison(df, "lon", "lat", ["m1"])
        assert result is None

    def test_savefig_creates_file(self, comparison_df, tmp_path):
        out = tmp_path / "comparison.png"
        plot_spatial_comparison(
            comparison_df,
            "lon",
            "lat",
            ["cas_qar", "cas_qgbm", "cas_xtft"],
            savefig=str(out),
        )
        assert out.exists()

    def test_scatter_per_panel_has_correct_point_count(self, comparison_df):
        axs = plot_spatial_comparison(
            comparison_df, "lon", "lat", ["cas_qar", "cas_qgbm", "cas_xtft"]
        )
        for ax in axs:
            sc = ax.collections[0]
            assert len(sc.get_offsets()) == len(comparison_df)


# ---------------------------------------------------------------------------
# Top-level export check
# ---------------------------------------------------------------------------


def test_top_level_exports():
    """All five functions must be importable directly from kdiagram."""
    import kdiagram as kd

    for name in [
        "plot_spatial_scatter",
        "plot_spatial_heatmap",
        "plot_spatial_uncertainty",
        "plot_spatial_coverage",
        "plot_spatial_comparison",
    ]:
        assert hasattr(kd, name), f"kdiagram.{name} not found"


def test_plot_namespace_exports():
    """All eight spatial functions must be accessible via kdiagram.plot."""
    import kdiagram.plot as kp

    for name in [
        "plot_spatial_scatter",
        "plot_spatial_heatmap",
        "plot_spatial_uncertainty",
        "plot_spatial_coverage",
        "plot_spatial_comparison",
        "plot_spatial_ordering",
        "plot_polar_from_spatial",
        "plot_paired_spatial_polar",
    ]:
        assert hasattr(kp, name), f"kdiagram.plot.{name} not found"


# ---------------------------------------------------------------------------
# Shared fixture for polar tests
# ---------------------------------------------------------------------------


@pytest.fixture
def polar_df():
    """60-point DataFrame suitable for polar diagnostic tests."""
    rng = np.random.default_rng(99)
    n = 60
    return pd.DataFrame(
        {
            "lon": rng.uniform(113.1, 113.6, n),
            "lat": rng.uniform(22.3, 22.8, n),
            "width": rng.uniform(0.5, 3.0, n),
            "width3": rng.uniform(0.5, 3.0, n) * 1.3,
            "width7": rng.uniform(0.5, 3.0, n) * 1.8,
            "order": np.arange(n, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# plot_spatial_ordering
# ---------------------------------------------------------------------------


class TestPlotSpatialOrdering:
    def test_returns_axes(self, polar_df):
        ax = plot_spatial_ordering(polar_df, "lon", "lat")
        assert isinstance(ax, Axes)

    def test_order_by_lat(self, polar_df):
        ax = plot_spatial_ordering(polar_df, "lon", "lat", order_by="lat")
        assert isinstance(ax, Axes)

    def test_order_by_lon(self, polar_df):
        ax = plot_spatial_ordering(polar_df, "lon", "lat", order_by="lon")
        assert isinstance(ax, Axes)

    def test_order_by_col(self, polar_df):
        ax = plot_spatial_ordering(polar_df, "lon", "lat", order_col="order")
        assert isinstance(ax, Axes)

    def test_no_arrows(self, polar_df):
        ax = plot_spatial_ordering(polar_df, "lon", "lat", show_arrows=False)
        assert isinstance(ax, Axes)

    def test_arrow_step(self, polar_df):
        ax = plot_spatial_ordering(
            polar_df, "lon", "lat", show_arrows=True, arrow_step=10
        )
        assert isinstance(ax, Axes)

    def test_custom_label_sites(self, polar_df):
        ax = plot_spatial_ordering(
            polar_df, "lon", "lat", label_sites=[0, 29, 59]
        )
        assert isinstance(ax, Axes)

    def test_no_colorbar(self, polar_df):
        ax = plot_spatial_ordering(polar_df, "lon", "lat", colorbar=False)
        assert isinstance(ax, Axes)

    def test_title_and_labels(self, polar_df):
        ax = plot_spatial_ordering(
            polar_df, "lon", "lat", title="My title", xlabel="X", ylabel="Y"
        )
        assert ax.get_title() == "My title"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

    def test_descending(self, polar_df):
        ax = plot_spatial_ordering(
            polar_df, "lon", "lat", order_ascending=False
        )
        assert isinstance(ax, Axes)

    def test_returns_none_on_missing_col(self, polar_df):
        with pytest.raises(ValueError):
            plot_spatial_ordering(
                polar_df, "lon", "lat", order_col="nonexistent"
            )

    def test_returns_none_all_nan(self, polar_df):
        df = polar_df.copy()
        df["lat"] = np.nan
        result = plot_spatial_ordering(df, "lon", "lat")
        assert result is None

    def test_existing_ax_reused(self, polar_df):
        fig, ax = plt.subplots()
        result = plot_spatial_ordering(polar_df, "lon", "lat", ax=ax)
        assert result is ax

    def test_savefig(self, polar_df, tmp_path):
        out = str(tmp_path / "order.png")
        plot_spatial_ordering(polar_df, "lon", "lat", savefig=out)
        import os

        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_polar_from_spatial
# ---------------------------------------------------------------------------


class TestPlotPolarFromSpatial:
    def test_returns_polar_axes(self, polar_df):
        ax = plot_polar_from_spatial(polar_df, "lon", "lat", "width")
        assert ax is not None
        # polar axes have a 'name' attribute equal to 'polar'
        assert ax.name == "polar"

    def test_order_by_lat(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", order_by="lat"
        )
        assert ax is not None

    def test_order_by_lon(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", order_by="lon"
        )
        assert ax is not None

    def test_order_col(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", order_col="order"
        )
        assert ax is not None

    def test_color_col(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", color_col="width3"
        )
        assert ax is not None

    def test_multi_horizon_ring_mode(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df,
            "lon",
            "lat",
            "width",
            horizon_cols=["width3", "width7"],
            horizon_labels=["H1", "H3", "H7"],
        )
        assert ax is not None

    def test_horizon_colors(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df,
            "lon",
            "lat",
            "width",
            horizon_cols=["width3"],
            horizon_colors=["steelblue", "tomato"],
        )
        assert ax is not None

    def test_no_colorbar(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", colorbar=False
        )
        assert ax is not None

    def test_custom_title(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", title="Polar diag"
        )
        assert "Polar diag" in ax.get_title()

    def test_no_site_labels(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", label_n_sites=0
        )
        assert ax is not None

    def test_counter_clockwise(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", clockwise=False
        )
        assert ax is not None

    def test_zero_loc_east(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", zero_loc="E"
        )
        assert ax is not None

    def test_returns_none_all_nan(self, polar_df):
        df = polar_df.copy()
        df["width"] = np.nan
        result = plot_polar_from_spatial(df, "lon", "lat", "width")
        assert result is None

    def test_missing_metric_col_raises(self, polar_df):
        with pytest.raises(ValueError):
            plot_polar_from_spatial(polar_df, "lon", "lat", "nonexistent")

    def test_savefig(self, polar_df, tmp_path):
        out = str(tmp_path / "polar.png")
        plot_polar_from_spatial(polar_df, "lon", "lat", "width", savefig=out)
        import os

        assert os.path.exists(out)

    def test_existing_polar_ax(self, polar_df):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        result = plot_polar_from_spatial(
            polar_df, "lon", "lat", "width", ax=ax
        )
        assert result is ax

    def test_horizon_string_col(self, polar_df):
        ax = plot_polar_from_spatial(
            polar_df,
            "lon",
            "lat",
            "width",
            horizon_cols="width3",
        )
        assert ax is not None


# ---------------------------------------------------------------------------
# plot_paired_spatial_polar
# ---------------------------------------------------------------------------


class TestPlotPairedSpatialPolar:
    def test_returns_list_of_two_axes(self, polar_df):
        result = plot_paired_spatial_polar(polar_df, "lon", "lat", "width")
        assert isinstance(result, list)
        assert len(result) == 2
        map_ax, pol_ax = result
        assert isinstance(map_ax, Axes)
        assert pol_ax.name == "polar"

    def test_order_by_lon(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", order_by="lon"
        )
        assert result is not None and len(result) == 2

    def test_order_col(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", order_col="order"
        )
        assert result is not None

    def test_multi_horizon_mode(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df,
            "lon",
            "lat",
            "width",
            horizon_cols=["width3", "width7"],
            horizon_labels=["H1", "H3", "H7"],
        )
        assert result is not None and len(result) == 2

    def test_map_label_sites(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df,
            "lon",
            "lat",
            "width",
            map_label_sites={0: "A", 29: "B", 59: "C"},
        )
        assert result is not None

    def test_show_ordering_arrows(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", show_ordering_arrows=True
        )
        assert result is not None

    def test_no_colorbar(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", colorbar=False
        )
        assert result is not None

    def test_custom_titles(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df,
            "lon",
            "lat",
            "width",
            title="Super",
            map_title="Map panel",
            polar_title="Polar panel",
        )
        map_ax, pol_ax = result
        assert map_ax.get_title() == "Map panel"
        assert "Polar panel" in pol_ax.get_title()

    def test_vmin_vmax(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", vmin=0.5, vmax=2.5
        )
        assert result is not None

    def test_color_col(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", color_col="width3"
        )
        assert result is not None

    def test_returns_none_all_nan(self, polar_df):
        df = polar_df.copy()
        df["width"] = np.nan
        result = plot_paired_spatial_polar(df, "lon", "lat", "width")
        assert result is None

    def test_missing_col_raises(self, polar_df):
        with pytest.raises(ValueError):
            plot_paired_spatial_polar(polar_df, "lon", "lat", "no_col")

    def test_savefig(self, polar_df, tmp_path):
        out = str(tmp_path / "paired.png")
        plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", savefig=out
        )
        import os

        assert os.path.exists(out)

    def test_descending_order(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", order_ascending=False
        )
        assert result is not None

    def test_counter_clockwise(self, polar_df):
        result = plot_paired_spatial_polar(
            polar_df, "lon", "lat", "width", clockwise=False
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Export checks — updated to cover all 8 functions
# ---------------------------------------------------------------------------


def test_top_level_exports_polar():
    """Three new polar functions must be accessible at the top-level kdiagram."""
    import kdiagram as kd

    for name in [
        "plot_spatial_ordering",
        "plot_polar_from_spatial",
        "plot_paired_spatial_polar",
    ]:
        assert hasattr(kd, name), f"kdiagram.{name} not found"
