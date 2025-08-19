# tests/test_utils_plot.py
import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from kdiagram.utils.plot import is_valid_kind, set_axis_grid


# ----------------------------
# Tests for set_axis_grid
# ----------------------------
def _grid_visible(ax):
    # Consider grid "on" if any x/y gridline is visible
    return any(gl.get_visible() for gl in ax.get_xgridlines()) or any(
        gl.get_visible() for gl in ax.get_ygridlines()
    )


def test_set_axis_grid_single_on_off():
    fig, ax = plt.subplots()
    try:
        set_axis_grid(
            ax, show_grid=True, grid_props={"linestyle": "--", "alpha": 0.3}
        )
        fig.canvas.draw()  # flush
        assert _grid_visible(ax) is True

        set_axis_grid(ax, show_grid=False)
        fig.canvas.draw()  # flush again
        assert _grid_visible(ax) is False
    finally:
        plt.close(fig)


def test_set_axis_grid_multiple_axes_list():
    fig, axs = plt.subplots(1, 2)
    try:
        set_axis_grid(
            list(axs),
            show_grid=True,
            grid_props={"linestyle": ":", "alpha": 0.7},
        )
        fig.canvas.draw()
        assert all(_grid_visible(a) for a in axs)

        set_axis_grid(list(axs), show_grid=False)
        fig.canvas.draw()
        assert all(_grid_visible(a) is False for a in axs)
    finally:
        plt.close(fig)


# ----------------------------
# Tests for is_valid_kind
# ----------------------------
@pytest.mark.parametrize(
    "inp, expected",
    [
        ("ScatterPlot", "scatter"),
        ("box_graph", "box"),
        ("linechart", "line"),
        ("HEAT_MAP", "heatmap"),
        ("densityplot", "density"),
        ("areachart", "area"),
        ("plotbox", "box"),
        ("violin_plot", "violin"),
        ("barchart", "bar"),
    ],
)
def test_is_valid_kind_aliases_and_suffixes(inp, expected):
    assert is_valid_kind(inp) == expected


def test_is_valid_kind_unknown_passthrough():
    # Unknown types should normalize but pass through
    assert is_valid_kind("SpiderWeb") == "spiderweb"


def test_is_valid_kind_with_validation_accepts():
    # Should accept an alias when a valid canonical is provided
    valid = ["scatter", "line", "bar"]
    assert is_valid_kind("sCatTeRpLoT", valid_kinds=valid) == "scatter"


def test_is_valid_kind_with_validation_rejects_raises():
    with pytest.raises(ValueError):
        is_valid_kind(
            "spiderweb", valid_kinds=["scatter", "line", "bar"], error="raise"
        )


def test_is_valid_kind_with_validation_rejects_nonraise():
    # When error != 'raise', invalid kinds return normalized kind without throwing
    out = is_valid_kind(
        "spiderweb", valid_kinds=["scatter", "line", "bar"], error="warn"
    )
    assert out == "spiderweb"
