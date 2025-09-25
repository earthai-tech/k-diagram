# ruff/black-friendly: keep lines short
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kdiagram.plot._properties import (
    _POLAR_LABEL_TABLE,
    _canon_acov,
    _mirror_if_clockwise,
    _resolve_span,
    place_polar_axis_labels,
)


def test_canon_and_span_and_mirror_and_errors():
    # alias mapping (case/sep)
    assert _canon_acov("full") == "default"
    assert _canon_acov("FULL") == "default"
    assert _canon_acov("quarter") == "quarter_circle"
    assert _canon_acov("quarter-circle") == "quarter_circle"
    assert _canon_acov("eighth") == "eighth_circle"

    # spans + error branch (lines 187-198)
    assert np.isclose(_resolve_span("default"), 2 * np.pi)
    assert np.isclose(_resolve_span("half_circle"), 1 * np.pi)
    assert np.isclose(_resolve_span("quarter_circle"), 0.5 * np.pi)
    assert np.isclose(_resolve_span("eighth_circle"), 0.25 * np.pi)

    try:
        _resolve_span("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid acov")

    # mirror utility
    assert _mirror_if_clockwise((0.2, 0.3), +1) == (0.2, 0.3)
    assert _mirror_if_clockwise((0.2, 0.3), -1) == (0.8, 0.3)


def test_place_labels_clear_default_and_positions():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # pre-set labels to ensure they get cleared
    ax.set_xlabel("A")
    ax.set_ylabel("B")

    # use quarter_circle / W (no mirror; no offset)
    x_txt, y_txt = place_polar_axis_labels(
        ax,
        x_label="theta",
        y_label="r",
        acov="quarter_circle",
        zero_location="W",
        direction=1,
        clear_default=True,
    )

    # defaults cleared
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""

    # positions match the table
    base = _POLAR_LABEL_TABLE["quarter_circle"]["W"]
    xx, _ = base["x"]
    yy, _ = base["y"]
    assert x_txt.get_position() == xx
    assert y_txt.get_position() == yy

    plt.close(fig)


def test_place_labels_mirror_offset_kwargs_and_no_clear():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # keep existing labels (exercise else branch 389->394)
    ax.set_xlabel("keep-x")
    ax.set_ylabel("keep-y")

    # choose coords that change when mirrored
    base = _POLAR_LABEL_TABLE["eighth_circle"]["W"]
    (x0, y0), _ = base["x"]
    (u0, v0), _ = base["y"]

    # expected after mirror + offsets
    x_off = (0.01, 0.02)
    y_off = (0.02, -0.01)
    exp_x = (1.0 - x0 + x_off[0], y0 + x_off[1])
    exp_y = (1.0 - u0 + y_off[0], v0 + y_off[1])

    x_txt, y_txt = place_polar_axis_labels(
        ax,
        x_label="theta",
        y_label="r",
        acov="eighth_circle",
        zero_location="W",
        direction=-1,
        x_offset=x_off,
        y_offset=y_off,
        clear_default=False,
        x_kw={"color": "red", "fontsize": 9},
        y_kw={"rotation": "horizontal", "color": "blue"},
    )

    # labels not cleared
    assert ax.get_xlabel() == "keep-x"
    assert ax.get_ylabel() == "keep-y"

    # positions reflect mirror + offsets
    assert np.allclose(x_txt.get_position(), exp_x)
    assert np.allclose(y_txt.get_position(), exp_y)

    # kwarg merge / override applied
    assert x_txt.get_color() == "red"
    assert y_txt.get_color() == "blue"
    # rotation overridden to horizontal
    rot = y_txt.get_rotation()
    if isinstance(rot, str):
        assert rot.lower().startswith("h")
    else:
        # numeric angle in degrees
        assert np.isclose(float(rot), 0.0)

    plt.close(fig)


def test_place_labels_fallback_zero_location_unknown():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # pass an unknown zero_location -> falls back to 'W'
    x_txt, y_txt = place_polar_axis_labels(
        ax,
        x_label="x",
        y_label="y",
        acov="default",
        zero_location="ZZ",  # triggers fallback
    )

    base = _POLAR_LABEL_TABLE["default"]["W"]
    xx, _ = base["x"]
    yy, _ = base["y"]
    assert x_txt.get_position() == xx
    assert y_txt.get_position() == yy

    plt.close(fig)
