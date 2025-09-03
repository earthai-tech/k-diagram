import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.text import Text

from kdiagram.plot._properties import place_polar_axis_labels


def _assert_text_artist(obj: Text):
    assert isinstance(obj, Text)
    # labels should be placed in axes coordinates (0..1)
    assert obj.get_transform() is obj.axes.transAxes


def test_place_polar_axis_labels_quarter_w_ccw():
    """Quarter-circle, zero at West, CCW."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        # simulate quarter coverage
        ax.set_thetamin(0)
        ax.set_thetamax(90)

        xt, yt = place_polar_axis_labels(
            ax,
            x_label="FPR",
            y_label="TPR",
            acov="quarter_circle",
            zero_location="W",
            direction=1,
            clear_default=True,  # if your helper supports this
        )

        _assert_text_artist(xt)
        _assert_text_artist(yt)

        xx, xy = xt.get_position()
        yx, yy = yt.get_position()

        # x-label roughly centered along x, slightly below the axes box
        assert 0.3 <= xx <= 0.7
        assert -0.25 <= xy <= 0.10

        # y-label near mid-height and within the axes box horizontally
        assert 0.35 <= yy <= 0.70
        assert -0.1 <= yx <= 1.1
    finally:
        plt.close(fig)


def test_place_polar_axis_labels_half_e_cw():
    """Half-circle, zero at East, clockwise direction."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        xt, yt = place_polar_axis_labels(
            ax,
            x_label="Angle",
            y_label="Radius",
            acov="half_circle",
            zero_location="E",
            direction=-1,  # clockwise
            clear_default=True,
        )

        _assert_text_artist(xt)
        _assert_text_artist(yt)

        # radial (y) label should be on the left for this config
        yx, yy = yt.get_position()
        assert yx < 0.5
        assert -0.1 <= yy <= 1.1
    finally:
        plt.close(fig)
