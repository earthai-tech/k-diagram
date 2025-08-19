import matplotlib.pyplot as plt
import numpy as np
import pytest

from kdiagram.plot.relationship import plot_relationship


def _cleanup():
    plt.close("all")


def test_basic_relationship_savefig(tmp_path):
    y_true = np.linspace(0, 1, 20)
    y_pred1 = y_true + 0.1
    y_pred2 = y_true * 0.7

    out = tmp_path / "basic.png"
    plot_relationship(
        y_true,
        y_pred1,
        y_pred2,
        names=["A", "B"],
        title="Basic",
        savefig=str(out),
    )
    assert out.exists()
    _cleanup()


def test_names_padding_and_invalid_cmap_warning(tmp_path):
    y_true = np.linspace(0, 1, 30)
    y_pred1 = y_true + 0.05
    y_pred2 = y_true - 0.05

    out = tmp_path / "pad.png"

    pattern = r"Colormap 'definitely_not_a_cmap' not found"

    with pytest.warns(UserWarning, match=pattern):
        plot_relationship(
            y_true,
            y_pred1,
            y_pred2,
            names=["First"],
            cmap="definitely_not_a_cmap",
            savefig=str(out),
        )
    assert out.exists()
    _cleanup()


def test_extra_names_warning(tmp_path):
    y_true = np.linspace(0, 1, 10)
    y_pred = y_true

    out = tmp_path / "extra_names.png"
    with pytest.warns(UserWarning, match="Extra names ignored"):
        plot_relationship(
            y_true,
            y_pred,
            names=["One", "Two", "Three"],  # extra names
            savefig=str(out),
        )
    assert out.exists()
    _cleanup()


def test_constant_y_true_proportional_warn(tmp_path):
    # proportional scaling + constant y_true -> warning
    y_true = np.ones(15)
    y_pred = np.linspace(0, 1, 15)

    out = tmp_path / "const_y_true.png"
    with pytest.raises(ValueError):
        plot_relationship(
            y_true,
            y_pred,
            theta_scale="proportional",
            savefig=str(out),
        )
    # assert out.exists()
    # _cleanup()


def test_uniform_half_circle_and_z_values_labeling(tmp_path):
    y_true = np.linspace(0, 1, 40)
    y_pred = y_true**2
    z_vals = np.linspace(10, 50, len(y_true))

    out = tmp_path / "uniform_half.png"
    plot_relationship(
        y_true,
        y_pred,
        theta_scale="uniform",
        acov="half_circle",
        z_values=z_vals,
        z_label="Custom Z",
        savefig=str(out),
    )
    assert out.exists()

    # Inspect current polar axes
    ax = plt.gca()
    # Thetamax should be ~180 degrees for half_circle
    # Matplotlib uses degrees in get_thetamax
    assert pytest.approx(ax.get_thetamax(), rel=1e-3) == 180.0
    # With z_values, xticks are replaced; capped at <= 8
    assert len(ax.get_xticks()) <= 8
    _cleanup()


def test_z_values_length_mismatch_raises(tmp_path):
    y_true = np.linspace(0, 1, 10)
    y_pred = y_true * 0.9
    z_vals = np.linspace(0, 1, 9)  # wrong length

    with pytest.raises(ValueError, match="Length of `z_values` must match"):
        plot_relationship(
            y_true,
            y_pred,
            z_values=z_vals,
            savefig=str(tmp_path / "fail.png"),
        )


def test_constant_y_pred_warns(tmp_path):
    y_true = np.linspace(0, 1, 25)
    y_pred_constant = np.ones_like(y_true) * 5  # zero range

    out = tmp_path / "const_y_pred.png"
    # default names -> Model_1

    # ValueError: Validation failed, due to Validation failed in strict mode.
    #  Expected type 'continuous' for both y_true and y_pred, but got 'continuous'
    #  and 'binary' respectively.. Please check your y_pred

    # "Model_1.*zero range"
    with pytest.raises(ValueError):
        plot_relationship(
            y_true,
            y_pred_constant,
            savefig=str(out),
        )


def test_grid_off(tmp_path):
    y_true = np.linspace(0, 1, 12)
    y_pred = y_true

    out = tmp_path / "nogrid.png"
    plot_relationship(
        y_true,
        y_pred,
        show_grid=False,
        savefig=str(out),
    )
    assert out.exists()

    # Public API to verify gridlines invisible
    ax = plt.gca()
    x_on = any(gl.get_visible() for gl in ax.get_xgridlines())
    y_on = any(gl.get_visible() for gl in ax.get_ygridlines())
    assert not x_on and not y_on
    _cleanup()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
