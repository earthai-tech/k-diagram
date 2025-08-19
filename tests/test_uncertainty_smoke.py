import matplotlib.pyplot as plt
import pandas as pd
import pytest

from kdiagram.plot.uncertainty import (
    plot_interval_width,
)


@pytest.fixture(autouse=True)
def mpl_agg(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)


def test_interval_width_basic(tmp_path):
    df = pd.DataFrame(
        {"low": [0, 1, 1], "up": [1, 2, 2], "theta": [0, 1, 2], "z": [0.2, 0.5, 0.7]}
    )

    pattern = r"currently ignored for positioning/ordering"

    with pytest.warns(UserWarning, match=pattern):
        plot_interval_width(
            df=df,
            q_cols=["low", "up"],
            theta_col="theta",
            z_col="z",
            acov="default",
            cbar=True,
            show_grid=True,
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
