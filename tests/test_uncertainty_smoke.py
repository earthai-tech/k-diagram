# tests/test_uncertainty_smoke.py
import numpy as np
import matplotlib.pyplot as plt
import pytest

from kdiagram.plot.uncertainty import (
    plot_interval_width, plot_coverage_diagnostic,
)

@pytest.fixture(autouse=True)
def mpl_agg(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)

def test_interval_width_basic(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"low":[0,1,1], "up":[1,2,2], "theta":[0,1,2], "z":[.2,.5,.7]})
    plot_interval_width(df=df, q_cols=["low","up"], theta_col="theta",
                        z_col="z", acov="default", cbar=True, show_grid=True)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])