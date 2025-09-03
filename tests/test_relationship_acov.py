from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kdiagram.plot.relationship import (
    plot_conditional_quantiles,
    plot_error_relationship,
    plot_relationship,
    plot_residual_relationship,
)


@pytest.fixture(autouse=True)
def _use_agg_backend():
    matplotlib.use("Agg")


ACOV_DEG = {
    "default": 360.0,
    "half_circle": 180.0,
    "quarter_circle": 90.0,
    "eighth_circle": 45.0,
}


def _make_reg_data(n: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    y_true = np.linspace(-2.0, 3.0, n)
    y_pred1 = y_true + rng.normal(0.0, 0.25, size=n)
    y_pred2 = 0.8 * y_true + rng.normal(0.0, 0.35, size=n)
    return y_true, y_pred1, y_pred2


def _make_quantile_preds(
    baseline: np.ndarray,
    q=(0.1, 0.5, 0.9),
    width: float = 1.0,
):
    # shape: (n_samples, n_quantiles)
    q10 = baseline - width
    q50 = baseline
    q90 = baseline + width
    Q = np.stack([q10, q50, q90], axis=1)
    return Q, np.asarray(q, dtype=float)


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_relationship_respects_acov(acov, deg, tmp_path):
    y_true, y1, y2 = _make_reg_data()
    ax = plot_relationship(
        y_true,
        y1,
        y2,
        names=["A", "B"],
        acov=acov,
        savefig=str(tmp_path / f"rel_{acov}.png"),
    )
    assert ax.name == "polar", "Expected a polar axes."
    assert pytest.approx(ax.get_thetamax(), rel=1e-3) == deg
    # Two point clouds expected (one per model)
    assert len(ax.collections) >= 2


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_relationship_reuses_external_ax(acov, deg, tmp_path):
    y_true, y1, y2 = _make_reg_data()
    fig, ext_ax = plt.subplots(subplot_kw={"projection": "polar"})
    ret_ax = plot_relationship(
        y_true,
        y1,
        y2,
        names=["A", "B"],
        acov=acov,
        ax=ext_ax,
        savefig=str(tmp_path / f"rel_ext_{acov}.png"),
    )
    assert ret_ax is ext_ax
    assert pytest.approx(ext_ax.get_thetamax(), rel=1e-3) == deg
    assert len(ext_ax.collections) >= 2


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_conditional_quantiles_respects_acov(acov, deg, tmp_path):
    y_true, y1, _ = _make_reg_data()
    Q, qs = _make_quantile_preds(y1, q=(0.1, 0.5, 0.9), width=0.8)
    ax = plot_conditional_quantiles(
        y_true=y_true,
        y_preds_quantiles=Q,
        quantiles=qs,
        bands=[80],  # use 80% band
        acov=acov,
        savefig=str(tmp_path / f"cq_{acov}.png"),
    )
    assert ax.name == "polar"
    assert pytest.approx(ax.get_thetamax(), rel=1e-3) == deg
    # Expect at least one PolyCollection from fill_between
    assert any(
        pc.__class__.__name__.endswith("PolyCollection")
        for pc in ax.collections
    )


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_residual_relationship_respects_acov(acov, deg, tmp_path):
    y_true, y1, y2 = _make_reg_data()
    ax = plot_residual_relationship(
        y_true,
        y1,
        y2,
        acov=acov,
        savefig=str(tmp_path / f"residual_{acov}.png"),
    )
    assert ax.name == "polar"
    assert pytest.approx(ax.get_thetamax(), rel=1e-3) == deg
    # Two scatter collections expected
    assert len(ax.collections) >= 2


@pytest.mark.parametrize("acov,deg", ACOV_DEG.items())
def test_plot_error_relationship_respects_acov(acov, deg, tmp_path):
    y_true, y1, y2 = _make_reg_data()
    ax = plot_error_relationship(
        y_true,
        y1,
        y2,
        acov=acov,
        savefig=str(tmp_path / f"err_{acov}.png"),
    )
    assert ax.name == "polar"
    assert pytest.approx(ax.get_thetamax(), rel=1e-3) == deg
    # Two scatter collections expected
    assert len(ax.collections) >= 2
