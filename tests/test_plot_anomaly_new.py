from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Use non-GUI backend for CI
matplotlib.use("Agg")

from kdiagram.plot import anomaly as ap

# -----------------------------
# Fixtures & helpers
# -----------------------------


def _df_with_hotspot(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    base = 20.0 * np.sin(x * np.pi / 60.0)
    y = base.copy()
    lo = base - 8.0
    up = base + 8.0
    # inject a hotspot (over-prediction)
    y[90:120] = up[90:120] + rng.uniform(6.0, 12.0, 30)
    return pd.DataFrame({"actual": y, "q_low": lo, "q_up": up})


def _df_no_anomaly(n=60, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n)
    lo = y - 5.0
    up = y + 5.0
    return pd.DataFrame({"actual": y, "q_low": lo, "q_up": up})


# -----------------------------
# plot_anomaly_severity
# -----------------------------


def test_plot_anomaly_severity_basic_and_save(tmp_path):
    df = _df_with_hotspot()
    out = tmp_path / "sev.png"
    ax = ap.plot_anomaly_severity(
        df,
        "actual",
        "q_low",
        "q_up",
        window_size=15,
        title="t",
        savefig=str(out),
        mask_angle=True,
        mask_radius=False,
    )
    # function returns Axes (not None) when anomalies exist
    assert ax is not None
    assert out.exists()


def test_plot_anomaly_severity_no_anomalies_warns():
    df = _df_no_anomaly()
    with pytest.warns(UserWarning, match="No anomalies"):
        ax = ap.plot_anomaly_severity(
            df, "actual", "q_low", "q_up", window_size=7
        )
    assert ax is None


def test_plot_anomaly_severity_empty_after_dropna_warns():
    df = pd.DataFrame(
        {"actual": [np.nan], "q_low": [np.nan], "q_up": [np.nan]}
    )
    with pytest.warns(UserWarning, match="empty after dropping"):
        ax = ap.plot_anomaly_severity(df, "actual", "q_low", "q_up")
    assert ax is None


# -----------------------------
# plot_anomaly_profile (fiery ring)
# -----------------------------


@pytest.mark.parametrize(
    "acov", ["default", "half_circle", "quarter_circle", "eighth_circle"]
)
@pytest.mark.parametrize("scale", ["linear", "sqrt", "log"])
def test_plot_anomaly_profile_variants(tmp_path, acov, scale):
    df = _df_with_hotspot()
    out = tmp_path / f"profile_{acov}_{scale}.png"
    ax = ap.plot_anomaly_profile(
        df,
        "actual",
        "q_low",
        "q_up",
        acov=acov,
        theta_bins=48,
        flare_scale=scale,
        flare_clip=3.0,
        jitter=1.0,
        max_flares_per_bin=5,
        savefig=str(out),
    )
    assert ax is not None
    assert out.exists()


def test_plot_anomaly_profile_jitter_zero_and_colors(tmp_path):
    df = _df_with_hotspot()
    out = tmp_path / "profile_j0.png"
    ax = ap.plot_anomaly_profile(
        df,
        "actual",
        "q_low",
        "q_up",
        jitter=0.0,
        colors=["#ff0000", "#0000ff"],
        savefig=str(out),
    )
    assert ax is not None
    assert out.exists()


def test_plot_anomaly_profile_bad_scale_errors():
    df = _df_with_hotspot()
    with pytest.raises(ValueError):
        ap.plot_anomaly_profile(
            df, "actual", "q_low", "q_up", flare_scale="bad"
        )


# -----------------------------
# plot_anomaly_glyphs
# -----------------------------


def test_plot_anomaly_glyphs_basic_and_save(tmp_path):
    df = _df_with_hotspot()
    out = tmp_path / "glyphs.png"
    ax = ap.plot_anomaly_glyphs(
        df,
        "actual",
        "q_low",
        "q_up",
        cmap="inferno",
        mask_angle=False,
        mask_radius=True,
        savefig=str(out),
    )
    assert ax is not None
    assert out.exists()


def test_plot_anomaly_glyphs_no_anomaly_warns():
    df = _df_no_anomaly()
    with pytest.warns(UserWarning):
        ax = ap.plot_anomaly_glyphs(df, "actual", "q_low", "q_up")
    assert ax is None


# -----------------------------
# plot_cas_profile (Cartesian)
# -----------------------------


def test_plot_cas_profile_basic_and_save(tmp_path):
    df = _df_with_hotspot()
    out = tmp_path / "profile_cart.png"
    ax = ap.plot_cas_profile(
        df,
        "actual",
        "q_low",
        "q_up",
        s=40,
        cmap="plasma",
        savefig=str(out),
    )
    assert ax is not None
    assert out.exists()


def test_plot_cas_profile_no_anomaly_warns():
    df = _df_no_anomaly()
    with pytest.warns(UserWarning):
        ax = ap.plot_cas_profile(df, "actual", "q_low", "q_up")
    assert ax is None


# -----------------------------
# _ensure_array_like / _prepare_sort_values / _order_index
# -----------------------------


def test_ensure_array_like_variants():
    k, a = ap._ensure_array_like(None)
    assert (k, a) == (None, None)
    k, a = ap._ensure_array_like("col")
    assert k == "col" and a is None
    k, a = ap._ensure_array_like([1, 2, 3])
    assert k is None and isinstance(a, np.ndarray)


def test_prepare_sort_values_dtype_paths():
    n = 12
    df = pd.DataFrame(
        {
            # Replaced: pd.date_range("2024-01-01", periods=n, freq="D")
            "dt": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(np.arange(n), unit="D"),
            "td": pd.to_timedelta(np.arange(n), unit="D"),
            "cat": pd.Categorical(list("aaabbbcccddd")),
            "b": np.tile([True, False], n // 2),
            "x": np.linspace(0, 1, n),
            "obj": [
                "1",
                "2",
                "3",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
            "z": np.arange(n),
        }
    )
    # string key (exact)
    v1 = ap._prepare_sort_values(df, "x")
    assert v1.shape == (n,)
    # case-insensitive / normalized key
    v2 = ap._prepare_sort_values(df, "DT")
    assert v2.shape == (n,)
    v3 = ap._prepare_sort_values(df, "  cat ")
    assert v3.shape == (n,)
    # array-like provided
    v4 = ap._prepare_sort_values(df, df["z"].values)
    assert (v4 == df["z"].values.astype(float)).all()

    # factorize object (non-numeric)
    with pytest.warns(UserWarning):
        v5 = ap._prepare_sort_values(df, "obj")
        assert v5.shape == (n,)

    # booleans
    v6 = ap._prepare_sort_values(df, "b")
    assert set(np.unique(v6)).issubset({0.0, 1.0})

    # timedelta path
    v7 = ap._prepare_sort_values(df, "td")
    assert v7.dtype == float

    # length mismatch error
    with pytest.raises(ValueError):
        ap._prepare_sort_values(df, np.arange(n - 1))

    # missing key error
    with pytest.raises(KeyError):
        ap._prepare_sort_values(df, "missing_col")


def test_order_index_variants_and_errors():
    df = pd.DataFrame(
        {
            "x": np.r_[5, 2, 9, 1, 7],
            # Use explicit timestamps; avoids internal range math that can overflow
            "t": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
        }
    )
    idx = ap._order_index(df, None)
    assert isinstance(idx, np.ndarray) and idx.shape == (5,)

    idx2, vals = ap._order_index(df, "t")
    assert len(idx2) == 5 and len(vals) == 5

    arr = np.array([10, 1, 3, 2, 0])
    idx3, vals3 = ap._order_index(df, arr)
    assert len(idx3) == 5 and (vals3 == arr).all()

    with pytest.raises(ValueError):
        ap._order_index(df, np.array([1, 2]))


# -----------------------------
# plot_glyphs (polar)
# -----------------------------


def test_plot_glyphs_sort_by_time_and_save(tmp_path):
    df = _df_with_hotspot()
    n = len(df)
    # Replaced: pd.date_range("2024-01-01", periods=n, freq="h")
    df["time"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.arange(n), unit="h"
    )

    out = tmp_path / "glyphs_time.png"
    ax = ap.plot_glyphs(
        df,
        "actual",
        "q_low",
        "q_up",
        sort_by="time",
        radius="magnitude",
        color_by="local_density",
        zero_at="E",
        clockwise=False,
        savefig=str(out),
    )
    assert ax is not None and out.exists()


def test_plot_glyphs_sort_by_array_none_and_errors(tmp_path):
    df = _df_with_hotspot()
    n = len(df)
    arr = np.arange(n)[::-1]
    ax = ap.plot_glyphs(
        df,
        "actual",
        "q_low",
        "q_up",
        sort_by=arr,
        radius="severity",
        color_by="severity",
        show_path=False,
    )
    assert ax is not None

    # sort_by None falls back to index
    ax2 = ap.plot_glyphs(df, "actual", "q_low", "q_up", sort_by=None)
    assert ax2 is not None

    # invalid radius
    with pytest.raises(ValueError):
        ap.plot_glyphs(df, "actual", "q_low", "q_up", radius="not_a_field")

    # invalid color_by
    with pytest.raises(ValueError):
        ap.plot_glyphs(df, "actual", "q_low", "q_up", color_by="nope")


# -----------------------------
# plot_cas_layers (Cartesian layers)
# -----------------------------


def test_plot_cas_layers_basic_sortby_none_and_save(tmp_path):
    df = _df_with_hotspot()
    out = tmp_path / "layers.png"
    axes = ap.plot_cas_layers(
        df,
        "actual",
        "q_low",
        "q_up",
        sort_by=None,
        title="layers",
        savefig=str(out),
    )
    # returns (ax, ax2) when show_severity True (default)
    assert isinstance(axes, tuple) and len(axes) == 2
    assert out.exists()


def test_plot_cas_layers_category_ticks_and_no_bottom(tmp_path):
    # two categories with distinct distributions
    nA, nB = 60, 80
    yA = np.random.normal(0, 3, nA)
    yB = np.random.normal(5, 3, nB)
    qlA, quA = yA - 6, yA + 6
    qlB, quB = yB - 4, yB + 4
    df = pd.DataFrame(
        {
            "category": ["A"] * nA + ["B"] * nB,
            "actual": np.r_[yA, yB],
            "q_low": np.r_[qlA, qlB],
            "q_up": np.r_[quA, quB],
        }
    )
    # Show single panel, no density line
    out = tmp_path / "layers_cat.png"
    ax = ap.plot_cas_layers(
        df,
        "actual",
        "q_low",
        "q_up",
        sort_by="category",
        show_severity=False,
        show_density=False,
        xlabel="cat",
        savefig=str(out),
    )
    assert ax is not None
    assert out.exists()


def test_plot_cas_layers_bad_sort_by_array_len_raises():
    df = _df_with_hotspot()
    with pytest.raises(ValueError):
        ap._order_index(df, np.array([1, 2, 3]))
