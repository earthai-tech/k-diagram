import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

# Prefer the public import if you've re-exported it in __init__.py
import kdiagram as kd

# If not re-exported, fall back to the helper directly:
# from kdiagram.utils.plot import savefig as kd_savefig


def test_savefig_uses_current_figure_when_fig_or_ax_is_none(tmp_path):
    """When a figure exists and fig_or_ax is None, kd.savefig should
    resolve the *current* figure and save it to disk."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    out = kd.savefig(tmp_path / "out.png")  # fig_or_ax omitted
    assert out is not None
    assert out.exists()
    # Optional: ensure it saved a non-empty file
    assert out.stat().st_size > 0

    plt.close(fig)


def test_savefig_warns_and_returns_none_when_no_active_figure(tmp_path):
    """If there is no active figure and fig_or_ax is None, the helper
    should warn and return None."""
    plt.close("all")  # ensure no figures exist

    with pytest.warns(
        UserWarning, match="No active Matplotlib figure to save"
    ):
        out = kd.savefig(tmp_path / "out.png")  # fig_or_ax omitted

    assert out is None


def _is_open(fig) -> bool:
    """Return True if the given Figure is still open."""
    return fig.number in plt.get_fignums()


def test_savefig_auto_does_not_close_when_explicit_figure(tmp_path):
    """close='auto' must NOT close when a Figure is explicitly provided."""
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    out = kd.savefig(tmp_path / "exp_fig.png", fig)  # explicit Figure
    assert out is not None and out.exists()

    # Should remain open because it was not auto-fetched.
    assert _is_open(fig)
    plt.close(fig)


def test_savefig_auto_does_not_close_when_explicit_axes(tmp_path):
    """close='auto' must NOT close when an Axes is explicitly provided."""
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    out = kd.savefig(tmp_path / "exp_ax.png", ax)  # explicit Axes
    assert out is not None and out.exists()

    # Should remain open because it was not auto-fetched.
    assert _is_open(fig)
    plt.close(fig)


def test_savefig_close_true_always_closes_with_explicit_fig(tmp_path):
    """close=True must close even when a Figure is explicitly provided."""
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    out = kd.savefig(tmp_path / "force_close.png", fig, close=True)
    assert out is not None and out.exists()
    assert not _is_open(fig)  # forced close


def test_savefig_close_false_never_closes_even_auto_fetched(tmp_path):
    """close=False must not close even when the figure was auto-fetched."""
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    out = kd.savefig(tmp_path / "no_close.png", close=False)  # auto-fetched
    assert out is not None and out.exists()
    assert _is_open(fig)  # still open
    plt.close(fig)


def test_savefig_unique_name_increment_when_exists(tmp_path):
    """If overwrite=False (default) and the file exists, kd.savefig should
    create 'name (1).ext', 'name (2).ext', ..."""
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    base = tmp_path / "dup.png"
    out1 = kd.savefig(base, fig)  # saves dup.png
    assert out1.name == "dup.png" and out1.exists()
    # keep fig open to reuse path logic; saving again without overwrite should bump suffix
    out2 = kd.savefig(base, fig)
    assert out2.name in {"dup (1).png", "dup(1).png"} and out2.exists()
    plt.close(fig)
