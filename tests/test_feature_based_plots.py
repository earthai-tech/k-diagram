import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from kdiagram.plot.feature_based import plot_feature_fingerprint


def _close_ax(ax):
    try:
        plt.close(ax.figure)
    except Exception:
        pass

# def test_rejects_empty_importances_via_decorator():
#     with pytest.raises(ValueError, match="Argument 'importances' is empty"):
#         plot_feature_fingerprint([])

def test_basic_success_normalized_save(tmp_path):
    # 3 layers, 4 features
    imp = np.array([
        [1, 2, 3, 4],
        [4, 1, 0, 2],   # includes 0 to test normalization mask
        [0, 0, 0, 0],   # all zero row -> remains zeros
    ])
    out = tmp_path / "fingerprint.png"
    ax = plot_feature_fingerprint(imp, savefig=str(out))
    assert out.exists()
    # a couple of sanity checks on plot
    assert len(ax.get_xticklabels()) == 4
    _close_ax(ax)


def test_features_too_few_autofilled(tmp_path):
    imp = np.ones((2, 3))
    # provide fewer names than n_features -> auto-extend
    features = ["f1"]
    out = tmp_path / "a.png"
    ax = plot_feature_fingerprint(imp, features=features, savefig=str(out))
    assert out.exists()
    lbls = [t.get_text() for t in ax.get_xticklabels()]
    assert lbls == ["f1", "feature 2", "feature 3"]
    _close_ax(ax)


def test_features_too_many_truncated_warn(tmp_path):
    imp = np.ones((2, 2))
    features = ["f1", "f2", "f3", "f4"]
    out = tmp_path / "b.png"
    with pytest.warns(UserWarning, match="More feature names"):
        ax = plot_feature_fingerprint(imp, features=features, savefig=str(out))
    assert out.exists()
    lbls = [t.get_text() for t in ax.get_xticklabels()]
    assert lbls == ["f1", "f2"]  # truncated
    _close_ax(ax)


def test_labels_too_few_warn_and_autofill(tmp_path):
    imp = np.ones((3, 2))
    labels = ["L1"]  # too few -> warn + extend
    out = tmp_path / "c.png"
    with pytest.warns(UserWarning, match="Fewer labels .* provided"):
        ax = plot_feature_fingerprint(imp, labels=labels, savefig=str(out))
    assert out.exists()
    # should have 3 legend entries after autofill
    handles, legend_labels = ax.get_legend_handles_labels()
    assert len(legend_labels) == 3
    _close_ax(ax)


def test_labels_too_many_warn_and_truncate(tmp_path):
    imp = np.ones((2, 2))
    labels = ["A", "B", "C"]  # too many -> warn + truncate
    out = tmp_path / "d.png"
    with pytest.warns(UserWarning, match="More labels .* provided"):
        ax = plot_feature_fingerprint(imp, labels=labels, savefig=str(out))
    assert out.exists()
    handles, legend_labels = ax.get_legend_handles_labels()
    assert legend_labels == ["A", "B"]
    _close_ax(ax)


def test_cmap_list_too_short_warn_repeats(tmp_path):
    imp = np.ones((3, 3))
    # color list shorter than n_layers -> warn & repeat
    cmap_list = ["red"]
    out = tmp_path / "e.png"
    with pytest.warns(UserWarning, match="fewer colors .* than layers"):
        ax = plot_feature_fingerprint(imp, cmap=cmap_list, savefig=str(out))
    assert out.exists()
    _close_ax(ax)


def test_invalid_cmap_string_falls_back_to_tab10_warn(tmp_path):
    imp = np.ones((2, 3))
    out = tmp_path / "f.png"
    with pytest.warns(UserWarning, match="Invalid cmap 'notacmap'"):
        ax = plot_feature_fingerprint(imp, cmap="notacmap", savefig=str(out))
    assert out.exists()
    _close_ax(ax)


def test_no_normalize_no_fill_no_grid(tmp_path):
    imp = np.array([[1, 2, 3]])
    out = tmp_path / "g.png"
    ax = plot_feature_fingerprint(
        imp,
        normalize=False,
        fill=False,
        show_grid=False,
        title="NoNorm",
        savefig=str(out),
    )
    assert out.exists()
    # grid off: no visible gridlines
    x_on = any(gl.get_visible() for gl in ax.get_xgridlines())
    y_on = any(gl.get_visible() for gl in ax.get_ygridlines())
    assert not x_on and not y_on
    _close_ax(ax)



def test_importances_as_dataframe_hits_values_branch(tmp_path):
    imp_df = pd.DataFrame([[0, 1, 2], [2, 1, 0]], columns=list("abc"))
    out = tmp_path / "h.png"
    # normalize=True path uses `.values` when DataFrame is detected
    ax = plot_feature_fingerprint(imp_df, savefig=str(out))
    assert out.exists()
    _close_ax(ax)

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])