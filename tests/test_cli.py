from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

# # Use a non-interactive backend
# matplotlib.use("Agg")
import kdiagram.cli as cli


@pytest.fixture(autouse=True)
def _mpl_agg(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")


@pytest.fixture
def stub_plots(monkeypatch):
    # stub every imported plot function to a sentinel
    called = {}

    def make_stub(name):
        def f(*a, **k):
            called[name] = (a, k)

        return f

    for fn in [
        "plot_coverage",
        "plot_model_drift",
        "plot_velocity",
        "plot_interval_consistency",
        "plot_anomaly_magnitude",
        "plot_uncertainty_drift",
        "plot_actual_vs_predicted",
        "plot_coverage_diagnostic",
        "plot_interval_width",
        "plot_temporal_uncertainty",
        "taylor_diagram",
        "plot_taylor_diagram_in",
        "plot_taylor_diagram",
        "plot_feature_fingerprint",
        "plot_relationship",
    ]:
        if hasattr(cli, fn):
            monkeypatch.setattr(cli, fn, make_stub(fn), raising=False)
    # stub IO helpers
    monkeypatch.setattr(
        cli,
        "read_csv_to_df",
        lambda p: pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    )
    monkeypatch.setattr(
        cli, "read_csv_to_numpy", lambda p, delimiter=",": np.array([1, 2, 3])
    )
    # stub show/savefig
    monkeypatch.setattr(cli.plt, "show", lambda: None)
    monkeypatch.setattr(cli.plt, "savefig", lambda *a, **k: None)
    return called


def run_cli(argv):
    # emulate command-line
    old = sys.argv[:]
    try:
        sys.argv = ["kdiagram"] + argv
        cli.main()
    finally:
        sys.argv = old


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        run_cli(["--version"])
    out = capsys.readouterr()
    assert "kdiagram" in out.err or out.out  # depending on argparse version


@pytest.mark.parametrize(
    "argv,called_fn",
    [
        (
            ["plot_coverage", "true.csv", "pred1.csv", "--q", "0.1", "0.9"],
            "plot_coverage",
        ),
        (
            [
                "plot_model_drift",
                "data.csv",
                "--q10-cols",
                "q10",
                "--q90-cols",
                "q90",
                "--horizons",
                "h1",
            ],
            "plot_model_drift",
        ),
        (["plot_velocity", "data.csv", "--q50-cols", "q50"], "plot_velocity"),
        (
            [
                "plot_interval_consistency",
                "data.csv",
                "--qlow-cols",
                "q10",
                "--qup-cols",
                "q90",
            ],
            "plot_interval_consistency",
        ),
        (
            [
                "plot_anomaly_magnitude",
                "data.csv",
                "--actual-col",
                "obs",
                "--q-cols",
                "q10",
                "q90",
            ],
            "plot_anomaly_magnitude",
        ),
        (
            [
                "plot_uncertainty_drift",
                "data.csv",
                "--qlow-cols",
                "q10a",
                "q10b",
                "--qup-cols",
                "q90a",
                "q90b",
            ],
            "plot_uncertainty_drift",
        ),
        (
            [
                "plot_actual_vs_predicted",
                "data.csv",
                "--actual-col",
                "obs",
                "--pred-col",
                "q50",
            ],
            "plot_actual_vs_predicted",
        ),
        (
            [
                "plot_coverage_diagnostic",
                "data.csv",
                "--actual-col",
                "obs",
                "--q-cols",
                "q10",
                "q90",
            ],
            "plot_coverage_diagnostic",
        ),
        (
            ["plot_interval_width", "data.csv", "--q-cols", "q10", "q90"],
            "plot_interval_width",
        ),
        (
            [
                "plot_temporal_uncertainty",
                "data.csv",
                "--q-cols",
                "q10",
                "q50",
                "q90",
            ],
            "plot_temporal_uncertainty",
        ),
    ],
)
def test_cli_commands_smoke(stub_plots, argv, called_fn):
    run_cli(argv)
    assert called_fn in stub_plots


def test_handle_figsize_good_and_bad(capsys, monkeypatch):
    assert cli._handle_figsize("8,8", (1, 1)) == (8.0, 8.0)
    # bad string → fallback with error printed
    r = cli._handle_figsize("oops", (9, 9))
    assert r == (9, 9)


def test_handle_savefig_show_save_error(monkeypatch, capsys):
    # simulate savefig failure
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(cli.plt, "savefig", boom)
    cli._handle_savefig_show("out.png")
    out = capsys.readouterr()
    assert "Error saving plot" in out.err


@pytest.fixture(autouse=True)
def no_gui(monkeypatch, tmp_path):
    """Silence plt.show and intercept savefig to avoid GUI/file I/O."""
    called = {"show": 0, "savefig": []}

    def fake_show(*a, **k):
        called["show"] += 1

    def fake_savefig(path, *a, **k):
        called["savefig"].append(os.fspath(path))
        # actually write something small so path exists
        with open(path, "wb") as f:
            f.write(b"PNG")

    monkeypatch.setattr(cli.plt, "show", fake_show, raising=True)
    monkeypatch.setattr(cli.plt, "savefig", fake_savefig, raising=True)
    return called


@pytest.fixture
def small_arrays(tmp_path):
    """Create simple CSVs for y_true + two preds."""
    t = tmp_path / "true.csv"
    p1 = tmp_path / "p1.csv"
    p2 = tmp_path / "p2.csv"
    t.write_text("1\n2\n3\n", encoding="utf-8")
    p1.write_text("1.1\n2.1\n3.1\n", encoding="utf-8")
    p2.write_text("0.9\n1.9\n2.9\n", encoding="utf-8")
    return t, p1, p2


@pytest.fixture
def small_df(tmp_path):
    """Create a small CSV DataFrame with quantile-ish columns."""
    df = pd.DataFrame(
        {
            "obs": [1.0, 2.0, 3.0],
            "q10_A": [0.7, 1.5, 2.6],
            "q90_A": [1.3, 2.6, 3.6],
            "q10_B": [0.6, 1.4, 2.5],
            "q90_B": [1.2, 2.5, 3.5],
            "q50_A": [1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_read_csv_helpers(tmp_path, capsys):
    # DataFrame loader ok
    p = tmp_path / "df.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
    df = cli.read_csv_to_df(str(p))
    assert list(df.columns) == ["a"]

    # Missing file -> None and stderr line
    out = cli.read_csv_to_df(str(p) + ".missing")
    assert out is None
    err = capsys.readouterr().err
    assert "File not found" in err

    # NumPy loader: vector
    q = tmp_path / "arr.csv"
    q.write_text("1\n2\n3\n", encoding="utf-8")
    arr = cli.read_csv_to_numpy(str(q))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)

    # NumPy loader: single value -> reshape to (1,)
    s = tmp_path / "single.csv"
    s.write_text("3.14\n", encoding="utf-8")
    arr1 = cli.read_csv_to_numpy(str(s))
    assert arr1.shape == (1,)


def test_plot_coverage_command(monkeypatch, small_arrays, no_gui, tmp_path):
    t, p1, p2 = small_arrays
    recorded = {}

    def fake_plot_coverage(y_true, *y_preds, names=None, q=None, **k):
        recorded["y_true"] = np.asarray(y_true)
        recorded["y_preds"] = [np.asarray(x) for x in y_preds]
        recorded["names"] = names
        recorded["q"] = q

    monkeypatch.setattr(
        cli, "plot_coverage", fake_plot_coverage, raising=True
    )
    out_png = tmp_path / "cov.png"

    argv = [
        "kdiagram",
        "plot_coverage",
        os.fspath(t),
        os.fspath(p1),
        os.fspath(p2),
        "--names",
        "A",
        "B",
        "--q",
        "0.1",
        "0.9",
        "--kind",
        "line",
        "--savefig",
        os.fspath(out_png),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    # Confirm plumbing
    assert recorded["y_true"].shape == (3,)
    assert len(recorded["y_preds"]) == 2
    assert recorded["names"] == ["A", "B"]
    assert recorded["q"] == [0.1, 0.9]
    assert os.path.exists(out_png)
    assert no_gui["savefig"] and no_gui["savefig"][-1] == os.fspath(out_png)


def test_plot_interval_consistency_figsize_fallback(monkeypatch, small_df):
    called = {}

    def fake_plot_interval_consistency(df, figsize=None, **k):
        called["figsize"] = figsize
        called["cols"] = (k.get("qlow_cols"), k.get("qup_cols"))

    monkeypatch.setattr(
        cli,
        "plot_interval_consistency",
        fake_plot_interval_consistency,
        raising=True,
    )

    argv = [
        "kdiagram",
        "plot_interval_consistency",
        os.fspath(small_df),
        "--qlow-cols",
        "q10_A",
        "q10_B",
        "--qup-cols",
        "q90_A",
        "q90_B",
        "--figsize",
        "not,a,tuple",  # invalid -> should fallback to default (9, 9)
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert called["figsize"] == (9, 9)
    assert called["cols"] == (["q10_A", "q10_B"], ["q90_A", "q90_B"])


def test_taylor_diagram_stats_mode(monkeypatch, tmp_path):
    used = {}

    def fake_taylor_diagram(**kwargs):
        used.update(kwargs)

    monkeypatch.setattr(
        cli, "taylor_diagram", fake_taylor_diagram, raising=True
    )

    argv = [
        "kdiagram",
        "taylor_diagram",
        "--stddev",
        "1.0",
        "2.0",
        "--corrcoef",
        "0.9",
        "0.8",
        "--ref-std",
        "1.1",
        "--fig-size",
        "7,5",
        "--title",
        "TD",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert used["stddev"] == [1.0, 2.0]
    assert used["corrcoef"] == [0.9, 0.8]
    assert used["ref_std"] == 1.1
    assert used["fig_size"] == (7.0, 5.0)
    assert used["title"] == "TD"


def test_taylor_diagram_arrays_mode(monkeypatch, small_arrays):
    t, p1, p2 = small_arrays
    used = {}

    def fake_taylor_diagram(**kwargs):
        used.update(kwargs)

    monkeypatch.setattr(
        cli, "taylor_diagram", fake_taylor_diagram, raising=True
    )

    argv = [
        "kdiagram",
        "taylor_diagram",
        "--reference-file",
        os.fspath(t),
        "--y-preds-files",
        os.fspath(p1),
        os.fspath(p2),
        "--marker",
        "x",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    # In arrays mode we expect 'reference' and 'y_preds' to be present
    assert "reference" in used
    assert "y_preds" in used
    assert isinstance(used["y_preds"], list)
    assert len(used["y_preds"]) == 2
    assert used["marker"] == "x"


def test_plot_taylor_diagram_in_rejects_wrong_nargs(
    monkeypatch, small_arrays, capsys
):
    """Test that argparse exits if the wrong number of args is given."""
    t_file, p1_file, _ = small_arrays  # Correctly unpack the fixture

    argv = [
        "kdiagram",
        "plot_taylor_diagram_in",
        os.fspath(t_file),
        os.fspath(p1_file),
        "--norm-range",
        "0.5",  # Only one argument, but nargs=2 is expected
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Argparse will call sys.exit(2), which raises SystemExit.
    # We catch this to confirm the parser is working correctly.
    with pytest.raises(SystemExit) as e:
        cli.main()

    assert e.value.code == 2

    # Check for the STANDARD argparse error message
    err = capsys.readouterr().err
    assert "expected 2 arguments" in err


def test_plot_taylor_diagram_in_rejects_bad_values(
    monkeypatch, small_arrays, capsys
):
    """Test custom validation for non-numeric values in --norm-range."""
    t_file, p1_file, _ = small_arrays  # Correctly unpack the fixture

    def mock_plot_func(*args, **kwargs):
        pytest.fail(
            "Plotting function should not be called with invalid args."
        )

    # We mock the plotting function itself to ensure it's not called
    # when validation fails inside the CLI helper function.
    monkeypatch.setattr(cli, "plot_taylor_diagram_in", mock_plot_func)

    argv = [
        "kdiagram",
        "plot_taylor_diagram_in",
        os.fspath(t_file),
        os.fspath(p1_file),
        "--norm-range",
        "bad",
        "values",  # Correct number of args, but bad values
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # The CLI main function should catch the error and print a message.
    # We expect it to exit gracefully.
    with pytest.raises(SystemExit) as e:
        cli.main()

    assert e.value.code == 2
    err = capsys.readouterr().err
    # Check for the standard argparse error message for invalid types.
    assert "invalid float value: 'bad'" in err


def test_plot_feature_fingerprint_matrix_and_flags(monkeypatch, tmp_path):
    used = {}

    def fake_plot_feature_fingerprint(**kwargs):
        used.update(kwargs)

    monkeypatch.setattr(
        cli,
        "plot_feature_fingerprint",
        fake_plot_feature_fingerprint,
        raising=True,
    )

    # 2x3 "matrix" CSV
    mat = tmp_path / "M.csv"
    np.savetxt(
        mat, np.array([[1, 2, 3], [4, 5, 6]], dtype=float), delimiter=","
    )

    argv = [
        "kdiagram",
        "plot_feature_fingerprint",
        os.fspath(mat),
        "--features",
        "f1",
        "f2",
        "f3",
        "--labels",
        "L1",
        "L2",
        "--no-normalize",
        "--no-fill",
        "--figsize",
        "6,6",
        "--cmap",
        "tab20",
        "--title",
        "FF",
        "--show-grid",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert used["features"] == ["f1", "f2", "f3"]
    assert used["labels"] == ["L1", "L2"]
    assert used["normalize"] is False
    assert used["fill"] is False
    assert used["figsize"] == (6.0, 6.0)
    assert used["cmap"] == "tab20"
    assert used["title"] == "FF"
    assert used["show_grid"] is True


def test_plot_relationship_with_optional_z(
    monkeypatch, small_arrays, tmp_path
):
    y_true, p1, p2 = small_arrays
    zf = tmp_path / "z.csv"
    zf.write_text("10\n20\n30\n", encoding="utf-8")
    used = {}

    def fake_plot_relationship(*a, **k):
        used["kwargs"] = k

    monkeypatch.setattr(
        cli, "plot_relationship", fake_plot_relationship, raising=True
    )

    argv = [
        "kdiagram",
        "plot_relationship",
        os.fspath(y_true),
        os.fspath(p1),
        os.fspath(p2),
        "--names",
        "A",
        "B",
        "--z-values-file",
        os.fspath(zf),
        "--xlabel",
        "rad",
        "--ylabel",
        "ang",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    k = used["kwargs"]
    assert k["names"] == ["A", "B"]
    assert k["xlabel"] == "rad"
    assert k["ylabel"] == "ang"
    # z_values is passed as array (or None if loader failed)
    assert k["z_values"] is not None


def test_version_flag(monkeypatch, capsys):
    # Make sure version prints and exits
    monkeypatch.setattr(cli.kdiagram, "__version__", "9.9.9", raising=False)
    with pytest.raises(SystemExit) as ex:
        monkeypatch.setattr(sys, "argv", ["kdiagram", "--version"])
        cli.main()
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "9.9.9" in out
