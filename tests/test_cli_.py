import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import kdiagram.cli as cli


@pytest.fixture(autouse=True)
def _matplotlib_agg_backend(monkeypatch):
    # keep GUI backends out of CI
    import matplotlib

    matplotlib.use("Agg", force=True)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)


def _write_csv(path, rows):
    path = str(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if (
            isinstance(rows, (list, tuple))
            and rows
            and not isinstance(rows[0], (list, tuple))
        ):
            for r in rows:
                w.writerow([r])
        elif isinstance(rows, (list, tuple)):
            w.writerows(rows)
        else:
            f.write(str(rows))
    return path


def _stub_calls(sink):
    def _stub(*args, **kwargs):
        # ensure there's a figure to save
        plt.figure()
        sink.append({"args": args, "kwargs": kwargs})

    return _stub


def test_read_csv_to_df_success_and_errors(tmp_path, capsys):
    p = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)

    df = cli.read_csv_to_df(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

    # not found
    df2 = cli.read_csv_to_df(str(tmp_path / "missing.csv"))
    assert df2 is None
    _, err = capsys.readouterr()
    assert "File not found" in err

    # passing a directory trips the generic exception branch
    df3 = cli.read_csv_to_df(str(tmp_path))
    assert df3 is None


def test_read_csv_to_numpy_scalar_array_and_notfound(tmp_path, capsys):
    p_scalar = _write_csv(tmp_path / "val.csv", [5])
    arr_s = cli.read_csv_to_numpy(p_scalar)
    assert isinstance(arr_s, np.ndarray)
    assert arr_s.ndim == 1 and arr_s.size == 1

    p_arr = _write_csv(tmp_path / "arr.csv", [[1, 2, 3], [4, 5, 6]])
    arr = cli.read_csv_to_numpy(p_arr)
    assert arr.shape == (2, 3)

    arr2 = cli.read_csv_to_numpy(str(tmp_path / "nope.csv"))
    assert arr2 is None
    _, err = capsys.readouterr()
    assert "File not found" in err


def test_handle_figsize_valid_and_invalid(capsys):
    assert cli._handle_figsize("8,6", (1, 1)) == (8.0, 6.0)

    # non-numeric
    out = cli._handle_figsize("bad", (9, 9))
    assert out == (9, 9)
    _, err = capsys.readouterr()
    assert "Invalid format for figsize" in err

    # wrong arity
    out = cli._handle_figsize("1,2,3", (7, 7))
    assert out == (7, 7)


def test_handle_savefig_show_writes_file_and_handles_show_error(
    tmp_path, monkeypatch, capsys
):
    # save path creates a file
    plt.figure()
    out = tmp_path / "plot.png"
    cli._handle_savefig_show(str(out))
    assert out.exists() and out.stat().st_size > 0

    # simulate show() failure -> friendly stderr note
    def _boom(*a, **k):
        raise RuntimeError("no backend")

    monkeypatch.setattr(plt, "show", _boom)
    cli._handle_savefig_show(None)
    _, err = capsys.readouterr()
    assert "Use --savefig to save to file" in err


def test_cli_plot_interval_width_invokes_function_and_saves(
    tmp_path, monkeypatch
):
    p = tmp_path / "df.csv"
    pd.DataFrame({"low": [1, 2], "up": [3, 4], "z": [2, 3]}).to_csv(
        p, index=False
    )

    calls = []
    monkeypatch.setattr(cli, "plot_interval_width", _stub_calls(calls))

    out = tmp_path / "iw.png"
    argv = [
        "k-diagram",
        "plot_interval_width",
        str(p),
        "--q-cols",
        "low",
        "up",
        "--z-col",
        "z",
        "--savefig",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert calls, "plot_interval_width was not called"
    assert out.exists()


def test_cli_plot_coverage_numpy_inputs(tmp_path, monkeypatch):
    yt = _write_csv(tmp_path / "y_true.csv", [1, 2, 3])
    p1 = _write_csv(tmp_path / "p1.csv", [1, 2, 2.5])
    p2 = _write_csv(tmp_path / "p2.csv", [0.5, 2.5, 3])

    calls = []
    monkeypatch.setattr(cli, "plot_coverage", _stub_calls(calls))

    argv = [
        "k-diagram",
        "plot_coverage",
        yt,
        p1,
        p2,
        "--q",
        "0.1",
        "0.9",
        "--names",
        "A",
        "B",
        "--savefig",
        str(tmp_path / "cov.png"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert calls and isinstance(calls[0]["kwargs"].get("q"), list)


def test_cli_taylor_diagram_stats_and_arrays_and_errors(
    tmp_path, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(cli, "taylor_diagram", _stub_calls(calls))

    # stats mode (no ref-std -> defaults to 1)
    argv = [
        "k-diagram",
        "taylor_diagram",
        "--stddev",
        "0.8",
        "0.9",
        "--corrcoef",
        "0.7",
        "0.6",
        "--savefig",
        str(tmp_path / "td_stats.png"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls
    assert calls[-1]["kwargs"].get("ref_std") == 1

    # arrays mode
    yt = _write_csv(tmp_path / "ref.csv", [1, 2, 3])
    p1 = _write_csv(tmp_path / "p1.csv", [1, 2, 2.8])
    p2 = _write_csv(tmp_path / "p2.csv", [0.7, 2.2, 3.1])

    calls.clear()
    argv = [
        "k-diagram",
        "taylor_diagram",
        "--reference-file",
        yt,
        "--y-preds-files",
        p1,
        p2,
        "--names",
        "M1",
        "M2",
        "--savefig",
        str(tmp_path / "td_arrays.png"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls

    # error: both modes provided
    calls.clear()
    argv = [
        "k-diagram",
        "taylor_diagram",
        "--stddev",
        "0.8",
        "--corrcoef",
        "0.7",
        "--reference-file",
        yt,
        "--y-preds-files",
        p1,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert not calls
    _, err = capsys.readouterr()
    assert "Provide EITHER" in err or "Must provide either" in err

    # error: neither provided (parse ok, handler prints error)
    calls.clear()
    argv = [
        "k-diagram",
        "taylor_diagram",
        "--marker",
        "o",
        "--fig-size",
        "8,6",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert not calls


def test_cli_plot_feature_fingerprint_reshapes_1d(tmp_path, monkeypatch):
    p = _write_csv(tmp_path / "imp.csv", [0.1, 0.2, 0.3])

    called = []

    def _stub(importances, **kwargs):
        plt.figure()
        assert importances.ndim == 2 and importances.shape[0] == 1
        called.append(True)

    monkeypatch.setattr(cli, "plot_feature_fingerprint", _stub)

    argv = [
        "k-diagram",
        "plot_feature_fingerprint",
        p,
        "--features",
        "f1",
        "f2",
        "f3",
        "--labels",
        "L1",
        "--savefig",
        str(tmp_path / "ff.png"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert called


def test_version_flag_prints_and_exits(monkeypatch):
    argv = ["k-diagram", "--version"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as ex:
        cli.main()
    assert ex.value.code == 0
