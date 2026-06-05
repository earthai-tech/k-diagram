"""
export_figures_multi_format.py
==============================
Re-runs every CAS paper figure script and saves outputs in multiple
formats, then copies the requested vector formats to
``examples/figures/vectors/`` for Adobe Illustrator editing.

How it works
------------
The script monkey-patches ``matplotlib.figure.Figure.savefig`` *before*
running each figure script via ``runpy.run_path``.  Every call to
``fig.savefig(path)`` inside the figure script therefore also writes the
same figure in all additional requested formats alongside the original file.

Usage (from the repo root)
--------------------------
# Default: add SVG + EPS for every figure, copy to figures/vectors/
python examples/cas/scripts/export_figures_multi_format.py

# Explicit format list
python examples/cas/scripts/export_figures_multi_format.py --formats svg eps pdf

# Only specific scripts
python examples/cas/scripts/export_figures_multi_format.py \\
    --scripts figure1_rearrangement.py figA1_h_sensitivity.py

# Only copy already-generated vector files (skip re-running scripts)
python examples/cas/scripts/export_figures_multi_format.py --skip-run

Outputs
-------
data/cas/outputs/          -- all formats written alongside existing PNG/PDF
examples/figures/vectors/  -- clean-named SVG + EPS (or requested formats)
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

from matplotlib.figure import Figure

# ── repo-aware paths ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_OUTDIR = _REPO / "data" / "cas" / "outputs"
_VECTORS = _REPO / "examples" / "figures" / "vectors"

# ── figure registry ───────────────────────────────────────────────────────────
# Each entry: (script_filename, [(output_stem, clean_name), ...])
#   output_stem : filename stem the script saves (in data/cas/outputs/)
#   clean_name  : stem used in examples/figures/vectors/
FIGURE_REGISTRY: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "figure1_rearrangement.py",
        [("figure1_rearrangement", "figure1_rearrangement")],
    ),
    (
        "figure2_fan_charts.py",
        [("fig5", "figure5_fan_charts")],
    ),
    (
        "figure3_rank_stability.py",
        [("figure3_rank_stability", "figure3_rank_stability")],
    ),
    (
        "figure4_risk_index.py",
        [("figure4_risk_index", "figure4_risk_index")],
    ),
    (
        "figA1_h_sensitivity.py",
        [("figA1_h_sensitivity", "figA1_h_sensitivity")],
    ),
    (
        "figA2_lambda_crossing.py",
        [("figA2_lambda_crossing", "figA2_lambda_crossing")],
    ),
    (
        "figA3_kernel_robustness.py",
        [("figA3_kernel_robustness", "figA3_kernel_robustness")],
    ),
    (
        "figA4_horizon_profile.py",
        [("figA4_horizon_profile", "figA4_horizon_profile")],
    ),
    (
        "figA5_directional_cas.py",
        [("figA5_directional_cas", "figA5_directional_cas")],
    ),
]

# ── per-format savefig kwargs ─────────────────────────────────────────────────
# SVG  : pure vector; dpi irrelevant for vector elements
# EPS  : rasterises transparent artists at 300 dpi (EPS has no alpha channel)
# PDF  : pure vector
# PNG  : raster at 300 dpi
FORMAT_KWARGS: dict[str, dict] = {
    "svg": {},
    "eps": {"dpi": 300},
    "pdf": {},
    "png": {"dpi": 300},
    "ps": {"dpi": 300},
}

# ── matplotlib hook ───────────────────────────────────────────────────────────
_orig_savefig = Figure.savefig
_active_extra_fmts: list[str] = []


def _patched_savefig(
    self: Figure, filename: object, **kwargs: object
) -> None:
    """Drop-in replacement for Figure.savefig that also writes extra formats."""
    _orig_savefig(self, filename, **kwargs)  # original save

    p = Path(str(filename))
    current_ext = p.suffix.lstrip(".").lower()

    for fmt in _active_extra_fmts:
        if fmt == current_ext:
            continue  # already saved in this format

        extra_path = p.with_suffix(f".{fmt}")

        # Build kwargs: strip any caller-supplied 'format', merge format defaults
        kw: dict = {k: v for k, v in kwargs.items() if k != "format"}
        kw.update(FORMAT_KWARGS.get(fmt, {}))

        try:
            _orig_savefig(self, extra_path, format=fmt, **kw)
            print(f"    [{fmt.upper():>3}] {extra_path.name}")
        except Exception as exc:
            print(
                f"    [WARN] {fmt.upper()} failed for {extra_path.name}: {exc}"
            )


def _install_hook(extra_fmts: list[str]) -> None:
    global _active_extra_fmts
    _active_extra_fmts = extra_fmts
    Figure.savefig = _patched_savefig  # type: ignore[method-assign]


def _uninstall_hook() -> None:
    Figure.savefig = _orig_savefig  # type: ignore[method-assign]


# ── per-script runner ─────────────────────────────────────────────────────────


def run_figure_script(script: Path, extra_fmts: list[str]) -> bool:
    """Run one figure script with the multi-format hook active.  Returns True on success."""
    print(f"\n{'-' * 62}")
    print(f"  Script : {script.name}")
    print(f"{'-' * 62}")
    _install_hook(extra_fmts)
    try:
        runpy.run_path(str(script), run_name="__main__")
        return True
    except SystemExit:
        return True  # some scripts call sys.exit(0)
    except Exception as exc:
        print(f"  [ERROR] {script.name} raised: {exc}")
        return False
    finally:
        _uninstall_hook()


# ── copy to vectors/ ──────────────────────────────────────────────────────────


def copy_to_vectors(
    registry: list[tuple[str, list[tuple[str, str]]]],
    vector_fmts: list[str],
    vectors_dir: Path,
    outdir: Path,
) -> None:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 62}")
    print(f"  Copying vector files -> {vectors_dir}")
    print(f"{'=' * 62}")

    ok = missing = 0
    for _script, mappings in registry:
        for out_stem, clean_name in mappings:
            for fmt in vector_fmts:
                src = outdir / f"{out_stem}.{fmt}"
                dst = vectors_dir / f"{clean_name}.{fmt}"
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  OK  {clean_name}.{fmt:<4}  <-  {src.name}")
                    ok += 1
                else:
                    print(f"  XX  MISSING: {src.name}")
                    missing += 1

    print(f"\n  {ok} file(s) copied, {missing} missing.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export CAS paper figures in multiple formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["svg", "eps"],
        choices=["png", "pdf", "svg", "eps", "ps"],
        metavar="FMT",
        help="Extra formats to generate (default: svg eps).",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip re-running scripts; only copy already-generated files to vectors/.",
    )
    p.add_argument(
        "--scripts",
        nargs="+",
        metavar="SCRIPT",
        help="Run only these scripts (filename only, e.g. figure1_rearrangement.py).",
    )
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy files to examples/figures/vectors/.",
    )
    return p.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # deduplicate while preserving order
    extra_fmts: list[str] = list(dict.fromkeys(args.formats))

    # formats that should land in vectors/ (vector formats only)
    vector_fmts = [f for f in extra_fmts if f in ("svg", "eps", "pdf")]

    # filter registry if --scripts was given
    registry = FIGURE_REGISTRY
    if args.scripts:
        requested = set(args.scripts)
        registry = [(s, m) for s, m in FIGURE_REGISTRY if s in requested]
        if not registry:
            print(
                "[ERROR] None of the requested scripts matched the registry.\n"
                f"  Available: {[s for s, _ in FIGURE_REGISTRY]}"
            )
            sys.exit(1)

    print("=" * 62)
    print("  CAS paper — multi-format figure export")
    print(f"  Extra formats  : {', '.join(extra_fmts)}")
    print(f"  Scripts dir    : {_HERE}")
    print(f"  Outputs dir    : {_OUTDIR}")
    print(f"  Vectors dir    : {_VECTORS}")
    print("=" * 62)

    # ── run scripts ───────────────────────────────────────────────────────────
    if not args.skip_run:
        errors: list[str] = []
        for script_name, _ in registry:
            script = _HERE / script_name
            if not script.exists():
                print(f"\n[SKIP] {script_name} — file not found")
                continue
            ok = run_figure_script(script, extra_fmts)
            if not ok:
                errors.append(script_name)

        if errors:
            print(f"\n[WARN] {len(errors)} script(s) failed: {errors}")
    else:
        print("\n[--skip-run] Skipping script execution.")

    # ── copy to vectors/ ──────────────────────────────────────────────────────
    if not args.no_copy and vector_fmts:
        copy_to_vectors(registry, vector_fmts, _VECTORS, _OUTDIR)
    elif not vector_fmts:
        print(
            "\n[INFO] No vector formats in --formats; skipping copy to vectors/."
        )

    print("\n[Done] export_figures_multi_format.py complete.")


if __name__ == "__main__":
    main()
