# License: Apache 2.0 Licence
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Command Line Interface (CLI) for the K-Diagram package.

Allows users to generate diagnostic plots directly from the command line
by providing data via CSV files.
"""


from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import package itself for version (if available)
try:
    import kdiagram
except ImportError:
    # Define a dummy if running script directly before install
    class kdiagram:
        __version__ = "unknown"

else:
    # --- Import plotting functions ---
    # Assuming the structure kdiagram -> plot -> uncertainty.py
    # Import ALL uncertainty plot functions intended for CLI
    from kdiagram.plot.evaluation import (
        plot_taylor_diagram,
        plot_taylor_diagram_in,
        taylor_diagram,
    )
    from kdiagram.plot.feature_based import plot_feature_fingerprint
    from kdiagram.plot.relationship import plot_relationship
    from kdiagram.plot.uncertainty import (
        plot_actual_vs_predicted,  
        plot_anomaly_magnitude,
        plot_coverage,
        plot_coverage_diagnostic,  
        plot_interval_consistency,
        plot_interval_width,  
        plot_model_drift,
        plot_temporal_uncertainty,  
        plot_uncertainty_drift,
        plot_velocity,
    )


# ... (rest of imports like kdiagram package for version) ...
def read_csv_to_df(filepath: str) -> pd.DataFrame | None:
    r"""
    Reads data from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}", file=sys.stderr)
        return None


def read_csv_to_numpy(filepath: str, delimiter=",") -> np.ndarray | None:
    r"""
    Reads data from a CSV file into a NumPy array.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str, optional
        Delimiter used in the CSV file, by default ','.

    Returns
    -------
    np.ndarray or None
        Loaded NumPy array, or None if an error occurs.
    """
    try:
        arr = np.genfromtxt(filepath, delimiter=delimiter)
        # Handle case where genfromtxt returns a scalar for single-value files
        if arr.ndim == 0:
            arr = arr.reshape(-1)
        print(f"Successfully loaded data from: {filepath}")
        return arr
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}", file=sys.stderr)
        return None


# --- CLI Command Functions ---


def _cli_plot_coverage(args):
    """Handler for the 'plot_coverage' command."""
    y_true = read_csv_to_numpy(args.y_true)
    if y_true is None:
        return  # Error already printed by read_csv_to_numpy

    y_preds_list = []
    for pred_file in args.y_preds_files:
        y_pred = read_csv_to_numpy(pred_file)
        if y_pred is None:
            return  # Error reading one of the prediction files
        y_preds_list.append(y_pred)

    if not y_preds_list:
        print("Error: No prediction files were successfully loaded.", file=sys.stderr)
        return

    print("Generating Coverage Plot...")
    plot_coverage(
        y_true,
        *y_preds_list,  # Unpack the list of prediction arrays
        names=args.names,
        q=args.q,
    )
    if args.savefig:
        plt.savefig(args.savefig)
        print(f"Plot saved to {args.savefig}")
    else:
        plt.show()


def _cli_plot_model_drift(args):
    """Handler for the 'plot_model_drift' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return

    print("Generating Model Drift Plot...")
    plot_model_drift(
        df=df,
        q10_cols=args.q10_cols,
        q90_cols=args.q90_cols,
        horizons=args.horizons,
        # Add other relevant args from your function signature if needed
    )
    if args.savefig:
        plt.savefig(args.savefig)
        print(f"Plot saved to {args.savefig}")
    else:
        plt.show()


def _cli_plot_velocity(args):
    """Handler for the 'plot_velocity' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return

    print("Generating Velocity Plot...")
    plot_velocity(
        df=df,
        q50_cols=args.q50_cols,
        theta_col=args.theta_col,
        # Add other relevant args from your function signature if needed
    )
    if args.savefig:
        plt.savefig(args.savefig)
        print(f"Plot saved to {args.savefig}")
    else:
        plt.show()


def _cli_plot_interval_consistency(args):
    """Handler for the 'plot_interval_consistency' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return

    # Convert figsize from string "width,height" to tuple (width, height)
    try:
        figsize = tuple(map(float, args.figsize.split(","))) if args.figsize else (9, 9)
    except ValueError:
        print(
            "Error: Invalid format for figsize. Expected 'width,height'"
            " (e.g., '8,8'). Using default (9,9).",
            file=sys.stderr,
        )
        figsize = (9, 9)

    print("Generating Interval Consistency Plot...")
    plot_interval_consistency(
        df=df,
        qlow_cols=args.qlow_cols,
        qup_cols=args.qup_cols,
        q50_cols=args.q50_cols,
        theta_col=args.theta_col,
        use_cv=args.use_cv,
        cmap=args.cmap,
        acov=args.acov,
        title=args.title,
        figsize=figsize,  # Use converted tuple
        s=args.s,
        alpha=args.alpha,
        show_grid=args.show_grid,
        mask_angle=args.mask_angle,
        # savefig handled outside
    )
    if args.savefig:
        plt.savefig(args.savefig)
        print(f"Plot saved to {args.savefig}")
    else:
        plt.show()


def _cli_plot_anomaly_magnitude(args):
    """Handler for the 'plot_anomaly_magnitude' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return

    # Convert figsize from string "width,height" to tuple (width, height)
    try:
        figsize = tuple(map(float, args.figsize.split(","))) if args.figsize else (8, 8)
    except ValueError:
        print(
            "Error: Invalid format for figsize. Expected 'width,height' (e.g., '8,8')."
            " Using default (8,8).",
            file=sys.stderr,
        )
        figsize = (8, 8)

    print("Generating Anomaly Magnitude Plot...")
    plot_anomaly_magnitude(
        df=df,
        actual_col=args.actual_col,
        q_cols=args.q_cols,  # Should be exactly two: [lower_q_col, upper_q_col]
        theta_col=args.theta_col,
        acov=args.acov,
        title=args.title,
        figsize=figsize,  # Use converted tuple
        cmap_under=args.cmap_under,
        cmap_over=args.cmap_over,
        s=args.s,
        alpha=args.alpha,
        show_grid=args.show_grid,
        verbose=args.verbose,
        cbar=args.cbar,
        mask_angle=args.mask_angle,
        # savefig handled outside
    )
    if args.savefig:
        plt.savefig(args.savefig)
        print(f"Plot saved to {args.savefig}")
    else:
        plt.show()


def _cli_plot_uncertainty_drift(args):
    """Handler for the 'plot_uncertainty_drift' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return

    # Convert figsize from string "width,height" to tuple (width, height)
    try:
        figsize = tuple(map(float, args.figsize.split(","))) if args.figsize else (9, 9)
    except ValueError:
        print(
            "Error: Invalid format for figsize. Expected 'width,height' (e.g., '8,8')."
            " Using default (9,9).",
            file=sys.stderr,
        )
        figsize = (9, 9)

    print("Generating Uncertainty Drift Plot...")
    plot_uncertainty_drift(
        df=df,
        qlow_cols=args.qlow_cols,
        qup_cols=args.qup_cols,
        dt_labels=args.dt_labels,
        theta_col=args.theta_col,
        acov=args.acov,
        base_radius=args.base_radius,
        band_height=args.band_height,
        cmap=args.cmap,
        label=args.label,
        alpha=args.alpha,
        figsize=figsize,  # Use converted tuple
        title=args.title,
        show_grid=args.show_grid,
        show_legend=args.show_legend,
        mask_angle=args.mask_angle,
        # savefig handled outside
    )
    if args.savefig:
        plt.savefig(args.savefig)
        print(f"Plot saved to {args.savefig}")
    else:
        plt.show()


def _handle_figsize(figsize_str: str, default_size: tuple) -> tuple:
    """Helper to parse figsize string 'w,h'."""
    if not figsize_str:
        return default_size
    try:
        size = tuple(map(float, figsize_str.split(",")))
        if len(size) != 2:
            raise ValueError("Figsize must have two values.")
        return size
    except ValueError as e:
        print(
            f"Error: Invalid format for figsize '{figsize_str}'. "
            f"Expected 'width,height' (e.g., '8,8'). Using default "
            f"{default_size}. Original error: {e}",
            file=sys.stderr,
        )
        return default_size


def _handle_savefig_show(savefig_path: str | None):
    """Helper to save or show plot."""
    if savefig_path:
        try:
            plt.savefig(savefig_path, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {savefig_path}")
        except Exception as e:
            print(f"Error saving plot to {savefig_path}: {e}", file=sys.stderr)
    else:
        try:
            plt.show()
        except Exception as e:
            # Handle cases where plt.show might fail (e.g., no backend)
            # but don't crash the CLI.
            print(
                f"Note: Could not display plot interactively ({e})."
                f" Use --savefig to save to file.",
                file=sys.stderr,
            )


def _cli_plot_actual_vs_predicted(args):
    """Handler for the 'plot_actual_vs_predicted' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return
    figsize = _handle_figsize(args.figsize, (8, 8))

    print("Generating Actual vs Predicted Plot...")
    # Note: actual_props and pred_props omitted from CLI for simplicity
    plot_actual_vs_predicted(
        df=df,
        actual_col=args.actual_col,
        pred_col=args.pred_col,
        theta_col=args.theta_col,
        acov=args.acov,
        figsize=figsize,
        title=args.title,
        line=args.line,
        r_label=args.r_label,
        # cmap=args.cmap, # cmap is currently unused in function
        alpha=args.alpha,
        show_grid=args.show_grid,
        show_legend=args.show_legend,
        mask_angle=args.mask_angle,
    )
    _handle_savefig_show(args.savefig)


def _cli_plot_coverage_diagnostic(args):
    """Handler for the 'plot_coverage_diagnostic' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return
    figsize = _handle_figsize(args.figsize, (8, 8))

    print("Generating Coverage Diagnostic Plot...")
    plot_coverage_diagnostic(
        df=df,
        actual_col=args.actual_col,
        q_cols=args.q_cols,
        theta_col=args.theta_col,
        acov=args.acov,
        figsize=figsize,
        title=args.title,
        show_grid=args.show_grid,
        # grid_props omitted for CLI simplicity
        cmap=args.cmap,
        alpha=args.alpha,
        s=args.s,
        as_bars=args.as_bars,
        coverage_line_color=args.coverage_line_color,
        fill_gradient=args.fill_gradient,
        gradient_cmap=args.gradient_cmap,
        # gradient_levels omitted for CLI simplicity
        mask_angle=args.mask_angle,
        verbose=args.verbose,
    )
    _handle_savefig_show(args.savefig)


def _cli_plot_interval_width(args):
    """Handler for the 'plot_interval_width' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return
    figsize = _handle_figsize(args.figsize, (8, 8))

    print("Generating Interval Width Plot...")
    plot_interval_width(
        df=df,
        q_cols=args.q_cols,
        theta_col=args.theta_col,
        z_col=args.z_col,
        acov=args.acov,
        figsize=figsize,
        title=args.title,
        cmap=args.cmap,
        s=args.s,
        alpha=args.alpha,
        show_grid=args.show_grid,
        # grid_props omitted for CLI simplicity
        cbar=args.cbar,
        mask_angle=args.mask_angle,
    )
    _handle_savefig_show(args.savefig)


def _cli_plot_temporal_uncertainty(args):
    """Handler for the 'plot_temporal_uncertainty' command."""
    df = read_csv_to_df(args.filepath)
    if df is None:
        return
    figsize = _handle_figsize(args.figsize, (8, 8))

    # Handle 'auto' q_cols case - requires detect_quantiles_in logic
    # For simplicity in CLI, we might mandate explicit columns here.
    # If 'auto' was kept:
    # if isinstance(args.q_cols, str) and args.q_cols.lower() == 'auto':
    #    from .utils.diagnose_q import detect_quantiles_in # Requires import
    #    q_cols_list = detect_quantiles_in(df)
    #    if not q_cols_list:
    #        print("Error: 'auto' detected no quantile columns.", file=sys.stderr)
    #        return
    # else:
    #    q_cols_list = args.q_cols
    # Simplified: assume args.q_cols is always a list from nargs='+'
    q_cols_list = args.q_cols

    print("Generating Temporal Uncertainty Plot...")
    plot_temporal_uncertainty(
        df=df,
        q_cols=q_cols_list,
        theta_col=args.theta_col,
        names=args.names,
        acov=args.acov,
        figsize=figsize,
        title=args.title,
        cmap=args.cmap,
        normalize=args.normalize,
        show_grid=args.show_grid,
        # grid_props omitted for CLI simplicity
        alpha=args.alpha,
        s=args.s,
        dot_style=args.dot_style,
        legend_loc=args.legend_loc,
        mask_label=args.mask_label,
        mask_angle=args.mask_angle,
    )
    _handle_savefig_show(args.savefig)


# --- Add NEW Handler Functions ---


def _cli_taylor_diagram(args):
    """Handler for the 'taylor_diagram' command."""
    # Determine input mode: stats or arrays
    use_stats_mode = args.stddev is not None and args.corrcoef is not None
    use_array_mode = args.y_preds_files is not None and args.reference_file is not None

    if use_stats_mode and use_array_mode:
        print(
            "Error: Provide EITHER --stddev/--corrcoef/--ref-std OR "
            "--y-preds-files/--reference-file, not both.",
            file=sys.stderr,
        )
        return
    if not use_stats_mode and not use_array_mode:
        print(
            "Error: Must provide either --stddev/--corrcoef/--ref-std OR "
            "--y-preds-files/--reference-file.",
            file=sys.stderr,
        )
        return

    figsize = _handle_figsize(args.fig_size, (8, 6))
    print("Generating Taylor Diagram...")

    if use_stats_mode:
        if args.ref_std is None:
            # ref_std defaults to 1 if not given with stats,
            # per function signature
            print(
                "Warning: --ref-std not provided with stats," " using default=1.",
                file=sys.stderr,
            )
        taylor_diagram(
            stddev=args.stddev,
            corrcoef=args.corrcoef,
            names=args.names,
            ref_std=args.ref_std if args.ref_std is not None else 1,
            cmap=args.cmap,
            draw_ref_arc=args.draw_ref_arc,
            radial_strategy=args.radial_strategy,
            norm_c=args.norm_c,
            power_scaling=args.power_scaling,
            marker=args.marker,
            # ref_props omitted for CLI simplicity
            fig_size=figsize,
            # size_props omitted for CLI simplicity
            title=args.title,
        )
    else:  # use_array_mode
        reference = read_csv_to_numpy(args.reference_file)
        if reference is None:
            return

        y_preds_list = []
        for pred_file in args.y_preds_files:
            y_pred = read_csv_to_numpy(pred_file)
            if y_pred is None:
                return
            y_preds_list.append(y_pred)

        if not y_preds_list:
            print("Error: No prediction files loaded successfully.", file=sys.stderr)
            return

        # ref_std will be calculated internally from reference array
        taylor_diagram(
            y_preds=y_preds_list,
            reference=reference,
            names=args.names,
            # ref_std calculated internally
            cmap=args.cmap,
            draw_ref_arc=args.draw_ref_arc,
            radial_strategy=args.radial_strategy,
            norm_c=args.norm_c,
            power_scaling=args.power_scaling,
            marker=args.marker,
            fig_size=figsize,
            title=args.title,
        )

    _handle_savefig_show(args.savefig)  # Handles show/save


def _cli_plot_taylor_diagram_in(args):
    """Handler for the 'plot_taylor_diagram_in' command."""
    reference = read_csv_to_numpy(args.reference_file)
    if reference is None:
        return

    y_preds_list = []
    for pred_file in args.y_preds_files:
        y_pred = read_csv_to_numpy(pred_file)
        if y_pred is None:
            return
        y_preds_list.append(y_pred)

    if not y_preds_list:
        print("Error: No prediction files loaded successfully.", file=sys.stderr)
        return

    figsize = _handle_figsize(args.fig_size, (10, 8))
    norm_range_tuple = None
    if args.norm_range:
        try:
            norm_range_tuple = tuple(map(float, args.norm_range))
            if len(norm_range_tuple) != 2:
                raise ValueError()
        except ValueError:
            print(
                "Error: Invalid format for --norm-range. Expected " "'min,max'.",
                file=sys.stderr,
            )
            return

    print("Generating Taylor Diagram with Background...")
    plot_taylor_diagram_in(
        *y_preds_list,
        reference=reference,
        names=args.names,
        acov=args.acov,
        zero_location=args.zero_location,
        direction=args.direction,
        only_points=args.only_points,
        ref_color=args.ref_color,
        draw_ref_arc=args.draw_ref_arc,
        angle_to_corr=args.angle_to_corr,
        marker=args.marker,
        corr_steps=args.corr_steps,
        cmap=args.cmap,
        shading=args.shading,
        shading_res=args.shading_res,
        radial_strategy=args.radial_strategy,
        norm_c=args.norm_c,
        norm_range=norm_range_tuple,
        cbar=(args.cbar == "True"),  # Convert string 'True'/'False'/'off'
        fig_size=figsize,
        title=args.title,
    )
    _handle_savefig_show(args.savefig)


def _cli_plot_taylor_diagram(args):
    """Handler for the 'plot_taylor_diagram' command."""
    reference = read_csv_to_numpy(args.reference_file)
    if reference is None:
        return

    y_preds_list = []
    for pred_file in args.y_preds_files:
        y_pred = read_csv_to_numpy(pred_file)
        if y_pred is None:
            return
        y_preds_list.append(y_pred)

    if not y_preds_list:
        print("Error: No prediction files loaded successfully.", file=sys.stderr)
        return

    figsize = _handle_figsize(args.fig_size, (10, 8))
    print("Generating Basic Taylor Diagram...")
    # Note: Signature for draw_ref_arc/angle_to_corr was '...'
    # Pass defaults or make them arguments if signature is finalized
    plot_taylor_diagram(
        *y_preds_list,
        reference=reference,
        names=args.names,
        acov=args.acov,
        zero_location=args.zero_location,
        direction=args.direction,
        only_points=args.only_points,
        ref_color=args.ref_color,
        # draw_ref_arc=True, # Example default if needed
        # angle_to_corr=True, # Example default if needed
        marker=args.marker,
        corr_steps=args.corr_steps,
        fig_size=figsize,
        title=args.title,
    )
    _handle_savefig_show(args.savefig)


def _cli_plot_feature_fingerprint(args):
    """Handler for the 'plot_feature_fingerprint' command."""
    # Assume CSV contains only the numerical importance matrix
    importances_arr = read_csv_to_numpy(args.importances_file)
    if importances_arr is None:
        return
    # Ensure it's 2D
    if importances_arr.ndim == 1:
        importances_arr = importances_arr.reshape(1, -1)  # Reshape 1D to 2D
    elif importances_arr.ndim > 2:
        print(
            f"Error: Input file '{args.importances_file}' should contain"
            " a 2D matrix of importances.",
            file=sys.stderr,
        )
        return

    figsize = _handle_figsize(args.figsize, (8, 8))
    print("Generating Feature Fingerprint Plot...")
    plot_feature_fingerprint(
        importances=importances_arr,
        features=args.features,
        labels=args.labels,
        normalize=args.normalize,
        fill=args.fill,
        cmap=args.cmap,
        title=args.title,
        figsize=figsize,
        show_grid=args.show_grid,
    )
    _handle_savefig_show(args.savefig)


def _cli_plot_relationship(args):
    """Handler for the 'plot_relationship' command."""
    y_true = read_csv_to_numpy(args.y_true_file)
    if y_true is None:
        return

    y_preds_list = []
    for pred_file in args.y_preds_files:
        y_pred = read_csv_to_numpy(pred_file)
        if y_pred is None:
            return
        y_preds_list.append(y_pred)

    if not y_preds_list:
        print("Error: No prediction files loaded successfully.", file=sys.stderr)
        return

    z_values_arr = None
    if args.z_values_file:
        z_values_arr = read_csv_to_numpy(args.z_values_file)
        if z_values_arr is None:
            return  # Error reading z_values

    figsize = _handle_figsize(args.figsize, (8, 8))
    print("Generating Relationship Plot...")
    # Note: grid_props, color_palette omitted from CLI for simplicity
    plot_relationship(
        y_true,
        *y_preds_list,
        names=args.names,
        title=args.title,
        theta_offset=args.theta_offset,
        theta_scale=args.theta_scale,
        acov=args.acov,
        figsize=figsize,
        cmap=args.cmap,
        s=args.s,
        alpha=args.alpha,
        legend=args.legend,
        show_grid=args.show_grid,
        # grid_props=None,
        # color_palette=None,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        z_values=z_values_arr,
        z_label=args.z_label,
    )
    _handle_savefig_show(args.savefig)


# --- Main CLI Parser Setup ---


def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="K-Diagram: CLI for Forecasting Uncertainty " "Visualization.",
        epilog=(
            "Example: k-diagram plot_coverage true_vals.csv "
            "preds1.csv preds2.csv --names ModelA ModelB --q 0.1 0.9"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {kdiagram.__version__}",
        # Assuming __version__ is in kdiagram/__init__.py
    )

    # Create subparsers for commands (plot types)
    # 'dest' helps identify which subparser was called
    # 'required=True' ensures a command must be provided
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="Available commands",
        metavar="<command>",
        help="Use '<command> --help' for specific command options.",
    )

    # --- Subparser: plot_coverage ---
    parser_coverage = subparsers.add_parser(
        "plot_coverage",
        help="Plot coverage score (requires NumPy arrays from CSVs).",
        description=(
            "Generates bar/line/pie/radar plot showing the empirical "
            "coverage rate(s)."  # Adjusted description slightly
        ),
    )
    parser_coverage.add_argument(
        "y_true",
        type=str,
        help="Path to CSV file with true target values (single column).",
    )
    parser_coverage.add_argument(
        "y_preds_files",
        type=str,
        nargs="+",  # One or more prediction files
        help=(
            "Paths to CSV files with predicted values/bounds. Each file "
            "represents a model or set of quantiles."
        ),
    )
    parser_coverage.add_argument(
        "--names",
        type=str,
        nargs="*",
        help="Optional names for the models corresponding to y_preds_files.",
    )
    parser_coverage.add_argument(
        "--q",
        type=float,
        nargs="+",
        required=True,
        help=(
            "Quantile levels used for the prediction intervals "
            "(e.g., 0.1 0.9 for 80%% interval)."
        ),
    )
    # --- Add arguments for plot_coverage specific options ---
    parser_coverage.add_argument(
        "--kind",
        type=str,
        default="line",
        choices=["line", "bar", "pie", "radar"],
        help="Type of plot to generate (default: 'line').",
    )
    parser_coverage.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for 'pie' or 'radar' gradient (default: 'viridis').",
    )
    parser_coverage.add_argument(
        "--cov-fill",
        action="store_true",
        default=False,
        help="Fill area in radar plot (default: False).",
    )
    parser_coverage.add_argument(
        "--figsize",
        type=str,
        default=None,  # Use function default
        help="Figure size 'width,height' (e.g., '8,6').",
    )
    parser_coverage.add_argument(
        "--title",
        type=str,
        default=None,  # Use function default
        help="Optional plot title.",
    )
    parser_coverage.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    parser_coverage.set_defaults(func=_cli_plot_coverage)  # Link handler

    # --- Subparser: plot_model_drift ---
    parser_drift = subparsers.add_parser(
        "plot_model_drift",
        help="Plot model drift over time horizons (DataFrame input).",
        description=(
            "Visualizes how prediction interval widths change across "
            "different time horizons using a polar bar chart."
        ),
    )
    parser_drift.add_argument(
        "filepath",
        type=str,
        help="Path to CSV with model predictions incl. quantile columns.",
    )
    parser_drift.add_argument(
        "--q10-cols",  # Use hyphenated names
        dest="q10_cols",  # Map to variable name
        type=str,
        nargs="+",
        required=True,
        help="Column names for lower quantile predictions (e.g., Q10).",
    )
    parser_drift.add_argument(
        "--q90-cols",
        dest="q90_cols",
        type=str,
        nargs="+",
        required=True,
        help="Column names for upper quantile predictions (e.g., Q90).",
    )
    parser_drift.add_argument(
        "--horizons",
        type=str,
        nargs="+",
        required=True,
        help="Labels for time horizons corresponding to quantile columns.",
    )
    # --- Add arguments for plot_model_drift specific options ---
    parser_drift.add_argument(
        "--color-metric-cols",
        dest="color_metric_cols",
        type=str,
        nargs="+",
        default=None,
        help="Optional columns to use for coloring bars.",
    )
    parser_drift.add_argument(
        "--acov",
        type=str,
        default="quarter_circle",  # Function default
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'quarter_circle').",
    )
    parser_drift.add_argument(
        "--value-label",
        dest="value_label",
        type=str,
        default="Uncertainty Width (Q90 - Q10)",  # Function default
        help="Label for the radial axis.",
    )
    parser_drift.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",  # Function default
        help="Colormap for bars (default: 'coolwarm').",
    )
    parser_drift.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    parser_drift.add_argument(
        "--title",
        type=str,
        default="Model Forecast Drift Over Time",  # Function default
        help="Plot title.",
    )
    parser_drift.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    parser_drift.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    parser_drift.add_argument(
        "--annotate",
        action="store_true",
        dest="annotate",
        default=True,
        help="Annotate bars with values (default).",
    )
    parser_drift.add_argument(
        "--no-annotate",
        action="store_false",
        dest="annotate",
        help="Do not annotate bars.",
    )
    parser_drift.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    parser_drift.set_defaults(func=_cli_plot_model_drift)

    # --- Subparser: plot_velocity ---
    parser_velocity = subparsers.add_parser(
        "plot_velocity",
        help="Plot velocity diagnostic (DataFrame input).",
        description="Visualizes the rate of change (velocity) of median "
        "predictions.",
    )
    parser_velocity.add_argument(
        "filepath",
        type=str,
        help="Path to CSV with predictions incl. median (Q50) columns.",
    )
    parser_velocity.add_argument(
        "--q50-cols",
        dest="q50_cols",
        type=str,
        nargs="+",
        required=True,
        help="Column names for the median (Q50) predictions.",
    )
    parser_velocity.add_argument(
        "--theta-col",
        dest="theta_col",
        type=str,
        default=None,
        help="Optional column name for angular position (ignored).",
    )
    # --- Add arguments for plot_velocity specific options ---
    parser_velocity.add_argument(
        "--cmap",
        type=str,
        default="viridis",  # Function default
        help="Colormap for points (default: 'viridis').",
    )
    parser_velocity.add_argument(
        "--acov",
        type=str,
        default="default",  # Function default
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    parser_velocity.add_argument(
        "--normalize",
        action="store_true",
        dest="normalize",
        default=True,
        help="Normalize radius (velocity) to [0, 1] (default).",
    )
    parser_velocity.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Plot raw velocity values.",
    )
    parser_velocity.add_argument(
        "--use-abs-color",
        dest="use_abs_color",
        action="store_true",
        default=True,
        help="Color points by avg Q50 magnitude (default).",
    )
    parser_velocity.add_argument(
        "--use-velocity-color",  # More intuitive flag name
        dest="use_abs_color",
        action="store_false",
        help="Color points by velocity instead of Q50 magnitude.",
    )
    parser_velocity.add_argument(
        "--figsize",
        type=str,
        default="9,9",
        help="Figure size 'width,height' (e.g., '9,9').",
    )
    parser_velocity.add_argument(
        "--title",
        type=str,
        default=None,  # Function default
        help="Optional plot title.",
    )
    parser_velocity.add_argument(
        "--s",
        type=float,  # Allow float size
        default=30,  # Function default
        help="Marker size (default: 30).",
    )
    parser_velocity.add_argument(
        "--alpha",
        type=float,
        default=0.85,  # Function default
        help="Transparency for points (default: 0.85).",
    )
    parser_velocity.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    parser_velocity.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    parser_velocity.add_argument(
        "--cbar",
        action="store_true",
        dest="cbar",
        default=True,
        help="Show color bar (default).",
    )
    parser_velocity.add_argument(
        "--no-cbar", action="store_false", dest="cbar", help="Hide color bar."
    )
    parser_velocity.add_argument(
        "--mask-angle",
        action="store_true",
        default=False,
        help="Hide angular tick labels (default: False).",
    )
    parser_velocity.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    parser_velocity.set_defaults(func=_cli_plot_velocity)

    # --- Subparser: plot_interval_consistency ---
    p_ic = subparsers.add_parser(
        "plot_interval_consistency",
        help="Plot interval consistency (DataFrame input).",
        description="Visualizes consistency of prediction interval width "
        "over time using Std Dev or CV.",
    )
    p_ic.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_ic.add_argument(
        "--qlow-cols",
        dest="qlow_cols",
        nargs="+",
        required=True,
        help="List of columns for lower quantiles (e.g., Q10).",
    )
    p_ic.add_argument(
        "--qup-cols",
        dest="qup_cols",
        nargs="+",
        required=True,
        help="List of columns for upper quantiles (e.g., Q90).",
    )
    p_ic.add_argument(
        "--q50-cols",
        dest="q50_cols",
        nargs="+",
        default=None,
        help="List of columns for median quantiles (optional, for color).",
    )
    p_ic.add_argument(
        "--theta-col",
        dest="theta_col",
        type=str,
        default=None,
        help="Column name for angular position (optional, ignored).",
    )
    p_ic.add_argument(
        "--use-cv",
        dest="use_cv",
        action="store_true",
        default=True,  # Func default
        help=("Use Coefficient of Variation (CV) for radial axis " "(default: True)."),
    )
    p_ic.add_argument(
        "--use-stddev",  # More intuitive flag name
        action="store_false",
        dest="use_cv",
        help="Use Standard Deviation for radial axis instead of CV.",
    )
    p_ic.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",  # Func default
        help="Colormap for points (default: 'coolwarm').",
    )
    p_ic.add_argument(
        "--acov",
        type=str,
        default="default",  # Func default
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_ic.add_argument(
        "--title",
        type=str,
        default=None,  # Func default uses "Prediction..."
        help="Plot title.",
    )
    p_ic.add_argument(
        "--figsize",
        type=str,
        default="9,9",
        help="Figure size 'width,height' (e.g., '9,9').",
    )
    p_ic.add_argument(
        "--s", type=int, default=30, help="Marker size (default: 30)."  # Func default
    )
    p_ic.add_argument(
        "--alpha",
        type=float,
        default=0.85,  # Func default
        help="Transparency level (default: 0.85).",
    )
    p_ic.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_ic.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_ic.add_argument(
        "--mask-angle",
        action="store_true",
        default=False,
        help="Hide angular tick labels (default: False).",
    )
    p_ic.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_ic.set_defaults(func=_cli_plot_interval_consistency)

    # --- Subparser: plot_anomaly_magnitude ---
    p_am = subparsers.add_parser(
        "plot_anomaly_magnitude",
        help="Plot anomaly magnitude (DataFrame input).",
        description="Visualizes magnitude/type of prediction anomalies "
        "(where actual is outside interval).",
    )
    p_am.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_am.add_argument(
        "--actual-col",
        dest="actual_col",
        required=True,
        type=str,
        help="Column name containing actual values.",
    )
    p_am.add_argument(
        "--q-cols",
        dest="q_cols",
        required=True,
        nargs=2,
        metavar=("LOW_Q", "UP_Q"),
        help="Two column names: lower and upper quantile bounds.",
    )
    p_am.add_argument(
        "--theta-col",
        dest="theta_col",
        type=str,
        default=None,
        help="Optional column for angular position.",
    )
    p_am.add_argument(
        "--acov",
        type=str,
        default="default",  # Func default
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_am.add_argument(
        "--title",
        type=str,
        default="Anomaly Magnitude Polar Plot",  # Func default
        help="Plot title.",
    )
    p_am.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_am.add_argument(
        "--cmap-under",
        dest="cmap_under",
        type=str,
        default="Blues",  # Func default
        help="Colormap for under-predictions (default: 'Blues').",
    )
    p_am.add_argument(
        "--cmap-over",
        dest="cmap_over",
        type=str,
        default="Reds",  # Func default
        help="Colormap for over-predictions (default: 'Reds').",
    )
    p_am.add_argument(
        "--s", type=int, default=30, help="Marker size (default: 30)."  # Func default
    )
    p_am.add_argument(
        "--alpha",
        type=float,
        default=0.8,  # Func default
        help="Transparency level (default: 0.8).",
    )
    p_am.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_am.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_am.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1],  # Func default is 1
        help="Verbosity level (0: silent, 1: print counts) (default: 1).",
    )
    p_am.add_argument(
        "--cbar",
        action="store_true",
        default=False,
        help="Show color bar (default: False).",
    )
    p_am.add_argument(
        "--mask-angle",
        action="store_true",
        default=False,
        help="Hide angular tick labels (default: False).",
    )
    p_am.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_am.set_defaults(func=_cli_plot_anomaly_magnitude)

    # --- Subparser: plot_uncertainty_drift ---
    p_ud = subparsers.add_parser(
        "plot_uncertainty_drift",
        help="Plot uncertainty drift using rings (DataFrame input).",
        description=(
            "Visualizes how prediction interval width patterns "
            "evolve over time steps using concentric rings."
        ),
    )
    p_ud.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_ud.add_argument(
        "--qlow-cols",
        dest="qlow_cols",
        required=True,
        nargs="+",
        help="List of lower quantile columns, one per time step.",
    )
    p_ud.add_argument(
        "--qup-cols",
        dest="qup_cols",
        required=True,
        nargs="+",
        help="List of upper quantile columns, one per time step.",
    )
    p_ud.add_argument(
        "--dt-labels",
        dest="dt_labels",
        nargs="+",
        default=None,
        help="Optional labels for time steps (must match number of cols).",
    )
    p_ud.add_argument(
        "--theta-col",
        dest="theta_col",
        type=str,
        default=None,
        help="Optional column for angular position (ignored).",
    )
    p_ud.add_argument(
        "--acov",
        type=str,
        default="default",  # Func default
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_ud.add_argument(
        "--base-radius",
        dest="base_radius",
        type=float,
        default=0.15,  # Func default
        help="Base radius for innermost ring (default: 0.15).",
    )
    p_ud.add_argument(
        "--band-height",
        dest="band_height",
        type=float,
        default=0.15,  # Func default
        help="Scaling factor for width effect on radius (default: 0.15).",
    )
    p_ud.add_argument(
        "--cmap",
        type=str,
        default="tab10",  # Func default
        help="Colormap for rings (default: 'tab10').",
    )
    p_ud.add_argument(
        "--label",
        type=str,
        default="Time Step",  # Func default
        help="Label prefix for legend entries (default: 'Time Step').",
    )
    p_ud.add_argument(
        "--alpha",
        type=float,
        default=0.85,  # Func default
        help="Transparency for lines (default: 0.85).",
    )
    p_ud.add_argument(
        "--figsize",
        type=str,
        default="9,9",
        help="Figure size 'width,height' (e.g., '9,9').",
    )
    p_ud.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_ud.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_ud.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_ud.add_argument(
        "--show-legend",
        action="store_true",
        dest="show_legend",
        default=True,
        help="Show legend (default).",
    )
    p_ud.add_argument(
        "--no-show-legend",
        action="store_false",
        dest="show_legend",
        help="Hide legend.",
    )
    p_ud.add_argument(
        "--mask-angle",  # Renamed from mute_degree in func
        dest="mask_angle",
        action="store_true",
        default=True,  # Func default was mute_degree=True
        help="Hide angular tick labels (default).",
    )
    p_ud.add_argument(
        "--show-angle-labels",  # More intuitive opposite flag
        action="store_false",
        dest="mask_angle",
        help="Show angular tick labels.",
    )
    p_ud.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_ud.set_defaults(func=_cli_plot_uncertainty_drift)

    # --- Add subparsers for Evaluation, Feature-Based, Relationship ---
    # (Defined in the previous response - Omitted here for brevity)
    # p_td = subparsers.add_parser("taylor_diagram", ...)
    # p_tdi = subparsers.add_parser("plot_taylor_diagram_in", ...)
    # p_tdb = subparsers.add_parser("plot_taylor_diagram", ...)
    # p_ff = subparsers.add_parser("plot_feature_fingerprint", ...)
    # p_rel = subparsers.add_parser("plot_relationship", ...)
    # --- END Other Subparsers ---

    # --- Parse Arguments and Execute ---
    # (Keep existing logic)
    # ...

    # --- Entry Point Guard ---
    # (Keep existing logic)
    # ...
    # def main():
    #     # Main parser
    #     parser = argparse.ArgumentParser(
    #         description="K-Diagram: CLI for Forecasting Uncertainty Visualization.",
    #         epilog=("Example: k-diagram plot_anomaly_magnitude data.csv "
    #                 "--actual-col=obs --q-cols=p10 p90 --savefig=plot.png")
    #     )
    #     parser.add_argument(
    #         '--version', action='version',
    #         version=f'%(prog)s {kdiagram.__version__}'
    #     )

    #     # Create subparsers for commands (plot types)
    #     subparsers = parser.add_subparsers(
    #         dest="command",
    #         required=True,
    #         title="Available commands (use <command> --help for details)",
    #         metavar="<command>",
    #     )

    # --- Add Subparsers for ALL Uncertainty Plots ---

    # (Existing subparsers for plot_coverage, plot_model_drift,
    #  plot_velocity, plot_interval_consistency, plot_anomaly_magnitude,
    #  plot_uncertainty_drift are assumed here - Omitted for brevity)

    # --- Subparser: plot_actual_vs_predicted ---
    p_avp = subparsers.add_parser(
        "plot_actual_vs_predicted",
        help="Compare actual vs predicted values point-by-point.",
        description="Generates a polar plot comparing actual observations "
        "against point predictions (e.g., Q50).",
    )
    p_avp.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_avp.add_argument(
        "--actual-col",
        required=True,
        type=str,
        help="Column name containing the actual observed values.",
    )
    p_avp.add_argument(
        "--pred-col",
        required=True,
        type=str,
        help="Column name containing the predicted values (e.g., Q50).",
    )
    p_avp.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Optional column for angular position (currently ignored).",
    )
    p_avp.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_avp.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_avp.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_avp.add_argument(
        "--line",
        action="store_true",
        default=True,  # Plot lines by default
        help="Plot data as lines (default).",
    )
    p_avp.add_argument(
        "--no-line",
        action="store_false",
        dest="line",
        help="Plot data as scatter points instead of lines.",
    )
    p_avp.add_argument(
        "--r-label", type=str, default=None, help="Label for the radial axis."
    )
    p_avp.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Transparency for difference lines (default: 0.3).",
    )
    p_avp.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_avp.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_avp.add_argument(
        "--show-legend",
        action="store_true",
        dest="show_legend",
        default=True,
        help="Show legend (default).",
    )
    p_avp.add_argument(
        "--no-show-legend",
        action="store_false",
        dest="show_legend",
        help="Hide legend.",
    )
    p_avp.add_argument(
        "--mask-angle",
        action="store_true",
        default=False,
        help="Hide angular tick labels.",
    )
    p_avp.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_avp.set_defaults(func=_cli_plot_actual_vs_predicted)

    # --- Subparser: plot_coverage_diagnostic ---
    p_cd = subparsers.add_parser(
        "plot_coverage_diagnostic",
        help="Diagnose interval coverage point-by-point.",
        description="Generates a polar plot showing whether each actual "
        "value falls within its prediction interval.",
    )
    p_cd.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_cd.add_argument(
        "--actual-col",
        required=True,
        type=str,
        help="Column name containing the actual observed values.",
    )
    p_cd.add_argument(
        "--q-cols",
        required=True,
        nargs=2,
        metavar=("LOW_Q", "UP_Q"),
        help="Two column names: lower and upper quantile bounds.",
    )
    p_cd.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Optional column for angular position (currently ignored).",
    )
    p_cd.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_cd.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_cd.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_cd.add_argument(
        "--cmap",
        type=str,
        default="RdYlGn",
        help="Colormap for coverage points/bars (default: 'RdYlGn').",
    )
    p_cd.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Transparency for points/bars (default: 0.85).",
    )
    p_cd.add_argument(
        "--s",
        type=int,
        default=35,
        help="Marker size if using scatter points (default: 35).",
    )
    p_cd.add_argument(
        "--as-bars",
        action="store_true",
        default=False,
        help="Display coverage as bars instead of scatter points.",
    )
    p_cd.add_argument(
        "--coverage-line-color",
        type=str,
        default="r",
        help="Color for the average coverage line (default: 'r').",
    )
    p_cd.add_argument(
        "--fill-gradient",
        action="store_true",
        dest="fill_gradient",
        default=True,
        help="Fill background with gradient up to avg coverage (default).",
    )
    p_cd.add_argument(
        "--no-fill-gradient",
        action="store_false",
        dest="fill_gradient",
        help="Do not fill background with gradient.",
    )
    p_cd.add_argument(
        "--gradient-cmap",
        type=str,
        default="Greens",
        help="Colormap for background gradient (default: 'Greens').",
    )
    p_cd.add_argument(
        "--mask-angle",
        action="store_true",
        default=True,  # Default True here
        help="Hide angular tick labels (default).",
    )
    p_cd.add_argument(
        "--no-mask-angle",
        action="store_false",
        dest="mask_angle",
        help="Show angular tick labels.",
    )
    p_cd.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_cd.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_cd.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="Verbosity level (0: silent, 1: print coverage) (default: 0).",
    )
    p_cd.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_cd.set_defaults(func=_cli_plot_coverage_diagnostic)

    # --- Subparser: plot_interval_width ---
    p_iw = subparsers.add_parser(
        "plot_interval_width",
        help="Visualize prediction interval width across samples.",
        description="Generates a polar scatter plot where radius is the "
        "interval width (Qup-Qlow).",
    )
    p_iw.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_iw.add_argument(
        "--q-cols",
        required=True,
        nargs=2,
        metavar=("LOW_Q", "UP_Q"),
        help="Two column names: lower and upper quantile bounds.",
    )
    p_iw.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Optional column for angular position (currently ignored).",
    )
    p_iw.add_argument(
        "--z-col",
        type=str,
        default=None,
        help="Optional column name for color mapping (e.g., Q50).",
    )
    p_iw.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_iw.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_iw.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_iw.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for points (used for z-col or radius) " "(default: 'viridis').",
    )
    p_iw.add_argument("--s", type=int, default=30, help="Marker size (default: 30).")
    p_iw.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Transparency for points (default: 0.8).",
    )
    p_iw.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_iw.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_iw.add_argument(
        "--cbar",
        action="store_true",
        dest="cbar",
        default=True,
        help="Show color bar (default, relevant if z-col used).",
    )
    p_iw.add_argument(
        "--no-cbar", action="store_false", dest="cbar", help="Hide color bar."
    )
    p_iw.add_argument(
        "--mask-angle",
        action="store_true",
        default=True,  # Default True
        help="Hide angular tick labels (default).",
    )
    p_iw.add_argument(
        "--no-mask-angle",
        action="store_false",
        dest="mask_angle",
        help="Show angular tick labels.",
    )
    p_iw.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_iw.set_defaults(func=_cli_plot_interval_width)

    # --- Subparser: plot_temporal_uncertainty ---
    p_tu = subparsers.add_parser(
        "plot_temporal_uncertainty",
        help="General polar scatter plot for multiple data series.",
        description="Visualizes multiple data columns (e.g., different "
        "quantiles for one time step) on a polar plot.",
    )
    p_tu.add_argument("filepath", type=str, help="Path to the CSV data file.")
    p_tu.add_argument(
        "--q-cols",
        required=True,
        nargs="+",
        help="List of column names to plot as different series.",
    )
    p_tu.add_argument(
        "--theta-col",
        type=str,
        default=None,
        help="Optional column (used for NaN alignment, ignored for pos).",
    )
    p_tu.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for each series in q-cols (for legend).",
    )
    p_tu.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_tu.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_tu.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_tu.add_argument(
        "--cmap",
        type=str,
        default="tab10",
        help="Colormap used to color different series (default: 'tab10').",
    )
    p_tu.add_argument(
        "--normalize",
        action="store_true",
        dest="normalize",
        default=True,
        help="Normalize each series to [0, 1] radially (default).",
    )
    p_tu.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Plot raw values instead of normalizing radially.",
    )
    p_tu.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_tu.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_tu.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency for points (default: 0.7).",
    )
    p_tu.add_argument("--s", type=int, default=25, help="Marker size (default: 25).")
    p_tu.add_argument(
        "--dot-style",
        type=str,
        default="o",
        help="Marker style for points (e.g., 'o', 'x', '^') (default: 'o').",
    )
    p_tu.add_argument(
        "--legend-loc",
        type=str,
        default="upper right",
        help="Location for the legend (default: 'upper right').",
    )
    p_tu.add_argument(
        "--mask-label",
        action="store_true",
        default=False,
        help="Hide radial tick labels.",
    )
    p_tu.add_argument(
        "--mask-angle",
        action="store_true",
        default=True,  # Default True
        help="Hide angular tick labels (default).",
    )
    p_tu.add_argument(
        "--no-mask-angle",
        action="store_false",
        dest="mask_angle",
        help="Show angular tick labels.",
    )
    p_tu.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_tu.set_defaults(func=_cli_plot_temporal_uncertainty)

    # cli.py -> main() function

    # ... (Main parser and subparsers setup as before) ...
    # --- Subparsers for Uncertainty Plots ---
    # (Keep all 10 subparsers defined previously)
    # parser_coverage = subparsers.add_parser(...)
    # parser_drift = subparsers.add_parser(...)
    # parser_velocity = subparsers.add_parser(...)
    # parser_consistency = subparsers.add_parser(...)
    # parser_anomaly = subparsers.add_parser(...)
    # parser_unc_drift = subparsers.add_parser(...)
    # p_avp = subparsers.add_parser(...)
    # p_cd = subparsers.add_parser(...)
    # p_iw = subparsers.add_parser(...)
    # p_tu = subparsers.add_parser(...)
    # --- END Uncertainty Subparsers ---

    # --- Subparser: taylor_diagram ---
    p_td = subparsers.add_parser(
        "taylor_diagram",
        help="Flexible Taylor Diagram (stats or arrays input).",
        description="Generates a Taylor Diagram comparing predictions to a "
        "reference using standard deviation and correlation. "
        "Accepts pre-calculated stats or raw data arrays.",
    )
    # --- Input Group (Mutually Exclusive Logic handled in handler) ---
    p_td.add_argument(
        "--stddev",
        type=float,
        nargs="+",
        default=None,
        help="List of standard deviations (use with --corrcoef, --ref-std).",
    )
    p_td.add_argument(
        "--corrcoef",
        type=float,
        nargs="+",
        default=None,
        help="List of correlation coefficients (use with --stddev, --ref-std).",
    )
    p_td.add_argument(
        "--ref-std",
        type=float,
        default=None,
        help="Standard deviation of the reference (use with --stddev/corrcoef "
        "or calculated automatically if using arrays).",
    )
    p_td.add_argument(
        "--reference-file",
        type=str,
        default=None,
        metavar="FILEPATH",
        help="Path to CSV file with reference values (use with --y-preds-files).",
    )
    p_td.add_argument(
        "--y-preds-files",
        type=str,
        nargs="+",
        default=None,
        metavar="FILEPATH",
        help="Paths to CSV files with prediction values (use with --reference-file).",
    )
    # --- END Input Group ---
    p_td.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for the models/predictions.",
    )
    p_td.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Optional colormap name for background shading.",
    )
    p_td.add_argument(
        "--draw-ref-arc",
        action="store_true",
        default=False,
        help="Draw reference std dev as an arc instead of a point.",
    )
    p_td.add_argument(
        "--no-draw-ref-arc",
        action="store_false",
        dest="draw_ref_arc",
        help="Draw reference std dev as a point (default).",
    )
    p_td.add_argument(
        "--radial-strategy",
        type=str,
        default="rwf",
        choices=["rwf", "convergence", "center_focus", "performance"],
        help="Strategy for background mesh generation (default: 'rwf').",
    )
    p_td.add_argument(
        "--norm-c",
        action="store_true",
        default=False,
        help="Normalize background mesh colors.",
    )
    p_td.add_argument(
        "--power-scaling",
        type=float,
        default=1.0,
        help="Exponent for background normalization (default: 1.0).",
    )
    p_td.add_argument(
        "--marker",
        type=str,
        default="o",
        help="Marker style for prediction points (default: 'o').",
    )
    p_td.add_argument(
        "--fig-size",
        type=str,
        default="8,6",
        help="Figure size 'width,height' (e.g., '8,6').",
    )
    p_td.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_td.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_td.set_defaults(func=_cli_taylor_diagram)

    # --- Subparser: plot_taylor_diagram_in ---
    p_tdi = subparsers.add_parser(
        "plot_taylor_diagram_in",
        help="Taylor Diagram with background shading.",
        description="Generates a Taylor Diagram with a background colormap "
        "based on correlation or performance. Requires raw data arrays.",
    )
    p_tdi.add_argument(
        "reference_file",
        type=str,
        metavar="REFERENCE_CSV",
        help="Path to CSV file with reference values.",
    )
    p_tdi.add_argument(
        "y_preds_files",
        type=str,
        nargs="+",
        metavar="PREDICTION_CSV",
        help="Paths to CSV files with prediction values.",
    )
    p_tdi.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for the predictions.",
    )
    p_tdi.add_argument(
        "--acov",
        type=str,
        default="half_circle",
        choices=["default", "half_circle"],
        help=(
            "Angular coverage ('default': pi, 'half_circle': pi/2)"
            "(default: 'half_circle')."
        ),
    )
    p_tdi.add_argument(
        "--zero-location",
        type=str,
        default="E",
        choices=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        help="Position of correlation=1 (default: 'E').",
    )
    p_tdi.add_argument(
        "--direction",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="Angle direction (-1: clockwise, 1: counter-clockwise) (default: -1).",
    )
    p_tdi.add_argument(
        "--only-points",
        action="store_true",
        default=False,
        help="Plot only markers, no radial lines to origin.",
    )
    p_tdi.add_argument(
        "--ref-color",
        type=str,
        default="red",
        help="Color for the reference arc/line (default: 'red').",
    )
    p_tdi.add_argument(
        "--draw-ref-arc",
        action="store_true",
        dest="draw_ref_arc",
        default=True,
        help="Draw reference std dev as an arc (default).",
    )
    p_tdi.add_argument(
        "--no-draw-ref-arc",
        action="store_false",
        dest="draw_ref_arc",
        help="Draw reference std dev as a point/line instead of arc.",
    )
    p_tdi.add_argument(
        "--angle-to-corr",
        action="store_true",
        dest="angle_to_corr",
        default=True,
        help="Label angular axis with correlation values (default).",
    )
    p_tdi.add_argument(
        "--no-angle-to-corr",
        action="store_false",
        dest="angle_to_corr",
        help="Label angular axis with degrees.",
    )
    p_tdi.add_argument(
        "--marker",
        type=str,
        default="o",
        help="Marker style for prediction points (default: 'o').",
    )
    p_tdi.add_argument(
        "--corr-steps",
        type=int,
        default=6,
        help="Number of correlation ticks if angle_to_corr=True (default: 6).",
    )
    p_tdi.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for background shading (default: 'viridis').",
    )
    p_tdi.add_argument(
        "--shading",
        type=str,
        default="auto",
        choices=["auto", "gouraud", "nearest"],
        help="Background shading method (default: 'auto').",
    )
    p_tdi.add_argument(
        "--shading-res",
        type=int,
        default=300,
        help="Resolution for background mesh (default: 300).",
    )
    p_tdi.add_argument(
        "--radial-strategy",
        type=str,
        default="performance",
        choices=["convergence", "norm_r", "performance"],
        help="Strategy for background calculation (default: 'performance').",
    )
    p_tdi.add_argument(
        "--norm-c",
        action="store_true",
        default=False,
        help="Normalize background colors.",
    )
    p_tdi.add_argument(
        "--norm-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Range [min, max] for background normalization if norm_c is True.",
    )
    p_tdi.add_argument(
        "--cbar",
        type=str,
        default="off",
        choices=["True", "False", "off"],
        help="Show colorbar ('True', 'False', 'off') (default: 'off').",
    )
    p_tdi.add_argument(
        "--fig-size",
        type=str,
        default="10,8",
        help="Figure size 'width,height' (e.g., '10,8').",
    )
    p_tdi.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_tdi.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_tdi.set_defaults(func=_cli_plot_taylor_diagram_in)

    # --- Subparser: plot_taylor_diagram ---
    p_tdb = subparsers.add_parser(
        "plot_taylor_diagram",
        help="Basic Taylor Diagram from arrays.",
        description="Generates a standard Taylor Diagram comparing predictions "
        "to a reference. Requires raw data arrays.",
    )
    p_tdb.add_argument(
        "reference_file",
        type=str,
        metavar="REFERENCE_CSV",
        help="Path to CSV file with reference values.",
    )
    p_tdb.add_argument(
        "y_preds_files",
        type=str,
        nargs="+",
        metavar="PREDICTION_CSV",
        help="Paths to CSV files with prediction values.",
    )
    p_tdb.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for the predictions.",
    )
    p_tdb.add_argument(
        "--acov",
        type=str,
        default="half_circle",
        choices=["default", "half_circle"],
        help="Angular coverage (default: 'half_circle').",
    )
    p_tdb.add_argument(
        "--zero-location",
        type=str,
        default="W",
        choices=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        help="Position of correlation=1 (default: 'W').",
    )
    p_tdb.add_argument(
        "--direction",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="Angle direction (default: -1).",
    )
    p_tdb.add_argument(
        "--only-points",
        action="store_true",
        default=False,
        help="Plot only markers, no radial lines.",
    )
    p_tdb.add_argument(
        "--ref-color",
        type=str,
        default="red",
        help="Color for reference arc/line (default: 'red').",
    )
    p_tdb.add_argument(
        "--marker", type=str, default="o", help="Marker style (default: 'o')."
    )
    p_tdb.add_argument(
        "--corr-steps",
        type=int,
        default=6,
        help="Number of correlation ticks (default: 6).",
    )
    p_tdb.add_argument(
        "--fig-size",
        type=str,
        default="10,8",
        help="Figure size 'width,height' (e.g., '10,8').",
    )
    p_tdb.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_tdb.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    # Add draw_ref_arc, angle_to_corr if they become explicit args
    p_tdb.set_defaults(func=_cli_plot_taylor_diagram)

    # --- Subparser: plot_feature_fingerprint ---
    p_ff = subparsers.add_parser(
        "plot_feature_fingerprint",
        help="Visualize feature importance profiles (radar chart).",
        description="Generates a radar chart comparing feature importance "
        "across different layers (e.g., models, years).",
    )
    p_ff.add_argument(
        "importances_file",
        type=str,
        metavar="IMPORTANCES_CSV",
        help="Path to CSV file containing the importance matrix "
        "(rows=layers, columns=features).",
    )
    p_ff.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="List of feature names (corresponding to columns).",
    )
    p_ff.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="List of layer names (corresponding to rows).",
    )
    p_ff.add_argument(
        "--normalize",
        action="store_true",
        dest="normalize",
        default=True,
        help="Normalize importances within each layer to [0, 1] (default).",
    )
    p_ff.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Plot raw importance values.",
    )
    p_ff.add_argument(
        "--fill",
        action="store_true",
        dest="fill",
        default=True,
        help="Fill the area under the radar lines (default).",
    )
    p_ff.add_argument(
        "--no-fill",
        action="store_false",
        dest="fill",
        help="Do not fill the area under the radar lines.",
    )
    p_ff.add_argument(
        "--cmap",
        type=str,
        default="tab10",
        help="Colormap for coloring different layers (default: 'tab10').",
    )
    p_ff.add_argument(
        "--title",
        type=str,
        default="Feature Impact Fingerprint",
        help="Plot title (default: 'Feature Impact Fingerprint').",
    )
    p_ff.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_ff.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_ff.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_ff.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_ff.set_defaults(func=_cli_plot_feature_fingerprint)

    # --- Subparser: plot_relationship ---
    p_rel = subparsers.add_parser(
        "plot_relationship",
        help="Visualize relationship between true and predicted values.",
        description="Generates a polar scatter plot mapping true values "
        "to angle and normalized predictions to radius.",
    )
    p_rel.add_argument(
        "y_true_file",
        type=str,
        metavar="TRUE_CSV",
        help="Path to CSV file with true values.",
    )
    p_rel.add_argument(
        "y_preds_files",
        type=str,
        nargs="+",
        metavar="PRED_CSV",
        help="Paths to CSV files with prediction values (one per model).",
    )
    p_rel.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for the prediction series.",
    )
    p_rel.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p_rel.add_argument(
        "--theta-offset",
        type=float,
        default=0,
        help="Angular offset in radians (default: 0).",
    )
    p_rel.add_argument(
        "--theta-scale",
        type=str,
        default="proportional",
        choices=["proportional", "uniform"],
        help="How y_true maps to angle (default: 'proportional').",
    )
    p_rel.add_argument(
        "--acov",
        type=str,
        default="default",
        choices=["default", "half_circle", "quarter_circle", "eighth_circle"],
        help="Angular coverage (default: 'default').",
    )
    p_rel.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="Figure size 'width,height' (e.g., '8,8').",
    )
    p_rel.add_argument(
        "--cmap",
        type=str,
        default="tab10",
        help="Colormap used to generate default colors (default: 'tab10').",
    )
    p_rel.add_argument(
        "--s",
        type=float,
        default=50,  # Allow float size
        help="Marker size (default: 50).",
    )
    p_rel.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency for points (default: 0.7).",
    )
    p_rel.add_argument(
        "--legend",
        action="store_true",
        dest="legend",
        default=True,
        help="Show legend (default).",
    )
    p_rel.add_argument(
        "--no-legend", action="store_false", dest="legend", help="Hide legend."
    )
    p_rel.add_argument(
        "--show-grid",
        action="store_true",
        dest="show_grid",
        default=True,
        help="Show grid lines (default).",
    )
    p_rel.add_argument(
        "--no-show-grid",
        action="store_false",
        dest="show_grid",
        help="Hide grid lines.",
    )
    p_rel.add_argument(
        "--xlabel", type=str, default=None, help="Label for radial axis."
    )
    p_rel.add_argument(
        "--ylabel", type=str, default=None, help="Label for angular axis."
    )
    p_rel.add_argument(
        "--z-values-file",
        type=str,
        default=None,
        metavar="Z_CSV",
        help="Optional path to CSV file with values for angular labels.",
    )
    p_rel.add_argument(
        "--z-label", type=str, default=None, help="Label for z-values if provided."
    )
    p_rel.add_argument(
        "--savefig",
        type=str,
        metavar="FILEPATH",
        help="Save plot to file instead of displaying.",
    )
    p_rel.set_defaults(func=_cli_plot_relationship)

    # --- Parse Arguments and Execute ---
    # (Keep existing logic)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


# --- Entry Point Guard ---
# This allows the script to be run directly (python kdiagram/cli.py ...)
# for testing, but is not strictly necessary when installed via setup.py
if __name__ == "__main__":
    # Need to access __version__ - assuming it's in the package's __init__
    try:
        from kdiagram import __version__ as pkg_version
    except ImportError:
        # Fallback if run directly before installation or __version__ isn't exposed
        pkg_version = "unknown"

    # A bit redundant if setup.py handles it, but useful for direct script execution
    class KDiagramNamespace:
        __version__ = pkg_version

    kdiagram = KDiagramNamespace()
    main()
