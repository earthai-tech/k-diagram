# -*- coding: utf-8 -*-
# test_evaluation_plots.py
# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Pytest suite for testing model evaluation functions (Taylor Diagrams) in
kdiagram.plot.evaluation.
"""

import pytest
import numpy as np
import pandas as pd # May be needed if functions internally use it
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# --- Import functions to test ---
# Adjust the import path based on your project structure
from kdiagram.plot.evaluation import (
    taylor_diagram,
    plot_taylor_diagram_in,
    plot_taylor_diagram
)

# --- Pytest Configuration ---
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

# --- Fixtures ---

@pytest.fixture(autouse=True)
def close_plots():
    """Fixture to close all matplotlib plots after each test."""
    yield
    plt.close('all')

@pytest.fixture(scope="module") # Reuse data across tests in this module
def sample_data_taylor():
    """Provides sample data for Taylor Diagram tests."""
    np.random.seed(101)
    n_points = 150
    reference = np.random.normal(0, 1.0, n_points) # Reference std dev = 1.0

    # Model A: High correlation, slightly lower std dev
    pred_a = reference * 0.8 + np.random.normal(0, 0.4, n_points)
    # Model B: Lower correlation, higher std dev
    pred_b = reference * 0.5 + np.random.normal(0, 1.1, n_points)
    # Model C: Good correlation, similar std dev
    pred_c = reference * 0.95 + np.random.normal(0, 0.3, n_points)

    y_preds = [pred_a, pred_b, pred_c]
    names = ["Model A", "Model B", "Model C"]

    # Precompute stats
    stds = [np.std(p) for p in y_preds]
    corrs = [np.corrcoef(p, reference)[0, 1] for p in y_preds]
    ref_std_val = np.std(reference)

    return {
        "reference": reference,
        "y_preds": y_preds,
        "names": names,
        "stds": stds,
        "corrs": corrs,
        "ref_std": ref_std_val
    }

# --- Test Functions ---

# == Tests for taylor_diagram ==
def test_taylor_diagram_runs_with_arrays(sample_data_taylor):
    """Test taylor_diagram runs okay with raw data arrays."""
    data = sample_data_taylor
    try:
        # Note: function doesn't explicitly return ax in provided code
        taylor_diagram(
            y_preds=data['y_preds'],
            reference=data['reference'],
            names=data['names'],
            title="Test with Arrays"
        )
        assert len(plt.get_fignums()) > 0, "Figure should be created"
    except Exception as e:
        pytest.fail(f"taylor_diagram (with arrays) raised exception: {e}")

def test_taylor_diagram_runs_with_stats(sample_data_taylor):
    """Test taylor_diagram runs okay with precomputed stats."""
    data = sample_data_taylor
    try:
        taylor_diagram(
            stddev=data['stds'],
            corrcoef=data['corrs'],
            ref_std=data['ref_std'],
            names=data['names'],
            title="Test with Stats"
        )
        assert len(plt.get_fignums()) > 0, "Figure should be created"
    except Exception as e:
        pytest.fail(f"taylor_diagram (with stats) raised exception: {e}")

@pytest.mark.parametrize("cmap, strategy", [
    ('viridis', 'rwf'),
    ('plasma', 'performance'),
    ('magma', 'convergence'),
    ('cividis', 'center_focus'),
    (None, None) # Test without background
])
def test_taylor_diagram_backgrounds(sample_data_taylor, cmap, strategy):
    """Test different background cmap and strategy options."""
    data = sample_data_taylor
    try:
        taylor_diagram(
            stddev=data['stds'], corrcoef=data['corrs'], ref_std=data['ref_std'],
            names=data['names'], cmap=cmap, radial_strategy=strategy,
            norm_c=(cmap is not None), # Normalize if cmap is used
            title=f"Test BG: {cmap}/{strategy}"
        )
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"taylor_diagram background test failed: {e}")

def test_taylor_diagram_ref_arc(sample_data_taylor):
    """Test draw_ref_arc=True option."""
    data = sample_data_taylor
    try:
        taylor_diagram(
            stddev=data['stds'], corrcoef=data['corrs'], ref_std=data['ref_std'],
            names=data['names'], draw_ref_arc=True
        )
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"taylor_diagram draw_ref_arc failed: {e}")

def test_taylor_diagram_error_no_input(sample_data_taylor):
    """Test ValueError if neither stats nor arrays are given."""
    with pytest.raises(ValueError, match="Provide either stddev.*or y_preds"):
        taylor_diagram()
    with pytest.raises(ValueError, match="Provide either stddev.*or y_preds"):
        taylor_diagram(stddev=sample_data_taylor['stds']) # Missing corrcoef
    with pytest.raises(ValueError, match="Provide either stddev.*or y_preds"):
        taylor_diagram(y_preds=sample_data_taylor['y_preds']) # Missing reference

def test_taylor_diagram_error_inconsistent_stats(sample_data_taylor):
    """Test error if stddev and corrcoef have different lengths."""
    data = sample_data_taylor
    with pytest.raises(ValueError): # check_consistent_length raises ValueError
        taylor_diagram(
            stddev=data['stds'][:-1], # Shorter list
            corrcoef=data['corrs'],
            ref_std=data['ref_std']
        )

# == Tests for plot_taylor_diagram_in ==

@pytest.mark.parametrize("acov", ['default', 'half_circle', None]) # None defaults to half
@pytest.mark.parametrize("zero_loc", ['N', 'E', 'W'])
@pytest.mark.parametrize("direction", [-1, 1])
def test_plot_taylor_diagram_in_orientations(
    sample_data_taylor, acov, zero_loc, direction):
    """Test plot_taylor_diagram_in with orientation params."""
    data = sample_data_taylor
    try:
        # Note: function doesn't explicitly return ax
        plot_taylor_diagram_in(
            *data['y_preds'], reference=data['reference'], names=data['names'],
            acov=acov, zero_location=zero_loc, direction=direction,
            title=f"Test IN: {acov}/{zero_loc}/{direction}"
        )
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_taylor_diagram_in orientation test failed: {e}")

@pytest.mark.parametrize("strategy", [
    'convergence', 'norm_r', 'performance', None]) # None defaults to performance
@pytest.mark.parametrize("norm_c", [True, False])
@pytest.mark.parametrize("cbar_opt", [True, False, 'off'])
def test_plot_taylor_diagram_in_background(
    sample_data_taylor, strategy, norm_c, cbar_opt):
    """Test plot_taylor_diagram_in background/cbar options."""
    data = sample_data_taylor
    # Expect warning for unsupported strategies
    if strategy in ['rwf', 'center_focus']:
        with pytest.warns(UserWarning, match="'rwf'|'center_focus' is not available"):
             plot_taylor_diagram_in(
                *data['y_preds'], reference=data['reference'], names=data['names'],
                radial_strategy=strategy, norm_c=norm_c, cbar=cbar_opt,
                title=f"Test IN BG: {strategy}/{norm_c}/{cbar_opt}"
            )
        assert len(plt.get_fignums()) > 0
    else:
        try:
            plot_taylor_diagram_in(
                *data['y_preds'], reference=data['reference'], names=data['names'],
                radial_strategy=strategy, norm_c=norm_c, cbar=cbar_opt,
                title=f"Test IN BG: {strategy}/{norm_c}/{cbar_opt}"
            )
            assert len(plt.get_fignums()) > 0
        except Exception as e:
            pytest.fail(
                f"plot_taylor_diagram_in background test failed: {e}"
            )

def test_plot_taylor_diagram_in_error_mismatched_lengths(sample_data_taylor):
    """Test error for mismatched prediction/reference lengths."""
    data = sample_data_taylor
    ref_short = data['reference'][:-10]
    with pytest.raises(ValueError, match="must be the same length"):
        plot_taylor_diagram_in(
            *data['y_preds'], reference=ref_short
        )

def test_plot_taylor_diagram_in_invalid_enums(sample_data_taylor):
    """Test error for invalid enum-like string options."""
    data = sample_data_taylor
    with pytest.raises(ValueError): # validate_params raises ValueError
         plot_taylor_diagram_in(
            *data['y_preds'], reference=data['reference'],
            zero_location='invalid_loc'
        )
    with pytest.raises(ValueError): # validate_params raises ValueError
         plot_taylor_diagram_in(
            *data['y_preds'], reference=data['reference'],
            acov='invalid_acov'
        )
    # Direction check is internal, test boundary
    with pytest.warns(UserWarning, match="direction must be -1 or 1"):
        plot_taylor_diagram_in(
           *data['y_preds'], reference=data['reference'], direction=0
        )

# == Tests for plot_taylor_diagram ==

@pytest.mark.parametrize("acov", ['default', 'half_circle'])
@pytest.mark.parametrize("zero_loc", ['N', 'S', 'W'])
def test_plot_taylor_diagram_basic_runs_ok(
    sample_data_taylor, acov, zero_loc):
    """Test basic plot_taylor_diagram runs okay."""
    data = sample_data_taylor
    try:
        # Note: signature uses ... for draw_ref_arc, angle_to_corr - testing defaults
        # Also doesn't explicitly return ax
        plot_taylor_diagram(
            *data['y_preds'], reference=data['reference'], names=data['names'],
            acov=acov, zero_location=zero_loc,
            title=f"Test Basic TD: {acov}/{zero_loc}"
        )
        assert len(plt.get_fignums()) > 0
    except Exception as e:
        pytest.fail(f"plot_taylor_diagram basic test failed: {e}")

def test_plot_taylor_diagram_error_mismatched_lengths(sample_data_taylor):
    """Test error for mismatched prediction/reference lengths."""
    data = sample_data_taylor
    ref_short = data['reference'][:-10]
    # Assertion is internal in the function
    with pytest.raises(AssertionError, match="must be of the same length"):
        plot_taylor_diagram(
            *data['y_preds'], reference=ref_short
        )

def test_plot_taylor_diagram_invalid_enums(sample_data_taylor):
    """Test error for invalid enum-like string options."""
    data = sample_data_taylor
    # Assuming validate_params decorator applies here too
    with pytest.raises(ValueError): # validate_params raises ValueError
         plot_taylor_diagram(
            *data['y_preds'], reference=data['reference'],
            zero_location='invalid_loc'
        )
    with pytest.raises(ValueError): # validate_params raises ValueError
         plot_taylor_diagram(
            *data['y_preds'], reference=data['reference'],
            acov='invalid_acov'
        )
    # Direction check is internal
    with pytest.warns(UserWarning, match="direction should be either 1"):
        plot_taylor_diagram(
            *data['y_preds'], reference=data['reference'], direction=5
        )

if __name__=='__main__': 
    pytest.main([__file__]) 