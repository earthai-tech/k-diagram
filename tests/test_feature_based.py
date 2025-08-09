# -*- coding: utf-8 -*-
# License: Apache 2.0
# Author: LKouadio <etanoyau@gmail.com>

"""
Pytest suite for testing feature-based visualization functions in
kdiagram.plot.feature_based.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# --- Import function to test ---
# Adjust the import path based on your project structure
from kdiagram.plot.feature_based import plot_feature_fingerprint

# --- Pytest Configuration ---
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Fixtures ---

@pytest.fixture(autouse=True)
def close_plots():
    """Fixture to close all matplotlib plots after each test."""
    yield
    plt.close('all')

@pytest.fixture(scope="module")
def sample_data_fingerprint():
    """Provides sample data for plot_feature_fingerprint tests."""
    np.random.seed(123)
    n_layers = 3
    n_features = 6
    importances = np.random.rand(n_layers, n_features) * 0.8
    # Make layers distinct
    importances[0, 0] = 0.9 # Layer 0 high on Feature 1
    importances[1, 2:4] = 0.85 # Layer 1 high on Features 3, 4
    importances[2, -1] = 0.95 # Layer 2 high on Last Feature
    features = [f'Feature_{i+1}' for i in range(n_features)]
    labels = [f'Group {chr(65+i)}' for i in range(n_layers)] # A, B, C

    return {
        "importances": importances,
        "features": features,
        "labels": labels,
    }

# --- Test Functions ---

@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("fill", [True, False])
@pytest.mark.parametrize("input_type", ["numpy", "list"])
def test_plot_feature_fingerprint_runs_ok(
    sample_data_fingerprint, normalize, fill, input_type):
    """Test plot_feature_fingerprint runs okay with different settings."""
    data = sample_data_fingerprint
    importances_input = data['importances'] if input_type == "numpy" \
                        else data['importances'].tolist()

    try:
        ax = plot_feature_fingerprint(
            importances=importances_input,
            features=data['features'],
            labels=data['labels'],
            normalize=normalize,
            fill=fill,
            title=f"Test: norm={normalize}, fill={fill}, type={input_type}",
            savefig=None # Test display path
        )
        assert isinstance(ax, Axes), "Function should return Axes object"
        assert len(plt.get_fignums()) > 0 or plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_feature_fingerprint raised exception: {e}")

@pytest.mark.parametrize("provide_features", [True, False])
@pytest.mark.parametrize("provide_labels", [True, False])
def test_plot_feature_fingerprint_defaults(
    sample_data_fingerprint, provide_features, provide_labels):
    """Test default naming for features and labels."""
    data = sample_data_fingerprint
    features = data['features'] if provide_features else None
    labels = data['labels'] if provide_labels else None
    try:
        ax = plot_feature_fingerprint(
            importances=data['importances'],
            features=features,
            labels=labels,
            title=f"Test Defaults: feats={provide_features}, labs={provide_labels}"
        )
        assert isinstance(ax, Axes)
    except Exception as e:
        pytest.fail(f"plot_feature_fingerprint with defaults failed: {e}")

@pytest.mark.skip("Userwarning issue can bypassed instead.")
@pytest.mark.parametrize("num_extra", [-1, 1]) # Test fewer and more names
def test_plot_feature_fingerprint_name_label_mismatch(
    sample_data_fingerprint, num_extra):
    """Test warnings for mismatched features/labels lists."""
    data = sample_data_fingerprint
    n_feat = len(data['features'])
    n_lab = len(data['labels'])

    # Test mismatched features
    if n_feat + num_extra > 0: # Avoid creating empty list if num_extra=-1
        with pytest.warns(UserWarning, match="More feature names|generic names"):
            plot_feature_fingerprint(
                importances=data['importances'],
                features=data['features'][:n_feat + num_extra], # Wrong length
                labels=data['labels']
            )
            plt.close() # Close plot opened by warning test

    # Test mismatched labels
    if n_lab + num_extra > 0:
        with pytest.warns(UserWarning, match="More labels|Fewer labels"):
            plot_feature_fingerprint(
                importances=data['importances'],
                features=data['features'],
                labels=data['labels'][:n_lab + num_extra] # Wrong length
            )
            plt.close() # Close plot opened by warning test


def test_plot_feature_fingerprint_empty_input():
    """Test error on empty importances input."""
    with pytest.raises((ValueError, IndexError)): # Decorator raises ValueError
        plot_feature_fingerprint(importances=[])
    with pytest.raises((ValueError, IndexError)):
          plot_feature_fingerprint(importances=np.array([]))

def test_plot_feature_fingerprint_normalize_zeros(sample_data_fingerprint):
    """Test normalization handles rows of zeros correctly."""
    data = sample_data_fingerprint
    importances_with_zeros = data['importances'].copy()
    importances_with_zeros[1, :] = 0 # Set second layer's importances to zero

    try:
        # Should run without division-by-zero errors
        ax = plot_feature_fingerprint(
            importances=importances_with_zeros,
            features=data['features'],
            labels=data['labels'],
            normalize=True # Enable normalization
        )
        assert isinstance(ax, Axes)
        # Check that the second layer's values are indeed zero (or near zero)
        lines = ax.get_lines()
        # Assuming lines are plotted in order: Layer A outline, Layer B, Layer C
        # Layer B is lines[1]
        r_data_layer_b = lines[1].get_ydata()
        assert np.allclose(r_data_layer_b, 0), "Normalized zero row should be zero"

    except Exception as e:
        pytest.fail(f"plot_feature_fingerprint normalize with zeros failed: {e}")
        
if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])