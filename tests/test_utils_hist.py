import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kdiagram.utils.hist import plot_hist_kde

# Ensure tests run without displaying plots
plt.switch_backend("Agg")


def test_plot_hist_kde_with_numpy_array():
    # Test with numpy array data
    data = np.random.normal(0, 1, 1000)
    grid, pdf = plot_hist_kde(data, bins=30, show_kde=True)

    # Check the types of the outputs
    assert isinstance(grid, np.ndarray), "Grid should be of type np.ndarray"
    assert isinstance(pdf, np.ndarray), "PDF should be of type np.ndarray"

    # Check the shape of the returned arrays (should match the number of grid points)
    assert grid.shape[0] == pdf.shape[0], (
        "Grid and PDF should have the same length"
    )


def test_plot_hist_kde_with_pandas_series():
    # Test with pandas Series data
    data = pd.Series(np.random.normal(0, 1, 1000))
    grid, pdf = plot_hist_kde(data, bins=30, show_kde=True)

    assert isinstance(grid, np.ndarray), "Grid should be of type np.ndarray"
    assert isinstance(pdf, np.ndarray), "PDF should be of type np.ndarray"
    assert grid.shape[0] == pdf.shape[0], (
        "Grid and PDF should have the same length"
    )


def test_plot_hist_kde_with_pandas_dataframe():
    # Test with pandas DataFrame and specifying a column
    df = pd.DataFrame({"values": np.random.normal(0, 1, 1000)})
    grid, pdf = plot_hist_kde(df, column="values", bins=30, show_kde=True)

    assert isinstance(grid, np.ndarray), "Grid should be of type np.ndarray"
    assert isinstance(pdf, np.ndarray), "PDF should be of type np.ndarray"
    assert grid.shape[0] == pdf.shape[0], (
        "Grid and PDF should have the same length"
    )


def test_plot_hist_kde_savefig():
    # Test with savefig functionality
    data = np.random.normal(0, 1, 1000)
    save_path = "test_hist_kde_output.png"

    # Run the function and save the plot
    grid, pdf = plot_hist_kde(data, bins=30, show_kde=True, savefig=save_path)

    # Check if the file was saved correctly
    import os

    assert os.path.exists(save_path), f"File {save_path} was not saved"

    # Clean up by removing the saved file after the test
    os.remove(save_path)


def test_plot_hist_kde_with_normalize_kde():
    # Test normalization of KDE
    data = np.random.normal(0, 1, 1000)
    grid, pdf = plot_hist_kde(
        data, bins=30, show_kde=True, normalize_kde=True
    )

    # Check if the PDF is normalized between 0 and 1
    assert pdf.max() <= 1.0, "PDF values should be normalized to [0, 1]"
    assert pdf.min() >= 0.0, "PDF values should be non-negative"
