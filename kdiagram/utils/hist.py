import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

from .handlers import columns_manager
from .plot import (
    prepare_data_for_kde,
    setup_plot_axes,
    add_kde_to_plot,
    add_histogram_to_plot,
    normalize_pdf,
    set_axis_grid,
)

def plot_hist_kde(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    column: Optional[str] = None,
    *,
    bins: int = 50,
    x_label: Optional[str] = None, 
    title: str = "Distribution (Histogram + KDE)",
    bandwidth: Optional[float] = None,
    show_kde: bool = True,
    savefig: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (8, 6),
    kde_color: str = "orange",
    hist_color: str = "skyblue",
    hist_edge_color: str = "white",
    kde_line_width: float = 2,
    hist_alpha: float = 0.7,
    normalize_kde: bool = False,
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    **hist_kws, # Pass extra keywords to the histogram
) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure the data is a valid type and convert it to np.ndarray
    if isinstance(data, ( pd.DataFrame, pd.Series)):
        if isinstance(data, pd.DataFrame):
            columns = columns_manager(column, empty_as_none=True)
            if columns is None:
                raise ValueError(
                    "If a DataFrame is provided, the 'column' "
                    "parameter must specify the series to plot."
                )
            column = columns[0]
            series_data = data[column]
        else: 
            series_data = data

        # Auto-set x_label from series name if not provided by user
        if x_label is None and series_data.name:
            x_label = str(series_data.name)
            
        data = series_data.values
      
    # Default x_label if still not set
    if x_label is None:
        x_label = "Value"
        
    data = np.asarray(data)

    # Prepare the data for KDE
    grid, pdf = prepare_data_for_kde(data, bandwidth=bandwidth)

    if normalize_kde:
        pdf = normalize_pdf(pdf)

    # Create the plot axes
    ax = setup_plot_axes(figsize=figsize, title=title, x_label= x_label)

    # Set grid properties
    set_axis_grid(ax, show_grid=show_grid, grid_props=grid_props)

    add_histogram_to_plot(
        data,
        ax,
        bins=bins,
        hist_color=hist_color,
        hist_edge_color=hist_edge_color,
        hist_alpha=hist_alpha,
        **hist_kws
    )

    # Add KDE to the plot if requested
    if show_kde:
        add_kde_to_plot(
            grid,
            pdf,
            ax,
            color=kde_color,
            line_width=kde_line_width,
        )

    # Customize axis labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    
    # Let legend be added after all elements are plotted
    ax.legend()

    # Save or display the plot
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return grid, pdf


plot_hist_kde.__doc__=r"""
Plot histogram and Kernel Density Estimate (KDE) for uncertainty 
evaluation.

This function combines a histogram and a Kernel Density Estimate 
(KDE) to visualize the distribution of the provided data. It allows 
users to evaluate the uncertainty in predictions by plotting the 
histogram of the data along with an optional KDE to estimate the 
probability density function.

Parameters
----------
data : Union[np.ndarray, pd.Series, pd.DataFrame]
    The data to be plotted. This can be a numpy array, a pandas 
    Series, or a pandas DataFrame. If a DataFrame is provided, the 
    'column' parameter must be specified to select the column to plot.

column : Optional[str], default=None
    The name of the column to plot if the input data is a DataFrame. 
    If data is a Series, this parameter is ignored.

bins : int, default=50
    The number of bins to use in the histogram.

x_label : str, default='Value'
    The label for the x-axis.

title : str, default='Distribution (Histogram + KDE)'
    The title of the plot.

bandwidth : Optional[float], default=None
    The bandwidth for the Kernel Density Estimate. If None, the 
    bandwidth will be estimated using Silverman's rule of thumb.

show_kde : bool, default=True
    Whether or not to display the KDE on the plot. If False, only 
    the histogram will be plotted.

savefig : Optional[str], default=None
    The file path where the plot will be saved. If None, the plot 
    will be displayed on the screen.

dpi : int, default=300
    The resolution of the saved plot (dots per inch) when savefig 
    is specified.

figsize : Tuple[float, float], default=(8, 6)
    The size of the plot in inches.

kde_color : str, default='orange'
    The color of the KDE line.

hist_color : str, default='skyblue'
    The color of the histogram bars.

hist_edge_color : str, default='white'
    The color of the edges of the histogram bars.

kde_line_width : float, default=2
    The line width of the KDE line.

hist_alpha : float, default=0.7
    The transparency level of the histogram bars. A value between 
    0 and 1.

hist_edge_alpha : float, default=1.0
    The transparency level of the histogram edges. A value between 
    0 and 1.

normalize_kde : bool, default=False
    If True, the KDE will be normalized so that the maximum value 
    is 1.

show_grid : bool, default=True
    Whether or not to display a grid on the plot.

grid_props : Optional[dict], default=None
    A dictionary of grid properties. If provided, these will be 
    applied to customize the grid appearance. By default, a dotted 
    grid with 0.7 alpha is used.

**kws : additional keyword arguments
    Additional keyword arguments that can be passed to customize 
    the plot, such as adjusting the axis properties or applying 
    specific formatting.

Returns
-------
grid : np.ndarray
    The x-values grid for the KDE evaluation.

pdf : np.ndarray
    The estimated probability density function (PDF) values computed 
    from the KDE.

Notes
-----
- The function estimates the KDE using a Gaussian kernel with a 
  specified or automatically calculated bandwidth.
- The KDE can be normalized to fit the range [0, 1], which is 
  useful for comparison purposes, especially when overlaid with 
  histograms.
- The function automatically handles different input data types, 
  such as pandas DataFrames, Series, or numpy arrays.

Examples
--------
>>> import numpy as np
>>> from kdiagram.utils import plot_hist_kde
>>> data = np.random.normal(0, 1, 1000)
>>> plot_hist_kde(data, bins=30, kde_color='blue')

>>> import pandas as pd
>>> df = pd.DataFrame({'values': np.random.normal(0, 1, 1000)})
>>> plot_hist_kde(df, column='values', bins=30, show_kde=True)

>>> plot_hist_kde(data, bins=30, title="Histogram with KDE", 
>>>                savefig="output.png")

See Also
--------
- `scipy.stats.gaussian_kde`: For the Kernel Density Estimate 
  implementation.
- `matplotlib.pyplot.hist`: For plotting histograms in matplotlib.
- `pandas.Series.hist`: For creating histograms from pandas Series.

References
----------
.. [1] Silverman, B. W. (1986). *Density Estimation for Statistics 
       and Data Analysis*. CRC Press.
.. [2] Scott, D. W. (2015). *Multivariate Density Estimation: 
       Theory, Practice, and Visualization*. Wiley-Interscience.
"""

