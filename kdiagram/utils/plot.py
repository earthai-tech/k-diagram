# -*- coding: utf-8 -*-

from __future__ import print_function
import re
from typing import Optional, List

def set_axis_grid(
    ax, 
    show_grid: bool=True, 
    grid_props: dict = None
) -> None:
    """
    Robustly set grid properties on one or more matplotlib axes.

    Parameters:
        ax: A matplotlib Axes object or a list/tuple of Axes.
        show_grid: If True, enable the grid with the specified properties.
                   If False, disable the grid.
        grid_props: A dictionary of grid properties to pass to ax.grid.
                    The key 'visible' will be removed to avoid conflicts.

    Returns:
        None
    """
    # Ensure grid_props is a dictionary.
    props = grid_props.copy() if grid_props is not None else {
        'linestyle': ':', 'alpha': 0.7}
    # Remove the 'visible' key if present to avoid dual assignment.
    props.pop("visible", None)
    
    # If multiple axes are provided, iterate over each.
    if isinstance(ax, (list, tuple)):
        for a in ax:
            a.grid(show_grid, **props)
    else:
        a = ax
        a.grid(show_grid, **props)


def is_valid_kind(
    kind: str,
    valid_kinds: Optional[List[str]] = None,
    error: str = 'raise',
) -> str:
    """
    Normalizes and validates plot type specifications,
    handling aliases and suffixes.

    Parameters:
        kind (str): Input plot type specification (flexible formatting).
        valid_kinds (Optional[List[str]]): 
            Acceptable plot types to validate against.
        error (str): Error handling mode 
        ('raise' to raise errors, others to return normalized kind).

    Returns:
        str: Normalized canonical plot type or custom kind.

    Raises:
        ValueError: If invalid plot type is provided and `error` is 'raise`.
    """
    SUFFIXES = ('plot', 'graph', 'chart', 'diagram', 'visual')

    # Expanded alias mappings
    KIND_ALIASES = {
        'boxplot': 'box',
        'boxgraph': 'box',
        'boxchart': 'box',
        'plotbox': 'box',
        'box_plot': 'box',
        'violinplot': 'violin',
        'violingraph': 'violin',
        'violinchart': 'violin',
        'violin_plot': 'violin',
        'scatterplot': 'scatter',
        'scattergraph': 'scatter',
        'scatterchart': 'scatter',
        'lineplot': 'line',
        'linegraph': 'line',
        'linechart': 'line',
        'barchart': 'bar',
        'bargraph': 'bar',
        'barplot': 'bar',
        'plotbar': 'bar',
        'histogram': 'hist',
        'histplot': 'hist',
        'heatmap': 'heatmap',
        'heat_map': 'heatmap',
        'plotdensity': 'density',
        'plot_density': 'density',
        'densityplot': 'density',
        'densitygraph': 'density',
        'areachart': 'area',
        'areagraph': 'area'
    }

    # Canonical regex patterns (match anywhere in string)
    CANONICAL_PATTERNS = {
        'box': re.compile(r'box'),
        'violin': re.compile(r'violin'),
        'scatter': re.compile(r'scatter'),
        'line': re.compile(r'line'),
        'bar': re.compile(r'bar'),
        'hist': re.compile(r'hist'),
        'heatmap': re.compile(r'heatmap'),
        'density': re.compile(r'density'),
        'area': re.compile(r'area')
    }

    def normalize(k: str) -> str:
        """Normalize input: clean, lowercase, remove suffixes."""
        # Remove non-alphanumeric chars and underscores
        k_clean = re.sub(r'[\W_]+', '', k.strip().lower())
        # Remove suffixes from the end
        for suffix in SUFFIXES:
            if k_clean.endswith(suffix):
                k_clean = k_clean[:-len(suffix)]
                break
        return k_clean

    normalized = normalize(kind)

    # 1. Check exact aliases
    canonical = KIND_ALIASES.get(normalized)

    # 2. Search for canonical patterns if no alias found
    if not canonical:
        for pattern, regex in CANONICAL_PATTERNS.items():
            if regex.search(normalized):
                canonical = pattern
                break

    final_kind = canonical if canonical else normalized

    # Validation against allowed kinds
    if valid_kinds is not None:
        # Normalize valid kinds using same rules
        valid_normalized = {normalize(k): k for k in valid_kinds}
        final_normalized = normalize(final_kind)

        # Check matches against original valid kinds or their normalized forms
        valid_match = False
        for valid_norm, orig_kind in valid_normalized.items():
            if (final_normalized == valid_norm or 
                final_normalized == normalize(orig_kind)):
                valid_match = True
                break

        if not valid_match and error == 'raise':
            allowed = ', '.join(f"'{k}'" for k in valid_kinds)
            raise ValueError(f"Invalid plot type '{kind}'. Allowed: {allowed}")

    return final_kind
