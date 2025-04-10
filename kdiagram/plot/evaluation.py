# -*- coding: utf-8 -*-


from __future__ import annotations 
import warnings 
from numbers import Integral
from typing import ( 
    Optional, Tuple, List,
)
import numpy as np
import matplotlib.pyplot as plt 

from ..compat.sklearn import validate_params, StrOptions 
from ..utils.handlers import columns_manager 
from ..utils.mathext import minmax_scaler 
from ..utils.validator import ( 
    check_consistent_length, 
    contains_nested_objects, 
    validate_yy, 
    validate_length_range, 
    
)

from ._properties import TDG_DIRECTIONS

__all__= [    
    'plot_taylor_diagram',
    'plot_taylor_diagram_in',     
    'taylor_diagram', 
   ]

def taylor_diagram(
    stddev=None,
    corrcoef=None,
    y_preds=None,
    reference=None,
    names=None,
    ref_std=1,
    cmap=None,
    draw_ref_arc=False,
    radial_strategy="rwf",
    norm_c=False,
    power_scaling=1.0,
    marker='o',
    ref_props=None,
    fig_size=None,
    size_props=None,
    title=None,
    savefig=None,
):
    r"""
    Plot a Taylor diagram to compare multiple predictions against
    a reference by visualizing their correlation and standard
    deviation. This function can accept either precomputed
    statistics (i.e. `stddev` and `corrcoef`) or the actual arrays
    (`y_preds` and `reference`) from which these statistics will be
    derived.

    The radial axis represents the standard deviation (std. dev.),
    while the angular axis represents the correlation with the
    reference (with angle :math:`\theta = \arccos(\rho)`).

    Parameters
    ----------
    stddev : list of float or None, optional
        List of standard deviations for each prediction. If
        `None`, the standard deviations are computed internally
        from `y_preds`. The length of `stddev` should match the
        number of models if provided.

    corrcoef : list of float or None, optional
        List of correlation coefficients for each prediction
        against the reference. If `None`, these are computed
        internally from `y_preds`. Must match the length of
        `stddev` if provided.

    y_preds : list of array-like or None, optional
        One or more prediction arrays (e.g. model outputs).
        Each array must share the same length as `reference`.
        Required if `stddev` or `corrcoef` is not provided.

    reference : array-like or None, optional
        Reference (observed) array used for computing correlation
        and std. dev. of predictions if `stddev` or `corrcoef`
        is not given. Must share length with each prediction in
        `y_preds`.

    names : list of str or None, optional
        Labels for each prediction array. Must match the number
        of models in `y_preds` or in `stddev`/`corrcoef`. If
        `None`, default labels of the form "Model_i" are used.

    ref_std : float, optional
        Standard deviation of the reference if already known or
        desired to be set explicitly. If predictions are provided
        (`y_preds` and `reference`), this is computed as
        `np.std(reference)` by default.

    cmap : str or None, optional
        Matplotlib colormap for the background shading. If not
        `None`, a contour fill is created based on the chosen
        `radial_strategy`, visualizing different performance
        or weighting zones. For example, `'viridis'` or
        `'plasma'`.

    draw_ref_arc : bool, optional
        If `True`, an arc is drawn at the reference's standard
        deviation, highlighting that radial distance. If `False`,
        a point is placed at angle `0` with radial distance
        `ref_std`. Default is `False`.

    radial_strategy : {'rwf', 'convergence', 'center_focus',
                       'performance'}, optional
        Strategy for computing the background mesh (when
        `cmap` is not `None`):
        * ``'rwf'``: Radial weighting function that uses
          correlation and deviation distance in an exponential
          form.
        * ``'convergence'``: A simple radial function of `r`.
        * ``'center_focus'``: Focus on a center region in the
          (theta, r) space using an exponential decay from the
          center.
        * ``'performance'``: Highlight the region near the best
          performing model (max correlation, optimal std. dev.).

    norm_c : bool, optional
        If `True`, the generated background mesh is normalized to
        the range [0, 1] before plotting. This can highlight
        relative differences more clearly. Default is `False`.

    power_scaling : float, optional
        When `norm_c` is `True`, the normalized background mesh
        can be exponentiated by this factor. Useful for adjusting
        contrast. Default is `1.0`.

    marker : str, optional
        Marker style for the points representing each prediction.
        Defaults to `'o'`.

    ref_props : dict or None, optional
        Dictionary of reference plot properties, such as line
        style, color, or width. Supported keys include:
        * ``'label'``: Legend label for the reference.
        * ``'lc'``: Line color/style for the reference arc.
        * ``'color'``: Color/style for the reference point.
        * ``'lw'``: Line width.
        If not given, defaults to a green line and black point.

    fig_size : (float, float) or None, optional
        Figure size in inches, e.g. ``(width, height)``.
        Defaults to ``(8, 6)``.

    size_props : dict or None, optional
        Optional dictionary to control tick and label sizes.
        For instance: 
        ``{'ticks': 12, 'labels': 14}``.
        Can be used to adjust the font sizes of the radial and
        angular ticks and labels.

    title : str or None, optional
        Title of the figure. If `None`, defaults to
        ``"Taylor Diagram"``.

    savefig : str or None, optional
        Path to save the figure (e.g. ``"diagram.png"``). If
        `None`, the figure is displayed instead of being saved.

    Notes
    -----

    The Taylor diagram simultaneously shows two statistics for
    each model prediction :math:`p` compared to a reference
    :math:`r`:

    1. **Standard Deviation**:
       .. math::
          \sigma_p = \sqrt{\frac{1}{n}
          \sum_{i=1}^{n}\bigl(p_i - \bar{p}\bigr)^2}

       where :math:`\bar{p}` is the mean of :math:`p`.

    2. **Correlation**: :math:`\rho`
       .. math::
          \rho = \frac{\mathrm{Cov}(p, r)}
          {\sigma_p \; \sigma_r}

       where :math:`\mathrm{Cov}(p, r)` is the covariance between
       :math:`p` and :math:`r`, and :math:`\sigma_r` is the
       standard deviation of :math:`r`.

    The diagram uses polar coordinates with radius corresponding
    to the standard deviation, and the angle
    :math:`\theta = \arccos(\rho)` representing correlation.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.ml_viz import taylor_diagram
    >>> # Generate synthetic data
    >>> ref = np.random.randn(100)
    >>> preds = [
    ...     ref + 0.1 * np.random.randn(100),
    ...     1.2 * ref + 0.5 * np.random.randn(100),
    ... ]
    >>> # Basic usage (auto-compute stddev and corrcoef)
    >>> taylor_diagram(y_preds=preds, reference=ref)

    See Also
    --------
    numpy.std : Compute standard deviation.
    numpy.corrcoef : Compute correlation coefficients.

    References
    ----------
    .. [1] Taylor, K. E. (2001). Summarizing multiple aspects of
           model performance in a single diagram. *Journal of
           Geophysical Research*, 106(D7), 7183-7192.
    """

    # Create polar subplot
    fig, ax = plt.subplots(
        subplot_kw={'projection': 'polar'},
        figsize=fig_size or (8, 6)
    )

    # Handle reference properties
    ref_props = ref_props or {}
    ref_label = ref_props.pop('label', 'Reference')
    ref_color = ref_props.pop('lc', 'red')
    ref_point = ref_props.pop('color', 'k*')
    ref_lw = ref_props.pop('lw', 2)

    # Compute stddev and corrcoef from predictions if needed
    if (stddev is None or corrcoef is None):
        if (y_preds is None or reference is None):
            raise ValueError(
                "Provide either stddev and corrcoef, "
                "or y_preds and reference."
            )
        if not contains_nested_objects(y_preds, strict= True): 
            y_preds =[y_preds]
            
        y_preds = [
            validate_yy(reference, pred, flatten="auto")[1] 
            for pred in y_preds
        ]
        
        stddev = [np.std(pred) for pred in y_preds]
        corrcoef = [
            np.corrcoef(pred, reference)[0, 1] 
            for pred in y_preds
        ]
        ref_std = np.std(reference)
    
    # Re-check consistency
    check_consistent_length(stddev, corrcoef)
    
    # Ensure `names` matches number of models
    if names is not None:
        names= columns_manager(names)
        if len(names) < len(stddev):
            additional = [
                f"Model_{i + 1}" 
                for i in range(len(stddev) - len(names))
            ]
            names = names + additional
    else:
        names = [
            f"Model_{i + 1}" 
            for i in range(len(stddev))
        ]

    # Generate background if cmap is provided
    if cmap:
        theta_bg, r_bg = np.meshgrid(
            np.linspace(0, np.pi / 2, 500),
            np.linspace(0, max(stddev) + 0.5, 500)
        )

        # Compute background based on strategy
        if radial_strategy == "convergence":
            background = r_bg
        elif radial_strategy == "rwf":
            corr_bg = np.cos(theta_bg)
            std_diff = (r_bg - ref_std) ** 2
            background = (
                np.exp(-std_diff / 0.1) * 
                corr_bg ** 2
            )
        elif radial_strategy == "center_focus":
            center_std = (max(stddev) + ref_std) / 2
            std_diff = (r_bg - center_std) ** 2
            theta_diff = (theta_bg - np.pi / 4) ** 2
            background = (
                np.exp(-std_diff / 0.1) *
                np.exp(-theta_diff / 0.2)
            )
        elif radial_strategy == "performance":
            best_idx = np.argmax(corrcoef)
            std_best = stddev[best_idx]
            corr_best = corrcoef[best_idx]
            theta_best = np.arccos(corr_best)
            std_diff = (r_bg - std_best) ** 2
            theta_diff = (theta_bg - theta_best) ** 2
            background = (
                np.exp(-std_diff / 0.05) *
                np.exp(-theta_diff / 0.05)
            )

        # Normalize background if requested
        if norm_c:
            background = minmax_scaler(background )
            # background = (
            #     (background - np.min(background)) /
            #     (np.max(background) - np.min(background))
            # )
            background = background ** power_scaling

        # Plot the colored contour
        ax.contourf(
            theta_bg, 
            r_bg, 
            background,
            levels=100,
            cmap=cmap,
            alpha=0.8
        )

    # Draw reference point or arc
    if draw_ref_arc:
        t_arc = np.linspace(0, np.pi / 2, 500)
        ax.plot(
            t_arc,
            [ref_std] * len(t_arc),
            ref_color,
            linewidth=ref_lw,
            label=ref_label
        )
    else:
        ax.plot(
            0,
            ref_std,
            ref_point,
            markersize=12,
            label=ref_label
        )

    # Plot data points
    for i, (std_val, corr_val) in enumerate(
        zip(stddev, corrcoef)
    ):
        theta_pt = np.arccos(corr_val)
        ax.plot(
            theta_pt,
            std_val,
            marker,
            label=names[i],
            markersize=10
        )

    # Add correlation lines (dotted radial lines)
    t_corr = np.linspace(0, np.pi / 2, 100)
    for r_line in np.linspace(0, 1, 11):
        ax.plot(
            t_corr,
            [r_line * ref_std] * len(t_corr),
            'k--',
            alpha=0.3
        )

    # Add standard deviation circles
    for r_circ in np.linspace(0, max(stddev) + 0.5, 5):
        ax.plot(
            np.linspace(0, np.pi / 2, 100),
            [r_circ] * 100,
            'k--',
            alpha=0.3
        )

    # Set axis limits
    ax.set_xlim(0, np.pi / 2)
    ax.set_ylim(0, max(stddev) + 0.5)

    # Set x-ticks for correlation
    ax.set_xticks(
        np.arccos(np.linspace(0, 1, 6))
    )
    ax.set_xticklabels(
        ['1.0', '0.8', '0.6', '0.4', '0.2', '0.0']
    )

    # Axis labels
    ax.set_xlabel("Standard Deviation", labelpad=20)

    # Correlation text label on the plot
    ax.text(
        0.85,
        0.7,
        "Correlation",
        ha='center',
        rotation_mode="anchor",
        rotation=-45,
        transform=ax.transAxes
    )

    # Set size of ticks and labels if provided
    if size_props:
        tick_size = size_props.get('ticks', 10)
        label_size = size_props.get('label', 12)
        ax.tick_params(
            axis='both',
            labelsize=tick_size
        )
        # X-label
        for label in ax.xaxis.get_label():
            label.set_size(label_size)
        # We might want to set radial labels if any,
        # but let's keep it minimal. 
        
    # Legend and title
    ax.legend(loc='upper right')
    plt.title(title or 'Taylor Diagram')

    # Save or show figure
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()


@validate_params ({
    'reference': ['array-like'], 
    'names': [str, 'array-like', None ], 
    'acov': [StrOptions({'default', 'half_circle'}), None], 
    'zero_location': [StrOptions({'N','NE','E','S','SW','W','NW', 'SE'})], 
    'direction': [Integral]
    })
def plot_taylor_diagram_in(
    *y_preds,
    reference,
    names=None,
    acov=None,
    zero_location='E',
    direction=-1,
    only_points=False,
    ref_color='red',
    draw_ref_arc=True,
    angle_to_corr=True,
    marker='o',
    corr_steps=6,
    cmap="viridis",
    shading='auto',
    shading_res=300,
    radial_strategy=None, 
    norm_c=False, 
    norm_range=None, 
    cbar="off",
    fig_size=None,
    title=None,
    savefig=None,
):
    r"""
    Plot a Taylor Diagram with a background color map encoding the
    correlation domain in polar form. This function provides a visually
    appealing layout where the radial axis represents the standard
    deviation of each prediction, while the angular axis is derived from
    the correlation with the reference.

    Parameters
    ----------
    *y_preds : array-like
        One or more prediction arrays. Each array must be of the same
        length as `reference`. Each array-like object typically has
        shape :math:`(n,)`, although multi-dimensional inputs can be
        flattened internally.

    reference : array-like
        The reference (observed) array of shape :math:`(n,)`. It must
        share the same length as each array in `*y_preds`.

    names : list of str or None, optional
        Labels for each of the arrays in `*y_preds`. If provided, must
        match the number of prediction arrays. If `None`, each prediction
        is labeled as "Pred i" automatically.

    acov : {'default', 'half_circle'}, optional
        Angular coverage of the diagram:
        - ``'default'``: The diagram covers an angle of
          :math:`\\pi` (180 degrees).
        - ``'half_circle'``: The diagram covers an angle of
          :math:`\\pi/2` (90 degrees).
        If `acov` is `None`, it defaults to ``'half_circle'`` in the
        current implementation.

    zero_location : {'N','NE','E','S','SW','W','NW','SE'}, optional
        The position on the polar axis that corresponds to a correlation
        of :math:`1.0`. For example, ``'W'`` (west) places the
        correlation :math:`\\rho=1` to the left on the polar plot,
        whereas ``'N'`` (north) places it at the top.
        Default is ``'E'``.

    direction : int, optional
        Rotation direction for increasing angles. A value of
        ``1`` sets a counter-clockwise rotation; ``-1`` sets a clockwise
        rotation. Default is ``-1``.

    only_points : bool, optional
        If `True`, only the point markers for each prediction are
        plotted, omitting the radial lines that connect each marker to
        the origin. If `False`, radial lines are drawn. Default is
        `False`.

    ref_color : str, optional
        Color used to represent the reference standard deviation either
        as an arc (if `draw_ref_arc` is `True`) or as a radial line (if
        `draw_ref_arc` is `False`). Any valid matplotlib color is
        accepted. Default is ``'red'``.

    draw_ref_arc : bool, optional
        If `True`, an arc is drawn at the reference standard deviation
        to highlight its radial position. If `False`, a radial line is
        drawn from the origin to the reference standard deviation.
        Default is `True`.

    angle_to_corr : bool, optional
        If `True`, the angular axis (theta) is labeled in terms of
        correlation values from 0 to 1 (mapping angle
        :math:`\\theta = \\arccos(\\rho)`), so that perfect correlation
        :math:`\\rho = 1.0` maps to :math:`\\theta = 0`. If `False`,
        the angular axis is displayed in degrees. Default is `True`.

    marker : str, optional
        Marker style used for plotting each prediction point.
        For example, ``'o'`` for a circle or ``'^'`` for a triangle.
        See matplotlib marker documentation for available options.
        Default is ``'o'``.

    corr_steps : int, optional
        Number of correlation intervals to be labeled on the angular
        axis when `angle_to_corr` is `True`. A value of 6 creates
        correlation tick labels from 0.00 to 1.00 in steps of 0.20.
        Default is 6.

    cmap : str, optional
        Colormap name used for the background mesh showing the
        correlation domain. Any valid matplotlib colormap can be used,
        such as ``'viridis'`` or ``'turbo'``. Default is ``'viridis'``.

    shading : {'auto', 'gouraud', 'nearest'}, optional
        The shading method passed to matplotlib's ``pcolormesh`` for
        rendering the background. Default is ``'auto'``.

    shading_res : int, optional
        Resolution factor for generating the background mesh grid in
        both radial and angular dimensions. Larger values produce a
        smoother background. Default is 300.

    radial_strategy:  str, optional
        Defines how the radial background is generated.
        - `'convergence'`: Correlation is mapped using :math:`cos(theta)`,
          where :math:`theta` represents the angular displacement. This
          results in a color gradient converging from high correlation (1)
          to low correlation (0 or -1, depending on the plot coverage).
        - `'norm_r'`: Standardizes the radial distance by normalizing `r`
          to the range `[0, 1]`, where `r` is scaled by the maximum 
          radius (`rad_limit`).
        - `'performance'`: Colors are mapped based on distance from 
          the best-performing model, 
          using an exponential decay function to highlight the 
          best-performing region.
        - `'rwf'` and `'center_focus'`: These are unsupported in this 
          function. Consider using :func:`gofast.plot.plot_taylor_diagram`
          instead.

    norm_c :bool, optional) 
        If ``True``, normalizes the color values for a better
        visual contrast. Ensures that the color distribution is 
        balanced across the plot by scaling values between a
        predefined range. Defaults to ``False``.

    norm_range: tuple, optional
        Specifies the normalization range for color scaling 
        when ``norm_c=True``.
        The format should be `(min_value, max_value)`, where:
        - `min_value`: The lower bound for normalization.
        - `max_value`: The upper bound for normalization.
        If `None`, it defaults to `(0, 1)`.

    cbar : {'off', True, False}, optional
        Determines whether a colorbar is displayed:
        - ``'off'`` or `False`: No colorbar is shown.
        - `True`: A colorbar is added to the figure.
        Default is ``'off'``.

    fig_size : (float, float), optional
        Figure size in inches, e.g., ``(width, height)``. If `None`,
        a default size of approximately ``(10, 8)`` is used.

    title : str, optional
        Title of the diagram. If `None`, defaults to ``"Taylor Diagram"``.

    savefig : str or None, optional
        If provided with a string path such as ``"diagram.png"``, the
        figure is saved to that path. If `None`, the figure is only
        displayed. Default is `None`.

    Notes
    -----
    **Mathematical Formulation**

    The Taylor diagram displays two key statistics for each prediction
    :math:`p` compared to the reference :math:`r`:

    1. **Correlation** (:math:`\\rho`):
       .. math::
          \\rho = \\mathrm{corrcoef}(p, r)[0,1]

       where :math:`\\mathrm{corrcoef}` is the Pearson correlation
       coefficient, which can also be expressed as:

       .. math::
          \\rho = \\frac{\\mathrm{Cov}(p, r)}{\\sigma_p \\sigma_r}

       with :math:`\\mathrm{Cov}(p, r)` being the covariance, and
       :math:`\\sigma_p`, :math:`\\sigma_r` the standard deviations of
       :math:`p` and :math:`r`.

    2. **Standard Deviation** (:math:`\\sigma`):
       .. math::
          \\sigma = \\sqrt{\\frac{1}{n}
          \\sum_{i=1}^n\\left(p_i - \\bar{p}\\right)^2}

       where :math:`\\bar{p}` is the mean of the prediction array
       :math:`p`.

    On the diagram, the radial distance from the origin corresponds to
    the standard deviation of the prediction, and the polar angle
    corresponds to :math:`\\arccos(\\rho)` when `angle_to_corr` is
    `True`.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.ml_viz import plot_taylor_diagram_in
    >>> # Generate some synthetic data
    >>> np.random.seed(42)
    >>> reference = np.random.normal(0, 1, 100)
    >>> y_preds = [
    ...     reference + np.random.normal(0, 0.3, 100),
    ...     reference * 0.9 + np.random.normal(0, 0.8, 100)
    ... ]
    >>> plot_taylor_diagram_in(
    ...     *y_preds,
    ...     reference=reference,
    ...     names=['Model A', 'Model B'],
    ...     acov='half_circle',
    ...     zero_location='N',
    ...     direction=1,
    ...     fig_size=(8, 8)
    ... )

    See Also
    --------
    - :func:`numpy.corrcoef` : Function to compute correlation.
    - :func:`numpy.std` : Function to compute standard deviation.

    References
    ----------
    .. [1] Taylor, K. E. (2001). Summarizing multiple aspects of model
           performance in a single diagram. *Journal of Geophysical
           Research*, 106(D7), 7183-7192.
    """

    # Flatten the reference and predictions
    reference = np.ravel(reference)
    y_preds = [np.ravel(yp) for yp in y_preds]
    n = reference.size
    for p in y_preds:
        if p.size != n:
            raise ValueError(
                "All predictions and reference must be the same length."
            )

    # correlation & stdev
    corrs = [np.corrcoef(p, reference)[0,1] for p in y_preds]
    stds  = [np.std(p) for p in y_preds]
    ref_std = np.std(reference)

    # Setup figure & polar axis

    fig = plt.figure(figsize=fig_size or (10,8))
    ax  = fig.add_subplot(111, polar=True)

    # Decide coverage
    acov = acov or "half_circle"
    if acov == "half_circle":
        angle_max = np.pi/2
    else:
        angle_max = np.pi

    # radial limit
    rad_limit = max(max(stds), ref_std)*1.2

    # Create a mesh for background
    theta_grid = np.linspace(0, angle_max, shading_res)
    r_grid     = np.linspace(0, rad_limit, shading_res)
    TH, RR     = np.meshgrid(theta_grid, r_grid)

    if radial_strategy=="convergence": 
        # correlation => cos(TH)
        # correlation = cos(TH) if half or full circle
        # (when angle=0 => correlation=1, angle= pi/2 => corr=0, angle= pi => corr=-1)
        CC = np.cos(TH)  # from 1..-1 or 1..0 depending on coverage
    elif radial_strategy =="norm_r": 
        CC = RR / rad_limit  # Normalizes r to range [0, 1]
    else: 
        if radial_strategy in {'rwf', 'center_focus'}: 
            warnings.warn(
                f"'{radial_strategy}' is not available in the current"
                " plot. Consider using `gofast.plot.taylor_diagram`"
                " for better support. Alternatively, choose from"
                " 'convergence', 'norm_r', or 'performance'."
                " Defaulting to 'performance' visualization."
            )
        # Fallback to performance 
        best_idx = np.argmax(corrs)
        std_best = stds[best_idx]
        corr_best = corrs[best_idx]
        theta_best = np.arccos(corr_best)
        std_diff = (RR - std_best) ** 2
        theta_diff = (TH - theta_best) ** 2

        CC = (
            np.exp(-std_diff / 0.05) *
            np.exp(-theta_diff / 0.05)
        )
        
    # Define color values based on radial distance (normalized)
    if norm_c:
        if norm_range is None: 
            norm_range = (0, 1)
        norm_range = validate_length_range(
            norm_range, param_name="Normalized Range"
            )
        CC= minmax_scaler(CC, feature_range=norm_range)

    # plot background
    c = ax.pcolormesh(
        TH,
        RR,
        CC,
        cmap=cmap,
        shading=shading,
        vmin=-1 if angle_max==np.pi else 0,
        vmax=1
    )

    # convert each correlation to an angle
    angles = np.arccos(corrs)
    radii  = stds

    # pick distinct colors
    colors = plt.cm.Set1(np.linspace(0,1,len(y_preds)))
    names= columns_manager(names, empty_as_none=False)
    # plot predictions
    for i,(ang,rd) in enumerate(zip(angles,radii)):
        label = (names[i] if (names and i<len(names))
                 else f"Pred {i+1}")
        if not only_points:
            ax.plot([ang, ang],[0,rd],
                    color=colors[i], lw=2, alpha=0.8)
        ax.plot(ang, rd, marker=marker,
                color=colors[i], label=label)

    # reference arc
    if draw_ref_arc:
        arc_t = np.linspace(0, angle_max, 300)
        ax.plot(arc_t, [ref_std]*300,
                color=ref_color, lw=2, label="Reference")
    else:
        ax.plot([0,0],[0, ref_std],
                color=ref_color, lw=2, label="Reference")
        ax.plot(0, ref_std, marker=marker,
                color=ref_color)

    # set coverage
    ax.set_thetamax(np.degrees(angle_max))

    # direction
    if direction not in (-1,1):
        warnings.warn(
            "direction must be -1 or 1; using 1."
        )
        direction=1
    ax.set_theta_direction(direction)
    ax.set_theta_zero_location(zero_location)
    
    # Use coordinates and positions to avoid overlapping 
    CORR_POS =TDG_DIRECTIONS[str(direction)]["CORR_POS"]
    STD_POS =TDG_DIRECTIONS[str(direction)]["STD_POS"]
    
    corr_pos = CORR_POS.get(zero_location)[0]
    corr_kw= CORR_POS.get(zero_location)[1]
    std_pos = STD_POS.get(zero_location)[0]
    std_kw = STD_POS.get(zero_location)[1]
    
    # angle => corr labels
    if angle_to_corr:
        corr_ticks = np.linspace(0,1,corr_steps)
        angles_deg = np.degrees(np.arccos(corr_ticks))
        ax.set_thetagrids(
            angles_deg,
            labels=[f"{ct:.2f}" for ct in corr_ticks]
        )
        ax.text(
            *corr_pos, 
            "Correlation",
            ha='center', va='bottom',
            transform=ax.transAxes, 
            **corr_kw 
        )
        ax.text(
            *std_pos,
            'Standard Deviation',
            ha='center', va='bottom',
            transform=ax.transAxes, 
            **std_kw
        )
        
    else:
        ax.text(*corr_pos, "Angle (degrees)",
                ha='center', va='bottom',
                transform=ax.transAxes,
                **corr_kw 
                )
        
        ax.text(
            *std_pos,
            'Standard Deviation',
            ha='center', va='bottom',
            transform=ax.transAxes, 
            **std_kw

        )
        
    ax.set_ylim(0, rad_limit)
    ax.set_rlabel_position(15)
    title = title or "Taylor Diagram"
    ax.set_title(title, pad=60)
 
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.1))

    if cbar not in ["off", False]:
        fig.colorbar(c, ax=ax, pad=0.1, label="Correlation")
        
    plt.tight_layout()
    plt.show()


@validate_params ({
    'reference': ['array-like'], 
    'names': [str, 'array-like', None ], 
    'acov': [StrOptions({'default', 'half_circle'})], 
    'zero_location': [StrOptions({'N','NE','E','S','SW','W','NW', 'SE'})], 
    'direction': [Integral]
    })
def plot_taylor_diagram(
    *y_preds: np.ndarray,
    reference:  np.ndarray,
    names: Optional[List[str]] = None,
    acov: str = "half_circle",
    zero_location: str = 'W',
    direction: int = -1,
    only_points: bool = False,
    ref_color: str = 'red',
    draw_ref_arc: bool = ...,
    angle_to_corr: bool = ..., 
    marker='o', 
    corr_steps=6,
    fig_size: Optional[Tuple[int, int]] = None,
    title: Optional[str]=None, 
    savefig: Optional[str]=None, 
):
    """
    Plots a Taylor Diagram, which is used to graphically summarize 
    how closely a set of predictions match observations. The diagram 
    displays the correlation between each prediction and the 
    observations (`reference`) as the angular coordinate and the 
    standard deviation as the radial coordinate.

    Parameters
    ----------
    y_preds : variable number of `ArrayLike`
        Each argument is a one-dimensional array containing the 
        predictions from different models. Each prediction array 
        should be the same length as the `reference` data array.

    reference : `ArrayLike`
        A one-dimensional array containing the reference data against 
        which the predictions are compared. This should have the same 
        length as each prediction array.

    names : list of `str`, optional
        A list of names for each set of predictions. If provided, this 
        list should be the same length as the number of `y_preds`. 
        If not provided, predictions will be labeled as "Prediction 1", 
        "Prediction 2", etc.

    acov : `str`, optional
        Determines the angular coverage of the plot.
        
        - `"default"`: The plot spans 180 degrees, typically covering
          from the West (270°) to the East (90°).
        - `"half_circle"`: The plot spans 90 degrees, which can be 
          useful for focused comparisons.
          
        **Default:** `"half_circle"`

    zero_location : `str`, optional
        Specifies the location of the zero-degree angle on the polar plot.
        This determines where the correlation coefficient of 1 (perfect
        correlation) is placed on the diagram.
        
        **Accepted Values:**
        
        - `'N'`: North (top of the plot, 0°)
        - `'NE'`: Northeast (45°)
        - `'E'`: East (right side, 90°)
        - `'SE'`: Southeast (135°)
        - `'S'`: South (bottom, 180°)
        - `'SW'`: Southwest (225°)
        - `'W'`: West (left side, 270°)
        - `'NW'`: Northwest (315°)
        
        **Default:** `'W'` (West)

        **Effects:**
        
        - **Positioning:** Changes the orientation of the diagram by rotating 
          the zero-degree line.
        - **Interpretation:** Influences how the angular coordinates
          correspond to correlation values.
        
        **Example:**
        
        - Setting `zero_location='N'` places the zero-degree correlation 
          at the top of the plot.
        - Setting `zero_location='E'` places it on the right side.

    direction : `int`, optional
        Determines the direction in which the angles increase on the polar plot.
        
        **Accepted Values:**
        
        - `1`: Counter-clockwise direction. Angles increase in the 
          traditional mathematical sense, moving from the zero location 
          upwards.
        - `-1`: Clockwise direction. Angles increase in the opposite 
          direction, moving from the zero location downwards.
        
        **Default:** `-1` (Clockwise)

        **Effects:**
        
        - **Angle Progression:** Dictates whether the angles move clockwise or
          counter-clockwise from the zero location.
        - **Visual Interpretation:** Affects the layout of the correlations 
          on the diagram.
        
        **Example:**
        
        - `direction=1`: Correlation angles increase counter-clockwise, which 
          might align with standard mathematical conventions.
        - `direction=-1`: Correlation angles increase clockwise, which might 
          be preferred for specific visualization standards.

    fig_size : `tuple`, optional
        The size of the figure in inches as a tuple `(width, height)`. 
        If not provided, defaults to `(10, 8)`.
        
    only_points : :class:`bool`, optional
        If `only_points` is ``True``, only the markers for each prediction
        are  drawn; the radial line from the origin to the marker is omitted.
        Default is ``False``.

    ref_color : :class:`str`, optional
        Color to use for the reference arc or line. Default is ``'red'``.

    draw_ref_arc : :class:`bool`, optional
        If `draw_ref_arc` is ``True``, the  reference standard deviation 
        is shown as an arc in the diagram. If ``False``, a radial line is 
        drawn in its place. Default is ``True``.

    angle_to_corr : :class:`bool`, optional
        If `angle_to_corr` is ``True``, the angular ticks are replaced with 
        correlation values from ``0.0`` to ``1.0``. If ``False``, angles in 
        degrees are displayed. Default is ``True``.

    marker : :class:`str`, optional
        Marker style (e.g. ``'o'`` for circles, ``'s'`` for squares). 
        Default is ``'o'``.

    corr_steps : :class:`int`, optional
        Number of ticks between 0 and 1 when `angle_to_corr` is ``True``. 
        Default is ``6``.

    set_corr_angle : :class:`bool`, optional
        If `set_corr_angle` is ``True``, the correlation approach is used 
        to set angle ticks automatically. This is typically used along with
        `angle_to_corr`. Default is ``True``.
        
    title : str, optional
        Title of the diagram. If `None`, defaults to ``"Taylor Diagram"``.

    savefig : str or None, optional
        If provided with a string path such as ``"diagram.png"``, the
        figure is saved to that path. If `None`, the figure is only
        displayed. Default is `None`.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.ml_viz import plot_taylor_diagram
    >>> y_preds = [
    ...     np.random.normal(loc=0, scale=1, size=100),
    ...     np.random.normal(loc=0, scale=1.5, size=100)
    ... ]
    >>> reference = np.random.normal(loc=0, scale=1, size=100)
    >>> plot_taylor_diagram(
    ...     *y_preds, 
    ...     reference=reference, 
    ...     names=['Model A', 'Model B'], 
    ...     acov='half_circle',
    ...     zero_location='N',
    ...     direction=1,
    ...     fig_size=(12, 10)
    ... )

    Notes
    -----
    Taylor diagrams provide a visual way of assessing multiple 
    aspects of prediction performance in terms of their ability to 
    reproduce observational data. It's particularly useful in the 
    field of meteorology but can be applied broadly to any predictive 
    models that need comparison to a reference.

    The angular coordinate on the Taylor Diagram represents the 
    correlation coefficient :math:`R` between each prediction and the 
    reference, calculated as:

    .. math::
        R = \frac{\sum_{i=1}^n (y_i - \bar{y})(x_i - \bar{x})}
        {\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2} 
        \sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}}

    where :math:`y_i` are the predictions, :math:`x_i` are the 
    observations, and :math:`\bar{y}` and :math:`\bar{x}` are the 
    means of the predictions and observations, respectively.

    The radial coordinate represents the standard deviation :math:`\sigma`
    of the predictions, calculated as:

    .. math::
        \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2}


    References
    ----------
    .. [1] K. P. Taylor, "Summarizing multiple aspects of model performance 
       in a single diagram," Journal of Geophysical Research, vol. 106, 
       no. D7, pp. 7183-7192, 2001.
    """

    # Convert inputs to 1D numpy arrays
    y_preds = [np.asarray(pred).flatten() for pred in y_preds]
    reference = np.asarray(reference).flatten()

    # Check consistency of lengths
    assert all(pred.size == reference.size for pred in y_preds), (
        "All predictions and the reference must be of the same length."
    )

    # Compute correlation and std dev for each prediction
    correlations = [np.corrcoef(pred, reference)[0, 1] for pred in y_preds]
    standard_deviations = [np.std(pred) for pred in y_preds]
    reference_std = np.std(reference)

    # standard_deviations= normalize_array(
    #     standard_deviations, normalize = "auto", method="01"
    #     )
    # correlations= normalize_array(
    #     correlations, normalize = "auto", method="01"
    #     )
    # Create figure and polar subplot
    fig = plt.figure(figsize=fig_size or (10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Convert correlation to angles (in radians)
    # angle = arccos(corr), so perfect correlation = 0 rad,
    # zero correlation = pi/2 rad, negative correlation = > pi/2, etc.
    angles = np.arccos(correlations)
    radii = standard_deviations

    # Plot each prediction
    # Use a color cycle so lines/points are more distinguishable
    colors = plt.cm.Set1(np.linspace(0, 1, len(y_preds)))
    for i, (angle, radius) in enumerate(zip(angles, radii)):
        label = names[i] if (names and i < len(names)) else f'Prediction {i+1}'
        if not only_points:
            # Draw the radial line from origin to the point
            ax.plot([angle, angle], [0, radius], color=colors[i], lw=2, alpha=0.8)
        ax.plot(angle, radius, marker, color=colors[i], label=label)

    # Draw the reference as a red arc if requested
    # This arc will have radius = reference_std
    if draw_ref_arc:
        if acov == "half_circle":
            theta_arc = np.linspace(0, np.pi/2, 300)
        else:
            theta_arc = np.linspace(0, np.pi, 300)

        ax.plot(theta_arc, [reference_std]*len(theta_arc),
                color=ref_color, lw=2, label='Reference')
    else:
        # If not drawing the arc, revert to a radial line as fallback
        ax.plot([0, 0], [0, reference_std], color=ref_color, lw=2, label='Reference')
        ax.plot(0, reference_std, marker, color=ref_color)

    # Set coverage (max angle)
    if acov == "half_circle":
        ax.set_thetamax(90)  # degrees
    else:
        ax.set_thetamax(180) # degrees (default)

    # Set direction (1=counterclockwise, -1=clockwise)
    if direction not in [-1, 1]:
        warnings.warn(
            "direction should be either 1 (CCW) or -1 (CW). "
            f"Got {direction}. Resetting to 1 (CCW)."
        )
        direction = 1
    ax.set_theta_direction(direction)
    ax.set_theta_zero_location(zero_location)
    
    CORR_POS =TDG_DIRECTIONS[str(direction)]["CORR_POS"]
    STD_POS =TDG_DIRECTIONS[str(direction)]["STD_POS"]
    
    corr_pos = CORR_POS.get(zero_location)[0]
    corr_kw= CORR_POS.get(zero_location)[1]
    std_pos = STD_POS.get(zero_location)[0]
    std_kw = STD_POS.get(zero_location)[1]
 
    # Replace angle ticks with correlation values if requested
    if angle_to_corr:
        # We'll map correlation ticks [0..1] -> angle via arccos
        # e.g. 1 -> 0 rad, 0 -> pi/2 or pi, depending on coverage
        # Just pick some correlation tick steps
        corr_ticks = np.linspace(0, 1, corr_steps)  # 0, 0.2, 0.4, 0.6, 0.8, 1
        angle_ticks = np.degrees(np.arccos(corr_ticks))
        # In half-circle mode, correlation from 0..1 fits in 0..pi/2,
        # in default mode, 0..1 fits in 0..pi. This still works generally.
        ax.set_thetagrids(angle_ticks, labels=[f"{ct:.2f}" for ct in corr_ticks])
        # We can label this dimension as 'Correlation'
        ax.set_ylabel('')  # remove default 0.5, 1.06
  
        ax.text(*corr_pos,  'Correlation', ha='center', va='center', 
                transform=ax.transAxes, **corr_kw)
        ax.text(*std_pos,  'Standard Deviation', ha='center', va='center', 
                transform=ax.transAxes, **std_kw)
    else:
        # Keep angle as degrees
        ax.set_ylabel('')  # remove default
        ax.text(*corr_pos, 'Angle (degrees)', ha='center', va='center',
                transform=ax.transAxes)

    # Adjust radial label (std dev)
    # This tries to reduce label overlap
    ax.set_rlabel_position(22.5)
    ax.set_title( title or 'Taylor Diagram', pad=60) #50
    # ax.set_xlabel('Standard Deviation', labelpad=15)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
    # plt.subplots_adjust(top=0.8)

    plt.tight_layout()
    plt.show()