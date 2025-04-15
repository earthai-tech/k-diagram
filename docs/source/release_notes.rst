.. _release_notes:

===============
Release Notes
===============

This document tracks the changes, new features, and bug fixes for
each release of the `k-diagram` package.

----------------
Version 1.0.3
----------------
*(Released: 2025-04-15)* 

This release focuses on improving documentation robustness, fixing bugs
identified during documentation generation and testing, refining the
dataset API, and enhancing visual styling.

✨ Enhancements
~~~~~~~~~~~~~~~~~~

* **Added** new synthetic dataset generators for specific use cases:
    * :func:`~kdiagram.datasets.make_taylor_data`
    * :func:`~kdiagram.datasets.make_multi_model_quantile_data`
    * :func:`~kdiagram.datasets.make_cyclical_data`
    * :func:`~kdiagram.datasets.make_fingerprint_data`
* **Added** :func:`~kdiagram.datasets.load_zhongshan_subsidence`
    function to load packaged sample data, including logic for
    caching and optional downloading.
* **Refined** dataset API: Introduced ``as_frame`` parameter to
    dataset loading/generation functions (like
    :func:`~kdiagram.datasets.load_uncertainty_data`) to return
    either a structured :class:`~kdiagram.bunch.Bunch` object (with
    metadata and ``DESCR``) or a plain :class:`pandas.DataFrame`.
* **Refactored** model comparison plotting into
    :mod:`kdiagram.plot.comparison` with the function
    :func:`~kdiagram.plot.comparison.plot_model_comparison`.
* **Added** utility module :mod:`kdiagram.utils.metric_utils`
    including a :func:`~kdiagram.utils.metric_utils.get_scorer`
    function.
* **Added** utility module :mod:`kdiagram.utils.io` and dataset
    property helpers in :mod:`kdiagram.datasets._property` (adapted
    from `gofast`).
* **Improved** CLI implementation (`kdiagram.cli`) with argument
    parsing for all core plotting functions and better help messages.
* **Enhanced** documentation theme (`custom.css`) with a refined
    color palette derived from the package logo, improved table
    styling, clearer admonitions, and smoother hover effects.
* **Added** `CODE_OF_CONDUCT.md` to the project.

🐛 Bug Fixes
~~~~~~~~~~~~~~~

* **Fixed** numerous ReStructuredText formatting errors in function
    docstrings across multiple modules (e.g., `evaluation.py`,
    `uncertainty.py`, `feature_based.py`), resolving critical errors
    and warnings during strict Sphinx builds (`-W`). This includes
    corrections to section headers, underlines, indentation, blank
    lines, list formatting, and footnote references. (#nnn)
* **Fixed** `ValueError` in
    :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`
    during normalization (`normalize=True`) caused by incorrect NumPy
    broadcasting. (#nnn)
* **Fixed** logic in dataset loading functions (e.g.,
    `load_uncertainty_data`, `load_zhongshan_subsidence`) to
    correctly populate `q10_cols`, `q50_cols`, `q90_cols` attributes
    in the returned Bunch object by using consistent dictionary keys.
    (#nnn)
* **Fixed** Matplotlib warning in
    :func:`~kdiagram.plot.relationship.plot_relationship` by using
    `color=` instead of `c=` for scatter plots with single color
    specifications. (#nnn)
* **Fixed** potential division-by-zero errors in normalization steps
    within dataset generators and plotting functions when input data
    has zero range.
* **Corrected** logic in internal dataset download helper
    (`download_file_if_missing`) to prioritize package resources,
    then cache, then download (to cache), resolving issues with file
    location and potentially incorrect download paths. (#nnn)
* **(Potentially Fixed)** Addressed underlying configuration issues
    in `setup.py` (e.g., removed internal dependency installs,
    hardcoded version temporarily) that likely caused `twine check`
    errors due to missing Name/Version metadata in wheel files. (#nnn)

📝 Documentation
~~~~~~~~~~~~~~~~~~~

* **Fixed** errors in `index.rst` related to incorrect usage of
    `sphinx-design` directives (`container :margin:`, `panels`,
    `button-ref :text:`). Replaced `include` directive with direct
    RST content where `myst-parser` failed. (#nnn)
* **Fixed** numerous minor RST warnings (e.g., title underlines,
    toctree entries, broken links) across documentation files.
* **Restructured** Gallery into sub-directories (`gallery/index.rst`
    linking to `plots/`, `utils/`, etc. - *Self-correction based on user actions*: now links to `uncertainty.rst`, `evaluation.rst`, etc.)
* **Added** detailed "Analysis and Interpretation" sections to all
    examples in the Plot Gallery (`gallery/*.rst`) using `topic`
    directives.
* **Added** User Guide pages for `Datasets`, `Model Comparison`,
    and `Motivation`. Updated User Guide index. (#nnn)
* **Added** `CODE_OF_CONDUCT.md` and linked from `CONTRIBUTING.rst`.
* **Added** `CITING.rst` with instructions for citing software and
    related papers.
* **Added** `GLOSSARY.rst`.
* **Added** new badges (Build Status, Python Versions, etc.) to
    `README.md`. Corrected Markdown comment syntax. Added HTML image
    tags to control image size in README.
* **Updated** `docs/requirements.txt` and `.readthedocs.yml` for
    correct documentation builds, including setting
    `fail_on_warning: false` temporarily.
* **Cleaned up** docstrings for utility and compatibility modules
    with proper attribution.


----------------
Version 1.0.0
----------------
*(Released: 2025-04-10)*

Initial Release
~~~~~~~~~~~~~~~~~

This is the first public release of the `k-diagram` package.

**Key Features Included:**

* **Uncertainty Visualization Suite (`kdiagram.plot.uncertainty`):**
    * :func:`~kdiagram.plot.uncertainty.plot_actual_vs_predicted`:
        Compare actual vs. point predictions.
    * :func:`~kdiagram.plot.uncertainty.plot_anomaly_magnitude`:
        Visualize magnitude and type of prediction interval failures.
    * :func:`~kdiagram.plot.uncertainty.plot_coverage`: Calculate
        and plot overall coverage scores (bar, line, pie, radar).
    * :func:`~kdiagram.plot.uncertainty.plot_coverage_diagnostic`:
        Diagnose point-wise interval coverage on a polar plot.
    * :func:`~kdiagram.plot.uncertainty.plot_interval_consistency`:
        Assess stability of interval width over time (Std Dev / CV).
    * :func:`~kdiagram.plot.uncertainty.plot_interval_width`:
        Visualize prediction interval width magnitude across samples.
    * :func:`~kdiagram.plot.uncertainty.plot_model_drift`: Track
        average interval width drift across forecast horizons (polar bars).
    * :func:`~kdiagram.plot.uncertainty.plot_temporal_uncertainty`:
        General polar scatter for comparing multiple series (e.g., quantiles).
    * :func:`~kdiagram.plot.uncertainty.plot_uncertainty_drift`:
        Visualize drift of uncertainty patterns using concentric rings.
    * :func:`~kdiagram.plot.uncertainty.plot_velocity`: Visualize
        rate of change (velocity) of median predictions.
* **Model Evaluation (`kdiagram.plot.evaluation`):**
    * Taylor Diagram functions (:func:`~kdiagram.plot.evaluation.taylor_diagram`,
        :func:`~kdiagram.plot.evaluation.plot_taylor_diagram_in`,
        :func:`~kdiagram.plot.evaluation.plot_taylor_diagram`) for
        summarizing model skill (correlation, standard deviation, RMSD).
* **Feature Importance (`kdiagram.plot.feature_based`):**
    * :func:`~kdiagram.plot.feature_based.plot_feature_fingerprint`:
        Radar charts for comparing feature importance profiles.
* **Relationship Visualization (`kdiagram.plot.relationship`):**
    * :func:`~kdiagram.plot.relationship.plot_relationship`: Polar
        scatter mapping true values to angle and predictions to radius.
* **Utility Functions (`kdiagram.utils`):**
    * Helpers for detecting, building names for, and reshaping quantile
        data in DataFrames (:func:`~kdiagram.utils.detect_quantiles_in`,
        :func:`~kdiagram.utils.build_q_column_names`,
        :func:`~kdiagram.utils.reshape_quantile_data`,
        :func:`~kdiagram.utils.melt_q_data`,
        :func:`~kdiagram.utils.pivot_q_data`).
* **Command-Line Interface (CLI):**
    * `k-diagram` command for generating core plots directly from CSV
        files via the terminal.
* **Documentation:**
    * Initial version including Installation Guide, Quick Start, User
        Guide (concepts & interpretation), Plot Gallery, Utility Examples,
        API Reference, Contribution Guidelines, and License.