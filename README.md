<div align="center">

  <!-- Centered Logo -->
  <img src="docs/source/_static/k_diagram.svg" alt="k-diagram logo" width="300"><br>

  <!-- Centered Title -->
  <h1>Polar Diagnostics for Forecast Uncertainty</h1>

  <!-- Centered Badges -->
<p>
    <a href="https://github.com/earthai-tech/k-diagram/actions/workflows/python-package-conda.yml"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/earthai-tech/k-diagram/python-package-conda.yml?branch=main&style=flat-square"></a>
    <a href="https://k-diagram.readthedocs.io/en/latest/?badge=latest"><img alt="Docs Status" src="https://readthedocs.org/projects/k-diagram/badge/?version=latest&style=flat-square"></a>
    <a href="https://github.com/earthai-tech/k-diagram/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/earthai-tech/k-diagram?style=flat-square&logo=apache&color=purple"></a>
    <a href="https://github.com/psf/black"><img alt="Black" src="https://img.shields.io/badge/code%20style-Black-000000.svg?style=flat-square"></a>
    <a href="https://github.com/earthai-tech/k-diagram/blob/main/CONTRIBUTING.md"><img alt="Contributions" src="https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=flat-square"></a>
    <a href="https://codecov.io/gh/earthai-tech/k-diagram"><img alt="Codecov" src="https://codecov.io/gh/earthai-tech/k-diagram/branch/main/graph/badge.svg"></a>
  <img alt="GitHub Release" src="https://img.shields.io/github/v/release/earthai-tech/k-diagram">

  </p>
  
  <!-- Centered Introduction -->
  <p>
    <em>k-diagram</em> provides polar diagnostic plots to evaluate
    forecast models with an emphasis on uncertainty. It helps you look
    beyond single metrics and understand <strong>where</strong> and <strong>why</strong> models
    behave as they do.
  </p>

</div>

-----------------------------------------------------

## ✨ Why k-diagram?

Key questions it helps answer:

- **Forecast uncertainty** — Are prediction intervals well calibrated?
- **Model drift** — Does performance degrade over time or horizon?
- **Anomalies** — Where do predictions miss, and by how much?
- **Patterns** — How does accuracy vary across conditions or locations?

The package is designed with applied settings in mind, including
environmental forecasting (subsidence, floods, climate impacts), but
is general enough for many time-series and geospatial tasks.

-----

## 📥 Installation

### From PyPI (recommended)

```bash
pip install k-diagram
````

This installs *k-diagram* and the scientific Python stack it depends
on (NumPy, Pandas, SciPy, Matplotlib, Seaborn, scikit-learn). Python
3.9+ is supported.

### Development install (editable)

If you plan to contribute or run tests locally:

```bash
git clone https://github.com/earthai-tech/k-diagram.git
cd k-diagram
pip install -e .[dev]
```

The `[dev]` extra installs pytest, coverage, Sphinx, Ruff (Black), and
other developer tools defined in `pyproject.toml`.

### Reproducible dev environment via conda

We ship an `environment.yml` mirroring our CI setup. It includes
runtime deps plus test and docs tooling.

```bash
git clone https://github.com/earthai-tech/k-diagram.git
cd k-diagram
conda env create -f environment.yml
conda activate k-diagram-dev
python -m pip install . --no-deps --force-reinstall
```

> For more detailed instructions, including how to **build the documentation locally**, 
> please see the full [Installation Guide](https://k-diagram.readthedocs.io/en/latest/installation#building-documentation) 
> in our documentation.
> 
> Tip: Prefer a virtual environment (either `venv` or `conda`) to keep
> project dependencies isolated.

-----

## ⚡ Quick Start

Visualize how the entire spatial pattern of forecast uncertainty evolves 
over multiple time steps with `plot_uncertainty_drift`. This plot uses 
concentric rings to represent each forecast period, revealing at a glance 
how the "map" of uncertainty changes over time.

```python
# Code Snippet
import kdiagram as kd
# (Requires df with multiple qlow/qup cols like sample_data_drift_uncertainty)
# Example using dummy data generation:
import pandas as pd
import numpy as np
years = range(2023, 2028) # 2028 excluded
N=100
df_drift = pd.DataFrame({'id': range(N)})
qlow_cols, qup_cols = [], []
for i, year in enumerate(years):
   ql, qu = f'q10_{year}', f'q90_{year}'
   qlow_cols.append(ql); qup_cols.append(qu)
   base = np.random.rand(N)*5; width=(np.random.rand(N)+0.5)*(1+i)
   df_drift[ql] = base; df_drift[qu]=base+width

kd.plot_uncertainty_drift(
    df_drift,
    qlow_cols=qlow_cols,
    qup_cols=qup_cols,
    acov="half_circle", 
    dt_labels=[str(y) for y in years],
    title='Uncertainty Drift (Interval Width)'
)
```

<p align="center">
    <img
    src="docs/source/_static/readme_uncertainty_drift_plot.png"
    alt="Uncertainty Drift Plot"
    width="500"
    />
</p>

-----

## 📊 Explore the Visualizations

The Quick Start shows just one of the many specialized plots available. 
The full documentation gallery showcases the complete suite of 
diagnostic tools. Discover how to:

- **Compare Models:** Use radar charts to weigh the trade-offs between 
  accuracy, speed, and other metrics.
- **Diagnose Reliability:** Go beyond accuracy with calibration spirals to 
  see if your probability forecasts are trustworthy.
- **Analyze Error Structures:** Uncover hidden biases and patterns in your 
  model's mistakes with polar residual plots.
- **Understand Feature Effects:** Visualize feature importance "fingerprints" 
  and complex two-way feature interactions.

➡️ **Explore the [Full Gallery](https://k-diagram.readthedocs.io/en/latest/gallery/index.html)**

-----
## 📚 Documentation

For detailed usage, API reference, and more examples, please 
visit the official documentation:

**[k-diagram.readthedocs.io](https://k-diagram.readthedocs.io/)** 

-----

## 💻 Using the CLI

`k-diagram` also provides a command-line interface for generating 
plots directly from CSV files.

**Check available commands:**

```bash
k-diagram --help
```

**Example: Generate a Coverage Diagnostic plot:**

```bash
k-diagram plot_coverage_diagnostic data.csv \
    --actual-col actual_obs \
    --q-cols q10_pred q90_pred \
    --title "Coverage for My Model" \
    --savefig coverage_plot.png
```

See **[k-diagram <command> --help](https://k-diagram.readthedocs.io/en/latest/cli/introduction.html)**
for options specific to each plot type).

-----

## 🙌 Contributing

Contributions are welcome. Please:

1.  Check the **[Issues Tracker](https://github.com/earthai-tech/k-diagram/issues)** 
    for existing bugs or ideas.
2.  Fork the repository.
3.  Create a new branch for your feature or fix.
4.  Make your changes and add tests.
5.  Submit a Pull Request.

Please refer to the [CONTRIBUTING](https://k-diagram.readthedocs.io/en/latest/contributing.html) 
page or the contributing section in the documentation 
for more detailed guidelines.

-----

## 📜 License

`k-diagram` is distributed under the terms of the **Apache License 2.0**. 
See the [LICENSE](https://github.com/earthai-tech/k-diagram/blob/main/LICENSE) 
file for details.

-----

## 📞 Contact & Support

  * **Bug Reports & Feature Requests:** The best place to report issues,
    ask questions about usage, or request new features is the
    [**GitHub Issues**](https://github.com/earthai-tech/k-diagram/issues) page for the project.

  * **Author Contact:** For direct inquiries related to the project's
    origins or specific collaborations, you can reach the author:

      * **Name:** Laurent Kouadio
      * 📧 **Email:** [etanoyau@gmail.com](mailto:etanoyau@gmail.com)

---

[badge-ci]: https://img.shields.io/github/actions/workflow/status/earthai-tech/k-diagram/python-package-conda.yml?branch=main&style=flat-square
[link-ci]: https://github.com/earthai-tech/k-diagram/actions/workflows/python-package-conda.yml
[badge-docs]: https://readthedocs.org/projects/k-diagram/badge/?version=latest&style=flat-square
[link-docs]: https://k-diagram.readthedocs.io/en/latest/?badge=latest
[badge-license]: https://img.shields.io/github/license/earthai-tech/k-diagram?style=flat-square&logo=apache&color=purple
[badge-black]: https://img.shields.io/badge/code%20style-Black-000000.svg?style=flat-square
[link-black]: https://github.com/psf/black
[badge-contrib]: https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=flat-square
[link-contrib]: https://github.com/earthai-tech/k-diagram/blob/main/CONTRIBUTING.md
[badge-codecov]: https://codecov.io/gh/earthai-tech/k-diagram/branch/main/graph/badge.svg
[link-codecov]: https://codecov.io/gh/earthai-tech/k-diagram


