# k-diagram: Rethinking Forecasting Uncertainty via Polar-Based Visualization

`k-diagram` is a Python package designed to provide specialized diagnostic polar plots, called "k-diagrams," for comprehensive model evaluation and forecast analysis. These visualizations focus on **forecast uncertainty**, **model drift**, **interval coverage**, **anomaly magnitude**, **actual vs. predicted performance**, **feature influence**, and other related diagnostics, all using polar coordinates.

The package is specifically designed for uncertainty-aware forecasting and is ideal for applications in environmental sciences, where forecasting uncertainty plays a key role in decision-making, such as in **land subsidence**, **flood forecasting**, and **climate prediction**.

---

## Key Features

- **Polar-based Visualizations**: Use polar diagrams to visualize multi-dimensional uncertainty and model behavior.
- **Uncertainty Diagnostics**: Analyze and visualize uncertainty in forecast intervals, model drift over time, anomaly magnitudes, and more.
- **Forecast Evaluation**: Visualize the coverage, consistency, and performance of your forecasting models.
- **Feature Impact**: Provide insight into which features dominate the predictions of your model over different spatial and temporal regions.
- **Model Degradation**: Track how forecast uncertainty grows with the forecast horizon and identify regions where predictions become less reliable.

---

## Installation

### Install via pip
You can install the package directly from PyPI:

```bash
pip install k-diagram
```

Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/your-username/k-diagram.git
cd k-diagram
pip install .
```

---

## Usage

After installing `k-diagram`, you can import the package into your Python code for various plotting and diagnostic tasks:

```python
import kdiagram as kd

# Example usage of the package to plot a coverage diagnostic
kd.plot_coverage_diagnostic(df, actual_col='subsidence_2023', q_cols=['subsidence_2023_q10', 'subsidence_2023_q90'])
```

### Example Plots Available

- **Plot Actual vs Predicted**: Compare the actual values against the predicted values to assess model accuracy.
- **Plot Interval Consistency**: Visualize the consistency of your model's prediction intervals.
- **Plot Anomaly Magnitude**: Identify regions where the model over- or under-predicts compared to actual values.
- **Plot Model Drift**: Track how uncertainty grows with the forecast horizon over time.
- **Plot Feature Fingerprint**: Visualize the importance of various input features across time or spatial zones.

---

## Available Functions

- `plot_actual_vs_predicted(df, actual_col, qlow_col, qup_col)`
- `plot_anomaly_magnitude(df, actual_col, qlow_col, qup_col)`
- `plot_coverage_diagnostic(df, actual_col, qcols)`
- `plot_feature_fingerprint(df, features, importances)`
- `plot_interval_consistency(df, q10_col, q90_col)`
- `plot_interval_width(df, q10_col, q90_col)`
- `plot_model_drift(df, q_cols)`
- `plot_temporal_uncertainty(df, q_cols)`
- `plot_uncertainty_drift(df, q_cols)`
- `plot_velocity(df, q_cols)`

---

## Example Code Snippets

**Example 1: Plotting Coverage Diagnostics**
```python
import kdiagram as kd
import pandas as pd

# Load your dataframe (replace with actual data loading process)
df = pd.read_csv('your_subsidence_data.csv')

# Plot coverage diagnostic
kd.plot_coverage_diagnostic(df, actual_col='subsidence_2023', qlow_col='subsidence_2023_q10', qup_col='subsidence_2023_q90')
```

**Example 2: Plotting Interval Consistency**
```python
import kdiagram as kd
import pandas as pd

# Load your dataframe (replace with actual data loading process)
df = pd.read_csv('your_subsidence_data.csv')

# Plot interval consistency
kd.plot_interval_consistency(df, q10_col='subsidence_2023_q10', q90_col='subsidence_2023_q90')
```

---

## Contributing

We welcome contributions to `kdiagram`. If you have ideas for improvements or bug fixes, feel free to submit a pull request. Please follow the steps below to contribute:

1. Fork the repository.
2. Clone your forked repository.
3. Create a new branch (`git checkout -b feature-branch`).
4. Make your changes and commit them.
5. Push to your forked repository (`git push origin feature-branch`).
6. Create a pull request to the `main` branch of this repository.

---

## License

`k-diagram` is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, feel free to reach out to the author:

- **Email**: etanoyau@gmail.com
- **GitHub**: https://github.com/your-username/kdiagram

---

This `README.md` provides clear instructions on how to install and use the `kdiagram` package, with detailed usage examples, installation steps, and contribution guidelines. It also gives an overview of the package’s functionality and provides relevant references.
