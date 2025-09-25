# Author: LKouadio <etanoyau@gmail.com>
# License: Apache License 2.0

from __future__ import annotations

from numbers import Integral, Real
from typing import (
    Literal,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

from .compat.sklearn import Hidden, Interval, StrOptions, validate_params
from .utils.validator import check_consistent_length, exist_features

__all__ = ["clustered_anomaly_severity", "cluster_aware_severity_score"]


ArrayLike = Union[np.ndarray, pd.Series]


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
        "window_size": [Integral],
        "sort_by": ["array-like", None],
        "normalize": [StrOptions({"none", "band", "mad"})],
        "density_source": [StrOptions({"indicator", "magnitude"})],
        "kernel": [StrOptions({"box", "triangular", "gaussian", "epan"})],
        "eps": [Hidden(Interval(Real, 0, 1, closed="neither"))],
        "gamma": [Interval(Real, 0, None, closed="neither")],
        "lambda_": [Interval(Real, 0, None, closed="neither")],
        "multioutput": [StrOptions({"uniform_average", "raw_values"})],
        "nan_policy": [StrOptions({"omit", "propagate", "raise"})],
        "return_details": [bool],
    }
)
def cluster_aware_severity_score(
    y_true,
    y_pred,
    *,
    sample_weight: ArrayLike | None = None,
    window_size: int = 21,
    sort_by: ArrayLike | None = None,
    normalize: str = "band",
    density_source: str = "indicator",
    kernel: str = "box",
    lambda_: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-12,
    multioutput: str = "uniform_average",
    nan_policy: str = "omit",
    return_details: bool = False,
):
    y_true = check_array(y_true, ensure_2d=False, ensure_all_finite=False)
    y_pred = check_array(
        y_pred, ensure_2d=False, allow_nd=True, ensure_all_finite=False
    )

    check_consistent_length(y_true, y_pred, sample_weight)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    # allow (n,), (n,n_out) for sort_by
    if sort_by is not None:
        sort_by = check_array(
            sort_by, ensure_2d=False, ensure_all_finite=False
        )
        check_consistent_length(y_true, sort_by)
        if sort_by.ndim == 1:
            sort_by = sort_by.reshape(-1, 1)

    bounds = _split_bounds(y_pred)
    # each (L,U) set must contain at least some finite values
    for i, (lo_i, up_i) in enumerate(bounds):
        lo_fin = np.isfinite(lo_i)
        up_fin = np.isfinite(up_i)
        # if one side is entirely non-finite OR both sides
        # are entirely non-finite
        if not lo_fin.any() or not up_fin.any():
            raise ValueError(
                f"y_pred interval set {i} has no finite "
                "lower/upper bounds across all samples."
            )

    n_out = max(1, y_true.shape[1])

    #  broadcast single-output target to the number of (L,U) sets
    if n_out == 1 and len(bounds) > 1:
        y_true = np.repeat(y_true, len(bounds), axis=1)
        n_out = len(bounds)

    # reject single interval pair when y_true has multiple outputs
    if len(bounds) == 1 and n_out > 1:
        raise ValueError(
            f"y_pred provides 1 interval set for {n_out} outputs; "
            "provide one (L,U) pair per output."
        )

    if len(bounds) not in (1, n_out):
        raise ValueError(
            "y_pred outputs mismatch: got "
            f"{len(bounds)} sets for {n_out} outputs."
        )

    scores = []
    details = [] if return_details else None

    for i in range(len(bounds)):
        lo_i, up_i = bounds[i]
        y_i = y_true[:, min(i, y_true.shape[1] - 1)]
        sb_i = None
        if sort_by is not None:
            sb_i = sort_by[:, min(i, sort_by.shape[1] - 1)]

        res = _cas_core(
            y_i,
            lo_i,
            up_i,
            window_size=window_size,
            sort_by=sb_i,
            normalize=normalize,
            density_source=density_source,
            kernel=kernel,
            lambda_=lambda_,
            gamma=gamma,
            eps=eps,
            sample_weight=sample_weight,
            return_details=return_details,
            nan_policy=nan_policy,
        )

        if return_details:
            sc, det = res
            scores.append(sc)
            details.append(det)
        else:
            scores.append(res)

    if len(scores) == 1:
        return (scores[0], details[0]) if return_details else scores[0]

    if multioutput == "raw_values":
        out = np.asarray(scores, dtype=float)
    elif multioutput == "uniform_average":
        out = float(np.mean(scores))
    else:
        raise ValueError(
            "multioutput must be 'raw_values' or 'uniform_average'"
        )

    if return_details:
        return out, details
    return out


cluster_aware_severity_score.__doc__ = r"""
Compute the Cluster-Aware Severity (CAS) score.

This metric evaluates prediction intervals by penalizing not
only the magnitude of interval failures (anomalies) but also
their local concentration in time or space. CAS highlights
models that generate **runs** of misses, which are often more
operationally risky than isolated errors with similar size.

Formally, for observation :math:`y_t` and interval
:math:`[L_t, U_t]` at level :math:`1-\alpha`, define the signed
excess and magnitude
:math:`m_t=\max(L_t-y_t,0)+\max(y_t-U_t,0)`. With the band
width :math:`w_t=U_t-L_t` and small :math:`\varepsilon>0`,
the normalized excess is
:math:`\tilde m_t = m_t / (w_t+\varepsilon)`. Let
:math:`A_t=\mathbf{1}\{y_t<L_t \text{ or } y_t>U_t\}` and
:math:`d_t` be a centered kernel average of either indicators
(:math:`A_t`) or magnitudes (:math:`\tilde m_t`) over a window
of size ``window_size``. The pointwise severity is

.. math::

   S_t \;=\; \tilde m_t \Bigl(1 + \lambda\, d_t^{\gamma}\Bigr),

with :math:`\lambda\ge 0` and :math:`\gamma\ge 1`. The CAS
score is the average :math:`n^{-1}\sum_t S_t`. Lower values
indicate fewer and less clustered violations.

Parameters
----------
y_true : array-like of shape (n_samples,)
    or (n_samples, n_outputs)
    Ground-truth targets. For multioutput, the same
    prediction interval (from ``y_pred``) is applied to each
    output unless your wrapper expands bounds per output.

y_pred : array-like of shape (n_samples, 2)
    Predicted interval bounds. Column 0 is the lower bound
    :math:`L_t`; column 1 is the upper bound :math:`U_t`.

sample_weight : array-like of shape (n_samples,), default=None
    Optional weights for averaging the final severities.

window_size : int, default=21
    Half-width plus one for the centered smoothing window used
    to compute :math:`d_t`. Larger values capture longer runs.

sort_by : array-like of shape (n_samples,), optional
    Key used to order samples before computing :math:`d_t`.
    Typical choices are time, a spatial coordinate, or any
    ordering that makes clustering meaningful.

normalize : {'band', 'mad', 'none'}, default='band'
    Normalization for the excess :math:`m_t`.
    
    - 'band': divide by :math:`w_t=U_t-L_t` (unit-free).
    - 'mad': divide by a robust global scale (median absolute
      deviation).
    - 'none': no normalization (units of the series).

density_source : {'indicator', 'magnitude'}, default='indicator'
    Source for computing :math:`d_t`.
    
    - 'indicator': kernel average of :math:`A_t` (0/1 misses),
      matching the CAS definition in the paper.
    - 'magnitude': kernel average of normalized magnitude
      (more sensitive to large single misses).

kernel : {'box', 'triangular', 'epan', 'gaussian'}, default='box'
    Smoothing kernel for :math:`d_t`. 'box' emphasizes run
    length; smooth kernels emphasize local prevalence.

lambda_ : float, default=1.0
    Cluster penalty weight :math:`\lambda`. Larger values
    increase the contribution of :math:`d_t`.

gamma : float, default=1.0
    Density nonlinearity :math:`\gamma`. Values :math:`>1`
    accentuate dense clusters relative to sparse ones.

eps : float, default=1e-12
    Small positive number used in the band normalization
    denominator :math:`(w_t+\varepsilon)`.

multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
    Aggregation across outputs when ``y_true`` is 2D.
    
    - 'raw_values': return per-output scores.
    - 'uniform_average': return the average score.

nan_policy : {'omit', 'propagate', 'raise'}, default='omit'
    How to handle NaN/inf in any inputs (y_true, bounds,
    sort_by, sample_weight). After optional sorting, a mask is
    built over all required columns.
    
    - 'omit'       : drop invalid rows before computing CAS.
    - 'propagate'  : return NaN (and `None` for details).
    - 'raise'      : raise ValueError with a row count.

return_details : bool, default=False
    If True, also return a DataFrame with per-sample fields
    (``is_anomaly``, ``type``, ``magnitude``, ``local_density``,
    ``severity``). For multioutput, a list of DataFrames may
    be returned.

Returns
-------
score : float or ndarray of shape (n_outputs,)
    The CAS score. Smaller is better.

(score, details) : tuple
    Returned if ``return_details=True``. ``details`` contains
    the per-sample components used to compute CAS.

Notes
-----
CAS complements proper scoring rules and coverage by focusing
on **organization** of errors rather than only their average
frequency or size. It is translation-invariant and, with
``normalize='band'``, unit-free. Setting ``lambda_=0`` reduces
CAS to an average normalized excess outside the interval,
akin to the distance penalty in interval/Winkler scores. In
contrast, ``lambda_>0`` increases the score when violations
cluster, capturing **burstiness** that aggregate scores may
blur. The default density source ('indicator') follows the
definition in the paper and is recommended for diagnostics.

Time complexity for a box kernel with window ``W`` is
:math:`\mathcal{O}(nW)` and memory :math:`\mathcal{O}(n)`. With
FFT-based convolution for smooth kernels, the cost is typically
:math:`\mathcal{O}(n\log n)`.

Examples
--------
Basic usage
^^^^^^^^^^^
>>> import numpy as np
>>> y_true = np.array([10, 25, 30, 45, 50])
>>> y_pred = np.array([[8, 12], [24, 26], [32, 33],
...                    [44, 46], [48, 52]])
>>> cas = cluster_aware_severity_score(
...     y_true, y_pred, window_size=3
... )
>>> float(cas)  # doctest: +SKIP

Sorting to control clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>>> sort_key = np.array([0, 2, 4, 1, 3])
>>> cas_unsorted = cluster_aware_severity_score(
...     y_true, y_pred, window_size=3
... )
>>> cas_sorted = cluster_aware_severity_score(
...     y_true, y_pred, window_size=3, sort_by=sort_key
... )
>>> (float(cas_sorted), float(cas_unsorted))  # doctest: +SKIP

Adjusting density source and kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>>> cas_mag = cluster_aware_severity_score(
...     y_true, y_pred, window_size=5,
...     density_source="magnitude", kernel="triangular"
... )
>>> float(cas_mag)  # doctest: +SKIP

Weighting and stronger cluster penalty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>>> w = np.array([1, 1, 5, 1, 1])
>>> cas_w = cluster_aware_severity_score(
...     y_true, y_pred, sample_weight=w,
...     lambda_=2.0, gamma=2.0
... )
>>> float(cas_w)  # doctest: +SKIP

See Also
--------
clustered_anomaly_severity
    Helper that accepts arrays or DataFrame columns and returns
    the CAS score (and details if requested).
kdiagram.utils.plot.plot_cas_layers
    Layered, publication-ready line plot of intervals, severity
    stems, and anomalies.
kdiagram.utils.plot.plot_anomaly_glyphs
    Polar glyph visualization that emphasizes clustering.

References
----------
.. [1] Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007).
       Probabilistic forecasts, calibration and sharpness.
       *JRSS Series B*, 69(2), 243–268.
.. [2] Koenker, R., & Xiao, Z. (2006). Quantile autoregression.
       *JASA*, 101, 980–990.
.. [3] Podsztavek, O., Jordan, A. I., Tvrdík, P., & Polsterer,
       K. L. (2024). Automatic Miscalibration Diagnosis:
       Interpreting PIT Histograms. *ESANN*.
.. [4] Sokol, A. (2025). Fan charts 2.0: Flexible forecast
       distributions with expert judgement. *International
       Journal of Forecasting*, 41(3), 1148–1164.
"""


@overload
def clustered_anomaly_severity(
    y_true: str,
    y_qlow: str,
    y_qup: str,
    data: pd.DataFrame,
    window_size: int = 21,
    return_details: Literal[False] = False,
) -> float: ...


@overload
def clustered_anomaly_severity(
    y_true: str,
    y_qlow: str,
    y_qup: str,
    data: pd.DataFrame,
    return_details: Literal[True],
    window_size: int = 21,
) -> tuple[float, pd.DataFrame]: ...


@overload
def clustered_anomaly_severity(
    y_true: ArrayLike,
    y_qlow: ArrayLike,
    y_qup: ArrayLike,
    data: Literal[None] = None,
    window_size: int = 21,
    return_details: Literal[False] = False,
) -> float: ...


@overload
def clustered_anomaly_severity(
    y_true: ArrayLike,
    y_qlow: ArrayLike,
    y_qup: ArrayLike,
    return_details: Literal[True],
    data: Literal[None] = None,
    window_size: int = 21,
) -> tuple[float, pd.DataFrame]: ...


@validate_params(
    {
        "data": [pd.DataFrame, None],
        "window_size": [Integral],
        "return_details": [bool],
    }
)
def clustered_anomaly_severity(
    y_true: str | np.ndarray | pd.Series,
    y_qlow: str | np.ndarray | pd.Series,
    y_qup: str | np.ndarray | pd.Series,
    data: pd.DataFrame | None = None,
    window_size: int = 21,
    return_details: bool = False,
) -> float | tuple[float, pd.DataFrame]:
    if data is not None:
        # Mode 1: User provided a DataFrame and column names
        if not all(isinstance(col, str) for col in [y_true, y_qlow, y_qup]):
            raise TypeError(
                "If 'data' is provided, 'y_true', 'y_qlow',"
                " and 'y_qup' must be column name strings."
            )

        required_cols = [y_true, y_qlow, y_qup]
        exist_features(data, features=required_cols)

        df = data[required_cols].copy()
        df.columns = ["y_true", "y_qlow", "y_qup"]

    else:
        # Mode 2: User provided array-like objects
        if any(isinstance(arr, str) for arr in [y_true, y_qlow, y_qup]):
            raise TypeError(
                "If 'data' is None, inputs must be array-like, not strings."
            )

        # Ensure inputs are 1D numpy arrays
        y_true_arr = check_array(
            y_true,
            ensure_2d=False,
        )
        y_qlow_arr = check_array(
            y_qlow,
            ensure_2d=False,
        )
        y_qup_arr = check_array(
            y_qup,
            ensure_2d=False,
        )

        df = pd.DataFrame(
            {
                "y_true": y_true_arr,
                "y_qlow": y_qlow_arr,
                "y_qup": y_qup_arr,
            }
        )

    # ---- 2) Delegate to core scorer (paper-consistent) -------
    #      we keep sort_by=None and sample_weight=None here to
    #       preserve API. _cas_score already computes details.
    res = _cas_core(
        df["y_true"].to_numpy(),
        df["y_qlow"].to_numpy(),
        df["y_qup"].to_numpy(),
        window_size=window_size,
        sort_by=None,
        normalize="band",  # unit-free
        density_source="indicator",  # matches paper
        kernel="box",  # rolling box by default
        lambda_=1.0,
        gamma=1.0,
        eps=1e-12,
        sample_weight=None,
        return_details=return_details,
    )

    # ---- 3) Return in the same contract ----------------------
    if return_details:
        score, details = res
        return float(score), details

    return float(res)


clustered_anomaly_severity.__doc__ = r"""
Computes the Clustered Anomaly Severity (CAS) score.

This function serves as a direct helper for calculating the
CAS score. It is designed to penalize not just the magnitude
of forecast failures but also their concentration, providing
a more nuanced view of reliability than standard metrics.

Parameters
----------
y_true : str or array-like
    The ground truth (correct) target values. Can be a column
    name if `data` is provided, or a 1D array-like object.

y_qlow : str or array-like
    The lower bound of the prediction interval. Can be a column
    name if `data` is provided, or a 1D array-like object.

y_qup : str or array-like
    The upper bound of the prediction interval. Can be a column
    name if `data` is provided, or a 1D array-like object.

data : pd.DataFrame, optional
    Optional DataFrame containing the data for `y_true`,
    `y_qlow`, and `y_qup`. If provided, the aforementioned
    parameters must be strings representing column names.

window_size : int, default=21
    The size of the moving window used to calculate the local
    density of anomalies. A larger window considers a wider
    neighborhood for defining a "cluster".

return_details : bool, default=False
    If ``True``, the function returns a tuple containing the
    final CAS score and a DataFrame with detailed intermediate
    calculations (magnitude, density, severity, etc.).

Returns
-------
cas_score : float
    The calculated Clustered Anomaly Severity (CAS) score. A
    lower score indicates better performance (less severe and
    less clustered anomalies).

(cas_score, details_df) : tuple of (float, pd.DataFrame)
    Returned only if `return_details` is ``True``. The second
    element is a DataFrame containing per-sample calculations.

Notes
-----
The CAS score is composed of two main components, calculated
for each data point :math:`i`:

1.  **Anomaly Magnitude** (:math:`m_i`): The absolute
    distance from the true value to the nearest violated
    interval bound. It is zero if the point is covered.

    .. math::
       m_i =
       \begin{cases}
         y_{qlow,i} - y_{true,i} & \text{if } y_{true,i} < y_{qlow,i} \\
         y_{true,i} - y_{qup,i} & \text{if } y_{true,i} > y_{qup,i} \\
         0 & \text{otherwise}
       \end{cases}

2.  **Local Cluster Density** (:math:`d_i`): A measure of
    how concentrated anomalies are around point :math:`i`. It
    is calculated using a centered moving average of the
    magnitudes within a `window_size`.

The final severity for each point is :math:`s_i = m_i \cdot d_i`,
and the overall CAS score is the mean of these severities.

See Also
--------
cluster_aware_severity_score :
    A Scikit-learn compliant version of this metric with
    extended functionality (e.g., `sample_weight`, `sort_by`).
kdiagram.plot.uncertainty.plot_anomaly_severity :
    The specialized polar plot to visualize this metric.

References
----------
.. footbibliography::

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> # Example 1: Using NumPy arrays
>>> y_true = np.array([10, 25, 30, 45, 50])
>>> y_qlow = np.array([8, 24, 32, 44, 48])
>>> y_qup = np.array([12, 26, 33, 46, 52])
>>> cas = clustered_anomaly_severity(
...     y_true, y_qlow, y_qup, window_size=3
... )
>>> print(f"CAS Score (from arrays): {cas:.4f}")
CAS Score (from arrays): 0.2222

>>> # Example 2: Using a DataFrame and column names
>>> df = pd.DataFrame({
...     'actual': y_true, 'lower_bound': y_qlow, 'upper_bound': y_qup
... })
>>> cas_df, details = clustered_anomaly_severity(
...     'actual', 'lower_bound', 'upper_bound',
...     data=df, window_size=3, return_details=True
... )
>>> print(f"CAS Score (from DataFrame): {cas_df:.4f}")
CAS Score (from DataFrame): 0.2222
>>> print(details.head())
   y_true  y_qlow  y_qup  magnitude  is_anomaly   type  local_density  severity
0      10       8     12          0       False   none       0.000000  0.000000
1      25      24     26          0       False   none       0.666667  0.000000
2      30      32     33          2        True  under       0.666667  1.333333
3      45      44     46          0       False   none       0.666667  0.000000
4      50      48     52          0       False   none       0.000000  0.000000

"""


def _rolling_kernel(a: np.ndarray, w: int, kernel: str) -> np.ndarray:
    w = int(max(1, w))
    if kernel == "box":
        k = np.ones(w) / w
    elif kernel == "triangular":
        mid = (w - 1) / 2
        x = np.arange(w) - mid
        k = (1 - np.abs(x) / (mid + 1e-12)).clip(0)
        k = k / k.sum()
    elif kernel == "epan":
        mid = (w - 1) / 2
        x = (np.arange(w) - mid) / (mid + 1e-12)
        k = (0.75 * (1 - x**2)).clip(0)
        k = k / k.sum()
    elif kernel == "gaussian":
        sig = max(1.0, w / 4.0)
        x = np.arange(w) - (w - 1) / 2
        k = np.exp(-(x**2) / (2 * sig**2))
        k = k / k.sum()
    else:
        raise ValueError("bad kernel")
    pad = w // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    conv = np.convolve(ap, k, mode="valid")
    return conv


def _cas_core(
    y: ArrayLike,
    lo: ArrayLike,
    up: ArrayLike,
    *,
    window_size: int = 21,
    sort_by: ArrayLike | None = None,
    normalize: str = "band",
    density_source: str = "indicator",
    kernel: str = "box",
    lambda_: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-12,
    sample_weight: ArrayLike | None = None,
    return_details: bool = False,
    nan_policy: str = "omit",
):
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    up = np.asarray(up, float)

    if sort_by is not None:
        sb = np.asarray(sort_by)
        idx = np.argsort(sb)
        y, lo, up = y[idx], lo[idx], up[idx]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[idx]
        # after sorting, rebind sb
        sort_by = sb[idx]

    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(up)
    if sort_by is not None:
        mask &= np.isfinite(sort_by)
    if sample_weight is not None:
        mask &= np.isfinite(sample_weight)

    if not mask.all():
        if nan_policy == "raise":
            bad = (~mask).sum()
            raise ValueError(f"CAS: found {bad} NaN/inf rows; policy=raise.")
        if nan_policy == "propagate":
            return (np.nan, None) if return_details else np.nan
        # "omit"
        y, lo, up = y[mask], lo[mask], up[mask]
        if sort_by is not None:
            sort_by = sort_by[mask]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[mask]

    if y.size == 0:
        return (np.nan, pd.DataFrame()) if return_details else np.nan

    is_under = y < lo
    is_over = y > up
    A = np.where(is_under | is_over, 1.0, 0.0)

    m = np.where(is_under, lo - y, 0.0) + np.where(is_over, y - up, 0.0)

    if normalize == "band":
        w = (up - lo) + eps
        m = m / w
    elif normalize == "mad":
        med = np.median(y)
        mad = np.median(np.abs(y - med)) + eps
        m = m / mad

    src = A if density_source == "indicator" else m

    d = _rolling_kernel(src, window_size, kernel)
    d = np.clip(d, 0.0, 1.0)

    S = m * (1.0 + lambda_ * (d**gamma))

    if sample_weight is not None:
        sw = np.asarray(sample_weight, float)
        score = np.average(S, weights=sw)
    else:
        score = float(np.mean(S))

    if not return_details:
        return score

    typ = np.where(is_under, "under", np.where(is_over, "over", "none"))
    det = pd.DataFrame(
        {
            "y_true": y,
            "y_qlow": lo,
            "y_qup": up,
            "is_anomaly": A.astype(bool),
            "type": typ,
            "magnitude": m,
            "local_density": d,
            "severity": S,
        }
    )
    return score, det


def _split_bounds(y_pred):
    # supports (n,2), (n,n_out,2), or (n, 2*n_out)
    yp = np.asarray(y_pred)
    if yp.ndim == 2 and yp.shape[1] == 2:
        return [(yp[:, 0], yp[:, 1])]
    if yp.ndim == 3 and yp.shape[2] == 2:
        return [(yp[:, i, 0], yp[:, i, 1]) for i in range(yp.shape[1])]
    if yp.ndim == 2 and yp.shape[1] % 2 == 0:
        n_out = yp.shape[1] // 2
        return [(yp[:, 2 * i], yp[:, 2 * i + 1]) for i in range(n_out)]
    raise ValueError("y_pred must be (n,2), (n,n_out,2), or (n,2*n_out)")
