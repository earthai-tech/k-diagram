# Tests for CAS implementation correctness against paper formulas.
#
# Every numeric assertion is derived from first-principles (paper equations),
# not from code output.  The test class is deliberately monolithic so that
# a single run gives a complete picture of CAS health.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kdiagram.metrics import (
    _cas_core,
    _rolling_kernel,
    cluster_aware_severity_score,
    clustered_anomaly_severity,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 21          # odd → clean centre index at 10
LO = np.zeros(N)
UP = np.ones(N)  # width = 1 everywhere
BASE = np.full(N, 0.5, dtype=float)  # inside the interval everywhere


def _bounds(lo, up):
    return np.column_stack([lo, up])


# ---------------------------------------------------------------------------
# Paper property 1 – Non-negativity (eq. CAS_nonnegative)
# ---------------------------------------------------------------------------

class TestNonNegativity:
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_random_data_nonneg(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(50)
        lo = y - rng.uniform(0.2, 1.0, 50)
        up = y + rng.uniform(0.2, 1.0, 50)
        y_noisy = y + rng.standard_normal(50) * 0.8
        s = cluster_aware_severity_score(y_noisy, _bounds(lo, up))
        assert s >= 0.0

    @pytest.mark.parametrize("kernel", ["box", "triangular", "epan", "gaussian"])
    def test_all_kernels_nonneg(self, kernel):
        y = BASE.copy()
        y[5] = 1.8
        s = cluster_aware_severity_score(
            y, _bounds(LO, UP), kernel=kernel, lambda_=2.0, gamma=0.5
        )
        assert s >= 0.0


# ---------------------------------------------------------------------------
# Paper property 2 – CAS = 0 when no violations
# ---------------------------------------------------------------------------

class TestZeroNoViolations:
    def test_all_inside(self):
        assert cluster_aware_severity_score(BASE, _bounds(LO, UP)) == 0.0

    def test_on_boundary_not_violation(self):
        y = BASE.copy()
        y[3] = 0.0   # exactly on lower bound → not a violation (strict <)
        y[7] = 1.0   # exactly on upper bound → not a violation (strict >)
        assert cluster_aware_severity_score(y, _bounds(LO, UP)) == 0.0

    def test_clustered_anomaly_severity_zero(self):
        assert clustered_anomaly_severity(BASE, LO, UP) == 0.0


# ---------------------------------------------------------------------------
# Paper property 3 – Self-exclusion: isolated violation → d = 0 → CAS = e/n
# Paper eq. (6): sum over r≠t; isolated ⟹ δ_t = 0 ⟹ c_t = e_t
# Paper eq. (CAS_A): CAS_A = k·e / n  (k=1 violation, e=exceedance)
# ---------------------------------------------------------------------------

class TestSelfExclusion:
    def test_density_zero_for_isolated_violation(self):
        # Violation at t=10, all neighbours inside → d[10] must be 0.
        src = np.zeros(N)
        src[10] = 1.0
        d = _rolling_kernel(src, N, "triangular", exclude_self=True)
        assert d[10] == pytest.approx(0.0, abs=1e-12)

    def test_density_zero_box_kernel(self):
        src = np.zeros(N)
        src[10] = 1.0
        d = _rolling_kernel(src, N, "box", exclude_self=True)
        assert d[10] == pytest.approx(0.0, abs=1e-12)

    def test_cas_equals_exceedance_over_n_for_isolated(self):
        # Over-violation at t=10 only: y=1.5, lo=0, up=1, width=1, e=0.5
        y = BASE.copy()
        y[10] = 1.5
        expected = 0.5 / N  # c_10 = v_10 * e_10 * [1 + λ·0^γ] = e_10
        s = cluster_aware_severity_score(y, _bounds(LO, UP))
        assert s == pytest.approx(expected, abs=1e-12)

    def test_under_violation_exact(self):
        # Under-violation at t=10: y=−0.3, lo=0, up=1, a=(0−(−0.3))=0.3, e=0.3
        y = BASE.copy()
        y[10] = -0.3
        expected = 0.3 / N
        s = cluster_aware_severity_score(y, _bounds(LO, UP))
        assert s == pytest.approx(expected, abs=1e-12)

    def test_two_isolated_violations_additive(self):
        # Two violations far apart in a large array → CAS = (e1+e2)/n
        n = 61
        lo, up = np.zeros(n), np.ones(n)
        y = np.full(n, 0.5)
        y[5] = 1.7    # over by 0.7, e=0.7
        y[55] = 1.4   # over by 0.4, e=0.4  (distance=50 >> window=21)
        expected = (0.7 + 0.4) / n
        s = cluster_aware_severity_score(y, _bounds(lo, up))
        assert s == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Paper property 4 – Cluster sensitivity / rearrangement theorem
# Same violations; clustered ⟹ CAS_B > CAS_A (eq. CAS_cluster_sensitive)
# ---------------------------------------------------------------------------

class TestRearrangementTheorem:
    @pytest.mark.parametrize("kernel", ["box", "triangular", "epan", "gaussian"])
    def test_clustered_beats_isolated_all_kernels(self, kernel):
        n = 50
        lo, up = np.zeros(n), np.ones(n)

        # Isolated: 3 violations spaced >> window
        y_iso = np.full(n, 0.5)
        y_iso[2] = 1.5
        y_iso[25] = 1.5
        y_iso[48] = 1.5

        # Clustered: same 3 violations adjacent
        y_cl = np.full(n, 0.5)
        y_cl[24] = 1.5
        y_cl[25] = 1.5
        y_cl[26] = 1.5

        kw = dict(kernel=kernel, window_size=7, lambda_=1.0, gamma=1.0)
        cas_iso = cluster_aware_severity_score(y_iso, _bounds(lo, up), **kw)
        cas_cl = cluster_aware_severity_score(y_cl, _bounds(lo, up), **kw)
        assert cas_cl > cas_iso, (
            f"kernel={kernel}: expected CAS_clustered > CAS_isolated, "
            f"got {cas_cl:.6f} vs {cas_iso:.6f}"
        )

    def test_same_crps_proxy_different_cas(self):
        # Verify that two series with same number of violations and same total
        # exceedance differ in CAS when arrangement differs.
        n = 40
        lo, up = np.zeros(n), np.ones(n)
        y_iso = np.full(n, 0.5)
        y_iso[[3, 20, 37]] = 1.5

        y_cl = np.full(n, 0.5)
        y_cl[[19, 20, 21]] = 1.5

        # Same total exceedance (both: 3 × 0.5)
        assert np.isclose(
            np.sum(np.maximum(y_iso - up, 0) + np.maximum(lo - y_iso, 0)),
            np.sum(np.maximum(y_cl - up, 0) + np.maximum(lo - y_cl, 0)),
        )

        cas_iso = cluster_aware_severity_score(y_iso, _bounds(lo, up), window_size=5)
        cas_cl = cluster_aware_severity_score(y_cl, _bounds(lo, up), window_size=5)
        assert cas_cl > cas_iso


# ---------------------------------------------------------------------------
# Paper property 5 – Scale invariance (eq. scale_invariance)
# Affine transform y* = a·y+b, L* = a·L+b, U* = a·U+b  → CAS unchanged
# ---------------------------------------------------------------------------

class TestScaleInvariance:
    @pytest.mark.parametrize("a,b", [(3.0, 5.0), (0.1, -2.0), (100.0, 0.0)])
    def test_affine_invariance(self, a, b):
        y = BASE.copy()
        y[8] = 1.6
        y[9] = 1.7
        s1 = cluster_aware_severity_score(y, _bounds(LO, UP))
        s2 = cluster_aware_severity_score(
            a * y + b, _bounds(a * LO + b, a * UP + b)
        )
        assert s1 == pytest.approx(s2, abs=1e-9)


# ---------------------------------------------------------------------------
# Paper property 6 – λ=0 reduces CAS to average relative exceedance
# (eq. cas_expanded, first component only)
# ---------------------------------------------------------------------------

class TestLambdaZero:
    def test_lambda_zero_equals_mean_exceedance(self):
        y = BASE.copy()
        y[8:11] = 1.5   # 3 over-violations, e=0.5 each
        lo, up = LO, UP

        # With λ=0 the cluster term vanishes → CAS = (1/n)Σ v_t·e_t
        cas_l0 = cluster_aware_severity_score(y, _bounds(lo, up), lambda_=0.0)

        exc = np.sum(np.maximum(y - up, 0) + np.maximum(lo - y, 0))
        width = up - lo
        expected = float(np.mean(
            np.where((y < lo) | (y > up),
                     (np.maximum(y - up, 0) + np.maximum(lo - y, 0)) / (width + 1e-12),
                     0.0)
        ))
        assert cas_l0 == pytest.approx(expected, abs=1e-12)

    def test_lambda_positive_increases_score_when_clustered(self):
        y = BASE.copy()
        y[9:12] = 1.5
        cas_l0 = cluster_aware_severity_score(y, _bounds(LO, UP), lambda_=0.0)
        cas_l1 = cluster_aware_severity_score(y, _bounds(LO, UP), lambda_=1.0)
        assert cas_l1 > cas_l0


# ---------------------------------------------------------------------------
# Paper formula check – pointwise components via return_details
# (eqs. violation_indicator, relative_exceedance, local_density, pointwise_cas)
# ---------------------------------------------------------------------------

class TestPointwiseComponents:
    def _get_details(self, y, lo, up, **kw):
        _, det = cluster_aware_severity_score(
            y, _bounds(lo, up), return_details=True, **kw
        )
        return det

    def test_over_violation_type_and_magnitude(self):
        # y[2]=1.8, lo=0, up=1; a=(1.8-1)=0.8, w=1, e=0.8
        y = np.array([0.5, 0.5, 1.8, 0.5, 0.5], dtype=float)
        lo = np.zeros(5)
        up = np.ones(5)
        det = self._get_details(y, lo, up, window_size=3)
        assert det.loc[2, "is_anomaly"] is True or det.loc[2, "is_anomaly"] == True
        assert det.loc[2, "type"] == "over"
        assert det.loc[2, "magnitude"] == pytest.approx(0.8, abs=1e-12)

    def test_under_violation_type_and_magnitude(self):
        # y[2]=-0.2, lo=0, up=1; a=(0-(-0.2))=0.2, w=1, e=0.2
        y = np.array([0.5, 0.5, -0.2, 0.5, 0.5], dtype=float)
        lo = np.zeros(5)
        up = np.ones(5)
        det = self._get_details(y, lo, up, window_size=3)
        assert det.loc[2, "type"] == "under"
        assert det.loc[2, "magnitude"] == pytest.approx(0.2, abs=1e-12)

    def test_no_violation_type_none_and_zero_magnitude(self):
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        lo = np.zeros(5)
        up = np.ones(5)
        det = self._get_details(y, lo, up)
        assert (det["type"] == "none").all()
        assert (det["magnitude"] == 0.0).all()
        assert (det["severity"] == 0.0).all()

    def test_isolated_violation_density_zero(self):
        y = BASE.copy()
        y[10] = 1.5   # isolated: all neighbours inside interval
        det = self._get_details(y, LO, UP)
        assert det.loc[10, "local_density"] == pytest.approx(0.0, abs=1e-12)

    def test_clustered_violation_density_positive(self):
        y = BASE.copy()
        y[9] = 1.5
        y[10] = 1.5
        y[11] = 1.5
        det = self._get_details(y, LO, UP, window_size=21)
        # Centre of cluster must see its neighbours
        assert det.loc[10, "local_density"] > 0.0

    def test_severity_formula(self):
        # S_t = m_t × (1 + λ·d_t^γ)
        y = BASE.copy()
        y[9] = 1.5
        y[10] = 1.5
        det = self._get_details(y, LO, UP, lambda_=1.0, gamma=1.0, window_size=21)
        for i in [9, 10]:
            m = det.loc[i, "magnitude"]
            d = det.loc[i, "local_density"]
            expected_sev = m * (1.0 + 1.0 * d**1.0)
            assert det.loc[i, "severity"] == pytest.approx(expected_sev, abs=1e-12)

    def test_score_equals_mean_severity(self):
        y = BASE.copy()
        y[8:12] = 1.5
        score, det = cluster_aware_severity_score(
            y, _bounds(LO, UP), return_details=True
        )
        assert score == pytest.approx(det["severity"].mean(), abs=1e-12)


# ---------------------------------------------------------------------------
# Relative exceedance: wider interval → smaller e for same absolute overshoot
# ---------------------------------------------------------------------------

class TestRelativeExceedance:
    def test_wide_interval_lower_cas(self):
        y = np.array([2.0])
        # Narrow: lo=0, up=1; over by 1, width=1, e=1
        s_narrow = cluster_aware_severity_score(y, _bounds(np.array([0.0]), np.array([1.0])))
        # Wide: lo=0, up=3; inside – no violation
        s_wide = cluster_aware_severity_score(y, _bounds(np.array([0.0]), np.array([3.0])))
        assert s_narrow > s_wide

    def test_normalisation_band_vs_mad(self):
        y = BASE.copy()
        y[10] = 1.5
        s_band = cluster_aware_severity_score(y, _bounds(LO, UP), normalize="band")
        s_mad = cluster_aware_severity_score(y, _bounds(LO, UP), normalize="mad")
        # Both must be non-negative; they may differ in value
        assert s_band >= 0.0
        assert s_mad >= 0.0


# ---------------------------------------------------------------------------
# Triangular kernel: default per paper §4.4 (K(u)=(1-u)_+)
# ---------------------------------------------------------------------------

class TestTriangularDefault:
    def test_default_kernel_matches_explicit_triangular(self):
        y = BASE.copy()
        y[9] = 1.5
        y[10] = 1.5
        s_default = cluster_aware_severity_score(y, _bounds(LO, UP))
        s_tri = cluster_aware_severity_score(y, _bounds(LO, UP), kernel="triangular")
        assert s_default == pytest.approx(s_tri, abs=1e-15)

    def test_clustered_anomaly_severity_default_is_triangular(self):
        y = BASE.copy()
        y[9] = 1.5
        y[10] = 1.5
        s_helper = clustered_anomaly_severity(y, LO, UP)
        s_api = cluster_aware_severity_score(y, _bounds(LO, UP), kernel="triangular")
        assert s_helper == pytest.approx(s_api, abs=1e-12)


# ---------------------------------------------------------------------------
# API: both public functions, all calling conventions
# ---------------------------------------------------------------------------

class TestAPIConventions:
    def test_clustered_anomaly_severity_array_mode(self):
        y = BASE.copy()
        y[10] = 1.5
        s = clustered_anomaly_severity(y, LO, UP)
        assert isinstance(s, float) and s > 0.0

    def test_clustered_anomaly_severity_dataframe_mode(self):
        y = BASE.copy()
        y[10] = 1.5
        df = pd.DataFrame({"y": y, "lo": LO, "up": UP})
        s = clustered_anomaly_severity("y", "lo", "up", data=df)
        s_arr = clustered_anomaly_severity(y, LO, UP)
        assert s == pytest.approx(s_arr, abs=1e-12)

    def test_dataframe_return_details_columns(self):
        df = pd.DataFrame({"y": BASE, "lo": LO, "up": UP})
        s, det = clustered_anomaly_severity("y", "lo", "up", data=df, return_details=True)
        required = {"y_true", "y_qlow", "y_qup", "is_anomaly", "type",
                    "magnitude", "local_density", "severity"}
        assert required.issubset(set(det.columns))
        assert len(det) == N

    def test_cas_score_n2_bounds(self):
        y = BASE.copy()
        y[10] = 1.5
        s = cluster_aware_severity_score(y, _bounds(LO, UP))
        assert s == pytest.approx(0.5 / N, abs=1e-12)

    def test_cas_score_3d_multioutput(self):
        # (n, 2, 2): two outputs, each with their own (L, U)
        n = 20
        y = np.c_[np.full(n, 0.5), np.full(n, 0.5)]
        y[10, 0] = 1.5
        y[10, 1] = 1.5
        lo = np.zeros((n, 2))
        up = np.ones((n, 2))
        y_pred = np.stack([_bounds(lo[:, 0], up[:, 0]),
                           _bounds(lo[:, 1], up[:, 1])], axis=1)
        s = cluster_aware_severity_score(y, y_pred)
        assert isinstance(s, float) and s > 0.0

    def test_cas_score_wide_matrix_bounds(self):
        # (n, 4) → two (L,U) pairs
        n = 20
        y = np.full(n, 0.5)
        y[10] = 1.5
        y_pred = np.c_[np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]
        s = cluster_aware_severity_score(y, y_pred)
        assert isinstance(s, float)

    def test_multioutput_raw_values(self):
        n = 20
        y = np.c_[np.full(n, 0.5), np.full(n, 0.5)]
        y_pred = np.c_[np.zeros(n), np.ones(n), np.zeros(n) - 1, np.ones(n) + 1]
        s = cluster_aware_severity_score(y, y_pred, multioutput="raw_values")
        assert s.shape == (2,)
        assert (s >= 0).all()


# ---------------------------------------------------------------------------
# NaN handling (nan_policy)
# ---------------------------------------------------------------------------

class TestNaNPolicy:
    def _data_with_nan(self):
        y = BASE.copy()
        y[10] = 1.5
        y[5] = np.nan
        return y

    def test_omit_ignores_nan(self):
        y = self._data_with_nan()
        s = cluster_aware_severity_score(
            y, _bounds(LO, UP), nan_policy="omit"
        )
        assert np.isfinite(s) and s > 0.0

    def test_propagate_returns_nan(self):
        y = self._data_with_nan()
        s = cluster_aware_severity_score(
            y, _bounds(LO, UP), nan_policy="propagate"
        )
        assert np.isnan(s)

    def test_raise_raises_on_nan(self):
        y = self._data_with_nan()
        with pytest.raises(ValueError, match="NaN"):
            cluster_aware_severity_score(
                y, _bounds(LO, UP), nan_policy="raise"
            )


# ---------------------------------------------------------------------------
# sample_weight and sort_by
# ---------------------------------------------------------------------------

class TestWeightAndSort:
    def test_high_weight_on_violation_inflates_score(self):
        y = BASE.copy()
        y[2] = 1.5
        y[10] = 1.5
        lo, up = LO, UP
        w_uniform = np.ones(N)
        w_heavy = np.ones(N)
        w_heavy[2] = 1000.0   # heavily weight the first violation
        s_u = cluster_aware_severity_score(y, _bounds(lo, up), sample_weight=w_uniform)
        s_h = cluster_aware_severity_score(y, _bounds(lo, up), sample_weight=w_heavy)
        assert s_h > s_u

    def test_sort_by_reorders_before_density(self):
        # Three violations adjacent in original order → positive density.
        # After sort_by that spreads them out → lower density → lower CAS.
        n = 20
        lo, up = np.zeros(n), np.ones(n)
        y = np.full(n, 0.5)
        y[9] = 1.5
        y[10] = 1.5
        y[11] = 1.5

        cas_natural = cluster_aware_severity_score(
            y, _bounds(lo, up), window_size=5, kernel="box"
        )

        # Reorder so violations end up at positions 0, 10, 19 (well separated)
        sort_key = np.arange(n)
        sort_key[9] = 0
        sort_key[10] = 10
        sort_key[11] = 19

        cas_spread = cluster_aware_severity_score(
            y, _bounds(lo, up), sort_by=sort_key, window_size=5, kernel="box"
        )
        assert cas_natural > cas_spread


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_bad_kernel_raises(self):
        with pytest.raises(ValueError):
            cluster_aware_severity_score(BASE, _bounds(LO, UP), kernel="sinc")

    def test_bad_multioutput_raises(self):
        with pytest.raises(ValueError):
            cluster_aware_severity_score(
                BASE, _bounds(LO, UP), multioutput="invalid"
            )

    def test_wrong_y_pred_shape_raises(self):
        with pytest.raises(ValueError):
            cluster_aware_severity_score(BASE, BASE)   # 1-D y_pred

    def test_dataframe_mode_type_error(self):
        with pytest.raises(TypeError):
            clustered_anomaly_severity(BASE, "lo", "up", data=None)

    def test_dataframe_mode_non_string_col_raises(self):
        df = pd.DataFrame({"y": BASE, "lo": LO, "up": UP})
        with pytest.raises(TypeError):
            clustered_anomaly_severity(BASE, LO, UP, data=df)
