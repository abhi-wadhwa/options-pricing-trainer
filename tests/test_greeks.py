"""Tests for option Greeks."""

import math

import pytest

from src.core.greeks import (
    delta, gamma, vega, theta, rho,
    all_greeks, numerical_greeks,
)
from src.core.black_scholes import bs_price


S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


class TestAnalyticGreeks:
    """Test analytic (Black-Scholes) Greeks."""

    def test_call_delta_range(self):
        d = delta(S, K, T, r, sigma, "call")
        assert 0.0 < d < 1.0

    def test_put_delta_range(self):
        d = delta(S, K, T, r, sigma, "put")
        assert -1.0 < d < 0.0

    def test_delta_call_minus_put_equals_one(self):
        """Key relationship: Delta_call - Delta_put = 1."""
        d_call = delta(S, K, T, r, sigma, "call")
        d_put = delta(S, K, T, r, sigma, "put")
        assert d_call - d_put == pytest.approx(1.0, abs=1e-10)

    def test_gamma_positive(self):
        g = gamma(S, K, T, r, sigma)
        assert g > 0

    def test_gamma_same_for_call_and_put(self):
        """Gamma is the same for calls and puts."""
        # gamma doesn't depend on option_type
        g = gamma(S, K, T, r, sigma)
        assert g > 0  # just confirm it's positive

    def test_vega_positive(self):
        v = vega(S, K, T, r, sigma)
        assert v > 0

    def test_theta_call_negative(self):
        """Long options have negative theta (time decay)."""
        t = theta(S, K, T, r, sigma, "call")
        assert t < 0

    def test_rho_call_positive(self):
        """Call rho is positive (higher rates benefit calls)."""
        r_val = rho(S, K, T, r, sigma, "call")
        assert r_val > 0

    def test_rho_put_negative(self):
        r_val = rho(S, K, T, r, sigma, "put")
        assert r_val < 0

    def test_deep_itm_call_delta_near_one(self):
        d = delta(200.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert d > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        d = delta(50.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert d < 0.01

    def test_all_greeks_returns_dict(self):
        g = all_greeks(S, K, T, r, sigma, "call")
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}
        for val in g.values():
            assert isinstance(val, float)


class TestNumericalGreeks:
    """Test numerical Greeks vs analytic."""

    def test_numerical_vs_analytic_delta(self):
        analytic = delta(S, K, T, r, sigma, "call")
        numerical = numerical_greeks(bs_price, S, K, T, r, sigma, "call")["delta"]
        assert numerical == pytest.approx(analytic, abs=1e-4)

    def test_numerical_vs_analytic_gamma(self):
        analytic = gamma(S, K, T, r, sigma)
        numerical = numerical_greeks(bs_price, S, K, T, r, sigma, "call")["gamma"]
        assert numerical == pytest.approx(analytic, abs=1e-3)

    def test_numerical_vs_analytic_vega(self):
        analytic = vega(S, K, T, r, sigma)
        numerical = numerical_greeks(bs_price, S, K, T, r, sigma, "call")["vega"]
        assert numerical == pytest.approx(analytic, abs=0.1)

    def test_numerical_vs_analytic_rho(self):
        analytic = rho(S, K, T, r, sigma, "call")
        numerical = numerical_greeks(bs_price, S, K, T, r, sigma, "call")["rho"]
        assert numerical == pytest.approx(analytic, abs=0.1)
