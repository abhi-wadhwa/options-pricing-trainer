"""Tests for put-call parity and cross-model consistency.

Put-call parity (European, no dividends):
    C - P = S - K * e^{-rT}

This must hold for any pricing model that correctly prices European options.
"""

import math

import pytest

from src.core.black_scholes import bs_call, bs_put
from src.core.binomial import binomial_price
from src.core.monte_carlo import monte_carlo_price


class TestPutCallParityBS:
    """Put-call parity using Black-Scholes prices."""

    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (100, 100, 1.0, 0.05, 0.2),
        (100, 90, 0.5, 0.03, 0.3),
        (100, 110, 2.0, 0.08, 0.15),
        (50, 60, 0.25, 0.02, 0.4),
        (200, 180, 1.5, 0.06, 0.25),
    ])
    def test_put_call_parity(self, S, K, T, r, sigma):
        C = bs_call(S, K, T, r, sigma)
        P = bs_put(S, K, T, r, sigma)
        parity = S - K * math.exp(-r * T)
        assert C - P == pytest.approx(parity, abs=1e-10)


class TestPutCallParityBinomial:
    """Put-call parity using binomial tree prices (European)."""

    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (100, 100, 1.0, 0.05, 0.2),
        (100, 90, 0.5, 0.03, 0.3),
    ])
    def test_put_call_parity_binomial(self, S, K, T, r, sigma):
        C = binomial_price(S, K, T, r, sigma, N=500, option_type="call", american=False)
        P = binomial_price(S, K, T, r, sigma, N=500, option_type="put", american=False)
        parity = S - K * math.exp(-r * T)
        assert C - P == pytest.approx(parity, abs=0.1)


class TestCrossModelConsistency:
    """Different models should give similar prices for European options."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_bs_vs_binomial_call(self):
        bs = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        binom = binomial_price(self.S, self.K, self.T, self.r, self.sigma,
                               N=500, option_type="call")
        assert binom == pytest.approx(bs, abs=0.05)

    def test_bs_vs_binomial_put(self):
        bs = bs_put(self.S, self.K, self.T, self.r, self.sigma)
        binom = binomial_price(self.S, self.K, self.T, self.r, self.sigma,
                               N=500, option_type="put")
        assert binom == pytest.approx(bs, abs=0.05)

    def test_bs_vs_monte_carlo_call(self):
        bs = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        mc = monte_carlo_price(
            self.S, self.K, self.T, self.r, self.sigma,
            option_type="call", n_paths=200_000, seed=42,
        )
        # Monte Carlo should be within 3 standard errors
        assert mc["price"] == pytest.approx(bs, abs=3 * mc["std_error"] + 0.1)

    def test_bs_vs_monte_carlo_put(self):
        bs = bs_put(self.S, self.K, self.T, self.r, self.sigma)
        mc = monte_carlo_price(
            self.S, self.K, self.T, self.r, self.sigma,
            option_type="put", n_paths=200_000, seed=42,
        )
        assert mc["price"] == pytest.approx(bs, abs=3 * mc["std_error"] + 0.1)
