"""Tests for Black-Scholes pricing.

Reference values computed independently using known BS formulae.
"""

import math

import pytest

from src.core.black_scholes import bs_call, bs_put, bs_price


# Reference case: S=100, K=100, T=1, r=0.05, sigma=0.2
# Known BS call price: 10.45058358 (verified against multiple sources)
# Known BS put price:  5.57352502

class TestBlackScholesKnownValues:
    """Test BS prices against independently verified values."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_call_atm(self):
        price = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        assert price == pytest.approx(10.45058358, abs=1e-6)

    def test_put_atm(self):
        price = bs_put(self.S, self.K, self.T, self.r, self.sigma)
        assert price == pytest.approx(5.573526022256971, abs=1e-6)

    def test_itm_call(self):
        # Deep ITM call: S=120, K=100
        price = bs_call(120.0, 100.0, 1.0, 0.05, 0.2)
        # Intrinsic ~ 20, should be above that
        assert price > 20.0
        assert price == pytest.approx(26.169043946847296, abs=1e-5)

    def test_otm_put(self):
        # Deep OTM put: S=120, K=100
        price = bs_put(120.0, 100.0, 1.0, 0.05, 0.2)
        assert price < 5.0
        assert price > 0.0

    def test_short_expiry(self):
        # Near-expiry ATM option should be small but positive
        price = bs_call(100.0, 100.0, 0.01, 0.05, 0.2)
        assert 0.0 < price < 5.0

    def test_high_vol(self):
        # High vol increases option value
        low_vol_price = bs_call(100.0, 100.0, 1.0, 0.05, 0.1)
        high_vol_price = bs_call(100.0, 100.0, 1.0, 0.05, 0.5)
        assert high_vol_price > low_vol_price

    def test_bs_price_call(self):
        assert bs_price(self.S, self.K, self.T, self.r, self.sigma, "call") == \
               bs_call(self.S, self.K, self.T, self.r, self.sigma)

    def test_bs_price_put(self):
        assert bs_price(self.S, self.K, self.T, self.r, self.sigma, "put") == \
               bs_put(self.S, self.K, self.T, self.r, self.sigma)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            bs_price(self.S, self.K, self.T, self.r, self.sigma, "straddle")


class TestBlackScholesEdgeCases:
    """Edge cases and boundary behaviour."""

    def test_deep_itm_call_approaches_intrinsic(self):
        # Very short expiry, deep ITM
        price = bs_call(200.0, 100.0, 0.001, 0.05, 0.2)
        intrinsic = 200.0 - 100.0 * math.exp(-0.05 * 0.001)
        assert price == pytest.approx(intrinsic, abs=0.5)

    def test_deep_otm_call_near_zero(self):
        price = bs_call(50.0, 200.0, 0.1, 0.05, 0.2)
        assert price < 0.01

    def test_zero_rate(self):
        # Should still price correctly
        price = bs_call(100.0, 100.0, 1.0, 0.0, 0.2)
        assert price > 0.0

    def test_symmetry(self):
        """At-the-money forward, call and put should be equal."""
        F = 100.0 * math.exp(0.05 * 1.0)  # forward
        call_p = bs_call(100.0, F, 1.0, 0.05, 0.2)
        put_p = bs_put(100.0, F, 1.0, 0.05, 0.2)
        assert call_p == pytest.approx(put_p, abs=1e-8)
