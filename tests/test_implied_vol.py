"""Tests for implied volatility computation."""

import pytest

from src.core.implied_vol import implied_volatility
from src.core.black_scholes import bs_price


class TestImpliedVolatility:
    """Round-trip tests: compute BS price, then recover sigma."""

    S, K, T, r = 100.0, 100.0, 1.0, 0.05

    @pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_brent_roundtrip_call(self, true_sigma):
        market_price = bs_price(self.S, self.K, self.T, self.r, true_sigma, "call")
        recovered = implied_volatility(
            market_price, self.S, self.K, self.T, self.r,
            option_type="call", method="brent",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-8)

    @pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_brent_roundtrip_put(self, true_sigma):
        market_price = bs_price(self.S, self.K, self.T, self.r, true_sigma, "put")
        recovered = implied_volatility(
            market_price, self.S, self.K, self.T, self.r,
            option_type="put", method="brent",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-8)

    @pytest.mark.parametrize("true_sigma", [0.15, 0.25, 0.40])
    def test_newton_roundtrip(self, true_sigma):
        market_price = bs_price(self.S, self.K, self.T, self.r, true_sigma, "call")
        recovered = implied_volatility(
            market_price, self.S, self.K, self.T, self.r,
            option_type="call", method="newton",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-6)

    def test_otm_call(self):
        """Test with an OTM call."""
        S, K = 100.0, 120.0
        true_sigma = 0.25
        market_price = bs_price(S, K, 0.5, self.r, true_sigma, "call")
        recovered = implied_volatility(market_price, S, K, 0.5, self.r, "call")
        assert recovered == pytest.approx(true_sigma, abs=1e-8)

    def test_itm_put(self):
        """Test with an ITM put."""
        S, K = 90.0, 100.0
        true_sigma = 0.30
        market_price = bs_price(S, K, 1.0, self.r, true_sigma, "put")
        recovered = implied_volatility(market_price, S, K, 1.0, self.r, "put")
        assert recovered == pytest.approx(true_sigma, abs=1e-8)

    def test_invalid_price_below_intrinsic(self):
        """Price below intrinsic should raise ValueError."""
        # Deep ITM put: S=50, K=100, T=1, r=0.05
        # Intrinsic put = max(K*e^{-rT} - S, 0) = max(95.12 - 50, 0) ~ 45.12
        # Price of 10 is well below intrinsic
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_volatility(10.0, 50.0, 100.0, 1.0, 0.05, "put")

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, method="bisection")
