"""Tests for the binomial tree pricer."""

import pytest

from src.core.binomial import binomial_price, binomial_tree_data
from src.core.black_scholes import bs_call, bs_put


class TestBinomialConvergence:
    """Binomial prices should converge to BS for European options."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_european_call_converges_to_bs(self):
        bs = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        binom = binomial_price(self.S, self.K, self.T, self.r, self.sigma,
                               N=500, option_type="call", american=False)
        assert binom == pytest.approx(bs, abs=0.05)

    def test_european_put_converges_to_bs(self):
        bs = bs_put(self.S, self.K, self.T, self.r, self.sigma)
        binom = binomial_price(self.S, self.K, self.T, self.r, self.sigma,
                               N=500, option_type="put", american=False)
        assert binom == pytest.approx(bs, abs=0.05)

    def test_more_steps_is_more_accurate(self):
        bs = bs_call(self.S, self.K, self.T, self.r, self.sigma)
        binom_50 = binomial_price(self.S, self.K, self.T, self.r, self.sigma, N=50)
        binom_500 = binomial_price(self.S, self.K, self.T, self.r, self.sigma, N=500)
        assert abs(binom_500 - bs) < abs(binom_50 - bs)


class TestAmericanPricing:
    """American options should be worth at least as much as European."""

    def test_american_call_geq_european(self):
        # For non-dividend-paying stock, American call = European call
        euro = binomial_price(100, 100, 1, 0.05, 0.2, N=200, american=False)
        amer = binomial_price(100, 100, 1, 0.05, 0.2, N=200, american=True)
        assert amer >= euro - 1e-10

    def test_american_put_geq_european(self):
        # American put strictly >= European put
        euro = binomial_price(100, 100, 1, 0.05, 0.2, N=200, option_type="put", american=False)
        amer = binomial_price(100, 100, 1, 0.05, 0.2, N=200, option_type="put", american=True)
        assert amer >= euro - 1e-10

    def test_deep_itm_american_put_has_early_exercise_premium(self):
        # Deep ITM American put should have measurable early exercise premium
        euro = binomial_price(100, 130, 1, 0.05, 0.2, N=200, option_type="put", american=False)
        amer = binomial_price(100, 130, 1, 0.05, 0.2, N=200, option_type="put", american=True)
        assert amer > euro + 0.01  # Strict inequality

    def test_american_put_geq_intrinsic(self):
        # American put should always be worth at least intrinsic
        amer = binomial_price(80, 100, 1, 0.05, 0.2, N=200, option_type="put", american=True)
        intrinsic = max(100 - 80, 0)
        assert amer >= intrinsic


class TestBinomialTreeData:
    """Test the tree data generation for visualisation."""

    def test_tree_shape(self):
        data = binomial_tree_data(100, 100, 1, 0.05, 0.2, N=5)
        assert len(data["asset_tree"]) == 6  # steps 0..5
        assert len(data["option_tree"]) == 6
        for step in range(6):
            assert len(data["asset_tree"][step]) == step + 1
            assert len(data["option_tree"][step]) == step + 1

    def test_root_node_matches_price(self):
        data = binomial_tree_data(100, 100, 1, 0.05, 0.2, N=5, option_type="call")
        price = binomial_price(100, 100, 1, 0.05, 0.2, N=5, option_type="call")
        assert data["option_tree"][0][0] == pytest.approx(price, abs=1e-10)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            binomial_price(100, 100, 1, 0.05, 0.2, option_type="straddle")
