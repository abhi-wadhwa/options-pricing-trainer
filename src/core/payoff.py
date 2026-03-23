"""Multi-leg option strategy builder and payoff diagram computation.

Supports building arbitrarily complex strategies from individual option legs
and stock positions, then computing payoff at expiry and current theoretical
value across a range of underlying prices.

Predefined strategies:
    - Bull call spread
    - Bear put spread
    - Straddle / Strangle
    - Iron condor
    - Butterfly spread
    - Covered call
    - Protective put
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.core.black_scholes import bs_price


@dataclass
class OptionLeg:
    """A single leg of an options strategy.

    Attributes
    ----------
    option_type : {"call", "put", "stock"}
        Type of instrument.  "stock" represents a position in the underlying.
    strike : float
        Strike price (ignored for stock).
    premium : float
        Premium paid (positive) or received (negative) per unit.
    quantity : int
        Number of contracts.  Positive = long, negative = short.
    """

    option_type: Literal["call", "put", "stock"]
    strike: float = 0.0
    premium: float = 0.0
    quantity: int = 1

    def payoff_at_expiry(self, S: float | np.ndarray) -> float | np.ndarray:
        """Compute payoff at expiry for a single unit, *excluding* premium."""
        if self.option_type == "call":
            return np.maximum(S - self.strike, 0.0) * self.quantity
        elif self.option_type == "put":
            return np.maximum(self.strike - S, 0.0) * self.quantity
        else:  # stock
            return S * self.quantity

    def profit_at_expiry(self, S: float | np.ndarray) -> float | np.ndarray:
        """Payoff minus net premium paid (profit/loss)."""
        return self.payoff_at_expiry(S) - self.premium * self.quantity

    def current_value(
        self, S: float | np.ndarray, T: float, r: float, sigma: float,
    ) -> float | np.ndarray:
        """Theoretical value of the leg using Black-Scholes.

        For stock legs, value is simply S * quantity.
        """
        if self.option_type == "stock":
            return S * self.quantity  # type: ignore[return-value]

        if isinstance(S, np.ndarray):
            vals = np.array([
                bs_price(float(s), self.strike, T, r, sigma, self.option_type)
                for s in S
            ])
        else:
            vals = bs_price(S, self.strike, T, r, sigma, self.option_type)

        return vals * self.quantity  # type: ignore[return-value]


class StrategyBuilder:
    """Build multi-leg option strategies and compute payoff diagrams."""

    def __init__(self) -> None:
        self.legs: list[OptionLeg] = []

    def add_leg(self, leg: OptionLeg) -> "StrategyBuilder":
        """Add a leg to the strategy.  Returns self for chaining."""
        self.legs.append(leg)
        return self

    def clear(self) -> None:
        """Remove all legs."""
        self.legs.clear()

    def total_premium(self) -> float:
        """Net premium paid (positive = debit, negative = credit)."""
        return sum(leg.premium * leg.quantity for leg in self.legs)

    def payoff_at_expiry(self, S: np.ndarray) -> np.ndarray:
        """Total payoff at expiry across all legs (excluding premiums)."""
        total = np.zeros_like(S, dtype=float)
        for leg in self.legs:
            total += leg.payoff_at_expiry(S)
        return total

    def profit_at_expiry(self, S: np.ndarray) -> np.ndarray:
        """Total profit/loss at expiry (payoff minus premiums paid)."""
        total = np.zeros_like(S, dtype=float)
        for leg in self.legs:
            total += leg.profit_at_expiry(S)
        return total

    def current_value(
        self, S: np.ndarray, T: float, r: float, sigma: float,
    ) -> np.ndarray:
        """Total current theoretical value of the strategy."""
        total = np.zeros_like(S, dtype=float)
        for leg in self.legs:
            total += leg.current_value(S, T, r, sigma)
        # Subtract net premium to get profit/loss relative to entry
        total -= self.total_premium()
        return total

    def payoff_diagram_data(
        self,
        S_range: tuple[float, float] | None = None,
        n_points: int = 500,
        T: float = 0.0,
        r: float = 0.05,
        sigma: float = 0.2,
    ) -> dict:
        """Generate data for a payoff diagram.

        Parameters
        ----------
        S_range : tuple[float, float] or None
            Min and max underlying price.  If None, auto-compute from strikes.
        n_points : int
            Number of points to evaluate.
        T : float
            Time to expiry for current value curve (0 = at expiry).
        r : float
            Risk-free rate (for current value curve).
        sigma : float
            Volatility (for current value curve).

        Returns
        -------
        dict
            ``S`` : array of underlying prices
            ``expiry_pnl`` : array of P/L at expiry
            ``current_pnl`` : array of P/L at current time (if T > 0)
            ``breakevens`` : list of approximate breakeven prices
            ``max_profit`` : float
            ``max_loss`` : float
        """
        # Auto-range based on strikes
        strikes = [leg.strike for leg in self.legs if leg.option_type != "stock"]
        if not strikes:
            strikes = [100.0]

        if S_range is None:
            mid = np.mean(strikes)
            spread = max(np.ptp(strikes), mid * 0.2)
            S_range = (mid - spread * 1.5, mid + spread * 1.5)

        S = np.linspace(S_range[0], S_range[1], n_points)
        expiry_pnl = self.profit_at_expiry(S)

        result: dict = {
            "S": S,
            "expiry_pnl": expiry_pnl,
        }

        if T > 0:
            result["current_pnl"] = self.current_value(S, T, r, sigma)

        # Find breakevens (sign changes)
        sign_changes = np.where(np.diff(np.sign(expiry_pnl)))[0]
        breakevens: list[float] = []
        for idx in sign_changes:
            # Linear interpolation
            x0, x1 = S[idx], S[idx + 1]
            y0, y1 = expiry_pnl[idx], expiry_pnl[idx + 1]
            if y1 != y0:
                be = x0 - y0 * (x1 - x0) / (y1 - y0)
                breakevens.append(float(be))
        result["breakevens"] = breakevens

        result["max_profit"] = float(np.max(expiry_pnl))
        result["max_loss"] = float(np.min(expiry_pnl))

        return result


# ---------------------------------------------------------------------------
# Predefined strategies
# ---------------------------------------------------------------------------

def bull_call_spread(
    S: float, K_low: float, K_high: float, T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long call at K_low, short call at K_high."""
    builder = StrategyBuilder()
    premium_long = bs_price(S, K_low, T, r, sigma, "call")
    premium_short = bs_price(S, K_high, T, r, sigma, "call")
    builder.add_leg(OptionLeg("call", K_low, premium_long, 1))
    builder.add_leg(OptionLeg("call", K_high, premium_short, -1))
    return builder


def bear_put_spread(
    S: float, K_low: float, K_high: float, T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long put at K_high, short put at K_low."""
    builder = StrategyBuilder()
    premium_long = bs_price(S, K_high, T, r, sigma, "put")
    premium_short = bs_price(S, K_low, T, r, sigma, "put")
    builder.add_leg(OptionLeg("put", K_high, premium_long, 1))
    builder.add_leg(OptionLeg("put", K_low, premium_short, -1))
    return builder


def straddle(
    S: float, K: float, T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long call + long put at the same strike."""
    builder = StrategyBuilder()
    builder.add_leg(OptionLeg("call", K, bs_price(S, K, T, r, sigma, "call"), 1))
    builder.add_leg(OptionLeg("put", K, bs_price(S, K, T, r, sigma, "put"), 1))
    return builder


def strangle(
    S: float, K_put: float, K_call: float, T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long put at K_put (lower), long call at K_call (higher)."""
    builder = StrategyBuilder()
    builder.add_leg(OptionLeg("put", K_put, bs_price(S, K_put, T, r, sigma, "put"), 1))
    builder.add_leg(OptionLeg("call", K_call, bs_price(S, K_call, T, r, sigma, "call"), 1))
    return builder


def iron_condor(
    S: float,
    K_put_low: float, K_put_high: float,
    K_call_low: float, K_call_high: float,
    T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Short put spread + short call spread.

    Sell put at K_put_high, buy put at K_put_low,
    sell call at K_call_low, buy call at K_call_high.
    """
    builder = StrategyBuilder()
    builder.add_leg(OptionLeg("put", K_put_low, bs_price(S, K_put_low, T, r, sigma, "put"), 1))
    builder.add_leg(OptionLeg("put", K_put_high, bs_price(S, K_put_high, T, r, sigma, "put"), -1))
    builder.add_leg(OptionLeg("call", K_call_low, bs_price(S, K_call_low, T, r, sigma, "call"), -1))
    builder.add_leg(OptionLeg("call", K_call_high, bs_price(S, K_call_high, T, r, sigma, "call"), 1))
    return builder


def butterfly_spread(
    S: float, K_low: float, K_mid: float, K_high: float,
    T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long call at K_low, short 2 calls at K_mid, long call at K_high."""
    builder = StrategyBuilder()
    builder.add_leg(OptionLeg("call", K_low, bs_price(S, K_low, T, r, sigma, "call"), 1))
    builder.add_leg(OptionLeg("call", K_mid, bs_price(S, K_mid, T, r, sigma, "call"), -2))
    builder.add_leg(OptionLeg("call", K_high, bs_price(S, K_high, T, r, sigma, "call"), 1))
    return builder


def covered_call(
    S: float, K: float, T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long stock + short call."""
    builder = StrategyBuilder()
    builder.add_leg(OptionLeg("stock", 0.0, S, 1))
    builder.add_leg(OptionLeg("call", K, bs_price(S, K, T, r, sigma, "call"), -1))
    return builder


def protective_put(
    S: float, K: float, T: float, r: float, sigma: float,
) -> StrategyBuilder:
    """Long stock + long put."""
    builder = StrategyBuilder()
    builder.add_leg(OptionLeg("stock", 0.0, S, 1))
    builder.add_leg(OptionLeg("put", K, bs_price(S, K, T, r, sigma, "put"), 1))
    return builder
