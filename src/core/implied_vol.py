"""Implied volatility computation.

Given a market price for an option, solve the Black-Scholes equation in
reverse to find the volatility (sigma) that makes BS(S, K, T, r, sigma)
equal to the observed market price.

Uses Brent's method (scipy.optimize.brentq) for robust root-finding, with
an optional Newton-Raphson fast path that leverages analytic vega.
"""

from __future__ import annotations

import math
from typing import Literal

from scipy.optimize import brentq

from src.core.black_scholes import bs_price
from src.core.greeks import vega as bs_vega


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    method: Literal["brent", "newton"] = "brent",
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Compute implied volatility from a market price.

    Parameters
    ----------
    market_price : float
        Observed option price in the market.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    option_type : {"call", "put"}
        Option type.
    method : {"brent", "newton"}
        Root-finding method.  Brent is more robust; Newton is faster when
        starting close to the solution.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Implied volatility (annualized).

    Raises
    ------
    ValueError
        If the market price is outside the no-arbitrage bounds.
    RuntimeError
        If the solver fails to converge.
    """
    # Sanity checks
    intrinsic_call = max(S - K * math.exp(-r * T), 0.0)
    intrinsic_put = max(K * math.exp(-r * T) - S, 0.0)

    if option_type == "call":
        if market_price < intrinsic_call - 1e-8:
            raise ValueError(
                f"Market price {market_price:.6f} is below intrinsic value "
                f"{intrinsic_call:.6f} for a call."
            )
        if market_price > S:
            raise ValueError(
                f"Market price {market_price:.6f} exceeds spot {S:.6f} — "
                "no valid implied vol."
            )
    else:
        if market_price < intrinsic_put - 1e-8:
            raise ValueError(
                f"Market price {market_price:.6f} is below intrinsic value "
                f"{intrinsic_put:.6f} for a put."
            )
        if market_price > K * math.exp(-r * T):
            raise ValueError(
                f"Market price {market_price:.6f} exceeds PV(K) — "
                "no valid implied vol."
            )

    if method == "brent":
        return _brent_iv(market_price, S, K, T, r, option_type, tol, max_iter)
    elif method == "newton":
        return _newton_iv(market_price, S, K, T, r, option_type, tol, max_iter)
    else:
        raise ValueError(f"method must be 'brent' or 'newton', got '{method}'")


def _brent_iv(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str,
    tol: float,
    max_iter: int,
) -> float:
    """Brent's method for implied volatility."""
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    # Search in [1e-6, 10.0] — covers virtually all practical cases
    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
    except ValueError as exc:
        raise RuntimeError(
            f"Brent's method failed to find IV: {exc}.  "
            f"Market price={market_price}, S={S}, K={K}, T={T}, r={r}"
        ) from exc


def _newton_iv(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str,
    tol: float,
    max_iter: int,
) -> float:
    """Newton-Raphson using analytic vega."""
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma)
        if abs(v) < 1e-15:
            raise RuntimeError("Vega is near zero — Newton step undefined.")
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
        if sigma <= 0:
            sigma = 1e-6  # keep positive
    raise RuntimeError(f"Newton's method did not converge after {max_iter} iterations.")


def implied_vol_surface(
    market_prices: list[list[float]],
    S: float,
    strikes: list[float],
    maturities: list[float],
    r: float,
    option_type: Literal["call", "put"] = "call",
) -> list[list[float]]:
    """Compute an implied volatility surface from a grid of market prices.

    Parameters
    ----------
    market_prices : list[list[float]]
        2D grid where market_prices[i][j] is the price for strike[j] and
        maturity[i].
    S : float
        Spot price.
    strikes : list[float]
        List of strike prices (columns).
    maturities : list[float]
        List of maturities in years (rows).
    r : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    list[list[float]]
        2D grid of implied volatilities (same shape as market_prices).
        Entries are NaN where the solver fails.
    """
    surface: list[list[float]] = []
    for i, T in enumerate(maturities):
        row: list[float] = []
        for j, K in enumerate(strikes):
            try:
                iv = implied_volatility(
                    market_prices[i][j], S, K, T, r,
                    option_type=option_type,
                )
                row.append(iv)
            except (ValueError, RuntimeError):
                row.append(float("nan"))
        surface.append(row)
    return surface
