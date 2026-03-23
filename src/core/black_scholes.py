"""Black-Scholes pricing model for European options.

The Black-Scholes formula prices European call and put options under the
assumptions of log-normal stock prices, constant volatility, no dividends,
and continuous trading.

    C = S * N(d1) - K * e^{-rT} * N(d2)
    P = K * e^{-rT} * N(-d2) - S * N(-d1)

where:
    d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
"""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European call option using the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying (annualized).

    Returns
    -------
    float
        Theoretical call option price.
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European put option using the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying (annualized).

    Returns
    -------
    float
        Theoretical put option price.
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Price a European option using the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying (annualized).
    option_type : {"call", "put"}
        Type of option.

    Returns
    -------
    float
        Theoretical option price.
    """
    if option_type == "call":
        return bs_call(S, K, T, r, sigma)
    elif option_type == "put":
        return bs_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
