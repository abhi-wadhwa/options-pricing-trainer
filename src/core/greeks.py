"""Option Greeks — sensitivity measures.

Analytic Greeks for the Black-Scholes model and numerical (finite-difference)
Greeks for any pricing function.

Analytic formulae (European options):
    Delta_call  = N(d1)
    Delta_put   = N(d1) - 1
    Gamma       = n(d1) / (S * sigma * sqrt(T))
    Vega        = S * n(d1) * sqrt(T)
    Theta_call  = -(S * n(d1) * sigma) / (2*sqrt(T)) - r*K*e^{-rT}*N(d2)
    Theta_put   = -(S * n(d1) * sigma) / (2*sqrt(T)) + r*K*e^{-rT}*N(-d2)
    Rho_call    = K * T * e^{-rT} * N(d2)
    Rho_put     = -K * T * e^{-rT} * N(-d2)

where N is the standard normal CDF and n is the standard normal PDF.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

from scipy.stats import norm

from src.core.black_scholes import _d1, _d2, bs_price


# ---------------------------------------------------------------------------
# Analytic Greeks (Black-Scholes)
# ---------------------------------------------------------------------------

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Black-Scholes delta: dC/dS (call) or dP/dS (put)."""
    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


def gamma(S: float, K: float, T: float, r: float, sigma: float, **_kw) -> float:
    """Black-Scholes gamma: d2C/dS2 (same for calls and puts)."""
    d1 = _d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def vega(S: float, K: float, T: float, r: float, sigma: float, **_kw) -> float:
    """Black-Scholes vega: dC/d(sigma).

    Returns the change in price for a 1-unit (100 pp) change in vol.
    To get the change per 1 pp, divide by 100.
    """
    d1 = _d1(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * math.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Black-Scholes theta: -dC/dT (per year).

    Divide by 365 to get per-calendar-day theta.
    """
    d1 = _d1(S, K, T, r, sigma)
    d2_val = d1 - sigma * math.sqrt(T)
    first_term = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type == "call":
        return float(first_term - r * K * math.exp(-r * T) * norm.cdf(d2_val))
    else:
        return float(first_term + r * K * math.exp(-r * T) * norm.cdf(-d2_val))


def rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Black-Scholes rho: dC/dr."""
    d2_val = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(K * T * math.exp(-r * T) * norm.cdf(d2_val))
    else:
        return float(-K * T * math.exp(-r * T) * norm.cdf(-d2_val))


def all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict[str, float]:
    """Compute all analytic BS Greeks at once.

    Returns dict with keys: delta, gamma, vega, theta, rho.
    """
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho": rho(S, K, T, r, sigma, option_type),
    }


# ---------------------------------------------------------------------------
# Numerical Greeks (finite-difference)
# ---------------------------------------------------------------------------

def numerical_greeks(
    pricing_fn: Callable[..., float],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    dS: float | None = None,
    d_sigma: float = 0.01,
    dT: float = 1.0 / 365.0,
    dr: float = 0.01,
    **pricing_kwargs,
) -> dict[str, float]:
    """Compute Greeks numerically via central finite differences.

    ``pricing_fn`` must accept (S, K, T, r, sigma, option_type=..., **kwargs)
    and return a float price.

    Parameters
    ----------
    pricing_fn : callable
        Pricing function.
    S, K, T, r, sigma : float
        Option parameters.
    option_type : str
        'call' or 'put'.
    dS : float or None
        Bump size for S (default 1% of S).
    d_sigma : float
        Bump size for sigma (default 0.01).
    dT : float
        Bump size for T (default 1 day).
    dr : float
        Bump size for r (default 0.01).
    **pricing_kwargs
        Extra keyword arguments forwarded to pricing_fn.

    Returns
    -------
    dict
        Keys: delta, gamma, vega, theta, rho.
    """
    if dS is None:
        dS = S * 0.01

    def _price(s=S, k=K, t=T, rate=r, vol=sigma):
        return pricing_fn(s, k, t, rate, vol, option_type=option_type, **pricing_kwargs)

    base = _price()

    # Delta & Gamma
    p_up = _price(s=S + dS)
    p_dn = _price(s=S - dS)
    num_delta = (p_up - p_dn) / (2.0 * dS)
    num_gamma = (p_up - 2.0 * base + p_dn) / (dS ** 2)

    # Vega
    p_vol_up = _price(vol=sigma + d_sigma)
    p_vol_dn = _price(vol=max(sigma - d_sigma, 1e-6))
    num_vega = (p_vol_up - p_vol_dn) / (2.0 * d_sigma)

    # Theta (negative of dC/dT)
    if T - dT > 0:
        p_t_dn = _price(t=T - dT)
        num_theta = -(p_t_dn - base) / dT  # dC/dT but theta = -dC/dT
        # Actually: theta = (V(T-dT) - V(T)) / dT  (price decays)
        num_theta = (p_t_dn - base) / dT
    else:
        p_t_up = _price(t=T + dT)
        num_theta = -(p_t_up - base) / dT

    # Rho
    p_r_up = _price(rate=r + dr)
    p_r_dn = _price(rate=r - dr)
    num_rho = (p_r_up - p_r_dn) / (2.0 * dr)

    return {
        "delta": num_delta,
        "gamma": num_gamma,
        "vega": num_vega,
        "theta": num_theta,
        "rho": num_rho,
    }
