"""Monte Carlo simulation for option pricing.

Generates *N* paths of Geometric Brownian Motion (GBM):

    S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

where Z ~ N(0,1).  The option payoff is computed for each path, discounted
to present value, and averaged to obtain the price.

Supports:
    - European call/put
    - Asian (arithmetic average)
    - Barrier (up-and-out, down-and-out, up-and-in, down-and-in)
    - Lookback (floating strike)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


def _simulate_gbm(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM paths.

    Returns array of shape (n_paths, n_steps + 1) where column 0 is S0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * math.sqrt(dt) * Z
    log_returns = drift + diffusion
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 0] = math.log(S)
    log_paths[:, 1:] = np.cumsum(log_returns, axis=1) + math.log(S)
    return np.exp(log_paths)


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    exotic: Literal["european", "asian", "barrier", "lookback"] = "european",
    barrier: float | None = None,
    barrier_type: Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"] | None = None,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> dict:
    """Price an option via Monte Carlo simulation.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price (ignored for lookback).
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : {"call", "put"}
        Call or put.
    exotic : {"european", "asian", "barrier", "lookback"}
        Exotic option type.
    barrier : float or None
        Barrier level (required for barrier options).
    barrier_type : str or None
        Barrier direction (required for barrier options).
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps per path.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``price`` : float — discounted expected payoff
        ``std_error`` : float — standard error of the estimate
        ``paths`` : np.ndarray — sample of first 200 paths for plotting
    """
    paths = _simulate_gbm(S, T, r, sigma, n_paths, n_steps, seed)
    disc = math.exp(-r * T)

    if exotic == "european":
        final = paths[:, -1]
        if option_type == "call":
            payoffs = np.maximum(final - K, 0.0)
        else:
            payoffs = np.maximum(K - final, 0.0)

    elif exotic == "asian":
        avg = np.mean(paths[:, 1:], axis=1)  # arithmetic average excluding S0
        if option_type == "call":
            payoffs = np.maximum(avg - K, 0.0)
        else:
            payoffs = np.maximum(K - avg, 0.0)

    elif exotic == "barrier":
        if barrier is None or barrier_type is None:
            raise ValueError("barrier and barrier_type are required for barrier options")

        final = paths[:, -1]
        if option_type == "call":
            vanilla_payoffs = np.maximum(final - K, 0.0)
        else:
            vanilla_payoffs = np.maximum(K - final, 0.0)

        max_prices = np.max(paths, axis=1)
        min_prices = np.min(paths, axis=1)

        if barrier_type == "up-and-out":
            knocked = max_prices >= barrier
            payoffs = np.where(knocked, 0.0, vanilla_payoffs)
        elif barrier_type == "down-and-out":
            knocked = min_prices <= barrier
            payoffs = np.where(knocked, 0.0, vanilla_payoffs)
        elif barrier_type == "up-and-in":
            activated = max_prices >= barrier
            payoffs = np.where(activated, vanilla_payoffs, 0.0)
        elif barrier_type == "down-and-in":
            activated = min_prices <= barrier
            payoffs = np.where(activated, vanilla_payoffs, 0.0)
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

    elif exotic == "lookback":
        if option_type == "call":
            # Floating-strike lookback call: payoff = S_T - S_min
            payoffs = np.maximum(paths[:, -1] - np.min(paths, axis=1), 0.0)
        else:
            # Floating-strike lookback put: payoff = S_max - S_T
            payoffs = np.maximum(np.max(paths, axis=1) - paths[:, -1], 0.0)
    else:
        raise ValueError(f"Unknown exotic type: {exotic}")

    price = disc * np.mean(payoffs)
    std_error = disc * np.std(payoffs) / math.sqrt(n_paths)

    # Return a small sample of paths for plotting
    sample_paths = paths[:min(200, n_paths), :]

    return {
        "price": float(price),
        "std_error": float(std_error),
        "paths": sample_paths,
    }
