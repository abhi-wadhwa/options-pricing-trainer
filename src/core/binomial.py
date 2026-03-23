"""Cox-Ross-Rubinstein (CRR) binomial tree model for option pricing.

The binomial model discretises the life of the option into *N* time steps.
At each step the underlying price moves up by factor *u = e^{sigma * sqrt(dt)}*
or down by *d = 1/u*.  The risk-neutral probability of an up-move is

    p = (e^{r * dt} - d) / (u - d)

Prices are computed by backward induction from the terminal payoff.  For
American options the algorithm takes the maximum of the continuation value
and the early-exercise payoff at every node.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: Literal["call", "put"] = "call",
    american: bool = False,
) -> float:
    """Price an option using the CRR binomial tree.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility (annualized).
    N : int
        Number of time steps in the tree (default 200).
    option_type : {"call", "put"}
        Option type.
    american : bool
        If True, allow early exercise (American option).

    Returns
    -------
    float
        Option price.
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    q = 1.0 - p

    # Terminal asset prices at step N
    # S * u^j * d^(N-j) for j = 0..N
    asset_prices = S * np.power(u, np.arange(N, -1, -1)) * np.power(d, np.arange(0, N + 1))

    # Terminal option values
    if option_type == "call":
        option_values = np.maximum(asset_prices - K, 0.0)
    elif option_type == "put":
        option_values = np.maximum(K - asset_prices, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Backward induction
    for step in range(N - 1, -1, -1):
        # Asset prices at this step
        asset_at_step = S * np.power(u, np.arange(step, -1, -1)) * np.power(d, np.arange(0, step + 1))
        # Continuation value
        option_values = disc * (p * option_values[:-1] + q * option_values[1:])
        # Early exercise check for American options
        if american:
            if option_type == "call":
                exercise = np.maximum(asset_at_step - K, 0.0)
            else:
                exercise = np.maximum(K - asset_at_step, 0.0)
            option_values = np.maximum(option_values, exercise)

    return float(option_values[0])


def binomial_tree_data(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 10,
    option_type: Literal["call", "put"] = "call",
    american: bool = False,
) -> dict:
    """Build the full binomial tree and return node data for visualisation.

    Returns a dict with:
        - ``asset_tree``: list of arrays, asset_tree[i] has i+1 prices at step i
        - ``option_tree``: list of arrays, option values at each node
        - ``params``: dict of u, d, p, dt

    Parameters are the same as :func:`binomial_price` but *N* should be small
    (e.g. 5-10) for readable visualisation.
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    q = 1.0 - p

    # Build forward asset tree
    asset_tree: list[np.ndarray] = []
    for step in range(N + 1):
        prices = S * np.power(u, np.arange(step, -1, -1)) * np.power(d, np.arange(0, step + 1))
        asset_tree.append(prices)

    # Terminal payoff
    if option_type == "call":
        payoff_fn = lambda s: max(s - K, 0.0)
    else:
        payoff_fn = lambda s: max(K - s, 0.0)

    option_tree: list[np.ndarray] = [np.array([])] * (N + 1)
    option_tree[N] = np.array([payoff_fn(s) for s in asset_tree[N]])

    # Backward induction
    for step in range(N - 1, -1, -1):
        continuation = disc * (p * option_tree[step + 1][:-1] + q * option_tree[step + 1][1:])
        if american:
            exercise = np.array([payoff_fn(s) for s in asset_tree[step]])
            option_tree[step] = np.maximum(continuation, exercise)
        else:
            option_tree[step] = continuation

    return {
        "asset_tree": [arr.tolist() for arr in asset_tree],
        "option_tree": [arr.tolist() for arr in option_tree],
        "params": {"u": u, "d": d, "p": p, "dt": dt},
    }
