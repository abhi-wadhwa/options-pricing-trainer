"""Delta hedging simulator.

Simulates the dynamic delta hedge of a sold European option over its
lifetime.  Each day the hedger:

1. Computes the BS delta of the option.
2. Buys/sells shares of the underlying to be delta-neutral.
3. Finances the position at the risk-free rate.

At expiry the option is settled and the total P/L is computed.  Running
the simulation many times produces a P/L distribution that illustrates
hedging error due to discrete rebalancing, transaction costs, and
realised vs. implied volatility differences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.core.black_scholes import bs_price
from src.core.greeks import delta as bs_delta


@dataclass
class HedgeResult:
    """Result of a single delta-hedging simulation path.

    Attributes
    ----------
    pnl : float
        Terminal P/L of the hedged portfolio.
    stock_path : np.ndarray
        Simulated stock prices (length n_steps + 1).
    delta_path : np.ndarray
        Delta at each rebalance point.
    hedge_shares_path : np.ndarray
        Number of shares held at each rebalance point.
    cash_path : np.ndarray
        Cash balance at each rebalance point.
    """

    pnl: float
    stock_path: np.ndarray
    delta_path: np.ndarray
    hedge_shares_path: np.ndarray
    cash_path: np.ndarray


def delta_hedge_single_path(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    sigma_realised: float | None = None,
    option_type: Literal["call", "put"] = "call",
    n_steps: int = 252,
    seed: int | None = None,
    transaction_cost: float = 0.0,
) -> HedgeResult:
    """Run a single delta-hedging simulation.

    The hedger sells one option and delta-hedges using the *implied*
    volatility.  The stock moves according to *realised* volatility.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    sigma_implied : float
        Implied volatility used for hedging (BS delta computation).
    sigma_realised : float or None
        Realised volatility used for simulating the stock.  If None, uses
        sigma_implied (perfect vol forecast).
    option_type : {"call", "put"}
        Option type being hedged.
    n_steps : int
        Number of rebalancing steps (default 252 = daily for 1 year).
    seed : int or None
        Random seed.
    transaction_cost : float
        Proportional transaction cost per share traded (e.g. 0.001 = 10 bps).

    Returns
    -------
    HedgeResult
    """
    if sigma_realised is None:
        sigma_realised = sigma_implied

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Simulate stock path under realised vol
    Z = rng.standard_normal(n_steps)
    log_returns = (r - 0.5 * sigma_realised ** 2) * dt + sigma_realised * math.sqrt(dt) * Z
    stock_path = np.zeros(n_steps + 1)
    stock_path[0] = S0
    for i in range(n_steps):
        stock_path[i + 1] = stock_path[i] * math.exp(log_returns[i])

    # Option premium received at t=0
    option_premium = bs_price(S0, K, T, r, sigma_implied, option_type)

    # Hedging arrays
    delta_path = np.zeros(n_steps + 1)
    hedge_shares_path = np.zeros(n_steps + 1)
    cash_path = np.zeros(n_steps + 1)

    # Initial hedge
    remaining_T = T
    d = bs_delta(S0, K, remaining_T, r, sigma_implied, option_type)
    delta_path[0] = d
    shares = d  # Shares to hold to hedge short option
    hedge_shares_path[0] = shares
    # Cash = premium received - cost of shares - transaction costs
    cash_path[0] = option_premium - shares * S0 - abs(shares) * S0 * transaction_cost

    for i in range(1, n_steps + 1):
        S_i = stock_path[i]
        remaining_T = T - i * dt

        # Earn interest on cash
        cash_path[i] = cash_path[i - 1] * math.exp(r * dt)

        if remaining_T > 1e-10:
            d = bs_delta(S_i, K, remaining_T, r, sigma_implied, option_type)
        else:
            # At expiry, delta is 0 or 1
            if option_type == "call":
                d = 1.0 if S_i > K else 0.0
            else:
                d = -1.0 if S_i < K else 0.0

        delta_path[i] = d
        shares_needed = d
        trade = shares_needed - hedge_shares_path[i - 1]
        hedge_shares_path[i] = shares_needed

        # Buy/sell shares and pay transaction costs
        cash_path[i] -= trade * S_i + abs(trade) * S_i * transaction_cost

    # At expiry: liquidate share position and settle option
    S_T = stock_path[-1]
    final_cash = cash_path[-1] + hedge_shares_path[-1] * S_T

    # Option settlement (we are short the option)
    if option_type == "call":
        option_payoff = max(S_T - K, 0.0)
    else:
        option_payoff = max(K - S_T, 0.0)
    final_cash -= option_payoff  # pay option holder

    pnl = final_cash

    return HedgeResult(
        pnl=float(pnl),
        stock_path=stock_path,
        delta_path=delta_path,
        hedge_shares_path=hedge_shares_path,
        cash_path=cash_path,
    )


def delta_hedge_simulation(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    sigma_realised: float | None = None,
    option_type: Literal["call", "put"] = "call",
    n_steps: int = 252,
    n_simulations: int = 1000,
    transaction_cost: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Run many delta-hedging simulations and return P/L statistics.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    T : float
        Time to expiry.
    r : float
        Risk-free rate.
    sigma_implied : float
        Implied vol used for hedging.
    sigma_realised : float or None
        Realised vol for stock simulation.
    option_type : {"call", "put"}
        Option type.
    n_steps : int
        Rebalancing frequency.
    n_simulations : int
        Number of simulation paths.
    transaction_cost : float
        Proportional transaction cost.
    seed : int or None
        Master random seed.

    Returns
    -------
    dict
        ``pnls`` : np.ndarray of P/Ls
        ``mean_pnl`` : float
        ``std_pnl`` : float
        ``sample_paths`` : list of first 5 HedgeResult objects
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_simulations)

    pnls = np.zeros(n_simulations)
    sample_paths: list[HedgeResult] = []

    for i in range(n_simulations):
        result = delta_hedge_single_path(
            S0, K, T, r, sigma_implied, sigma_realised,
            option_type, n_steps, int(seeds[i]), transaction_cost,
        )
        pnls[i] = result.pnl
        if i < 5:
            sample_paths.append(result)

    return {
        "pnls": pnls,
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "sample_paths": sample_paths,
    }
