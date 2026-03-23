"""Demo script showcasing all features of the Options Pricing Trainer."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.core.black_scholes import bs_call, bs_put, bs_price
from src.core.binomial import binomial_price
from src.core.monte_carlo import monte_carlo_price
from src.core.greeks import all_greeks, numerical_greeks
from src.core.implied_vol import implied_volatility
from src.core.payoff import (
    OptionLeg, StrategyBuilder,
    bull_call_spread, iron_condor, straddle,
)
from src.core.delta_hedge import delta_hedge_simulation


def main():
    print("=" * 70)
    print("OPTIONS PRICING TRAINER — DEMO")
    print("=" * 70)

    # Parameters
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.20

    # ── 1. Black-Scholes ──────────────────────────────────────────────────
    print("\n1. BLACK-SCHOLES PRICING")
    print("-" * 40)
    call_price = bs_call(S, K, T, r, sigma)
    put_price = bs_put(S, K, T, r, sigma)
    print(f"   Call price: ${call_price:.8f}")
    print(f"   Put price:  ${put_price:.8f}")
    parity = call_price - put_price - (S - K * np.exp(-r * T))
    print(f"   Put-call parity error: {parity:.2e}")

    # ── 2. Binomial Tree ──────────────────────────────────────────────────
    print("\n2. BINOMIAL TREE PRICING")
    print("-" * 40)
    euro_call = binomial_price(S, K, T, r, sigma, N=500, option_type="call", american=False)
    amer_put = binomial_price(S, K, T, r, sigma, N=500, option_type="put", american=True)
    euro_put = binomial_price(S, K, T, r, sigma, N=500, option_type="put", american=False)
    print(f"   European call (500 steps): ${euro_call:.6f}")
    print(f"   European put  (500 steps): ${euro_put:.6f}")
    print(f"   American put  (500 steps): ${amer_put:.6f}")
    print(f"   Early exercise premium:    ${amer_put - euro_put:.6f}")

    # ── 3. Monte Carlo ────────────────────────────────────────────────────
    print("\n3. MONTE CARLO PRICING")
    print("-" * 40)
    mc_euro = monte_carlo_price(S, K, T, r, sigma, option_type="call", seed=42)
    mc_asian = monte_carlo_price(S, K, T, r, sigma, option_type="call", exotic="asian", seed=42)
    mc_lookback = monte_carlo_price(S, K, T, r, sigma, option_type="call", exotic="lookback", seed=42)
    print(f"   European call: ${mc_euro['price']:.6f} (SE: {mc_euro['std_error']:.6f})")
    print(f"   Asian call:    ${mc_asian['price']:.6f} (SE: {mc_asian['std_error']:.6f})")
    print(f"   Lookback call: ${mc_lookback['price']:.6f} (SE: {mc_lookback['std_error']:.6f})")

    # ── 4. Greeks ─────────────────────────────────────────────────────────
    print("\n4. GREEKS (ANALYTIC)")
    print("-" * 40)
    greeks = all_greeks(S, K, T, r, sigma, "call")
    for name, val in greeks.items():
        print(f"   {name:>6s}: {val:>12.8f}")

    print("\n   Numerical Greeks (BS pricing function):")
    num_greeks = numerical_greeks(bs_price, S, K, T, r, sigma, "call")
    for name, val in num_greeks.items():
        print(f"   {name:>6s}: {val:>12.8f}")

    # ── 5. Implied Volatility ─────────────────────────────────────────────
    print("\n5. IMPLIED VOLATILITY")
    print("-" * 40)
    market_price = bs_call(S, K, T, r, 0.25)  # True vol is 25%
    iv_brent = implied_volatility(market_price, S, K, T, r, "call", method="brent")
    iv_newton = implied_volatility(market_price, S, K, T, r, "call", method="newton")
    print(f"   Market price (true vol=25%): ${market_price:.6f}")
    print(f"   Recovered IV (Brent):  {iv_brent:.8f} ({iv_brent*100:.4f}%)")
    print(f"   Recovered IV (Newton): {iv_newton:.8f} ({iv_newton*100:.4f}%)")

    # ── 6. Strategy Payoff ────────────────────────────────────────────────
    print("\n6. STRATEGY PAYOFF DIAGRAMS")
    print("-" * 40)
    spread = bull_call_spread(S, 95.0, 105.0, T, r, sigma)
    data = spread.payoff_diagram_data()
    print(f"   Bull Call Spread (95/105):")
    print(f"   Net premium:  ${spread.total_premium():.4f}")
    print(f"   Max profit:   ${data['max_profit']:.4f}")
    print(f"   Max loss:     ${data['max_loss']:.4f}")
    if data["breakevens"]:
        print(f"   Breakeven(s): {', '.join(f'${be:.2f}' for be in data['breakevens'])}")

    ic = iron_condor(S, 85, 90, 110, 115, T, r, sigma)
    ic_data = ic.payoff_diagram_data()
    print(f"\n   Iron Condor (85/90/110/115):")
    print(f"   Net premium:  ${ic.total_premium():.4f}")
    print(f"   Max profit:   ${ic_data['max_profit']:.4f}")
    print(f"   Max loss:     ${ic_data['max_loss']:.4f}")

    # ── 7. Delta Hedging ──────────────────────────────────────────────────
    print("\n7. DELTA HEDGING SIMULATION")
    print("-" * 40)
    hedge_result = delta_hedge_simulation(
        S, K, T, r, sigma,
        sigma_realised=0.20,
        n_simulations=1000,
        seed=42,
    )
    print(f"   Simulations: 1000")
    print(f"   Mean P/L:    ${hedge_result['mean_pnl']:.4f}")
    print(f"   Std P/L:     ${hedge_result['std_pnl']:.4f}")
    print(f"   Option premium: ${bs_price(S, K, T, r, sigma, 'call'):.4f}")

    # With vol mismatch
    hedge_mismatch = delta_hedge_simulation(
        S, K, T, r, sigma,
        sigma_realised=0.30,  # Realised > implied
        n_simulations=1000,
        seed=42,
    )
    print(f"\n   With vol mismatch (implied=20%, realised=30%):")
    print(f"   Mean P/L: ${hedge_mismatch['mean_pnl']:.4f} (should be negative)")

    print("\n" + "=" * 70)
    print("Run `streamlit run src/viz/app.py` for the interactive UI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
