"""Streamlit interactive UI for Options Pricing Trainer.

Run with:
    streamlit run src/viz/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.black_scholes import bs_price, bs_call, bs_put
from src.core.binomial import binomial_price, binomial_tree_data
from src.core.monte_carlo import monte_carlo_price
from src.core.greeks import all_greeks, delta as bs_delta
from src.core.implied_vol import implied_volatility, implied_vol_surface
from src.core.payoff import (
    OptionLeg, StrategyBuilder,
    bull_call_spread, bear_put_spread, straddle, strangle,
    iron_condor, butterfly_spread, covered_call, protective_put,
)
from src.core.delta_hedge import delta_hedge_simulation


st.set_page_config(
    page_title="Options Pricing Trainer",
    page_icon="📈",
    layout="wide",
)

st.title("Options Pricing Trainer")
st.markdown("*Interactive education for options pricing, Greeks, and strategies*")

tab_pricer, tab_payoff, tab_tree, tab_hedge, tab_surface, tab_quiz = st.tabs([
    "Option Pricer",
    "Payoff Diagrams",
    "Binomial Tree",
    "Delta Hedging",
    "IV Surface",
    "Quiz Mode",
])


# ============================================================================
# TAB 1: Option Pricer
# ============================================================================
with tab_pricer:
    st.header("Black-Scholes Option Pricer")

    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, step=1.0, key="pricer_S")
        K = st.number_input("Strike Price (K)", value=105.0, min_value=0.01, step=1.0, key="pricer_K")
        T = st.number_input("Time to Expiry (T, years)", value=0.5, min_value=0.001, step=0.1, key="pricer_T")

    with col2:
        r = st.number_input("Risk-free Rate (r)", value=0.05, step=0.01, format="%.4f", key="pricer_r")
        sigma = st.number_input("Volatility (sigma)", value=0.20, min_value=0.001, step=0.01, format="%.4f", key="pricer_sigma")
        option_type = st.selectbox("Option Type", ["call", "put"], key="pricer_type")

    model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial (European)", "Binomial (American)", "Monte Carlo"])

    if st.button("Calculate", key="pricer_calc"):
        if model == "Black-Scholes":
            price = bs_price(S, K, T, r, sigma, option_type)
            st.success(f"**{option_type.title()} Price: ${price:.6f}**")
        elif model.startswith("Binomial"):
            american = "American" in model
            n_steps = st.session_state.get("binom_steps", 200)
            price = binomial_price(S, K, T, r, sigma, N=200, option_type=option_type, american=american)
            label = "American" if american else "European"
            st.success(f"**Binomial ({label}) {option_type.title()} Price: ${price:.6f}**")
        else:
            result = monte_carlo_price(S, K, T, r, sigma, option_type=option_type, n_paths=100_000, seed=42)
            st.success(f"**Monte Carlo {option_type.title()} Price: ${result['price']:.6f}** (SE: {result['std_error']:.6f})")

        # Always show Greeks
        greeks = all_greeks(S, K, T, r, sigma, option_type)
        st.subheader("Greeks")
        gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
        gcol1.metric("Delta", f"{greeks['delta']:.6f}")
        gcol2.metric("Gamma", f"{greeks['gamma']:.6f}")
        gcol3.metric("Vega", f"{greeks['vega']:.6f}")
        gcol4.metric("Theta", f"{greeks['theta']:.6f}")
        gcol5.metric("Rho", f"{greeks['rho']:.6f}")

        # Greeks surface: delta and gamma vs S
        S_range = np.linspace(S * 0.5, S * 1.5, 200)
        deltas = [bs_delta(s, K, T, r, sigma, option_type) for s in S_range]
        fig_greeks = go.Figure()
        fig_greeks.add_trace(go.Scatter(x=S_range, y=deltas, name="Delta", line=dict(width=2)))
        fig_greeks.update_layout(
            title="Delta vs Spot Price",
            xaxis_title="Spot Price",
            yaxis_title="Delta",
            template="plotly_white",
        )
        st.plotly_chart(fig_greeks, use_container_width=True)


# ============================================================================
# TAB 2: Payoff Diagrams
# ============================================================================
with tab_payoff:
    st.header("Strategy Payoff Diagrams")

    strategy_choice = st.selectbox(
        "Predefined Strategy",
        ["Custom", "Bull Call Spread", "Bear Put Spread", "Straddle",
         "Strangle", "Iron Condor", "Butterfly", "Covered Call", "Protective Put"],
        key="strategy_choice",
    )

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        p_S = st.number_input("Spot Price", value=100.0, min_value=0.01, key="pay_S")
        p_T = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.001, key="pay_T")
        p_r = st.number_input("Risk-free Rate", value=0.05, key="pay_r")
        p_sigma = st.number_input("Volatility", value=0.20, min_value=0.001, key="pay_sigma")

    builder = None

    if strategy_choice == "Bull Call Spread":
        with pcol2:
            k_low = st.number_input("Lower Strike", value=95.0, key="bs_kl")
            k_high = st.number_input("Upper Strike", value=105.0, key="bs_kh")
        builder = bull_call_spread(p_S, k_low, k_high, p_T, p_r, p_sigma)

    elif strategy_choice == "Bear Put Spread":
        with pcol2:
            k_low = st.number_input("Lower Strike", value=95.0, key="bps_kl")
            k_high = st.number_input("Upper Strike", value=105.0, key="bps_kh")
        builder = bear_put_spread(p_S, k_low, k_high, p_T, p_r, p_sigma)

    elif strategy_choice == "Straddle":
        with pcol2:
            k_strad = st.number_input("Strike", value=100.0, key="strad_k")
        builder = straddle(p_S, k_strad, p_T, p_r, p_sigma)

    elif strategy_choice == "Strangle":
        with pcol2:
            k_put = st.number_input("Put Strike", value=95.0, key="strang_kp")
            k_call = st.number_input("Call Strike", value=105.0, key="strang_kc")
        builder = strangle(p_S, k_put, k_call, p_T, p_r, p_sigma)

    elif strategy_choice == "Iron Condor":
        with pcol2:
            ic_kpl = st.number_input("Put Low Strike", value=90.0, key="ic_kpl")
            ic_kph = st.number_input("Put High Strike", value=95.0, key="ic_kph")
            ic_kcl = st.number_input("Call Low Strike", value=105.0, key="ic_kcl")
            ic_kch = st.number_input("Call High Strike", value=110.0, key="ic_kch")
        builder = iron_condor(p_S, ic_kpl, ic_kph, ic_kcl, ic_kch, p_T, p_r, p_sigma)

    elif strategy_choice == "Butterfly":
        with pcol2:
            bf_kl = st.number_input("Low Strike", value=90.0, key="bf_kl")
            bf_km = st.number_input("Mid Strike", value=100.0, key="bf_km")
            bf_kh = st.number_input("High Strike", value=110.0, key="bf_kh")
        builder = butterfly_spread(p_S, bf_kl, bf_km, bf_kh, p_T, p_r, p_sigma)

    elif strategy_choice == "Covered Call":
        with pcol2:
            cc_k = st.number_input("Call Strike", value=105.0, key="cc_k")
        builder = covered_call(p_S, cc_k, p_T, p_r, p_sigma)

    elif strategy_choice == "Protective Put":
        with pcol2:
            pp_k = st.number_input("Put Strike", value=95.0, key="pp_k")
        builder = protective_put(p_S, pp_k, p_T, p_r, p_sigma)

    else:
        # Custom strategy
        with pcol2:
            st.markdown("**Add legs manually:**")
            n_legs = st.number_input("Number of legs", value=2, min_value=1, max_value=8, key="n_legs")

        builder = StrategyBuilder()
        for i in range(int(n_legs)):
            with st.expander(f"Leg {i + 1}", expanded=(i == 0)):
                lcol1, lcol2, lcol3, lcol4 = st.columns(4)
                with lcol1:
                    ltype = st.selectbox("Type", ["call", "put", "stock"], key=f"leg_type_{i}")
                with lcol2:
                    lstrike = st.number_input("Strike", value=100.0, key=f"leg_strike_{i}")
                with lcol3:
                    lqty = st.number_input("Quantity (+long/-short)", value=1, step=1, key=f"leg_qty_{i}")
                with lcol4:
                    if ltype != "stock":
                        lprem = bs_price(p_S, lstrike, p_T, p_r, p_sigma, ltype)
                    else:
                        lprem = p_S
                    st.text(f"Premium: {lprem:.4f}")
                builder.add_leg(OptionLeg(ltype, lstrike, lprem, int(lqty)))

    if builder and st.button("Plot Payoff", key="plot_payoff"):
        data = builder.payoff_diagram_data(T=p_T, r=p_r, sigma=p_sigma)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["S"], y=data["expiry_pnl"],
            name="P/L at Expiry", line=dict(width=2.5, color="blue"),
        ))
        if "current_pnl" in data:
            fig.add_trace(go.Scatter(
                x=data["S"], y=data["current_pnl"],
                name=f"P/L Now (T={p_T:.2f}y)", line=dict(width=2, dash="dash", color="orange"),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        for be in data["breakevens"]:
            fig.add_vline(x=be, line_dash="dot", line_color="red", annotation_text=f"BE: {be:.1f}")

        fig.update_layout(
            title=f"{strategy_choice} Payoff Diagram",
            xaxis_title="Underlying Price at Expiry",
            yaxis_title="Profit / Loss",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Net Premium", f"${builder.total_premium():.4f}")
        mcol2.metric("Max Profit", f"${data['max_profit']:.4f}")
        mcol3.metric("Max Loss", f"${data['max_loss']:.4f}")
        if data["breakevens"]:
            st.info(f"Breakeven(s): {', '.join(f'${be:.2f}' for be in data['breakevens'])}")


# ============================================================================
# TAB 3: Binomial Tree Visualization
# ============================================================================
with tab_tree:
    st.header("Binomial Tree Visualization")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        t_S = st.number_input("Spot Price", value=100.0, key="tree_S")
        t_K = st.number_input("Strike Price", value=100.0, key="tree_K")
        t_T = st.number_input("Time to Expiry (years)", value=1.0, key="tree_T")
    with tcol2:
        t_r = st.number_input("Risk-free Rate", value=0.05, key="tree_r")
        t_sigma = st.number_input("Volatility", value=0.20, key="tree_sigma")
        t_N = st.slider("Number of Steps", min_value=2, max_value=10, value=5, key="tree_N")
        t_type = st.selectbox("Option Type", ["call", "put"], key="tree_type")
        t_american = st.checkbox("American (early exercise)", key="tree_american")

    if st.button("Build Tree", key="build_tree"):
        tree_data = binomial_tree_data(t_S, t_K, t_T, t_r, t_sigma, t_N, t_type, t_american)

        fig = go.Figure()

        # Plot asset price nodes
        for step, prices in enumerate(tree_data["asset_tree"]):
            for node_idx, price in enumerate(prices):
                opt_val = tree_data["option_tree"][step][node_idx]
                fig.add_trace(go.Scatter(
                    x=[step], y=[price],
                    mode="markers+text",
                    marker=dict(size=12, color="steelblue"),
                    text=[f"S={price:.1f}<br>V={opt_val:.2f}"],
                    textposition="top center",
                    textfont=dict(size=9),
                    showlegend=False,
                ))
                # Draw edges to next step
                if step < t_N:
                    next_prices = tree_data["asset_tree"][step + 1]
                    # Up edge
                    fig.add_trace(go.Scatter(
                        x=[step, step + 1], y=[price, next_prices[node_idx]],
                        mode="lines", line=dict(color="lightblue", width=1),
                        showlegend=False,
                    ))
                    # Down edge
                    fig.add_trace(go.Scatter(
                        x=[step, step + 1], y=[price, next_prices[node_idx + 1]],
                        mode="lines", line=dict(color="lightcoral", width=1),
                        showlegend=False,
                    ))

        style = "American" if t_american else "European"
        fig.update_layout(
            title=f"CRR Binomial Tree ({style} {t_type.title()}, {t_N} steps)",
            xaxis_title="Step",
            yaxis_title="Asset Price",
            template="plotly_white",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        params = tree_data["params"]
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("u (up factor)", f"{params['u']:.6f}")
        pcol2.metric("d (down factor)", f"{params['d']:.6f}")
        pcol3.metric("p (risk-neutral prob)", f"{params['p']:.6f}")
        pcol4.metric("Option Price", f"${tree_data['option_tree'][0][0]:.6f}")


# ============================================================================
# TAB 4: Delta Hedging Simulation
# ============================================================================
with tab_hedge:
    st.header("Delta Hedging Simulator")
    st.markdown("""
    Simulate selling an option and dynamically delta-hedging.
    See how rebalancing frequency, transaction costs, and vol mismatch
    affect hedging P/L.
    """)

    hcol1, hcol2 = st.columns(2)
    with hcol1:
        h_S = st.number_input("Spot Price", value=100.0, key="hedge_S")
        h_K = st.number_input("Strike Price", value=100.0, key="hedge_K")
        h_T = st.number_input("Time to Expiry (years)", value=0.25, key="hedge_T")
        h_r = st.number_input("Risk-free Rate", value=0.05, key="hedge_r")
    with hcol2:
        h_sigma_imp = st.number_input("Implied Volatility (hedge with)", value=0.20, key="hedge_sig_imp")
        h_sigma_real = st.number_input("Realised Volatility (actual)", value=0.20, key="hedge_sig_real")
        h_n_sims = st.slider("Number of Simulations", 100, 5000, 1000, key="hedge_nsims")
        h_tc = st.number_input("Transaction Cost (bps)", value=0.0, step=1.0, key="hedge_tc")
        h_type = st.selectbox("Option Type", ["call", "put"], key="hedge_type")

    if st.button("Run Simulation", key="run_hedge"):
        with st.spinner("Simulating..."):
            result = delta_hedge_simulation(
                h_S, h_K, h_T, h_r, h_sigma_imp,
                sigma_realised=h_sigma_real,
                option_type=h_type,
                n_simulations=h_n_sims,
                transaction_cost=h_tc / 10000.0,
                seed=42,
            )

        # P/L histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=result["pnls"], nbinsx=50,
            marker_color="steelblue", opacity=0.8,
        ))
        fig_hist.add_vline(x=result["mean_pnl"], line_dash="dash", line_color="red",
                           annotation_text=f"Mean: ${result['mean_pnl']:.4f}")
        fig_hist.update_layout(
            title="Delta Hedging P/L Distribution",
            xaxis_title="P/L ($)",
            yaxis_title="Frequency",
            template="plotly_white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Mean P/L", f"${result['mean_pnl']:.4f}")
        mcol2.metric("Std Dev", f"${result['std_pnl']:.4f}")
        mcol3.metric("Option Premium", f"${bs_price(h_S, h_K, h_T, h_r, h_sigma_imp, h_type):.4f}")

        # Show a sample path
        if result["sample_paths"]:
            sample = result["sample_paths"][0]
            fig_path = go.Figure()
            t_axis = np.linspace(0, h_T, len(sample.stock_path))
            fig_path.add_trace(go.Scatter(
                x=t_axis, y=sample.stock_path,
                name="Stock Price", line=dict(width=2),
            ))
            fig_path.add_trace(go.Scatter(
                x=t_axis, y=sample.delta_path,
                name="Delta", yaxis="y2", line=dict(width=1.5, dash="dash"),
            ))
            fig_path.update_layout(
                title="Sample Hedging Path",
                xaxis_title="Time (years)",
                yaxis=dict(title="Stock Price"),
                yaxis2=dict(title="Delta", overlaying="y", side="right"),
                template="plotly_white",
            )
            st.plotly_chart(fig_path, use_container_width=True)


# ============================================================================
# TAB 5: Implied Volatility Surface
# ============================================================================
with tab_surface:
    st.header("Implied Volatility Surface")
    st.markdown("""
    Generate a synthetic IV surface using a simple volatility smile model.
    In practice, IVs come from market prices.
    """)

    scol1, scol2 = st.columns(2)
    with scol1:
        s_S = st.number_input("Spot Price", value=100.0, key="surf_S")
        s_r = st.number_input("Risk-free Rate", value=0.05, key="surf_r")
        s_base_vol = st.number_input("ATM Volatility", value=0.20, min_value=0.01, key="surf_base_vol")
    with scol2:
        s_skew = st.number_input("Skew (vol increase per 10% OTM)", value=0.03, key="surf_skew")
        s_term = st.number_input("Term structure slope", value=0.02, key="surf_term")

    if st.button("Generate Surface", key="gen_surface"):
        strikes = np.linspace(s_S * 0.7, s_S * 1.3, 25)
        maturities = np.linspace(0.05, 2.0, 20)

        # Synthetic smile: IV = base + skew * |log(K/S)| + term * sqrt(T)
        iv_grid = np.zeros((len(maturities), len(strikes)))
        for i, T_val in enumerate(maturities):
            for j, K_val in enumerate(strikes):
                moneyness = abs(math.log(K_val / s_S))
                iv_grid[i, j] = s_base_vol + s_skew * moneyness * 10 + s_term * (math.sqrt(T_val) - math.sqrt(0.25))

        fig_surf = go.Figure(data=[go.Surface(
            z=iv_grid * 100,
            x=strikes,
            y=maturities,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)"),
        )])
        fig_surf.update_layout(
            title="Implied Volatility Surface",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Maturity (years)",
                zaxis_title="IV (%)",
            ),
            height=700,
        )
        st.plotly_chart(fig_surf, use_container_width=True)


# ============================================================================
# TAB 6: Quiz Mode
# ============================================================================
with tab_quiz:
    st.header("Quiz Mode")
    st.markdown("Test your intuition about option prices and Greeks!")

    if "quiz_params" not in st.session_state:
        st.session_state.quiz_params = None
        st.session_state.quiz_answer = None

    if st.button("Generate New Question", key="new_quiz"):
        rng = np.random.default_rng()
        q_S = float(rng.choice([80, 90, 100, 110, 120]))
        q_K = float(rng.choice([80, 90, 95, 100, 105, 110, 120]))
        q_T = float(rng.choice([0.25, 0.5, 1.0]))
        q_r = 0.05
        q_sigma = float(rng.choice([0.15, 0.20, 0.25, 0.30, 0.40]))
        q_type = str(rng.choice(["call", "put"]))

        price = bs_price(q_S, q_K, q_T, q_r, q_sigma, q_type)
        greeks_val = all_greeks(q_S, q_K, q_T, q_r, q_sigma, q_type)

        st.session_state.quiz_params = {
            "S": q_S, "K": q_K, "T": q_T, "r": q_r,
            "sigma": q_sigma, "type": q_type,
        }
        st.session_state.quiz_answer = {"price": price, "greeks": greeks_val}

    if st.session_state.quiz_params is not None:
        p = st.session_state.quiz_params
        st.info(
            f"**{p['type'].title()} Option**: S={p['S']}, K={p['K']}, "
            f"T={p['T']}y, r={p['r']}, sigma={p['sigma']}"
        )

        guess = st.number_input("Your price guess:", min_value=0.0, step=0.5, key="quiz_guess")

        if st.button("Check Answer", key="check_quiz"):
            ans = st.session_state.quiz_answer
            actual = ans["price"]
            error = abs(guess - actual)
            pct_error = (error / actual * 100) if actual > 0 else 0

            if pct_error < 5:
                st.success(f"Excellent! Actual price: ${actual:.4f} (error: {pct_error:.1f}%)")
            elif pct_error < 15:
                st.warning(f"Close! Actual price: ${actual:.4f} (error: {pct_error:.1f}%)")
            else:
                st.error(f"Not quite. Actual price: ${actual:.4f} (error: {pct_error:.1f}%)")

            st.subheader("Full Answer")
            acol1, acol2 = st.columns(2)
            with acol1:
                st.metric("BS Price", f"${actual:.6f}")
            with acol2:
                g = ans["greeks"]
                for name, val in g.items():
                    st.text(f"{name:>6s}: {val:>12.8f}")


# Footer
st.markdown("---")
st.markdown(
    "*Options Pricing Trainer — educational tool for learning "
    "Black-Scholes, binomial trees, Monte Carlo simulation, and Greeks.*"
)
