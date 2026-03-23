# Options Pricing Trainer

Interactive options pricing education tool implementing Black-Scholes, binomial trees, Monte Carlo simulation, Greeks analytics, implied volatility computation, multi-leg strategy payoff diagrams, and delta hedging simulation.

## Features

- **Black-Scholes Pricing** — Closed-form European call/put pricing
- **CRR Binomial Tree** — American and European option pricing with early exercise
- **Monte Carlo Simulation** — Path simulation for vanilla and exotic options (Asian, barrier, lookback)
- **Greeks Suite** — Analytic (BS) and numerical (finite-difference) delta, gamma, vega, theta, rho
- **Implied Volatility** — Newton-Raphson and Brent's method to invert the BS formula
- **Payoff Diagram Builder** — Multi-leg strategies with breakeven analysis
- **Delta Hedging Simulator** — Dynamic hedging P/L distribution with vol mismatch and transaction costs
- **Interactive Streamlit UI** — Pricer, payoff diagrams, binomial tree visualisation, IV surface, quiz mode

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the interactive UI
streamlit run src/viz/app.py

# Run tests
pytest tests/ -v

# CLI usage
python -m src.cli price --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --type call
python -m src.cli greeks --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --type call
python -m src.cli iv --market-price 5.0 --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --type call
python -m src.cli binomial --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --steps 200 --type put --american
python -m src.cli mc --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --paths 100000 --type call --exotic asian

# Run the demo
python examples/demo.py
```

## Docker

```bash
docker build -t options-pricing-trainer .
docker run -p 8501:8501 options-pricing-trainer
```

## Theory

### Black-Scholes Model

The Black-Scholes formula prices European options under the assumption of log-normal stock prices, constant volatility, no dividends, and continuous trading.

**Call price:**

$$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

**Put price:**

$$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

where:

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) \cdot T}{\sigma \sqrt{T}}$$

$$d_2 = d_1 - \sigma \sqrt{T}$$

- $S$ = spot price, $K$ = strike, $T$ = time to expiry, $r$ = risk-free rate, $\sigma$ = volatility
- $N(\cdot)$ = standard normal CDF

### Binomial Model (CRR)

The Cox-Ross-Rubinstein model discretises price movement into up/down steps:

$$u = e^{\sigma \sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad p = \frac{e^{r \Delta t} - d}{u - d}$$

Option values are computed by backward induction. For American options, at each node:

$$V_{\text{node}} = \max\left(\text{exercise value},\; e^{-r\Delta t}\left[p \cdot V_{\text{up}} + (1-p) \cdot V_{\text{down}}\right]\right)$$

### Monte Carlo Simulation

Stock paths are simulated under geometric Brownian motion:

$$S(t + \Delta t) = S(t) \cdot \exp\left[\left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} \cdot Z\right]$$

where $Z \sim \mathcal{N}(0,1)$. The option price is the discounted expected payoff:

$$V = e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N} \text{payoff}(S_T^{(i)})$$

Supported exotic payoffs:
- **Asian**: payoff based on arithmetic average price
- **Barrier**: knock-in/knock-out based on price path extremes
- **Lookback**: payoff based on path minimum or maximum

### Greeks

The option Greeks measure sensitivity to input parameters:

| Greek | Formula (Call) | Interpretation |
|-------|----------------|----------------|
| **Delta** ($\Delta$) | $N(d_1)$ | Price sensitivity to spot |
| **Gamma** ($\Gamma$) | $\frac{n(d_1)}{S\sigma\sqrt{T}}$ | Rate of change of delta |
| **Vega** ($\mathcal{V}$) | $S \cdot n(d_1) \cdot \sqrt{T}$ | Sensitivity to volatility |
| **Theta** ($\Theta$) | $-\frac{S \cdot n(d_1) \cdot \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$ | Time decay |
| **Rho** ($\rho$) | $KTe^{-rT}N(d_2)$ | Sensitivity to interest rate |

Key relationships:
- $\Delta_{\text{call}} - \Delta_{\text{put}} = 1$
- Put-call parity: $C - P = S - Ke^{-rT}$
- $\Gamma$ is the same for calls and puts

### Implied Volatility

Given a market price $V_{\text{market}}$, solve for $\sigma$:

$$\text{BS}(S, K, T, r, \sigma) = V_{\text{market}}$$

Solved using Brent's method (robust) or Newton-Raphson (fast, using analytic vega):

$$\sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - V_{\text{market}}}{\text{vega}(\sigma_n)}$$

### Delta Hedging

The delta hedging simulator sells an option and dynamically rebalances a stock position to remain delta-neutral:

1. Sell option, receive premium $V_0$
2. At each rebalance: hold $\Delta_t$ shares, finance at rate $r$
3. At expiry: settle the option and liquidate shares
4. P/L = final cash position

With perfect continuous hedging and correct vol forecast, P/L = 0. Discrete rebalancing and vol mismatch create hedging error.

## Project Structure

```
options-pricing-trainer/
├── src/
│   ├── core/
│   │   ├── black_scholes.py   # BS call/put pricing
│   │   ├── binomial.py        # CRR binomial tree
│   │   ├── monte_carlo.py     # MC simulation (vanilla + exotics)
│   │   ├── greeks.py          # Analytic + numerical Greeks
│   │   ├── implied_vol.py     # IV solver (Brent + Newton)
│   │   ├── payoff.py          # Multi-leg strategy builder
│   │   └── delta_hedge.py     # Delta hedging simulator
│   ├── viz/
│   │   └── app.py             # Streamlit interactive UI
│   └── cli.py                 # Command-line interface
├── tests/
│   ├── test_black_scholes.py  # BS price verification
│   ├── test_binomial.py       # Binomial convergence + American >= European
│   ├── test_greeks.py         # Analytic vs numerical Greeks
│   ├── test_implied_vol.py    # IV round-trip tests
│   └── test_parity.py        # Put-call parity + cross-model consistency
├── examples/
│   └── demo.py                # Full feature demo
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .github/workflows/ci.yml
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_parity.py -v
```

Tests verify:
- BS prices match known values to 6+ decimal places
- Put-call parity: $C - P = S - Ke^{-rT}$ to machine precision
- Greeks relationships: $\Delta_{\text{call}} - \Delta_{\text{put}} = 1$
- American option prices $\geq$ European option prices
- Binomial converges to BS as steps increase
- Monte Carlo converges to BS within standard error bounds
- Implied volatility round-trips perfectly

## License

MIT
