"""Command-line interface for the options pricing trainer.

Usage:
    python -m src.cli price --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --type call
    python -m src.cli greeks --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --type call
    python -m src.cli iv --market-price 5.0 --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --type call
    python -m src.cli binomial --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --steps 200 --type put --american
    python -m src.cli mc --spot 100 --strike 105 --expiry 0.5 --rate 0.05 --vol 0.2 --paths 100000 --type call --exotic european
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="options-pricing-trainer",
        description="Interactive options pricing education tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- Common arguments ---
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--spot", "-S", type=float, required=True, help="Spot price")
        p.add_argument("--strike", "-K", type=float, required=True, help="Strike price")
        p.add_argument("--expiry", "-T", type=float, required=True, help="Time to expiry (years)")
        p.add_argument("--rate", "-r", type=float, default=0.05, help="Risk-free rate")
        p.add_argument("--vol", type=float, default=0.2, help="Volatility")
        p.add_argument("--type", dest="option_type", choices=["call", "put"], default="call")

    # Price
    p_price = sub.add_parser("price", help="Price an option using Black-Scholes")
    add_common(p_price)

    # Greeks
    p_greeks = sub.add_parser("greeks", help="Compute all BS Greeks")
    add_common(p_greeks)

    # Implied vol
    p_iv = sub.add_parser("iv", help="Compute implied volatility")
    p_iv.add_argument("--market-price", type=float, required=True)
    p_iv.add_argument("--spot", "-S", type=float, required=True)
    p_iv.add_argument("--strike", "-K", type=float, required=True)
    p_iv.add_argument("--expiry", "-T", type=float, required=True)
    p_iv.add_argument("--rate", "-r", type=float, default=0.05)
    p_iv.add_argument("--type", dest="option_type", choices=["call", "put"], default="call")

    # Binomial
    p_binom = sub.add_parser("binomial", help="Price via binomial tree")
    add_common(p_binom)
    p_binom.add_argument("--steps", "-N", type=int, default=200)
    p_binom.add_argument("--american", action="store_true")

    # Monte Carlo
    p_mc = sub.add_parser("mc", help="Price via Monte Carlo")
    add_common(p_mc)
    p_mc.add_argument("--paths", type=int, default=100_000)
    p_mc.add_argument("--exotic", choices=["european", "asian", "barrier", "lookback"], default="european")
    p_mc.add_argument("--barrier-level", type=float, default=None)
    p_mc.add_argument("--barrier-type", choices=["up-and-out", "down-and-out", "up-and-in", "down-and-in"], default=None)
    p_mc.add_argument("--seed", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "price":
        from src.core.black_scholes import bs_price

        price = bs_price(args.spot, args.strike, args.expiry, args.rate, args.vol, args.option_type)
        print(f"Black-Scholes {args.option_type} price: {price:.8f}")

    elif args.command == "greeks":
        from src.core.greeks import all_greeks

        g = all_greeks(args.spot, args.strike, args.expiry, args.rate, args.vol, args.option_type)
        print(f"Greeks for {args.option_type} (S={args.spot}, K={args.strike}, T={args.expiry}, r={args.rate}, sigma={args.vol}):")
        for name, val in g.items():
            print(f"  {name:>6s}: {val:>12.8f}")

    elif args.command == "iv":
        from src.core.implied_vol import implied_volatility

        iv = implied_volatility(
            args.market_price, args.spot, args.strike, args.expiry,
            args.rate, args.option_type,
        )
        print(f"Implied volatility: {iv:.6f} ({iv * 100:.2f}%)")

    elif args.command == "binomial":
        from src.core.binomial import binomial_price

        price = binomial_price(
            args.spot, args.strike, args.expiry, args.rate, args.vol,
            N=args.steps, option_type=args.option_type, american=args.american,
        )
        style = "American" if args.american else "European"
        print(f"Binomial ({style}) {args.option_type} price ({args.steps} steps): {price:.8f}")

    elif args.command == "mc":
        from src.core.monte_carlo import monte_carlo_price

        result = monte_carlo_price(
            args.spot, args.strike, args.expiry, args.rate, args.vol,
            option_type=args.option_type,
            exotic=args.exotic,
            barrier=args.barrier_level,
            barrier_type=args.barrier_type,
            n_paths=args.paths,
            seed=args.seed,
        )
        print(f"Monte Carlo ({args.exotic}) {args.option_type} price: {result['price']:.6f}")
        print(f"  Standard error: {result['std_error']:.6f}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
