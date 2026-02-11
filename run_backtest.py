from __future__ import annotations

import argparse

import pandas as pd

from src.attribution import AttributionEngine
from src.backtest import Backtester
from src.data import DataLoader
from src.factors import FactorModel
from src.portfolio import PortfolioConstructor
from src.reporting import ReportExporter
from src.robustness import RobustnessAnalyzer

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "NVDA",
    "V",
    "UNH",
    "HD",
    "MA",
    "LLY",
    "AVGO",
    "COST",
    "MRK",
    "KO",
    "PEP",
]


def run_pipeline(
    start: str,
    end: str,
    rebalance: str,
    transaction_cost_bps: float,
    method: str,
):
    data_loader = DataLoader(DEFAULT_TICKERS)
    bundle = data_loader.build_bundle(start, end)

    factor_model = FactorModel()
    tech = factor_model.build_technical_factors(bundle.prices)
    fund = factor_model.build_fundamental_factors(
        bundle.fundamentals,
        dates=bundle.prices.index,
        market_caps=bundle.market_caps,
    )

    raw_factors = {**tech, **fund}
    std_factors = factor_model.combine_and_standardize(raw_factors)

    factor_weights = {
        "value": 0.25,
        "momentum": 0.25,
        "quality": 0.20,
        "low_vol": 0.15,
        "size": 0.10,
        "trend": 0.05,
    }

    score = factor_model.composite_score(std_factors, factor_weights)
    constructor = PortfolioConstructor()
    returns = bundle.prices.pct_change().fillna(0)
    weights = constructor.construct(score, returns, bundle.sectors, method=method)

    backtester = Backtester(transaction_cost_bps=transaction_cost_bps, periods_per_year=(12 if rebalance == "M" else 52))
    result = backtester.run(bundle.prices, bundle.benchmark, weights, rebalance=rebalance)

    attribution = AttributionEngine()
    factor_contrib = attribution.factor_contribution(result.weights, std_factors, result.portfolio_returns)
    regime = attribution.regime_attribution(result.portfolio_returns, result.benchmark_returns)

    robustness = RobustnessAnalyzer()
    mc = robustness.monte_carlo_ci(result.portfolio_returns)
    stress = robustness.stress_period_performance(
        result.portfolio_returns,
        windows={
            "covid_crash": ("2020-02-01", "2020-04-30"),
            "rate_shock_2022": ("2022-01-01", "2022-12-31"),
            "recent": ("2023-01-01", "2024-12-31"),
        },
    )

    exporter = ReportExporter()
    exporter.export_all(
        holdings=result.weights,
        transactions=result.transactions,
        metrics=result.metrics,
        portfolio_returns=result.portfolio_returns,
        benchmark_returns=result.benchmark_returns,
        factor_contrib=factor_contrib,
    )
    regime.to_csv("outputs/regime_attribution.csv")
    mc.to_csv("outputs/monte_carlo_ci.csv", index=False)
    stress.to_csv("outputs/stress_test.csv", index=False)

    print("=== Performance Summary ===")
    print(result.metrics.round(4))
    print("\n=== Regime Attribution ===")
    print(regime.round(4))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-factor portfolio backtest")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--rebalance", choices=["M", "W"], default="M")
    parser.add_argument("--transaction-cost-bps", type=float, default=15.0)
    parser.add_argument(
        "--method",
        choices=["equal_weighted", "score_weighted", "risk_parity"],
        default="score_weighted",
    )
    args = parser.parse_args()

    run_pipeline(
        start=args.start,
        end=args.end,
        rebalance=args.rebalance,
        transaction_cost_bps=args.transaction_cost_bps,
        method=args.method,
    )


if __name__ == "__main__":
    main()
