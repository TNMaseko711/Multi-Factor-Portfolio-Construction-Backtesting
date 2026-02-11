# Executive Summary (Template)

## Strategy Overview
This project implements a long-only multi-factor equity strategy that combines value, momentum, quality, low-volatility, and size factors.

## Key Findings
- Fill after running `run_backtest.py` and reviewing `outputs/performance_summary.csv`.
- Compare against S&P 500 benchmark in `outputs/returns_series.csv`.

## Risk & Controls
- Name cap: 5%
- Sector cap: 25%
- Turnover cap: 40% per rebalance
- Transaction cost: configurable (default 15 bps)

## Attribution
- Factor contribution output: `outputs/factor_contribution.csv`
- Regime analysis output: `outputs/regime_attribution.csv`

## Recommendations
- Test weekly and monthly rebalance sensitivity.
- Incorporate point-in-time universe membership to reduce survivorship bias.
- Extend with optimizer overlays for production use.
