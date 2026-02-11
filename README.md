# Multi-Factor Portfolio Construction & Backtesting

A modular quantitative finance research project for building and backtesting a **long-only multi-factor equity strategy** with risk controls, attribution, and robustness tests.

## Features
- Universe handling for S&P 500 or custom ticker lists
- Factor model with value, momentum, quality, low-volatility, and size signals
- Composite ranking with configurable factor weights
- Portfolio construction:
  - Equal-weight top bucket
  - Score-weighted top bucket
  - Risk-parity-inspired inverse-vol weighting
- Constraint-aware position sizing
  - Max single-name weight
  - Sector cap
  - Turnover cap
- Walk-forward backtest with transaction costs and benchmark comparison
- Risk analytics: volatility, VaR/CVaR, max drawdown, drawdown duration
- Attribution:
  - Factor contribution decomposition
  - Fama-French style regression (if factor data supplied)
  - Regime split analysis
- Sensitivity and robustness utilities:
  - Parameter sweeps
  - Block bootstrap Monte Carlo confidence intervals
  - Stress-period slicing
- Output exports:
  - Holdings history CSV
  - Transactions CSV
  - Performance summary CSV

## Project Structure
```
.
├── notebooks/
│   └── multifactor_research.ipynb
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── attribution.py
│   ├── backtest.py
│   ├── data.py
│   ├── factors.py
│   ├── metrics.py
│   ├── portfolio.py
│   ├── reporting.py
│   └── robustness.py
├── run_backtest.py
├── requirements.txt
└── README.md
```

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run backtest:
   ```bash
   python run_backtest.py --start 2015-01-01 --end 2024-12-31 --rebalance M
   ```
3. Review generated CSVs in `outputs/`.

## Notes
- The project uses `yfinance` for market and fundamental data.
- Fundamental fields can be sparse across history via free APIs. The pipeline handles missing data with robust cross-sectional median imputation and winsorization.
- To reduce survivorship bias, pass a point-in-time ticker list for each rebalance date where available.

## Optional Extensions
- Streamlit dashboard from exported artifacts
- Kelly sizing overlay
- Regime-switching risk-on/risk-off signal
- Mean-variance and Black-Litterman allocation comparison
