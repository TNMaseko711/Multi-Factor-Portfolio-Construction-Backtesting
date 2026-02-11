from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import (
    annualized_return,
    annualized_volatility,
    drawdown_stats,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    var_cvar,
)


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transactions: pd.DataFrame
    metrics: pd.Series


class Backtester:
    def __init__(self, transaction_cost_bps: float = 15.0, periods_per_year: int = 12):
        self.transaction_cost_bps = transaction_cost_bps
        self.periods_per_year = periods_per_year

    def run(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series,
        weights: pd.DataFrame,
        rebalance: str = "M",
    ) -> BacktestResult:
        returns = prices.pct_change().fillna(0)
        benchmark_rets = benchmark.pct_change().fillna(0)

        rebal_dates = returns.resample(rebalance).last().index
        w = weights.reindex(rebal_dates).fillna(0)

        monthly_returns = returns.resample(rebalance).apply(lambda x: (1 + x).prod() - 1)
        monthly_bm = benchmark_rets.resample(rebalance).apply(lambda x: (1 + x).prod() - 1)

        aligned_w = w.reindex(monthly_returns.index).ffill().fillna(0)
        gross = (aligned_w.shift(1).fillna(0) * monthly_returns).sum(axis=1)

        turnover = aligned_w.diff().abs().sum(axis=1).fillna(aligned_w.abs().sum(axis=1))
        tx_cost = turnover * (self.transaction_cost_bps / 10_000)
        net = gross - tx_cost

        tx = pd.DataFrame(
            {
                "date": aligned_w.index,
                "turnover": turnover.values,
                "transaction_cost": tx_cost.values,
            }
        )

        max_dd, max_dd_dur, _ = drawdown_stats(net)
        var95, cvar95 = var_cvar(net, 0.95)
        var99, cvar99 = var_cvar(net, 0.99)

        metrics = pd.Series(
            {
                "total_return": (1 + net).prod() - 1,
                "cagr": annualized_return(net, self.periods_per_year),
                "volatility": annualized_volatility(net, self.periods_per_year),
                "sharpe": sharpe_ratio(net, periods_per_year=self.periods_per_year),
                "sortino": sortino_ratio(net, periods_per_year=self.periods_per_year),
                "max_drawdown": max_dd,
                "max_drawdown_duration": max_dd_dur,
                "var_95": var95,
                "cvar_95": cvar95,
                "var_99": var99,
                "cvar_99": cvar99,
                "information_ratio": information_ratio(net, monthly_bm, self.periods_per_year),
                "avg_turnover": turnover.mean(),
            }
        )

        return BacktestResult(
            portfolio_returns=net,
            benchmark_returns=monthly_bm.reindex(net.index).fillna(0),
            weights=aligned_w,
            turnover=turnover,
            transactions=tx,
            metrics=metrics,
        )
