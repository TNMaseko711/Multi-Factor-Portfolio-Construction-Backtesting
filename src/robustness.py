from __future__ import annotations

import numpy as np
import pandas as pd


class RobustnessAnalyzer:
    def parameter_sweep(
        self,
        run_fn,
        param_grid: dict[str, list],
    ) -> pd.DataFrame:
        rows = []
        for rebalance in param_grid.get("rebalance", ["M"]):
            for tc in param_grid.get("transaction_cost_bps", [15.0]):
                result = run_fn(rebalance=rebalance, transaction_cost_bps=tc)
                rows.append(
                    {
                        "rebalance": rebalance,
                        "transaction_cost_bps": tc,
                        **result.metrics.to_dict(),
                    }
                )
        return pd.DataFrame(rows)

    def monte_carlo_ci(
        self,
        returns: pd.Series,
        n_sims: int = 1000,
        block_size: int = 6,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        r = returns.dropna().values
        n = len(r)
        if n == 0:
            return pd.DataFrame()

        sims = []
        for _ in range(n_sims):
            chunks = []
            while sum(len(c) for c in chunks) < n:
                start = rng.integers(0, max(1, n - block_size))
                chunks.append(r[start : start + block_size])
            sample = np.concatenate(chunks)[:n]
            sims.append((1 + sample).prod() - 1)

        sims = np.array(sims)
        return pd.DataFrame(
            {
                "metric": ["total_return"],
                "p05": [np.quantile(sims, 0.05)],
                "p50": [np.quantile(sims, 0.50)],
                "p95": [np.quantile(sims, 0.95)],
            }
        )

    def stress_period_performance(
        self,
        returns: pd.Series,
        windows: dict[str, tuple[str, str]],
    ) -> pd.DataFrame:
        rows = []
        for name, (start, end) in windows.items():
            s = returns.loc[start:end]
            rows.append(
                {
                    "window": name,
                    "start": start,
                    "end": end,
                    "total_return": (1 + s).prod() - 1 if not s.empty else np.nan,
                    "vol": s.std() * np.sqrt(12) if not s.empty else np.nan,
                    "hit_rate": (s > 0).mean() if not s.empty else np.nan,
                }
            )
        return pd.DataFrame(rows)
