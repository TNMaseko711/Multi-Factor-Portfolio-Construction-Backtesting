from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportExporter:
    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        holdings: pd.DataFrame,
        transactions: pd.DataFrame,
        metrics: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_contrib: pd.DataFrame | None = None,
    ) -> None:
        holdings.to_csv(self.output_dir / "holdings_history.csv")
        transactions.to_csv(self.output_dir / "transactions.csv", index=False)
        metrics.to_csv(self.output_dir / "performance_summary.csv", header=["value"])
        pd.concat(
            [
                portfolio_returns.rename("portfolio"),
                benchmark_returns.rename("benchmark"),
            ],
            axis=1,
        ).to_csv(self.output_dir / "returns_series.csv")

        if factor_contrib is not None:
            factor_contrib.to_csv(self.output_dir / "factor_contribution.csv")
