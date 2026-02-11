from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


class AttributionEngine:
    def factor_contribution(
        self,
        weights: pd.DataFrame,
        factor_scores: dict[str, pd.DataFrame],
        portfolio_returns: pd.Series,
    ) -> pd.DataFrame:
        exposures = {}
        for name, scores in factor_scores.items():
            common = weights.index.intersection(scores.index)
            aligned_w = weights.reindex(common).fillna(0)
            aligned_s = scores.reindex(common).fillna(0)
            exposures[name] = (aligned_w * aligned_s).sum(axis=1)

        expo_df = pd.DataFrame(exposures).reindex(portfolio_returns.index).fillna(method="ffill").fillna(0)
        contrib = expo_df.mul(portfolio_returns, axis=0)
        contrib.columns = [f"{c}_contribution" for c in contrib.columns]
        return contrib

    def fama_french_regression(
        self,
        portfolio_returns: pd.Series,
        ff_factors: pd.DataFrame,
    ) -> pd.Series:
        data = pd.concat([portfolio_returns.rename("strategy"), ff_factors], axis=1).dropna()
        if data.empty:
            return pd.Series(dtype=float)
        y = data["strategy"]
        x = sm.add_constant(data.drop(columns=["strategy"]))
        model = sm.OLS(y, x).fit()
        out = model.params.rename(lambda c: f"beta_{c}")
        out["r_squared"] = model.rsquared
        out["alpha_tstat"] = model.tvalues.get("const", np.nan)
        return out

    def regime_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:
        df = pd.concat([portfolio_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        df["regime"] = np.where(df["benchmark"] >= 0, "bull", "bear")
        summary = df.groupby("regime").agg(
            periods=("strategy", "size"),
            mean_return=("strategy", "mean"),
            vol=("strategy", "std"),
            hit_rate=("strategy", lambda x: (x > 0).mean()),
        )
        return summary
