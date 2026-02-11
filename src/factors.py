from __future__ import annotations

import numpy as np
import pandas as pd


class FactorModel:
    def __init__(self, winsor_pct: float = 0.01) -> None:
        self.winsor_pct = winsor_pct

    @staticmethod
    def _zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean(axis=1)
        std = df.std(axis=1).replace(0, np.nan)
        return df.sub(mean, axis=0).div(std, axis=0)

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        low = df.quantile(self.winsor_pct, axis=1)
        high = df.quantile(1 - self.winsor_pct, axis=1)
        return df.clip(lower=low, upper=high, axis=0)

    def build_technical_factors(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        returns = prices.pct_change()

        mom_12_1 = prices.shift(21).pct_change(252)
        vol_63 = returns.rolling(63).std() * np.sqrt(252)
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        trend = (sma_50 / sma_200) - 1

        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return {
            "momentum": mom_12_1,
            "low_vol": -vol_63,
            "trend": trend,
            "rsi": -(rsi - 50).abs(),
        }

    def build_fundamental_factors(
        self,
        fundamentals: pd.DataFrame,
        dates: pd.Index,
        market_caps: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        f = fundamentals.copy()

        value = (
            (1 / f["pb"]).rename("book_to_market")
            + (1 / f["pe"]).rename("earnings_yield")
        ) / 2

        quality = (
            f[["roe", "roa", "gross_margins"]].mean(axis=1)
            - f["debt_to_equity"].rank(pct=True)
        )

        size = -np.log(market_caps.iloc[0].replace(0, np.nan))

        value_df = pd.DataFrame([value] * len(dates), index=dates)
        quality_df = pd.DataFrame([quality] * len(dates), index=dates)
        size_df = pd.DataFrame([size] * len(dates), index=dates)

        return {
            "value": value_df,
            "quality": quality_df,
            "size": size_df,
        }

    def combine_and_standardize(self, raw_factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        cleaned: dict[str, pd.DataFrame] = {}
        for name, frame in raw_factors.items():
            c = frame.copy()
            c = c.replace([np.inf, -np.inf], np.nan)
            c = c.apply(lambda row: row.fillna(row.median()), axis=1)
            c = self._winsorize(c)
            cleaned[name] = self._zscore_cross_section(c)
        return cleaned

    def composite_score(
        self,
        standardized_factors: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> pd.DataFrame:
        shared_dates = sorted(set.intersection(*(set(f.index) for f in standardized_factors.values())))
        score = None
        for name, frame in standardized_factors.items():
            w = weights.get(name, 0.0)
            aligned = frame.loc[shared_dates]
            score = aligned * w if score is None else score.add(aligned * w, fill_value=0)
        return score.sort_index()
