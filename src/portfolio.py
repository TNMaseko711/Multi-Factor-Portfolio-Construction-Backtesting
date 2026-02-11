from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioConstructor:
    def __init__(
        self,
        top_quantile: float = 0.2,
        max_weight: float = 0.05,
        sector_cap: float = 0.25,
        turnover_cap: float = 0.4,
    ) -> None:
        self.top_quantile = top_quantile
        self.max_weight = max_weight
        self.sector_cap = sector_cap
        self.turnover_cap = turnover_cap

    def _select_universe(self, scores_row: pd.Series) -> pd.Index:
        cutoff = scores_row.quantile(1 - self.top_quantile)
        return scores_row[scores_row >= cutoff].dropna().index

    def _apply_sector_cap(self, weights: pd.Series, sectors: pd.Series) -> pd.Series:
        if weights.empty:
            return weights
        w = weights.copy()
        sec = sectors.reindex(w.index).fillna("Unknown")
        for _ in range(10):
            sector_totals = w.groupby(sec).sum()
            breaches = sector_totals[sector_totals > self.sector_cap]
            if breaches.empty:
                break
            for sector, total in breaches.items():
                members = sec[sec == sector].index
                scale = self.sector_cap / total
                w.loc[members] *= scale
            leftover = 1 - w.sum()
            eligible = w[w < self.max_weight].index
            if len(eligible) > 0 and leftover > 0:
                w.loc[eligible] += leftover / len(eligible)
                w = w.clip(upper=self.max_weight)
                w /= w.sum()
        return w / w.sum()

    def _turnover_limited(self, target: pd.Series, prev: pd.Series | None) -> pd.Series:
        if prev is None:
            return target
        prev = prev.reindex(target.index).fillna(0)
        diff = target - prev
        turnover = diff.abs().sum()
        if turnover <= self.turnover_cap:
            return target
        scale = self.turnover_cap / turnover
        adjusted = prev + diff * scale
        adjusted = adjusted.clip(lower=0)
        if adjusted.sum() == 0:
            return target
        return adjusted / adjusted.sum()

    def construct(
        self,
        score: pd.DataFrame,
        returns: pd.DataFrame,
        sectors: pd.Series,
        method: str = "score_weighted",
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        prev = None

        for dt in score.index:
            row = score.loc[dt].dropna()
            selected = self._select_universe(row)
            if selected.empty:
                continue

            if method == "equal_weighted":
                target = pd.Series(1 / len(selected), index=selected)
            elif method == "risk_parity":
                vol = returns[selected].loc[:dt].tail(63).std().replace(0, np.nan)
                inv_vol = 1 / vol
                target = inv_vol / inv_vol.sum()
            else:
                raw = row[selected] - row[selected].min()
                if raw.sum() == 0:
                    target = pd.Series(1 / len(selected), index=selected)
                else:
                    target = raw / raw.sum()

            target = target.clip(upper=self.max_weight)
            target = target / target.sum()
            target = self._apply_sector_cap(target, sectors)
            target = self._turnover_limited(target, prev)

            weights.loc[dt, target.index] = target.values
            prev = weights.loc[dt]

        return weights
