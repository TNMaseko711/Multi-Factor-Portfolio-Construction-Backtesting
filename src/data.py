from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class DataBundle:
    prices: pd.DataFrame
    benchmark: pd.Series
    fundamentals: pd.DataFrame
    sectors: pd.Series
    market_caps: pd.DataFrame


class DataLoader:
    """Loads market and fundamental data for multi-factor research."""

    def __init__(self, tickers: Iterable[str], benchmark: str = "^GSPC") -> None:
        self.tickers = list(dict.fromkeys(tickers))
        self.benchmark = benchmark

    def load_prices(self, start: str, end: str) -> pd.DataFrame:
        px = yf.download(
            self.tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(name=self.tickers[0])
        return px.sort_index().dropna(how="all")

    def load_benchmark(self, start: str, end: str) -> pd.Series:
        bm = yf.download(
            self.benchmark,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )["Close"]
        return bm.sort_index().rename("benchmark")

    def load_fundamentals_snapshot(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        rows: list[dict] = []
        sectors: dict[str, str] = {}
        mkt_caps: dict[str, float] = {}

        for ticker in self.tickers:
            info = yf.Ticker(ticker).info
            rows.append(
                {
                    "ticker": ticker,
                    "pe": info.get("trailingPE", np.nan),
                    "pb": info.get("priceToBook", np.nan),
                    "roe": info.get("returnOnEquity", np.nan),
                    "debt_to_equity": info.get("debtToEquity", np.nan),
                    "revenue_growth": info.get("revenueGrowth", np.nan),
                    "roa": info.get("returnOnAssets", np.nan),
                    "gross_margins": info.get("grossMargins", np.nan),
                }
            )
            sectors[ticker] = info.get("sector", "Unknown")
            mkt_caps[ticker] = info.get("marketCap", np.nan)

        fundamentals = pd.DataFrame(rows).set_index("ticker")
        sector_series = pd.Series(sectors, name="sector")
        mkt_caps_df = pd.DataFrame([mkt_caps])
        mkt_caps_df.index = pd.Index([pd.Timestamp.utcnow().normalize()], name="date")
        return fundamentals, sector_series, mkt_caps_df

    def build_bundle(self, start: str, end: str) -> DataBundle:
        prices = self.load_prices(start, end)
        benchmark = self.load_benchmark(start, end)
        fundamentals, sectors, market_caps = self.load_fundamentals_snapshot()

        return DataBundle(
            prices=prices,
            benchmark=benchmark,
            fundamentals=fundamentals,
            sectors=sectors,
            market_caps=market_caps,
        )
