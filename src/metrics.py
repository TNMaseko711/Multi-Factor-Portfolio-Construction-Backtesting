from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return total ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 12) -> float:
    excess = returns - rf / periods_per_year
    vol = annualized_volatility(excess, periods_per_year)
    return np.nan if vol == 0 else annualized_return(excess, periods_per_year) / vol


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 12) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0].std() * np.sqrt(periods_per_year)
    return np.nan if downside == 0 else annualized_return(excess, periods_per_year) / downside


def drawdown_stats(returns: pd.Series) -> tuple[float, int, pd.Series]:
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    max_dd = dd.min()

    duration = 0
    max_duration = 0
    for v in dd:
        if v < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return float(max_dd), int(max_duration), dd


def var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    losses = -returns.dropna()
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean() if (losses >= var).any() else var
    return float(var), float(cvar)


def information_ratio(strategy: pd.Series, benchmark: pd.Series, periods_per_year: int = 12) -> float:
    aligned = pd.concat([strategy, benchmark], axis=1).dropna()
    if aligned.empty:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std() * np.sqrt(periods_per_year)
    if te == 0:
        return np.nan
    return active.mean() * periods_per_year / te
