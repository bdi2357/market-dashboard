"""
Simple backtesting engine.

Strategy: price crosses N-day moving average (long only).
Position sizing: Kelly fraction and volatility-targeting.
Includes stress test (2008, 2020, 2022) and bootstrap Sharpe CI.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


STRESS_PERIODS = {
    "GFC 2008-09": ("2008-01-01", "2009-03-31"),
    "COVID 2020": ("2020-01-15", "2020-04-30"),
    "Rate Shock 2022": ("2022-01-01", "2022-12-31"),
}


def ma_crossover_signals(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Long signal when fast MA > slow MA.
    Returns series of {1, 0} positions.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    return signal.shift(1).fillna(0)  # avoid lookahead


def kelly_fraction(returns: pd.Series, max_leverage: float = 1.0) -> float:
    """
    Full Kelly fraction = (μ - rf) / σ²  (simplified, rf=0).
    Capped at max_leverage to limit over-betting.
    Half-Kelly is often preferred in practice.
    """
    mu = returns.mean() * 252
    sigma2 = returns.var() * 252
    kelly = mu / sigma2 if sigma2 > 0 else 0
    return float(np.clip(kelly, 0, max_leverage))


def vol_target_sizing(
    returns: pd.Series,
    target_vol: float = 0.15,
    lookback: int = 21,
) -> pd.Series:
    """
    Scale position so expected portfolio vol = target_vol (annualized).
    position_t = target_vol / (realized_vol_t * sqrt(252))
    Capped at 1.0 (no leverage).
    """
    realized = returns.rolling(lookback).std() * np.sqrt(252)
    sizing = (target_vol / realized).clip(upper=1.0).fillna(0)
    return sizing


def run_backtest(
    prices: pd.Series,
    fast_ma: int = 20,
    slow_ma: int = 50,
    sizing: str = "vol_target",
    target_vol: float = 0.15,
) -> dict:
    """
    Run MA crossover backtest.

    Returns:
        equity: portfolio equity curve (starting at 1.0)
        returns: daily returns
        positions: position series
        rolling_sharpe: 63-day rolling Sharpe
        rolling_var95: 63-day rolling 95% VaR
        summary: dict of performance stats
    """
    r = prices.pct_change().fillna(0)
    signals = ma_crossover_signals(prices, fast_ma, slow_ma)

    if sizing == "vol_target":
        size = vol_target_sizing(r, target_vol=target_vol)
    elif sizing == "kelly":
        # Rolling Kelly
        kelly = r.rolling(63).apply(
            lambda x: kelly_fraction(pd.Series(x)), raw=False
        ).fillna(0)
        size = kelly
    else:
        size = pd.Series(1.0, index=r.index)

    positions = signals * size
    strat_returns = positions.shift(1).fillna(0) * r

    equity = (1 + strat_returns).cumprod()

    rolling_sharpe = (
        strat_returns.rolling(63).mean() / strat_returns.rolling(63).std() * np.sqrt(252)
    )
    rolling_var95 = strat_returns.rolling(63).quantile(0.05).abs()

    # Summary stats
    ann_ret = (equity.iloc[-1] ** (252 / len(equity))) - 1
    ann_vol = strat_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum_ret = equity - 1
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1)
    max_dd = dd.min()

    summary = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": ann_ret / abs(max_dd) if max_dd != 0 else np.nan,
        "total_return": equity.iloc[-1] - 1,
        "n_trades": int((signals.diff().abs() > 0).sum()),
    }

    return {
        "equity": equity,
        "returns": strat_returns,
        "positions": positions,
        "rolling_sharpe": rolling_sharpe,
        "rolling_var95": rolling_var95,
        "drawdown": dd,
        "summary": summary,
    }


def stress_test(prices: pd.Series, **backtest_kwargs) -> pd.DataFrame:
    """
    Run backtest over 2008, 2020, 2022 stress periods.
    Returns DataFrame with period stats.
    """
    rows = []
    for label, (start, end) in STRESS_PERIODS.items():
        chunk = prices.loc[start:end]
        if len(chunk) < 20:
            rows.append({"period": label, "available_days": len(chunk), "note": "insufficient data"})
            continue
        result = run_backtest(chunk, **backtest_kwargs)
        s = result["summary"]
        rows.append({
            "period": label,
            "total_return": s["total_return"],
            "ann_vol": s["ann_vol"],
            "max_drawdown": s["max_drawdown"],
            "sharpe": s["sharpe"],
            "available_days": len(chunk),
        })
    return pd.DataFrame(rows).set_index("period")


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Parametric bootstrap confidence interval for annualized Sharpe ratio.
    Resamples daily returns with replacement n_boot times.
    Returns mean, lower, upper bounds at given CI level.
    """
    rng = np.random.default_rng(random_state)
    r = returns.dropna().values
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(r, size=len(r), replace=True)
        ann_r = sample.mean() * 252
        ann_v = sample.std() * np.sqrt(252)
        sharpes.append(ann_r / ann_v if ann_v > 0 else np.nan)
    sharpes = np.array(sharpes)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.nanmean(sharpes)),
        "lower": float(np.nanpercentile(sharpes, alpha * 100)),
        "upper": float(np.nanpercentile(sharpes, (1 - alpha) * 100)),
        "ci": ci,
    }
