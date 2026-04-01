"""
Risk metrics engine.

All annualization uses sqrt(252) for daily returns.
Returns are computed as log returns for volatility, simple returns elsewhere.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


# ── REALIZED RISK ─────────────────────────────────────────────────────────────

def rolling_realized_vol(prices: pd.Series, windows=(21, 63, 252)) -> pd.DataFrame:
    """
    Annualized realized volatility for each window.
    Uses log returns × sqrt(252).
    """
    lr = _log_returns(prices)
    result = {}
    for w in windows:
        result[f"vol_{w}d"] = lr.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(result, index=lr.index)


def vol_regime(vol_series: pd.Series, lookback: int = 504) -> pd.Series:
    """
    For each date, classify current vol vs its own 2-year rolling history.
    Returns: 'low' (bottom 25%), 'normal' (25-75%), 'high' (75-95%), 'extreme' (>95%)
    """
    def classify(x):
        window = vol_series.loc[:x.index[-1]].tail(lookback).dropna()
        if len(window) < 30:
            return "unknown"
        p = stats.percentileofscore(window, x.iloc[-1]) / 100
        if p < 0.25:
            return "low"
        elif p < 0.75:
            return "normal"
        elif p < 0.95:
            return "high"
        else:
            return "extreme"

    # Vectorized approximation using rolling percentile rank
    ranks = vol_series.rolling(lookback, min_periods=30).apply(
        lambda w: stats.percentileofscore(w, w.iloc[-1]) / 100, raw=False
    )
    labels = pd.cut(
        ranks,
        bins=[-np.inf, 0.25, 0.75, 0.95, np.inf],
        labels=["low", "normal", "high", "extreme"],
    )
    return labels


def max_drawdown(prices: pd.Series) -> dict:
    """
    Returns:
        max_dd: maximum drawdown (negative fraction)
        dd_start, dd_end: dates of drawdown
        dd_duration: calendar days of drawdown
        recovery_time: calendar days from trough to recovery (None if not recovered)
        dd_series: full drawdown time series
    """
    cum = prices / prices.iloc[0]
    roll_max = cum.cummax()
    dd = cum / roll_max - 1

    trough_idx = dd.idxmin()
    max_dd = dd.min()
    peak_idx = roll_max.loc[:trough_idx].idxmax()

    # Recovery: first date after trough where price >= peak
    after_trough = cum.loc[trough_idx:]
    peak_val = cum.loc[peak_idx]
    recovered = after_trough[after_trough >= peak_val]
    recovery_date = recovered.index[0] if not recovered.empty else None

    dd_duration = (trough_idx - peak_idx).days
    recovery_time = (recovery_date - trough_idx).days if recovery_date else None

    return {
        "max_drawdown": max_dd,
        "dd_start": peak_idx,
        "dd_end": trough_idx,
        "dd_duration_days": dd_duration,
        "recovery_days": recovery_time,
        "dd_series": dd,
    }


def calmar_ratio(prices: pd.Series) -> float:
    """Annualized return / |max drawdown|. Higher = better."""
    r = _simple_returns(prices)
    ann_return = (1 + r.mean()) ** 252 - 1
    mdd = abs(max_drawdown(prices)["max_drawdown"])
    return ann_return / mdd if mdd > 0 else np.nan


def sortino_ratio(prices: pd.Series, risk_free: float = 0.0) -> float:
    """
    (Annualized return - rf) / downside deviation.
    Downside deviation uses returns below rf/252.
    """
    r = _simple_returns(prices)
    ann_return = (1 + r.mean()) ** 252 - 1
    daily_rf = risk_free / 252
    downside = r[r < daily_rf] - daily_rf
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    return (ann_return - risk_free) / downside_std if downside_std > 0 else np.nan


def ulcer_index(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Rolling Ulcer Index = sqrt(mean of squared % drawdowns over window).
    Formula: UI = sqrt( sum((DD_i/peak_i)^2) / n )
    High values indicate prolonged drawdown stress.
    """
    roll_max = prices.rolling(window, min_periods=1).max()
    pct_dd = ((prices - roll_max) / roll_max) * 100
    return np.sqrt((pct_dd ** 2).rolling(window, min_periods=1).mean())


# ── TAIL RISK ─────────────────────────────────────────────────────────────────

def historical_var_cvar(returns: pd.Series, levels=(0.95, 0.99)) -> dict:
    """
    Historical simulation VaR and CVaR.
    VaR_{α} = -quantile(returns, 1-α)   [positive number = loss]
    CVaR_{α} = -mean(returns | returns < -VaR)
    """
    result = {}
    for level in levels:
        var = -np.percentile(returns, (1 - level) * 100)
        cvar = -returns[returns < -var].mean()
        result[f"var_{int(level*100)}"] = var
        result[f"cvar_{int(level*100)}"] = cvar
    return result


def parametric_var(returns: pd.Series, levels=(0.95, 0.99)) -> dict:
    """
    Parametric VaR under Normal and fitted Student-t distributions.
    Student-t df fitted via MLE.
    """
    mu, sigma = returns.mean(), returns.std()
    result = {}

    # Normal
    for level in levels:
        z = stats.norm.ppf(1 - level)
        result[f"var_normal_{int(level*100)}"] = -(mu + z * sigma)

    # Student-t: fit df via MLE
    try:
        df_t, loc_t, scale_t = stats.t.fit(returns, floc=mu)
        result["t_df"] = df_t
        for level in levels:
            z_t = stats.t.ppf(1 - level, df=df_t, loc=loc_t, scale=scale_t)
            result[f"var_t_{int(level*100)}"] = -z_t
    except Exception:
        result["t_df"] = np.nan
        for level in levels:
            result[f"var_t_{int(level*100)}"] = np.nan

    return result


def cornish_fisher_var(returns: pd.Series, levels=(0.95, 0.99)) -> dict:
    """
    Cornish-Fisher expansion adjusts Normal VaR for skewness (S) and excess kurtosis (K):
    z_cf = z + (z²-1)*S/6 + (z³-3z)*K/24 - (2z³-5z)*S²/36
    Captures fat tails better than parametric normal for non-Gaussian returns.
    Concern threshold: |skew| > 1 or excess kurtosis > 3.
    """
    mu, sigma = returns.mean(), returns.std()
    S = returns.skew()
    K = returns.kurtosis()  # excess kurtosis
    result = {"skewness": S, "excess_kurtosis": K}

    for level in levels:
        z = stats.norm.ppf(1 - level)
        z_cf = (
            z
            + (z**2 - 1) * S / 6
            + (z**3 - 3 * z) * K / 24
            - (2 * z**3 - 5 * z) * S**2 / 36
        )
        result[f"var_cf_{int(level*100)}"] = -(mu + z_cf * sigma)

    return result


def rolling_skew_kurt(prices: pd.Series, window: int = 63) -> pd.DataFrame:
    """Rolling skewness and excess kurtosis. Flag |skew|>1 or kurt>3 as tail-risk."""
    r = _simple_returns(prices)
    return pd.DataFrame({
        "skewness": r.rolling(window).skew(),
        "excess_kurtosis": r.rolling(window).kurt(),
        "skew_flag": r.rolling(window).skew().abs() > 1,
        "kurt_flag": r.rolling(window).kurt() > 3,
    }, index=r.index)


# ── RELATIVE RISK ─────────────────────────────────────────────────────────────

def rolling_beta(stock_prices: pd.Series, bench_prices: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling OLS beta of stock vs benchmark.
    Beta = Cov(r_stock, r_bench) / Var(r_bench)
    β>1: amplifies market moves; β<0: inverse relationship.
    """
    rs = _simple_returns(stock_prices)
    rb = _simple_returns(bench_prices)
    aligned = pd.DataFrame({"s": rs, "b": rb}).dropna()

    betas = []
    idx = []
    for i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[i - window:i]
        cov = chunk.cov().iloc[0, 1]
        var_b = chunk["b"].var()
        betas.append(cov / var_b if var_b > 0 else np.nan)
        idx.append(aligned.index[i - 1])

    return pd.Series(betas, index=idx, name="beta")


def rolling_correlation(stock_prices: pd.Series, bench_prices: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Pearson correlation. Values near 1 = high systematic exposure."""
    rs = _simple_returns(stock_prices)
    rb = _simple_returns(bench_prices)
    return rs.rolling(window).corr(rb)


def residual_vol(stock_prices: pd.Series, bench_prices: pd.Series, window: int = 63) -> pd.Series:
    """
    Idiosyncratic (residual) volatility = total vol - systematic vol.
    systematic_vol = beta * bench_vol
    idio_vol = sqrt(total_var - beta² * bench_var)
    High idio vol signals stock-specific risk not captured by the benchmark.
    """
    rs = _simple_returns(stock_prices)
    rb = _simple_returns(bench_prices)
    aligned = pd.DataFrame({"s": rs, "b": rb}).dropna()

    idio = []
    idx = []
    for i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[i - window:i]
        var_s = chunk["s"].var()
        var_b = chunk["b"].var()
        cov = chunk.cov().iloc[0, 1]
        beta = cov / var_b if var_b > 0 else 0
        systematic_var = beta**2 * var_b
        idio_var = max(var_s - systematic_var, 0)
        idio.append(np.sqrt(idio_var * 252))
        idx.append(aligned.index[i - 1])

    return pd.Series(idio, index=idx, name="idio_vol")


def information_ratio(stock_prices: pd.Series, bench_prices: pd.Series) -> float:
    """
    IR = annualized active return / tracking error.
    Active return = stock return - benchmark return.
    IR > 0.5 is considered good; > 1.0 is excellent.
    """
    rs = _simple_returns(stock_prices)
    rb = _simple_returns(bench_prices)
    active = rs - rb
    active = active.dropna()
    ann_active = active.mean() * 252
    te = active.std() * np.sqrt(252)
    return ann_active / te if te > 0 else np.nan


# ── MACRO REGIME ──────────────────────────────────────────────────────────────

def classify_macro_regime(macro_df: pd.DataFrame) -> pd.Series:
    """
    Regime classification using 63-day rolling change in DGS10 and BAMLH0A0HYM2.
    - RISK_ON:    rates falling AND spreads tightening
    - RISK_OFF:   rates rising  AND spreads widening
    - STAGFLATION: rates rising  AND equities under pressure (no equity here → use spread proxy)
    - MIXED:      other combinations

    Returns a Series of regime labels aligned to macro_df.index.
    """
    if macro_df.empty or "DGS10" not in macro_df.columns:
        return pd.Series(dtype=str)

    w = 63
    rates_chg = macro_df["DGS10"].diff(w)
    spread_chg = macro_df["BAMLH0A0HYM2"].diff(w) if "BAMLH0A0HYM2" in macro_df.columns else pd.Series(0, index=macro_df.index)

    def label(r, s):
        if pd.isna(r) or pd.isna(s):
            return "UNKNOWN"
        if r > 0.2 and s > 0.2:
            return "RISK_OFF"
        if r < -0.2 and s < -0.2:
            return "RISK_ON"
        if r > 0.2 and s < -0.1:
            return "STAGFLATION_PROXY"
        return "MIXED"

    return pd.Series(
        [label(r, s) for r, s in zip(rates_chg, spread_chg)],
        index=macro_df.index,
        name="macro_regime",
    )


def vix_regime(vix_series: pd.Series) -> pd.Series:
    """
    VIX regimes:
    - low:     VIX < 15   (complacency, tight spreads)
    - medium:  15 ≤ VIX < 25 (normal uncertainty)
    - high:    25 ≤ VIX < 35 (elevated stress)
    - extreme: VIX ≥ 35  (crisis, fat-tail risk elevated)
    """
    return pd.cut(
        vix_series,
        bins=[-np.inf, 15, 25, 35, np.inf],
        labels=["low", "medium", "high", "extreme"],
    )


def conditional_returns_by_regime(stock_prices: pd.Series, regime_series: pd.Series) -> pd.DataFrame:
    """
    For each regime label, compute:
    - mean daily return, annualized return, annualized vol, Sharpe (rf=0)
    Returns a DataFrame indexed by regime.
    """
    r = _simple_returns(stock_prices)
    aligned = pd.DataFrame({"ret": r, "regime": regime_series}).dropna()
    rows = []
    for regime, grp in aligned.groupby("regime"):
        n = len(grp)
        mean_r = grp["ret"].mean()
        ann_r = (1 + mean_r) ** 252 - 1
        ann_vol = grp["ret"].std() * np.sqrt(252)
        sharpe = ann_r / ann_vol if ann_vol > 0 else np.nan
        rows.append({
            "regime": regime,
            "n_days": n,
            "mean_daily_ret": mean_r,
            "ann_return": ann_r,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        })
    return pd.DataFrame(rows).set_index("regime")


# ── LIQUIDITY RISK ────────────────────────────────────────────────────────────

def amihud_illiquidity(prices: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio:
    ILLIQ_t = |R_t| / (P_t × V_t)   averaged over rolling window.
    Higher values = less liquid. Units: (% move per dollar volume).
    Concern: ratio in top quartile of its own history.
    """
    r = prices.pct_change().abs()
    dollar_vol = prices * volume
    daily_illiq = r / dollar_vol.replace(0, np.nan)
    return daily_illiq.rolling(window).mean() * 1e6  # scale for readability


def hl_spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """
    High-low spread proxy for bid-ask spread (Corwin & Schultz, 2012 simplified):
    spread = (High - Low) / Close
    Rolling mean as the liquidity proxy. Higher = wider effective spread.
    """
    hl = (high - low) / close
    return hl.rolling(window).mean()


def volume_trend(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to its rolling N-day average. >1.5 = surge, <0.5 = dry-up."""
    return volume / volume.rolling(window).mean()


# ── COMPOSITE RISK SCORE ──────────────────────────────────────────────────────

def composite_risk_score(
    prices: pd.Series,
    volume: pd.Series,
    bench_prices: pd.Series,
    macro_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Single composite risk score (0=lowest risk, 100=highest risk).

    Component weights:
    - Realized vol regime rank (25%)
    - |Max drawdown| magnitude (20%)
    - |Skewness| + excess kurtosis (tail shape) (15%)
    - CVaR 99% magnitude (20%)
    - Beta to benchmark (10%)
    - Amihud illiquidity (10%)

    Each component is mapped to 0-100 via percentile vs S&P 500 universe,
    or vs the stock's own history when universe data isn't available.
    """
    r = _simple_returns(prices)
    lr = _log_returns(prices)

    vol_21 = lr.rolling(21).std().iloc[-1] * np.sqrt(252)
    mdd = abs(max_drawdown(prices)["max_drawdown"])
    tail = cornish_fisher_var(r)
    cf_cvar_99 = historical_var_cvar(r)["cvar_99"]
    skew = abs(r.skew())
    kurt = max(r.kurtosis(), 0)
    beta = rolling_beta(prices, bench_prices).iloc[-1] if len(prices) > 70 else 1.0
    illiq = amihud_illiquidity(prices, volume).iloc[-1]

    # Map each to 0-100 using a simple sigmoid-style normalization
    def norm(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))

    components = {
        "Realized Volatility (21d)": norm(vol_21, 0.10, 0.80) * 0.25,
        "Max Drawdown": norm(mdd, 0.05, 0.70) * 0.20,
        "Tail Shape (Skew+Kurt)": norm(skew + kurt / 5, 0, 4) * 0.15,
        "CVaR 99%": norm(cf_cvar_99, 0.01, 0.10) * 0.20,
        "Beta": norm(abs(beta), 0.3, 2.5) * 0.10,
        "Illiquidity": norm(illiq if not np.isnan(illiq) else 0, 0, 5) * 0.10,
    }
    total = sum(components.values())
    return {"total": round(total, 1), "components": components}
