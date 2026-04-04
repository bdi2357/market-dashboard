"""
Sector-Specific Price Shock Analysis
=====================================
Identifies historical episodes when key factors (oil, rates, FX, etc.)
moved by shock magnitudes and measures how the stock responded.

Entirely free data: yfinance for factor prices, FRED for macro context.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ── SECTOR → SHOCK FACTOR MAPPING ────────────────────────────────────────────

SECTOR_SHOCK_MAP: dict[str, dict] = {
    "Industrials": {
        "Crude Oil (WTI)": {
            "ticker": "USO", "shocks": [-40, -20, +20, +40, +60],
            "transmission": "cost_shock",
            "note": "Oil is key fuel/feedstock cost for industrials.",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, -10, +10, +15],
            "transmission": "revenue_fx",
            "note": "Strong USD hits international revenues.",
        },
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, -50, +100, +200],
            "is_rate": True, "transmission": "debt_refinancing",
            "note": "Leverage-sensitive. Rate rise increases refinancing cost.",
        },
    },
    "Energy": {
        "WTI Crude Oil": {
            "ticker": "USO", "shocks": [-50, -30, +30, +50],
            "transmission": "revenue_shock",
            "note": "Direct revenue driver for E&P companies.",
        },
        "Natural Gas": {
            "ticker": "UNG", "shocks": [-50, -30, +30, +50, +100],
            "transmission": "revenue_shock",
            "note": "Key revenue driver for gas-weighted producers.",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, +15],
            "transmission": "revenue_fx",
            "note": "Oil priced globally in USD.",
        },
    },
    "Materials": {
        "Gold": {
            "ticker": "GLD", "shocks": [-30, -20, +20, +30, +50],
            "transmission": "revenue_shock",
            "note": "Gold miners: ~3x operating leverage to gold price.",
        },
        "Natural Gas": {
            "ticker": "UNG", "shocks": [-50, +50, +100],
            "transmission": "cost_shock",
            "note": "Primary feedstock for chemicals/fertilizers.",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, +15],
            "transmission": "revenue_fx",
        },
    },
    "Consumer Staples": {
        "Agriculture (WEAT)": {
            "ticker": "WEAT", "shocks": [-30, +30, +60],
            "transmission": "cost_shock",
            "note": "Wheat/corn/soy are key COGS inputs for food companies.",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, +15],
            "transmission": "revenue_fx",
        },
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, +100, +200],
            "is_rate": True, "transmission": "multiple_compression",
        },
    },
    "Consumer Discretionary": {
        "Crude Oil (fuel cost)": {
            "ticker": "USO", "shocks": [-30, +30, +50],
            "transmission": "cost_shock",
            "note": "Higher oil reduces consumer discretionary spending.",
        },
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, +100, +200],
            "is_rate": True, "transmission": "multiple_compression",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, +15],
            "transmission": "revenue_fx",
        },
    },
    "Financials": {
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-200, -100, +100, +200],
            "is_rate": True, "transmission": "nim_expansion",
            "note": "Asset-sensitive banks benefit from rate rises.",
        },
        "HY Credit Spreads": {
            "ticker": "HYG", "shocks": [-200, +200, +400, +700],
            "transmission": "credit_losses",
            "note": "Wider spreads signal rising credit losses ahead.",
        },
        "Yield Curve (2Y via SHY)": {
            "ticker": "SHY", "shocks": [-100, +100],
            "transmission": "nim_structural",
            "note": "Yield curve flattening compresses NIM structurally.",
        },
    },
    "Technology": {
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, -50, +100, +200],
            "is_rate": True, "transmission": "multiple_compression",
            "note": "Long-duration growth stocks most sensitive to rate rises.",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, +15],
            "transmission": "revenue_fx",
            "note": "Large international revenue exposure.",
        },
        "S&P 500 (beta)": {
            "ticker": "SPY", "shocks": [-40, -20, -10, +10, +20],
            "transmission": "market_beta",
        },
    },
    "Utilities": {
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, +100, +200],
            "is_rate": True, "transmission": "multiple_and_debt",
            "note": "Bond proxies + heavily indebted. Rate rise = double hit.",
        },
        "Natural Gas": {
            "ticker": "UNG", "shocks": [-50, +50, +100],
            "transmission": "cost_shock",
            "note": "Gas-fired generation feedstock cost.",
        },
    },
    "Real Estate": {
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, +100, +200],
            "is_rate": True, "transmission": "multiple_and_debt",
            "note": "Cap rate expansion compresses REIT NAV; refinancing risk.",
        },
        "S&P 500 (beta)": {
            "ticker": "SPY", "shocks": [-40, -20, +20],
            "transmission": "market_beta",
        },
    },
    "Health Care": {
        "S&P 500 (beta)": {
            "ticker": "SPY", "shocks": [-40, -20, -10, +10, +20],
            "transmission": "market_beta",
        },
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, +200],
            "is_rate": True, "transmission": "multiple_compression",
            "note": "Unprofitable biotechs very rate-sensitive (long duration).",
        },
    },
    "Communication Services": {
        "10Y Treasury Rate": {
            "ticker": "^TNX", "shocks": [-100, +200],
            "is_rate": True, "transmission": "multiple_compression",
        },
        "USD Index": {
            "ticker": "UUP", "shocks": [-15, +15],
            "transmission": "revenue_fx",
        },
        "S&P 500 (beta)": {
            "ticker": "SPY", "shocks": [-40, -20, +20],
            "transmission": "market_beta",
        },
    },
}

_DEFAULT_FACTORS = {
    "S&P 500 (beta)": {
        "ticker": "SPY", "shocks": [-40, -20, -10, +10, +20],
        "transmission": "market_beta",
    },
    "10Y Treasury Rate": {
        "ticker": "^TNX", "shocks": [-100, +100, +200],
        "is_rate": True, "transmission": "multiple_compression",
    },
}


def get_sector_factors(sector: str) -> dict:
    """Return the factor shock map for the given sector."""
    # Try exact match first, then partial
    if sector in SECTOR_SHOCK_MAP:
        return SECTOR_SHOCK_MAP[sector]
    for key, factors in SECTOR_SHOCK_MAP.items():
        if key.lower() in sector.lower() or sector.lower() in key.lower():
            return factors
    return _DEFAULT_FACTORS


# ── data helpers ──────────────────────────────────────────────────────────────

def _fetch_monthly(ticker: str, start: date, end: date) -> Optional[pd.Series]:
    """Fetch monthly total-return price series via yfinance."""
    try:
        raw = yf.Ticker(ticker).history(
            start=start - timedelta(days=90),
            end=end + timedelta(days=10),
            auto_adjust=True,
        )
        if raw.empty:
            return None
        raw.index = raw.index.tz_localize(None)
        monthly = raw["Close"].resample("ME").last()
        return monthly
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


def _monthly_returns(s: pd.Series) -> pd.Series:
    """Simple monthly returns from price series."""
    return s.pct_change().dropna()


# ── CORE: analyze_price_shock ─────────────────────────────────────────────────

def _find_shock_episodes(factor_ret: pd.Series, shock_pct: float) -> pd.DatetimeIndex:
    """
    Find months where factor return exceeded |shock_pct|% in the required direction.
    Enforces 1-month buffer between episodes.
    """
    threshold = shock_pct / 100.0
    if threshold > 0:
        mask = factor_ret >= threshold
    else:
        mask = factor_ret <= threshold

    episodes = []
    last_ep = None
    for dt in factor_ret.index:
        if mask.get(dt, False):
            if last_ep is None or (dt - last_ep).days > 30:
                episodes.append(dt)
                last_ep = dt
    return pd.DatetimeIndex(episodes)


def _conditional_distribution(
    stock_ret: pd.Series,
    episodes: pd.DatetimeIndex,
    lag: int = 0,
) -> dict:
    """
    For each episode date, get stock return `lag` months later.
    lag=0: same month, lag=1: next month, etc.
    """
    responses = []
    for ep_dt in episodes:
        try:
            # Find the target month
            target_idx = stock_ret.index.get_indexer([ep_dt], method="nearest")[0]
            target_idx += lag
            if 0 <= target_idx < len(stock_ret):
                responses.append(float(stock_ret.iloc[target_idx]))
        except Exception:
            continue

    if not responses:
        return {}

    arr = np.array(responses)
    return {
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "worst": round(float(np.percentile(arr, 5)), 4),
        "best": round(float(np.percentile(arr, 95)), 4),
        "pct_negative": round(float((arr < 0).mean() * 100), 1),
        "n": len(responses),
    }


def _confidence_label(n: int) -> str:
    if n >= 10:
        return "HIGH"
    if n >= 5:
        return "MEDIUM"
    if n >= 3:
        return "LOW"
    return "INSUFFICIENT"


def analyze_price_shock(
    ticker: str,
    sector: str,
    prices: pd.Series,
    start: date,
    end: date,
) -> dict:
    """
    For each relevant factor in the sector shock map:
    1. Download factor monthly price history
    2. Find historical shock episodes
    3. Measure stock response (same-month, +1M, 3M window)
    4. Return conditional distributions + current shock assessment

    Args:
        ticker: stock symbol
        sector: GICS sector string
        prices: daily stock close price series
        start, end: date range

    Returns:
        dict with historical_responses, current_shock_risk, factor analysis
    """
    factors = get_sector_factors(sector)

    # Monthly stock returns
    stock_monthly = prices.resample("ME").last()
    stock_ret = _monthly_returns(stock_monthly)
    stock_ret.index = stock_ret.index.normalize()

    results: dict = {
        "sector": sector,
        "ticker": ticker,
        "factors_analyzed": list(factors.keys()),
        "historical_responses": {},
        "current_shock_risk": {},
        "compound_scenario": None,
        "available": True,
    }

    if len(stock_ret) < 18:
        results["available"] = False
        results["error"] = "Insufficient price history (need 18+ months)"
        return results

    for factor_name, factor_cfg in factors.items():
        factor_ticker = factor_cfg["ticker"]
        shocks = factor_cfg["shocks"]
        is_rate = factor_cfg.get("is_rate", False)

        # Fetch factor data
        if is_rate:
            # For rate tickers like ^TNX, compute level change in bps
            factor_prices = _fetch_monthly(factor_ticker, start, end)
            if factor_prices is None:
                continue
            # Rate changes in bps
            factor_ret_raw = factor_prices.diff().dropna() * 100  # pct → bps
        else:
            factor_prices = _fetch_monthly(factor_ticker, start, end)
            if factor_prices is None:
                continue
            factor_ret_raw = _monthly_returns(factor_prices)

        factor_ret_raw.index = factor_ret_raw.index.normalize()

        # Align with stock returns
        common = stock_ret.index.intersection(factor_ret_raw.index)
        if len(common) < 12:
            continue

        s_ret = stock_ret.loc[common]
        f_ret = factor_ret_raw.loc[common]

        # Current shock risk assessment
        last_3m = f_ret.iloc[-3:]
        momentum = float(last_3m.mean())
        if is_rate:
            current_risk = (
                "ELEVATED" if abs(momentum) > 30 else
                "NORMAL" if abs(momentum) > 10 else "SUBDUED"
            )
        else:
            current_risk = (
                "ELEVATED" if abs(momentum) > 0.05 else
                "NORMAL" if abs(momentum) > 0.02 else "SUBDUED"
            )
        results["current_shock_risk"][factor_name] = current_risk

        # Historical shock episodes for each shock magnitude
        shock_responses: dict = {}
        for shock in shocks:
            episodes = _find_shock_episodes(f_ret, shock)
            n_episodes = len(episodes)

            if n_episodes < 3:
                shock_responses[f"{shock:+.0f}{'bps' if is_rate else '%'}"] = {
                    "episodes_found": n_episodes,
                    "confidence": "INSUFFICIENT",
                }
                continue

            # Measure stock response at different lags
            same_month = _conditional_distribution(s_ret, episodes, lag=0)
            next_month = _conditional_distribution(s_ret, episodes, lag=1)

            # 3-month cumulative: compute compound return
            three_m_rets = []
            for ep_dt in episodes:
                try:
                    idx0 = s_ret.index.get_indexer([ep_dt], method="nearest")[0]
                    window = s_ret.iloc[idx0: idx0 + 3]
                    if len(window) >= 2:
                        cum = float((1 + window).prod() - 1)
                        three_m_rets.append(cum)
                except Exception:
                    continue

            three_m = {}
            if three_m_rets:
                arr = np.array(three_m_rets)
                three_m = {
                    "median": round(float(np.median(arr)), 4),
                    "worst": round(float(np.percentile(arr, 5)), 4),
                    "best": round(float(np.percentile(arr, 95)), 4),
                    "pct_negative": round(float((arr < 0).mean() * 100), 1),
                }

            shock_label = f"{shock:+.0f}{'bps' if is_rate else '%'}"
            shock_responses[shock_label] = {
                "episodes_found": n_episodes,
                "confidence": _confidence_label(n_episodes),
                "same_month": same_month,
                "next_month": next_month,
                "three_month": three_m,
            }

        results["historical_responses"][factor_name] = {
            "transmission": factor_cfg.get("transmission", "unknown"),
            "note": factor_cfg.get("note", ""),
            "shocks": shock_responses,
        }

    # ── Compound shock scenario ───────────────────────────────────────────────
    # Simple heuristic: combine top 2 risk factors
    elevated_factors = [
        f for f, risk in results["current_shock_risk"].items()
        if risk == "ELEVATED"
    ]
    if len(elevated_factors) >= 2:
        results["compound_scenario"] = {
            "name": "Multi-Factor Stress",
            "factors": elevated_factors[:3],
            "description": (
                f"Multiple risk factors simultaneously elevated for {ticker} "
                f"in {sector}: {', '.join(elevated_factors[:3])}. "
                "Historical analogues: 2022 (rate + energy shock), "
                "2008 (credit + equity crash), 2020 (COVID demand collapse)."
            ),
        }
    elif sector in ("Energy", "Industrials"):
        results["compound_scenario"] = {
            "name": "Stagflation Shock",
            "factors": ["Oil +40%", "Rates +200bps", "Recession -2% GDP"],
            "description": (
                "1970s-style stagflation analogue: oil spike + rate rise "
                "simultaneously. For energy producers: mixed (revenue up, "
                "cost of capital up). For industrials: predominantly negative."
            ),
        }
    elif "Financials" in sector:
        results["compound_scenario"] = {
            "name": "Credit Crisis",
            "factors": ["HY Spreads +400bps", "Rate Inversion", "GDP -2%"],
            "description": (
                "2008/2023-style credit crunch: spread widening + yield curve "
                "inversion + recessionary credit losses all hitting simultaneously."
            ),
        }
    else:
        results["compound_scenario"] = {
            "name": "Risk-Off Shock",
            "factors": ["SPY -30%", "VIX >35", "HY Spreads +300bps"],
            "description": (
                "Broad risk-off episode (2008, 2020, 2022): equity drawdown "
                "combined with credit spread widening and elevated volatility."
            ),
        }

    return results
