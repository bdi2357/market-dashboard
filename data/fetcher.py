"""
Data fetching layer.
- Equity OHLCV + metadata via yfinance (Ticker.history, auto_adjust=True, tz stripped)
- Macro series via FRED (fredapi): DGS10, BAMLH0A0HYM2, VIXCLS, T10Y2Y
- Parquet caching under data/cache/ to avoid re-fetching
- S&P 500 constituent list from Wikipedia
"""

import os
import re
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Path to secrets.toml relative to this file (../../.streamlit/secrets.toml)
_SECRETS_PATH = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"


def _get_fred_key() -> str:
    """
    Return FRED API key from (in order):
    1. FRED_API_KEY environment variable
    2. .streamlit/secrets.toml  (parsed directly — no Streamlit dependency)
    """
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        return key
    if _SECRETS_PATH.exists():
        try:
            text = _SECRETS_PATH.read_text(encoding="utf-8")
            m = re.search(r'FRED_API_KEY\s*=\s*["\']([^"\']+)["\']', text)
            if m:
                return m.group(1)
        except Exception:
            pass
    return ""

# FRED series used by macro regime module
FRED_SERIES = {
    "DGS10": "10Y Treasury Yield",
    "BAMLH0A0HYM2": "HY Credit Spread",
    "VIXCLS": "VIX",
    "T10Y2Y": "Yield Curve (10Y-2Y)",
}

# Sector ETF map (GICS sector name -> ETF ticker)
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Communication Services": "XLC",
}


def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.parquet"


def _load_cache(key: str, max_age_hours: int = 4) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600
    if age > max_age_hours:
        return None
    return pd.read_parquet(path)


def _save_cache(key: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(key))


def fetch_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Return OHLCV DataFrame with timezone-naive DatetimeIndex."""
    key = f"ohlcv_{ticker}_{start}_{end}"
    cached = _load_cache(key)
    if cached is not None:
        return cached
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    _save_cache(key, df)
    return df


def fetch_ticker_info(ticker: str) -> dict:
    """Return sector, industry, market cap, name from yfinance."""
    key = f"info_{ticker}"
    path = _cache_path(key)
    if path.exists():
        try:
            return pd.read_parquet(path).iloc[0].to_dict()
        except Exception:
            pass
    t = yf.Ticker(ticker)
    info = t.info or {}
    result = {
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("marketCap", None),
        "name": info.get("longName", ticker),
        "exchange": info.get("exchange", ""),
    }
    pd.DataFrame([result]).to_parquet(path)
    return result


def get_sector_etf(ticker: str) -> str:
    """Return the sector ETF for this ticker, defaulting to SPY."""
    info = fetch_ticker_info(ticker)
    return SECTOR_ETF_MAP.get(info.get("sector", ""), "SPY")


def fetch_macro(start: date, end: date) -> pd.DataFrame:
    """
    Return DataFrame with columns [DGS10, BAMLH0A0HYM2, VIXCLS, T10Y2Y].
    Falls back to pandas_datareader if fredapi key not set.
    """
    key = f"macro_{start}_{end}"
    cached = _load_cache(key, max_age_hours=24)
    if cached is not None:
        return cached

    fred_key = _get_fred_key()
    frames = {}

    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            for series_id in FRED_SERIES:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                frames[series_id] = s
        except Exception as e:
            print(f"fredapi error: {e}, falling back to pandas_datareader")

    if not frames:
        try:
            import pandas_datareader.data as web
            for series_id in FRED_SERIES:
                s = web.DataReader(series_id, "fred", start, end)[series_id]
                frames[series_id] = s
        except Exception as e:
            print(f"pandas_datareader error: {e}")
            # Return empty frame — macro tab will show a warning
            return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.ffill()
    _save_cache(key, df)
    return df


def fetch_sp500_tickers() -> list[str]:
    """Scrape S&P 500 constituents from Wikipedia."""
    key = "sp500_tickers"
    path = _cache_path(key)
    if path.exists():
        age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age < 24:
            return pd.read_parquet(path)["ticker"].tolist()
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0][["Symbol"]].rename(columns={"Symbol": "ticker"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df.to_parquet(path)
        return df["ticker"].tolist()
    except Exception:
        return []


def fetch_peers(ticker: str, start: date, end: date, top_n: int = 10) -> dict[str, pd.Series]:
    """
    Return closing price series for top-N S&P 500 peers in the same sector.
    Falls back to a hardcoded list if Wikipedia fetch fails.
    """
    info = fetch_ticker_info(ticker)
    sector = info.get("sector", "")
    sp500 = fetch_sp500_tickers()

    if sp500 and sector and sector != "Unknown":
        # Filter by sector
        sector_peers = []
        for t in sp500:
            if t == ticker:
                continue
            try:
                ti = fetch_ticker_info(t)
                if ti.get("sector") == sector:
                    sector_peers.append((t, ti.get("market_cap") or 0))
            except Exception:
                pass
        sector_peers.sort(key=lambda x: x[1], reverse=True)
        peers = [t for t, _ in sector_peers[:top_n]]
    else:
        peers = ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "AVGO", "ORCL", "CRM", "AMD"]
        peers = [p for p in peers if p != ticker][:top_n]

    result = {}
    for p in peers:
        df = fetch_ohlcv(p, start, end)
        if not df.empty:
            result[p] = df["Close"]
    return result
