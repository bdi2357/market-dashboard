"""
Polygon.io data fetcher.
- Primary data source: unlimited daily requests on free tier, 5 calls/minute.
- No IP blocking on Streamlit Cloud (unlike Yahoo Finance).
- Parquet caching under data/cache/ (same TTLs as fetcher.py).

Free tier limits:
  - 5 API calls / minute  → 12-second minimum between requests
  - Unlimited daily requests
  - OHLCV, ticker details, news, previous close, limited financials
"""

import os
import time
import hashlib
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

POLYGON_BASE = "https://api.polygon.io"

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Minimum seconds between consecutive Polygon calls (free tier: 5/min).
# 13s gives a small buffer over the 12s theoretical minimum.
_POLY_RATE_INTERVAL = 13.0
_last_call_ts: float = 0.0


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.parquet"


def _load_cache(key: str, max_age_hours: float) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600
    if age > max_age_hours:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(key))
    except Exception:
        pass


# ── Rate-limited API caller ────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Read key from os.environ at call time (injected by app.py from st.secrets)."""
    return os.environ.get("POLYGON_API_KEY", "")


def _polygon_get(url: str, params: dict) -> dict:
    """
    Rate-limited Polygon REST call.
    Enforces _POLY_RATE_INTERVAL between requests.
    On 429 waits 15 s and retries once.
    """
    global _last_call_ts

    api_key = _get_api_key()
    if not api_key:
        raise Exception("POLYGON_API_KEY not configured")

    # Enforce minimum interval between calls
    elapsed = time.time() - _last_call_ts
    if elapsed < _POLY_RATE_INTERVAL:
        time.sleep(_POLY_RATE_INTERVAL - elapsed)

    call_params = dict(params)
    call_params["apiKey"] = api_key

    _last_call_ts = time.time()
    response = requests.get(url, params=call_params, timeout=30)

    if response.status_code == 429:
        time.sleep(15)
        _last_call_ts = time.time()
        response = requests.get(url, params=call_params, timeout=30)

    if response.status_code != 200:
        raise Exception(
            f"Polygon API error {response.status_code}: "
            f"{response.text[:200]}"
        )

    return response.json()


# ── OHLCV ──────────────────────────────────────────────────────────────────────

def polygon_fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily adjusted OHLCV from Polygon /v2/aggs endpoint.
    Returns DataFrame with timezone-naive DatetimeIndex (same contract as
    fetch_ohlcv in fetcher.py).
    Cached 4 hours.
    """
    key = f"poly_ohlcv_{ticker}_{start}_{end}"
    cached = _load_cache(key, max_age_hours=4)
    if cached is not None:
        return cached

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    data = _polygon_get(url, params)

    if data.get("status") == "ERROR":
        raise Exception(f"Polygon error: {data.get('error', 'Unknown')}")

    results = data.get("results", [])
    if not results:
        raise Exception(
            f"No Polygon data for {ticker} ({start} → {end}). "
            "Check that the ticker is valid and listed on a US exchange."
        )

    df = pd.DataFrame(results)
    df = df.rename(columns={
        "t": "Date",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
    })

    # Timestamps are milliseconds UTC → normalize to date only
    df["Date"] = pd.to_datetime(df["Date"], unit="ms").dt.normalize()
    df = df.set_index("Date").sort_index()

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    _save_cache(key, df)
    return df


# ── Ticker details ─────────────────────────────────────────────────────────────

# Rough SIC description → GICS sector mapping for SECTOR_ETF_MAP compatibility.
# Only the most common mappings; anything not matched falls through to "Unknown".
_SIC_TO_GICS: dict[str, str] = {
    "services-prepackaged software": "Technology",
    "services-computer programming": "Technology",
    "electronic computers": "Technology",
    "semiconductors and related devices": "Technology",
    "telephone communications": "Communication Services",
    "cable & other pay television services": "Communication Services",
    "retail-eating & drinking places": "Consumer Discretionary",
    "retail-department stores": "Consumer Discretionary",
    "retail-auto dealers & gas stations": "Consumer Discretionary",
    "state commercial banks": "Financials",
    "national commercial banks": "Financials",
    "investment offices": "Financials",
    "security brokers, dealers, and flotation companies": "Financials",
    "insurance carriers": "Financials",
    "pharmaceutical preparations": "Health Care",
    "services-hospitals": "Health Care",
    "services-health services": "Health Care",
    "crude petroleum & natural gas": "Energy",
    "oil and gas field services": "Energy",
    "electric services": "Utilities",
    "water supply": "Utilities",
    "steel works, blast furnaces": "Materials",
    "mining & quarrying of nonmetallic minerals": "Materials",
    "air transportation, scheduled": "Industrials",
    "trucking & warehousing": "Industrials",
    "real estate investment trusts": "Real Estate",
    "retail stores, not elsewhere classified": "Consumer Staples",
    "grocery stores": "Consumer Staples",
}


def _sic_to_sector(sic_desc: str) -> str:
    if not sic_desc:
        return "Unknown"
    key = sic_desc.lower().strip()
    for pattern, sector in _SIC_TO_GICS.items():
        if pattern in key:
            return sector
    return "Unknown"


def polygon_fetch_ticker_details(ticker: str) -> dict:
    """
    Fetch company details from Polygon /v3/reference/tickers.
    Returns dict matching the shape expected by the rest of the app.
    Cached 24 hours.
    """
    key = f"poly_info_{ticker}"
    cached = _load_cache(key, max_age_hours=24)
    if cached is not None:
        return cached.iloc[0].to_dict()

    if not _get_api_key():
        return {}

    url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
    try:
        data = _polygon_get(url, {})
    except Exception as e:
        print(f"polygon_fetch_ticker_details error: {e}")
        return {}

    result = data.get("results", {})
    if not result:
        return {}

    sic_desc = result.get("sic_description", "")
    sector = _sic_to_sector(sic_desc)
    address = result.get("address", {})

    info = {
        "sector": sector,
        "industry": sic_desc or "Unknown",
        "market_cap": result.get("market_cap", 0) or 0,
        "name": result.get("name", ticker),
        "exchange": result.get("primary_exchange", ""),
        "description": result.get("description", ""),
        "website": result.get("homepage_url", ""),
        "country": address.get("state", ""),
        "currency": result.get("currency_name", "USD"),
        "total_employees": result.get("total_employees", 0) or 0,
    }

    pd.DataFrame([info]).to_parquet(_cache_path(key))
    return info


# ── Previous close ─────────────────────────────────────────────────────────────

def polygon_fetch_previous_close(ticker: str) -> dict:
    """Get previous trading day OHLCV from Polygon /v2/aggs/ticker/{t}/prev."""
    if not _get_api_key():
        return {}

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/prev"
    try:
        data = _polygon_get(url, {"adjusted": "true"})
        results = data.get("results", [])
        if results:
            r = results[0]
            return {
                "price": r.get("c"),
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "volume": r.get("v"),
            }
    except Exception as e:
        print(f"polygon_fetch_previous_close error: {e}")
    return {}


# ── News ───────────────────────────────────────────────────────────────────────

def polygon_fetch_news(ticker: str, limit: int = 10) -> list[dict]:
    """
    Fetch recent news articles for ticker from Polygon /v2/reference/news.
    Cached 30 minutes.
    Returns list of dicts with keys: title, url, published, source, summary, tickers.
    """
    key = f"poly_news_{ticker}_{limit}"
    cached = _load_cache(key, max_age_hours=0.5)
    if cached is not None:
        return cached.to_dict(orient="records")

    if not _get_api_key():
        return []

    url = f"{POLYGON_BASE}/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": limit,
        "sort": "published_utc",
        "order": "desc",
    }

    try:
        data = _polygon_get(url, params)
        results = data.get("results", [])
        items = [
            {
                "title": r.get("title", ""),
                "url": r.get("article_url", ""),
                "published": r.get("published_utc", ""),
                "source": r.get("publisher", {}).get("name", ""),
                "summary": r.get("description", ""),
                "tickers": r.get("tickers", []),
            }
            for r in results
        ]
        if items:
            _save_cache(key, pd.DataFrame(items))
        return items
    except Exception as e:
        print(f"polygon_fetch_news error: {e}")
        return []


# ── Financials ─────────────────────────────────────────────────────────────────

def polygon_fetch_financials(ticker: str) -> dict:
    """
    Fetch quarterly financial statements from Polygon /vX/reference/financials.
    Note: Full access requires Stocks Starter plan; free tier returns what is
    available. Returns {} if not accessible.
    Cached 24 hours.
    """
    key = f"poly_fin_{ticker}"
    cached = _load_cache(key, max_age_hours=24)
    if cached is not None:
        return cached.iloc[0].to_dict()

    if not _get_api_key():
        return {}

    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {
        "ticker": ticker,
        "limit": 8,
        "sort": "period_of_report_date",
        "order": "desc",
        "timeframe": "quarterly",
    }

    try:
        data = _polygon_get(url, params)
        results = data.get("results", [])
        if not results:
            return {}

        latest = results[0]
        financials = latest.get("financials", {})
        income = financials.get("income_statement", {})
        balance = financials.get("balance_sheet", {})
        cash_flow = financials.get("cash_flow_statement", {})

        def _v(section: dict, field: str) -> float:
            return section.get(field, {}).get("value", 0) or 0

        result = {
            "revenue": _v(income, "revenues"),
            "net_income": _v(income, "net_income_loss"),
            "operating_income": _v(income, "operating_income_loss"),
            "total_assets": _v(balance, "assets"),
            "total_liabilities": _v(balance, "liabilities"),
            "equity": _v(balance, "equity_attributable_to_parent"),
            "operating_cash_flow": _v(
                cash_flow, "net_cash_flow_from_operating_activities"
            ),
            "capex": abs(
                _v(cash_flow, "net_cash_flow_from_investing_activities")
            ),
            "period": latest.get("period_of_report_date", ""),
            "quarters_available": len(results),
        }

        pd.DataFrame([result]).to_parquet(_cache_path(key))
        return result
    except Exception as e:
        print(f"polygon_fetch_financials error: {e}")
        return {}
