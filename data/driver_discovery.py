"""
Dynamic Risk Driver Discovery
==============================
3-layer system that discovers company-specific external risk drivers
without any hardcoded sector mappings.

Layer 1: EDGAR 10-K Item 1A risk factors (what the company says its own risks are)
Layer 2: Map to tradeable price proxies (commodity prices, rates, FX, indices)
Layer 3: News velocity scoring via Tavily (what is actually moving in headlines NOW)

Combined relevance score drives the AI report structure:
  relevance = baseline_weight × (1 + news_velocity×0.5) × (1.3 if price_moving)
"""

import re
import json
import time
import logging
import requests
import yfinance as yf
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {"User-Agent": "market-dashboard itaybd@gmail.com"}


# ── LAYER 2: FACTOR → PRICE PROXY MAP ────────────────────────────────────────

FACTOR_PRICE_MAP = {
    # Energy / Transport
    "jet fuel":         "CL=F",
    "crude oil":        "CL=F",
    "natural gas":      "NG=F",
    "fuel":             "CL=F",
    "energy cost":      "XLE",
    "gasoline":         "RB=F",
    "oil price":        "CL=F",
    # Rates / Credit
    "interest rate":    "^TNX",
    "federal funds":    "^IRX",
    "credit spread":    "HYG",
    "borrowing cost":   "^TNX",
    "default rate":     "HYG",
    "yield curve":      "^TNX",
    "treasury":         "^TNX",
    "mortgage rate":    "^TNX",
    # FX
    "usd":              "DX-Y.NYB",
    "dollar":           "DX-Y.NYB",
    "euro":             "EURUSD=X",
    "chinese yuan":     "CNY=X",
    "yen":              "JPY=X",
    "rmb":              "CNY=X",
    "currency":         "DX-Y.NYB",
    # Metals / Materials
    "gold":             "GC=F",
    "silver":           "SI=F",
    "copper":           "HG=F",
    "aluminum":         "ALI=F",
    "steel":            "SLX",
    "lithium":          "LIT",
    "iron ore":         "SLX",
    # Agriculture
    "wheat":            "WEAT",
    "corn":             "ZC=F",
    "soybean":          "ZS=F",
    "agricultural":     "WEAT",
    "fertilizer":       "WEAT",
    # Technology
    "ai chip":          "NVDA",
    "semiconductor":    "SOXX",
    "memory":           "MU",
    "dram":             "MU",
    "cloud":            "CLOU",
    "data center":      "NVDA",
    "ai demand":        "NVDA",
    "export control":   "SOXX",
    "chip":             "SOXX",
    # Consumer / Macro
    "consumer spending": "XLY",
    "consumer confidence": "XLY",
    "inflation":         "TIP",
    "housing":           "XHB",
    "freight":           "XLI",
    "shipping":          "XLI",
}


# ── LAYER 1: EDGAR 10-K EXTRACTION ───────────────────────────────────────────

def _fetch_10k_text(cik: str) -> str | None:
    """Fetch Item 1A text from the latest 10-K filing for a given CIK."""
    try:
        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        time.sleep(0.12)
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        filings = resp.json().get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])

        # Find the most recent 10-K
        tenk_idx = next((i for i, f in enumerate(forms) if f == "10-K"), None)
        if tenk_idx is None:
            return None

        acc = accessions[tenk_idx].replace("-", "")
        doc = docs[tenk_idx]
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"

        time.sleep(0.12)
        doc_resp = requests.get(doc_url, headers=_HEADERS, timeout=20)
        if doc_resp.status_code != 200:
            return None

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(doc_resp.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        # Extract Item 1A section
        m = re.search(
            r"(?i)item\s+1a[\.\s:]+risk\s+factors(.*?)(?:item\s+1b|item\s+2)",
            text, re.DOTALL
        )
        return m.group(1)[:8000] if m else text[:4000]
    except Exception as e:
        logger.warning(f"10-K fetch failed for CIK {cik}: {e}")
        return None


def _extract_factors_from_text(risk_text: str, openai_client) -> list[dict]:
    """Use LLM to extract structured risk factors from 10-K text."""
    prompt = f"""Extract the top 8 EXTERNAL risk factors from this 10-K filing.
External = driven by outside prices, rates, regulations, or demand shifts.

For each factor return a JSON object with:
- name: 3-5 word label
- external_factor: specific price/metric driving it (e.g. "jet fuel price", "10Y Treasury rate", "AI chip demand") — null if purely internal
- category: one of COMMODITY|MACRO|REGULATORY|COMPETITIVE|FINANCIAL|TECHNOLOGY|GEOPOLITICAL
- description: one sentence
- measurability: HIGH (has a tradeable proxy) | MEDIUM | LOW
- baseline_weight: float from 0.5 to 1.5
  1.5 = existential (fuel for airlines, oil for E&P, gold for gold miners)
  1.3 = highly material (rates for banks, AI demand for semis)
  1.0 = significant
  0.7 = secondary
  0.5 = minor

Return ONLY a valid JSON array. No markdown fences, no other text.

10-K text:
{risk_text}"""

    try:
        resp = openai_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        factors = json.loads(raw)
        for f in factors:
            f.setdefault("baseline_weight", 1.0)
            f.setdefault("news_velocity", 0.0)
            f.setdefault("current_relevance_score", f["baseline_weight"])
            f["source"] = "10-K"
        return factors
    except Exception as e:
        logger.warning(f"LLM factor extraction failed: {e}")
        return []


def extract_10k_risk_factors(ticker: str, cik: str, openai_client) -> list[dict]:
    """Layer 1: get factors from EDGAR 10-K, fall back to empty list."""
    text = _fetch_10k_text(cik)
    if not text:
        return []
    return _extract_factors_from_text(text, openai_client)


def generate_drivers_from_description(
    ticker: str, company: str, sector: str, industry: str, openai_client
) -> list[dict]:
    """Fallback: LLM generates drivers from yfinance business description."""
    try:
        info = yf.Ticker(ticker).info
        desc = (info.get("longBusinessSummary") or "")[:2000]
    except Exception:
        desc = ""

    prompt = f"""Identify the top 8 EXTERNAL risk factors for this company.
Focus on specific external prices, rates, or metrics that directly impact P&L.

Company: {company} ({ticker})
Sector: {sector} / Industry: {industry}
Description: {desc}

Examples of specific external factors (not generic):
- Airlines → "jet fuel price" (baseline_weight: 1.5)
- Banks → "federal funds rate" (1.3), "loan default rate" (1.2)
- Semiconductors → "AI data center capex" (1.4), "export control restrictions" (1.2)
- Gold miners → "gold spot price" (1.5)
- Retailers → "consumer confidence index" (1.2)
- REITs → "10Y Treasury rate" (1.4)

Return ONLY a valid JSON array with fields: name, external_factor, category, description, measurability, baseline_weight.
No markdown fences, no other text."""

    try:
        resp = openai_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        factors = json.loads(raw)
        for f in factors:
            f.setdefault("baseline_weight", 1.0)
            f.setdefault("news_velocity", 0.0)
            f.setdefault("current_relevance_score", f["baseline_weight"])
            f["source"] = "LLM_description"
        return factors
    except Exception as e:
        logger.warning(f"LLM description-based driver generation failed: {e}")
        return _hardcoded_fallback(sector)


def _hardcoded_fallback(sector: str) -> list[dict]:
    """Last-resort fallback: minimal generic drivers if LLM also fails."""
    base = [
        {"name": "Market conditions", "external_factor": "S&P 500 level",
         "category": "MACRO", "description": "Broad market regime impact on valuation.",
         "measurability": "HIGH", "baseline_weight": 1.0,
         "news_velocity": 0.0, "current_relevance_score": 1.0, "source": "fallback"},
        {"name": "Interest rate environment", "external_factor": "10Y Treasury rate",
         "category": "MACRO", "description": "Rate level affects discount rates and debt cost.",
         "measurability": "HIGH", "baseline_weight": 1.0,
         "news_velocity": 0.0, "current_relevance_score": 1.0, "source": "fallback"},
        {"name": "Credit conditions", "external_factor": "HY credit spread",
         "category": "FINANCIAL", "description": "Spread widening signals tightening financial conditions.",
         "measurability": "HIGH", "baseline_weight": 0.8,
         "news_velocity": 0.0, "current_relevance_score": 0.8, "source": "fallback"},
    ]
    return base


# ── LAYER 2: MAP TO PRICE PROXIES ────────────────────────────────────────────

def map_to_price_proxies(factors: list[dict]) -> list[dict]:
    """Attach tradeable proxy tickers and compute recent price moves."""
    for factor in factors:
        ext = (factor.get("external_factor") or "").lower()
        if not ext:
            continue
        for keyword, proxy_ticker in FACTOR_PRICE_MAP.items():
            if keyword in ext:
                factor["price_proxy"] = proxy_ticker
                try:
                    data = yf.Ticker(proxy_ticker).history(period="3mo", auto_adjust=True)
                    if data.empty or len(data) < 5:
                        break
                    closes = data["Close"].dropna()
                    closes.index = closes.index.tz_localize(None)
                    r1m = float((closes.iloc[-1] / closes.iloc[max(-22, -len(closes))] - 1) * 100)
                    r3m = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)
                    factor["factor_1m_return"] = round(r1m, 2)
                    factor["factor_3m_return"] = round(r3m, 2)
                    factor["factor_is_moving"] = abs(r1m) > 5.0
                except Exception:
                    pass
                break
    return factors


# ── LAYER 3: NEWS VELOCITY ────────────────────────────────────────────────────

def score_news_velocity(
    factors: list[dict], ticker: str, company: str, tavily_client
) -> list[dict]:
    """
    For each factor, search Tavily for recent news and score velocity (0.0–2.0).
    Velocity > 1.0 means the factor is getting significant news coverage.
    """
    if tavily_client is None:
        for f in factors:
            f.setdefault("news_velocity", 0.0)
        return factors

    for factor in factors:
        ext = factor.get("external_factor")
        if not ext:
            factor.setdefault("news_velocity", 0.0)
            continue
        try:
            r = tavily_client.search(
                query=f"{ext} {company} impact risk outlook",
                search_depth="basic",
                max_results=5,
            )
            results = r.get("results", [])
            high_quality = sum(1 for x in results if x.get("score", 0) > 0.7)
            velocity = round(min(2.0, high_quality * 0.5 + len(results) * 0.2), 2)
            factor["news_velocity"] = velocity
            factor["news_results"] = [
                {"title": x.get("title", ""), "url": x.get("url", "")}
                for x in results[:3]
            ]
        except Exception:
            factor.setdefault("news_velocity", 0.0)

    return factors


# ── COMBINED RELEVANCE SCORE ──────────────────────────────────────────────────

def compute_relevance(factors: list[dict]) -> list[dict]:
    """
    relevance = baseline_weight × (1 + news_velocity×0.5) × (1.3 if price_moving)
    Sort descending and assign priority labels.
    """
    for f in factors:
        f["current_relevance_score"] = round(
            f.get("baseline_weight", 1.0)
            * (1 + f.get("news_velocity", 0.0) * 0.5)
            * (1.3 if f.get("factor_is_moving", False) else 1.0),
            3,
        )

    factors.sort(key=lambda x: x["current_relevance_score"], reverse=True)
    for i, f in enumerate(factors):
        f["priority"] = (
            "PRIMARY" if i < 3 else
            "SECONDARY" if i < 6 else
            "MONITORING"
        )
    return factors


# ── DRIVER RELEVANCE FILTERING ───────────────────────────────────────────────

# Terms that are clearly irrelevant for a given sector
_SECTOR_EXCLUSIONS: dict[str, list[str]] = {
    "Technology": [
        "jet fuel", "crude oil", "oil price", "natural gas", "fuel cost",
        "wheat", "corn", "soybean", "agricultural", "fertilizer",
        "freight rate", "shipping cost", "iron ore", "steel price",
    ],
    "Semiconductors": [
        "jet fuel", "crude oil", "oil price", "natural gas",
        "wheat", "corn", "soybean", "agricultural", "fertilizer",
    ],
    "Information Technology": [
        "jet fuel", "crude oil", "oil price", "natural gas", "fuel cost",
        "wheat", "corn", "soybean", "agricultural", "fertilizer",
    ],
    "Airlines": [
        "ai chip demand", "semiconductor demand", "cloud subscription",
        "software revenue", "data center", "memory price",
    ],
    "Financials": [
        "jet fuel", "crude oil", "oil price", "agricultural", "wheat",
        "corn", "soybean", "semiconductor demand",
    ],
    "Consumer Staples": [
        "ai chip", "semiconductor", "jet fuel", "crude oil",
        "data center", "cloud demand",
    ],
    "Energy": [
        "ai chip", "semiconductor", "cloud demand", "wheat", "corn",
        "soybean", "agricultural",
    ],
    "Healthcare": [
        "jet fuel", "crude oil", "oil price", "wheat", "corn",
        "soybean", "agricultural", "semiconductor demand",
    ],
}


def filter_irrelevant_drivers(
    factors: list[dict],
    ticker: str,
    sector: str,
    industry: str,
) -> list[dict]:
    """
    Remove drivers whose external_factor clearly doesn't apply
    to this company's sector/industry.
    """
    # Build exclusion list from sector and industry lookups
    exclusions: list[str] = []
    for key, excl_list in _SECTOR_EXCLUSIONS.items():
        if key.lower() in sector.lower() or key.lower() in industry.lower():
            exclusions.extend(excl_list)

    if not exclusions:
        return factors

    filtered = []
    for factor in factors:
        ext = (factor.get("external_factor") or "").lower()
        desc = (factor.get("description") or "").lower()
        name = (factor.get("name") or "").lower()
        combined = f"{ext} {desc} {name}"
        if any(excl in combined for excl in exclusions):
            logger.debug(
                f"Filtered irrelevant driver '{factor.get('name')}' "
                f"for {ticker} ({sector}/{industry})"
            )
        else:
            filtered.append(factor)
    return filtered


def validate_drivers_with_llm(
    factors: list[dict],
    ticker: str,
    company: str,
    sector: str,
    industry: str,
    openai_client,
) -> list[dict]:
    """
    Ask the LLM to review and remove any drivers that don't actually
    affect this specific company. Returns filtered list.
    Falls back to the original list on any error.
    """
    if not openai_client or not factors:
        return factors

    factor_lines = "\n".join(
        f"- {f.get('name','')}: {f.get('external_factor','')}"
        for f in factors
    )

    prompt = f"""Company: {company} ({ticker})
Sector: {sector} / Industry: {industry}

Review these risk drivers and return ONLY the names that are genuinely relevant.

Drivers:
{factor_lines}

Rules (be strict):
- For Technology/Semiconductors: oil/fuel/agricultural are NOT relevant
  unless there is a specific reason. AI demand, export controls, chip
  competition, TSMC risk, data center capex ARE relevant.
- For Airlines: fuel costs are PRIMARY. AI chip demand is NOT relevant.
- For Banks/Financials: credit spread, fed funds, yield curve ARE relevant.
  jet fuel, agricultural, semiconductor demand are NOT.
- For Energy companies: oil/gas prices ARE relevant. Semiconductor demand
  and agricultural prices are NOT.
- Remove anything that is clearly a generic macro factor with no
  specific transmission to this company's P&L.

Return ONLY a valid JSON array of driver NAMES to keep. No other text."""

    try:
        resp = openai_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        keep_names = json.loads(raw)
        keep_lower = [k.lower() for k in keep_names]
        validated = [
            f for f in factors
            if any(k in f.get("name", "").lower() for k in keep_lower)
        ]
        # Safety: if LLM removes everything, fall back to filter_irrelevant_drivers result
        return validated if len(validated) >= 3 else factors
    except Exception as e:
        logger.warning(f"LLM driver validation failed: {e}")
        return factors


# ── MASTER FUNCTION ───────────────────────────────────────────────────────────

def discover_risk_drivers(
    ticker: str,
    company: str,
    sector: str,
    industry: str,
    cik: str | None,
    tavily_client,
    openai_client,
    horizon: str,
    force_refresh: bool = False,
) -> dict:
    """
    Full 3-layer driver discovery. Cached 24 hours per ticker.

    Returns:
        {ticker, company, horizon, data_quality, timestamp,
         factors (all), primary_drivers (top 3)}
    """
    cache_path = CACHE_DIR / f"drivers_{ticker.upper()}.json"

    if not force_refresh and cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass

    # Layer 1: EDGAR 10-K or LLM fallback
    data_quality = "unknown"
    factors: list[dict] = []

    if cik and openai_client:
        try:
            factors = extract_10k_risk_factors(ticker, cik, openai_client)
            if factors:
                data_quality = "EDGAR_10K"
        except Exception as e:
            logger.warning(f"10-K extraction failed: {e}")

    if not factors and openai_client:
        try:
            factors = generate_drivers_from_description(
                ticker, company, sector, industry, openai_client
            )
            if factors:
                data_quality = "LLM_description"
        except Exception as e:
            logger.warning(f"LLM description fallback failed: {e}")

    if not factors:
        factors = _hardcoded_fallback(sector)
        data_quality = "hardcoded_fallback"

    # Layer 2: price proxies
    factors = map_to_price_proxies(factors)

    # Layer 3: news velocity
    factors = score_news_velocity(factors, ticker, company, tavily_client)

    # Combined relevance score + relevance filtering
    factors = compute_relevance(factors)

    # Remove drivers clearly irrelevant to this sector (fast rule-based pass)
    factors = filter_irrelevant_drivers(factors, ticker, sector, industry)

    # LLM validation pass — removes any remaining mismatches (only if openai available)
    if openai_client and data_quality != "hardcoded_fallback":
        factors = validate_drivers_with_llm(
            factors, ticker, company, sector, industry, openai_client
        )

    # Re-score after filtering (priority labels may have shifted)
    factors = compute_relevance(factors)

    result = {
        "ticker": ticker,
        "company": company,
        "sector": sector,
        "industry": industry,
        "horizon": horizon,
        "data_quality": data_quality,
        "timestamp": datetime.now().isoformat(),
        "factors": factors,
        "primary_drivers": [f for f in factors if f["priority"] == "PRIMARY"],
    }

    try:
        cache_path.write_text(json.dumps(result, indent=2, default=str))
    except Exception:
        pass

    return result
