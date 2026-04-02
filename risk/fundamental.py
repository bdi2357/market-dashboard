"""
Fundamental Risk Engine
=======================
All data pulled from yfinance — no paid APIs required.

Sections:
- Valuation Risk (P/E, P/B, P/S, EV/EBITDA, PEG, ERP)
- Balance Sheet Risk (D/E, Z-Score, coverage, Altman)
- Earnings Quality (accruals ratio, surprise history, consistency)
- Cash Flow Risk (FCF, FCF yield, shareholder yield)
- Growth/Value Classification + Regime Mismatch
- News Materiality Scoring
- Composite Fundamental Risk Score (0-100)
"""

import re
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional

# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(info: dict, key: str, default=np.nan):
    v = info.get(key)
    return v if v is not None else default


def _pct_rank(value: float, peer_values: list) -> float:
    """Percentile rank of value in peer_values list (0=lowest, 100=highest)."""
    vals = [v for v in peer_values if v is not None and not np.isnan(v)]
    if not vals or np.isnan(value):
        return np.nan
    return float(np.sum(np.array(vals) <= value) / len(vals) * 100)


# ── peer group ────────────────────────────────────────────────────────────────

def fetch_peer_info(ticker: str, max_peers: int = 20) -> pd.DataFrame:
    """
    Build sector peer group using S&P 500 constituents from Wikipedia.
    Returns DataFrame of peer info dicts (one row per peer).
    Falls back to a hardcoded list if Wikipedia is unavailable.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        sp500 = []

    t0 = yf.Ticker(ticker)
    info0 = t0.info or {}
    sector = info0.get("sector", "")
    industry = info0.get("industry", "")

    rows = []
    candidates = [s for s in sp500 if s != ticker] if sp500 else [
        "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "AVGO",
        "ORCL", "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT",
        "LRCX", "KLAC", "MRVL", "ADI", "NXPI",
    ]

    for sym in candidates:
        if len(rows) >= max_peers:
            break
        try:
            info = yf.Ticker(sym).info or {}
            if info.get("sector") != sector:
                continue
            rows.append({
                "ticker": sym,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "priceToBook": info.get("priceToBook"),
                "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "pegRatio": info.get("pegRatio"),
                "revenueGrowth": info.get("revenueGrowth"),
                "debtToEquity": info.get("debtToEquity"),
                "currentRatio": info.get("currentRatio"),
                "returnOnEquity": info.get("returnOnEquity"),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ── VALUATION RISK ────────────────────────────────────────────────────────────

def valuation_risk(ticker: str, fred_dgs10: Optional[float] = None) -> dict:
    """
    Returns valuation metrics, peer percentile ranks, ERP, and flags.

    Flags:
    - PEG_HIGH: PEG > 2  (priced for perfection)
    - VALUATION_STRETCHED: above 80th percentile on 3+ metrics
    - ERP_COMPRESSED: equity risk premium < 1%
    """
    info = yf.Ticker(ticker).info or {}
    peers_df = fetch_peer_info(ticker)

    metrics = {
        "trailingPE":   _safe(info, "trailingPE"),
        "forwardPE":    _safe(info, "forwardPE"),
        "priceToBook":  _safe(info, "priceToBook"),
        "priceToSales": _safe(info, "priceToSalesTrailing12Months"),
        "evToEbitda":   _safe(info, "enterpriseToEbitda"),
        "pegRatio":     _safe(info, "pegRatio"),
    }

    # Sector percentile ranks
    ranks = {}
    for col, key in [
        ("trailingPE", "trailingPE"),
        ("forwardPE", "forwardPE"),
        ("priceToBook", "priceToBook"),
        ("priceToSales", "priceToSalesTrailing12Months"),
        ("evToEbitda", "enterpriseToEbitda"),
    ]:
        if col in peers_df.columns:
            peer_vals = peers_df[col].dropna().tolist()
            ranks[col + "_pct"] = _pct_rank(metrics.get(col, np.nan), peer_vals)
        else:
            ranks[col + "_pct"] = np.nan

    # Equity Risk Premium = Earnings Yield - 10Y Treasury
    erp = np.nan
    if not np.isnan(metrics["trailingPE"]) and metrics["trailingPE"] > 0:
        earnings_yield = 1 / metrics["trailingPE"]
        rf = (fred_dgs10 / 100) if fred_dgs10 else 0.043  # fallback ~4.3%
        erp = earnings_yield - rf

    # Flags
    flags = []
    peg = metrics["pegRatio"]
    if not np.isnan(peg) and peg > 2:
        flags.append("PEG_HIGH")

    stretched_count = sum(
        1 for k in ["trailingPE_pct", "forwardPE_pct", "priceToBook_pct", "priceToSales_pct", "evToEbitda_pct"]
        if not np.isnan(ranks.get(k, np.nan)) and ranks.get(k, 0) > 80
    )
    if stretched_count >= 3:
        flags.append("VALUATION_STRETCHED")

    if not np.isnan(erp) and erp < 0.01:
        flags.append("ERP_COMPRESSED")

    # Score: average of percentile ranks (higher rank = higher valuation risk)
    valid_ranks = [v for v in ranks.values() if not np.isnan(v)]
    score = float(np.mean(valid_ranks)) if valid_ranks else 50.0

    return {
        "metrics": metrics,
        "peer_ranks": ranks,
        "equity_risk_premium": erp,
        "flags": flags,
        "score": score,
        "peer_count": len(peers_df),
    }


# ── BALANCE SHEET RISK ────────────────────────────────────────────────────────

def balance_sheet_risk(ticker: str) -> dict:
    """
    Altman Z-Score, leverage, and coverage ratios.

    Altman Z (public manufacturers):
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

    Thresholds: Z > 2.99 safe | 1.81-2.99 grey zone | < 1.81 distress

    Flags:
    - ALTMAN_DISTRESS: Z < 1.81
    - ALTMAN_GREY: 1.81 <= Z < 2.99
    - INTEREST_COVERAGE_LOW: interest coverage < 2
    - DEBT_SURGE: total debt grew > 20% YoY
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    bs = t.balance_sheet        # columns = dates (most recent first)
    cf = t.cashflow
    inc = t.financials

    def _get(df, *rows):
        """Try row names in order, return most recent value."""
        if df is None or df.empty:
            return np.nan
        for row in rows:
            for idx in df.index:
                if str(idx).lower() == str(row).lower():
                    v = df.loc[idx].iloc[0]
                    return float(v) if pd.notna(v) else np.nan
        return np.nan

    # Balance sheet items
    total_assets       = _get(bs, "Total Assets")
    total_liabilities  = _get(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
    current_assets     = _get(bs, "Current Assets")
    current_liabilities= _get(bs, "Current Liabilities")
    inventory          = _get(bs, "Inventory")
    retained_earnings  = _get(bs, "Retained Earnings")
    total_debt         = _get(bs, "Total Debt", "Long Term Debt")
    stockholders_eq    = _get(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")

    # Income statement items
    ebit               = _get(inc, "EBIT", "Operating Income")
    interest_expense   = _get(inc, "Interest Expense")
    revenue            = _get(inc, "Total Revenue")

    # Cash flow
    operating_cf       = _get(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")

    market_cap         = _safe(info, "marketCap", np.nan)

    # Altman Z-Score
    z_score = np.nan
    if not any(np.isnan(x) for x in [total_assets, current_assets, current_liabilities,
                                       retained_earnings, ebit, market_cap,
                                       total_liabilities, revenue]) and total_assets > 0:
        x1 = (current_assets - current_liabilities) / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities if total_liabilities > 0 else np.nan
        x5 = revenue / total_assets
        if not np.isnan(x4):
            z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    # Ratios
    debt_to_equity     = total_debt / stockholders_eq if (stockholders_eq and stockholders_eq > 0) else np.nan
    current_ratio      = current_assets / current_liabilities if (current_liabilities and current_liabilities > 0) else np.nan
    quick_ratio        = _safe(info, "quickRatio")
    interest_coverage  = abs(ebit / interest_expense) if (not np.isnan(ebit) and not np.isnan(interest_expense) and interest_expense != 0) else np.nan
    net_debt           = total_debt - _get(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments") if not np.isnan(total_debt) else np.nan

    # YoY debt change
    debt_yoy_pct = np.nan
    if bs is not None and not bs.empty and len(bs.columns) >= 2:
        debt_now  = bs.loc[[i for i in bs.index if "total debt" in str(i).lower() or "long term debt" in str(i).lower()]]
        if not debt_now.empty:
            d0 = float(debt_now.iloc[0, 0]) if pd.notna(debt_now.iloc[0, 0]) else np.nan
            d1 = float(debt_now.iloc[0, 1]) if pd.notna(debt_now.iloc[0, 1]) else np.nan
            if not np.isnan(d0) and not np.isnan(d1) and d1 != 0:
                debt_yoy_pct = (d0 - d1) / abs(d1)

    # Net Debt / EBITDA
    ebitda = _safe(info, "ebitda")
    net_debt_ebitda = net_debt / ebitda if (not np.isnan(net_debt) and not np.isnan(ebitda) and ebitda > 0) else np.nan

    # Flags
    flags = []
    if not np.isnan(z_score):
        if z_score < 1.81:
            flags.append("ALTMAN_DISTRESS")
        elif z_score < 2.99:
            flags.append("ALTMAN_GREY")
    if not np.isnan(interest_coverage) and interest_coverage < 2:
        flags.append("INTEREST_COVERAGE_LOW")
    if not np.isnan(debt_yoy_pct) and debt_yoy_pct > 0.20:
        flags.append("DEBT_SURGE")

    # Score: 0=safe, 100=distress
    score = 50.0
    subscores = []
    if not np.isnan(z_score):
        # Map Z: <1.81 -> 100, >2.99 -> 0
        subscores.append(float(np.clip((2.99 - z_score) / (2.99 - 1.81) * 100, 0, 100)))
    if not np.isnan(debt_to_equity):
        subscores.append(float(np.clip(debt_to_equity / 3 * 100, 0, 100)))
    if not np.isnan(interest_coverage):
        subscores.append(float(np.clip((5 - interest_coverage) / 5 * 100, 0, 100)))
    if subscores:
        score = float(np.mean(subscores))

    return {
        "z_score": z_score,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "interest_coverage": interest_coverage,
        "net_debt_ebitda": net_debt_ebitda,
        "debt_yoy_pct": debt_yoy_pct,
        "flags": flags,
        "score": score,
    }


# ── EARNINGS QUALITY & SURPRISE ───────────────────────────────────────────────

def earnings_quality(ticker: str) -> dict:
    """
    Accruals ratio, EPS surprise history, revenue growth consistency.

    Accruals ratio = (Net Income - CFO) / Total Assets
    High (>5%) signals low earnings quality (income outpaces cash).

    Flags:
    - EARNINGS_RISK: 2+ consecutive misses OR accruals ratio > 5%
    - ACCRUALS_HIGH: accruals ratio > 5%
    - CONSECUTIVE_MISSES: 2+ consecutive negative surprises
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    bs   = t.balance_sheet
    cf   = t.cashflow
    inc  = t.financials

    def _get(df, *rows):
        if df is None or df.empty:
            return np.nan
        for row in rows:
            for idx in df.index:
                if str(idx).lower() == str(row).lower():
                    v = df.loc[idx].iloc[0]
                    return float(v) if pd.notna(v) else np.nan
        return np.nan

    # Accruals ratio
    net_income   = _get(inc, "Net Income")
    cfo          = _get(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    total_assets = _get(bs, "Total Assets")

    accruals_ratio = np.nan
    if not any(np.isnan(x) for x in [net_income, cfo, total_assets]) and total_assets > 0:
        accruals_ratio = (net_income - cfo) / total_assets

    # EPS surprise history
    surprise_history = []
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            ed = ed.dropna(subset=["EPS Estimate", "Reported EPS"])
            ed = ed.head(8)
            for dt, row in ed.iterrows():
                est = row.get("EPS Estimate", np.nan)
                act = row.get("Reported EPS", np.nan)
                if pd.notna(est) and pd.notna(act) and est != 0:
                    surprise_pct = (act - est) / abs(est)
                    surprise_history.append({
                        "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                        "estimate": round(float(est), 4),
                        "actual": round(float(act), 4),
                        "surprise_pct": round(float(surprise_pct), 4),
                        "beat": bool(act >= est),
                    })
    except Exception:
        pass

    # Consecutive misses
    consecutive_misses = 0
    for item in surprise_history:
        if not item["beat"]:
            consecutive_misses += 1
        else:
            break

    # Revenue growth consistency (std dev of quarterly growth)
    rev_growth_std = np.nan
    try:
        q_inc = t.quarterly_financials
        if q_inc is not None and not q_inc.empty:
            for idx in q_inc.index:
                if "total revenue" in str(idx).lower():
                    rev = q_inc.loc[idx].dropna().sort_index()
                    if len(rev) >= 3:
                        g = rev.pct_change().dropna()
                        rev_growth_std = float(g.std())
                    break
    except Exception:
        pass

    # Flags
    flags = []
    if not np.isnan(accruals_ratio) and abs(accruals_ratio) > 0.05:
        flags.append("ACCRUALS_HIGH")
    if consecutive_misses >= 2:
        flags.append("CONSECUTIVE_MISSES")
    if flags:
        flags.append("EARNINGS_RISK")

    # Score
    score = 50.0
    subscores = []
    if not np.isnan(accruals_ratio):
        subscores.append(float(np.clip(abs(accruals_ratio) / 0.10 * 100, 0, 100)))
    if consecutive_misses > 0:
        subscores.append(float(np.clip(consecutive_misses / 4 * 100, 0, 100)))
    if surprise_history:
        beat_rate = np.mean([s["beat"] for s in surprise_history])
        subscores.append(float((1 - beat_rate) * 100))
    if subscores:
        score = float(np.mean(subscores))

    return {
        "accruals_ratio": accruals_ratio,
        "surprise_history": surprise_history,
        "consecutive_misses": consecutive_misses,
        "rev_growth_std": rev_growth_std,
        "flags": flags,
        "score": score,
    }


# ── CASH FLOW RISK ────────────────────────────────────────────────────────────

def cashflow_risk(ticker: str) -> dict:
    """
    FCF = Operating CF - CapEx
    FCF Yield = FCF / Market Cap
    Shareholder yield = FCF yield + buyback yield + dividend yield

    Flags:
    - FCF_NEGATIVE_STREAK: FCF < 0 for 2+ consecutive years
    - FCF_YIELD_NEGATIVE_PREMIUM: FCF yield < 0 while P/E > 20
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    cf = t.cashflow

    annual_fcf = []
    annual_ocf = []
    annual_capex = []
    annual_revenue = []

    if cf is not None and not cf.empty:
        ocf_row = next(
            (idx for idx in cf.index if "operating cash flow" in str(idx).lower()
             or "cash flow from continuing operating" in str(idx).lower()),
            None
        )
        capex_row = next(
            (idx for idx in cf.index if "capital expenditure" in str(idx).lower()
             or "purchase of ppe" in str(idx).lower()),
            None
        )
        for col in cf.columns[:4]:  # last 4 years
            ocf = float(cf.loc[ocf_row, col]) if ocf_row and pd.notna(cf.loc[ocf_row, col]) else np.nan
            capex = float(cf.loc[capex_row, col]) if capex_row and pd.notna(cf.loc[capex_row, col]) else np.nan
            annual_ocf.append(ocf)
            annual_capex.append(capex if not np.isnan(capex) else 0)
            fcf = ocf - abs(capex) if not np.isnan(ocf) and not np.isnan(capex) else (ocf if not np.isnan(ocf) else np.nan)
            annual_fcf.append(fcf)

    market_cap   = _safe(info, "marketCap")
    latest_fcf   = annual_fcf[0] if annual_fcf and not np.isnan(annual_fcf[0]) else np.nan
    fcf_yield    = latest_fcf / market_cap if (not np.isnan(latest_fcf) and not np.isnan(market_cap) and market_cap > 0) else np.nan

    # FCF margin trend (need revenue)
    try:
        inc = t.financials
        rev_row = next((idx for idx in inc.index if "total revenue" in str(idx).lower()), None)
        if rev_row:
            annual_revenue = [
                float(inc.loc[rev_row, col]) if pd.notna(inc.loc[rev_row, col]) else np.nan
                for col in inc.columns[:4]
            ]
    except Exception:
        annual_revenue = []

    fcf_margins = []
    for fcf, rev in zip(annual_fcf, annual_revenue):
        if not np.isnan(fcf) and not np.isnan(rev) and rev > 0:
            fcf_margins.append(fcf / rev)
        else:
            fcf_margins.append(np.nan)

    # FCF volatility
    valid_fcf = [f for f in annual_fcf if not np.isnan(f)]
    fcf_vol = float(np.std(valid_fcf) / abs(np.mean(valid_fcf))) if len(valid_fcf) >= 2 and np.mean(valid_fcf) != 0 else np.nan

    # Shareholder yield
    dividend_yield = _safe(info, "dividendYield", 0) or 0
    buyback_yield  = _safe(info, "buybackYield", 0) or 0
    shareholder_yield = (fcf_yield or 0) + buyback_yield + dividend_yield

    # Consecutive negative FCF
    neg_streak = 0
    for f in annual_fcf:
        if not np.isnan(f) and f < 0:
            neg_streak += 1
        else:
            break

    trailing_pe = _safe(info, "trailingPE")
    flags = []
    if neg_streak >= 2:
        flags.append("FCF_NEGATIVE_STREAK")
    if not np.isnan(fcf_yield) and fcf_yield < 0 and not np.isnan(trailing_pe) and trailing_pe > 20:
        flags.append("FCF_YIELD_NEGATIVE_PREMIUM")

    # Score
    score = 50.0
    subscores = []
    if not np.isnan(fcf_yield):
        subscores.append(float(np.clip((-fcf_yield + 0.02) / 0.06 * 100, 0, 100)))
    if not np.isnan(fcf_vol):
        subscores.append(float(np.clip(fcf_vol / 2 * 100, 0, 100)))
    subscores.append(min(neg_streak / 3 * 100, 100))
    if subscores:
        score = float(np.mean(subscores))

    return {
        "annual_fcf": annual_fcf,
        "annual_ocf": annual_ocf,
        "annual_capex": annual_capex,
        "fcf_yield": fcf_yield,
        "fcf_margins": fcf_margins,
        "fcf_volatility": fcf_vol,
        "dividend_yield": dividend_yield,
        "buyback_yield": buyback_yield,
        "shareholder_yield": shareholder_yield,
        "neg_fcf_streak": neg_streak,
        "flags": flags,
        "score": score,
    }


# ── GROWTH / VALUE CLASSIFICATION ────────────────────────────────────────────

def growth_value_classification(ticker: str, macro_regime: str = "MIXED") -> dict:
    """
    Classify stock as:
      Deep Value / Value / Blend / Growth / Aggressive Growth

    Based on trailing P/E percentile within sector + revenue growth rate.

    Regime mismatch:
    - Aggressive Growth in RISK_OFF → REGIME_MISMATCH
    - Deep Value in RISK_ON → mild mismatch (opportunity signal)
    """
    info = yf.Ticker(ticker).info or {}
    peers_df = fetch_peer_info(ticker, max_peers=20)

    pe = _safe(info, "trailingPE")
    rev_growth = _safe(info, "revenueGrowth")  # YoY

    pe_pct = np.nan
    if "trailingPE" in peers_df.columns:
        pe_pct = _pct_rank(pe, peers_df["trailingPE"].dropna().tolist())

    # Classify by P/E percentile AND growth
    label = "Blend"
    if not np.isnan(pe_pct) and not np.isnan(rev_growth):
        if pe_pct < 20 and rev_growth < 0.05:
            label = "Deep Value"
        elif pe_pct < 40:
            label = "Value"
        elif pe_pct > 80 and rev_growth > 0.20:
            label = "Aggressive Growth"
        elif pe_pct > 60:
            label = "Growth"
    elif not np.isnan(pe_pct):
        if pe_pct < 25:
            label = "Deep Value"
        elif pe_pct < 45:
            label = "Value"
        elif pe_pct > 75:
            label = "Growth"

    # Regime mismatch
    flags = []
    regime_favors = {
        "RISK_ON": "Growth",
        "RISK_OFF": "Value",
        "STAGFLATION_PROXY": "Value",
        "MIXED": "Blend",
    }
    favored = regime_favors.get(macro_regime, "Blend")
    mismatch_pairs = {
        ("Aggressive Growth", "RISK_OFF"),
        ("Aggressive Growth", "STAGFLATION_PROXY"),
        ("Deep Value", "RISK_ON"),
    }
    if (label, macro_regime) in mismatch_pairs:
        flags.append("REGIME_MISMATCH")

    return {
        "classification": label,
        "pe_percentile": pe_pct,
        "revenue_growth": rev_growth,
        "macro_regime": macro_regime,
        "regime_favors": favored,
        "flags": flags,
    }


# ── NEWS MATERIALITY ──────────────────────────────────────────────────────────

# Keywords that indicate material news
_MATERIAL_PATTERNS = {
    3: [
        r"earning[s]?\s+(report|beat|miss|results|surprise)",
        r"guidance\s+(raise|cut|lower|increase|update)",
        r"merger|acquisition|acquires?|acquired|takeover|buyout",
        r"spinoff|spin-off|divestiture|divests?",
        r"ceo|cfo|chief executive|chief financial|executive\s+chair",
        r"credit\s+rating|downgrad|upgrad.*rating|moody|s&p|fitch",
        r"sec\s+(investigation|charges|subpoena|settlement)",
        r"fda\s+(approv|reject|denial|complete response)",
        r"regulatory\s+(approv|block|fine|penalty)",
        r"share\s+buyback|repurchase\s+program",
        r"dividend\s+(cut|suspend|increase|initiat)",
        r"debt\s+(issu|offer|refinanc|default|restructur)",
        r"layoff|restructur|workforce\s+reduction",
    ],
    2: [
        r"analyst\s+(upgrade|downgrade|initiat|reiterat)",
        r"price\s+target\s+(raised?|lowered?|cut|increased?)",
        r"earnings\s+per\s+share|eps",
        r"revenue\s+(beat|miss|exceed|below)",
        r"operating\s+(income|margin|profit)",
        r"interest\s+rate|federal\s+reserve|fed\s+(hike|cut|pause)",
        r"inflation|cpi|pce|gdp",
        r"supply\s+chain|semiconductor\s+(shortage|supply)",
        r"partnership|joint\s+venture|strategic\s+alliance",
        r"product\s+(launch|recall|delay)",
        r"antitrust|ftc|doj\s+investig",
    ],
}

_NOT_MATERIAL = [
    r"stock\s+(up|down|rises?|falls?|gains?|drops?)\s+\d",
    r"(monday|tuesday|wednesday|thursday|friday|weekend)\s+market",
    r"3\s+stocks?\s+to\s+(buy|watch|avoid)",
    r"best\s+stocks?\s+to\s+buy",
    r"here['']s\s+why",
]

_SENTIMENT_POSITIVE = [
    r"beat|exceed|surpass|record|strong|growth|gain|raise|upgrade|approv|buyback|partnership",
]
_SENTIMENT_NEGATIVE = [
    r"miss|below|weak|cut|lower|downgrade|decline|loss|investigat|recall|layoff|fine|default|concern|risk",
]


def _score_headline(title: str) -> tuple[int, str]:
    """Returns (materiality_score 1-3, sentiment tag)."""
    t = title.lower()

    # Check not-material patterns first
    for pat in _NOT_MATERIAL:
        if re.search(pat, t):
            return 1, "NEUTRAL"

    # Score materiality
    score = 1
    for mat_score, patterns in sorted(_MATERIAL_PATTERNS.items(), reverse=True):
        for pat in patterns:
            if re.search(pat, t):
                score = mat_score
                break
        if score == mat_score:
            break

    # Sentiment
    pos = any(re.search(p, t) for p in _SENTIMENT_POSITIVE)
    neg = any(re.search(p, t) for p in _SENTIMENT_NEGATIVE)
    if pos and neg:
        sentiment = "MIXED"
    elif pos:
        sentiment = "POSITIVE"
    elif neg:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return score, sentiment


def _categorize_headline(title: str, ticker: str, ticker_name: str) -> str:
    t = title.lower()
    name_lower = ticker_name.lower() if ticker_name else ""
    ticker_lower = ticker.lower()
    if ticker_lower in t or (name_lower and any(w in t for w in name_lower.split()[:2] if len(w) > 3)):
        return "Company News"
    macro_words = ["federal reserve", "fed ", "inflation", "interest rate", "treasury", "gdp", "recession", "cpi", "pce"]
    if any(w in t for w in macro_words):
        return "Macro News"
    return "Sector News"


def fetch_material_news(ticker: str, ticker_name: str = "") -> list[dict]:
    """
    Return list of material news items (score >= 2), sorted by materiality desc.
    Each item: {date, title, source, url, materiality, sentiment, category}
    """
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    results = []
    for item in news:
        content = item.get("content", {})
        title = content.get("title") or item.get("title", "")
        if not title:
            continue

        score, sentiment = _score_headline(title)
        if score < 2:
            continue

        # Extract date
        pub = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(pub, (int, float)):
            date_str = pd.Timestamp(pub, unit="s").strftime("%Y-%m-%d")
        elif isinstance(pub, str):
            date_str = pub[:10]
        else:
            date_str = ""

        # Extract source
        provider = content.get("provider", {})
        source = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher", "")

        # Extract URL
        click_through = content.get("clickThroughUrl", {}) or {}
        url = click_through.get("url") or item.get("link", "")

        category = _categorize_headline(title, ticker, ticker_name)

        results.append({
            "date": date_str,
            "title": title,
            "source": source,
            "url": url,
            "materiality": score,
            "sentiment": sentiment,
            "category": category,
        })

    results.sort(key=lambda x: x["materiality"], reverse=True)
    return results


# ── COMPOSITE FUNDAMENTAL SCORE ───────────────────────────────────────────────

def composite_fundamental_score(
    val_result: dict,
    bs_result: dict,
    eq_result: dict,
    cf_result: dict,
    gv_result: dict,
) -> dict:
    """
    Weighted composite fundamental risk score (0=safe, 100=highest risk).

    Weights:
    - Valuation:      25%
    - Balance Sheet:  30%
    - Earnings Quality: 20%
    - Cash Flow:      20%
    - Regime Mismatch: 5%
    """
    weights = {
        "Valuation Risk":      (val_result.get("score", 50), 0.25),
        "Balance Sheet Risk":  (bs_result.get("score", 50),  0.30),
        "Earnings Quality":    (eq_result.get("score", 50),  0.20),
        "Cash Flow Risk":      (cf_result.get("score", 50),  0.20),
        "Regime Mismatch":     (100 if "REGIME_MISMATCH" in gv_result.get("flags", []) else 20, 0.05),
    }

    total = sum(score * w for score, w in weights.values())
    components = {k: round(score * w, 1) for k, (score, w) in weights.items()}

    # All flags
    all_flags = (
        val_result.get("flags", [])
        + bs_result.get("flags", [])
        + eq_result.get("flags", [])
        + cf_result.get("flags", [])
        + gv_result.get("flags", [])
    )

    return {
        "total": round(total, 1),
        "components": components,
        "all_flags": all_flags,
    }
