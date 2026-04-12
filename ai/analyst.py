"""
AI Research Analyst
===================
Combines all computed risk signals with live web research (Tavily)
and DeepSeek via OpenRouter to generate institutional-grade reports.

API keys loaded from .env at project root via python-dotenv.
"""

import os
import logging
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

# ── lazy clients (instantiated once, only if keys present) ────────────────────

_openai_client = None
_tavily_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            return None
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )
        except Exception as e:
            logger.warning(f"OpenRouter client init failed: {e}")
            return None
    return _openai_client


def _get_tavily():
    global _tavily_client
    if _tavily_client is None:
        key = os.getenv("TAVILY_API_KEY", "")
        if not key:
            return None
        try:
            from tavily import TavilyClient
            _tavily_client = TavilyClient(api_key=key)
        except Exception as e:
            logger.warning(f"Tavily client init failed: {e}")
            return None
    return _tavily_client


# ── PART 1: BUILD TICKER CONTEXT ──────────────────────────────────────────────

def build_ticker_context(
    ticker: str,
    prices: pd.Series,
    volume: pd.Series,
    bench_prices: pd.Series,
    info: dict,
    macro_df: pd.DataFrame,
    fund_scores: dict | None = None,
    insider_signals: dict | None = None,
    inst_signals: dict | None = None,
    activist_df: pd.DataFrame | None = None,
    factor_results: dict | None = None,
    driver_profile: dict | None = None,
    horizon: str = "1 Month",
) -> dict:
    """
    Collect all computed metrics into a single context dict for the LLM.
    Imports risk functions lazily to avoid circular imports.
    """
    from risk.metrics import (
        rolling_realized_vol, vol_regime, max_drawdown, calmar_ratio,
        sortino_ratio, ulcer_index, historical_var_cvar, cornish_fisher_var,
        rolling_beta, residual_vol, information_ratio, composite_risk_score,
        amihud_illiquidity,
    )

    r = prices.pct_change().dropna()
    lr = np.log(prices / prices.shift(1)).dropna()

    # ── Market risk ───────────────────────────────────────────────────────────
    vol_df = rolling_realized_vol(prices, windows=[21, 63, 252])
    vol_21 = float(vol_df["vol_21d"].iloc[-1]) if not vol_df.empty else np.nan
    vol_63 = float(vol_df["vol_63d"].iloc[-1]) if not vol_df.empty else np.nan

    # Vol regime + percentile
    _vr_series = vol_regime(vol_df["vol_21d"].dropna())
    _current_vol_regime = str(_vr_series.iloc[-1]) if not _vr_series.empty else "unknown"
    _vol_rank = vol_df["vol_21d"].dropna()
    _vol_pct = float(
        (_vol_rank <= _vol_rank.iloc[-1]).mean() * 100
    ) if len(_vol_rank) > 5 else np.nan

    mdd_data = max_drawdown(prices)
    var_data = historical_var_cvar(r)
    score_data = composite_risk_score(prices, volume, bench_prices)
    _beta = float(rolling_beta(prices, bench_prices).iloc[-1]) if len(prices) > 70 else np.nan
    _ires = residual_vol(prices, bench_prices)
    _idio_pct = float(_ires.iloc[-1] / vol_21 * 100) if vol_21 and not np.isnan(_ires.iloc[-1]) else np.nan

    ann_ret = float((1 + r.mean()) ** 252 - 1)
    ann_vol = float(r.std() * np.sqrt(252))
    _ir = information_ratio(prices, bench_prices)

    # ── Macro ─────────────────────────────────────────────────────────────────
    macro_ctx: dict = {}
    if macro_df is not None and not macro_df.empty:
        last = macro_df.ffill().iloc[-1]
        macro_ctx = {
            "yield_10y": float(last.get("DGS10", np.nan)),
            "hy_spread": float(last.get("BAMLH0A0HYM2", np.nan)),
            "yield_curve": float(last.get("T10Y2Y", np.nan)),
            "vix_level": float(last.get("VIXCLS", np.nan)),
        }
        # VIX regime
        vix = macro_ctx["vix_level"]
        macro_ctx["vix_regime"] = (
            "low" if vix < 15 else "medium" if vix < 25 else
            "high" if vix < 35 else "extreme"
        ) if not np.isnan(vix) else "unknown"

        # Macro regime classification
        from risk.metrics import classify_macro_regime
        macro_regime_s = classify_macro_regime(macro_df)
        macro_ctx["macro_regime"] = str(macro_regime_s.iloc[-1]) if not macro_regime_s.empty else "UNKNOWN"

    # ── Fundamentals ─────────────────────────────────────────────────────────
    fund_ctx: dict = {}
    if fund_scores:
        fund_ctx["fundamental_risk_score"] = fund_scores.get("total", np.nan)
        mkt = score_data["total"]
        fund_ctx["divergence"] = abs(mkt - fund_scores.get("total", mkt))
        # Pull individual fundamental results if stored
        for key in ["pe_ratio", "forward_pe", "pe_sector_percentile", "ev_ebitda",
                    "pb_ratio", "peg_ratio", "altman_z", "altman_zone",
                    "fcf_yield", "fcf_trend", "fcf_margin",
                    "debt_equity", "net_debt_ebitda", "interest_coverage",
                    "accruals_ratio", "earnings_surprise_trend",
                    "revenue_growth_consistency", "all_flags"]:
            if key in fund_scores:
                fund_ctx[key] = fund_scores[key]

    # ── Smart money ───────────────────────────────────────────────────────────
    smart_ctx: dict = {}
    if insider_signals:
        cluster = insider_signals.get("cluster_signal")
        smart_ctx["insider_signal"] = (
            "CLUSTER_BUY" if cluster == "BUY" else
            "CLUSTER_SELL" if cluster == "SELL" else "NEUTRAL"
        )
        smart_ctx["insider_net_30d"] = insider_signals.get("net_30d", 0)
        smart_ctx["insider_net_90d"] = insider_signals.get("net_90d", 0)
        bsr = insider_signals.get("buy_sell_ratio_90d", np.nan)
        smart_ctx["insider_buy_sell_ratio"] = float(bsr) if not (isinstance(bsr, float) and np.isnan(bsr)) else None

    if inst_signals:
        smart_ctx["funds_adding"] = inst_signals.get("n_adding", 0)
        smart_ctx["funds_reducing"] = inst_signals.get("n_reducing", 0)
        smart_ctx["new_positions"] = inst_signals.get("n_new", 0)
        smart_ctx["closed_positions"] = inst_signals.get("n_closed", 0)
        smart_ctx["inst_signal"] = inst_signals.get("signal", "NEUTRAL")

    if activist_df is not None and not activist_df.empty:
        smart_ctx["activist_present"] = True
        row = activist_df.iloc[0]
        smart_ctx["activist_name"] = str(row.get("filer_name", ""))
        smart_ctx["activist_pct"] = float(row.get("ownership_pct", 0) or 0)
        smart_ctx["activist_intent"] = bool(row.get("activist_intent", False))
    else:
        smart_ctx["activist_present"] = False

    # ── Factor decomposition (optional) ──────────────────────────────────────
    factor_ctx: dict = {}
    if factor_results:
        for k in ["alpha", "alpha_tstat", "r_squared", "systematic_risk_pct",
                  "idiosyncratic_risk_pct", "beta_market", "beta_size",
                  "beta_value", "beta_profitability", "beta_investment",
                  "beta_momentum", "total_vol_annualized", "idio_vol",
                  "factor_contributions", "factor_model_quality",
                  "interpretation_flags"]:
            if k in factor_results:
                val = factor_results[k]
                if isinstance(val, pd.Series):
                    continue  # skip time-series objects
                factor_ctx[k] = val

    # ── Driver profile (optional) ─────────────────────────────────────────────
    driver_ctx: dict = {}
    if driver_profile:
        primary = driver_profile.get("primary_drivers", [])
        driver_ctx["top_driver_name"] = primary[0]["name"] if primary else None
        driver_ctx["top_driver_factor"] = primary[0].get("external_factor") if primary else None
        driver_ctx["top_driver_score"] = primary[0].get("current_relevance_score") if primary else None
        driver_ctx["top_driver_moving"] = primary[0].get("factor_is_moving", False) if primary else False
        driver_ctx["top_driver_1m_return"] = primary[0].get("factor_1m_return") if primary else None
        driver_ctx["data_quality"] = driver_profile.get("data_quality")

    return {
        "ticker": ticker,
        "company": info.get("name", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("market_cap"),
        "horizon": horizon,
        # Market risk
        "composite_risk_score": score_data["total"],
        "ann_volatility": round(ann_vol, 4),
        "ann_return": round(ann_ret, 4),
        "vol_21d": round(vol_21, 4) if not np.isnan(vol_21) else None,
        "vol_63d": round(vol_63, 4) if not np.isnan(vol_63) else None,
        "vol_regime": _current_vol_regime,
        "vol_percentile": round(_vol_pct, 1) if not np.isnan(_vol_pct) else None,
        "max_drawdown": round(float(mdd_data["max_drawdown"]), 4),
        "calmar": round(float(calmar_ratio(prices)), 3),
        "sortino": round(float(sortino_ratio(prices)), 3),
        "ulcer_index": round(float(ulcer_index(prices).iloc[-1]), 4),
        "var_95": round(float(var_data["var_95"]), 4),
        "var_99": round(float(var_data["var_99"]), 4),
        "cvar_95": round(float(var_data["cvar_95"]), 4),
        "cvar_99": round(float(var_data["cvar_99"]), 4),
        "skewness": round(float(r.skew()), 3),
        "excess_kurtosis": round(float(r.kurtosis()), 3),
        "beta_spy": round(_beta, 3) if not np.isnan(_beta) else None,
        "idiosyncratic_risk_pct": round(_idio_pct, 1) if _idio_pct and not np.isnan(_idio_pct) else None,
        "information_ratio": round(float(_ir), 3) if _ir is not None and not np.isnan(_ir) else None,
        # Macro
        **macro_ctx,
        # Fundamentals
        **fund_ctx,
        # Smart money
        **smart_ctx,
        # Factor model
        **factor_ctx,
        # Driver profile
        **driver_ctx,
    }


def _fmt_val(v) -> str:
    """Format a context value for LLM prompt."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if np.isnan(v):
            return "N/A"
        if abs(v) < 1:
            return f"{v:.4f}"
        return f"{v:.2f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt_val(vv)}" for k, vv in v.items())
    if isinstance(v, list):
        return "; ".join(str(x) for x in v[:10])
    return str(v)


# ── PART 2: WEB RESEARCH ──────────────────────────────────────────────────────

_NEWS_DOMAINS = [
    "reuters.com", "wsj.com", "ft.com", "seekingalpha.com",
    "sec.gov", "barrons.com", "bloomberg.com", "cnbc.com",
    "marketwatch.com", "thestreet.com",
]


def fetch_web_research(
    ticker: str,
    company: str,
    sector: str,
    driver_profile: dict | None = None,
    horizon: str = "1 Month",
) -> list[dict]:
    """
    Returns up to 10 research results.
    Sources (in order):
      1. Polygon.io news (structured, reliable, free tier)
      2. Tavily web search (driver-specific + earnings + credit queries)
    """
    results: list[dict] = []
    seen_urls: set[str] = set()

    # 1. Polygon news (primary — structured, no rate issues for news endpoint)
    try:
        from data.polygon_fetcher import polygon_fetch_news
        for item in polygon_fetch_news(ticker, limit=5):
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append({
                    "title": item["title"],
                    "url": url,
                    "content": item.get("summary", "")[:500],
                    "score": 0.9,
                    "source": item.get("source", "Polygon News"),
                })
    except Exception as e:
        logger.debug(f"Polygon news fetch failed: {e}")

    # 2. Tavily web search
    client = _get_tavily()
    if not client:
        logger.warning("Tavily key missing — skipping web research")
        return results

    queries: list[str] = []

    # Driver-specific queries (highest priority)
    if driver_profile:
        for d in driver_profile.get("primary_drivers", [])[:3]:
            ext = d.get("external_factor", "")
            if ext:
                queries.append(f"{ext} outlook {company} impact")

    # Horizon-specific near-term catalysts
    if horizon in ("1 Week", "1 Month"):
        queries.append(f"{ticker} {company} news catalyst this week")
        queries.append(f"{sector} risk near term 2026")

    # Always include earnings + credit
    queries.append(f"{ticker} earnings guidance analyst 2026")
    queries.append(f"{company} credit debt refinancing")

    for query in queries[:7]:
        try:
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_domains=_NEWS_DOMAINS,
            )
            for r in resp.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": (r.get("content", "") or "")[:500],
                        "score": float(r.get("score", 0)),
                    })
        except Exception as e:
            logger.warning(f"Tavily query failed ({query[:50]}…): {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


# ── PART 3: GENERATE RESEARCH REPORT ─────────────────────────────────────────

def _make_analyst_system(horizon: str, top_driver_name: str | None, top_driver_factor: str | None, top_driver_score: float | None, top_driver_moving: bool) -> str:
    moving_str = "YES — actively moving" if top_driver_moving else "stable"
    driver_note = ""
    if top_driver_name:
        _score_fmt = f"{top_driver_score:.2f}" if top_driver_score is not None else "N/A"
        _active_str = "⚠️ ACTIVE" if top_driver_moving else "stable"
        driver_note = f"""
PRIMARY DRIVER IN EFFECT: {top_driver_name}
External factor: {top_driver_factor}
Relevance score: {_score_fmt} — {_active_str}
"""
    return f"""You are a senior equity risk analyst at a quantitative hedge fund.
Your edge: surfacing NON-OBVIOUS risks that standard providers (Yahoo Finance, WSJ) miss.
{driver_note}
ABSOLUTE RULES:
1. Address PRIMARY drivers (highest relevance score) BEFORE generic metrics like vol or beta.
   Generic vol is already on Yahoo Finance — it adds zero value here.
2. If a driver has relevance_score > 2.0 it is ACTIVELY IN PLAY. Lead with it.
   Example: crude up 18% + airlines in news → lead with fuel cost, hedge ratio, unhedged P&L impact.
3. Time horizon is {horizon}:
   1 Week  → near-term catalysts, scheduled events, earnings, macro data releases
   1 Month → trend, guidance changes, sector rotation
   3 Months → regime change, fundamental shifts
   1 Year  → thesis, balance sheet trajectory, competitive moat
4. MINIMALISM: one key insight per sentence. Max 4 sentences per section. No bullets > 3 items.
5. YOUR UNIQUE VALUE: connect sector driver to company-specific numbers from EDGAR/fundamentals.
   BAD: "fuel prices are rising"
   GOOD: "Jet fuel up 18% in 3 months. AAL hedges ~40% per 10-K. Unhedged 60% hits COGS directly."
6. Every number you cite must come from the data provided. No fabrication."""


def _build_context_block(ctx: dict) -> str:
    """Format context dict into clean grouped LLM prompt block."""
    sections = {
        "Market Risk": [
            "composite_risk_score", "ann_return", "ann_volatility",
            "vol_21d", "vol_63d", "vol_regime", "vol_percentile",
            "max_drawdown", "calmar", "sortino", "ulcer_index",
            "var_95", "var_99", "cvar_95", "cvar_99",
            "skewness", "excess_kurtosis",
            "beta_spy", "idiosyncratic_risk_pct", "information_ratio",
        ],
        "Macro": [
            "macro_regime", "yield_10y", "hy_spread", "yield_curve",
            "vix_level", "vix_regime",
        ],
        "Fundamentals": [
            "fundamental_risk_score", "divergence",
            "pe_ratio", "forward_pe", "pe_sector_percentile",
            "ev_ebitda", "pb_ratio", "peg_ratio",
            "altman_z", "altman_zone",
            "fcf_yield", "fcf_margin", "debt_equity",
            "net_debt_ebitda", "interest_coverage",
            "accruals_ratio", "all_flags",
        ],
        "Smart Money": [
            "insider_signal", "insider_net_30d", "insider_net_90d",
            "insider_buy_sell_ratio",
            "funds_adding", "funds_reducing", "new_positions", "closed_positions",
            "inst_signal", "activist_present", "activist_name", "activist_pct",
        ],
        "Factor Model": [
            "alpha", "alpha_tstat", "r_squared",
            "systematic_risk_pct", "idiosyncratic_risk_pct",
            "beta_market", "beta_size", "beta_value",
            "beta_profitability", "beta_investment", "beta_momentum",
            "factor_contributions", "factor_model_quality",
        ],
    }
    lines = []
    for section, keys in sections.items():
        block = []
        for k in keys:
            if k in ctx and ctx[k] is not None:
                block.append(f"  {k}: {_fmt_val(ctx[k])}")
        if block:
            lines.append(f"\n[{section}]")
            lines.extend(block)
    return "\n".join(lines)


def _build_driver_block(driver_profile: dict | None) -> str:
    """Format driver profile into LLM prompt block."""
    if not driver_profile:
        return "(No driver profile available)"
    lines = []
    for d in driver_profile.get("factors", [])[:6]:
        status = "⚠️ ACTIVE" if d.get("factor_is_moving") else "stable"
        r1m = d.get("factor_1m_return")
        r1m_str = f"{r1m:+.1f}%" if r1m is not None else "N/A"
        vel = d.get("news_velocity", 0)
        lines.append(
            f"  [{d.get('priority','?')}] {d['name']} | factor: {d.get('external_factor','?')} | "
            f"relevance: {d.get('current_relevance_score',0):.2f} | 1M: {r1m_str} | "
            f"news: {vel:.1f}/2.0 | {status}"
        )
    return "\n".join(lines)


def generate_research_report(
    context: dict,
    web_results: list[dict],
    driver_profile: dict | None = None,
) -> str:
    """
    Generate insight-first, driver-led research report via DeepSeek on OpenRouter.
    Returns plain-text Markdown string.
    """
    client = _get_openai()
    if not client:
        return "_AI report unavailable: OPENROUTER_API_KEY not set._"

    horizon = context.get("horizon", "1 Month")
    ticker = context["ticker"]
    company = context["company"]
    sector = context["sector"]
    industry = context["industry"]

    top_driver_name = context.get("top_driver_name")
    top_driver_factor = context.get("top_driver_factor")
    top_driver_score = context.get("top_driver_score")
    top_driver_moving = context.get("top_driver_moving", False)

    system_prompt = _make_analyst_system(
        horizon, top_driver_name, top_driver_factor,
        top_driver_score, top_driver_moving
    )

    ctx_block = _build_context_block(context)
    driver_block = _build_driver_block(driver_profile)

    web_block = ""
    if web_results:
        web_lines = []
        for i, r in enumerate(web_results, 1):
            domain = r["url"].split("/")[2] if r.get("url") else "unknown"
            snippet = (r.get("content") or "")[:200].replace("\n", " ")
            web_lines.append(f"{i}. {r['title']} | {domain}\n   {snippet}")
        web_block = "\n".join(web_lines)
    else:
        web_block = "(No web research available)"

    # Format key interpolated values safely
    risk_score = context.get("composite_risk_score", "N/A")
    td_name = top_driver_name or "primary sector driver"
    td_factor = top_driver_factor or "N/A"

    user_prompt = f"""Analyze {ticker} ({company}, {sector}/{industry}) for {horizon} risk horizon.

=== TOP RISK DRIVERS (auto-ranked by relevance score) ===
{driver_block}

=== QUANTITATIVE SNAPSHOT ===
{ctx_block}

=== WEB RESEARCH (recent, driver-targeted) ===
{web_block}

=== REPORT — {horizon} HORIZON ===
Write exactly these sections:

## WHAT YOU NEED TO KNOW RIGHT NOW
1 short paragraph. Lead with the highest relevance_score driver.
If any driver has relevance_score > 2.0: start there — it is actively in play.
What should an experienced trader act on today that is NOT already in Yahoo Finance or WSJ?

## TOP 3 RISKS FOR {horizon}
Specific to THIS company for THIS horizon.
Risk #1 must be the highest-relevance driver if it is active.
Each risk: what → why NOW → specific trigger level to watch.

## SECTOR DRIVER DEEP DIVE
The #1 driver: {td_name} / factor: {td_factor}
- Current state of {td_factor}: [from web research or data]
- Company-specific exposure: [from fundamentals/EDGAR data]
- Estimated P&L impact if driver shocks: [calculate from actual margins/COGS]
- Hedging or mitigation in place: [from 10-K knowledge if applicable]
- Outlook for {horizon}: [from web research]

## WHAT STANDARD PROVIDERS ARE MISSING
One paragraph on what is not in the price and not covered by Yahoo Finance/WSJ.
Why does it matter specifically for {horizon}?

## NEAR-TERM CATALYSTS
From web research only. Events in next {horizon} that could move the stock.
If none found: state explicitly "No material catalysts identified for {horizon} horizon."

## RISK SCORE CONTEXT: {risk_score}/100
What specifically drives this score for {company}?
Is it appropriately priced vs sector peers?

## BOTTOM LINE
One sentence: risk/reward for {horizon}. Confidence: HIGH/MEDIUM/LOW and why."""

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return f"_Report generation failed: {e}_"


# ── PART 4: CHAT ──────────────────────────────────────────────────────────────

def chat_with_analyst(
    question: str,
    context: dict,
    chat_history: list[dict],
    driver_profile: dict | None = None,
    use_web: bool = True,
) -> str:
    """
    Answer a follow-up question in PM-briefing format.
    Always references primary sector driver. Fetches web for macro questions.
    """
    client = _get_openai()
    if not client:
        return "_Chat unavailable: OPENROUTER_API_KEY not set._"

    horizon = context.get("horizon", "1 Month")
    top_driver_name = context.get("top_driver_name", "primary risk factor")
    top_driver_factor = context.get("top_driver_factor", "unknown")
    top_driver_score = context.get("top_driver_score")
    top_driver_moving = context.get("top_driver_moving", False)
    score_str = f"{top_driver_score:.2f}" if top_driver_score is not None else "N/A"

    ctx_block = _build_context_block(context)

    _sector = context.get("sector", "")
    _industry = context.get("industry", "")
    _ticker_sym = context.get("ticker", "?")

    # Build sector-specific relevance guidance
    _sector_lower = _sector.lower()
    _industry_lower = _industry.lower()
    if any(s in _sector_lower or s in _industry_lower for s in ["technology", "semiconductor", "software"]):
        _sector_rules = (
            f"SECTOR RULES for {_ticker_sym} ({_sector}/{_industry}):\n"
            "- DO mention: AI demand, chip competition (AMD/Intel), export controls (China), "
            "data center capex, TSMC manufacturing risk, USD/Asia FX.\n"
            "- DO NOT mention crude oil, jet fuel, wheat, freight rates, or agricultural prices "
            "— they are not relevant to this company's P&L."
        )
    elif any(s in _sector_lower or s in _industry_lower for s in ["airline", "aviation"]):
        _sector_rules = (
            f"SECTOR RULES for {_ticker_sym} ({_sector}/{_industry}):\n"
            "- ALWAYS lead with jet fuel / crude oil if risk-related — it is the primary cost driver.\n"
            "- DO NOT mention AI chip demand, semiconductor supply, cloud subscriptions."
        )
    elif any(s in _sector_lower or s in _industry_lower for s in ["bank", "financial", "insurance"]):
        _sector_rules = (
            f"SECTOR RULES for {_ticker_sym} ({_sector}/{_industry}):\n"
            "- Lead with: federal funds rate, credit spreads, loan default risk, yield curve.\n"
            "- DO NOT mention jet fuel, crude oil, agricultural prices, or semiconductor demand."
        )
    elif any(s in _sector_lower or s in _industry_lower for s in ["energy", "oil", "gas"]):
        _sector_rules = (
            f"SECTOR RULES for {_ticker_sym} ({_sector}/{_industry}):\n"
            "- Lead with: crude oil / natural gas price, reserve life, E&P capex cycle.\n"
            "- DO NOT mention AI chip demand, semiconductor, or agricultural prices."
        )
    else:
        _sector_rules = (
            f"Only cite factors that directly affect {_ticker_sym}'s P&L. "
            "Do not use generic macro factors that belong on Yahoo Finance."
        )

    system_content = f"""You are a senior equity risk analyst. Answer like briefing a PM before market open.

Stock: {_ticker_sym} | {context.get('company', _ticker_sym)} | {_sector} / {_industry}
Horizon: {horizon}

PRIMARY RISK DRIVER RIGHT NOW:
{top_driver_name} — {top_driver_factor}
Relevance: {score_str}/3.0+ | Moving: {"YES" if top_driver_moving else "no"}

{_sector_rules}

=== CURRENT DATA ===
{ctx_block}

RESPONSE FORMAT — always use this structure:
🎯 [Direct answer in one sentence]

KEY FACTORS:
- [Most important — must relate to primary driver first if risk-related]
- [Second factor]
- [Third factor — optional]

⚠️ WATCH: [1-2 specific signals to monitor with trigger levels]

CONFIDENCE: HIGH/MEDIUM/LOW

RULES:
- Never start with "DATA SHOWS:" — you are an analyst, not a terminal.
- Always mention {top_driver_name} if the question is risk-related.
- Generic vol/beta answers belong on Yahoo Finance. Your value is sector-specific + company-specific.
- Strictly follow the SECTOR RULES above. Violating them makes the analysis useless.
- Total response under 200 words."""

    # Fetch web context for macro/sector questions
    web_ctx = ""
    if use_web:
        domain_keywords = [
            "sector", "industry", "fuel", "rates", "recession", "regulation",
            "competition", "macro", "inflation", "fed", "interest", "credit",
            "oil", "gas", "currency", "usd", "yield", "spread", "tariff",
        ]
        if any(kw in question.lower() for kw in domain_keywords):
            try:
                web_hits = fetch_web_research(
                    context.get("ticker", _ticker_sym),
                    context.get("company", _ticker_sym),
                    context.get("sector", ""),
                    driver_profile=driver_profile, horizon=horizon,
                )
                if web_hits:
                    snippets = [
                        f"- {r['title']}: {(r.get('content') or '')[:150]}"
                        for r in web_hits[:4]
                    ]
                    web_ctx = "\n\n=== RECENT WEB CONTEXT ===\n" + "\n".join(snippets)
            except Exception:
                pass

    messages = [{"role": "system", "content": system_content + web_ctx}]
    for msg in chat_history[-10:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=messages,
            max_tokens=500,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return f"_Chat error: {e}_"


# ── PART 5: SCENARIO EVALUATION ──────────────────────────────────────────────

_BASE_SCENARIOS = {
    "Recession": {
        "gdp_change": -2, "unemployment_delta": +3,
        "spread_delta": +200, "equity_drawdown": -30,
    },
    "Rate Shock +200bps": {
        "rate_delta_bps": +200, "duration_impact": True,
    },
    "Credit Crunch": {
        "hy_spread_target": 700,
    },
}

_SECTOR_SCENARIOS = {
    "Industrials":          {"Fuel Shock +40%": {"fuel_price_delta_pct": +40}},
    "Airlines":             {"Fuel Shock +40%": {"fuel_price_delta_pct": +40}},
    "Energy":               {"Oil Crash -40%": {"oil_delta_pct": -40}},
    "Technology":           {"Multiple Compression -30%": {"pe_compression_pct": -30}},
    "Financials":           {"Credit Losses": {"loss_rate_delta_bps": +200}},
    "Real Estate":          {"Cap Rate Expansion": {"rate_delta_bps": +150}},
    "Healthcare":           {"Reimbursement Cut": {"revenue_impact_pct": -15}},
    "Consumer Discretionary": {"Consumer Downturn": {"revenue_impact_pct": -20}},
}


def get_scenarios_for_sector(sector: str) -> dict:
    """Return base scenarios + sector-specific scenario for the given sector."""
    scenarios = dict(_BASE_SCENARIOS)
    for key, scenario_map in _SECTOR_SCENARIOS.items():
        if key.lower() in sector.lower():
            scenarios.update(scenario_map)
            break
    if len(scenarios) == len(_BASE_SCENARIOS):
        scenarios["Sector Disruption"] = {"revenue_impact_pct": -20}
    return scenarios


def evaluate_scenario(scenario_name: str, params: dict, context: dict) -> dict:
    """
    Use LLM to evaluate a specific scenario given the current context.
    Returns structured dict with estimated impacts.
    """
    client = _get_openai()
    if not client:
        return {"error": "OPENROUTER_API_KEY not set"}

    ctx_brief = {
        k: context[k] for k in [
            "ticker", "company", "sector", "composite_risk_score",
            "beta_spy", "max_drawdown", "cvar_99",
            "debt_equity", "interest_coverage", "altman_zone",
            "fundamental_risk_score", "divergence",
        ] if k in context
    }

    prompt = f"""Analyze the impact of scenario "{scenario_name}" on {context['ticker']} ({context['company']}, {context['sector']}).

Scenario parameters: {params}

Current company profile:
{_fmt_val(ctx_brief)}

Respond ONLY with a JSON object with these exact keys:
{{
  "scenario": "{scenario_name}",
  "probability": "LOW|MEDIUM|HIGH",
  "estimated_price_impact_pct": <number>,
  "estimated_eps_impact_pct": <number>,
  "key_transmission": "<1 sentence: HOW this scenario hits this company>",
  "historical_precedent": "<1 sentence: similar past episode>",
  "ticker_specific": "<2 sentences: why THIS company is more/less vulnerable given its actual balance sheet>",
  "hedging_implication": "<1 sentence: what risk managers would do>"
}}"""

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a quantitative risk analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        import json
        text = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        return json.loads(text)
    except Exception as e:
        logger.error(f"Scenario eval failed: {e}")
        return {"scenario": scenario_name, "error": str(e)}


# ── PART 6: QUICK SYNOPSIS (for Risk Scorecard) ───────────────────────────────

def get_quick_synopsis(
    ticker: str,
    risk_score: float,
    fundamental_score: float | None,
    divergence: float | None,
    vol_regime: str,
    altman_zone: str | None,
) -> str:
    """
    Two-sentence max synopsis for the Risk Scorecard tab.
    Fast call — small prompt, max_tokens=150.
    """
    client = _get_openai()
    if not client:
        return ""

    fund_part = f"fundamental score {fundamental_score:.0f}/100, divergence {divergence:.0f}pts, " if fundamental_score else ""
    altman_part = f"Altman Z: {altman_zone}" if altman_zone else ""

    prompt = (
        f"In exactly 2 sentences, summarize the key risk insight for {ticker}: "
        f"risk score {risk_score:.0f}/100, {fund_part}"
        f"vol regime {vol_regime}, {altman_part}. "
        f"Be specific with numbers. No generic statements."
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Synopsis failed: {e}")
        return ""


