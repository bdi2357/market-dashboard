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

    return {
        "ticker": ticker,
        "company": info.get("name", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("market_cap"),
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


def fetch_web_research(ticker: str, company: str, sector: str) -> list[dict]:
    """
    Run 5 targeted Tavily searches and return top-10 results by relevance.
    Returns empty list if Tavily key missing or any error occurs.
    """
    client = _get_tavily()
    if not client:
        logger.warning("Tavily key missing — skipping web research")
        return []

    queries = [
        f"{company} {ticker} earnings guidance analyst 2026",
        f"{sector} industry risk macro outlook 2026",
        f"{ticker} SEC insider filing Form 4 recent",
        f"{company} credit rating debt refinancing",
        f"{sector} institutional flows hedge fund positioning",
    ]

    seen_urls: set = set()
    results: list[dict] = []

    for query in queries:
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
            logger.warning(f"Tavily query failed ({query[:40]}…): {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


# ── PART 3: GENERATE RESEARCH REPORT ─────────────────────────────────────────

_ANALYST_SYSTEM = """You are a senior equity risk analyst at a quantitative hedge fund with 20 years of experience. You specialize in:
- Multi-factor risk decomposition
- Behavioral finance and market regime analysis
- Fundamental credit and equity analysis
- Insider and institutional flow interpretation

Your analysis style:
- ALWAYS cite actual numbers from the data provided
- NEVER make generic statements
  BAD: "volatility is elevated"
  GOOD: "21d realized vol of 47% sits at 89th percentile of its own 2-year history"
- Every claim must be specific to THIS stock's actual values
- Flag data quality issues honestly
- Your audience: experienced traders and researchers
- Distinguish clearly: DATA SHOWS vs YOUR INTERPRETATION"""


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


def generate_research_report(context: dict, web_results: list[dict]) -> str:
    """
    Generate full institutional research report via DeepSeek on OpenRouter.
    Returns plain-text Markdown report string.
    """
    client = _get_openai()
    if not client:
        return "_AI report unavailable: OPENROUTER_API_KEY not set._"

    ctx_block = _build_context_block(context)

    web_block = ""
    if web_results:
        web_lines = []
        for i, r in enumerate(web_results, 1):
            domain = r["url"].split("/")[2] if r["url"] else "unknown"
            snippet = r["content"][:200].replace("\n", " ")
            web_lines.append(f"{i}. {r['title']} | {domain}\n   {snippet}")
        web_block = "\n".join(web_lines)
    else:
        web_block = "(No web research available — Tavily key not configured)"

    ticker = context["ticker"]
    company = context["company"]
    sector = context["sector"]
    industry = context["industry"]
    risk_score = context.get("composite_risk_score", "N/A")
    fund_score = context.get("fundamental_risk_score", "N/A")
    divergence = context.get("divergence", "N/A")
    vol_regime_val = context.get("vol_regime", "unknown")
    vol_pct = context.get("vol_percentile", "N/A")
    kurtosis = context.get("excess_kurtosis", "N/A")
    cvar_99 = context.get("cvar_99", "N/A")
    cvar_str = f"{cvar_99:.2%}" if isinstance(cvar_99, float) else str(cvar_99)
    macro_regime = context.get("macro_regime", "UNKNOWN")
    insider_signal = context.get("insider_signal", "NEUTRAL")
    funds_adding = context.get("funds_adding", 0)
    funds_reducing = context.get("funds_reducing", 0)
    divergence_str = f"{divergence:.1f}" if isinstance(divergence, float) else str(divergence)
    fund_score_str = f"{fund_score:.0f}" if isinstance(fund_score, float) else str(fund_score)

    user_prompt = f"""Generate a professional research report for {ticker} ({company}, {sector} / {industry}).

=== QUANTITATIVE SNAPSHOT ===
{ctx_block}

=== RECENT WEB RESEARCH ===
{web_block}

=== REPORT STRUCTURE ===
Write exactly these sections with these headers:

## EXECUTIVE SUMMARY
3-4 sentences. What is the single dominant risk theme right now? What should an experienced trader focus on?

## RISK PROFILE: {risk_score}/100
Explain what drives the score. Is risk appropriately priced vs sector peers? What's the #1 risk driver?

## VOLATILITY REGIME
Current: {vol_regime_val} at {vol_pct}th percentile.
Kurtosis {kurtosis} means: [interpret fat tails].
What does the return distribution shape tell us about the TYPE of risk (jump risk vs diffuse vol)?

## TAIL RISK ASSESSMENT
CVaR99 of {cvar_str} interpreted in plain terms.
Is tail risk systematic (market beta) or idiosyncratic?
What specific scenarios could trigger tail events for THIS company in THIS sector?

## SMART MONEY POSITIONING
Insider signal: {insider_signal}
Funds adding: {funds_adding} | Reducing: {funds_reducing}
Synthesize the complete smart money narrative.
Flag contradictions between insider and institutional signals.
Historical base rate: when insiders show this pattern, what typically happens over 30/60/90 days?

## FUNDAMENTAL vs MARKET RISK DIVERGENCE
Divergence: {divergence_str} points
({risk_score} market vs {fund_score_str} fundamental)
What is the market pricing that fundamentals don't confirm?
Resolution: which signal has historically been correct in similar setups?

## MACRO REGIME SENSITIVITY
Regime: {macro_regime}
How does {company}'s specific business model interact with: current rate level, credit spreads, sector dynamics, and the VIX regime? Quantify where possible.

## KEY RISKS — MONITOR THESE
List exactly 5 risks with specific trigger levels:
Format: "RISK: [name] | Trigger: [specific metric at specific level] | Historical impact: [what happened]"

## RECENT MATERIAL DEVELOPMENTS
From web research — material news ONLY (earnings, guidance, credit, M&A, regulatory, sector disruption).
For each: [Source] [Headline] [Why it matters for risk]
Exclude generic price commentary.

=== CONFIDENCE RATINGS ===
Rate each section HIGH/MEDIUM/LOW based on data quality."""

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": _ANALYST_SYSTEM},
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
    use_web: bool = True,
) -> str:
    """
    Answer a follow-up question using full context + chat history.
    If question touches sector/macro keywords, fetch fresh web results.
    """
    client = _get_openai()
    if not client:
        return "_Chat unavailable: OPENROUTER_API_KEY not set._"

    ctx_block = _build_context_block(context)
    system_content = f"""{_ANALYST_SYSTEM}

=== STOCK UNDER ANALYSIS ===
Ticker: {context['ticker']} | Company: {context['company']}
Sector: {context['sector']} | Industry: {context['industry']}

=== CURRENT QUANTITATIVE DATA ===
{ctx_block}

Always ground answers in the ACTUAL numbers above. Be concise but precise."""

    # Optionally fetch web context for domain questions
    web_ctx = ""
    if use_web:
        domain_keywords = [
            "sector", "industry", "fuel", "rates", "recession", "regulation",
            "competition", "macro", "inflation", "fed", "interest", "credit",
            "oil", "gas", "currency", "usd", "yield", "spread",
        ]
        if any(kw in question.lower() for kw in domain_keywords):
            try:
                web_hits = fetch_web_research(
                    context["ticker"], context["company"], context["sector"]
                )
                if web_hits:
                    snippets = [
                        f"- {r['title']}: {r['content'][:150]}"
                        for r in web_hits[:4]
                    ]
                    web_ctx = "\n\n=== RECENT WEB CONTEXT ===\n" + "\n".join(snippets)
            except Exception:
                pass

    messages = [{"role": "system", "content": system_content + web_ctx}]
    # Add chat history (last 10 turns to stay within token budget)
    for msg in chat_history[-10:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=messages,
            max_tokens=1500,
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
