"""
Market Risk Assessment Dashboard
=================================
Multi-tab professional risk engine for research & backtesting.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Market Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import yfinance as yf

# Inject FRED API key from Streamlit secrets into the environment so
# data/fetcher.py (which reads os.environ) can find it without requiring
# the user to set the env var manually before launching Streamlit.
if "FRED_API_KEY" not in os.environ:
    try:
        os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

from data.fetcher import (
    fetch_ohlcv, fetch_ticker_info, get_sector_etf,
    fetch_macro, fetch_peers, SECTOR_ETF_MAP,
)
from risk.metrics import (
    rolling_realized_vol, vol_regime, max_drawdown,
    calmar_ratio, sortino_ratio, ulcer_index,
    historical_var_cvar, parametric_var, cornish_fisher_var,
    rolling_skew_kurt, rolling_beta, rolling_correlation,
    residual_vol, information_ratio,
    classify_macro_regime, vix_regime, conditional_returns_by_regime,
    amihud_illiquidity, hl_spread_proxy, volume_trend,
    composite_risk_score,
)
from backtest.engine import run_backtest, stress_test, bootstrap_sharpe_ci
from risk.fundamental import (
    valuation_risk, balance_sheet_risk, earnings_quality,
    cashflow_risk, growth_value_classification,
    fetch_material_news, composite_fundamental_score,
)
from data.edgar import (
    fetch_insider_transactions, compute_insider_signals,
    fetch_institutional_changes, compute_institutional_signals,
    fetch_activist_positions, fetch_xbrl_fundamentals,
)
from risk.factor_decomposition import factor_decomposition
from risk.shock_analysis import analyze_price_shock, get_sector_factors
from ai.analyst import (
    build_ticker_context, fetch_web_research,
    generate_research_report, chat_with_analyst,
    evaluate_scenario, get_scenarios_for_sector, get_quick_synopsis,
)
from data.driver_discovery import discover_risk_drivers

DARK = "plotly_dark"


def init_session_state():
    defaults = {
        "horizon": "1 Month",
        "data_loaded": False,
        "current_ticker": None,
        "fast_ma": 20,
        "slow_ma": 50,
        "sizing_method": "vol_target",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

REGIME_COLORS = {
    "RISK_ON": "#2ecc71",
    "RISK_OFF": "#e74c3c",
    "STAGFLATION_PROXY": "#f39c12",
    "MIXED": "#95a5a6",
    "UNKNOWN": "#7f8c8d",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=3*365))
    with col_b:
        end_date = st.date_input("End", value=date.today())
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)
    st.divider()
    horizon = st.radio(
        "Risk Horizon",
        ["1 Week", "1 Month", "3 Months", "1 Year"],
        index=1,
        horizontal=False,
        help="Drives AI report focus: near-term catalysts vs long-term thesis",
    )
    if st.session_state.get("horizon") != horizon:
        # Clear AI report when horizon changes
        for _k in list(st.session_state.keys()):
            if _k.startswith("ai_report_") or _k.startswith("ai_ctx_") or _k.startswith("synopsis_"):
                del st.session_state[_k]
        st.session_state["horizon"] = horizon
    st.divider()
    load = st.button("🚀 Load Data", type="primary", use_container_width=True)
    if st.session_state.get("data_loaded") and st.session_state.get("current_ticker") == ticker:
        st.caption(f"⚡ {ticker} loaded · click to refresh")

st.title("📊 Market Risk Dashboard")

# ── Data loading ──────────────────────────────────────────────────────────────
if load:
    _progress = st.progress(0, text="Fetching price data...")
    df = fetch_ohlcv(ticker, start_date, end_date)
    _progress.progress(20, text="Fetching benchmark data...")
    bench_df = fetch_ohlcv(benchmark, start_date, end_date)
    _progress.progress(35, text="Fetching ticker metadata...")
    info = fetch_ticker_info(ticker)
    sector_etf = get_sector_etf(ticker)
    _progress.progress(50, text=f"Fetching sector ETF ({sector_etf})...")
    sector_df = fetch_ohlcv(sector_etf, start_date, end_date)
    _progress.progress(70, text="Fetching FRED macro data (DGS10, VIX, spreads)...")
    macro_df = fetch_macro(start_date, end_date)
    _progress.progress(100, text="Done.")
    _progress.empty()

    st.session_state.df = df
    st.session_state.bench_df = bench_df
    st.session_state.sector_df = sector_df
    st.session_state.macro_df = macro_df
    st.session_state.info = info
    st.session_state.sector_etf = sector_etf
    st.session_state.data_loaded = True
    st.session_state.ticker = ticker
    st.session_state.benchmark = benchmark
    st.session_state["horizon"] = horizon
    st.session_state["current_ticker"] = ticker
    # Clear stale cached computations when new data is loaded
    for _k in list(st.session_state.keys()):
        if any(_k.startswith(p) for p in ("bt_", "fund_", "smart_money_",
                                           "ai_report_", "ai_ctx_", "synopsis_",
                                           "factor_", "shock_", "driver_")):
            del st.session_state[_k]

    # ── Driver discovery (3-layer, 24h JSON cache) ────────────────────────────
    with st.spinner("Discovering risk drivers (EDGAR + AI, cached 24h)..."):
        try:
            from data.edgar import resolve_cik as _rcik_load
            _cik_load = _rcik_load(ticker)
        except Exception:
            _cik_load = None
        try:
            from ai.analyst import _get_openai as _goa, _get_tavily as _gtv
            _dp = discover_risk_drivers(
                ticker,
                info.get("name", ticker),
                info.get("sector", "Unknown"),
                info.get("industry", "Unknown"),
                _cik_load,
                _gtv(),
                _goa(),
                horizon,
            )
            st.session_state[f"driver_{ticker}"] = _dp
        except Exception:
            st.session_state[f"driver_{ticker}"] = None

if not st.session_state.data_loaded:
    st.info("Configure settings in the sidebar and click **Load Data** to begin.")
    st.stop()

df = st.session_state.df
bench_df = st.session_state.bench_df
sector_df = st.session_state.sector_df
macro_df = st.session_state.macro_df
info = st.session_state.info
sector_etf = st.session_state.sector_etf
_ticker = st.session_state.ticker
_bench = st.session_state.benchmark

if df.empty:
    st.error(f"No data for '{_ticker}'. Check the symbol.")
    st.stop()

prices = df["Close"]
volume = df["Volume"]
bench_prices = bench_df["Close"] if not bench_df.empty else prices
sector_prices = sector_df["Close"] if not sector_df.empty else prices

simple_r = prices.pct_change().dropna()

# Retrieve horizon from session state (set in sidebar)
_horizon = st.session_state.get("horizon", "1 Month")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Risk Summary",
    "📊 Market Risk",
    "🌍 Macro & Regime",
    "🏛️ Smart Money & Factors",
    "📋 Fundamentals",
    "🤖 AI Analyst",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RISK SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Risk Scorecard — {_ticker}")

    # ── RISK ALERTS ───────────────────────────────────────────────────────────
    _r_alerts = []
    _kurt = simple_r.kurtosis()
    _skew = simple_r.skew()
    _ann_vol_alert = simple_r.std() * np.sqrt(252)
    _ann_ret_alert = (1 + simple_r.mean()) ** 252 - 1
    _mdd_alert = max_drawdown(prices)["max_drawdown"]
    _calmar_alert = calmar_ratio(prices)
    _var_data_alert = historical_var_cvar(simple_r)
    _cvar99_alert = _var_data_alert["cvar_99"]

    if _kurt > 5:
        _adj = (1 + (_kurt / 3) * 0.15)
        _r_alerts.append(("🔴", "EXTREME FAT TAILS",
            f"Excess kurtosis {_kurt:.2f} — normal VaR underestimates true tail risk by ~{(_adj-1)*100:.0f}%. "
            f"Cornish-Fisher adjustment is essential for this asset.",
            "#e74c3c"))
    elif _kurt > 3:
        _r_alerts.append(("🟠", "FAT TAILS",
            f"Excess kurtosis {_kurt:.2f} > 3 — return distribution has heavier tails than normal.",
            "#e67e22"))

    if _cvar99_alert > 0.10:
        _r_alerts.append(("🔴", "SEVERE TAIL RISK",
            f"CVaR 99% = {_cvar99_alert:.2%} — worst 1% of trading days average a {_cvar99_alert:.2%} loss. "
            f"Threshold: >10%.",
            "#e74c3c"))

    if _mdd_alert < -0.40:
        _r_alerts.append(("🟠", "DEEP DRAWDOWN HISTORY",
            f"Max drawdown = {_mdd_alert:.1%} — asset has experienced severe capital destruction. "
            f"Threshold: < -40%.",
            "#e67e22"))

    if not np.isnan(_calmar_alert) and _calmar_alert < 0.10:
        _r_alerts.append(("🟠", "POOR RISK-ADJUSTED RETURN",
            f"Calmar ratio = {_calmar_alert:.3f} — annualized return barely compensates for drawdown risk. "
            f"Threshold: < 0.10.",
            "#e67e22"))

    if _ann_vol_alert > 0.40:
        _r_alerts.append(("🟡", "HIGH VOLATILITY ASSET",
            f"Annualized vol = {_ann_vol_alert:.1%} — significantly above typical equity range (15-25%). "
            f"Threshold: > 40%.",
            "#f39c12"))

    if _r_alerts:
        st.markdown("**⚠️ Active Risk Alerts**")
        for icon, title, msg, color in _r_alerts:
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:8px 14px;border-radius:4px;margin-bottom:6px'>"
                f"<b style='color:{color}'>{icon} {title}</b><br>"
                f"<span style='font-size:13px;color:#ddd'>{msg}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

    col_info, col_score = st.columns([1, 1])

    with col_info:
        st.markdown(f"**Name:** {info.get('name', _ticker)}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}  |  **Industry:** {info.get('industry', 'N/A')}")
        mcap = info.get("market_cap")
        st.markdown(f"**Market Cap:** {'${:,.0f}M'.format(mcap/1e6) if mcap else 'N/A'}")
        st.markdown(f"**Sector ETF:** {sector_etf}")

    score_data = composite_risk_score(prices, volume, bench_prices)
    total_score = score_data["total"]
    components = score_data["components"]

    with col_score:
        color = "#2ecc71" if total_score < 33 else "#f39c12" if total_score < 66 else "#e74c3c"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_score,
            title={"text": "Composite Risk Score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "#1a3a1a"},
                    {"range": [33, 66], "color": "#3a2e0a"},
                    {"range": [66, 100], "color": "#3a0a0a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": total_score},
            },
        ))
        gauge.update_layout(height=250, template=DARK, margin=dict(t=30, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    # ── DUAL GAUGE: Market Risk vs Fundamental Risk ───────────────────────────
    _fund_score_val = st.session_state.fund_scores["total"] if st.session_state.get("fund_scores") else None
    _gauge_cols = st.columns(2 if _fund_score_val is not None else 1)

    def _make_gauge(title, value, height=260):
        _gc = "#2ecc71" if value < 30 else "#f39c12" if value < 60 else "#e67e22" if value < 80 else "#e74c3c"
        _fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _gc},
                "steps": [
                    {"range": [0, 30],  "color": "#0d2b0d"},
                    {"range": [30, 60], "color": "#2b1f00"},
                    {"range": [60, 80], "color": "#2b0f00"},
                    {"range": [80, 100],"color": "#1a0000"},
                ],
            },
        ))
        _fig.update_layout(height=height, template=DARK, margin=dict(t=40, b=10, l=20, r=20))
        return _fig

    with _gauge_cols[0]:
        st.plotly_chart(_make_gauge("Market Risk Score", total_score), use_container_width=True)
    if _fund_score_val is not None:
        with _gauge_cols[1]:
            st.plotly_chart(_make_gauge("Fundamental Risk Score", _fund_score_val), use_container_width=True)
        _div_val = abs(total_score - _fund_score_val)
        _div_c = "#2ecc71" if _div_val < 20 else "#f39c12" if _div_val < 40 else "#e74c3c"
        st.markdown(
            f"<div style='text-align:center;padding:6px;background:{_div_c}22;"
            f"border-radius:4px;margin-bottom:12px'>"
            f"<b style='color:{_div_c}'>Divergence: {_div_val:.0f} pts</b> — "
            f"{'Aligned' if _div_val < 20 else 'Mild divergence' if _div_val < 40 else 'Strong divergence — see Fundamental Risk tab'}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Component breakdown bar chart
    comp_names = list(components.keys())
    comp_values = [components[k] for k in comp_names]
    fig_bar = go.Figure(go.Bar(
        y=comp_names, x=comp_values, orientation="h",
        marker_color=[("#2ecc71" if v < 8 else "#f39c12" if v < 16 else "#e74c3c") for v in comp_values],
        text=[f"{v:.1f}" for v in comp_values], textposition="outside",
    ))
    fig_bar.update_layout(
        title="Risk Component Breakdown (weighted contribution to score)",
        xaxis_title="Weighted Score",
        template=DARK, height=300, margin=dict(l=200),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Key summary metrics
    st.subheader("Summary Statistics")
    # Divergence indicator (lazy — only shown if fundamental data already loaded)
    if st.session_state.get("fund_scores"):
        fs = st.session_state.fund_scores
        mkt_score = total_score
        fund_score = fs["total"]
        div = abs(mkt_score - fund_score)
        if div < 20:
            div_color, div_label = "#2ecc71", "🟢 ALIGNED — Market and fundamental risk agree"
        elif div < 40:
            div_color, div_label = "#f39c12", "🟡 MILD DIVERGENCE — Signals mixed, monitor closely"
        else:
            div_color, div_label = "#e74c3c", "🔴 STRONG DIVERGENCE"
            if mkt_score > fund_score:
                div_label += " — Market may be OVERPRICING RISK (potential opportunity if fundamentals hold)"
            else:
                div_label += " — Market may be UNDERPRICING RISK (fundamentals deteriorating beneath calm surface)"
        st.markdown(
            f"<div style='background:{div_color}22;border-left:4px solid {div_color};"
            f"padding:10px 16px;border-radius:4px;margin-bottom:12px'>"
            f"<b style='color:{div_color}'>{div_label}</b><br>"
            f"<small>Market Risk Score: {mkt_score:.0f} | Fundamental Risk Score: {fund_score:.0f} | Divergence: {div:.0f} pts</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # ── AI SYNOPSIS ───────────────────────────────────────────────────────────
    _syn_key = f"synopsis_{_ticker}"
    if _syn_key not in st.session_state:
        _fs_val = st.session_state.fund_scores.get("total") if st.session_state.get("fund_scores") else None
        _div_syn = abs(total_score - _fs_val) if _fs_val is not None else None
        _vr_syn = vol_regime(rolling_realized_vol(prices, [21])["vol_21d"].dropna())
        _vr_syn_str = str(_vr_syn.iloc[-1]) if not _vr_syn.empty else "unknown"
        _az_syn = st.session_state.fund_scores.get("altman_zone") if st.session_state.get("fund_scores") else None
        try:
            _synopsis = get_quick_synopsis(
                _ticker, total_score, _fs_val, _div_syn, _vr_syn_str, _az_syn
            )
        except Exception:
            _synopsis = ""
        st.session_state[_syn_key] = _synopsis
    if st.session_state.get(_syn_key):
        st.info(f"🤖 {st.session_state[_syn_key]}")

    # ── DRIVER PROFILE CARD ───────────────────────────────────────────────────
    _dp_card = st.session_state.get(f"driver_{_ticker}")
    if _dp_card and _dp_card.get("factors"):
        _p_icons = {"PRIMARY": "🔴", "SECONDARY": "🟡", "MONITORING": "⚪"}
        _vel_icon = lambda v: "🔥" if v > 1.0 else "📈" if v > 0.5 else "➖"
        st.markdown("**📡 Top Risk Drivers**")
        _dc_rows = []
        for _df in _dp_card["factors"][:5]:
            _vel = _df.get("news_velocity", 0)
            _r1m = _df.get("factor_1m_return")
            _dc_rows.append({
                "": f"{_p_icons.get(_df.get('priority',''), '⚪')}",
                "Driver": _df.get("name", ""),
                "Factor": _df.get("external_factor") or "—",
                "1M Move": f"{_r1m:+.1f}%" if _r1m is not None else "—",
                "News": _vel_icon(_vel),
            })
        st.dataframe(pd.DataFrame(_dc_rows), use_container_width=True, hide_index=True)
        _dq = _dp_card.get("data_quality", "")
        st.caption(f"Source: {_dq} · {str(_dp_card.get('timestamp',''))[:10]}")

    mdd_data = max_drawdown(prices)
    var_data = historical_var_cvar(simple_r)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    ann_vol = simple_r.std() * np.sqrt(252)
    ann_ret = (1 + simple_r.mean()) ** 252 - 1
    m1.metric("Ann. Return", f"{ann_ret:.1%}")
    m2.metric("Ann. Volatility", f"{ann_vol:.1%}")
    m3.metric("Sharpe (rf=0)", f"{ann_ret/ann_vol:.2f}" if ann_vol > 0 else "N/A")
    m4.metric("Max Drawdown", f"{mdd_data['max_drawdown']:.1%}")
    m5.metric("Calmar", f"{calmar_ratio(prices):.2f}")
    m6.metric("Sortino", f"{sortino_ratio(prices):.2f}")

    m7, m8, m9, m10, m11, m12 = st.columns(6)
    m7.metric("VaR 95% (1d)", f"{var_data['var_95']:.2%}")
    m8.metric("CVaR 95% (1d)", f"{var_data['cvar_95']:.2%}")
    m9.metric("VaR 99% (1d)", f"{var_data['var_99']:.2%}")
    m10.metric("CVaR 99% (1d)", f"{var_data['cvar_99']:.2%}")
    m11.metric("Skewness", f"{simple_r.skew():.2f}")
    m12.metric("Excess Kurtosis", f"{simple_r.kurtosis():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET RISK (Volatility · Tail Risk · Relative Risk · Backtester)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    _mkt_s1, _mkt_s2, _mkt_s3, _mkt_s4 = st.tabs([
        "📈 Volatility Surface", "⚠️ Tail Risk", "🔗 Relative Risk", "🧪 Backtester",
    ])

with _mkt_s1:
    st.subheader("Volatility Cone & Regime")

    vol_df = rolling_realized_vol(prices, windows=[10, 21, 63, 252])
    skk_df = rolling_skew_kurt(prices)

    # Vol cone chart
    fig_vol = go.Figure()
    colors = {"vol_10d": "#e74c3c", "vol_21d": "#f39c12", "vol_63d": "#2ecc71", "vol_252d": "#3498db"}
    for col, color in colors.items():
        label = col.replace("vol_", "").replace("d", "d Realized Vol")
        fig_vol.add_trace(go.Scatter(x=vol_df.index, y=vol_df[col], name=label, line=dict(color=color)))

    # Percentile bands for 21d vol + spike annotations
    v21 = vol_df["vol_21d"].dropna()
    p95 = pd.Series(dtype=float)
    if len(v21) > 40:
        p25 = v21.rolling(252, min_periods=60).quantile(0.25)
        p75 = v21.rolling(252, min_periods=60).quantile(0.75)
        p95 = v21.rolling(252, min_periods=60).quantile(0.95)
        fig_vol.add_trace(go.Scatter(x=v21.index, y=p95, name="95th pct (21d)", line=dict(color="white", dash="dot", width=1)))
        fig_vol.add_trace(go.Scatter(x=v21.index, y=p75, name="75th pct (21d)", line=dict(color="gray", dash="dot", width=1)))
        fig_vol.add_trace(go.Scatter(x=v21.index, y=p25, name="25th pct (21d)", line=dict(color="gray", dash="dot", width=1), fill="tonexty", fillcolor="rgba(100,100,100,0.1)"))

        # Detect spikes: 10d vol > 95th pct of its own history, group into events
        v10 = vol_df["vol_10d"].dropna()
        p95_10 = v10.rolling(252, min_periods=60).quantile(0.95)
        spike_mask = v10 > p95_10
        spike_events = []
        in_spike = False
        spike_start = None
        spike_peak_val = 0
        spike_peak_date = None
        for dt in v10.index:
            if spike_mask.get(dt, False):
                if not in_spike:
                    in_spike = True
                    spike_start = dt
                    spike_peak_val = v10[dt]
                    spike_peak_date = dt
                elif v10[dt] > spike_peak_val:
                    spike_peak_val = v10[dt]
                    spike_peak_date = dt
            else:
                if in_spike:
                    spike_events.append((spike_peak_date, spike_peak_val, p95_10.get(spike_peak_date, spike_peak_val)))
                    in_spike = False
        if in_spike:
            spike_events.append((spike_peak_date, spike_peak_val, p95_10.get(spike_peak_date, spike_peak_val)))

        # Annotate top-5 spikes by magnitude
        spike_events.sort(key=lambda x: x[1], reverse=True)
        for sp_date, sp_val, sp_p95 in spike_events[:5]:
            pct_above = (sp_val / sp_p95 - 1) * 100 if sp_p95 > 0 else 0
            fig_vol.add_annotation(
                x=sp_date, y=sp_val,
                text=f"Spike +{pct_above:.0f}% above 95th",
                showarrow=True, arrowhead=2, arrowcolor="#e74c3c",
                font=dict(size=10, color="#e74c3c"),
                bgcolor="rgba(30,30,46,0.85)",
                yshift=10,
            )

    fig_vol.update_layout(title="Realized Volatility Cone with Spike Annotations", yaxis_title="Annualized Vol", template=DARK, height=420)
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── VOL REGIME COLOR BAR ──────────────────────────────────────────────────
    if len(v21) > 40:
        _regime_s = vol_regime(v21)
        _regime_color_map = {"low": "#2ecc71", "normal": "#3498db", "high": "#f39c12", "extreme": "#e74c3c"}
        _rdf = pd.DataFrame({"regime": _regime_s, "y": 1}, index=v21.index).dropna()
        fig_rbar = go.Figure()
        for _reg, _rc in _regime_color_map.items():
            _mask = _rdf["regime"] == _reg
            if _mask.any():
                fig_rbar.add_trace(go.Bar(
                    x=_rdf.index[_mask], y=[1]*_mask.sum(),
                    name=_reg, marker_color=_rc, showlegend=True,
                ))
        fig_rbar.update_layout(
            barmode="stack", title="Vol Regime Timeline (21d)",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=80, template=DARK, margin=dict(t=30, b=20),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_rbar, use_container_width=True)

    # ── TOP VOL SPIKES TABLE ──────────────────────────────────────────────────
    if len(v21) > 40 and spike_events:
        st.subheader("Top Vol Spike Periods")
        _spike_rows = []
        for sp_date, sp_val, sp_p95 in spike_events[:5]:
            _pct_above = (sp_val / sp_p95 - 1) * 100 if sp_p95 > 0 else 0
            _spike_rows.append({
                "Date": str(sp_date)[:10],
                "10d Realized Vol": f"{sp_val:.1%}",
                "95th Pct Threshold": f"{sp_p95:.1%}",
                "% Above Threshold": f"+{_pct_above:.0f}%",
            })
        st.dataframe(pd.DataFrame(_spike_rows), use_container_width=True, hide_index=True)

    # Vol regime coloring
    regime_labels = vol_regime(vol_df["vol_21d"].dropna())
    regime_colors_map = {"low": "#2ecc71", "normal": "#3498db", "high": "#f39c12", "extreme": "#e74c3c", "unknown": "#7f8c8d"}

    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(
        x=vol_df.index, y=vol_df["vol_21d"],
        mode="lines", name="21d Vol",
        line=dict(color="white", width=1),
    ))
    # Add colored background by regime
    for regime, color in regime_colors_map.items():
        mask = (regime_labels == regime)
        if mask.any():
            idx = vol_df["vol_21d"].index[vol_df["vol_21d"].index.isin(mask[mask].index)]
            fig_regime.add_trace(go.Scatter(
                x=idx, y=vol_df["vol_21d"].reindex(idx),
                mode="markers", marker=dict(color=color, size=3),
                name=f"Regime: {regime}", showlegend=True,
            ))
    fig_regime.update_layout(title="Vol Regime (21d) — colored by quartile vs 2Y history", template=DARK, height=300)
    st.plotly_chart(fig_regime, use_container_width=True)

    # Skew/Kurtosis
    fig_sk = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling 63d Skewness", "Rolling 63d Excess Kurtosis"])
    fig_sk.add_trace(go.Scatter(x=skk_df.index, y=skk_df["skewness"], name="Skewness", line=dict(color="#e74c3c")), row=1, col=1)
    fig_sk.add_hline(y=1, line_dash="dot", line_color="white", row=1, col=1)
    fig_sk.add_hline(y=-1, line_dash="dot", line_color="white", row=1, col=1)
    fig_sk.add_trace(go.Scatter(x=skk_df.index, y=skk_df["excess_kurtosis"], name="Excess Kurtosis", line=dict(color="#f39c12")), row=2, col=1)
    fig_sk.add_hline(y=3, line_dash="dot", line_color="white", row=2, col=1)
    fig_sk.update_layout(template=DARK, height=400, title="Higher moments — flag when |skew|>1 or kurt>3")
    st.plotly_chart(fig_sk, use_container_width=True)

    # ── INTERPRETIVE COMMENTARY ───────────────────────────────────────────────
    _v21_last = vol_df["vol_21d"].dropna().iloc[-1] if not vol_df["vol_21d"].dropna().empty else np.nan
    _v252_last = vol_df["vol_252d"].dropna().iloc[-1] if not vol_df["vol_252d"].dropna().empty else np.nan
    _skew_last = skk_df["skewness"].dropna().iloc[-1] if not skk_df["skewness"].dropna().empty else np.nan
    _kurt_last  = skk_df["excess_kurtosis"].dropna().iloc[-1] if not skk_df["excess_kurtosis"].dropna().empty else np.nan
    _vol_bullets = []
    if not np.isnan(_v21_last):
        _regime_now = str(vol_regime(vol_df["vol_21d"].dropna()).iloc[-1]) if not vol_df["vol_21d"].dropna().empty else "unknown"
        _vol_bullets.append(f"Current 21d realized vol is **{_v21_last:.1%}** (annualized) — vol regime is **{_regime_now}** relative to 2-year history.")
    if not np.isnan(_v21_last) and not np.isnan(_v252_last):
        _ratio = _v21_last / _v252_last
        _dir = "above" if _ratio > 1 else "below"
        _vol_bullets.append(f"Short-term vol ({_v21_last:.1%}) is **{_ratio:.1f}x** the 1-year average ({_v252_last:.1%}) — vol is currently {_dir} its long-run baseline.")
    if not np.isnan(_skew_last):
        _skew_interp = "right-skewed (large positive outliers more likely)" if _skew_last > 0.5 else "left-skewed (large negative outliers more likely)" if _skew_last < -0.5 else "roughly symmetric"
        _vol_bullets.append(f"Rolling skewness ({_skew_last:.2f}) — returns are {_skew_interp}.")
    if not np.isnan(_kurt_last) and _kurt_last > 3:
        _vol_bullets.append(f"Excess kurtosis ({_kurt_last:.2f}) is well above 3 — extreme moves occur far more often than a normal distribution predicts. Standard deviation understates risk.")
    if _vol_bullets:
        _bullet_html = "".join(f"<li style='margin-bottom:4px'>{b}</li>" for b in _vol_bullets)
        st.markdown(
            f"<div style='background:#1a1a2e;border-left:4px solid #3498db;padding:12px 16px;border-radius:4px;margin-top:8px'>"
            f"<b style='color:#3498db'>What this means for {_ticker}:</b>"
            f"<ul style='margin-top:8px;color:#ddd;font-size:13px'>{_bullet_html}</ul></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MARKET RISK — TAIL RISK
# ══════════════════════════════════════════════════════════════════════════════
with _mkt_s2:
    st.subheader("Tail Risk Analysis")

    r = simple_r
    hist = historical_var_cvar(r)
    param = parametric_var(r)
    cf = cornish_fisher_var(r)

    # Return distribution with VaR lines
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=r, nbinsx=100, name="Daily Returns",
        marker_color="#3498db", opacity=0.7,
        histnorm="probability density",
    ))
    # Normal fit overlay
    xs = np.linspace(r.min(), r.max(), 300)
    fig_dist.add_trace(go.Scatter(
        x=xs, y=pd.Series(xs).apply(lambda x: __import__("scipy.stats", fromlist=["norm"]).norm.pdf(x, r.mean(), r.std())),
        name="Normal Fit", line=dict(color="white", dash="dash"),
    ))
    var_lines = [
        ("VaR 95% Hist", -hist["var_95"], "#f39c12"),
        ("VaR 99% Hist", -hist["var_99"], "#e74c3c"),
        ("CVaR 99%", -hist["cvar_99"], "#c0392b"),
        ("VaR 99% CF", -cf.get("var_cf_99", hist["var_99"]), "#8e44ad"),
    ]
    for label, x_val, color in var_lines:
        fig_dist.add_vline(x=x_val, line_color=color, line_dash="dot", annotation_text=label, annotation_font_color=color)
    fig_dist.update_layout(title="Return Distribution with VaR/CVaR Overlays", template=DARK, height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

    # QQ plot
    from scipy import stats as sp_stats
    (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(r, dist="norm")
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Quantiles", marker=dict(color="#3498db", size=3)))
    fig_qq.add_trace(go.Scatter(
        x=[min(osm), max(osm)],
        y=[slope * min(osm) + intercept, slope * max(osm) + intercept],
        mode="lines", name="Normal line", line=dict(color="white", dash="dash"),
    ))
    fig_qq.update_layout(title="QQ Plot vs Normal (fat tails = points above line at extremes)", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", template=DARK, height=350)
    st.plotly_chart(fig_qq, use_container_width=True)

    # VaR comparison table
    st.subheader("VaR Comparison Table")
    var_table = pd.DataFrame({
        "Method": ["Historical", "Historical", "Normal", "Normal", "Student-t", "Student-t", "Cornish-Fisher", "Cornish-Fisher"],
        "Level": ["95%", "99%", "95%", "99%", "95%", "99%", "95%", "99%"],
        "VaR (1d)": [
            f"{hist['var_95']:.4f}", f"{hist['var_99']:.4f}",
            f"{param.get('var_normal_95', 0):.4f}", f"{param.get('var_normal_99', 0):.4f}",
            f"{param.get('var_t_95', 0):.4f}", f"{param.get('var_t_99', 0):.4f}",
            f"{cf.get('var_cf_95', 0):.4f}", f"{cf.get('var_cf_99', 0):.4f}",
        ],
        "CVaR (1d)": [
            f"{hist['cvar_95']:.4f}", f"{hist['cvar_99']:.4f}",
            "—", "—", "—", "—", "—", "—",
        ],
    })
    st.dataframe(var_table, use_container_width=True)

    t_df = param.get("t_df", np.nan)
    if not np.isnan(t_df):
        st.caption(f"Student-t fitted degrees of freedom: **{t_df:.2f}** (lower df = fatter tails; df<5 = extreme fat tails)")

    # Worst 10 days
    st.subheader("Worst 10 Trading Days")
    worst = r.nsmallest(10).reset_index()
    worst.columns = ["Date", "Return"]
    worst["Return"] = worst["Return"].map("{:.2%}".format)
    st.dataframe(worst, use_container_width=True)

    # ── INTERPRETIVE COMMENTARY ───────────────────────────────────────────────
    _tail_bullets = []
    _k_val = simple_r.kurtosis()
    _s_val = simple_r.skew()
    _cvar99 = hist["cvar_99"]
    _var99  = hist["var_99"]
    _var_cf = cf.get("var_cf_99", _var99)
    if _k_val > 3:
        _underest = (_var_cf / _var99 - 1) * 100 if _var99 > 0 else 0
        _tail_bullets.append(f"Fat tails are **{'extreme' if _k_val > 5 else 'significant'}** (kurtosis {_k_val:.2f}) — standard normal VaR underestimates true tail risk by approximately **{_underest:.0f}%** (Cornish-Fisher correction).")
    _tail_bullets.append(f"On the worst 1% of trading days, expect losses of **{_cvar99:.2%} or more** (CVaR 99%).")
    if abs(_s_val) > 0.3:
        _skew_msg = (f"Return distribution is **right-skewed ({_s_val:.2f})** — occasional large positive outliers offset frequent small losses."
                     if _s_val > 0 else
                     f"Return distribution is **left-skewed ({_s_val:.2f})** — large negative returns are more frequent than large positive ones. Downside risk is asymmetric.")
        _tail_bullets.append(_skew_msg)
    if _var_cf > _var99 * 1.1:
        _tail_bullets.append(f"Cornish-Fisher VaR 99% ({_var_cf:.2%}) is materially higher than historical VaR ({_var99:.2%}) — use CF-adjusted estimates for position sizing and risk limits.")
    _bullet_html = "".join(f"<li style='margin-bottom:4px'>{b}</li>" for b in _tail_bullets)
    st.markdown(
        f"<div style='background:#1a1a2e;border-left:4px solid #e74c3c;padding:12px 16px;border-radius:4px;margin-top:8px'>"
        f"<b style='color:#e74c3c'>What this means for {_ticker}:</b>"
        f"<ul style='margin-top:8px;color:#ddd;font-size:13px'>{_bullet_html}</ul></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MARKET RISK — RELATIVE RISK
# ══════════════════════════════════════════════════════════════════════════════
with _mkt_s3:
    st.subheader("Relative Risk Analysis")

    beta_spy = rolling_beta(prices, bench_prices)
    beta_sector = rolling_beta(prices, sector_prices)
    corr_spy = rolling_correlation(prices, bench_prices)
    corr_sector = rolling_correlation(prices, sector_prices)
    idio = residual_vol(prices, bench_prices)
    sys_vol = rolling_realized_vol(prices, windows=[63])["vol_63d"]
    ir = information_ratio(prices, sector_prices)

    # Rolling beta
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Scatter(x=beta_spy.index, y=beta_spy, name=f"Beta vs {_bench}", line=dict(color="#3498db")))
    fig_beta.add_trace(go.Scatter(x=beta_sector.index, y=beta_sector, name=f"Beta vs {sector_etf}", line=dict(color="#e74c3c")))
    fig_beta.add_hline(y=1, line_dash="dot", line_color="white")
    fig_beta.update_layout(title="Rolling 63d Beta", yaxis_title="Beta", template=DARK, height=350)
    st.plotly_chart(fig_beta, use_container_width=True)

    # Idio vs systematic decomposition
    aligned_sys = sys_vol.reindex(idio.index).ffill()
    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(
        x=idio.index, y=idio, name="Idiosyncratic Vol",
        fill="tozeroy", mode="lines", line=dict(color="#e74c3c"),
        stackgroup="vol",
    ))
    fig_decomp.add_trace(go.Scatter(
        x=aligned_sys.index, y=(aligned_sys - idio).clip(lower=0),
        name="Systematic Vol",
        fill="tonexty", mode="lines", line=dict(color="#3498db"),
        stackgroup="vol",
    ))
    fig_decomp.update_layout(
        title="Idiosyncratic vs Systematic Vol Decomposition (annualized, 63d rolling)",
        yaxis_title="Annualized Vol", template=DARK, height=350,
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    st.metric(f"Information Ratio vs {sector_etf}", f"{ir:.3f}" if not np.isnan(ir) else "N/A",
              help="IR > 0.5 = good alpha generation; > 1.0 = excellent")

    # Correlation with sector peers
    st.subheader("Correlation Heatmap — Sector Peers")
    with st.spinner("Loading peer data..."):
        peers = fetch_peers(_ticker, start_date, end_date, top_n=10)

    if peers:
        all_prices = {_ticker: prices}
        all_prices.update(peers)
        price_df = pd.DataFrame(all_prices).dropna()
        ret_df = price_df.pct_change().dropna()
        corr_matrix = ret_df.corr()

        fig_heat = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(title="Correlation Matrix — Sector Peers", template=DARK, height=450)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Could not load peer data.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MACRO & REGIME
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Macro Regime Analysis")

    if macro_df.empty:
        st.warning("Macro data unavailable. Set FRED_API_KEY environment variable or ensure pandas-datareader is working.")
    else:
        macro_regimes = classify_macro_regime(macro_df)
        current_regime = str(macro_regimes.iloc[-1]) if len(macro_regimes) > 0 else "UNKNOWN"
        regime_color = REGIME_COLORS.get(current_regime, "#7f8c8d")

        # ── EXACT NUMBERS DRIVING THE REGIME ─────────────────────────────────
        _window_days = 63
        _dgs10_now = float(macro_df["DGS10"].dropna().iloc[-1]) if "DGS10" in macro_df.columns and not macro_df["DGS10"].dropna().empty else np.nan
        _dgs10_prev = float(macro_df["DGS10"].dropna().iloc[-_window_days]) if "DGS10" in macro_df.columns and len(macro_df["DGS10"].dropna()) > _window_days else np.nan
        _dgs10_chg = (_dgs10_now - _dgs10_prev) * 100 if not np.isnan(_dgs10_now) and not np.isnan(_dgs10_prev) else np.nan

        _hy_now = float(macro_df["BAMLH0A0HYM2"].dropna().iloc[-1]) if "BAMLH0A0HYM2" in macro_df.columns and not macro_df["BAMLH0A0HYM2"].dropna().empty else np.nan
        _hy_prev = float(macro_df["BAMLH0A0HYM2"].dropna().iloc[-_window_days]) if "BAMLH0A0HYM2" in macro_df.columns and len(macro_df["BAMLH0A0HYM2"].dropna()) > _window_days else np.nan
        _hy_chg = (_hy_now - _hy_prev) * 100 if not np.isnan(_hy_now) and not np.isnan(_hy_prev) else np.nan

        _yc_now = float(macro_df["T10Y2Y"].dropna().iloc[-1]) if "T10Y2Y" in macro_df.columns and not macro_df["T10Y2Y"].dropna().empty else np.nan

        # Metric cards row
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        def _macro_card(col, label, value, change_bps, change_label, unit=""):
            _dir = "RISING ▲" if (change_bps or 0) > 5 else "FALLING ▼" if (change_bps or 0) < -5 else "FLAT →"
            _dc = "#e74c3c" if _dir.startswith("R") else "#2ecc71" if _dir.startswith("F") else "#95a5a6"
            _val_str = f"{value:.2f}{unit}" if not np.isnan(value) else "N/A"
            _chg_str = f"{change_bps:+.0f} bps (3m)" if change_bps is not None and not np.isnan(change_bps) else ""
            col.markdown(
                f"<div style='background:#1e1e2e;border-radius:6px;padding:12px;border-top:3px solid {_dc}'>"
                f"<div style='font-size:12px;color:#aaa'>{label}</div>"
                f"<div style='font-size:22px;font-weight:bold;color:white'>{_val_str}</div>"
                f"<div style='font-size:11px;color:#aaa'>{_chg_str}</div>"
                f"<div style='font-size:12px;font-weight:bold;color:{_dc}'>{_dir}</div>"
                f"</div>", unsafe_allow_html=True)

        if not np.isnan(_dgs10_now):
            _macro_card(_mc1, "10Y Treasury Yield", _dgs10_now, _dgs10_chg, "rates", "%")
        if not np.isnan(_hy_now):
            _macro_card(_mc2, "HY Credit Spread", _hy_now, _hy_chg, "spreads", "%")
        if not np.isnan(_yc_now):
            _yc_label = "INVERTED ▼" if _yc_now < 0 else "FLAT →" if _yc_now < 0.3 else "NORMAL ▲"
            _yc_dc = "#e74c3c" if _yc_now < 0 else "#f39c12" if _yc_now < 0.3 else "#2ecc71"
            _mc3.markdown(
                f"<div style='background:#1e1e2e;border-radius:6px;padding:12px;border-top:3px solid {_yc_dc}'>"
                f"<div style='font-size:12px;color:#aaa'>Yield Curve (10Y-2Y)</div>"
                f"<div style='font-size:22px;font-weight:bold;color:white'>{_yc_now:.2f}%</div>"
                f"<div style='font-size:12px;font-weight:bold;color:{_yc_dc}'>{_yc_label}</div>"
                f"</div>", unsafe_allow_html=True)

        # Traffic light
        _mc4.markdown(
            f"<div style='background:{regime_color};border-radius:6px;padding:12px;text-align:center;height:100%'>"
            f"<div style='font-size:12px;color:white;opacity:0.8'>Current Regime</div>"
            f"<div style='font-size:20px;font-weight:bold;color:white;margin-top:4px'>{current_regime}</div>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 2×2 REGIME MATRIX ────────────────────────────────────────────────
        st.subheader("Macro Regime Matrix")
        _matrix_cells = [
            ("Rates ↑ + Spreads ↑", "RISK OFF", "#e74c3c", "Tightest credit, highest rate burden. Defensive positioning."),
            ("Rates ↑ + Spreads ↓", "REFLATIONARY", "#e67e22", "Growth expanding, rates catching up. Cyclicals may outperform."),
            ("Rates ↓ + Spreads ↑", "RECESSION FEAR", "#c0392b", "Flight to safety. Credit stress rising. Risk-off equities."),
            ("Rates ↓ + Spreads ↓", "RISK ON", "#2ecc71", "Easy financial conditions. Risk assets broadly favored."),
        ]
        # Highlight current regime cell
        _rate_rising = not np.isnan(_dgs10_chg) and _dgs10_chg > 5
        _spread_widening = not np.isnan(_hy_chg) and _hy_chg > 5
        _active_matrix = {
            (True, True): "RISK OFF",
            (True, False): "REFLATIONARY",
            (False, True): "RECESSION FEAR",
            (False, False): "RISK ON",
        }.get((_rate_rising, _spread_widening), current_regime)

        _m1, _m2 = st.columns(2)
        for idx, (label, rname, color, desc) in enumerate(_matrix_cells):
            _col = _m1 if idx % 2 == 0 else _m2
            _is_active = (rname == _active_matrix)
            _border = f"3px solid {color}" if _is_active else f"1px solid {color}44"
            _bg = f"{color}33" if _is_active else f"{color}11"
            _active_badge = " ← CURRENT" if _is_active else ""
            _col.markdown(
                f"<div style='background:{_bg};border:{_border};border-radius:8px;padding:12px;margin-bottom:8px'>"
                f"<b style='color:{color}'>{label}</b><b style='color:{color}'>{_active_badge}</b><br>"
                f"<span style='font-size:16px;font-weight:bold;color:white'>{rname}</span><br>"
                f"<span style='font-size:12px;color:#ccc'>{desc}</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CHARTS ──────────────────────────────────────────────────────────
        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["10Y Treasury Yield (%)", "HY Credit Spread (%)", "Yield Curve (10Y-2Y, %)"])

        for series_id, row in [("DGS10", 1), ("BAMLH0A0HYM2", 2), ("T10Y2Y", 3)]:
            if series_id in macro_df.columns:
                fig_macro.add_trace(
                    go.Scatter(x=macro_df.index, y=macro_df[series_id], name=series_id, line=dict(width=1.5)),
                    row=row, col=1,
                )

        # Yield curve zero line
        if "T10Y2Y" in macro_df.columns:
            fig_macro.add_hline(y=0, row=3, col=1, line_dash="dot", line_color="white", line_width=1)

        # Regime bands (block-compressed)
        if len(macro_regimes) > 0:
            prev_r = str(macro_regimes.iloc[0])
            block_start = macro_regimes.index[0]
            for t, r in macro_regimes.items():
                r = str(r)
                if r != prev_r:
                    fig_macro.add_vrect(x0=block_start, x1=t,
                        fillcolor=REGIME_COLORS.get(prev_r, "gray"), opacity=0.08, line_width=0)
                    block_start = t
                    prev_r = r
            fig_macro.add_vrect(x0=block_start, x1=macro_regimes.index[-1],
                fillcolor=REGIME_COLORS.get(prev_r, "gray"), opacity=0.08, line_width=0)

        fig_macro.update_layout(template=DARK, height=550, title="Macro Indicators with Regime Bands")
        st.plotly_chart(fig_macro, use_container_width=True)

        # ── CONDITIONAL PERFORMANCE TABLE ─────────────────────────────────
        st.subheader(f"{_ticker} Historical Performance by Macro Regime")
        aligned_macro = macro_regimes.reindex(prices.index, method="ffill").dropna()
        cond_macro = conditional_returns_by_regime(prices.reindex(aligned_macro.index), aligned_macro)
        if not cond_macro.empty:
            st.dataframe(
                cond_macro.style.format({
                    "mean_daily_ret": "{:.4f}", "ann_return": "{:.2%}",
                    "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
                }).background_gradient(subset=[c for c in ["ann_return", "sharpe"] if c in cond_macro.columns], cmap="RdYlGn"),
                use_container_width=True,
            )
            # Plain-English summary for current regime
            if current_regime in cond_macro.index:
                _cr = cond_macro.loc[current_regime]
                st.info(
                    f"In the current **{current_regime}** regime, **{_ticker}** has historically returned "
                    f"**{_cr['ann_return']:.1%}** annualized with **{_cr['ann_vol']:.1%}** vol "
                    f"(Sharpe {_cr['sharpe']:.2f}) over **{int(_cr['n_days'])}** trading days."
                )

        # VIX regime conditional returns
        if "VIXCLS" in macro_df.columns:
            vix_s = macro_df["VIXCLS"].dropna()
            vix_regimes = vix_regime(vix_s)
            st.subheader("Conditional Performance by VIX Regime")
            vix_df_merged = pd.DataFrame({"vix": vix_s, "vix_regime": vix_regimes})
            merged = vix_df_merged.join(simple_r.rename("ret"), how="inner").dropna()
            cond_vix = conditional_returns_by_regime(prices.reindex(merged.index), merged["vix_regime"].astype(str))
            if not cond_vix.empty:
                st.dataframe(
                    cond_vix.style.format({
                        "mean_daily_ret": "{:.4f}", "ann_return": "{:.2%}",
                        "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
                    }),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# MARKET RISK — BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════
with _mkt_s4:
    _bt_cols = st.columns(3)
    st.session_state["fast_ma"] = _bt_cols[0].number_input(
        "Fast MA", min_value=5, max_value=100,
        value=st.session_state["fast_ma"], step=5)
    st.session_state["slow_ma"] = _bt_cols[1].number_input(
        "Slow MA", min_value=10, max_value=300,
        value=st.session_state["slow_ma"], step=10)
    _sz_opts = ["vol_target", "kelly", "fixed"]
    st.session_state["sizing_method"] = _bt_cols[2].selectbox(
        "Position Sizing", _sz_opts,
        index=_sz_opts.index(st.session_state["sizing_method"]))
    fast_ma = st.session_state["fast_ma"]
    slow_ma = st.session_state["slow_ma"]
    sizing_method = st.session_state["sizing_method"]

    st.subheader(f"MA Crossover Backtest — {_ticker}")
    st.caption(f"Strategy: Long when {fast_ma}d MA > {slow_ma}d MA | Sizing: {sizing_method}")

    _bt_key = f"bt_{_ticker}_{fast_ma}_{slow_ma}_{sizing_method}"
    if _bt_key not in st.session_state:
        with st.spinner("Running backtest..."):
            _bt = run_backtest(prices, fast_ma=int(fast_ma), slow_ma=int(slow_ma), sizing=sizing_method)
            _ci = bootstrap_sharpe_ci(_bt["returns"], n_boot=500)
            _stress = stress_test(prices, fast_ma=int(fast_ma), slow_ma=int(slow_ma))
            st.session_state[_bt_key] = (_bt, _ci, _stress)
    bt, ci, stress = st.session_state[_bt_key]

    s = bt["summary"]
    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
    bm1.metric("Total Return", f"{s['total_return']:.2%}")
    bm2.metric("Ann. Return", f"{s['ann_return']:.2%}")
    bm3.metric("Sharpe", f"{s['sharpe']:.2f}")
    bm4.metric("Max Drawdown", f"{s['max_drawdown']:.2%}")
    bm5.metric("# Trades", s["n_trades"])

    st.caption(f"Sharpe 95% CI (bootstrap, n=1000): [{ci['lower']:.2f}, {ci['upper']:.2f}]")

    # Equity curve with drawdown shading
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        subplot_titles=["Equity Curve", "Drawdown"])

    fig_bt.add_trace(go.Scatter(
        x=bt["equity"].index, y=bt["equity"], name="Strategy",
        line=dict(color="#2ecc71"), fill="tozeroy", fillcolor="rgba(46,204,113,0.05)",
    ), row=1, col=1)

    # Buy-and-hold reference
    bah = (1 + simple_r).cumprod()
    bah_aligned = bah.reindex(bt["equity"].index)
    fig_bt.add_trace(go.Scatter(
        x=bah_aligned.index, y=bah_aligned, name="Buy & Hold",
        line=dict(color="#3498db", dash="dash"),
    ), row=1, col=1)

    fig_bt.add_trace(go.Scatter(
        x=bt["drawdown"].index, y=bt["drawdown"],
        fill="tozeroy", name="Drawdown",
        line=dict(color="#e74c3c"), fillcolor="rgba(231,76,60,0.3)",
    ), row=2, col=1)

    fig_bt.update_layout(template=DARK, height=500)
    st.plotly_chart(fig_bt, use_container_width=True)

    # Rolling Sharpe + Rolling VaR
    fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Rolling 63d Sharpe", "Rolling 63d VaR 95%"])
    fig_roll.add_trace(go.Scatter(x=bt["rolling_sharpe"].index, y=bt["rolling_sharpe"], name="Rolling Sharpe", line=dict(color="#f39c12")), row=1, col=1)
    fig_roll.add_hline(y=0, row=1, col=1, line_dash="dot", line_color="white")
    fig_roll.add_trace(go.Scatter(x=bt["rolling_var95"].index, y=bt["rolling_var95"], name="Rolling VaR 95%", line=dict(color="#e74c3c")), row=2, col=1)
    fig_roll.update_layout(template=DARK, height=400)
    st.plotly_chart(fig_roll, use_container_width=True)

    # Stress test
    st.subheader("Stress Test Results")
    if not stress.empty:
        numeric_cols = {"total_return": "{:.2%}", "ann_vol": "{:.2%}", "max_drawdown": "{:.2%}", "sharpe": "{:.2f}"}
        fmt = {k: v for k, v in numeric_cols.items() if k in stress.columns}
        gradient_cols = [c for c in ["total_return", "max_drawdown"] if c in stress.columns]
        styler = stress.style.format(fmt)
        if gradient_cols:
            styler = styler.background_gradient(subset=gradient_cols, cmap="RdYlGn")
        st.dataframe(styler, use_container_width=True)
    else:
        st.info("No stress test data available for the selected period.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"Fundamental Risk Analysis — {_ticker}")

    _fund_key = f"fund_{_ticker}"
    if _fund_key not in st.session_state:
        with st.spinner("Loading fundamental data (first run ~20s, cached after)..."):
            fred_rate = None
            if not macro_df.empty and "DGS10" in macro_df.columns:
                fred_rate = float(macro_df["DGS10"].dropna().iloc[-1])

            from risk.metrics import classify_macro_regime as _cmr
            _mr = _cmr(macro_df) if not macro_df.empty else pd.Series(dtype=str)
            current_regime = str(_mr.iloc[-1]) if len(_mr) > 0 else "MIXED"

            val_r   = valuation_risk(_ticker, fred_dgs10=fred_rate)
            bs_r    = balance_sheet_risk(_ticker)
            eq_r    = earnings_quality(_ticker)
            cf_r    = cashflow_risk(_ticker)
            gv_r    = growth_value_classification(_ticker, macro_regime=current_regime)
            fund_scores = composite_fundamental_score(val_r, bs_r, eq_r, cf_r, gv_r)
            ticker_name = info.get("name", _ticker)
            news_items  = fetch_material_news(_ticker, ticker_name=ticker_name)
            st.session_state[_fund_key] = (val_r, bs_r, eq_r, cf_r, gv_r, fund_scores, news_items)

    val_r, bs_r, eq_r, cf_r, gv_r, fund_scores, news_items = st.session_state[_fund_key]
    st.session_state.fund_scores = fund_scores

    # ── DIVERGENCE ALERT BANNER ───────────────────────────────────────────────
    mkt_score  = composite_risk_score(prices, volume, bench_prices)["total"]
    fund_total = fund_scores["total"]
    div        = abs(mkt_score - fund_total)

    if div < 20:
        div_color = "#2ecc71"
        div_text  = "🟢 ALIGNED — Market and fundamental risk agree"
        div_detail = f"Both scores are within {div:.0f} pts of each other. No material divergence signal."
    elif div < 40:
        div_color = "#f39c12"
        div_text  = "🟡 MILD DIVERGENCE — Signals mixed, monitor closely"
        div_detail = f"Market Risk: {mkt_score:.0f} | Fundamental Risk: {fund_total:.0f} | Gap: {div:.0f} pts"
    else:
        div_color = "#e74c3c"
        if mkt_score > fund_total:
            div_text   = "🔴 STRONG DIVERGENCE — Market may be OVERPRICING RISK"
            div_detail = (
                f"Market Risk Score ({mkt_score:.0f}) >> Fundamental Score ({fund_total:.0f}). "
                "Price action is unusually volatile/stressed relative to fundamentals. "
                "Potential opportunity if business quality holds."
            )
        else:
            div_text   = "🔴 STRONG DIVERGENCE — Market may be UNDERPRICING RISK"
            div_detail = (
                f"Fundamental Risk Score ({fund_total:.0f}) >> Market Risk Score ({mkt_score:.0f}). "
                "Fundamentals are deteriorating beneath a calm price surface. "
                "Consider reducing exposure or hedging."
            )

    st.markdown(
        f"<div style='background:{div_color}22;border-left:5px solid {div_color};"
        f"padding:14px 18px;border-radius:6px;margin-bottom:20px'>"
        f"<div style='font-size:18px;font-weight:bold;color:{div_color}'>{div_text}</div>"
        f"<div style='margin-top:6px;color:#ccc'>{div_detail}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Active flags
    all_flags = fund_scores.get("all_flags", [])
    if all_flags:
        flag_str = "  ".join([f"`{f}`" for f in all_flags])
        st.warning(f"**Active Risk Flags:** {flag_str}")

    # ── FOUR SCORE CARDS ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    def _score_card(col, label, score, key_lines, flags):
        color = "#2ecc71" if score < 33 else "#f39c12" if score < 66 else "#e74c3c"
        flag_html = "".join(
            f"<div style='font-size:11px;color:{color};margin-top:2px'>⚠ {f}</div>"
            for f in flags
        )
        lines_html = "".join(
            f"<div style='font-size:12px;color:#ccc'>{l}</div>" for l in key_lines
        )
        col.markdown(
            f"<div style='background:#1e1e2e;border-radius:8px;padding:14px;"
            f"border-top:3px solid {color}'>"
            f"<div style='font-size:13px;color:#aaa'>{label}</div>"
            f"<div style='font-size:36px;font-weight:bold;color:{color}'>{score:.0f}</div>"
            f"{lines_html}{flag_html}</div>",
            unsafe_allow_html=True,
        )

    def _fmt(v, fmt=".2f", suffix=""):
        return f"{v:{fmt}}{suffix}" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "N/A"

    _score_card(c1, "Valuation Risk", val_r["score"],
        [f"P/E: {_fmt(val_r['metrics']['trailingPE'])}",
         f"Fwd P/E: {_fmt(val_r['metrics']['forwardPE'])}",
         f"EV/EBITDA: {_fmt(val_r['metrics']['evToEbitda'])}",
         f"ERP: {_fmt(val_r['equity_risk_premium'], '.2%')}"],
        val_r["flags"])

    _score_card(c2, "Balance Sheet Risk", bs_r["score"],
        [f"Z-Score: {_fmt(bs_r['z_score'])}",
         f"D/E: {_fmt(bs_r['debt_to_equity'])}",
         f"Interest Cov: {_fmt(bs_r['interest_coverage'])}",
         f"Current: {_fmt(bs_r['current_ratio'])}"],
        bs_r["flags"])

    _score_card(c3, "Earnings Quality", eq_r["score"],
        [f"Accruals: {_fmt(eq_r['accruals_ratio'], '.2%')}",
         f"Consec. Misses: {eq_r['consecutive_misses']}",
         f"Rev Growth Std: {_fmt(eq_r['rev_growth_std'], '.2%')}"],
        eq_r["flags"])

    _score_card(c4, "Cash Flow Risk", cf_r["score"],
        [f"FCF Yield: {_fmt(cf_r['fcf_yield'], '.2%')}",
         f"Shareholder Yield: {_fmt(cf_r['shareholder_yield'], '.2%')}",
         f"Neg FCF Streak: {cf_r['neg_fcf_streak']}y"],
        cf_r["flags"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GROWTH/VALUE + REGIME ─────────────────────────────────────────────────
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Classification", gv_r["classification"])
    gc2.metric("Regime Favors", gv_r["regime_favors"])
    gc3.metric("P/E Sector Percentile", f"{_fmt(gv_r['pe_percentile'], '.0f')}th")
    if "REGIME_MISMATCH" in gv_r["flags"]:
        st.error(f"⚠️ REGIME MISMATCH: {gv_r['classification']} stock in {gv_r['macro_regime']} regime. "
                 f"Regime currently favors {gv_r['regime_favors']} stocks.")

    # ── ALTMAN Z-SCORE COMPONENTS ────────────────────────────────────────────
    st.subheader("Altman Z-Score Breakdown")
    _z = bs_r.get("z_score", np.nan)
    if not np.isnan(_z):
        _z_color = "#2ecc71" if _z > 2.99 else "#f39c12" if _z > 1.81 else "#e74c3c"
        _z_label = "SAFE ZONE (Z > 2.99)" if _z > 2.99 else "GREY ZONE (1.81 < Z < 2.99)" if _z > 1.81 else "DISTRESS ZONE (Z < 1.81)"
        st.markdown(
            f"<div style='background:{_z_color}22;border:1px solid {_z_color};border-radius:6px;padding:14px;margin-bottom:10px'>"
            f"<div style='font-size:13px;color:#aaa'>Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5</div>"
            f"<div style='font-size:28px;font-weight:bold;color:{_z_color}'>Z = {_z:.2f} — {_z_label}</div>"
            f"<div style='font-size:12px;color:#ccc;margin-top:6px'>"
            f"X1=Working Capital/Assets &nbsp;|&nbsp; X2=Retained Earnings/Assets &nbsp;|&nbsp; "
            f"X3=EBIT/Assets &nbsp;|&nbsp; X4=Mkt Cap/Total Liabilities &nbsp;|&nbsp; X5=Revenue/Assets"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        # Gauge for Z-score
        _fig_z = go.Figure(go.Indicator(
            mode="gauge+number",
            value=_z,
            title={"text": "Altman Z-Score"},
            gauge={
                "axis": {"range": [0, 5]},
                "bar": {"color": _z_color},
                "steps": [
                    {"range": [0, 1.81], "color": "#3a0a0a"},
                    {"range": [1.81, 2.99], "color": "#3a2e0a"},
                    {"range": [2.99, 5], "color": "#0d2b0d"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": _z},
            },
        ))
        _fig_z.update_layout(height=220, template=DARK, margin=dict(t=30, b=10))
        st.plotly_chart(_fig_z, use_container_width=True)
    else:
        st.info("Altman Z-Score could not be computed (insufficient balance sheet data).")

    # ── PEER COMPARISON TABLE ─────────────────────────────────────────────────
    st.subheader("Valuation vs Sector Peers")
    # Build multi-ticker peer comparison (fetch live data for key peers)
    _sector_info = info.get("sector", "")
    # Industry-specific hardcoded peers for common sectors (as fallback / supplement)
    _industry_peers = {
        "Airlines": ["DAL", "UAL", "LUV", "SAVE", "JBLU"],
        "Oil & Gas E&P": ["CVX", "COP", "EOG", "PXD", "DVN"],
        "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],
        "Banks": ["JPM", "BAC", "WFC", "C", "GS"],
        "Biotech": ["AMGN", "GILD", "REGN", "VRTX", "BIIB"],
    }
    _industry = info.get("industry", "")
    _hardcoded_peers = next(
        (v for k, v in _industry_peers.items() if k.lower() in _industry.lower()),
        []
    )
    # Use cached peer info from val_r, supplement with hardcoded if peer_count is low
    _peer_tickers = ([r["ticker"] for _, r in val_r.get("_peer_df_rows", {}).items()]
                     if "_peer_df_rows" in val_r else [])
    if len(_peer_tickers) < 3 and _hardcoded_peers:
        _peer_tickers = [p for p in _hardcoded_peers if p != _ticker]

    _peer_compare_rows = []
    for _pt in _peer_tickers[:6]:
        try:
            _pi = yf.Ticker(_pt).info or {}
            _peer_fcf_yield = np.nan
            try:
                _pcf = yf.Ticker(_pt).cashflow
                if _pcf is not None and not _pcf.empty:
                    _pocf_row = next((i for i in _pcf.index if "operating cash flow" in str(i).lower()), None)
                    _pcap_row = next((i for i in _pcf.index if "capital expenditure" in str(i).lower()), None)
                    if _pocf_row and _pcap_row:
                        _pocf = float(_pcf.loc[_pocf_row].iloc[0]) if pd.notna(_pcf.loc[_pocf_row].iloc[0]) else np.nan
                        _pcap = float(_pcf.loc[_pcap_row].iloc[0]) if pd.notna(_pcf.loc[_pcap_row].iloc[0]) else np.nan
                        _pmcap = _pi.get("marketCap")
                        if not any(np.isnan(x) for x in [_pocf, _pcap]) and _pmcap:
                            _peer_fcf_yield = (_pocf - abs(_pcap)) / _pmcap
            except Exception:
                pass
            _peer_compare_rows.append({
                "Ticker": _pt,
                "P/E": _fmt(_pi.get("trailingPE")),
                "P/B": _fmt(_pi.get("priceToBook")),
                "EV/EBITDA": _fmt(_pi.get("enterpriseToEbitda")),
                "D/E": _fmt(_pi.get("debtToEquity")),
                "FCF Yield": f"{_peer_fcf_yield:.1%}" if not np.isnan(_peer_fcf_yield) else "N/A",
            })
        except Exception:
            continue

    # Insert subject ticker first
    _own_fcf_yield = cf_r.get("fcf_yield", np.nan)
    _subject_row = {
        "Ticker": f"★ {_ticker}",
        "P/E": _fmt(val_r["metrics"]["trailingPE"]),
        "P/B": _fmt(val_r["metrics"]["priceToBook"]),
        "EV/EBITDA": _fmt(val_r["metrics"]["evToEbitda"]),
        "D/E": _fmt(bs_r.get("debt_to_equity")),
        "FCF Yield": f"{_own_fcf_yield:.1%}" if not np.isnan(_own_fcf_yield) else "N/A",
    }
    _all_peer_rows = [_subject_row] + _peer_compare_rows
    if _all_peer_rows:
        st.dataframe(pd.DataFrame(_all_peer_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Peer comparison data unavailable. Sector peers could not be loaded.")

    # Sector percentile ranking (from val_r)
    peer_metrics = [
        ("Trailing P/E",  val_r["metrics"]["trailingPE"],   val_r["peer_ranks"].get("trailingPE_pct")),
        ("Forward P/E",   val_r["metrics"]["forwardPE"],    val_r["peer_ranks"].get("forwardPE_pct")),
        ("Price/Book",    val_r["metrics"]["priceToBook"],  val_r["peer_ranks"].get("priceToBook_pct")),
        ("Price/Sales",   val_r["metrics"]["priceToSales"], val_r["peer_ranks"].get("priceToSales_pct")),
        ("EV/EBITDA",     val_r["metrics"]["evToEbitda"],   val_r["peer_ranks"].get("evToEbitda_pct")),
    ]
    prank_rows = []
    for label, value, rank in peer_metrics:
        prank_rows.append({
            "Metric": label,
            f"{_ticker} Value": _fmt(value),
            "Sector Percentile": f"{rank:.0f}th" if rank is not None and not np.isnan(rank) else "N/A",
            "Signal": ("🔴 Expensive" if rank and rank > 75 else "🟢 Cheap" if rank and rank < 25 else "🟡 Neutral") if rank is not None and not np.isnan(rank) else "—",
        })
    st.caption("Sector percentile rank (0th = cheapest in sector, 100th = most expensive):")
    st.dataframe(pd.DataFrame(prank_rows), use_container_width=True, hide_index=True)

    # ── EARNINGS SURPRISE HISTORY ─────────────────────────────────────────────
    st.subheader("Earnings Surprise History (last 8 quarters)")
    if eq_r["surprise_history"]:
        surp_df = pd.DataFrame(eq_r["surprise_history"])
        fig_surp = go.Figure(go.Bar(
            x=surp_df["date"],
            y=surp_df["surprise_pct"] * 100,
            marker_color=["#2ecc71" if b else "#e74c3c" for b in surp_df["beat"]],
            text=[f"{v:.1f}%" for v in surp_df["surprise_pct"] * 100],
            textposition="outside",
            name="EPS Surprise %",
        ))
        fig_surp.add_hline(y=0, line_color="white", line_dash="dot")
        fig_surp.update_layout(
            title="EPS Surprise % (positive = beat, negative = miss)",
            yaxis_title="Surprise %", template=DARK, height=300,
        )
        st.plotly_chart(fig_surp, use_container_width=True)
    else:
        st.info("No earnings surprise data available.")

    # ── FCF WATERFALL ────────────────────────────────────────────────────────
    st.subheader("Free Cash Flow — 4-Year Trend")
    if any(not np.isnan(f) for f in cf_r["annual_fcf"]):
        n_years = min(4, len(cf_r["annual_fcf"]))
        year_labels = [f"Y-{i}" for i in range(n_years - 1, -1, -1)]
        fcf_vals = [f / 1e9 if not np.isnan(f) else 0 for f in cf_r["annual_fcf"][:n_years]][::-1]
        margins  = [m * 100 if m is not None and not np.isnan(m) else None for m in cf_r["fcf_margins"][:n_years]][::-1]

        fig_fcf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fcf.add_trace(go.Bar(
            x=year_labels, y=fcf_vals,
            name="FCF ($B)",
            marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in fcf_vals],
        ), secondary_y=False)
        valid_margins = [(l, m) for l, m in zip(year_labels, margins) if m is not None]
        if valid_margins:
            fig_fcf.add_trace(go.Scatter(
                x=[l for l, _ in valid_margins],
                y=[m for _, m in valid_margins],
                name="FCF Margin %", mode="lines+markers",
                line=dict(color="#3498db", width=2),
            ), secondary_y=True)
        fig_fcf.update_layout(title="Free Cash Flow & Margin", template=DARK, height=300)
        fig_fcf.update_yaxes(title_text="FCF ($B)", secondary_y=False)
        fig_fcf.update_yaxes(title_text="FCF Margin %", secondary_y=True)
        st.plotly_chart(fig_fcf, use_container_width=True)
    else:
        st.info("FCF data unavailable.")

    # ── MATERIAL NEWS FEED ───────────────────────────────────────────────────
    st.subheader("Material News")
    if news_items:
        SENTIMENT_COLOR = {
            "POSITIVE": "#2ecc71",
            "NEGATIVE": "#e74c3c",
            "NEUTRAL": "#95a5a6",
            "MIXED": "#f39c12",
        }
        for category in ["Company News", "Sector News", "Macro News"]:
            cat_items = [n for n in news_items if n["category"] == category]
            if not cat_items:
                continue
            st.markdown(f"**{category}**")
            for item in cat_items:
                sc = item["sentiment"]
                color = SENTIMENT_COLOR.get(sc, "#95a5a6")
                stars = "★" * item["materiality"] + "☆" * (3 - item["materiality"])
                title_display = f"[{item['title']}]({item['url']})" if item["url"] else item["title"]
                st.markdown(
                    f"<div style='border-left:3px solid {color};padding:6px 12px;margin-bottom:6px;background:#1a1a2e;border-radius:0 4px 4px 0'>"
                    f"<span style='color:{color};font-size:11px'>{sc}</span> "
                    f"<span style='color:#888;font-size:11px'>{stars} · {item['date']} · {item['source']}</span><br>"
                    f"<span style='font-size:13px'>{title_display}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No material news found for this ticker.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SMART MONEY & FACTORS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    _sm_s1, _sm_s2 = st.tabs(["🏛️ Smart Money", "📊 Factor & Shock"])

with _sm_s1:
    st.subheader(f"Smart Money — SEC EDGAR Data for {_ticker}")
    st.caption("Form 4 insider transactions · 13F institutional holdings · 13D/13G activist detection · XBRL fundamentals")

    _sm_key = f"smart_money_{_ticker}"
    if _sm_key not in st.session_state:
        _sm_progress = st.progress(0, text="Resolving CIK from EDGAR...")
        try:
            from data.edgar import resolve_cik as _rcik
            _cik_val = _rcik(_ticker)
            _sm_progress.progress(15, text="Fetching Form 4 insider transactions...")
            _insider_df = fetch_insider_transactions(_ticker, days_back=180)
            _sm_progress.progress(40, text="Computing insider signals...")
            _insider_signals = compute_insider_signals(_insider_df)
            _sm_progress.progress(55, text="Fetching 13F institutional holdings...")
            _inst_df = fetch_institutional_changes(_ticker)
            _sm_progress.progress(75, text="Computing institutional signals...")
            _inst_signals = compute_institutional_signals(_inst_df)
            _sm_progress.progress(85, text="Checking 13D/13G activist filings...")
            _activist_df = fetch_activist_positions(_ticker)
            _sm_progress.progress(95, text="Fetching XBRL fundamentals...")
            _xbrl = fetch_xbrl_fundamentals(_ticker)
            _sm_progress.progress(100, text="Done.")
            _sm_progress.empty()
            st.session_state[_sm_key] = (_cik_val, _insider_df, _insider_signals,
                                          _inst_df, _inst_signals, _activist_df, _xbrl)
        except Exception as _e:
            _sm_progress.empty()
            st.error(f"EDGAR data load failed: {_e}")
            st.session_state[_sm_key] = ("ERROR", None, {}, None, {}, None, {})

    (_cik_val, _insider_df, _insider_signals,
     _inst_df, _inst_signals, _activist_df, _xbrl) = st.session_state[_sm_key]

    if not _cik_val or _cik_val == "ERROR":
        st.warning(f"EDGAR data unavailable for {_ticker}. CIK could not be resolved.")
        _cik_val = None
        _insider_df = pd.DataFrame()
        _insider_signals = {}
        _inst_df = pd.DataFrame()
        _inst_signals = {"n_adding": 0, "n_reducing": 0, "n_new": 0, "n_closed": 0, "signal": "NEUTRAL"}
        _activist_df = pd.DataFrame()
        _xbrl = {"source": "unavailable"}
    else:
        st.caption(f"CIK: {_cik_val}  |  Data source: SEC EDGAR (free API)")

    # ── SECTION 1: INSIDER ACTIVITY ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Section 1: Insider Activity (Form 4)")

    # Cluster alert banner
    _cluster = _insider_signals.get("cluster_signal")
    _cluster_names = _insider_signals.get("cluster_insiders", [])
    if _cluster == "BUY":
        st.markdown(
            f"<div style='background:#2ecc7122;border-left:5px solid #2ecc71;padding:12px 18px;border-radius:6px;margin-bottom:12px'>"
            f"<b style='color:#2ecc71;font-size:16px'>🟢 INSIDER BUYING CLUSTER</b><br>"
            f"<span style='color:#ccc'>{len(_cluster_names)} insiders bought in the last 30 days: {', '.join(_cluster_names[:5])}</span>"
            f"</div>", unsafe_allow_html=True)
    elif _cluster == "SELL":
        st.markdown(
            f"<div style='background:#e74c3c22;border-left:5px solid #e74c3c;padding:12px 18px;border-radius:6px;margin-bottom:12px'>"
            f"<b style='color:#e74c3c;font-size:16px'>🔴 INSIDER SELLING CLUSTER</b><br>"
            f"<span style='color:#ccc'>{len(_cluster_names)} insiders sold in the last 30 days: {', '.join(_cluster_names[:5])}</span>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.info("🟡 No insider cluster signal detected in the last 30 days.")

    # Summary metrics
    _si = _insider_signals
    _ic1, _ic2, _ic3, _ic4 = st.columns(4)
    def _fmt_dollars(v):
        if v == 0 or np.isnan(v): return "$0"
        sign = "+" if v > 0 else ""
        if abs(v) >= 1e6: return f"{sign}${abs(v)/1e6:.1f}M"
        if abs(v) >= 1e3: return f"{sign}${abs(v)/1e3:.0f}K"
        return f"{sign}${abs(v):.0f}"
    _ic1.metric("Net Insider Flow (30d)", _fmt_dollars(_si["net_30d"]),
                help="Positive = net buying, negative = net selling (open market only)")
    _ic2.metric("Net Insider Flow (90d)", _fmt_dollars(_si["net_90d"]))
    _bsr = _si.get("buy_sell_ratio_90d", np.nan)
    _ic3.metric("Buy/Sell Ratio (90d)", f"{_bsr:.2f}" if not np.isnan(_bsr) else "N/A",
                help=">0.6 = bullish signal; <0.4 = bearish")
    _ic4.metric("Weighted Net (CEO/CFO 2×)", _fmt_dollars(_si["weighted_net_90d"]))

    if not _insider_df.empty:
        # Price chart with insider transactions overlaid
        _fig_ins = go.Figure()
        _fig_ins.add_trace(go.Scatter(
            x=prices.index, y=prices,
            name="Price", line=dict(color="#3498db", width=1.5),
        ))

        _ins_plot = _insider_df.copy()
        _ins_plot["date"] = pd.to_datetime(_ins_plot["date"], errors="coerce")
        _ins_plot = _ins_plot.dropna(subset=["date"])
        _ins_plot = _ins_plot[(_ins_plot["date"] >= prices.index[0]) & (_ins_plot["date"] <= prices.index[-1])]

        if not _ins_plot.empty:
            # Map transaction dates to nearest price dates
            _ins_plot["price_at_date"] = _ins_plot["date"].apply(
                lambda d: float(prices.reindex([d], method="nearest").iloc[0]) if len(prices) > 0 else np.nan
            )

            _buys_plt = _ins_plot[_ins_plot["is_buy"]]
            _sells_plt = _ins_plot[~_ins_plot["is_buy"]]

            max_dv = _ins_plot["dollar_value"].max() if not _ins_plot.empty else 1
            _size_scale = lambda dv: max(8, min(30, int(dv / max_dv * 25) + 8))

            if not _buys_plt.empty:
                _fig_ins.add_trace(go.Scatter(
                    x=_buys_plt["date"],
                    y=_buys_plt["price_at_date"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        color="#2ecc71",
                        size=[_size_scale(v) for v in _buys_plt["dollar_value"]],
                        line=dict(color="white", width=1),
                    ),
                    name="Insider Buy",
                    text=[f"{r['name']} ({r['title']})<br>${r['dollar_value']:,.0f} ({r['shares']:,.0f} sh @ ${r['price']:.2f})"
                          for _, r in _buys_plt.iterrows()],
                    hovertemplate="%{text}<extra></extra>",
                ))
            if not _sells_plt.empty:
                _fig_ins.add_trace(go.Scatter(
                    x=_sells_plt["date"],
                    y=_sells_plt["price_at_date"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        color="#e74c3c",
                        size=[_size_scale(v) for v in _sells_plt["dollar_value"]],
                        line=dict(color="white", width=1),
                    ),
                    name="Insider Sale",
                    text=[f"{r['name']} ({r['title']})<br>${r['dollar_value']:,.0f} ({r['shares']:,.0f} sh @ ${r['price']:.2f})"
                          for _, r in _sells_plt.iterrows()],
                    hovertemplate="%{text}<extra></extra>",
                ))

        _fig_ins.update_layout(
            title=f"{_ticker} Price with Insider Transactions (▲ Buy / ▼ Sell, size = $ value)",
            yaxis_title="Price ($)", template=DARK, height=400,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(_fig_ins, use_container_width=True)

        # Transactions table
        st.subheader("Recent Insider Transactions (last 180 days)")
        _show_df = _insider_df.head(30).copy()
        _show_df["dollar_value"] = _show_df["dollar_value"].apply(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
        _show_df["conviction_pct"] = _show_df["conviction_pct"].apply(lambda v: f"{v:.1f}%" if pd.notna(v) and not np.isnan(v) else "N/A")
        _show_df["type_label"] = _show_df["is_buy"].map({True: "🟢 BUY", False: "🔴 SELL"})
        _display_cols = ["date", "name", "title", "type_label", "shares", "price", "dollar_value", "conviction_pct"]
        _avail_cols = [c for c in _display_cols if c in _show_df.columns]
        st.dataframe(_show_df[_avail_cols].rename(columns={
            "type_label": "Type", "conviction_pct": "Conviction",
            "dollar_value": "$ Value",
        }), use_container_width=True, hide_index=True)
    else:
        st.info(f"No open-market insider transactions found for {_ticker} in the last 180 days via EDGAR.")

    # ── SECTION 2: INSTITUTIONAL / HEDGE FUND HOLDINGS ───────────────────────
    st.markdown("---")
    st.subheader("🏦 Section 2: Institutional Holdings (13F)")

    _inst_sig = _inst_signals
    _is1, _is2, _is3, _is4 = st.columns(4)
    _is1.metric("Funds Adding", _inst_sig["n_adding"], help="# funds that increased position this quarter")
    _is2.metric("Funds Reducing", _inst_sig["n_reducing"])
    _is3.metric("New Positions", _inst_sig["n_new"], help="Funds that initiated a new position (most bullish)")
    _is4.metric("Closed Positions", _inst_sig["n_closed"], help="Funds that fully exited (most bearish)")

    _inst_color = "#2ecc71" if _inst_sig["signal"] == "ACCUMULATION" else "#e74c3c" if _inst_sig["signal"] == "DISTRIBUTION" else "#95a5a6"
    st.markdown(
        f"<div style='background:{_inst_color}22;border-left:4px solid {_inst_color};padding:10px 16px;border-radius:4px;margin:8px 0'>"
        f"<b style='color:{_inst_color}'>Institutional Signal: {_inst_sig['signal']}</b>"
        f"</div>", unsafe_allow_html=True)

    if not _inst_df.empty:
        # Color-coded change table
        _inst_display = _inst_df.copy()
        _inst_display["change_pct_fmt"] = _inst_display["change_pct"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "NEW" if True else "N/A"
        )
        for _, row in _inst_df.iterrows():
            if row["signal"] == "NEW":
                _inst_display.loc[_, "change_pct_fmt"] = "NEW ★"
            elif row["signal"] == "CLOSED":
                _inst_display.loc[_, "change_pct_fmt"] = "CLOSED ✗"

        _inst_display["curr_value_fmt"] = _inst_display["curr_value"].apply(
            lambda v: f"${v/1e6:.1f}M" if pd.notna(v) and v > 0 else "—"
        )
        _inst_display["change_shares_fmt"] = _inst_display["change_shares"].apply(
            lambda v: f"{v:+,.0f}" if pd.notna(v) else "N/A"
        )
        st.dataframe(
            _inst_display[["fund_name", "date_reported", "curr_shares", "change_shares_fmt",
                           "change_pct_fmt", "curr_value_fmt", "signal"]].rename(columns={
                "fund_name": "Fund", "date_reported": "Filed",
                "curr_shares": "Current Shares", "change_shares_fmt": "Change",
                "change_pct_fmt": "Change %", "curr_value_fmt": "Market Value",
                "signal": "Signal",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Limited institutional 13F data available for this ticker. Large funds may not hold this position.")

    # ── SECTION 3: ACTIVIST WATCH ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Section 3: Activist Watch (13D/13G)")

    if not _activist_df.empty:
        latest_activist = _activist_df.iloc[0]
        _a_color = "#e74c3c" if latest_activist["activist_intent"] else "#f39c12"
        _a_type = "ACTIVIST (13D — control/influence intent)" if latest_activist["activist_intent"] else "PASSIVE (13G — investment only)"
        st.markdown(
            f"<div style='background:{_a_color}33;border:2px solid {_a_color};border-radius:8px;padding:16px;margin-bottom:12px'>"
            f"<div style='font-size:18px;font-weight:bold;color:{_a_color}'>⚡ ACTIVIST POSITION DETECTED</div>"
            f"<div style='margin-top:8px;color:#ddd'>"
            f"<b>{latest_activist['filer_name']}</b> — {_a_type}<br>"
            f"Ownership: <b>{'%.1f%%' % latest_activist['ownership_pct'] if pd.notna(latest_activist['ownership_pct']) else 'See filing'}</b> | "
            f"Filed: {latest_activist['filing_date']}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            _activist_df[["filer_name", "form_type", "filing_date", "ownership_pct", "amendment", "activist_intent"]].rename(columns={
                "filer_name": "Filer", "form_type": "Form", "filing_date": "Filed",
                "ownership_pct": "% Owned", "amendment": "Amendment", "activist_intent": "Activist (13D)",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.success(f"No activist positions detected for {_ticker} (threshold: >5% ownership via 13D/13G filing).")

    # ── SECTION 4: XBRL FUNDAMENTALS VS YFINANCE ─────────────────────────────
    st.markdown("---")
    st.subheader("📑 Section 4: EDGAR XBRL Fundamentals")

    if _xbrl.get("source") == "unavailable":
        st.warning(f"XBRL data unavailable for {_ticker}.")
    else:
        _xc1, _xc2, _xc3 = st.columns(3)
        def _b(v, label, col):
            formatted = f"${v/1e9:.2f}B" if not np.isnan(v) and abs(v) >= 1e9 else f"${v/1e6:.0f}M" if not np.isnan(v) else "N/A"
            col.metric(label, formatted, help="Source: EDGAR XBRL 10-K")

        _b(_xbrl.get("latest_revenue", np.nan), "Revenue (XBRL)", _xc1)
        _b(_xbrl.get("latest_net_income", np.nan), "Net Income (XBRL)", _xc2)
        _b(_xbrl.get("latest_fcf", np.nan), "FCF (XBRL)", _xc3)

        _xc4, _xc5, _xc6 = st.columns(3)
        _b(_xbrl.get("latest_total_assets", np.nan), "Total Assets", _xc4)
        _b(_xbrl.get("net_debt", np.nan), "Net Debt", _xc5)
        _ic = _xbrl.get("interest_coverage", np.nan)
        _xc6.metric("Interest Coverage", f"{_ic:.1f}×" if not np.isnan(_ic) else "N/A")

        # FCF time series from XBRL
        _fcf_s = _xbrl.get("fcf_series", pd.Series(dtype=float))
        if not _fcf_s.empty:
            st.subheader("FCF Trend (from EDGAR XBRL 10-K filings)")
            _fig_xfcf = go.Figure(go.Bar(
                x=[str(d)[:4] for d in _fcf_s.index],
                y=(_fcf_s / 1e9).round(3),
                marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in _fcf_s],
                name="FCF ($B)",
                text=[f"${v/1e9:.2f}B" for v in _fcf_s],
                textposition="outside",
            ))
            _fig_xfcf.update_layout(
                title=f"{_ticker} Free Cash Flow (EDGAR XBRL)",
                yaxis_title="FCF ($B)", template=DARK, height=300,
            )
            st.plotly_chart(_fig_xfcf, use_container_width=True)

        # Accruals ratio
        _acc = _xbrl.get("accruals_ratio", np.nan)
        if not np.isnan(_acc):
            _acc_c = "#e74c3c" if abs(_acc) > 0.05 else "#2ecc71"
            st.markdown(
                f"<div style='background:{_acc_c}22;border-left:3px solid {_acc_c};padding:8px 14px;border-radius:4px'>"
                f"<b style='color:{_acc_c}'>Accruals Ratio (EDGAR): {_acc:.2%}</b> — "
                f"{'⚠️ High accruals (>5%): earnings quality concern' if abs(_acc) > 0.05 else '✓ Normal earnings quality'}"
                f"</div>", unsafe_allow_html=True)

        st.caption("Source: EDGAR XBRL (10-K filings). All values in USD.")


# ══════════════════════════════════════════════════════════════════════════════
# SMART MONEY & FACTORS — FACTOR & SHOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with _sm_s2:
    st.subheader(f"Factor & Shock Analysis — {_ticker}")

    _factor_key = f"factor_{_ticker}"
    _shock_key = f"shock_{_ticker}"

    if _factor_key not in st.session_state:
        with st.spinner("Running Fama-French factor decomposition..."):
            try:
                _fd = factor_decomposition(prices, start_date, end_date)
            except Exception as _fe:
                _fd = {"available": False, "error": str(_fe)}
        st.session_state[_factor_key] = _fd
    if _shock_key not in st.session_state:
        with st.spinner("Analyzing sector-specific price shocks..."):
            try:
                _sd = analyze_price_shock(_ticker, info.get("sector", "Unknown"), prices, start_date, end_date)
            except Exception as _se:
                _sd = {"available": False, "error": str(_se)}
        st.session_state[_shock_key] = _sd

    _fd = st.session_state[_factor_key]
    _sd = st.session_state[_shock_key]

    # ── SECTION 1: FAMA-FRENCH DECOMPOSITION ─────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Section 1: Fama-French 5-Factor + Momentum Decomposition")

    if not _fd.get("available", False):
        st.warning(f"Factor decomposition unavailable: {_fd.get('error', 'Unknown error')}. "
                   "Requires pandas-datareader and internet access to Ken French library.")
    else:
        # Metric row
        _fc1, _fc2, _fc3, _fc4 = st.columns(4)
        _fc1.metric("Systematic Risk", f"{_fd['systematic_risk_pct']:.1f}%",
                    help="% of return variance explained by FF factors")
        _fc2.metric("Idiosyncratic Risk", f"{_fd['idiosyncratic_risk_pct']:.1f}%",
                    help="% unexplained = pure company-specific risk")
        _fc3.metric("Alpha (ann.)", f"{_fd['alpha']:.1%}",
                    help=f"t-stat: {_fd['alpha_tstat']:.1f} | p: {_fd['alpha_pvalue']:.3f}")
        _fc4.metric("Model Quality", _fd["factor_model_quality"],
                    help=f"R²={_fd['r_squared']:.3f} | {_fd['n_months']} months")

        # Stacked bar: variance decomposition
        _fc_data = _fd.get("factor_contributions", {})
        if _fc_data:
            _idio_share = _fd["idiosyncratic_risk_pct"]
            _sys_share = _fd["systematic_risk_pct"]
            _bar_labels = list(_fc_data.keys()) + ["Idiosyncratic"]
            # Factor contributions are % of systematic, scale to % of total
            _bar_vals = [v * _sys_share / 100 for v in _fc_data.values()] + [_idio_share]
            _bar_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#95a5a6"]
            _fig_decomp = go.Figure(go.Bar(
                x=_bar_labels, y=_bar_vals,
                marker_color=_bar_colors[:len(_bar_labels)],
                text=[f"{v:.1f}%" for v in _bar_vals],
                textposition="outside",
            ))
            _fig_decomp.update_layout(
                title=f"{_ticker} — Variance Decomposition (% of Total Risk)",
                yaxis_title="% of Total Variance",
                template=DARK, height=350,
                yaxis=dict(range=[0, max(_bar_vals) * 1.3]),
            )
            st.plotly_chart(_fig_decomp, use_container_width=True)

        # Factor loadings table
        _factor_rows = [
            ("Market (Mkt-RF)", _fd.get("beta_market"), _fd.get("tstat_market")),
            ("Size (SMB)", _fd.get("beta_size"), _fd.get("tstat_size")),
            ("Value (HML)", _fd.get("beta_value"), _fd.get("tstat_value")),
            ("Profitability (RMW)", _fd.get("beta_profitability"), _fd.get("tstat_profitability")),
            ("Investment (CMA)", _fd.get("beta_investment"), _fd.get("tstat_investment")),
            ("Momentum (Mom)", _fd.get("beta_momentum"), _fd.get("tstat_momentum")),
        ]
        _interpretations = {
            "Market (Mkt-RF)": lambda b: "High mkt sensitivity" if b > 1.5 else "Low beta defensive" if b < 0.5 else "Average market exposure",
            "Size (SMB)": lambda b: "Small-cap tilt" if b > 0.3 else "Large-cap tilt" if b < -0.3 else "Size-neutral",
            "Value (HML)": lambda b: "Value stock" if b > 0.3 else "Growth/momentum" if b < -0.3 else "Blend",
            "Profitability (RMW)": lambda b: "High profitability" if b > 0.2 else "Low profitability" if b < -0.2 else "Average profitability",
            "Investment (CMA)": lambda b: "Conservative investment" if b > 0.2 else "Aggressive investment" if b < -0.2 else "Average investment",
            "Momentum (Mom)": lambda b: "Strong momentum" if b > 0.3 else "Contrarian/reversal" if b < -0.3 else "Neutral momentum",
        }
        _tbl_data = []
        for fname, beta, tstat in _factor_rows:
            if beta is None:
                continue
            sig = "YES ✓" if tstat is not None and abs(tstat) > 2.0 else "no"
            interp = _interpretations.get(fname, lambda b: "")(beta)
            _tbl_data.append({
                "Factor": fname,
                "Beta": f"{beta:.3f}",
                "T-stat": f"{tstat:.1f}" if tstat is not None else "N/A",
                "Significant?": sig,
                "Interpretation": interp,
            })
        if _tbl_data:
            st.dataframe(pd.DataFrame(_tbl_data), use_container_width=True, hide_index=True)

        # Alpha box
        _at = _fd.get("alpha_tstat", 0)
        _aa = _fd.get("alpha", 0)
        if abs(_at) > 2.0:
            _ac = "#2ecc71" if _aa > 0 else "#e74c3c"
            st.markdown(
                f"<div style='background:{_ac}22;border-left:4px solid {_ac};padding:10px 16px;border-radius:4px;margin:8px 0'>"
                f"<b style='color:{_ac}'>{'✅' if _aa > 0 else '❌'} Significant Alpha: {_aa:.1%} annualized (t={_at:.1f})</b><br>"
                f"<span style='color:#ddd;font-size:13px'>"
                f"{'This stock has historically generated returns BEYOND what factor exposures predict.' if _aa > 0 else 'This stock has historically UNDERPERFORMED relative to its factor exposures.'}"
                f"</span></div>", unsafe_allow_html=True)
        else:
            st.info(f"Factor model explains returns well (R²={_fd['r_squared']:.3f}). No statistically significant alpha (t={_at:.1f}).")

        # Interpretation flags
        _flags = _fd.get("interpretation_flags", [])
        if _flags:
            for _fl in _flags:
                st.warning(_fl)

        # Rolling factor exposure chart
        _rb = _fd.get("rolling_beta_market")
        _rr2 = _fd.get("rolling_r2")
        if _rb is not None and not _rb.empty and _rr2 is not None and not _rr2.empty:
            _fig_roll = go.Figure()
            _fig_roll.add_trace(go.Scatter(
                x=_rb.index, y=_rb, name="Rolling β Market",
                line=dict(color="#3498db", width=2),
            ))
            _fig_roll.add_hline(y=1.0, line_dash="dot", line_color="gray",
                                annotation_text="β=1")
            _fig_roll.update_layout(
                title=f"{_ticker} — Rolling Market Beta (63-month window)",
                yaxis_title="Beta", template=DARK, height=280,
            )
            st.plotly_chart(_fig_roll, use_container_width=True)

        # Idiosyncratic risk panel
        st.markdown("**Pure Idiosyncratic Risk** (after removing all factor exposures)")
        _ic1, _ic2, _ic3 = st.columns(3)
        _ic1.metric("Idio Vol (ann.)", f"{_fd['idio_vol']:.1%}")
        _ic2.metric("Idio VaR 95% (monthly)", f"{_fd['idio_var_95']:.2%}")
        _ic3.metric("Idio CVaR 99% (monthly)", f"{_fd['idio_cvar_99']:.2%}" if _fd.get("idio_cvar_99") else "N/A")

    # ── SECTION 2: SECTOR-SPECIFIC SHOCK ANALYSIS ─────────────────────────────
    st.markdown("---")
    _sector_val = info.get("sector", "Unknown")
    _industry_val = info.get("industry", "Unknown")
    st.subheader(f"💥 Section 2: Price Shock Sensitivity — {_sector_val}")

    if not _sd.get("available", False):
        st.warning(f"Shock analysis unavailable: {_sd.get('error', 'Need 18+ months of data')}")
    else:
        # Current shock risk indicators
        _csr = _sd.get("current_shock_risk", {})
        if _csr:
            _risk_cols = st.columns(min(len(_csr), 4))
            _risk_colors = {"ELEVATED": "#e74c3c", "NORMAL": "#f39c12", "SUBDUED": "#2ecc71"}
            for i, (factor, level) in enumerate(_csr.items()):
                with _risk_cols[i % 4]:
                    _rc = _risk_colors.get(level, "#95a5a6")
                    st.markdown(
                        f"<div style='background:{_rc}22;border:1px solid {_rc};border-radius:6px;"
                        f"padding:8px 12px;text-align:center;margin-bottom:8px'>"
                        f"<b style='color:{_rc}'>{level}</b><br>"
                        f"<small style='color:#ccc'>{factor}</small>"
                        f"</div>", unsafe_allow_html=True)

        # Historical shock response tables per factor
        _hr = _sd.get("historical_responses", {})
        for factor_name, factor_data in _hr.items():
            with st.expander(f"📈 {factor_name} — Historical Shock Responses", expanded=False):
                _note = factor_data.get("note", "")
                _trans = factor_data.get("transmission", "")
                if _note:
                    st.caption(f"Transmission: **{_trans}** | {_note}")

                _shock_dict = factor_data.get("shocks", {})
                if not _shock_dict:
                    st.info("No shock data available.")
                    continue

                # Build heatmap data
                _shock_labels = list(_shock_dict.keys())
                _same_m = []
                _next_m = []
                _three_m = []
                _conf = []
                _n_eps = []

                for slabel in _shock_labels:
                    sd_entry = _shock_dict[slabel]
                    _n_eps.append(sd_entry.get("episodes_found", 0))
                    _conf.append(sd_entry.get("confidence", "INSUFFICIENT"))
                    same = sd_entry.get("same_month", {})
                    nxt = sd_entry.get("next_month", {})
                    thr = sd_entry.get("three_month", {})
                    _same_m.append(same.get("median", np.nan))
                    _next_m.append(nxt.get("median", np.nan))
                    _three_m.append(thr.get("median", np.nan))

                _hm_df = pd.DataFrame({
                    "Shock": _shock_labels,
                    "Episodes": _n_eps,
                    "Confidence": _conf,
                    "Same Month (median)": [f"{v:.1%}" if not np.isnan(v) else "N/A" for v in _same_m],
                    "Next Month (median)": [f"{v:.1%}" if not np.isnan(v) else "N/A" for v in _next_m],
                    "3-Month (median)": [f"{v:.1%}" if not np.isnan(v) else "N/A" for v in _three_m],
                })
                st.dataframe(_hm_df, use_container_width=True, hide_index=True)

                # Worst case panel
                _worst_3m = min(
                    (v for v in _three_m if not np.isnan(v)), default=np.nan
                )
                if not np.isnan(_worst_3m) and _worst_3m < -0.05:
                    st.markdown(
                        f"<div style='background:#e74c3c22;border-left:3px solid #e74c3c;"
                        f"padding:8px 14px;border-radius:4px;margin-top:8px'>"
                        f"<b style='color:#e74c3c'>Worst median 3-month response: {_worst_3m:.1%}</b> "
                        f"across all shock magnitudes for this factor."
                        f"</div>", unsafe_allow_html=True)

        # Compound shock scenario
        _comp = _sd.get("compound_scenario")
        if _comp:
            st.markdown("---")
            st.subheader("🌪️ Compound Shock Scenario")
            st.markdown(
                f"<div style='background:#8e44ad22;border-left:4px solid #8e44ad;"
                f"padding:12px 18px;border-radius:6px'>"
                f"<b style='color:#8e44ad;font-size:16px'>{_comp['name']}</b><br>"
                f"<span style='color:#ccc'>{_comp['description']}</span>"
                f"</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI ANALYST
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"🤖 AI Research Analyst — {_ticker}")

    _ai_key = f"ai_report_{_ticker}"
    _ctx_key = f"ai_ctx_{_ticker}"

    # Check if API keys are present
    import os as _os
    _has_openrouter = bool(_os.getenv("OPENROUTER_API_KEY"))
    _has_tavily = bool(_os.getenv("TAVILY_API_KEY"))

    if not _has_openrouter:
        st.error(
            "**OPENROUTER_API_KEY not found.** Add it to your `.env` file:\n"
            "```\nOPENROUTER_API_KEY=sk-or-...\nTAVILY_API_KEY=tvly-...\n```"
        )
        st.stop()

    if not _has_tavily:
        st.warning("**TAVILY_API_KEY not set** — web research disabled. Report will use only computed metrics.")

    # ── Deferred generation — show button until user triggers ─────────────────
    _dp_ai = st.session_state.get(f"driver_{_ticker}")
    _top_driver_name = (_dp_ai.get("primary_drivers", [{}])[0].get("name", "primary sector driver")
                        if _dp_ai and _dp_ai.get("primary_drivers") else "primary sector driver")

    if _ai_key not in st.session_state:
        _gen_col1, _gen_col2 = st.columns([3, 1])
        with _gen_col1:
            st.write(
                f"Generate **{_horizon}** risk analysis for **{_ticker}** using DeepSeek + live web research.  \n"
                f"Will lead with: **{_top_driver_name}**"
            )
        with _gen_col2:
            _generate_clicked = st.button("Generate Analysis →", type="primary", key="gen_ai")
        if not _generate_clicked:
            st.info(
                f"AI analysis will:\n"
                f"- Lead with {_ticker}'s highest-relevance risk driver\n"
                f"- Search live news and analyst reports\n"
                f"- Generate sector-specific risk narrative\n"
                f"- Focus on **{_horizon}** horizon\n\n"
                f"Takes ~15–20 seconds to generate."
            )
            st.stop()

        _smart_sm = st.session_state.get(f"smart_money_{_ticker}")
        _insider_sig_ai = _smart_sm[2] if _smart_sm else None
        _inst_sig_ai = _smart_sm[4] if _smart_sm else None
        _activist_ai = _smart_sm[5] if _smart_sm else None
        _factor_ai = st.session_state.get(f"factor_{_ticker}")

        _s1 = st.empty()
        _s1.write("🤖 AI analyst reading metrics...")
        try:
            _ai_ctx = build_ticker_context(
                _ticker, prices, volume, bench_prices, info, macro_df,
                fund_scores=st.session_state.get("fund_scores"),
                insider_signals=_insider_sig_ai,
                inst_signals=_inst_sig_ai,
                activist_df=_activist_ai,
                factor_results=_factor_ai if _factor_ai and _factor_ai.get("available") else None,
                driver_profile=_dp_ai,
                horizon=_horizon,
            )
            st.session_state[_ctx_key] = _ai_ctx

            _s1.write("🔍 Searching recent news and analyst reports...")
            _web_res = fetch_web_research(
                _ticker, info.get("name", _ticker), info.get("sector", ""),
                driver_profile=_dp_ai, horizon=_horizon,
            )

            _s1.write("✍️ Writing research report...")
            _report = generate_research_report(_ai_ctx, _web_res, driver_profile=_dp_ai)
            st.session_state[_ai_key] = (_report, _web_res, len(_web_res))
            _s1.empty()
        except Exception as _ae:
            _s1.empty()
            st.error(f"Report generation failed: {_ae}")
            st.session_state[_ai_key] = (f"_Error: {_ae}_", [], 0)

    _report_text, _web_res_cached, _n_web = st.session_state[_ai_key]
    _ai_ctx = st.session_state.get(_ctx_key, {})
    _driver_profile_ai = st.session_state.get(f"driver_{_ticker}")

    # ── Display report as expandable sections ─────────────────────────────────
    import re as _re
    _sections = _re.split(r"\n(?=##\s)", _report_text)
    _conf_colors = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}

    for _i, _sec in enumerate(_sections):
        if not _sec.strip():
            continue
        _lines = _sec.strip().split("\n")
        _header = _lines[0].lstrip("#").strip()
        _body = "\n".join(_lines[1:]).strip()

        # Detect confidence badge at end of section
        _conf_match = _re.search(r"\bConfidence[:\s]+(HIGH|MEDIUM|LOW)\b", _body, _re.IGNORECASE)
        _conf_label = _conf_match.group(1).upper() if _conf_match else None

        _is_exec = "EXECUTIVE" in _header.upper()
        with st.expander(_header, expanded=_is_exec):
            if _conf_label:
                _bc = _conf_colors.get(_conf_label, "gray")
                st.markdown(
                    f"<span style='background:{'#2ecc71' if _bc=='green' else '#e67e22' if _bc=='orange' else '#e74c3c'}33;"
                    f"color:{'#2ecc71' if _bc=='green' else '#f39c12' if _bc=='orange' else '#e74c3c'};"
                    f"padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold'>"
                    f"Confidence: {_conf_label}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown(_body)

    # Report footer
    from datetime import datetime as _dt
    st.caption(
        f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Model: DeepSeek (via OpenRouter) | "
        f"Web sources: {_n_web} | "
        f"⚠️ For research purposes only — not financial advice"
    )

    # Download button
    st.download_button(
        "📄 Download Report",
        data=_report_text,
        file_name=f"{_ticker}_risk_report_{_dt.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
    )

    # Regenerate button
    if st.button("🔄 Regenerate Report", key="regen_report"):
        for _k in (_ai_key, _ctx_key):
            if _k in st.session_state:
                del st.session_state[_k]
        st.rerun()

    # ── SECTION 2: RESEARCH CHAT ──────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask the Analyst")

    _chat_key = f"chat_{_ticker}"
    if _chat_key not in st.session_state:
        st.session_state[_chat_key] = []

    # Suggested questions
    st.write("**Suggested questions:**")
    _sq_cols = st.columns(3)
    _sector_str = _ai_ctx.get("sector", "")
    _suggested_qs = [
        "What's the biggest hidden risk here?",
        "Is valuation justified given the risk profile?",
        "How would a recession impact this specifically?",
        "What would change your view bullish or bearish?",
        "Compare to the 2020 COVID crash setup",
        f"Is the {_sector_str} sector at risk from current macro?",
    ]
    for _qi, _q in enumerate(_suggested_qs):
        if _sq_cols[_qi % 3].button(_q, key=f"sq_{_qi}_{_ticker}"):
            st.session_state["_pending_q"] = _q

    # Display chat history
    for _msg in st.session_state[_chat_key]:
        with st.chat_message(_msg["role"]):
            st.write(_msg["content"])

    # Handle pending suggested question or typed input
    _question = st.chat_input("Ask anything about this stock or sector...", key=f"chat_input_{_ticker}")
    if "_pending_q" in st.session_state:
        _question = st.session_state.pop("_pending_q")

    if _question and _ai_ctx:
        st.session_state[_chat_key].append({"role": "user", "content": _question})
        with st.chat_message("user"):
            st.write(_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    _answer = chat_with_analyst(
                        _question, _ai_ctx,
                        st.session_state[_chat_key][:-1],
                        driver_profile=_driver_profile_ai,
                    )
                except Exception as _ce:
                    _answer = f"_Chat error: {_ce}_"
            st.write(_answer)
            st.caption("⚠️ Research only — not financial advice")
        st.session_state[_chat_key].append({"role": "assistant", "content": _answer})

    if st.session_state[_chat_key]:
        if st.button("🗑️ Clear chat", key=f"clear_chat_{_ticker}"):
            st.session_state[_chat_key] = []

    # ── SECTION 3: SCENARIO ANALYSIS ─────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Scenario Analysis")
    st.write("Click a scenario to get AI-powered impact analysis specific to this company's balance sheet.")

    if _ai_ctx:
        _scenarios = get_scenarios_for_sector(_ai_ctx.get("sector", ""))
        _sc_cols = st.columns(min(len(_scenarios), 4))
        for _si, (_sname, _sparams) in enumerate(_scenarios.items()):
            with _sc_cols[_si % 4]:
                st.markdown(f"**{_sname}**")
                st.caption(", ".join(f"{k}={v}" for k, v in list(_sparams.items())[:2]))
                if st.button("Analyze", key=f"scenario_{_si}_{_ticker}"):
                    with st.spinner(f"Analyzing {_sname}..."):
                        try:
                            _sresult = evaluate_scenario(_sname, _sparams, _ai_ctx)
                        except Exception as _sce:
                            _sresult = {"error": str(_sce)}
                    # Display result
                    if "error" in _sresult:
                        st.error(_sresult["error"])
                    else:
                        _prob_c = "#e74c3c" if _sresult.get("probability") == "HIGH" else "#f39c12" if _sresult.get("probability") == "MEDIUM" else "#2ecc71"
                        st.markdown(
                            f"<div style='background:{_prob_c}22;border-left:3px solid {_prob_c};"
                            f"padding:10px 14px;border-radius:4px;margin-top:8px'>"
                            f"<b style='color:{_prob_c}'>Probability: {_sresult.get('probability','?')}</b><br>"
                            f"<b>Price impact:</b> {_sresult.get('estimated_price_impact_pct','?')}%<br>"
                            f"<b>Transmission:</b> {_sresult.get('key_transmission','')}<br>"
                            f"<b>Company-specific:</b> {_sresult.get('ticker_specific','')}<br>"
                            f"<b>Precedent:</b> {_sresult.get('historical_precedent','')}<br>"
                            f"<b>Hedging:</b> {_sresult.get('hedging_implication','')}"
                            f"</div>", unsafe_allow_html=True)
