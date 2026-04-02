"""
Market Risk Assessment Dashboard
=================================
Multi-tab professional risk engine for research & backtesting.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Risk Dashboard", layout="wide", page_icon="📊")

DARK = "plotly_dark"
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
    fast_ma = st.number_input("Fast MA (backtest)", min_value=5, max_value=100, value=20)
    slow_ma = st.number_input("Slow MA (backtest)", min_value=10, max_value=300, value=50)
    sizing_method = st.selectbox("Position Sizing", ["vol_target", "kelly", "fixed"])
    load = st.button("Load Data", type="primary")

st.title("📊 Market Risk Dashboard")

# ── Data loading ──────────────────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

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
    # Clear stale cached computations when new data is loaded
    for _k in list(st.session_state.keys()):
        if _k.startswith("bt_") or _k.startswith("fund_"):
            del st.session_state[_k]

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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Risk Scorecard",
    "📈 Volatility Surface",
    "⚠️ Tail Risk",
    "🔗 Relative Risk",
    "🌍 Macro Regime",
    "🧪 Backtester",
    "📋 Fundamental Risk",
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
# TAB 2 — VOLATILITY SURFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
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
# TAB 3 — TAIL RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
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
# TAB 4 — RELATIVE RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
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
# TAB 5 — MACRO REGIME
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
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
# TAB 6 — BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
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
# TAB 7 — FUNDAMENTAL RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
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
