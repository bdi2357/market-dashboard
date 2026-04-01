"""
Market Risk Assessment Dashboard
=================================
Multi-tab professional risk engine for research & backtesting.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta

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
    with st.spinner("Fetching data..."):
        df = fetch_ohlcv(ticker, start_date, end_date)
        bench_df = fetch_ohlcv(benchmark, start_date, end_date)
        info = fetch_ticker_info(ticker)
        sector_etf = get_sector_etf(ticker)
        sector_df = fetch_ohlcv(sector_etf, start_date, end_date)
        macro_df = fetch_macro(start_date, end_date)

        st.session_state.df = df
        st.session_state.bench_df = bench_df
        st.session_state.sector_df = sector_df
        st.session_state.macro_df = macro_df
        st.session_state.info = info
        st.session_state.sector_etf = sector_etf
        st.session_state.data_loaded = True
        st.session_state.ticker = ticker
        st.session_state.benchmark = benchmark

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Risk Scorecard",
    "📈 Volatility Surface",
    "⚠️ Tail Risk",
    "🔗 Relative Risk",
    "🌍 Macro Regime",
    "🧪 Backtester",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RISK SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Risk Scorecard — {_ticker}")
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

    # Component breakdown
    comp_df = pd.DataFrame(
        [(k, v / 0.01 if k == list(components.keys())[0] else v, v) for k, v in components.items()],
        columns=["Factor", "_raw", "Weighted Score"],
    )
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

    # Percentile bands for 21d vol
    v21 = vol_df["vol_21d"].dropna()
    if len(v21) > 40:
        p25 = v21.rolling(252, min_periods=60).quantile(0.25)
        p75 = v21.rolling(252, min_periods=60).quantile(0.75)
        p95 = v21.rolling(252, min_periods=60).quantile(0.95)
        fig_vol.add_trace(go.Scatter(x=v21.index, y=p95, name="95th pct (21d)", line=dict(color="white", dash="dot", width=1)))
        fig_vol.add_trace(go.Scatter(x=v21.index, y=p75, name="75th pct (21d)", line=dict(color="gray", dash="dot", width=1)))
        fig_vol.add_trace(go.Scatter(x=v21.index, y=p25, name="25th pct (21d)", line=dict(color="gray", dash="dot", width=1), fill="tonexty", fillcolor="rgba(100,100,100,0.1)"))

    fig_vol.update_layout(title="Realized Volatility Cone", yaxis_title="Annualized Vol", template=DARK, height=400)
    st.plotly_chart(fig_vol, use_container_width=True)

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
        current_regime = macro_regimes.iloc[-1] if len(macro_regimes) > 0 else "UNKNOWN"
        regime_color = REGIME_COLORS.get(current_regime, "#7f8c8d")

        # Traffic light
        col_tl, col_desc = st.columns([1, 3])
        with col_tl:
            st.markdown(
                f"<div style='background:{regime_color};padding:20px;border-radius:10px;"
                f"text-align:center;font-size:24px;font-weight:bold;color:white'>"
                f"{current_regime}</div>",
                unsafe_allow_html=True,
            )
        with col_desc:
            descriptions = {
                "RISK_ON": "Rates falling, spreads tightening — risk assets typically favored.",
                "RISK_OFF": "Rates rising, spreads widening — defensive positioning warranted.",
                "STAGFLATION_PROXY": "Rates rising while credit spreads tighten — unusual stress.",
                "MIXED": "No clear directional signal from rates or credit spreads.",
                "UNKNOWN": "Insufficient data to classify regime.",
            }
            st.info(descriptions.get(current_regime, ""))

        # Yield curve + credit spread chart with regime bands
        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["10Y Treasury Yield", "HY Credit Spread", "Yield Curve (10Y-2Y)"])

        for series_id, row in [("DGS10", 1), ("BAMLH0A0HYM2", 2), ("T10Y2Y", 3)]:
            if series_id in macro_df.columns:
                fig_macro.add_trace(
                    go.Scatter(x=macro_df.index, y=macro_df[series_id], name=series_id, line=dict(width=1.5)),
                    row=row, col=1,
                )

        # Add regime background bands
        for i in range(len(macro_regimes) - 1):
            t0 = macro_regimes.index[i]
            t1 = macro_regimes.index[i + 1]
            r = macro_regimes.iloc[i]
            fig_macro.add_vrect(
                x0=t0, x1=t1,
                fillcolor=REGIME_COLORS.get(r, "gray"),
                opacity=0.08, line_width=0,
            )

        fig_macro.update_layout(template=DARK, height=600, title="Macro Indicators with Regime Bands")
        st.plotly_chart(fig_macro, use_container_width=True)

        # VIX regime
        if "VIXCLS" in macro_df.columns:
            vix_s = macro_df["VIXCLS"].dropna()
            vix_regimes = vix_regime(vix_s)
            st.subheader("VIX Regime Analysis")

            # Align stock returns to VIX dates
            vix_df_merged = pd.DataFrame({
                "vix": vix_s,
                "vix_regime": vix_regimes,
            })
            stock_r = simple_r.rename("stock_ret")
            merged = vix_df_merged.join(stock_r, how="inner").dropna()

            cond_table = conditional_returns_by_regime(
                prices.reindex(merged.index),
                merged["vix_regime"].astype(str),
            )

            st.dataframe(
                cond_table.style.format({
                    "mean_daily_ret": "{:.4f}",
                    "ann_return": "{:.2%}",
                    "ann_vol": "{:.2%}",
                    "sharpe": "{:.2f}",
                }),
                use_container_width=True,
            )

        # Conditional returns by macro regime
        st.subheader(f"Conditional Performance of {_ticker} by Macro Regime")
        aligned_macro = macro_regimes.reindex(prices.index, method="ffill").dropna()
        cond_macro = conditional_returns_by_regime(prices.reindex(aligned_macro.index), aligned_macro)
        if not cond_macro.empty:
            fig_cond = go.Figure()
            colors_regime = [REGIME_COLORS.get(r, "#95a5a6") for r in cond_macro.index]
            fig_cond.add_trace(go.Bar(
                x=cond_macro.index.astype(str),
                y=cond_macro["ann_return"],
                name="Ann. Return",
                marker_color=colors_regime,
                text=[f"{v:.1%}" for v in cond_macro["ann_return"]],
                textposition="outside",
            ))
            fig_cond.update_layout(title=f"{_ticker} Annualized Return by Macro Regime", yaxis_title="Ann. Return", yaxis_tickformat=".0%", template=DARK, height=300)
            st.plotly_chart(fig_cond, use_container_width=True)
            st.dataframe(cond_macro.style.format({"mean_daily_ret": "{:.4f}", "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"MA Crossover Backtest — {_ticker}")
    st.caption(f"Strategy: Long when {fast_ma}d MA > {slow_ma}d MA | Sizing: {sizing_method}")

    with st.spinner("Running backtest..."):
        bt = run_backtest(prices, fast_ma=int(fast_ma), slow_ma=int(slow_ma), sizing=sizing_method)
        ci = bootstrap_sharpe_ci(bt["returns"], n_boot=1000)
        stress = stress_test(prices, fast_ma=int(fast_ma), slow_ma=int(slow_ma))

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
        st.dataframe(
            stress.style.format({
                "total_return": "{:.2%}",
                "ann_vol": "{:.2%}",
                "max_drawdown": "{:.2%}",
                "sharpe": "{:.2f}",
            }).background_gradient(subset=["total_return", "max_drawdown"], cmap="RdYlGn"),
            use_container_width=True,
        )
    else:
        st.info("No stress test data available for the selected period.")
