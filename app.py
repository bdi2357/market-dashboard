import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Market Dashboard", layout="wide")
st.title("Market Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
with col2:
    period_options = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    period_label = st.selectbox("Period", list(period_options.keys()), index=2)
with col3:
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])

end_date = date.today()
start_date = end_date - timedelta(days=period_options[period_label])

@st.cache_data(ttl=300)
def load_data(symbol, start, end):
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if not df.empty:
        df.index = df.index.tz_localize(None)
    return df

with st.spinner(f"Loading {ticker}..."):
    df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error(f"No data found for '{ticker}'. Check the ticker symbol.")
    st.stop()

# Summary metrics
last_price = df["Close"].iloc[-1].item()
prev_price = df["Close"].iloc[-2].item() if len(df) > 1 else last_price
change = last_price - prev_price
pct_change = (change / prev_price) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Price", f"${last_price:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
m2.metric("Period High", f"${df['High'].max().item():.2f}")
m3.metric("Period Low", f"${df['Low'].min().item():.2f}")
m4.metric("Avg Volume", f"{int(df['Volume'].mean()):,}")

# Chart
fig = go.Figure()
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=df["Close"].squeeze(),
        name=ticker,
    ))
else:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"].squeeze(),
        mode="lines",
        name=ticker,
        line=dict(width=2),
    ))

fig.update_layout(
    title=f"{ticker} — {period_label}",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    height=500,
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# Volume bar chart
vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"].squeeze(), name="Volume", marker_color="steelblue"))
vol_fig.update_layout(title="Volume", height=200, template="plotly_dark", margin=dict(t=30, b=20))
st.plotly_chart(vol_fig, use_container_width=True)
