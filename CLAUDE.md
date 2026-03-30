# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-page Streamlit dashboard that displays interactive stock price charts and basic metrics using yfinance for data and Plotly for visualization.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Architecture

All application logic lives in `app.py` — there is intentionally no multi-file structure. The data flow is:

1. User selects a ticker, period, and chart type via Streamlit sidebar widgets.
2. `load_data()` fetches OHLCV data from Yahoo Finance via yfinance and caches it for 5 minutes (`@st.cache_data(ttl=300)`).
3. Plotly `go.Candlestick` or `go.Scatter` traces are built from the DataFrame and rendered with `st.plotly_chart`.
4. A second Plotly bar chart renders volume below the price chart.

## Key Notes

- Data is cached per `(symbol, start, end)` tuple for 5 minutes to avoid redundant API calls during interaction.
- `yf.Ticker.fast_info` is used for lightweight metadata (no full `info` dict download).
- `.item()` is called on single-element pandas results to extract scalars safely (avoids deprecation warnings in recent pandas/yfinance versions).
