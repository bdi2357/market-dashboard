# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A professional multi-tab Streamlit risk assessment dashboard for equity research and backtesting. Combines yfinance equity data with FRED macro data to compute a comprehensive suite of risk metrics across realized risk, tail risk, relative risk, macro regime, and liquidity dimensions.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Set FRED API key (optional — falls back to pandas-datareader without it)
export FRED_API_KEY=your_key_here
```

## Architecture

```
market-dashboard/
├── app.py                  # Streamlit UI — 6 tabs
├── data/
│   ├── fetcher.py          # Data fetching + parquet caching
│   └── cache/              # Auto-created, gitignored
├── risk/
│   └── metrics.py          # All risk metric implementations
└── backtest/
    └── engine.py           # MA crossover backtester
```

### Data Layer (`data/fetcher.py`)
- `fetch_ohlcv(ticker, start, end)` — yfinance `Ticker.history(auto_adjust=True)` with tz stripped to naive DatetimeIndex. Cached 4h.
- `fetch_macro(start, end)` — FRED series via `fredapi` (preferred) or `pandas-datareader` fallback. Cached 24h. Requires `FRED_API_KEY` env var for fredapi.
- `fetch_peers(ticker, start, end)` — top-N sector peers by market cap from S&P 500 Wikipedia list.
- Cache key is MD5 hash of `(symbol, start, end)` → `data/cache/<hash>.parquet`.

### Risk Engine (`risk/metrics.py`)

#### Realized Risk
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| Realized Vol | `std(log_returns) × √252` | >40% annualized = high; >80% = extreme |
| Vol Regime | Rolling percentile rank vs 2Y history | "extreme" (>95th pct) = elevated tail risk |
| Max Drawdown | `price/rolling_max - 1` minimum | >30% = significant; >50% = severe |
| Calmar Ratio | `ann_return / |max_drawdown|` | <0.5 = poor; >1.0 = acceptable |
| Sortino Ratio | `ann_excess_return / downside_std` | <1.0 = risk-adjusted underperformance |
| Ulcer Index | `√(mean(pct_drawdown²))` over window | Higher = more prolonged drawdown stress |

#### Tail Risk
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| Historical VaR α | `-quantile(returns, 1-α)` | VaR 99% > 3% daily = high tail risk |
| CVaR (ES) α | `-mean(returns | returns < -VaR)` | CVaR/VaR ratio > 1.5 = fat tails |
| Parametric VaR (Normal) | `-(μ + z_α × σ)` | Underestimates if returns non-normal |
| Parametric VaR (Student-t) | MLE-fitted df; `-(μ + t_α × σ)` | Fitted df < 5 = extreme fat tails |
| Cornish-Fisher VaR | `z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36` | Use when |skew|>1 or kurt>3 |
| Skewness | `E[(r-μ)³]/σ³` | |skew| > 1 = significant asymmetry |
| Excess Kurtosis | `E[(r-μ)⁴]/σ⁴ - 3` | >3 = fat tails beyond normal |

#### Relative Risk
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| Rolling Beta | `Cov(r_s, r_b) / Var(r_b)` over 63d | β>2 = high market sensitivity; β<0 = inverse |
| Idiosyncratic Vol | `√(σ²_total - β² × σ²_bench)` | High idio vol = stock-specific event risk |
| Information Ratio | `ann_active_return / tracking_error` | <0 = underperforming benchmark; >0.5 = good |
| Rolling Correlation | Pearson correlation of returns over 63d | >0.9 = near-pure systematic exposure |

#### Macro Regime Classification
Regime is determined by 63-day rolling change in DGS10 (10Y yield) and BAMLH0A0HYM2 (HY spread):
- **RISK_ON**: rates Δ < -0.2 AND spread Δ < -0.2
- **RISK_OFF**: rates Δ > +0.2 AND spread Δ > +0.2
- **STAGFLATION_PROXY**: rates Δ > +0.2 AND spread Δ < -0.1
- **MIXED**: all other combinations

VIX regime thresholds: low <15, medium 15-25, high 25-35, extreme >35.

#### Liquidity Risk
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| Amihud Illiquidity | `mean(|R_t| / dollar_volume_t)` × 1e6 | Rising trend = drying liquidity |
| HL Spread Proxy | `mean((High-Low)/Close)` over 21d | Spike = widening effective spread |
| Volume Trend | `volume / rolling_20d_avg` | <0.5 = dry-up; >2.0 = surge |

### Composite Risk Score (`risk/metrics.py:composite_risk_score`)
Single 0-100 score from weighted components:
- Realized Vol (21d): 25%
- Max Drawdown: 20%
- CVaR 99%: 20%
- Tail Shape (|skew| + kurt/5): 15%
- Beta: 10%
- Amihud Illiquidity: 10%

Each component is linearly mapped to 0-100 between hardcoded lo/hi bounds representing low and extreme risk levels.

### Backtesting Engine (`backtest/engine.py`)
- Strategy: long when fast MA > slow MA (no shorting)
- Position sizing: `vol_target` (default) scales size so portfolio vol = 15% annualized; `kelly` uses rolling 63d Kelly fraction; `fixed` = full position
- `stress_test()` reruns the strategy over GFC 2008-09, COVID 2020, Rate Shock 2022 subperiods
- `bootstrap_sharpe_ci()` resamples daily returns 1000× for Sharpe confidence intervals

## Caching
All parquet cache files live in `data/cache/` (gitignored). Cache TTL: 4h for price data, 24h for macro/info. Delete `data/cache/` to force fresh fetches.

## Environment Variables
- `FRED_API_KEY` — optional; without it, macro data falls back to `pandas-datareader` FRED reader (rate-limited but functional)

### Fundamental Risk Engine (`risk/fundamental.py`)

#### Valuation Risk
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| Trailing P/E | Price / TTM EPS | >80th sector percentile = expensive |
| Forward P/E | Price / NTM EPS estimate | >80th pct + high = priced for perfection |
| Price/Book | Market Cap / Book Value | >80th pct = stretched for financials/industrials |
| EV/EBITDA | Enterprise Value / EBITDA | >15× = high; >20× = very high |
| PEG Ratio | P/E / EPS Growth Rate | >2 = growth fully priced in |
| Equity Risk Premium | Earnings Yield - 10Y Treasury yield | <1% = bonds competitive vs equities |

Flag `VALUATION_STRETCHED`: above 80th sector percentile on 3+ of the above metrics.

#### Balance Sheet Risk — Altman Z-Score
Z = 1.2×(Working Capital/Assets) + 1.4×(Retained Earnings/Assets) + 3.3×(EBIT/Assets) + 0.6×(Mkt Cap/Total Liab) + 1.0×(Revenue/Assets)
- Z > 2.99: Safe zone
- 1.81 ≤ Z < 2.99: Grey zone — monitor
- Z < 1.81: Distress zone — flag `ALTMAN_DISTRESS`

Other thresholds: Interest Coverage < 2× → `INTEREST_COVERAGE_LOW`; Total Debt YoY > 20% → `DEBT_SURGE`

#### Earnings Quality
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| Accruals Ratio | (Net Income - CFO) / Total Assets | >5% = low quality; earnings exceed cash |
| EPS Surprise Trend | Actual vs consensus estimate | 2+ consecutive misses = `CONSECUTIVE_MISSES` |
| Revenue Growth Std | Std dev of quarterly revenue growth | High std dev = lumpy, unreliable revenue |

#### Cash Flow Risk
| Metric | Formula | Concern Threshold |
|--------|---------|-------------------|
| FCF | Operating CF - CapEx | Negative for 2+ years = `FCF_NEGATIVE_STREAK` |
| FCF Yield | FCF / Market Cap | <0% while P/E >20 = `FCF_YIELD_NEGATIVE_PREMIUM` |
| FCF Volatility | std(annual FCF) / mean(FCF) | >1.0 = highly unpredictable cash generation |
| Shareholder Yield | FCF yield + dividend yield + buyback yield | Higher = more capital returned to shareholders |

#### Growth/Value + Regime Classification
Stock is classified into: Deep Value / Value / Blend / Growth / Aggressive Growth
based on sector P/E percentile and revenue growth rate.

Regime mismatch flags (`REGIME_MISMATCH`):
- Aggressive Growth in RISK_OFF or STAGFLATION_PROXY regime
- Deep Value in RISK_ON regime (mild — opportunity signal)

#### Divergence Score (Market Risk vs Fundamental Risk)
divergence = |composite_risk_score - composite_fundamental_score|
- <20: Aligned
- 20-40: Mild divergence — monitor
- >40: Strong divergence — actionable signal:
  - Market risk high + Fundamental risk low → market overpricing risk
  - Market risk low + Fundamental risk high → market underpricing risk (most dangerous)

#### News Materiality Scoring
Headlines scored 1-3; only 2-3 shown in the feed.
Score 3: Earnings results, M&A, CEO change, regulatory action, credit rating change, FCF/dividend change
Score 2: Analyst upgrades/downgrades, price target changes, macro events with sector impact
Score 1 (filtered): Generic price commentary, listicles, unrelated news
