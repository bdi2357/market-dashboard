"""
Fama-French 5-Factor + Momentum Decomposition
==============================================
Uses Ken French's free data library via pandas_datareader.
Decomposes stock returns into systematic factor exposures and
pure idiosyncratic (company-specific) risk.

Factors:
  Mkt-RF : Market excess return
  SMB    : Small minus Big (size)
  HML    : High minus Low (value/growth)
  RMW    : Robust minus Weak (profitability)
  CMA    : Conservative minus Aggressive (investment)
  Mom    : Momentum (separate dataset)
"""

import warnings
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
FACTOR_LABELS = {
    "Mkt-RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "Mom": "Momentum",
}


# ── data fetching ─────────────────────────────────────────────────────────────

def _ff_cache_path(name: str) -> Path:
    h = hashlib.md5(name.encode()).hexdigest()[:10]
    return CACHE_DIR / f"ff_{h}.parquet"


def _load_ff_cache(name: str, max_age_days: int = 7) -> Optional[pd.DataFrame]:
    p = _ff_cache_path(name)
    if not p.exists():
        return None
    age = (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")).total_seconds() / 86400
    if age > max_age_days:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def fetch_ff5_factors(start: date) -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 5-Factor + Momentum data from Ken French library.
    Returns monthly factor returns (decimals, not percent).
    Cached weekly.
    """
    cache_key = f"ff5mom_{start}"
    cached = _load_ff_cache(cache_key)
    if cached is not None:
        return cached

    try:
        import pandas_datareader.data as web

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff5_raw = web.DataReader(
                "F-F_Research_Data_5_Factors_2x3", "famafrench", start=start
            )
            mom_raw = web.DataReader(
                "F-F_Momentum_Factor", "famafrench", start=start
            )

        ff5 = ff5_raw[0] / 100
        mom = mom_raw[0] / 100

        # Align and merge
        ff5.index = ff5.index.to_timestamp()
        mom.index = mom.index.to_timestamp()
        mom = mom.rename(columns={"Mom": "Mom"})

        df = ff5.join(mom[["Mom"]], how="inner")
        df.index = pd.to_datetime(df.index)

        # Rename RF to just RF
        if "RF" not in df.columns and "Mkt-RF" in df.columns:
            pass  # RF column may be separate

        df.to_parquet(_ff_cache_path(cache_key))
        return df

    except Exception as e:
        logger.warning(f"FF5 fetch failed: {e}")
        return None


# ── OLS helper ────────────────────────────────────────────────────────────────

def _run_ols(y: pd.Series, X: pd.DataFrame):
    """Run OLS with statsmodels. Returns result object."""
    import statsmodels.api as sm
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, missing="drop")
    return model.fit()


# ── CORE: factor_decomposition ────────────────────────────────────────────────

def factor_decomposition(
    prices: pd.Series,
    start: date,
    end: date,
    rolling_window: int = 63,
) -> dict:
    """
    Full Fama-French 5-Factor + Momentum decomposition.

    Args:
        prices: Daily close price series (DatetimeIndex, tz-naive)
        start, end: date range used for data (determines FF data window)
        rolling_window: days for rolling regression (default 63 = 3 months)

    Returns:
        dict with factor loadings, variance decomposition, rolling series,
        idiosyncratic returns and VaR, interpretation flags.
    """
    # Need at least 18 months of history for meaningful decomposition
    ff_start = start - timedelta(days=90)
    ff_df = fetch_ff5_factors(ff_start)

    if ff_df is None or ff_df.empty:
        return {"error": "Fama-French data unavailable", "available": False}

    # Convert daily prices to monthly returns
    monthly_prices = prices.resample("ME").last()
    monthly_ret = monthly_prices.pct_change().dropna()
    monthly_ret.index = monthly_ret.index.normalize()

    # Align FF data to same monthly frequency
    ff_aligned = ff_df.copy()
    ff_aligned.index = ff_aligned.index.normalize()

    # Align on common dates
    common_idx = monthly_ret.index.intersection(ff_aligned.index)
    if len(common_idx) < 18:
        return {
            "error": f"Insufficient overlapping monthly data: {len(common_idx)} months (need 18+)",
            "available": False,
        }

    r = monthly_ret.loc[common_idx]
    ff = ff_aligned.loc[common_idx]

    # RF rate for excess returns
    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0, index=common_idx)
    r_excess = r - rf

    # Factor matrix (exclude RF column)
    factor_cols = [c for c in FACTORS if c in ff.columns]
    X = ff[factor_cols]

    # ── Full-sample OLS ───────────────────────────────────────────────────────
    try:
        result = _run_ols(r_excess, X)
    except Exception as e:
        return {"error": f"OLS failed: {e}", "available": False}

    params = result.params
    tvalues = result.tvalues
    pvalues = result.pvalues
    r2 = result.rsquared

    alpha_monthly = float(params.get("const", 0))
    alpha_ann = (1 + alpha_monthly) ** 12 - 1
    alpha_t = float(tvalues.get("const", 0))
    alpha_p = float(pvalues.get("const", 1))

    betas = {c: float(params.get(c, 0)) for c in factor_cols}
    tstats = {c: float(tvalues.get(c, 0)) for c in factor_cols}

    # Residuals = idiosyncratic returns
    idio_returns_m = pd.Series(result.resid, index=common_idx)

    # ── Variance decomposition ────────────────────────────────────────────────
    total_vol_ann = float(r_excess.std() * np.sqrt(12))
    idio_vol_ann = float(idio_returns_m.std() * np.sqrt(12))
    systematic_vol_ann = float(np.sqrt(max(total_vol_ann**2 - idio_vol_ann**2, 0)))

    systematic_pct = float(r2 * 100)
    idio_pct = float((1 - r2) * 100)

    # Factor contributions to systematic variance
    # Use |beta * factor_std| as proxy for factor contribution
    factor_contribs: dict[str, float] = {}
    total_sys_var = 0.0
    raw_contribs: dict[str, float] = {}
    for c in factor_cols:
        b = betas[c]
        f_std = float(X[c].std())
        contrib = (b * f_std) ** 2
        raw_contribs[c] = contrib
        total_sys_var += contrib

    for c in factor_cols:
        label = FACTOR_LABELS.get(c, c)
        factor_contribs[label] = (
            round(raw_contribs[c] / total_sys_var * 100, 1) if total_sys_var > 0 else 0.0
        )

    # ── Idiosyncratic VaR ─────────────────────────────────────────────────────
    idio_var_95 = float(-np.percentile(idio_returns_m.dropna(), 5))
    idio_cvar_99 = float(
        -idio_returns_m[idio_returns_m <= np.percentile(idio_returns_m, 1)].mean()
    ) if len(idio_returns_m) > 10 else np.nan

    # ── Model quality ─────────────────────────────────────────────────────────
    try:
        aic = float(result.aic)
    except Exception:
        aic = np.nan

    if r2 > 0.70:
        quality = "GOOD"
    elif r2 > 0.40:
        quality = "ADEQUATE"
    else:
        quality = "POOR"

    # ── Rolling regression (rolling_window months) ────────────────────────────
    min_periods = max(12, rolling_window // 2)
    rolling_r2 = pd.Series(dtype=float, index=common_idx)
    rolling_beta_mkt = pd.Series(dtype=float, index=common_idx)
    rolling_idio_vol = pd.Series(dtype=float, index=common_idx)
    rolling_alpha = pd.Series(dtype=float, index=common_idx)

    for i in range(len(common_idx)):
        if i < min_periods:
            continue
        window_idx = common_idx[max(0, i - rolling_window):i + 1]
        r_w = r_excess.loc[window_idx].dropna()
        X_w = X.loc[window_idx].dropna()
        common_w = r_w.index.intersection(X_w.index)
        if len(common_w) < min_periods:
            continue
        try:
            res_w = _run_ols(r_w.loc[common_w], X_w.loc[common_w])
            rolling_r2.iloc[i] = res_w.rsquared
            rolling_beta_mkt.iloc[i] = float(res_w.params.get("Mkt-RF", np.nan))
            rolling_idio_vol.iloc[i] = float(pd.Series(res_w.resid).std() * np.sqrt(12))
            rolling_alpha.iloc[i] = float(res_w.params.get("const", np.nan))
        except Exception:
            continue

    # ── Interpretation flags ──────────────────────────────────────────────────
    flags = []
    if abs(alpha_t) > 2.0:
        flags.append(f"SIGNIFICANT ALPHA: {alpha_ann:.1%} annualized (t={alpha_t:.1f})")
    if idio_pct > 60:
        flags.append(f"PRIMARILY IDIOSYNCRATIC RISK: {idio_pct:.0f}% unexplained by factors")
    if betas.get("Mkt-RF", 0) > 1.5:
        flags.append(f"HIGH MARKET SENSITIVITY: β_mkt={betas['Mkt-RF']:.2f}")
    if rolling_r2.dropna().std() > 0.15:
        flags.append("UNSTABLE FACTOR EXPOSURES: R² varies significantly over time")

    return {
        "available": True,
        # Alpha
        "alpha": round(alpha_ann, 4),
        "alpha_monthly": round(alpha_monthly, 4),
        "alpha_tstat": round(alpha_t, 2),
        "alpha_pvalue": round(alpha_p, 4),
        # Factor betas
        "beta_market": round(betas.get("Mkt-RF", np.nan), 3),
        "beta_size": round(betas.get("SMB", np.nan), 3),
        "beta_value": round(betas.get("HML", np.nan), 3),
        "beta_profitability": round(betas.get("RMW", np.nan), 3),
        "beta_investment": round(betas.get("CMA", np.nan), 3),
        "beta_momentum": round(betas.get("Mom", np.nan), 3),
        # T-statistics
        "tstat_market": round(tstats.get("Mkt-RF", np.nan), 2),
        "tstat_size": round(tstats.get("SMB", np.nan), 2),
        "tstat_value": round(tstats.get("HML", np.nan), 2),
        "tstat_profitability": round(tstats.get("RMW", np.nan), 2),
        "tstat_investment": round(tstats.get("CMA", np.nan), 2),
        "tstat_momentum": round(tstats.get("Mom", np.nan), 2),
        # Variance decomposition
        "r_squared": round(r2, 4),
        "systematic_risk_pct": round(systematic_pct, 1),
        "idiosyncratic_risk_pct": round(idio_pct, 1),
        # Volatility breakdown
        "total_vol_annualized": round(total_vol_ann, 4),
        "systematic_vol": round(systematic_vol_ann, 4),
        "idio_vol": round(idio_vol_ann, 4),
        # Factor contributions to systematic risk
        "factor_contributions": factor_contribs,
        # Rolling time series
        "rolling_r2": rolling_r2.dropna(),
        "rolling_beta_market": rolling_beta_mkt.dropna(),
        "rolling_idio_vol": rolling_idio_vol.dropna(),
        "rolling_alpha": rolling_alpha.dropna(),
        # Idiosyncratic returns and VaR
        "idio_returns": idio_returns_m,
        "idio_var_95": round(idio_var_95, 4),
        "idio_cvar_99": round(idio_cvar_99, 4) if not np.isnan(idio_cvar_99) else None,
        # Model quality
        "information_criterion_aic": round(aic, 1) if not np.isnan(aic) else None,
        "factor_model_quality": quality,
        "n_months": len(common_idx),
        "factor_cols": factor_cols,
        # Interpretation
        "interpretation_flags": flags,
    }
