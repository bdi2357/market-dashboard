"""
SEC EDGAR Data Layer
====================
Free SEC EDGAR REST API — no paid API key required.
All requests include required User-Agent header.
Rate limit: 10 req/sec max → 0.11s sleep between calls.

Parts:
1. CIK resolution (ticker → CIK)
2. Form 4 insider transactions (open-market buys/sells only)
3. 13F hedge fund holdings (quarterly position changes)
4. XBRL fundamentals (income statement, balance sheet, cash flow)
5. 13D/13G activist detection
"""

import re
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "market-dashboard itaybd@gmail.com"}
EDGAR_BASE = "https://data.sec.gov"
_SLEEP = 0.11  # stay well under 10 req/sec limit

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:14]
    return CACHE_DIR / f"edgar_{h}.parquet"


def _cache_json_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:14]
    return CACHE_DIR / f"edgar_{h}.json"


def _load_df_cache(key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    age_h = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600
    if age_h > max_age_hours:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_df_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(key))
    except Exception:
        pass


def _get(url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """GET with retries and rate-limit sleep."""
    time.sleep(_SLEEP)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(1)
    return None


def _get_text(url: str, retries: int = 3) -> Optional[str]:
    time.sleep(_SLEEP)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(1)
    return None


# ── PART 1: CIK RESOLUTION ────────────────────────────────────────────────────

_CIK_CACHE: dict[str, str] = {}  # in-memory cache across calls

def resolve_cik(ticker: str) -> Optional[str]:
    """
    Resolve ticker symbol → zero-padded 10-digit CIK.
    Uses EDGAR company tickers JSON (most reliable, no rate limit concern).
    """
    ticker_upper = ticker.upper()
    if ticker_upper in _CIK_CACHE:
        return _CIK_CACHE[ticker_upper]

    # Cache to disk for 24h
    key = f"cik_{ticker_upper}"
    path = CACHE_DIR / f"edgar_{hashlib.md5(key.encode()).hexdigest()[:14]}.txt"
    if path.exists():
        age_h = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age_h < 168:  # 1 week
            cik = path.read_text().strip()
            _CIK_CACHE[ticker_upper] = cik
            return cik

    # Fetch EDGAR company_tickers.json (maps ticker → CIK)
    data = _get("https://www.sec.gov/files/company_tickers.json")
    if not data:
        return None

    for entry in data.values():
        if str(entry.get("ticker", "")).upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            _CIK_CACHE[ticker_upper] = cik
            path.write_text(cik)
            return cik
    return None


def get_submissions(cik: str) -> Optional[dict]:
    """Fetch EDGAR submission metadata for a CIK."""
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    return _get(url)


# ── PART 2: FORM 4 INSIDER TRANSACTIONS ──────────────────────────────────────

def fetch_insider_transactions(ticker: str, days_back: int = 180) -> pd.DataFrame:
    """
    Parse Form 4 filings for open-market buys (P) and sells (S).
    Excludes: awards (A), dispositions (D/F/G/J), option exercises (M).

    Returns DataFrame with columns:
    date, name, title, transaction_type, shares, price, dollar_value,
    shares_after, conviction_pct, is_buy
    """
    cache_key = f"form4_{ticker}_{days_back}"
    cached = _load_df_cache(cache_key, max_age_hours=24)
    if cached is not None:
        return cached

    cik = resolve_cik(ticker)
    if not cik:
        return pd.DataFrame()

    submissions = get_submissions(cik)
    if not submissions:
        return pd.DataFrame()

    # Get Form 4 filing accession numbers
    filings = submissions.get("filings", {}).get("recent", {})
    if not filings:
        return pd.DataFrame()

    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    form4_entries = [
        (acc, dt) for form, acc, dt in zip(forms, accessions, dates)
        if form == "4" and dt >= cutoff
    ]

    rows = []
    for accession, filing_date in form4_entries[:50]:  # cap at 50 recent filings
        acc_clean = accession.replace("-", "")
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}.txt"
        # Try to find the XML document
        index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"
        index_text = _get_text(index_url)

        xml_link = None
        if index_text:
            # Find the Form 4 XML in the index
            m = re.search(r'href="(/Archives/edgar/data/[^"]+\.xml)"', index_text, re.IGNORECASE)
            if m:
                xml_link = "https://www.sec.gov" + m.group(1)

        if not xml_link:
            # Fallback: construct common Form 4 XML filename
            xml_link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/form4.xml"

        xml_text = _get_text(xml_link)
        if not xml_text or "<ownershipDocument>" not in xml_text:
            continue

        try:
            parsed = _parse_form4_xml(xml_text, filing_date)
            rows.extend(parsed)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Filter to open-market transactions only (P=purchase, S=sale)
    df = df[df["transaction_type"].isin(["P", "S"])].copy()
    if df.empty:
        return df

    df["is_buy"] = df["transaction_type"] == "P"
    df["dollar_value"] = df["shares"] * df["price"]
    df["conviction_pct"] = np.where(
        df["shares_after"] > 0,
        df["shares"] / df["shares_after"] * 100,
        np.nan,
    )
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    _save_df_cache(cache_key, df)
    return df


def _parse_form4_xml(xml: str, filing_date: str) -> list[dict]:
    """Extract transaction records from Form 4 XML."""
    rows = []

    # Extract reporter info
    name_m = re.search(r"<rptOwnerName>(.*?)</rptOwnerName>", xml, re.S)
    name = name_m.group(1).strip() if name_m else "Unknown"

    title_m = re.search(r"<officerTitle>(.*?)</officerTitle>", xml, re.S)
    title = title_m.group(1).strip() if title_m else ""

    # Is director / 10% owner?
    is_dir = re.search(r"<isDirector>1</isDirector>", xml)
    is_officer = re.search(r"<isOfficer>1</isOfficer>", xml)
    is_ten_pct = re.search(r"<isTenPercentOwner>1</isTenPercentOwner>", xml)

    if not title:
        if is_ten_pct:
            title = "10% Owner"
        elif is_dir:
            title = "Director"

    # Weight: CEO/CFO get 2x, others 1x
    title_lower = title.lower()
    weight = 2.0 if any(t in title_lower for t in ["chief executive", "ceo", "chief financial", "cfo"]) else 1.0

    # Find all nonDerivativeTransaction blocks
    for block in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", xml, re.S):
        try:
            tx_date_m = re.search(r"<transactionDate>\s*<value>(.*?)</value>", block, re.S)
            tx_date = tx_date_m.group(1).strip() if tx_date_m else filing_date

            tx_code_m = re.search(r"<transactionCode>(.*?)</transactionCode>", block, re.S)
            tx_code = tx_code_m.group(1).strip() if tx_code_m else ""

            shares_m = re.search(r"<transactionShares>\s*<value>([\d.,]+)</value>", block, re.S)
            shares = float(shares_m.group(1).replace(",", "")) if shares_m else 0

            price_m = re.search(r"<transactionPricePerShare>\s*<value>([\d.,]+)</value>", block, re.S)
            price = float(price_m.group(1).replace(",", "")) if price_m else 0

            after_m = re.search(r"<sharesOwnedFollowingTransaction>\s*<value>([\d.,]+)</value>", block, re.S)
            shares_after = float(after_m.group(1).replace(",", "")) if after_m else 0

            rows.append({
                "date": tx_date,
                "name": name,
                "title": title,
                "transaction_type": tx_code,
                "shares": shares,
                "price": price,
                "shares_after": shares_after,
                "weight": weight,
            })
        except Exception:
            continue

    return rows


def compute_insider_signals(df: pd.DataFrame) -> dict:
    """
    Compute aggregate insider signals from Form 4 transactions.

    Returns:
    - net_30d, net_60d, net_90d: net dollar value (buy=positive, sell=negative)
    - buy_sell_ratio_90d: buys/(buys+sells) by count
    - cluster_signal: 'BUY', 'SELL', or None
    - cluster_insiders: list of names in cluster
    - weighted_net_90d: CEO/CFO weighted net flow
    """
    if df.empty:
        return {
            "net_30d": 0, "net_60d": 0, "net_90d": 0,
            "buy_sell_ratio_90d": np.nan,
            "cluster_signal": None, "cluster_insiders": [],
            "weighted_net_90d": 0,
        }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    now = pd.Timestamp.now()

    def net_flow(days):
        mask = df["date"] >= (now - pd.Timedelta(days=days))
        sub = df[mask]
        buys = sub[sub["is_buy"]]["dollar_value"].sum()
        sells = sub[~sub["is_buy"]]["dollar_value"].sum()
        return buys - sells

    last90 = df[df["date"] >= (now - pd.Timedelta(days=90))]
    n_buys = last90["is_buy"].sum()
    n_sells = (~last90["is_buy"]).sum()
    bsr = n_buys / (n_buys + n_sells) if (n_buys + n_sells) > 0 else np.nan

    # Cluster detection: 3+ unique insiders same direction in 10 trading days
    cluster_signal = None
    cluster_insiders = []
    last30 = df[df["date"] >= (now - pd.Timedelta(days=30))]
    if not last30.empty:
        buy_names = last30[last30["is_buy"]]["name"].unique().tolist()
        sell_names = last30[~last30["is_buy"]]["name"].unique().tolist()
        if len(buy_names) >= 3:
            cluster_signal = "BUY"
            cluster_insiders = buy_names
        elif len(sell_names) >= 3:
            cluster_signal = "SELL"
            cluster_insiders = sell_names

    # Weighted net flow (CEO/CFO 2x)
    w_net = 0
    for _, row in last90.iterrows():
        sign = 1 if row["is_buy"] else -1
        w_net += sign * row["dollar_value"] * row.get("weight", 1.0)

    return {
        "net_30d": net_flow(30),
        "net_60d": net_flow(60),
        "net_90d": net_flow(90),
        "buy_sell_ratio_90d": bsr,
        "cluster_signal": cluster_signal,
        "cluster_insiders": cluster_insiders,
        "weighted_net_90d": w_net,
    }


# ── PART 3: 13F HEDGE FUND HOLDINGS ──────────────────────────────────────────

def fetch_institutional_changes(ticker: str) -> pd.DataFrame:
    """
    Use yfinance institutional_holders as seed, then fetch 13F data from EDGAR
    for a curated list of large funds to find position changes in the target ticker.

    Returns DataFrame: fund_name, cik, prev_shares, curr_shares, change_shares,
    change_pct, curr_value, pct_of_portfolio, signal (NEW/ADDED/REDUCED/CLOSED)
    """
    cache_key = f"13f_{ticker}"
    cached = _load_df_cache(cache_key, max_age_hours=168)  # refresh weekly
    if cached is not None:
        return cached

    import yfinance as yf
    t = yf.Ticker(ticker)
    inst = t.institutional_holders

    # Seed fund names from yfinance
    seed_funds = []
    if inst is not None and not inst.empty:
        holder_col = "Holder" if "Holder" in inst.columns else inst.columns[0]
        seed_funds = inst[holder_col].dropna().tolist()[:15]

    # Well-known fund CIKs (pre-resolved, reliable)
    KNOWN_FUND_CIKS = {
        "Vanguard Group": "0000102909",
        "BlackRock": "0001364742",
        "State Street": "0000093751",
        "Fidelity": "0000315066",
        "T. Rowe Price": "0001113169",
        "Invesco": "0000914203",
        "JPMorgan": "0000019617",
        "Goldman Sachs": "0000886982",
        "Morgan Stanley": "0000895421",
        "Wellington Management": "0000101899",
    }

    rows = []
    for fund_name, fund_cik in list(KNOWN_FUND_CIKS.items())[:8]:
        try:
            result = _fetch_13f_for_fund(fund_cik, fund_name, ticker)
            if result:
                rows.append(result)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    _save_df_cache(cache_key, df)
    return df


def _fetch_13f_for_fund(cik: str, fund_name: str, target_ticker: str) -> Optional[dict]:
    """
    Fetch latest 2 quarters of 13F for a fund and find position in target_ticker.
    Returns dict with position data or None if not found.
    """
    submissions = get_submissions(cik.zfill(10))
    if not submissions:
        return None

    filings = submissions.get("filings", {}).get("recent", {})
    if not filings:
        return None

    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    descriptions = filings.get("primaryDocument", [])

    # Get last 2 13F-HR filings
    thirteenf_entries = [
        (acc, dt, doc)
        for form, acc, dt, doc in zip(forms, accessions, dates, descriptions)
        if "13F-HR" in str(form)
    ][:2]

    if not thirteenf_entries:
        return None

    quarters = []
    for accession, filing_date, primary_doc in thirteenf_entries:
        shares, value = _parse_13f_holding(cik.zfill(10), accession, target_ticker)
        if shares is not None:
            quarters.append({"date": filing_date, "shares": shares, "value": value})

    if not quarters:
        return None

    curr = quarters[0]
    prev = quarters[1] if len(quarters) > 1 else {"shares": 0, "value": 0}

    change_shares = curr["shares"] - prev["shares"]
    change_pct = change_shares / prev["shares"] * 100 if prev["shares"] > 0 else None

    if prev["shares"] == 0 and curr["shares"] > 0:
        signal = "NEW"
    elif curr["shares"] == 0 and prev["shares"] > 0:
        signal = "CLOSED"
    elif change_shares > 0:
        signal = "ADDED"
    elif change_shares < 0:
        signal = "REDUCED"
    else:
        signal = "UNCHANGED"

    return {
        "fund_name": fund_name,
        "cik": cik,
        "filing_date": curr["date"],
        "curr_shares": curr["shares"],
        "prev_shares": prev["shares"],
        "change_shares": change_shares,
        "change_pct": change_pct,
        "curr_value": curr["value"],
        "signal": signal,
    }


def _parse_13f_holding(cik: str, accession: str, target_ticker: str) -> tuple[Optional[int], Optional[float]]:
    """
    Parse 13F XML to find shares and value held for target_ticker.
    Returns (shares, value) or (None, None).
    """
    acc_clean = accession.replace("-", "")
    # Try to get the index to find the infotable XML
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"
    index_text = _get_text(index_url)
    if not index_text:
        return None, None

    # Find infotable or primary_doc XML
    xml_links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.xml)"', index_text, re.IGNORECASE)
    # Prefer the information table (usually the 2nd XML file)
    infotable_link = None
    for link in xml_links:
        if "infotable" in link.lower() or "form13f" in link.lower():
            infotable_link = "https://www.sec.gov" + link
            break
    if not infotable_link and xml_links:
        infotable_link = "https://www.sec.gov" + xml_links[-1]  # last XML usually infotable

    if not infotable_link:
        return None, None

    xml_text = _get_text(infotable_link)
    if not xml_text:
        return None, None

    # Search for target ticker in holding entries
    ticker_upper = target_ticker.upper()
    # Common XBRL patterns for 13F info table entries
    # Match either <nameOfIssuer> blocks containing the ticker/company
    entries = re.findall(
        r"<infoTable>(.*?)</infoTable>",
        xml_text, re.S | re.IGNORECASE,
    )
    if not entries:
        entries = re.findall(r"<ns1:infoTable>(.*?)</ns1:infoTable>", xml_text, re.S | re.IGNORECASE)

    for entry in entries:
        name_m = re.search(r"<nameOfIssuer>(.*?)</nameOfIssuer>", entry, re.S | re.IGNORECASE)
        if not name_m:
            name_m = re.search(r"<ns1:nameOfIssuer>(.*?)</ns1:nameOfIssuer>", entry, re.S | re.IGNORECASE)
        if not name_m:
            continue
        issuer = name_m.group(1).strip().upper()
        # Fuzzy match: ticker appears in name or exact ticker match
        if ticker_upper not in issuer and not _ticker_matches_issuer(ticker_upper, issuer):
            continue

        shares_m = re.search(r"<sshPrnamt>([\d,]+)</sshPrnamt>", entry, re.S | re.IGNORECASE)
        if not shares_m:
            shares_m = re.search(r"<ns1:sshPrnamt>([\d,]+)</ns1:sshPrnamt>", entry, re.S | re.IGNORECASE)
        value_m = re.search(r"<value>([\d,]+)</value>", entry, re.S | re.IGNORECASE)
        if not value_m:
            value_m = re.search(r"<ns1:value>([\d,]+)</ns1:value>", entry, re.S | re.IGNORECASE)

        shares = int(shares_m.group(1).replace(",", "")) if shares_m else 0
        value = float(value_m.group(1).replace(",", "")) * 1000 if value_m else 0  # 13F values in $thousands
        return shares, value

    return None, None


# Map of common ticker → company name fragments for 13F matching
_TICKER_ISSUER_MAP = {
    "AAL": ["AMERICAN AIRLINES"],
    "DAL": ["DELTA AIR"],
    "UAL": ["UNITED AIRLINES"],
    "AAPL": ["APPLE"],
    "MSFT": ["MICROSOFT"],
    "GOOGL": ["ALPHABET", "GOOGLE"],
    "AMZN": ["AMAZON"],
    "TSLA": ["TESLA"],
    "NVDA": ["NVIDIA"],
    "META": ["META PLATFORMS", "FACEBOOK"],
}


def _ticker_matches_issuer(ticker: str, issuer_upper: str) -> bool:
    fragments = _TICKER_ISSUER_MAP.get(ticker, [ticker])
    return any(frag in issuer_upper for frag in fragments)


def compute_institutional_signals(df: pd.DataFrame) -> dict:
    """Aggregate institutional position changes into signals."""
    if df.empty:
        return {
            "n_adding": 0, "n_reducing": 0, "n_new": 0, "n_closed": 0,
            "net_share_change": 0, "signal": "NEUTRAL",
        }

    n_adding  = len(df[df["signal"].isin(["ADDED", "NEW"])])
    n_reducing = len(df[df["signal"].isin(["REDUCED", "CLOSED"])])
    n_new     = len(df[df["signal"] == "NEW"])
    n_closed  = len(df[df["signal"] == "CLOSED"])
    net       = df["change_shares"].sum()

    if n_new >= 2 or (n_adding >= 3 and n_adding > n_reducing * 2):
        signal = "ACCUMULATION"
    elif n_closed >= 2 or (n_reducing >= 3 and n_reducing > n_adding * 2):
        signal = "DISTRIBUTION"
    else:
        signal = "NEUTRAL"

    return {
        "n_adding": n_adding,
        "n_reducing": n_reducing,
        "n_new": n_new,
        "n_closed": n_closed,
        "net_share_change": net,
        "signal": signal,
    }


# ── PART 4: XBRL FUNDAMENTALS ────────────────────────────────────────────────

# XBRL concept names to try for each metric (order = preference)
XBRL_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomer"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": ["ShortTermBorrowings", "DebtCurrent"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "interest_expense": ["InterestExpense"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "inventory": ["InventoryNet"],
    "ebit": ["OperatingIncomeLoss"],  # proxy
}


def fetch_xbrl_facts(ticker: str) -> Optional[dict]:
    """
    Fetch all XBRL company facts from EDGAR.
    Returns raw facts dict or None. Cached 1 week.
    """
    cache_key = f"xbrl_{ticker}"
    path = CACHE_DIR / f"edgar_{hashlib.md5(cache_key.encode()).hexdigest()[:14]}_xbrl.parquet"

    # Use a simple JSON cache for the large facts blob
    json_path = CACHE_DIR / f"edgar_{hashlib.md5(cache_key.encode()).hexdigest()[:14]}_xbrl.json"
    if json_path.exists():
        age_h = (pd.Timestamp.now() - pd.Timestamp(json_path.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age_h < 168:
            import json
            try:
                with open(json_path) as f:
                    return json.load(f)
            except Exception:
                pass

    cik = resolve_cik(ticker)
    if not cik:
        return None

    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    data = _get(url)
    if not data:
        return None

    import json
    try:
        with open(json_path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

    return data


def _extract_xbrl_series(facts: dict, concept_key: str, form: str = "10-K") -> pd.Series:
    """
    Extract time series for a metric from XBRL facts.
    Returns pd.Series indexed by end date, annual values.
    Tries multiple concept names from XBRL_CONCEPTS[concept_key].
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    concepts = XBRL_CONCEPTS.get(concept_key, [])

    for concept in concepts:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        # Try USD, shares, USD/shares
        for unit_key in ["USD", "shares", "USD/shares"]:
            if unit_key not in units:
                continue
            entries = units[unit_key]
            # Filter to annual filings (form=10-K or 10-Q)
            annual = [
                e for e in entries
                if e.get("form") == form
                and e.get("filed")
                and e.get("val") is not None
            ]
            if not annual:
                continue
            s = pd.Series(
                {e["end"]: e["val"] for e in annual}
            )
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            # Remove duplicates (keep last filed)
            s = s[~s.index.duplicated(keep="last")]
            return s

    return pd.Series(dtype=float)


def fetch_xbrl_fundamentals(ticker: str) -> dict:
    """
    Compute key fundamental metrics from raw XBRL data.
    Returns dict with annual time series and latest values.
    Falls back to None for unavailable concepts.
    """
    facts = fetch_xbrl_facts(ticker)
    if not facts:
        return {"source": "unavailable"}

    result = {"source": "EDGAR XBRL", "ticker": ticker}

    # Extract annual series
    for key in XBRL_CONCEPTS:
        try:
            s = _extract_xbrl_series(facts, key, form="10-K")
            result[key] = s
        except Exception:
            result[key] = pd.Series(dtype=float)

    # Computed metrics (last 4 years)
    def _last(series, n=1):
        s = result.get(series)
        if s is None or (isinstance(s, pd.Series) and s.empty):
            return np.nan
        return float(s.iloc[-n]) if len(s) >= n else np.nan

    # FCF time series
    ocf = result.get("operating_cf", pd.Series(dtype=float))
    capex = result.get("capex", pd.Series(dtype=float))
    if not ocf.empty:
        # Align and compute FCF
        aligned = pd.DataFrame({"ocf": ocf, "capex": capex}).dropna(subset=["ocf"])
        aligned["capex"] = aligned["capex"].fillna(0)
        aligned["fcf"] = aligned["ocf"] - aligned["capex"].abs()
        result["fcf_series"] = aligned["fcf"].tail(8)
    else:
        result["fcf_series"] = pd.Series(dtype=float)

    # Latest values
    result["latest_revenue"] = _last("revenue")
    result["latest_net_income"] = _last("net_income")
    result["latest_operating_income"] = _last("operating_income")
    result["latest_total_assets"] = _last("total_assets")
    result["latest_total_liabilities"] = _last("total_liabilities")
    result["latest_cash"] = _last("cash")
    result["latest_long_term_debt"] = _last("long_term_debt")
    result["latest_operating_cf"] = _last("operating_cf")
    result["latest_fcf"] = float(result["fcf_series"].iloc[-1]) if not result["fcf_series"].empty else np.nan

    # Net debt
    ltd = _last("long_term_debt")
    std = _last("short_term_debt")
    cash = _last("cash")
    result["net_debt"] = (ltd if not np.isnan(ltd) else 0) + (std if not np.isnan(std) else 0) - (cash if not np.isnan(cash) else 0)

    # Accruals ratio
    ni = _last("net_income")
    ocf_v = _last("operating_cf")
    ta = _last("total_assets")
    result["accruals_ratio"] = (ni - ocf_v) / ta if not any(np.isnan(x) for x in [ni, ocf_v, ta]) and ta > 0 else np.nan

    # Interest coverage
    ebit = _last("ebit")
    ie = _last("interest_expense")
    result["interest_coverage"] = abs(ebit / ie) if not any(np.isnan(x) for x in [ebit, ie]) and ie != 0 else np.nan

    return result


# ── PART 5: 13D/13G ACTIVIST DETECTION ───────────────────────────────────────

def fetch_activist_positions(ticker: str) -> pd.DataFrame:
    """
    Fetch 13D and 13G filings for the target company.
    Returns DataFrame: filer_name, form_type, filing_date,
    ownership_pct, amendment, activist_intent
    """
    cache_key = f"activist_{ticker}"
    cached = _load_df_cache(cache_key, max_age_hours=168)  # weekly refresh
    if cached is not None:
        return cached

    cik = resolve_cik(ticker)
    if not cik:
        return pd.DataFrame()

    submissions = get_submissions(cik)
    if not submissions:
        return pd.DataFrame()

    filings = submissions.get("filings", {}).get("recent", {})
    if not filings:
        return pd.DataFrame()

    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    rows = []
    for form, acc, dt in zip(forms, accessions, dates):
        if form not in ("SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"):
            continue

        acc_clean = acc.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{acc}.txt"
        text = _get_text(doc_url)
        if not text:
            continue

        pct_m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", text[:5000])
        pct = float(pct_m.group(1)) if pct_m else np.nan

        filer_m = re.search(r"(?:FILED BY|REPORTING PERSON[S]?)[:\s]+([A-Z][^\n]{3,60})", text[:3000], re.IGNORECASE)
        filer = filer_m.group(1).strip()[:60] if filer_m else "Unknown"

        rows.append({
            "filer_name": filer,
            "form_type": form,
            "filing_date": dt,
            "ownership_pct": pct,
            "amendment": "/A" in form,
            "activist_intent": "13D" in form,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("filing_date", ascending=False).reset_index(drop=True)
        _save_df_cache(cache_key, df)
    return df
