"""US Treasury daily par-yield curve adapter (primary rates source).

Pulls the official daily Treasury par yield curve from home.treasury.gov — the same
data FRED republishes as the DGS* series, but fetched from the authoritative source
(reachable in environments where FRED is blocked). One HTTP request per calendar year;
returns a wide frame with canonical tenor columns (yields in percent).
"""
from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

SOURCE = "treasury"
_BASE = ("https://home.treasury.gov/resource-center/data-chart-center/"
         "interest-rates/daily-treasury-rates.csv/{year}/all")
_PARAMS = {"type": "daily_treasury_yield_curve", "_format": "csv"}
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
_TIMEOUT = 30

# CSV column label -> canonical tenor id
CANON = {
    "1 Mo": "UST1M",
    "2 Mo": "UST2M",
    "3 Mo": "UST3M",
    "4 Mo": "UST4M",
    "6 Mo": "UST6M",
    "1 Yr": "UST1Y",
    "2 Yr": "UST2Y",
    "3 Yr": "UST3Y",
    "5 Yr": "UST5Y",
    "7 Yr": "UST7Y",
    "10 Yr": "UST10Y",
    "20 Yr": "UST20Y",
    "30 Yr": "UST30Y",
}


def fetch_year(year: int) -> pd.DataFrame:
    """Wide yield-curve frame for one calendar year (empty frame on failure)."""
    url = _BASE.format(year=year)
    try:
        resp = requests.get(url, params=_PARAMS, headers={"User-Agent": _UA},
                            timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df = df.set_index("Date").sort_index()
        rename = {c: CANON[c] for c in df.columns if c in CANON}
        df = df[list(rename)].rename(columns=rename)
        df.index.name = "date"
        return df.apply(pd.to_numeric, errors="coerce")
    except Exception as exc:
        log.warning("treasury fetch failed for %s: %s", year, exc)
        return pd.DataFrame()


def fetch_history(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Wide yield-curve frame spanning the requested date range (year by year)."""
    start_year = int((start or "1990")[:4])
    end_year = int((end or str(date.today().year))[:4])
    frames = [fetch_year(y) for y in range(start_year, end_year + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    return out
