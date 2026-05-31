"""GDELT DOC 2.0 news-tone adapter (keyless).

The GDELT DOC API exposes a free, key-less time series of news *tone* and *volume* for an
arbitrary query, aggregated across global online news:

    https://api.gdeltproject.org/api/v2/doc/doc
        ?query=<q>&mode=TimelineTone|TimelineVolRaw&format=json&timespan=<N>months

We pull both modes and return a single daily frame with a ``value`` column (average tone,
positive = more favorable coverage) and a ``volume`` column (share-of-coverage intensity).
This is a **single-source** signal — there is no independent mirror to cross-check against —
so downstream it carries staleness/min-history QA only and is labeled single-source.

Extensible: a keyed news API (NewsAPI, etc.) can be added as a sibling adapter returning the
same ``(value, volume)`` frame shape without touching the signal layer.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from macro_advisor.ingest.base import PullResult

log = logging.getLogger(__name__)

SOURCE = "gdelt"
_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = 45


def _fetch_mode(query: str, mode: str, timespan: str) -> pd.Series | None:
    """Fetch one GDELT timeline mode -> a date-indexed float Series (None on failure)."""
    params = {"query": query, "mode": mode, "format": "json", "timespan": timespan}
    try:
        resp = requests.get(_BASE, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        # GDELT occasionally returns an HTML/text error with a 200; guard the JSON parse.
        payload = resp.json()
    except Exception as exc:
        log.warning("gdelt %s fetch failed for %r: %s", mode, query, exc)
        return None
    timelines = payload.get("timeline") or []
    if not timelines:
        return None
    data = timelines[0].get("data") or []
    if not data:
        return None
    raw = pd.DataFrame(data)
    if "date" not in raw or "value" not in raw:
        return None
    # GDELT dates look like "20220101T120000Z"; fall back to a generic parse if not.
    dt = pd.to_datetime(raw["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(raw["date"], errors="coerce")
    s = pd.Series(pd.to_numeric(raw["value"], errors="coerce").values, index=dt)
    s = s[s.index.notna()].dropna()
    if s.empty:
        return None
    # collapse to one observation per calendar day (GDELT is sub-daily for short spans)
    return s.groupby(s.index.normalize()).mean().sort_index()


def fetch(label: str, query: str, timespan_months: int = 24) -> PullResult:
    """Fetch news tone (+volume) for ``query``, returned under the series id ``label``."""
    timespan = f"{int(timespan_months)}months"
    tone = _fetch_mode(query, "TimelineTone", timespan)
    if tone is None or tone.empty:
        return PullResult(label, SOURCE, "series", df=None, status="error",
                          message=f"no GDELT tone for {query!r}", freq="D")
    vol = _fetch_mode(query, "TimelineVolRaw", timespan)
    df = tone.to_frame("value")
    if vol is not None and not vol.empty:
        df["volume"] = vol.reindex(df.index)
    df = df.sort_index()
    return PullResult(label, SOURCE, "series", df=df, status="ok", freq="D",
                      extra={"query": query, "single_source": True})


def parse_timeline_json(payload: dict, mode_value: str = "value") -> pd.Series:
    """Parse a GDELT timeline JSON payload into a date-indexed Series (test seam)."""
    data = (payload.get("timeline") or [{}])[0].get("data") or []
    raw = pd.DataFrame(data)
    if raw.empty or "date" not in raw:
        return pd.Series(dtype=float)
    dt = pd.to_datetime(raw["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(raw["date"], errors="coerce")
    s = pd.Series(pd.to_numeric(raw[mode_value], errors="coerce").values, index=dt)
    return s[s.index.notna()].dropna().groupby(level=0).mean().sort_index()
