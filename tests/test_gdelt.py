"""GDELT news-tone adapter — offline (requests mocked), no live network."""
from __future__ import annotations

import pandas as pd

from macro_advisor.ingest import gdelt


def _tone_payload():
    return {"timeline": [{"series": "Average Tone", "data": [
        {"date": "20240101T000000Z", "value": 1.5},
        {"date": "20240102T000000Z", "value": -0.5},
        {"date": "20240103T000000Z", "value": 0.8},
    ]}]}


def _vol_payload():
    return {"timeline": [{"series": "Volume Intensity", "data": [
        {"date": "20240101T000000Z", "value": 2.0},
        {"date": "20240102T000000Z", "value": 3.0},
        {"date": "20240103T000000Z", "value": 2.5},
    ]}]}


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_builds_tone_and_volume_frame(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        mode = params["mode"]
        return _FakeResp(_tone_payload() if mode == "TimelineTone" else _vol_payload())

    monkeypatch.setattr(gdelt.requests, "get", fake_get)
    res = gdelt.fetch("news_markets", "financial markets", timespan_months=24)
    assert res.ok and res.key == "gdelt:news_markets"
    assert list(res.df.columns) == ["value", "volume"]
    assert len(res.df) == 3
    assert res.df.index.is_monotonic_increasing
    assert res.extra.get("single_source") is True


def test_fetch_degrades_when_tone_missing(monkeypatch):
    # best-effort/single-source: an unreachable GDELT is "unavailable", not a hard error
    monkeypatch.setattr(gdelt.requests, "get",
                        lambda *a, **k: _FakeResp({"timeline": []}))
    res = gdelt.fetch("news_markets", "financial markets")
    assert not res.ok and res.status == "unavailable"


def test_parse_timeline_json_handles_gdelt_dates():
    s = gdelt.parse_timeline_json(_tone_payload())
    assert isinstance(s, pd.Series) and len(s) == 3
    assert s.index[0] == pd.Timestamp("2024-01-01")
