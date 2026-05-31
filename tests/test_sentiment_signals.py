"""Sentiment signal family: orientation + causal forward-fill (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.signals import sentiment as sent
from tests.conftest import bdays, value_frame


def _rising(n=600, lo=50.0, hi=100.0):
    return value_frame(np.linspace(lo, hi, n), bdays(n))


def test_consumer_sentiment_high_is_risk_on(store_factory):
    store = store_factory(series={"UMCSENT": _rising()})
    res = sent.consumer_sentiment(store)
    assert res is not None
    assert res.category == "sentiment"
    assert res.latest_score > 0          # rising/high consumer sentiment -> risk-on
    assert (res.score.dropna().abs() <= 1.0 + 1e-9).all()


def test_financial_conditions_tight_is_risk_off(store_factory):
    # NFCI rising = tightening conditions = stress
    store = store_factory(series={"NFCI": value_frame(np.linspace(-1.0, 1.5, 600), bdays(600))})
    res = sent.financial_conditions(store)
    assert res is not None
    assert res.latest_score < 0          # tighter conditions -> risk-off


def test_absent_series_returns_none(store_factory):
    store = store_factory(series={})
    assert sent.consumer_sentiment(store) is None
    assert sent.news_tone(store) is None


def test_daily_ffill_is_causal():
    # monthly observations; a future value must never fill an earlier date
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    s = pd.Series([10.0, 20.0, 30.0], index=idx)
    out = sent._daily_ffill(s)
    # every business day carries the most recent *prior-or-equal* observation
    assert out.loc["2024-02-15"] == 10.0          # before the Feb print -> still January
    assert out.loc["2024-03-01"] == 20.0          # after Feb print -> February
    assert out.index.is_monotonic_increasing
    assert out.max() <= 30.0                       # never sees beyond the last observation
