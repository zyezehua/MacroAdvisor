"""Publication-lag guard: sentiment/macro features are shifted so nothing leaks pre-release."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.config import load_config
from macro_advisor.predict import features
from macro_advisor.signals import compute_all
from tests.conftest import bdays


class _StoreLike:
    cfg = load_config()


def test_publication_lag_map_reads_universe_config():
    lags = features._publication_lag_bdays(_StoreLike())
    # calendar-day lags from universe.yaml -> business-day rows (round(d * 5/7))
    assert lags["consumer_sentiment"] == round(14 * 5 / 7)      # UMCSENT, 14d -> 10
    assert lags["financial_conditions"] == round(5 * 5 / 7)     # NFCI, 5d -> 4
    assert lags["news_tone"] == 1                                # near-real-time


def test_market_features_shifts_sentiment_forward(store_factory):
    # news_tone has a fixed 1-day lag regardless of universe config, so it exercises the shift
    n = 600
    idx = bdays(n)
    df = pd.DataFrame({"value": np.linspace(-1.0, 1.0, n),
                       "volume": np.linspace(1.0, 2.0, n)}, index=idx)
    store = store_factory(series={"news_markets": df})

    raw = compute_all(store)["news_tone"].score
    mf = features.market_features(store)
    assert "news_tone" in mf.columns
    # downstream (date, symbol) panels reorder on a 'date'-named index; sentiment signals
    # whose score index is unnamed must not drop the name (regression: train _stress_panel)
    assert mf.index.name == "date"
    # the panel column is the raw causal score shifted forward by one business day
    shifted = raw.shift(1)
    aligned = pd.concat([mf["news_tone"], shifted], axis=1).dropna()
    assert len(aligned) > 100
    assert np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])
