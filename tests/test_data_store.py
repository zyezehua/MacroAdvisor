"""MarketStore: loading, symbol->key mapping, and the integrity gate."""
from __future__ import annotations

import numpy as np
import pytest

from macro_advisor.data import DataIntegrityError, MissingSeriesError
from tests.conftest import bdays, price_frame, value_frame


def test_price_and_yield_load(store_factory):
    idx = bdays(300)
    store = store_factory(
        prices={"SPY": price_frame(np.linspace(400, 500, 300), idx)},
        series={"UST10Y": value_frame(np.linspace(3.5, 4.5, 300), idx)},
    )
    assert store.price("SPY").iloc[-1] == pytest.approx(500.0, rel=1e-3)
    assert store.yield_("UST10Y").iloc[-1] == pytest.approx(4.5, rel=1e-3)
    assert store.has_price("SPY") and store.has_yield("UST10Y")


def test_missing_series_raises(store_factory):
    store = store_factory()
    with pytest.raises(MissingSeriesError):
        store.price("NOPE")
    assert store.try_price("NOPE") is None


def test_error_flag_blocks_series(store_factory):
    idx = bdays(300)
    store = store_factory(
        prices={"SPY": price_frame(np.linspace(400, 500, 300), idx)},
        flags=[{"key": "yahoo:SPY", "code": "NONPOSITIVE_PRICE", "severity": "error"}],
    )
    with pytest.raises(DataIntegrityError):
        store.price("SPY")
    # info/warn flags must NOT block
    store2 = store_factory(
        prices={"SPY": price_frame(np.linspace(400, 500, 300), idx)},
        flags=[{"key": "yahoo:SPY", "code": "PRICE_JUMP", "severity": "info"}],
    )
    assert store2.price("SPY").iloc[-1] == pytest.approx(500.0, rel=1e-3)
