"""PnL attribution reconciles to the headline return and splits long vs short."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.backtest import attribution, engine
from macro_advisor.config import load_config


def test_attribution_reconciles_and_splits_long_short():
    cfg = load_config()
    dates = pd.bdate_range(end="2026-05-29", periods=300)
    up = pd.Series(np.linspace(100, 140, 300), index=dates, name="SPY")    # rising -> long wins
    down = pd.Series(np.linspace(100, 70, 300), index=dates, name="TLT")   # falling -> short wins
    prices = {"SPY": up, "TLT": down}

    pred_dates = dates[150:]
    rows = []
    for sym, d in (("SPY", 1), ("TLT", -1)):
        rows.append(pd.DataFrame({"date": pred_dates, "symbol": sym, "pred": float(d),
                                  "p_up": 0.9 if d == 1 else 0.0,
                                  "p_down": 0.9 if d == -1 else 0.0}))
    preds = pd.concat(rows).set_index(["date", "symbol"]).sort_index()

    out = engine.run(preds, prices, cfg=cfg)
    attr = attribution.attribute(out)

    per = attr["per_asset"]
    assert set(per["symbol"]) == {"SPY", "TLT"}
    # per-asset net PnL sums to the strategy's cumulative simple return (additive contributions);
    # tolerance reflects the 4-dp rounding applied for display.
    assert per["net_pnl"].sum() == pytest.approx(out["returns"].sum(), abs=1e-3)
    # waterfall: gross + (negative) cost == net
    wf = attr["waterfall"]
    assert wf["gross_pnl"] + wf["cost"] == pytest.approx(wf["net_pnl"], abs=1e-3)
    # both legs were profitable here: long SPY and short TLT
    assert attr["long_short"]["long_gross"] > 0
    assert attr["long_short"]["short_gross"] > 0
