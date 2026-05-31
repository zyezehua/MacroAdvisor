"""Walk-forward OOS engine: produces OOS predictions and is leakage-free."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict import walkforward


def _panel(n_dates=400, seed=0):
    """Synthetic (date, symbol) panel where direction is a noisy function of features."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-05-29", periods=n_dates)
    rows = []
    for sym in ("A", "B"):
        f1 = rng.normal(0, 1, n_dates)
        f2 = rng.normal(0, 1, n_dates)
        signal = f1 - 0.5 * f2 + rng.normal(0, 0.3, n_dates)
        direction = np.sign(signal).astype(int)
        rows.append(pd.DataFrame({"date": dates, "symbol": sym, "f1": f1, "f2": f2,
                                  "y": direction}))
    panel = pd.concat(rows).set_index(["date", "symbol"]).sort_index()
    return panel, panel["y"], ["f1", "f2"]


WF = dict(train_min_days=150, test_days=50, embargo_days=5)


def test_walk_forward_produces_oos_predictions():
    panel, y, feat = _panel()
    res = walkforward.walk_forward(panel, y, feat, model_name="linear", kind="clf",
                                   horizon=5, **WF)
    assert not res.empty
    assert {"pred", "p_up", "p_down", "y"}.issubset(res.columns)
    # OOS predictions only exist beyond the initial training window
    first_date = res.index.get_level_values("date").min()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    assert first_date >= all_dates[WF["train_min_days"]]


def test_walk_forward_is_leakage_free():
    """OOS predictions at shared dates must be identical when future rows are appended —
    proves a block's fit never sees future data."""
    panel, y, feat = _panel(n_dates=400, seed=2)
    full = walkforward.walk_forward(panel, y, feat, model_name="linear", kind="clf",
                                    horizon=5, **WF)
    # truncate to the first 300 dates and re-run
    dates = panel.index.get_level_values("date").unique().sort_values()
    keep = panel.index.get_level_values("date") <= dates[299]
    pt, yt = panel[keep], y[keep]
    trunc = walkforward.walk_forward(pt, yt, feat, model_name="linear", kind="clf",
                                     horizon=5, **WF)
    shared = full.index.intersection(trunc.index)
    assert len(shared) > 0
    assert (full.loc[shared, "pred"] == trunc.loc[shared, "pred"]).all()


def test_final_forecast_predicts_latest_date():
    panel, y, feat = _panel()
    preds, attrib = walkforward.final_forecast(panel, y, feat, model_name="linear", kind="clf")
    last = panel.index.get_level_values("date").max()
    assert (preds.index.get_level_values("date") == last).all()
    assert list(attrib.columns) == feat            # one attribution per feature
