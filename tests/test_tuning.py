"""Hyperparameter tuning picks from the grid via purged inner CV, and the full uplift
(calibration + tuning + weighting) keeps the outer walk-forward leakage-free."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict import walkforward, weighting
from macro_advisor.predict.models import make_model


def _xy_dates(n=600, seed=1):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range(end="2026-01-01", periods=n // 2).values, 2)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = np.sign(X["f1"] - 0.5 * X["f2"] + rng.normal(0, 0.5, n)).astype(int)
    return X, pd.Series(y), dates


def test_tuning_selects_a_grid_candidate():
    X, y, dates = _xy_dates()
    params = {"tune": {"enabled": True, "cv_splits": 3}}
    m = make_model("linear", "clf", params=params).fit(X, y, dates=dates, purge=5)
    assert m.params["hyper"]["C"] in {0.3, 1.0, 3.0}      # chose one of the grid values


def _panel(n_dates=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-05-29", periods=n_dates)
    rows = []
    for sym in ("A", "B"):
        f1, f2 = rng.normal(0, 1, n_dates), rng.normal(0, 1, n_dates)
        direction = np.sign(f1 - 0.5 * f2 + rng.normal(0, 0.3, n_dates)).astype(int)
        rows.append(pd.DataFrame({"date": dates, "symbol": sym, "f1": f1, "f2": f2, "y": direction}))
    panel = pd.concat(rows).set_index(["date", "symbol"]).sort_index()
    return panel, panel["y"], ["f1", "f2"]


def test_walk_forward_leakage_free_with_full_uplift():
    """OOS predictions at shared dates are unchanged when future rows are appended, even with
    calibration + tuning + sample weighting active — the inner splits never see the test block."""
    panel, y, feat = _panel()
    mp = {"calibrate": {"enabled": True, "cv_splits": 3},
          "tune": {"enabled": True, "cv_splits": 3}}
    wfn = weighting.make_weight_fn(
        {"enabled": True, "recency_halflife_days": 200, "uniqueness": True}, horizon=5)
    WF = dict(train_min_days=150, test_days=50, embargo_days=5)

    full = walkforward.walk_forward(panel, y, feat, model_name="gbm", kind="clf", horizon=5,
                                    model_params=mp, weight_fn=wfn, **WF)
    cut = panel.index.get_level_values("date").unique().sort_values()[300]
    keep = panel.index.get_level_values("date") <= cut
    trunc = walkforward.walk_forward(panel[keep], y[keep], feat, model_name="gbm", kind="clf",
                                     horizon=5, model_params=mp, weight_fn=wfn, **WF)
    shared = full.index.intersection(trunc.index)
    assert len(shared) > 0
    assert (full.loc[shared, "pred"] == trunc.loc[shared, "pred"]).all()
