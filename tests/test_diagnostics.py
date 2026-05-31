"""OOS diagnostics: Brier / log-loss summary, calibration curve, conviction table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict import diagnostics


def _oos(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p_up = rng.uniform(0, 1, n)
    p_down = rng.uniform(0, 1 - p_up)            # leave room for a flat class
    y = np.where(rng.uniform(size=n) < p_up, 1, np.where(rng.uniform(size=n) < p_down, -1, 0))
    pred = np.where(p_up >= p_down, 1, -1)
    return pd.DataFrame({"p_up": p_up, "p_down": p_down, "pred": pred, "y": y})


def test_summary_reports_finite_brier_logloss_hitrate():
    s = diagnostics.summary(_oos())
    assert s["n_oos"] == 400
    assert 0.0 <= s["hit_rate"] <= 1.0
    assert np.isfinite(s["brier_up"]) and np.isfinite(s["logloss"])


def test_reliability_curve_shape():
    rel = diagnostics.reliability(_oos(), n_bins=10)
    assert set(rel.columns) == {"bin_mid", "pred_mean", "emp_freq", "count"}
    assert rel["count"].sum() == 400
    assert ((rel["emp_freq"] >= 0) & (rel["emp_freq"] <= 1)).all()


def test_handles_unrealized_labels():
    """Tail rows with NaN ``y`` (forward window past the sample) must not break log-loss."""
    oos = _oos(200)
    oos.loc[oos.index[-15:], "y"] = np.nan        # unrealized labels at the block tail
    s = diagnostics.summary(oos)
    assert s["n_oos"] == 185                       # NaN-label rows excluded
    assert np.isfinite(s["logloss"]) and np.isfinite(s["brier_up"])
    assert not diagnostics.reliability(oos).empty
    assert not diagnostics.conviction_table(oos).empty


def test_conviction_table_buckets_are_ordered():
    conv = diagnostics.conviction_table(_oos())
    assert {"conv_lo", "conv_hi", "count", "hit_rate"}.issubset(conv.columns)
    assert (conv["conv_hi"] > conv["conv_lo"]).all()
