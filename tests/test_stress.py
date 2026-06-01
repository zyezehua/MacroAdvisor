"""Composite stress index: aggregation, decomposition, and weight renormalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.signals.base import SignalResult
from macro_advisor.stress import compute_stress

IDX = pd.bdate_range(end="2026-05-29", periods=80)


def _sig(name, category, score_val) -> SignalResult:
    s = pd.Series(np.full(len(IDX), float(score_val)), index=IDX)
    return SignalResult(name=name, category=category, score=s, raw=s,
                        direction="x", attribution=f"{name}={score_val}",
                        asof=IDX.max(), inputs=[name])

# one signal per component, names matching COMPONENT_OF
_NAMES = [("vix_level", "volatility"), ("hy_ig", "credit"), ("curve_2s10s", "rates"),
          ("momentum", "technical"), ("breadth", "technical"), ("dollar", "cross_asset")]


def _signals(score_val):
    return {n: _sig(n, c, score_val) for n, c in _NAMES}


def test_all_risk_on_is_calm(store_factory):
    store = store_factory()
    res = compute_stress(store, _signals(+1.0))     # stress = -score = -1 everywhere
    assert res.latent == pytest.approx(-1.0, abs=1e-6)
    assert res.level < 30 and res.label == "calm"


def test_all_risk_off_is_crisis(store_factory):
    store = store_factory()
    res = compute_stress(store, _signals(-1.0))
    assert res.latent == pytest.approx(1.0, abs=1e-6)
    assert res.level > 85 and res.label == "crisis"


def test_sentiment_only_tail_does_not_define_reading(store_factory):
    """A real-time news-tone date past the last market close must not collapse the reading to
    sentiment alone (coverage floor anchors `asof` to the last well-covered market date)."""
    store = store_factory()
    sigs = _signals(+0.5)
    ext = IDX.append(pd.bdate_range(start=IDX.max() + pd.offsets.BDay(1), periods=1))
    news = pd.Series(np.full(len(ext), -0.9), index=ext)         # sentiment extends one day past
    sigs["news_tone"] = SignalResult(name="news_tone", category="sentiment", score=news, raw=news,
                                     direction="x", attribution="news", asof=ext.max(),
                                     inputs=["news_tone"])
    res = compute_stress(store, sigs)
    assert res.asof == IDX.max()                                 # not the sentiment-only tail date
    assert len(res.components) > 1 and "volatility" in {c.component for c in res.components}


def test_neutral_is_normal(store_factory):
    store = store_factory()
    res = compute_stress(store, _signals(0.0))
    assert res.level == pytest.approx(50.0, abs=1e-6) and res.label == "normal"


def test_contributions_sum_to_latent(store_factory):
    store = store_factory()
    res = compute_stress(store, _signals(-0.4))
    total = sum(c.contribution for c in res.components)
    assert total == pytest.approx(res.latent, abs=1e-9)
    assert sum(c.weight for c in res.components) == pytest.approx(1.0)


def test_weight_renormalization_with_missing_components(store_factory):
    store = store_factory()
    # only volatility (0.25) + credit (0.20) present -> weights renormalize to sum 1
    sigs = {"vix_level": _sig("vix_level", "volatility", -1.0),
            "hy_ig": _sig("hy_ig", "credit", +1.0)}
    res = compute_stress(store, sigs)
    weights = {c.component: c.weight for c in res.components}
    assert weights["volatility"] == pytest.approx(0.25 / 0.45)
    assert weights["credit"] == pytest.approx(0.20 / 0.45)
    # latent = (0.25*1 + 0.20*(-1)) / 0.45
    assert res.latent == pytest.approx((0.25 - 0.20) / 0.45)
