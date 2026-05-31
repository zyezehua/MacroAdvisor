"""The stacking meta-learner fits, predicts valid probabilities, and stays explainable."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict.models import make_model


def _xy_dates(n=600, seed=2):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range(end="2026-01-01", periods=n // 2).values, 2)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = np.sign(X["f1"] - 0.5 * X["f2"] + rng.normal(0, 0.5, n)).astype(int)
    return X, pd.Series(y), dates


def test_stack_classifier_predicts_and_blends_attribution():
    X, y, dates = _xy_dates()
    params = {"stack": {"cv_splits": 3, "base_models": ["linear", "gbm"]}}
    m = make_model("stack", "clf", params=params).fit(X, y, dates=dates, purge=5)
    preds = m.predict(X)
    assert {"pred", "p_up", "p_down"}.issubset(preds.columns)
    assert ((preds["p_up"] >= 0) & (preds["p_up"] <= 1)).all()
    # base weights are a convex blend; attribution carries one column per feature
    w = m._base_weights()
    assert np.isclose(sum(w), 1.0) and all(x >= 0 for x in w)
    assert list(m.attribution(X.head(4)).columns) == ["f1", "f2"]


def test_stack_regressor_predicts():
    rng = np.random.default_rng(3)
    n = 600
    dates = np.repeat(pd.bdate_range(end="2026-01-01", periods=n // 2).values, 2)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(X["f1"] - 0.5 * X["f2"] + rng.normal(0, 0.2, n))
    m = make_model("stack", "reg", params={"stack": {"cv_splits": 3}}).fit(X, y, dates=dates, purge=5)
    preds = m.predict(X)
    assert "pred" in preds.columns and len(preds) == n and preds["pred"].notna().all()
