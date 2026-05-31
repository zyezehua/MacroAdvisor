"""Both model families honor the shared fit/predict/attribution interface."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.predict.models import make_model


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    signal = X["f1"] - 0.5 * X["f2"]
    y_clf = np.sign(signal + rng.normal(0, 0.2, n)).astype(int)
    y_reg = signal + rng.normal(0, 0.1, n)
    return X, pd.Series(y_clf), pd.Series(y_reg)


@pytest.mark.parametrize("name", ["linear", "gbm"])
def test_classifier_interface(name, xy):
    X, y_clf, _ = xy
    m = make_model(name, "clf").fit(X, y_clf)
    preds = m.predict(X)
    assert {"pred", "p_up", "p_down"}.issubset(preds.columns)
    assert ((preds["p_up"] >= 0) & (preds["p_up"] <= 1)).all()
    attrib = m.attribution(X.head(3))
    assert attrib.shape == (3, 2) and list(attrib.columns) == ["f1", "f2"]


@pytest.mark.parametrize("name", ["linear", "gbm"])
def test_regressor_interface(name, xy):
    X, _, y_reg = xy
    m = make_model(name, "reg").fit(X, y_reg)
    preds = m.predict(X)
    assert "pred" in preds.columns and len(preds) == len(X)
