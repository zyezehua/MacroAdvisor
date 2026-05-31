"""Two model families behind one interface, for the walk-forward engine.

  * ``LinearModel`` — interpretable: standardized logistic (direction) / ridge (magnitude/stress);
    attribution = coefficient × standardized-feature contribution.
  * ``GBMModel`` — LightGBM classifier/regressor; attribution = SHAP values.

Both expose the same ``fit`` / ``predict`` / ``attribution`` surface so ``walkforward`` can run
either without caring which it is. ``predict`` returns a DataFrame: classifiers add ``pred``
(-1/0/+1), ``p_up``, ``p_down``; regressors add ``pred``.

Heavy imports (sklearn / lightgbm / shap) are local to keep them out of the app import path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLF, REG = "clf", "reg"


class BaseModel:
    name = "base"

    def __init__(self, kind: str):
        assert kind in (CLF, REG)
        self.kind = kind
        self._fitted = None
        self._features: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:           # pragma: no cover - interface
        raise NotImplementedError

    def attribution(self, X: pd.DataFrame) -> pd.DataFrame:       # pragma: no cover - interface
        raise NotImplementedError


class LinearModel(BaseModel):
    name = "linear"

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._features = list(X.columns)
        if self.kind == CLF:
            est = LogisticRegression(max_iter=1000, C=1.0)
        else:
            est = Ridge(alpha=1.0)
        self._fitted = make_pipeline(StandardScaler(), est)
        self._fitted.fit(X.values, y.values)
        return self

    def predict(self, X):
        idx = X.index
        if self.kind == REG:
            return pd.DataFrame({"pred": self._fitted.predict(X.values)}, index=idx)
        est = self._fitted[-1]
        proba = self._fitted.predict_proba(X.values)
        classes = list(est.classes_)
        p_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(idx))
        p_down = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(idx))
        pred = self._fitted.predict(X.values)
        return pd.DataFrame({"pred": pred, "p_up": p_up, "p_down": p_down}, index=idx)

    def attribution(self, X):
        """coef × standardized feature, per row (signed contribution toward 'up'/value)."""
        scaler, est = self._fitted[0], self._fitted[-1]
        Xs = scaler.transform(X.values)
        if self.kind == CLF:
            coef_arr = np.atleast_2d(est.coef_)
            if coef_arr.shape[0] > 1:                      # multiclass: row toward 'up'
                classes = list(est.classes_)
                coef = coef_arr[classes.index(1) if 1 in classes else 0]
            else:                                          # binary: single row = positive class
                coef = coef_arr.ravel()
        else:
            coef = np.ravel(est.coef_)
        return pd.DataFrame(Xs * coef, columns=self._features, index=X.index)


class GBMModel(BaseModel):
    name = "gbm"

    def fit(self, X, y):
        import lightgbm as lgb

        self._features = list(X.columns)
        common = dict(n_estimators=200, num_leaves=15, learning_rate=0.05,
                      min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                      verbosity=-1, n_jobs=1)
        if self.kind == CLF:
            self._fitted = lgb.LGBMClassifier(**common)
        else:
            self._fitted = lgb.LGBMRegressor(**common)
        self._fitted.fit(X.values, y.values)
        return self

    def predict(self, X):
        idx = X.index
        if self.kind == REG:
            return pd.DataFrame({"pred": self._fitted.predict(X.values)}, index=idx)
        classes = list(self._fitted.classes_)
        proba = self._fitted.predict_proba(X.values)
        p_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(idx))
        p_down = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(idx))
        pred = self._fitted.predict(X.values)
        return pd.DataFrame({"pred": pred, "p_up": p_up, "p_down": p_down}, index=idx)

    def attribution(self, X):
        """SHAP values per row; falls back to broadcast feature_importances_ if SHAP fails."""
        try:
            import shap
            vals = shap.TreeExplainer(self._fitted).shap_values(X.values)
            if isinstance(vals, list):                       # multiclass -> pick 'up' class
                classes = list(self._fitted.classes_)
                vals = vals[classes.index(1) if 1 in classes else -1]
            return pd.DataFrame(vals, columns=self._features, index=X.index)
        except Exception:
            imp = np.asarray(self._fitted.feature_importances_, dtype=float)
            imp = imp / (imp.sum() or 1.0)
            return pd.DataFrame(np.tile(imp, (len(X), 1)), columns=self._features, index=X.index)


_REGISTRY = {"linear": LinearModel, "gbm": GBMModel}


def make_model(name: str, kind: str) -> BaseModel:
    return _REGISTRY[name](kind)
