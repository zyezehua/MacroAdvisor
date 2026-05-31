"""Model families behind one interface, for the walk-forward engine.

  * ``LinearModel`` — interpretable: standardized logistic (direction) / ridge (magnitude/stress);
    attribution = coefficient × standardized-feature contribution.
  * ``GBMModel`` — LightGBM classifier/regressor; attribution = SHAP values.

Both expose the same ``fit`` / ``predict`` / ``attribution`` surface so ``walkforward`` can run
either without caring which it is. ``predict`` returns a DataFrame: classifiers add ``pred``
(-1/0/+1), ``p_up``, ``p_down``; regressors add ``pred``.

Phase-4 ML uplift threads three optional, **leakage-safe** behaviours through ``fit`` (all default
off, so the Phase-2 behaviour is unchanged unless enabled in config):

  * **Probability calibration** (classifiers) — wraps the estimator in a purged-CV
    ``CalibratedClassifierCV`` so ``p_up``/``p_down`` are trustworthy (the recommender gates trade
    ideas on them). The inner CV is purged via :mod:`macro_advisor.predict.selection`.
  * **Sample weighting** — per-row weights (recency / label-uniqueness; computed by the engine)
    routed into the base estimator's ``fit``.
  * **Hyperparameter tuning** — a small grid scored by purged inner walk-forward CV (workstream B).

Calibration changes *probabilities*, not the qualitative driver story, so ``attribution`` always
comes from a separate base ``_explainer`` fit on the full training fold.

Heavy imports (sklearn / lightgbm / shap) are local to keep them out of the app import path.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLF, REG = "clf", "reg"


class BaseModel:
    name = "base"

    def __init__(self, kind: str, params: dict | None = None):
        assert kind in (CLF, REG)
        self.kind = kind
        self.params = params or {}
        self._fitted = None        # probability/value model used by predict()
        self._explainer = None     # base estimator on full fold, used by attribution()
        self._features: list[str] = []

    def fit(self, X, y, *, sample_weight=None, dates=None, purge=0):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:           # pragma: no cover - interface
        raise NotImplementedError

    def attribution(self, X: pd.DataFrame) -> pd.DataFrame:       # pragma: no cover - interface
        raise NotImplementedError

    # -- shared calibration -------------------------------------------------
    def _maybe_calibrate(self, base, X, y, sample_weight, dates, purge):
        """Return a calibrated classifier if configured & feasible, else ``base`` (refit).

        Uses purged inner K-fold so overlapping labels can't leak into the calibration map. Any
        failure (too little data, a fold missing a class) falls back to the plain base estimator —
        robustness over purity, since a miscalibration is worse than no calibration here.
        """
        cal = self.params.get("calibrate") or {}
        if self.kind != CLF or not cal.get("enabled") or dates is None:
            base.fit(X.values, y.values, **self._sw(sample_weight))
            return base
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV

        from macro_advisor.predict import selection
        n_splits = int(cal.get("cv_splits", 4))
        splits = selection.purged_kfold(dates, n_splits=n_splits, purge=purge)
        # need enough rows and >=2 classes in every train fold for calibration to be valid
        ok = bool(splits) and all(len(np.unique(y.values[tr])) >= 2 for tr, _ in splits)
        if not ok:
            log.info("%s: calibration skipped (insufficient purged folds)", self.name)
            base.fit(X.values, y.values, **self._sw(sample_weight))
            return base
        method = cal.get("method", "isotonic")
        cc = CalibratedClassifierCV(clone(base), method=method, cv=splits)
        try:
            cc.fit(X.values, y.values, **self._sw(sample_weight))
            return cc
        except Exception as exc:                                    # pragma: no cover - defensive
            log.warning("%s: calibration failed (%s); using uncalibrated", self.name, exc)
            base.fit(X.values, y.values, **self._sw(sample_weight))
            return base

    def _sw(self, sample_weight) -> dict:
        """Keyword args to route ``sample_weight`` into the base estimator's fit (or {})."""
        if sample_weight is None:
            return {}
        return {self._sw_key: np.asarray(sample_weight, dtype=float)}

    _sw_key = "sample_weight"

    # -- shared hyperparameter tuning --------------------------------------
    def _grid(self) -> list[dict]:                       # pragma: no cover - overridden
        return []

    def _maybe_tune(self, X, y, dates, purge):
        """Grid-search ``hyper`` via purged inner K-fold; set the best in ``self.params['hyper']``.

        Scored by mean negative log-loss (clf) / MSE (reg) on the purged validation folds. The
        purge gap keeps overlapping labels out of the validation set, so the chosen hyperparameters
        are honestly out-of-sample within the fold. No-op if tuning is disabled or infeasible.
        """
        tune = self.params.get("tune") or {}
        grid = self._grid()
        if not tune.get("enabled") or len(grid) < 2 or dates is None:
            return
        from macro_advisor.predict import selection
        splits = selection.purged_kfold(dates, n_splits=int(tune.get("cv_splits", 3)), purge=purge)
        if not splits:
            return
        base_hyper = dict(self.params.get("hyper", {}))
        best, best_score = None, -np.inf
        for cand in grid:
            self.params["hyper"] = {**base_hyper, **cand}
            score = self._cv_score(X, y, splits)
            if score > best_score:
                best_score, best = score, cand
        self.params["hyper"] = {**base_hyper, **(best or {})}
        log.info("%s: tuned hyper=%s (cv_score=%.4f)", self.name, self.params["hyper"], best_score)

    def _cv_score(self, X, y, splits) -> float:
        from sklearn.metrics import log_loss, mean_squared_error
        Xv, yv = X.values, y.values
        scores = []
        for tr, va in splits:
            if self.kind == CLF and len(np.unique(yv[tr])) < 2:
                continue
            est = self._build()
            est.fit(Xv[tr], yv[tr])
            if self.kind == CLF:
                proba = est.predict_proba(Xv[va])
                scores.append(-log_loss(yv[va], proba, labels=list(est.classes_)))
            else:
                scores.append(-mean_squared_error(yv[va], est.predict(Xv[va])))
        return float(np.mean(scores)) if scores else -np.inf


class LinearModel(BaseModel):
    name = "linear"

    def _build(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        p = self.params.get("hyper", {})
        if self.kind == CLF:
            est = LogisticRegression(max_iter=1000, C=float(p.get("C", 1.0)))
        else:
            est = Ridge(alpha=float(p.get("alpha", 1.0)))
        return Pipeline([("scaler", StandardScaler()), ("est", est)])

    _sw_key = "est__sample_weight"     # route weights to the pipeline's final step

    def _grid(self):
        return [{"C": 0.3}, {"C": 1.0}, {"C": 3.0}] if self.kind == CLF \
            else [{"alpha": 0.3}, {"alpha": 1.0}, {"alpha": 3.0}]

    def fit(self, X, y, *, sample_weight=None, dates=None, purge=0):
        self._features = list(X.columns)
        self._maybe_tune(X, y, dates, purge)
        # explainer: base pipeline on the full fold (drives attribution regardless of calibration)
        self._explainer = self._build()
        self._explainer.fit(X.values, y.values, **self._sw(sample_weight))
        if self.kind == REG:
            self._fitted = self._explainer
        else:
            self._fitted = self._maybe_calibrate(self._build(), X, y, sample_weight, dates, purge)
        return self

    def predict(self, X):
        idx = X.index
        if self.kind == REG:
            return pd.DataFrame({"pred": self._fitted.predict(X.values)}, index=idx)
        proba = self._fitted.predict_proba(X.values)
        classes = list(self._fitted.classes_)
        p_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(idx))
        p_down = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(idx))
        pred = self._fitted.predict(X.values)
        return pd.DataFrame({"pred": pred, "p_up": p_up, "p_down": p_down}, index=idx)

    def attribution(self, X):
        """coef × standardized feature, per row (signed contribution toward 'up'/value)."""
        scaler, est = self._explainer.named_steps["scaler"], self._explainer.named_steps["est"]
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

    def _build(self):
        import lightgbm as lgb

        p = self.params.get("hyper", {})
        common = dict(n_estimators=int(p.get("n_estimators", 200)),
                      num_leaves=int(p.get("num_leaves", 15)),
                      learning_rate=float(p.get("learning_rate", 0.05)),
                      min_child_samples=int(p.get("min_child_samples", 30)),
                      subsample=0.8, colsample_bytree=0.8, verbosity=-1, n_jobs=1)
        return lgb.LGBMClassifier(**common) if self.kind == CLF else lgb.LGBMRegressor(**common)

    def _grid(self):
        return [{"num_leaves": 7, "learning_rate": 0.05},
                {"num_leaves": 15, "learning_rate": 0.05},
                {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 300},
                {"num_leaves": 15, "learning_rate": 0.1}]

    def fit(self, X, y, *, sample_weight=None, dates=None, purge=0):
        self._features = list(X.columns)
        self._maybe_tune(X, y, dates, purge)
        self._explainer = self._build()
        self._explainer.fit(X.values, y.values, **self._sw(sample_weight))
        if self.kind == REG:
            self._fitted = self._explainer
        else:
            self._fitted = self._maybe_calibrate(self._build(), X, y, sample_weight, dates, purge)
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
            vals = shap.TreeExplainer(self._explainer).shap_values(X.values)
            if isinstance(vals, list):                       # multiclass -> pick 'up' class
                classes = list(self._explainer.classes_)
                vals = vals[classes.index(1) if 1 in classes else -1]
            return pd.DataFrame(vals, columns=self._features, index=X.index)
        except Exception:
            imp = np.asarray(self._explainer.feature_importances_, dtype=float)
            imp = imp / (imp.sum() or 1.0)
            return pd.DataFrame(np.tile(imp, (len(X), 1)), columns=self._features, index=X.index)


class StackModel(BaseModel):
    """Out-of-fold stacking ensemble over the base families (Phase 4, workstream D).

    Promotes the recommender's ad-hoc agreement average into a principled meta-learner. Base
    models produce **out-of-fold** predictions on purged inner folds (so the meta-learner never
    trains on in-sample base outputs), a logistic/ridge meta-learner learns the blend, and the
    bases are refit on the full fold for serving. Attribution stays explainable: it is the base
    attributions blended by each base's learned weight in the meta-learner.
    """
    name = "stack"

    def _base_params(self) -> dict:
        # bases keep calibration for trustworthy probabilities, but skip nested tuning/stacking
        return {"calibrate": self.params.get("calibrate")}

    def fit(self, X, y, *, sample_weight=None, dates=None, purge=0):
        from macro_advisor.predict import selection
        self._features = list(X.columns)
        self._base_names = list((self.params.get("stack") or {}).get("base_models", ["linear", "gbm"]))
        bp = self._base_params()
        cv = int((self.params.get("stack") or {}).get("cv_splits", 4))
        splits = selection.purged_kfold(dates, n_splits=cv, purge=purge) if dates is not None else []
        meta = self._oof_meta(X, y, bp, splits, dates, purge, sample_weight)
        self._meta = self._fit_meta(meta, y)
        self._bases = [make_model(nm, self.kind, params=bp).fit(
            X, y, sample_weight=sample_weight, dates=dates, purge=purge) for nm in self._base_names]
        self._explainer = self
        return self

    def _oof_meta(self, X, y, bp, splits, dates, purge, sw):
        per = 2 if self.kind == CLF else 1
        M = np.full((len(X), per * len(self._base_names)), np.nan)
        dvals = np.asarray(dates) if dates is not None else None
        swv = np.asarray(sw) if sw is not None else None
        for tr, va in splits:
            ytr = y.iloc[tr]
            if self.kind == CLF and len(np.unique(ytr.values)) < 2:
                continue
            dtr = dvals[tr] if dvals is not None else None
            swtr = swv[tr] if swv is not None else None
            for j, nm in enumerate(self._base_names):
                m = make_model(nm, self.kind, params=bp).fit(
                    X.iloc[tr], ytr, sample_weight=swtr, dates=dtr, purge=purge)
                pr = m.predict(X.iloc[va])
                if self.kind == CLF:
                    M[va, per * j], M[va, per * j + 1] = pr["p_up"].to_numpy(), pr["p_down"].to_numpy()
                else:
                    M[va, j] = pr["pred"].to_numpy()
        return M

    def _fit_meta(self, M, y):
        from sklearn.linear_model import LogisticRegression, Ridge
        mask = ~np.isnan(M).any(axis=1)
        if mask.sum() < 2:
            return None
        Mm, ym = M[mask], y.values[mask]
        if self.kind == CLF:
            if len(np.unique(ym)) < 2:
                return None
            return LogisticRegression(max_iter=1000).fit(Mm, ym)
        return Ridge(alpha=1.0).fit(Mm, ym)

    def _meta_features(self, X):
        cols = []
        for m in self._bases:
            pr = m.predict(X)
            cols += ([pr["p_up"].to_numpy(), pr["p_down"].to_numpy()] if self.kind == CLF
                     else [pr["pred"].to_numpy()])
        return np.column_stack(cols)

    def predict(self, X):
        idx = X.index
        M = self._meta_features(X)
        if self.kind == REG:
            pred = self._meta.predict(M) if self._meta is not None else M.mean(axis=1)
            return pd.DataFrame({"pred": pred}, index=idx)
        if self._meta is None:                              # degenerate: average base probas
            ups = M[:, 0::2].mean(axis=1)
            downs = M[:, 1::2].mean(axis=1)
            pred = np.where(ups >= downs, 1, -1)
            return pd.DataFrame({"pred": pred, "p_up": ups, "p_down": downs}, index=idx)
        proba, classes = self._meta.predict_proba(M), list(self._meta.classes_)
        p_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(idx))
        p_down = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(idx))
        return pd.DataFrame({"pred": self._meta.predict(M), "p_up": p_up, "p_down": p_down}, index=idx)

    def _base_weights(self):
        n = len(self._bases)
        if self._meta is None:
            return [1.0 / n] * n
        col_imp = np.abs(np.atleast_2d(self._meta.coef_)).mean(axis=0)
        per = 2 if self.kind == CLF else 1
        w = np.array([col_imp[per * j:per * (j + 1)].sum() for j in range(n)])
        s = w.sum()
        return list(w / s) if s > 0 else [1.0 / n] * n

    def attribution(self, X):
        """Base attributions blended by each base's learned weight in the meta-learner."""
        acc = None
        for wj, m in zip(self._base_weights(), self._bases):
            a = m.attribution(X) * wj
            acc = a if acc is None else acc.add(a, fill_value=0.0)
        return acc


_REGISTRY = {"linear": LinearModel, "gbm": GBMModel, "stack": StackModel}


def make_model(name: str, kind: str, *, params: dict | None = None) -> BaseModel:
    return _REGISTRY[name](kind, params=params)
