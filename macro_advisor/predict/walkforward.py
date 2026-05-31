"""Purged + embargoed walk-forward out-of-sample engine.

Operates on a **panel** indexed by ``(date, symbol)`` (a single pseudo-symbol for the stress
target). Splits are by *date*: for each forward test block, the model is fit only on rows whose
date precedes the block by a **purge gap** of ``horizon + embargo_days`` trading days — so no
training row's forward-looking label (window ``(t, t+h]``) can overlap the test block. This is the
only thing standing between us and look-ahead, and it's asserted in the tests.

``walk_forward`` returns concatenated OOS predictions (with the realized label joined) across all
blocks — the same series used by the backtester and the historical OOS view. ``final_forecast``
fits once on all available history for the *current* live forecast + attribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict.models import make_model


def _dates(panel: pd.DataFrame) -> np.ndarray:
    return panel.index.get_level_values("date").unique().sort_values().to_numpy()


def walk_forward(panel: pd.DataFrame, label: pd.Series, feat_cols: list[str],
                 *, model_name: str, kind: str, horizon: int,
                 train_min_days: int, test_days: int, embargo_days: int,
                 model_params: dict | None = None,
                 weight_fn=None) -> pd.DataFrame:
    """Run expanding-window walk-forward; return OOS predictions joined with realized labels.

    ``model_params`` is forwarded to the model family (calibration / tuning / hyper config).
    ``weight_fn(dates, y) -> sample_weight`` (Phase-4 workstream C) supplies per-row training
    weights; both are leakage-safe because they only ever see the training fold's own rows.
    """
    dates = _dates(panel)
    purge = horizon + embargo_days
    y_all = label.reindex(panel.index)
    out: list[pd.DataFrame] = []

    start = train_min_days
    while start < len(dates):
        test_dates = dates[start:start + test_days]
        if len(test_dates) == 0:
            break
        train_cutoff = dates[start - purge] if start - purge > 0 else None

        # training rows: strictly before the purge gap, with a known label
        train_mask = panel.index.get_level_values("date") < train_cutoff if train_cutoff is not None else np.zeros(len(panel), bool)
        Xtr, ytr = panel.loc[train_mask, feat_cols], y_all[train_mask]
        ok = ytr.notna().to_numpy()
        Xtr, ytr = Xtr[ok], ytr[ok]

        test_mask = panel.index.get_level_values("date").isin(test_dates)
        Xte = panel.loc[test_mask, feat_cols]

        if len(ytr) < train_min_days or Xte.empty or (kind == "clf" and ytr.nunique() < 2):
            start += test_days
            continue

        tr_dates = Xtr.index.get_level_values("date")
        sw = weight_fn(tr_dates, ytr) if weight_fn is not None else None
        model = make_model(model_name, kind, params=model_params).fit(
            Xtr, ytr, sample_weight=sw, dates=tr_dates, purge=purge)
        preds = model.predict(Xte)
        preds["y"] = y_all[test_mask].to_numpy()
        out.append(preds)
        start += test_days

    if not out:
        return pd.DataFrame()
    res = pd.concat(out).sort_index()
    res.attrs["model"] = model_name
    res.attrs["kind"] = kind
    res.attrs["horizon"] = horizon
    return res


def final_forecast(panel: pd.DataFrame, label: pd.Series, feat_cols: list[str],
                   *, model_name: str, kind: str,
                   model_params: dict | None = None, horizon: int = 0,
                   embargo_days: int = 0, weight_fn=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on all labelled history; return (latest-date predictions, attribution for them)."""
    y_all = label.reindex(panel.index)
    ok = y_all.notna().to_numpy()
    Xtr, ytr = panel.loc[ok, feat_cols], y_all[ok]
    if len(ytr) == 0 or (kind == "clf" and ytr.nunique() < 2):
        return pd.DataFrame(), pd.DataFrame()
    tr_dates = Xtr.index.get_level_values("date")
    sw = weight_fn(tr_dates, ytr) if weight_fn is not None else None
    model = make_model(model_name, kind, params=model_params).fit(
        Xtr, ytr, sample_weight=sw, dates=tr_dates, purge=horizon + embargo_days)

    last_date = panel.index.get_level_values("date").max()
    Xlatest = panel.loc[panel.index.get_level_values("date") == last_date, feat_cols]
    preds = model.predict(Xlatest)
    attrib = model.attribution(Xlatest)
    return preds, attrib
