"""OOS model diagnostics (Phase 4, workstream E).

Turns the concatenated out-of-sample predictions (``walk_forward`` output: ``pred``/``p_up``/
``p_down``/``y``) into the numbers that tell us whether the ML uplift is *real*:

  * **summary** — Brier score (was ``p_up`` honest?), multiclass log-loss, and OOS hit-rate.
  * **reliability** — a calibration curve: predicted ``p_up`` vs realized up-frequency by bin. A
    well-calibrated model sits on the diagonal; this is what justifies the conviction gate.
  * **conviction** — hit-rate by directional-conviction bucket (do high-conviction calls win more
    often?), which is the property the recommender actually trades on.

Pure functions over a DataFrame so they're trivially testable and carry no model/IO dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _realized(oos: pd.DataFrame) -> pd.DataFrame:
    """Rows with a realized label and valid probabilities.

    The tail of each walk-forward block has no realized ``y`` yet (the forward window runs past the
    sample), and those NaNs are not a valid class — drop them before scoring or ``log_loss`` rejects
    the whole block.
    """
    return oos.dropna(subset=["y", "p_up", "p_down"])


def _proba_matrix(oos: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """3-class probability matrix [p_down, p_flat, p_up] for classes [-1, 0, 1]."""
    p_up = oos["p_up"].to_numpy()
    p_down = oos["p_down"].to_numpy()
    p_flat = np.clip(1.0 - p_up - p_down, 0.0, 1.0)
    P = np.column_stack([p_down, p_flat, p_up])
    P = P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P, [-1, 0, 1]


def summary(oos: pd.DataFrame) -> dict:
    """Brier (up class), multiclass log-loss, and hit-rate for one (model, horizon) OOS block."""
    from sklearn.metrics import brier_score_loss, log_loss

    oos = _realized(oos)
    y = oos["y"].to_numpy()
    P, classes = _proba_matrix(oos)
    out = {"n_oos": int(len(oos)), "hit_rate": round(float((oos["pred"] == oos["y"]).mean()), 4)}
    try:
        out["brier_up"] = round(float(brier_score_loss((y == 1).astype(int), oos["p_up"])), 4)
    except Exception:
        out["brier_up"] = float("nan")
    try:
        out["logloss"] = round(float(log_loss(y, P, labels=classes)), 4)
    except Exception:
        out["logloss"] = float("nan")
    return out


def reliability(oos: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Calibration curve for ``p_up`` vs realized up-frequency, by probability bin."""
    oos = _realized(oos)
    df = pd.DataFrame({"p": oos["p_up"].to_numpy(), "up": (oos["y"].to_numpy() == 1).astype(int)})
    df = df.dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin_mid", "pred_mean", "emp_freq", "count"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = np.clip(np.digitize(df["p"], edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b, g in df.groupby("bin"):
        rows.append({"bin_mid": round((edges[b] + edges[b + 1]) / 2.0, 3),
                     "pred_mean": round(float(g["p"].mean()), 4),
                     "emp_freq": round(float(g["up"].mean()), 4), "count": int(len(g))})
    return pd.DataFrame(rows)


def conviction_table(oos: pd.DataFrame, edges=(0.5, 0.55, 0.6, 0.7, 1.01)) -> pd.DataFrame:
    """Hit-rate by directional-conviction bucket (over non-flat predictions)."""
    oos = _realized(oos)
    denom = (oos["p_up"] + oos["p_down"]).replace(0, np.nan)
    conv = oos[["p_up", "p_down"]].max(axis=1) / denom
    df = pd.DataFrame({"conv": conv.to_numpy(), "pred": oos["pred"].to_numpy(),
                       "y": oos["y"].to_numpy()})
    df = df[df["pred"] != 0].dropna()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = df[(df["conv"] >= lo) & (df["conv"] < hi)]
        if len(g):
            rows.append({"conv_lo": lo, "conv_hi": round(hi, 2), "count": int(len(g)),
                         "hit_rate": round(float((g["pred"] == g["y"]).mean()), 4)})
    return pd.DataFrame(rows)
