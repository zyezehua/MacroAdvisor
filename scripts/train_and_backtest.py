#!/usr/bin/env python
"""Phase-2a trainer: walk-forward OOS prediction + backtest, written as HF-shippable artifacts.

Run by the post-close cron (after the data pull). Builds the feature panel, runs the
purged/embargoed walk-forward for each model family × horizon (direction target) to produce
genuinely out-of-sample predictions, backtests them, and computes the current live forecast
(direction + expected magnitude) and the forward stress-index forecast.

Artifacts land under ``data/oos/`` and are uploaded to HF alongside the price cache:
  metrics.parquet   strategy + benchmark performance (Sortino/Sharpe/maxDD/hit/CAGR)
  equity.parquet    date×strategy equity curves (+ SPY benchmark)
  forecast.parquet  latest per (model,horizon,symbol): direction, p_up/p_down, expected return
  attrib.parquet    latest top feature drivers per (model,horizon)
  stress.parquet    latest forward stress-change forecast per (model,horizon)
  diagnostics.parquet  OOS Brier / log-loss / hit-rate / driver-stability per (model,horizon)
  reliability.parquet  calibration curve (predicted vs realized up-freq) per (model,horizon)
  conviction.parquet   hit-rate by conviction bucket per (model,horizon)
  oos_predictions.parquet  full OOS direction series (date×symbol×model×horizon) for live backtests
  meta.parquet      asof / universe / params

Examples:
    python scripts/train_and_backtest.py                 # full walk-forward
    python scripts/train_and_backtest.py --fast          # coarse steps (quick local check)
    python scripts/train_and_backtest.py --no-upload     # local; skip HF upload
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_advisor.backtest import engine                       # noqa: E402
from macro_advisor.config import load_config                    # noqa: E402
from macro_advisor.data import MarketStore                      # noqa: E402
from macro_advisor.predict import diagnostics, features, labels, walkforward, weighting  # noqa: E402
from macro_advisor.signals import compute_all                   # noqa: E402
from macro_advisor.storage import remote                        # noqa: E402
from macro_advisor.stress import compute_stress                 # noqa: E402

log = logging.getLogger("train_and_backtest")
ANN = 252.0


def _asset_label_series(prices, h, band):
    """Direction + forward-return labels for the universe, indexed by (date, symbol)."""
    parts = []
    for sym, px in prices.items():
        lab = labels.asset_labels(px, h, band)
        lab["symbol"] = sym
        parts.append(lab.set_index("symbol", append=True))
    df = pd.concat(parts).reorder_levels(["date", "symbol"]).sort_index()
    return df["direction"], df["fwd_ret"]


def _model_params(cfg, fast: bool) -> dict:
    """Assemble the Phase-4 model params (calibration / tuning / stacking) from config.

    ``--fast`` disables hyperparameter tuning (the heavy step) when ``tune.fast_disable`` is set,
    so a quick local run stays quick while a full nightly run gets the tuned models.
    """
    p = cfg.predict
    tune = dict(p.get("tune", {}))
    if fast and tune.get("fast_disable", True):
        tune["enabled"] = False
    return {"calibrate": p.get("calibrate", {}), "tune": tune, "stack": p.get("stack", {})}


def _importance_stability(panel, lab, feat, *, model, mp, h, embargo, attrib_now):
    """Spearman corr of feature importances now vs ~1y earlier — are the drivers stable?"""
    if attrib_now is None or attrib_now.empty:
        return float("nan")
    imp_now = attrib_now.abs().mean()
    dates = panel.index.get_level_values("date")
    cutoff = dates.max() - pd.Timedelta(days=int(round(252 * 7 / 5)))
    prev = panel[dates <= cutoff]
    if prev.empty:
        return float("nan")
    _, attrib_prev = walkforward.final_forecast(prev, lab, feat, model_name=model, kind="clf",
                                                model_params=mp, horizon=h, embargo_days=embargo)
    if attrib_prev is None or attrib_prev.empty:
        return float("nan")
    imp_prev = attrib_prev.abs().mean()
    common = imp_now.index.intersection(imp_prev.index)
    if len(common) < 3:
        return float("nan")
    return round(float(imp_now[common].corr(imp_prev[common], method="spearman")), 3)


def _stress_panel(store, level, h):
    mkt = features.market_features(store).add_prefix("mkt_").dropna()
    mkt = mkt.copy()
    mkt["symbol"] = "_MKT_"
    panel = mkt.set_index("symbol", append=True).reorder_levels(["date", "symbol"]).sort_index()
    feat = [c for c in panel.columns if c.startswith("mkt_")]
    lab = pd.Series(labels.stress_label(level, h)
                    .reindex(panel.index.get_level_values("date")).to_numpy(),
                    index=panel.index)
    return panel, feat, lab


def main() -> int:
    ap = argparse.ArgumentParser(description="MacroAdvisor Phase-2a trainer")
    ap.add_argument("--fast", action="store_true", help="coarse walk-forward steps for a quick run")
    ap.add_argument("--no-upload", action="store_true", help="skip HF upload")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    store = MarketStore()
    try:
        universe = cfg.yahoo_symbols("backtest_equity", "backtest_rates")
        prices = {s: p for s in universe if (p := store.try_price(s)) is not None}
        panel = features.build_panel(store, list(prices))
        if panel.empty:
            print("empty feature panel — is the cache populated?")
            return 1
        feat = features.feature_columns(panel)
        level = compute_stress(store, compute_all(store)).history

        wf = dict(cfg.backtest["walk_forward"])
        if args.fast:
            wf["test_days"] = 252
        models = cfg.predict.get("models", ["linear", "gbm"])
        band = float(cfg.predict.get("direction_neutral_band", 0.005))
        horizons = {"short": cfg.horizons["short"]["max_days"],
                    "med_long": cfg.horizons["med_long"]["max_days"]}
        mp = _model_params(cfg, fast=args.fast)             # Phase-4 calibration/tuning/stack params
        sw_cfg = cfg.predict.get("sample_weight", {})
        embargo = wf["embargo_days"]

        ust = store.try_yield("UST3M")
        rf_daily = (ust / 100.0 / ANN) if ust is not None else 0.0

        metrics_rows, equity_cols, fc_rows, attrib_rows, stress_rows = [], {}, [], [], []
        diag_rows, rel_rows, conv_rows, oos_pred_rows = [], [], [], []

        for hname, h in horizons.items():
            dir_lab, ret_lab = _asset_label_series(prices, h, band)
            spanel, sfeat, slab = _stress_panel(store, level, h)
            wfn = weighting.make_weight_fn(sw_cfg, h)        # leakage-safe per-fold training weights
            for m in models:
                tag = f"{m}_{hname}"
                log.info("walk-forward %s (h=%d)", tag, h)
                oos = walkforward.walk_forward(
                    panel, dir_lab, feat, model_name=m, kind="clf", horizon=h,
                    train_min_days=wf["train_min_days"], test_days=wf["test_days"],
                    embargo_days=embargo, model_params=mp, weight_fn=wfn)
                if not oos.empty:
                    bt = engine.run(oos, prices, cfg=cfg, rf_daily=rf_daily)
                    metrics_rows.append({"strategy": tag, "model": m, "horizon": hname,
                                         **bt["metrics"], "avg_gross": round(bt["avg_gross"], 3),
                                         "oos_hit_rate": round(float((oos["pred"] == oos["y"]).mean()), 4)})
                    equity_cols[tag] = bt["equity"]
                    # ship the OOS prediction series so the app can re-backtest live (Phase 5)
                    op = oos[["pred", "p_up", "p_down"]].reset_index()
                    op["model"], op["horizon"] = m, hname
                    op["p_up"] = op["p_up"].round(4); op["p_down"] = op["p_down"].round(4)
                    oos_pred_rows.append(op)
                    # -- Phase-4 OOS diagnostics --------------------------------
                    rel = diagnostics.reliability(oos); rel["model"], rel["horizon"] = m, hname
                    conv = diagnostics.conviction_table(oos); conv["model"], conv["horizon"] = m, hname
                    rel_rows.append(rel); conv_rows.append(conv)
                    diag = {"model": m, "horizon": hname, **diagnostics.summary(oos)}
                else:
                    diag = None

                # current live forecast: direction + expected magnitude
                fc_dir, attrib = walkforward.final_forecast(panel, dir_lab, feat, model_name=m,
                                                            kind="clf", model_params=mp,
                                                            horizon=h, embargo_days=embargo, weight_fn=wfn)
                fc_mag, _ = walkforward.final_forecast(panel, ret_lab, feat, model_name=m, kind="reg",
                                                       model_params=mp, horizon=h,
                                                       embargo_days=embargo, weight_fn=wfn)
                if not fc_dir.empty:
                    j = fc_dir.copy()
                    j["exp_ret"] = fc_mag["pred"] if not fc_mag.empty else float("nan")
                    j = j.reset_index()
                    j["model"], j["horizon"] = m, hname
                    fc_rows.append(j)
                    top = attrib.abs().mean().sort_values(ascending=False).head(8)
                    for f, v in top.items():
                        attrib_rows.append({"model": m, "horizon": hname, "feature": f,
                                            "importance": round(float(v), 5)})
                if diag is not None:
                    diag["importance_stability"] = (float("nan") if args.fast else
                        _importance_stability(panel, dir_lab, feat, model=m, mp=mp, h=h,
                                              embargo=embargo, attrib_now=attrib))
                    diag_rows.append(diag)

                # forward stress forecast (latest)
                sfc, _ = walkforward.final_forecast(spanel, slab, sfeat, model_name=m, kind="reg",
                                                    model_params=mp, horizon=h, embargo_days=embargo)
                if not sfc.empty:
                    stress_rows.append({"model": m, "horizon": hname,
                                        "current_stress": round(float(level.dropna().iloc[-1]), 1),
                                        "fwd_stress_chg": round(float(sfc["pred"].iloc[0]), 2)})

        # benchmark (SPY) over the strategy window
        if equity_cols and "SPY" in prices:
            idx = pd.concat(equity_cols.values(), axis=1).index
            bench = engine.benchmark(prices, "SPY", idx, rf_daily=rf_daily)
            equity_cols["SPY"] = bench["equity"]
            metrics_rows.append({"strategy": "SPY (buy&hold)", "model": "benchmark",
                                 "horizon": "-", **bench["metrics"]})

        # -- write artifacts ------------------------------------------------
        out = cfg.path("root") / "oos"
        out.mkdir(parents=True, exist_ok=True)
        asof = str(panel.index.get_level_values("date").max().date())
        pd.DataFrame(metrics_rows).to_parquet(out / "metrics.parquet")
        pd.DataFrame(equity_cols).sort_index().to_parquet(out / "equity.parquet")
        pd.concat(fc_rows, ignore_index=True).to_parquet(out / "forecast.parquet")
        pd.DataFrame(attrib_rows).to_parquet(out / "attrib.parquet")
        pd.DataFrame(stress_rows).to_parquet(out / "stress.parquet")
        # Phase-4 diagnostics (calibration / Brier / log-loss / conviction / driver stability)
        pd.DataFrame(diag_rows).to_parquet(out / "diagnostics.parquet")
        (pd.concat(rel_rows, ignore_index=True) if rel_rows else pd.DataFrame()).to_parquet(out / "reliability.parquet")
        (pd.concat(conv_rows, ignore_index=True) if conv_rows else pd.DataFrame()).to_parquet(out / "conviction.parquet")
        # OOS prediction series (date×symbol×model×horizon) for the live Strategy Backtest tab
        (pd.concat(oos_pred_rows, ignore_index=True) if oos_pred_rows else pd.DataFrame()).to_parquet(out / "oos_predictions.parquet")
        pd.DataFrame([{"asof": asof, "n_assets": len(prices), "models": ",".join(models),
                       "test_days": wf["test_days"]}]).to_parquet(out / "meta.parquet")

        print(f"\n=== Phase-2a artifacts written to {out} (asof {asof}) ===")
        print(pd.DataFrame(metrics_rows).to_string(index=False))
    finally:
        store.close()

    if args.no_upload:
        print("\n[--no-upload] skipping HF upload.")
        return 0
    if not os.getenv("HF_TOKEN"):
        print("\nHF_TOKEN not set — skipping upload.")
        return 0
    repo = remote.upload_cache(message="Refresh OOS artifacts (phase-2a)")
    print(f"\nUploaded artifacts to https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
