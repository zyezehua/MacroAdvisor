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
from macro_advisor.predict import features, labels, walkforward # noqa: E402
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

        ust = store.try_yield("UST3M")
        rf_daily = (ust / 100.0 / ANN) if ust is not None else 0.0

        metrics_rows, equity_cols, fc_rows, attrib_rows, stress_rows = [], {}, [], [], []

        for hname, h in horizons.items():
            dir_lab, ret_lab = _asset_label_series(prices, h, band)
            spanel, sfeat, slab = _stress_panel(store, level, h)
            for m in models:
                tag = f"{m}_{hname}"
                log.info("walk-forward %s (h=%d)", tag, h)
                oos = walkforward.walk_forward(
                    panel, dir_lab, feat, model_name=m, kind="clf", horizon=h,
                    train_min_days=wf["train_min_days"], test_days=wf["test_days"],
                    embargo_days=wf["embargo_days"])
                if not oos.empty:
                    bt = engine.run(oos, prices, cfg=cfg, rf_daily=rf_daily)
                    metrics_rows.append({"strategy": tag, "model": m, "horizon": hname,
                                         **bt["metrics"], "avg_gross": round(bt["avg_gross"], 3),
                                         "oos_hit_rate": round(float((oos["pred"] == oos["y"]).mean()), 4)})
                    equity_cols[tag] = bt["equity"]

                # current live forecast: direction + expected magnitude
                fc_dir, attrib = walkforward.final_forecast(panel, dir_lab, feat, model_name=m, kind="clf")
                fc_mag, _ = walkforward.final_forecast(panel, ret_lab, feat, model_name=m, kind="reg")
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

                # forward stress forecast (latest)
                sfc, _ = walkforward.final_forecast(spanel, slab, sfeat, model_name=m, kind="reg")
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
