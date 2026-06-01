# MacroAdvisor

An evidence-based, multi-asset market regime & trade advisory engine for US markets.

> **Status:** Phase 5 (tunable model-signal strategy backtests · PnL attribution · rich stats). See [Roadmap](#roadmap).

## What it does

MacroAdvisor ingests cross-asset market data, computes **explainable** quantitative,
technical, fundamental, policy and (hard-data) sentiment signals, derives a composite
**market stress level**, produces horizon-specific directional views, and ranks
**risk-budgeted trade ideas** — every conclusion is traceable to the factors that drove it.

All forecasting uses **walk-forward out-of-sample** validation with purged/embargoed
splits to eliminate look-ahead bias and leakage.

## MVP scope (locked)

| Dimension | Decision |
|---|---|
| Assets | Equities + Rates/Treasuries (risk-on/off axis) |
| Horizons | Short = 1–5 trading days · Med-long = 1–3 months |
| Modeling | Rules/factor + interpretable statistical models first; ML deferred |
| Backtest universe | Index & sector ETFs only (survivorship-bias-free) |
| Live-signal universe | S&P 500 + Nasdaq constituents (never used in historical OOS) |
| Trade expression | Liquid ETF/futures proxies; structured payoff shown as illustration only |
| Ranking | Configurable risk budget (Sortino-primary) + structured-product view |
| Data | Yahoo Finance + FRED, cross-checked; free & high-credibility only |
| Stack | Python · Streamlit · Plotly · scheduled refresh · local |

### Default risk profile (override in-app)

- Notional **$250,000** · Max drawdown **15%** · Leverage **1.0x**
- Per-position cap **15%** · Per-asset-class cap **60%**
- Ranking: **Sortino-primary**, Sharpe secondary · Hurdle: 3M T-bill

## Architecture

```
macro_advisor/
├── config.py        # YAML config loader (universe, risk budget, params)
├── storage/         # parquet price/series cache + SQLite metadata/provenance
├── ingest/          # source adapters (Yahoo, FRED), each stamping (source, ts, value)
├── crosscheck/      # reconciliation — flags source divergence before any signal computes
├── signals/         # signal library (quant, technical, fundamental, policy, sentiment)
├── stress/          # composite stress index with per-component decomposition
├── predict/         # label construction + walk-forward OOS engine
├── recommend/       # structured payoff units + risk-budget ranking
├── strategy/        # custom rules-builder + in-app backtest (Phase 3)
├── backtest/        # vectorized backtester with costs + walk-forward harness
└── dashboard/       # Streamlit app (Plotly)
```

## Data integrity

Every series carries `(source, pull_timestamp, value)`. The `crosscheck` layer compares
overlapping series (e.g. Yahoo vs Stooq close, FRED yield vs ETF-implied) and raises a
data-quality flag when they diverge beyond tolerance. **No signal is computed on
unreconciled data.** This is the defense against garbage-in inputs and headline/marketing noise.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/pull_data.py --full        # populate the data cache (Yahoo + Treasury curve)
python scripts/compute_signals.py         # signals + composite stress index (text summary)
streamlit run macro_advisor/dashboard/app.py   # interactive read-only dashboard
```

Add `--fred-extras` to `pull_data.py` to also pull credit OAS / real-yield / breakeven series,
which unlock the corresponding FRED-backed signals.

## Prediction & backtest (Phase 2a)

Walk-forward **out-of-sample** forecasting of per-asset forward returns (direction + magnitude)
and the forward stress-index path, with **model families shown side by side**: an interpretable
linear model (coefficient attribution), a gradient-boosted-tree model (SHAP attribution), and — as
of Phase 4 — a stacked meta-learner (see [ML uplift](#ml-uplift-phase-4)). Splits are **purged +
embargoed** so no training row's forward label overlaps its test block — the no-leakage guarantee
is asserted in the tests.

```bash
pip install -r requirements.txt -r requirements-phase2.txt   # lightgbm/shap for training only
python scripts/train_and_backtest.py            # full walk-forward + backtest -> data/oos/*
python scripts/train_and_backtest.py --fast     # coarse steps for a quick local check
```

- **Engine:** [predict/](macro_advisor/predict/) (features → labels → walk-forward → models) and
  [backtest/](macro_advisor/backtest/) (vectorized, costs + slippage, Sortino-primary metrics).
- **Universe:** ETF backtest universe only (survivorship-bias-free); live single names are never
  used in OOS.
- **Artifacts:** the **post-close** cron runs the trainer and ships `data/oos/*.parquet` to HF; the
  dashboard's **Predictions** and **Backtest** tabs read them (so the app needs no ML libraries).
  The intraday cron does not retrain.

Forecasts are research output, not investment advice.

## Trade ideas (Phase 2b)

Ranks the OOS forecasts into **risk-budgeted trade ideas** (liquid ETF proxies). An **ensemble**
(linear + GBM, agreement-aware) drives the headline list, with per-model detail; ideas are ranked
**Sortino-style** (`direction · expected return ÷ downside vol`, with a vol floor so near-cash
ETFs aren't over-rewarded) and gated by directional conviction.

- **Engine:** [recommend/](macro_advisor/recommend/) — `score` (ensemble + risk-adjusted ranking),
  `portfolio` (risk-budget construction), `payoff` (illustration), `engine` (orchestrator).
- **Portfolio:** built to the locked risk budget — per-position ≤ 15%, per-asset-class ≤ 60%,
  gross leverage ≤ 1.0, sized in $ against the $250k notional.
- **Structured payoffs:** the dashboard shows an *illustrative* options payoff-at-expiry for the
  top idea (no pricing/greeks) — purely to show how a view could be expressed.
- **Dashboard:** the **Trade Ideas** tab (horizon selector → ranked ideas, portfolio + allocation
  chart + caps usage, per-model detail, payoff illustration). Computed live in the app from the
  forecast artifacts — no extra deps or workflow changes.

Trade ideas are research & educational output only — **not investment advice**.

## Interactivity & sentiment (Phase 3)

Phase 3 makes the engine interactive and adds the deferred sentiment signal family.

- **Manual override UI** — a sidebar panel reshapes the Trade Ideas list live: notional,
  per-position / per-asset-class caps, gross leverage, ranking objective, conviction gate,
  model-agreement requirement, asset-class filter, and **pin / exclude** specific symbols. It
  patches the config in-memory ([`Config.with_overrides`](macro_advisor/config.py)) and re-runs the
  recommender — the caps are still enforced under any override.
- **Strategy Lab** — a rules-builder: compose a strategy from the signal library + stress level +
  per-asset technicals (`input op threshold`, weighted), pick the ETF universe / direction / sizing /
  rebalance cadence, and **backtest it in-app** against SPY, reusing the Phase-2 vectorized engine
  ([strategy/](macro_advisor/strategy/)). Export/import strategies as JSON. The rules are causal and
  returns are next-day, but thresholds are user-chosen — it is an **in-sample** tool, *not*
  walk-forward-validated (clearly labeled).
- **News/sentiment signals** ([signals/sentiment.py](macro_advisor/signals/sentiment.py)) — a new
  `sentiment` signal family and stress component, from **free, high-credibility** sources:
  - **FRED hard-sentiment** (keyless): U.Mich consumer sentiment (`UMCSENT`), Chicago Fed financial
    conditions (`NFCI`), St. Louis Fed financial stress (`STLFSI4`).
  - **GDELT news tone** ([ingest/gdelt.py](macro_advisor/ingest/gdelt.py), keyless): average news
    tone + volume for a query. **Single-source** — no cross-check mirror exists, so it carries
    staleness/min-history QA only and is labeled single-source and weighted modestly.
  - **No look-ahead:** low-frequency series are forward-filled only (never back-filled), and for the
    OOS feature panel each is shifted by its **publication lag** (`publication_lag_days` in
    `universe.yaml`) so a row never sees a survey before it was released. The adapter is left
    extensible for a keyed news API later. No social media (deferred for credibility).

```bash
python scripts/pull_data.py --full           # full pull now includes news/sentiment
python scripts/pull_data.py --full --no-sentiment   # opt out of the sentiment pull
```

## ML uplift (Phase 4)

Phase 4 squeezes genuine out-of-sample performance out of the Phase-2 harness **without
sacrificing the no-leakage / explainable guarantees**. Every model-selection step runs on a
**purged** inner split of the *training fold only* ([predict/selection.py](macro_advisor/predict/selection.py)) —
it never touches the outer OOS test block, so the walk-forward guarantee is preserved (and
asserted in [tests/test_tuning.py](tests/test_tuning.py)). All steps are config-gated and default-on
in a full nightly run; `--fast` skips the heavy ones.

- **Probability calibration** — classifier `p_up`/`p_down` are wrapped in a purged-CV
  `CalibratedClassifierCV`, so the conviction gate the recommender trades on is honest. Attribution
  is unaffected (it reads a separate base explainer fit on the full fold).
- **Sample weighting** ([predict/weighting.py](macro_advisor/predict/weighting.py)) — exponential
  recency decay + López-de-Prado label-uniqueness (overlapping forward windows are down-weighted),
  normalised to mean 1 so regularization strength is unchanged.
- **Leakage-safe hyperparameter tuning** — a small per-family grid scored by purged inner
  walk-forward CV (negative log-loss / MSE), replacing the hardcoded Phase-2 hyperparameters.
- **Stacking ensemble** (`stack`) — promotes the recommender's ad-hoc agreement average into a
  proper out-of-fold meta-learner over the base families; shown side-by-side as a third model and
  used as the headline ensemble in Trade Ideas. Attribution = base attributions blended by their
  learned meta weight, so it stays explainable.
- **OOS diagnostics** ([predict/diagnostics.py](macro_advisor/predict/diagnostics.py)) — Brier,
  log-loss, a calibration/reliability curve, hit-rate by conviction bucket, and feature-importance
  stability (driver stability ρ). Shipped as `diagnostics/reliability/conviction.parquet` and shown
  in the **Predictions → Model diagnostics** panel (the app reads parquet only — still no ML deps).

```bash
python scripts/train_and_backtest.py            # full uplift (calibration + tuning + stack + diag)
python scripts/train_and_backtest.py --fast     # coarse: tuning/diag skipped for a quick check
```

Config lives under `predict.{calibrate,sample_weight,tune,stack}` in
[config/settings.yaml](config/settings.yaml). Research output only — not investment advice.

## Strategy backtests (Phase 5)

Turns the fixed OOS model backtest into **named, tunable strategies** the user can re-backtest
**live** in the app, each with full **PnL attribution** and a **rich statistics** panel — all on
liquid ETF/futures proxies, all out-of-sample.

- **Default model-signal strategies** ([strategy/model_strategies.py](macro_advisor/strategy/model_strategies.py)) —
  *Ensemble Directional*, *Cross-Asset Ensemble*, *Stress-Gated Equity Trend*, *Risk-On/Off
  Rotation*, *High-Conviction Ensemble*. Each trades the walk-forward OOS model directions, not raw
  signals — so it's genuinely OOS (distinct from the in-sample Strategy Lab).
- **Tunable knobs:** model family · horizon · **conviction/signal threshold** · rebalance/roll
  frequency · **minimum holding period** (hysteresis, decoupled from rebalance) · long-short vs
  long-only · vol-target/equal sizing · leverage & per-position caps · trading costs · and a
  **stress gate** (defensive-equity or risk-on/off rotation across the equity & rates sleeves).
- **PnL attribution** ([backtest/attribution.py](macro_advisor/backtest/attribution.py)) — per-asset
  gross/cost/net contribution, long-vs-short gross, and a gross→cost→net waterfall, as additive
  contributions that reconcile to the headline return.
- **Rich statistics** ([backtest/metrics.py](macro_advisor/backtest/metrics.py) `extended`) —
  annualized vol, Calmar, win/loss + profit/payoff factor, VaR/CVaR(95), skew/kurtosis, beta/alpha
  vs SPY, drawdown duration, time-in-market, gross exposure, annual turnover, and a monthly-return
  table.
- **Live & dependency-free:** the trainer ships `data/oos/oos_predictions.parquet` (the full OOS
  direction series); the **Strategy Backtest** tab re-runs the vectorized backtester as the user
  moves the sliders — pure pandas, **no ML libraries** in the app.

Options / delta-hedged overlays are intentionally **deferred** (no credible free options/IV
history); the strategy spec leaves a clean seam for them. Research output only — not investment advice.

## Deployment (Streamlit Cloud)

The cache under `data/` is gitignored, so the deployed app gets its data from a **public
Hugging Face Hub dataset repo**. A GitHub Actions cron refreshes the data and uploads it; the
app re-pulls the latest snapshot on its own cache TTL (~30 min) — no reboot or extra secrets.

```
GitHub Actions cron ──► scripts/refresh_and_upload.py ──► HF dataset repo
                                                              │
                          Streamlit Cloud app ◄──────────────┘ (re-pulls every ~30 min)
```

- **Refresh:** [.github/workflows/refresh_postclose.yml](.github/workflows/refresh_postclose.yml)
  (full universe + FRED, 22:00 UTC weekdays) and
  [.github/workflows/refresh_intraday.yml](.github/workflows/refresh_intraday.yml)
  (core scope, hourly during US market hours).
- **Sync:** [macro_advisor/storage/remote.py](macro_advisor/storage/remote.py) — `upload_cache`
  (`HfApi.upload_folder`) and `sync_for_app` (`snapshot_download`, only changed files). Repo
  defaults to `zyezehua/macroadvisor-cache`; override with the `MACROADVISOR_HF_REPO` env var.
- **App:** [macro_advisor/dashboard/app.py](macro_advisor/dashboard/app.py) re-pulls the snapshot
  every ~30 min (anonymous read, since the repo is public) and recomputes signals/stress live. A
  locally-pulled dev cache is detected via a marker file and never overwritten.

### One-time setup

1. Create a Hugging Face **write** token at <https://huggingface.co/settings/tokens> → add it as
   repo secret `HF_TOKEN` (GitHub → Settings → Secrets and variables → Actions). The dataset repo
   auto-creates on first upload; set it **public**.
2. Seed the cache once: `HF_TOKEN=… python scripts/refresh_and_upload.py --scope full`.
3. Deploy on [share.streamlit.io](https://share.streamlit.io): main file
   `macro_advisor/dashboard/app.py`, **Python 3.13** (Streamlit Cloud's max; the data layer is 3.13-compatible).

No `FRED_API_KEY` is needed — the FRED adapter uses the keyless CSV endpoint.

## Roadmap

- **Phase 0** — scaffolding, config, data adapters (Yahoo+FRED), cross-check, storage ✓
- **Phase 1** — signal library + stress index + read-only dashboard ✓
- **Phase 2a** — walk-forward OOS prediction + backtester ✓
- **Phase 2b** — recommendation/ranking engine + trade-idea dashboard ✓
- **Phase 3** — manual override UI + custom-strategy lab + news/sentiment signals ✓
- **Phase 4** — ML uplift: calibration · sample weighting · leakage-safe tuning · stacking · diagnostics ✓
- **Phase 5** — tunable model-signal strategy backtests · PnL attribution · rich statistics ← *current*

## Disclaimer

For research and educational use only. Not investment advice.
