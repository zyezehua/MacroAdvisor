# MacroAdvisor

An evidence-based, multi-asset market regime & trade advisory engine for US markets.

> **Status:** Phase 1 (signal library + stress index + read-only dashboard). See [Roadmap](#roadmap).

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

## Deployment (Streamlit Cloud)

The cache under `data/` is gitignored, so the deployed app gets its data from a **public
Hugging Face Hub dataset repo**. A GitHub Actions cron refreshes the data and uploads it; the
app downloads it on boot; the cron then reboots the app to pick it up.

```
GitHub Actions cron ──► scripts/refresh_and_upload.py ──► HF dataset repo
                                                              │
                          Streamlit Cloud app ◄──────────────┘ (downloads on boot)
                          (rebooted by the cron after each refresh)
```

- **Refresh:** [.github/workflows/refresh_postclose.yml](.github/workflows/refresh_postclose.yml)
  (full universe + FRED, 22:00 UTC weekdays) and
  [.github/workflows/refresh_intraday.yml](.github/workflows/refresh_intraday.yml)
  (core scope, hourly during US market hours).
- **Sync:** [macro_advisor/storage/remote.py](macro_advisor/storage/remote.py) — `upload_cache`
  (`HfApi.upload_folder`) and `ensure_cache` (`snapshot_download`). Repo defaults to
  `zyezehua/macroadvisor-cache`; override with the `MACROADVISOR_HF_REPO` env var.
- **App:** [macro_advisor/dashboard/app.py](macro_advisor/dashboard/app.py) downloads the cache on
  first load (anonymous, since the repo is public) and recomputes signals/stress live.

### One-time setup

1. Create a Hugging Face **write** token → add repo secret `HF_TOKEN` (GitHub → Settings →
   Secrets and variables → Actions). The dataset repo auto-creates on first upload; make it public.
2. Seed the cache once: `HF_TOKEN=… python scripts/refresh_and_upload.py --scope full`.
3. Deploy on [share.streamlit.io](https://share.streamlit.io): main file
   `macro_advisor/dashboard/app.py`, **Python 3.13** (Streamlit Cloud's max; the data layer is 3.13-compatible).
4. To enable instant refresh-after-cron, add repo secrets `STREAMLIT_API_KEY` and
   `STREAMLIT_APP_ID` (from the Streamlit Cloud dashboard). Without them the app still picks up
   fresh data on its next cold start / cache expiry.

No `FRED_API_KEY` is needed — the FRED adapter uses the keyless CSV endpoint.

## Roadmap

- **Phase 0** — scaffolding, config, data adapters (Yahoo+FRED), cross-check, storage ✓
- **Phase 1** — signal library + stress index + read-only dashboard ← *current*
- **Phase 2** — prediction (walk-forward OOS) + recommendation engine + backtester
- **Phase 3** — manual override UI, custom strategies, then NLP/news/social sentiment

## Disclaimer

For research and educational use only. Not investment advice.
