# MacroAdvisor

An evidence-based, multi-asset market regime & trade advisory engine for US markets.

> **Status:** Phase 0 (scaffolding + data layer). See [Roadmap](#roadmap).

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
python scripts/pull_data.py --core   # smoke-test the data pipeline
```

## Roadmap

- **Phase 0** — scaffolding, config, data adapters (Yahoo+FRED), cross-check, storage ← *current*
- **Phase 1** — signal library + stress index + read-only dashboard
- **Phase 2** — prediction (walk-forward OOS) + recommendation engine + backtester
- **Phase 3** — manual override UI, custom strategies, then NLP/news/social sentiment

## Disclaimer

For research and educational use only. Not investment advice.
