"""MacroAdvisor read-only dashboard (Phase 1).

Three views (tabs):
  * Overview     — stress gauge, component decomposition, stress history.
  * Signals      — every signal: score, direction, attribution, provenance, sparkline.
  * Data Health  — Phase-0 provenance + QA flags, so the integrity story is visible.

Data source: the local parquet/SQLite cache. Locally it's populated by
``scripts/pull_data.py``; when deployed, it's downloaded on boot from the Hugging Face Hub
dataset repo that the GitHub Actions cron refreshes. Run with:
``streamlit run macro_advisor/dashboard/app.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts this file's directory on sys.path, not the repo root, so make the
# macro_advisor package importable regardless of how the app is launched.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from macro_advisor.config import load_config
from macro_advisor.data import MarketStore
from macro_advisor.recommend import payoff as payoff_mod
from macro_advisor.recommend import recommend as run_recommend
from macro_advisor.signals import compute_all
from macro_advisor.storage import remote
from macro_advisor.stress import StressResult, compute_stress
from macro_advisor.strategy import available_inputs, evaluate as eval_strategy, spec_from_json, spec_to_json
from macro_advisor.strategy.library import presets
from macro_advisor.strategy.spec import REBALANCES, Rule, StrategySpec
from macro_advisor.dashboard import theme

theme.apply_default()   # register the neon Plotly template as the chart default

_HNAME = {"short": "Short (1–5d)", "med_long": "Med-long (1–3m)"}

_CFG = load_config()
_TTL = int((_CFG.remote or {}).get("app_cache_ttl_min", 30)) * 60

_BAND_COLORS = theme.BAND_COLORS
_PAL = theme.PALETTE
_DIR_BADGE = {"risk_on": "🟢 on", "risk_off": "🔴 off", "neutral": "⚪ neutral"}


def _hf_token() -> str | None:
    """Optional HF token from Streamlit secrets (not needed for the public cache repo)."""
    try:
        return st.secrets.get("HF_TOKEN")  # type: ignore[no-any-return]
    except Exception:
        return None


@st.cache_data(ttl=_TTL, show_spinner="Fetching data cache from Hugging Face Hub…")
def _sync_cache() -> str:
    """Sync the data cache from HF Hub. Cached with a TTL, so the deployed app re-pulls the
    latest snapshot every TTL (no reboot needed); a local dev cache is left untouched."""
    try:
        return remote.sync_for_app(_CFG, token=_hf_token())
    except Exception as exc:  # surfaced by the no-data guard in main()
        return f"cache sync failed: {exc}"


@st.cache_data(ttl=_TTL, show_spinner="Computing signals + stress…")
def _load() -> dict:
    """Compute the engine once and return plain (cacheable) structures."""
    _sync_cache()
    store = MarketStore()
    try:
        signals = compute_all(store)
        stress = compute_stress(store, signals) if signals else None
        provenance = store.provenance()
        flags = store.flags()
    finally:
        store.close()

    signal_rows = [
        {
            "category": s.category,
            "signal": s.name,
            "score": round(s.latest_score, 3),
            "direction": s.direction,
            "attribution": s.attribution,
            "inputs": ", ".join(s.inputs),
            "asof": s.asof.date().isoformat() if pd.notna(s.asof) else "",
            "spark": s.score.tail(120).tolist(),
        }
        for s in sorted(signals.values(), key=lambda x: (x.category, x.name))
    ]
    return {
        "stress": stress,
        "signal_rows": signal_rows,
        "provenance": provenance,
        "flags": flags,
        "n_signals": len(signals),
    }


@st.cache_data(ttl=_TTL, show_spinner=False)
def _load_oos() -> dict:
    """Read the precomputed Phase-2 OOS artifacts (data/oos/*.parquet); {} if not present."""
    _sync_cache()
    out: dict[str, pd.DataFrame] = {}
    base = _CFG.path("root") / "oos"
    for name in ("metrics", "equity", "forecast", "attrib", "stress", "meta"):
        f = base / f"{name}.parquet"
        if f.exists():
            out[name] = pd.read_parquet(f)
    return out


def _sidebar_overrides(symbols: list[str]) -> tuple[object, bool]:
    """Render the live risk/ranking override panel; return (effective Config, overrides_active).

    Patches only ``risk_budget`` + ``recommend``, so the (cached) signals/stress are untouched
    and only the Trade Ideas tab recomputes — cheap, pure pandas."""
    rb, rc = _CFG.risk_budget, _CFG.recommend
    st.sidebar.header("⚙️ Risk & ranking overrides")
    st.sidebar.caption("Reshape the Trade Ideas list live. Defaults match the locked risk budget.")
    notional = st.sidebar.number_input("Notional ($)", min_value=10_000, max_value=10_000_000,
                                       value=int(rb["notional_usd"]), step=10_000)
    per_pos = st.sidebar.slider("Per-position cap", 0.05, 0.50, float(rb["per_position_cap"]), 0.01)
    per_cls = st.sidebar.slider("Per-asset-class cap", 0.20, 1.0, float(rb["per_asset_class_cap"]), 0.05)
    leverage = st.sidebar.slider("Gross leverage", 0.25, 2.0, float(rb["max_leverage"]), 0.05)
    objective = st.sidebar.selectbox("Ranking objective", ["sortino", "sharpe"],
                                     index=0 if rb.get("ranking_objective", "sortino") == "sortino" else 1)
    conviction = st.sidebar.slider("Min conviction", 0.50, 0.90, float(rc.get("min_conviction", 0.55)), 0.01)
    require_agree = st.sidebar.checkbox("Require model agreement", bool(rc.get("require_agreement", False)))
    classes = st.sidebar.multiselect("Asset classes", ["equities", "rates"], default=["equities", "rates"])
    pinned = st.sidebar.multiselect("Pin symbols (always include)", symbols, default=[])
    base_excl = list(rc.get("exclude_symbols", []))
    excluded = st.sidebar.multiselect("Exclude symbols", sorted(set(symbols) | set(base_excl)),
                                      default=[s for s in base_excl if s in symbols] or base_excl)

    patch = {
        "risk_budget": {"notional_usd": int(notional), "per_position_cap": float(per_pos),
                        "per_asset_class_cap": float(per_cls), "max_leverage": float(leverage),
                        "ranking_objective": objective},
        "recommend": {"min_conviction": float(conviction), "require_agreement": bool(require_agree),
                      "include_asset_classes": classes, "pinned_symbols": pinned,
                      "exclude_symbols": excluded},
    }
    active = (int(notional) != int(rb["notional_usd"]) or per_pos != rb["per_position_cap"]
              or per_cls != rb["per_asset_class_cap"] or leverage != rb["max_leverage"]
              or objective != rb.get("ranking_objective", "sortino")
              or conviction != rc.get("min_conviction", 0.55) or bool(require_agree) != bool(rc.get("require_agreement", False))
              or set(classes) != {"equities", "rates"} or pinned or set(excluded) != set(base_excl))
    if st.sidebar.button("Reset to defaults"):
        st.rerun()
    return _CFG.with_overrides(patch), bool(active)


def _render_strategy_lab(signal_names: list[str]) -> None:
    """Rules-builder + in-app backtest (reuses the Phase-2 engine via strategy.evaluate)."""
    st.subheader("🧪 Strategy Lab — build & backtest a custom rule")
    st.caption("⚠️ **Research only.** A user-defined, **in-sample** rule — *not* walk-forward-validated; "
               "hand-tuned thresholds can curve-fit. Signals are causal and returns are next-day, "
               "but this is an exploration tool, not advice.")

    inputs = available_inputs(signal_names=signal_names)
    input_names = sorted(inputs)
    etf_universe = _CFG.yahoo_symbols("backtest_equity", "backtest_rates")

    preset_map = presets()
    choice = st.selectbox("Start from", ["Blank"] + list(preset_map))
    seed = preset_map[choice] if choice in preset_map else None

    name = st.text_input("Strategy name", value=(seed.name if seed else "My strategy"))
    universe = st.multiselect("Universe (ETFs)", etf_universe,
                              default=(seed.universe if seed else ["SPY", "QQQ"]))
    seed_rules = ([r.to_dict() for r in seed.rules] if seed
                  else [{"input": input_names[0], "op": ">", "threshold": 0.0, "weight": 1.0}])
    rules_df = st.data_editor(
        pd.DataFrame(seed_rules), num_rows="dynamic", width="stretch", key=f"rules_{choice}",
        column_config={
            "input": st.column_config.SelectboxColumn("input", options=input_names, width="medium"),
            "op": st.column_config.SelectboxColumn("op", options=[">", ">=", "<", "<="]),
            "threshold": st.column_config.NumberColumn("threshold", format="%.3f"),
            "weight": st.column_config.NumberColumn("weight", format="%.2f"),
        })
    with st.expander("Input reference"):
        st.dataframe(pd.DataFrame([{"input": k, **v} for k, v in inputs.items()]),
                     hide_index=True, width="stretch")

    c1, c2, c3 = st.columns(3)
    direction = c1.radio("Direction", ["long_only", "long_short"], index=0)
    sizing = c2.radio("Sizing", ["vol_target", "equal"], index=0)
    rebalance = c3.radio("Rebalance", list(REBALANCES), index=list(REBALANCES).index("weekly"))
    cc1, cc2 = st.columns(2)
    per_pos = cc1.slider("Per-position cap", 0.05, 0.50, float(_CFG.risk_budget["per_position_cap"]), 0.01)
    leverage = cc2.slider("Gross leverage", 0.25, 2.0, float(_CFG.risk_budget["max_leverage"]), 0.05)

    rules = [Rule(input=str(r["input"]), op=str(r["op"]), threshold=float(r["threshold"]),
                  weight=float(r.get("weight", 1.0)))
             for r in rules_df.to_dict("records") if r.get("input")]
    try:
        spec = StrategySpec(name=name or "custom", universe=universe, rules=rules,
                            direction=direction, sizing=sizing, rebalance=rebalance,
                            caps={"per_position_cap": per_pos, "max_leverage": leverage}).validate()
    except ValueError as exc:
        st.info(f"Define a universe and at least one rule to run. ({exc})")
        return

    st.download_button("⬇️ Export strategy (JSON)", spec_to_json(spec),
                       file_name=f"{(name or 'strategy').replace(' ', '_')}.json", mime="application/json")
    up = st.file_uploader("Import strategy (JSON)", type=["json"])
    if up is not None:
        try:
            imported = spec_from_json(up.getvalue().decode("utf-8"))
            st.success(f"Imported '{imported.name}'. Adjust above, or rerun the page to load its rules.")
        except Exception as exc:
            st.error(f"Could not parse strategy JSON: {exc}")

    if not st.button("▶️ Run backtest", type="primary"):
        return
    store = MarketStore()
    try:
        with st.spinner("Backtesting (causal, costs applied)…"):
            out = eval_strategy(spec, store, _CFG)
    finally:
        store.close()
    if "error" in out:
        st.warning(out["error"])
        return

    s = out["summary"]
    st.caption(f"{s['n_assets']} asset(s) · {s['first']} → {s['last']} · objective: {s['objective']}")
    eq = out["equity"]
    fig = go.Figure()
    for col in eq.columns:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col], mode="lines", name=col,
                                 line=dict(width=2.5 if col == spec.name else 1.5)))
    fig.update_layout(title="Custom-strategy equity (growth of $1) vs SPY", height=420,
                      margin=dict(t=50, b=20, l=10, r=10), legend_title="strategy")
    st.plotly_chart(theme.style_fig(fig), width="stretch")
    mt = out["metrics"].copy()
    for c in ("max_drawdown", "cagr", "hit_rate"):
        if c in mt.columns:
            mt[c] = (mt[c] * 100).round(1)
    st.dataframe(mt, hide_index=True, width="stretch", column_config={
        "sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "max_drawdown": st.column_config.NumberColumn("max DD %", format="%.1f"),
        "cagr": st.column_config.NumberColumn("CAGR %", format="%.1f"),
        "hit_rate": st.column_config.NumberColumn("hit %", format="%.1f"),
        "avg_gross": st.column_config.NumberColumn("avg gross", format="%.2f"),
    })


def _gauge(stress: StressResult) -> go.Figure:
    color = _BAND_COLORS.get(stress.label, _PAL["muted"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress.level,
        number={"suffix": " / 100", "font": {"size": 40, "color": color}},
        title={"text": f"Market Stress — <b>{stress.label.upper()}</b>"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _PAL["muted"]},
            "bar": {"color": color, "thickness": 0.7},
            "bordercolor": _PAL["border"],
            "steps": theme.GAUGE_STEPS,
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    return theme.style_fig(fig)


def _decomposition(stress: StressResult) -> go.Figure:
    comps = stress.components
    names = [c.component for c in comps]
    vals = [c.contribution for c in comps]
    colors = [_PAL["down"] if v >= 0 else _PAL["up"] for v in vals]  # +stress red, -stress green
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    fig.update_layout(
        title=f"Component contribution to latent ({stress.latent:+.3f})",
        height=320, margin=dict(t=50, b=20, l=10, r=10),
        xaxis_title="weight × stress  (→ risk-off / ← risk-on)",
    )
    return theme.style_fig(fig)


def _history(stress: StressResult) -> go.Figure:
    h = stress.history.tail(504)  # ~2y
    fig = go.Figure(go.Scatter(x=h.index, y=h.values, mode="lines",
                               line=dict(color=_PAL["cyan"], width=2),
                               fill="tozeroy", fillcolor="rgba(14,165,233,0.08)"))
    for edge, col in [(30, _BAND_COLORS["calm"]), (55, _BAND_COLORS["normal"]),
                      (70, _BAND_COLORS["stressed"]), (85, _BAND_COLORS["crisis"])]:
        fig.add_hline(y=edge, line_dash="dot", line_color=col, opacity=0.4)
    fig.update_layout(title="Stress history (~2y)", height=320,
                      margin=dict(t=50, b=20, l=10, r=10), yaxis_range=[0, 100])
    return theme.style_fig(fig)


def main() -> None:
    st.set_page_config(page_title="MacroAdvisor", layout="wide", page_icon="📊")
    theme.inject_css()
    st.title("📊 MacroAdvisor")
    st.caption("Evidence-based market regime & stress read, OOS forecasts, risk-budgeted trade "
               "ideas, and a custom-strategy lab — Phase 3.  Research only; not investment advice.")

    data = _load()
    stress: StressResult | None = data["stress"]
    if stress is None:
        st.error(
            "No signals available — the data cache is empty.\n\n"
            f"- Cache sync: **{_sync_cache()}**\n"
            "- Locally: run `python scripts/pull_data.py --full`\n"
            "- Deployed: check the GitHub Actions refresh ran and the HF dataset repo "
            f"`{remote.resolve_repo(_CFG)}` has data."
        )
        st.stop()

    oos = _load_oos()
    signal_names = sorted(r["signal"] for r in data["signal_rows"])
    idea_symbols = (sorted(oos["forecast"]["symbol"].unique())
                    if "forecast" in oos else _CFG.yahoo_symbols("backtest_equity", "backtest_rates"))
    cfg_eff, overrides_active = _sidebar_overrides(idea_symbols)

    overview, signals_tab, predictions, backtest_tab, ideas_tab, strategy_tab, health = st.tabs(
        ["Overview", "Signals", "Predictions", "Backtest", "Trade Ideas", "Strategy Lab", "Data Health"])

    # -- Overview --------------------------------------------------------
    with overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Stress level", f"{stress.level:.1f}", stress.label.upper())
        c2.metric("Latent (z-space)", f"{stress.latent:+.3f}")
        c3.metric("Signals firing", str(stress.n_signals))
        st.caption(f"As of {stress.asof.date()}")

        left, right = st.columns([1, 1])
        left.plotly_chart(_gauge(stress), width="stretch")
        right.plotly_chart(_decomposition(stress), width="stretch")
        st.plotly_chart(_history(stress), width="stretch")

        st.subheader("Top drivers")
        for line in stress.top_drivers[:8]:
            st.markdown(f"- {line}")

    # -- Signals ---------------------------------------------------------
    with signals_tab:
        st.subheader("Signal library")
        df = pd.DataFrame(data["signal_rows"])
        df["direction"] = df["direction"].map(lambda d: _DIR_BADGE.get(d, d))
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn("score", format="%.2f"),
                "spark": st.column_config.LineChartColumn("score trend", y_min=-1, y_max=1),
                "attribution": st.column_config.TextColumn("attribution", width="large"),
            },
        )
        st.caption("Score in [-1, +1]: +risk-on / -risk-off, z-scored vs each signal's "
                   "own trailing window. Every signal is traceable to the listed inputs.")

    # -- Predictions -----------------------------------------------------
    with predictions:
        if "forecast" not in oos:
            st.info("No prediction artifacts yet. Run `python scripts/train_and_backtest.py` "
                    "(the post-close cron produces these).")
        else:
            fc, stress_fc = oos["forecast"], oos.get("stress", pd.DataFrame())
            attrib = oos.get("attrib", pd.DataFrame())
            meta = oos.get("meta", pd.DataFrame())
            if not meta.empty:
                st.caption(f"Walk-forward OOS forecast · as of {meta.iloc[0].get('asof','?')} · "
                           "+1 = up / −1 = down / 0 = flat over the horizon")
            for model_label, model_key in (("Interpretable (linear)", "linear"), ("ML (gradient-boosted trees)", "gbm")):
                st.subheader(model_label)
                mfc = fc[fc["model"] == model_key]
                if mfc.empty:
                    st.caption("— no forecast for this model —")
                    continue
                cols = st.columns(2)
                for col, (hk, htitle) in zip(cols, _HNAME.items()):
                    sub = mfc[mfc["horizon"] == hk]
                    with col:
                        st.markdown(f"**{htitle}**")
                        if not stress_fc.empty:
                            srow = stress_fc[(stress_fc["model"] == model_key) & (stress_fc["horizon"] == hk)]
                            if not srow.empty:
                                chg = float(srow.iloc[0]["fwd_stress_chg"])
                                st.metric("Stress forecast", f"{srow.iloc[0]['current_stress']:.0f} → "
                                          f"{srow.iloc[0]['current_stress']+chg:.0f}", f"{chg:+.1f}")
                        show = sub[["symbol", "pred", "p_up", "p_down", "exp_ret"]].copy()
                        show["dir"] = show["pred"].map({1: "🟢 up", -1: "🔴 down", 0: "⚪ flat"})
                        show = show[["symbol", "dir", "p_up", "p_down", "exp_ret"]].sort_values("exp_ret", ascending=False)
                        show["exp_ret"] = (show["exp_ret"] * 100).round(2)   # -> percent
                        st.dataframe(show, hide_index=True, width="stretch", column_config={
                            "p_up": st.column_config.NumberColumn("P(up)", format="%.2f"),
                            "p_down": st.column_config.NumberColumn("P(down)", format="%.2f"),
                            "exp_ret": st.column_config.NumberColumn("exp. return %", format="%.2f"),
                        })
                if not attrib.empty:
                    ma = attrib[attrib["model"] == model_key]
                    if not ma.empty:
                        top = (ma.groupby("feature")["importance"].mean()
                               .sort_values(ascending=False).head(8))
                        st.caption("Top drivers: " + " · ".join(f"{f} ({v:.3f})" for f, v in top.items()))
            st.caption("OOS = purged/embargoed walk-forward; the linear model is read off its "
                       "coefficients, the GBM off SHAP values. Research only — not investment advice.")

    # -- Backtest --------------------------------------------------------
    with backtest_tab:
        if "metrics" not in oos:
            st.info("No backtest artifacts yet. Run `python scripts/train_and_backtest.py`.")
        else:
            st.subheader("Out-of-sample performance")
            mt = oos["metrics"].copy()
            for c in ("max_drawdown", "cagr", "hit_rate", "oos_hit_rate"):
                if c in mt.columns:
                    mt[c] = (mt[c] * 100).round(1)          # fractions -> percent
            st.dataframe(mt, hide_index=True, width="stretch", column_config={
                "sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn("max DD %", format="%.1f"),
                "cagr": st.column_config.NumberColumn("CAGR %", format="%.1f"),
                "hit_rate": st.column_config.NumberColumn("hit %", format="%.1f"),
                "oos_hit_rate": st.column_config.NumberColumn("OOS dir. hit %", format="%.1f"),
            })
            if "equity" in oos and not oos["equity"].empty:
                eq = oos["equity"]
                fig = go.Figure()
                for c in eq.columns:
                    fig.add_trace(go.Scatter(x=eq.index, y=eq[c], mode="lines", name=c,
                                             line=dict(width=2.5 if c == "SPY" else 1.5)))
                fig.update_layout(title="OOS equity curves (growth of $1)", height=420,
                                  margin=dict(t=50, b=20, l=10, r=10), legend_title="strategy")
                st.plotly_chart(theme.style_fig(fig), width="stretch")
            st.caption("Sortino-primary (the locked ranking objective). Costs + slippage applied; "
                       "vol-targeted within the risk budget; only OOS-predicted dates trade.")

    # -- Trade Ideas -----------------------------------------------------
    with ideas_tab:
        st.subheader("Trade ideas")
        st.caption("⚠️ Research & educational output only — **not investment advice**. Expressed via "
                   "liquid ETF proxies; ranked Sortino-style (expected return ÷ downside vol).")
        if overrides_active:
            st.info("⚙️ **Overrides active** — ideas below reflect your sidebar risk/ranking settings, "
                    "not the default risk budget.")
        if "forecast" not in oos:
            st.info("No forecasts yet. Run `python scripts/train_and_backtest.py`.")
        else:
            store = MarketStore()
            try:
                rec = run_recommend(store, oos["forecast"], cfg_eff)
                spots = {}
                for hr in rec.horizons.values():
                    for s in hr.ensemble["symbol"]:
                        if s not in spots:
                            p = store.try_price(s)
                            if p is not None:
                                spots[s] = float(p.iloc[-1])
            finally:
                store.close()

            if rec.is_empty:
                st.info("No qualifying ideas at the current conviction threshold.")
            else:
                hsel = st.radio("Horizon", list(rec.horizons), horizontal=True,
                                format_func=lambda h: _HNAME.get(h, h))
                hr = rec.horizons[hsel]
                st.caption(f"As of {rec.asof}")

                ens = hr.ensemble
                if ens.empty:
                    st.info("No qualifying ideas for this horizon.")
                else:
                    show = ens[["symbol", "direction", "conviction", "exp_ret", "idea_score",
                                "asset_class", "agree"]].copy()
                    show.insert(0, "rank", range(1, len(show) + 1))
                    show["direction"] = show["direction"].map({1: "🟢 long", -1: "🔴 short"})
                    show["exp_ret"] = (show["exp_ret"] * 100).round(2)
                    show["agree"] = show["agree"].map({True: "✓ both", False: "—"})
                    st.markdown("**Ranked ideas (ensemble of linear + GBM)**")
                    st.dataframe(show, hide_index=True, width="stretch", column_config={
                        "conviction": st.column_config.NumberColumn("conviction", format="%.2f"),
                        "exp_ret": st.column_config.NumberColumn("exp. ret %", format="%.2f"),
                        "idea_score": st.column_config.NumberColumn("risk-adj score", format="%.2f"),
                    })

                    st.markdown(f"**Risk-budgeted portfolio** · ${hr.summary['notional']:,.0f} notional")
                    alloc = hr.allocation
                    if alloc.empty:
                        st.caption("No positions after applying the risk-budget caps.")
                    else:
                        disp = alloc.copy()
                        disp["weight"] = (disp["weight"] * 100).round(1)
                        st.dataframe(disp, hide_index=True, width="stretch", column_config={
                            "weight": st.column_config.NumberColumn("weight %", format="%.1f"),
                            "dollars": st.column_config.NumberColumn("$ amount", format="$%.0f"),
                        })
                        bar = go.Figure(go.Bar(
                            x=alloc["symbol"], y=(alloc["weight"] * 100),
                            marker_color=[_PAL["up"] if d == "long" else _PAL["down"] for d in alloc["direction"]]))
                        bar.update_layout(title="Allocation (%, signed)", height=300,
                                          margin=dict(t=40, b=20, l=10, r=10))
                        st.plotly_chart(theme.style_fig(bar), width="stretch")
                        s = hr.summary
                        cls = " · ".join(f"{k} {v*100:.0f}%" for k, v in s["by_class"].items())
                        st.caption(f"Gross {s['gross']*100:.0f}% (cap 100%) · net {s['net']*100:+.0f}% · "
                                   f"by class: {cls} (cap 60%) · per-position cap 15% · {s['n_positions']} positions")

                    with st.expander("Per-model ideas (linear vs GBM)"):
                        mcols = st.columns(len(rec.models))
                        for col, m in zip(mcols, rec.models):
                            pm = hr.per_model.get(m, pd.DataFrame())
                            col.markdown(f"**{m}**")
                            if pm.empty:
                                col.caption("— none —")
                            else:
                                t = pm[["symbol", "direction", "conviction", "idea_score"]].head(8).copy()
                                t["direction"] = t["direction"].map({1: "long", -1: "short"})
                                col.dataframe(t, hide_index=True, width="stretch")

                    # structured-payoff illustration for the top idea
                    top = ens.iloc[0]
                    spot = spots.get(top["symbol"])
                    if spot:
                        otm = float(_CFG.recommend.get("payoff_otm_pct", 0.03))
                        pay = payoff_mod.illustrate(spot, int(top["direction"]), otm_pct=otm)
                        st.markdown(f"**Illustrative payoff — {top['symbol']} ({pay.label})**")
                        fig = go.Figure(go.Scatter(x=pay.x, y=pay.y, mode="lines",
                                                   line=dict(color=_PAL["cyan"], width=2),
                                                   fill="tozeroy", fillcolor="rgba(14,165,233,0.07)"))
                        fig.add_hline(y=0, line_dash="dot", line_color=_PAL["muted"])
                        for name, xv in pay.markers.items():
                            fig.add_vline(x=xv, line_dash="dot", line_color=_PAL["border"],
                                          annotation_text=name, annotation_position="top")
                        fig.update_layout(height=320, margin=dict(t=30, b=20, l=10, r=10),
                                          xaxis_title=f"{top['symbol']} price at expiry",
                                          yaxis_title="P&L (per unit, illustrative)")
                        st.plotly_chart(theme.style_fig(fig), width="stretch")
                        st.caption("Illustration of how the directional view *could* be expressed with "
                                   "options — payoff shape at expiry only, no pricing/greeks. Not a recommendation.")

    # -- Strategy Lab ----------------------------------------------------
    with strategy_tab:
        _render_strategy_lab(signal_names)

    # -- Data Health -----------------------------------------------------
    with health:
        st.subheader("Provenance")
        prov = pd.DataFrame(data["provenance"])
        if not prov.empty:
            cols = [c for c in ["key", "source", "kind", "status", "start_date",
                                "end_date", "n_rows", "pull_ts"] if c in prov.columns]
            st.dataframe(prov[cols], width="stretch", hide_index=True)
            st.caption(f"Last pull: {prov['pull_ts'].max()}")
            # best-effort, single-source feeds (GDELT news tone) can be rate-limited; surface
            # that as "unavailable" rather than alarming — nothing else depends on them.
            if "status" in prov.columns:
                unavail = prov[prov["status"] == "unavailable"]
                if not unavail.empty:
                    names = ", ".join(unavail["key"].astype(str))
                    st.info(f"ℹ️ Best-effort source(s) currently **unavailable**: {names}. "
                            "GDELT news tone is single-source and occasionally rate-limited; "
                            "the `news_tone` signal is simply omitted until it returns — no impact "
                            "on the stress index, forecasts, or trade ideas.")

        st.subheader("QA flags")
        flags = pd.DataFrame(data["flags"])
        if flags.empty:
            st.success("No QA flags raised.")
        else:
            errs = (flags["severity"] == "error").sum()
            if errs:
                st.error(f"{errs} error-severity flag(s) — those series are withheld from signals.")
            st.dataframe(flags, width="stretch", hide_index=True)


main()
