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
from macro_advisor.signals import compute_all
from macro_advisor.storage import remote
from macro_advisor.stress import StressResult, compute_stress

_CFG = load_config()
_TTL = int((_CFG.remote or {}).get("app_cache_ttl_min", 30)) * 60

_BAND_COLORS = {
    "calm": "#2e7d32", "normal": "#9e9d24", "elevated": "#f9a825",
    "stressed": "#ef6c00", "crisis": "#c62828",
}
_DIR_BADGE = {"risk_on": "🟢 on", "risk_off": "🔴 off", "neutral": "⚪ neutral"}


def _hf_token() -> str | None:
    """Optional HF token from Streamlit secrets (not needed for the public cache repo)."""
    try:
        return st.secrets.get("HF_TOKEN")  # type: ignore[no-any-return]
    except Exception:
        return None


@st.cache_resource(ttl=_TTL, show_spinner="Fetching data cache from Hugging Face Hub…")
def _sync_cache() -> str:
    """Ensure a local data cache exists, downloading from HF Hub when absent (cloud boot).

    Cached as a resource so it runs once per container; intraday freshness is delivered by
    the cron's Streamlit-reboot step, which restarts the container and re-triggers this.
    """
    try:
        downloaded = remote.ensure_cache(_CFG, token=_hf_token())
        return "downloaded from HF Hub" if downloaded else "using local cache"
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


def _gauge(stress: StressResult) -> go.Figure:
    color = _BAND_COLORS.get(stress.label, "#607d8b")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stress.level,
        number={"suffix": " / 100", "font": {"size": 40}},
        title={"text": f"Market Stress — <b>{stress.label.upper()}</b>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#e8f5e9"},
                {"range": [30, 55], "color": "#f9fbe7"},
                {"range": [55, 70], "color": "#fff8e1"},
                {"range": [70, 85], "color": "#fff3e0"},
                {"range": [85, 100], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def _decomposition(stress: StressResult) -> go.Figure:
    comps = stress.components
    names = [c.component for c in comps]
    vals = [c.contribution for c in comps]
    colors = ["#c62828" if v >= 0 else "#2e7d32" for v in vals]  # +stress red, -stress green
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    fig.update_layout(
        title=f"Component contribution to latent ({stress.latent:+.3f})",
        height=320, margin=dict(t=50, b=20, l=10, r=10),
        xaxis_title="weight × stress  (→ risk-off / ← risk-on)",
    )
    return fig


def _history(stress: StressResult) -> go.Figure:
    h = stress.history.tail(504)  # ~2y
    fig = go.Figure(go.Scatter(x=h.index, y=h.values, mode="lines", line=dict(color="#37474f")))
    for edge, col in [(30, "#2e7d32"), (55, "#9e9d24"), (70, "#f9a825"), (85, "#c62828")]:
        fig.add_hline(y=edge, line_dash="dot", line_color=col, opacity=0.4)
    fig.update_layout(title="Stress history (~2y)", height=320,
                      margin=dict(t=50, b=20, l=10, r=10), yaxis_range=[0, 100])
    return fig


def main() -> None:
    st.set_page_config(page_title="MacroAdvisor", layout="wide", page_icon="📊")
    st.title("📊 MacroAdvisor")
    st.caption("Evidence-based market regime & stress read — Phase 1 (read-only).  "
               "Not investment advice.")

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

    overview, signals_tab, health = st.tabs(["Overview", "Signals", "Data Health"])

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

    # -- Data Health -----------------------------------------------------
    with health:
        st.subheader("Provenance")
        prov = pd.DataFrame(data["provenance"])
        if not prov.empty:
            cols = [c for c in ["key", "source", "kind", "status", "start_date",
                                "end_date", "n_rows", "pull_ts"] if c in prov.columns]
            st.dataframe(prov[cols], width="stretch", hide_index=True)
            st.caption(f"Last pull: {prov['pull_ts'].max()}")

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
