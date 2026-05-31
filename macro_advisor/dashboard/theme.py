"""Dashboard visual theme — "deep-space navy + neon glow" (tech/HUD look).

Centralizes every visual asset so ``app.py`` stays about logic, not styling:
  * ``PALETTE``        — the color constants.
  * ``BAND_COLORS``    — stress-band colors (calm→crisis), neon-tuned for dark.
  * ``GAUGE_STEPS``    — translucent dark band fills for the stress gauge.
  * ``inject_css``     — one-shot CSS (fonts, gradient background, glowing cards,
                          tab highlight, sidebar/button polish).
  * ``apply_default``  — registers a Plotly template and makes it the default so
                          every chart matches (transparent bg, neon colorway, grid).
  * ``style_fig``      — per-figure touch-up (transparent bg + font), for the few
                          figures that want it explicitly.

Pure styling: no data/logic here. Selectors are ``data-testid``/role based and
additive, so a selector miss simply doesn't apply (never breaks the app).
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------- palette
PALETTE = {
    "bg": "#0b1020",            # deep navy (not pure black)
    "bg2": "#0d1428",           # gradient end
    "panel": "#131a2e",         # cards / sidebar / tab panels
    "panel2": "#1a2238",        # raised panel
    "border": "#233049",        # hairline borders
    "grid": "#1e2940",          # chart gridlines
    "glow": "rgba(34,211,238,0.22)",
    "cyan": "#22d3ee",          # primary accent
    "blue": "#3b82f6",
    "violet": "#818cf8",
    "teal": "#2dd4bf",
    "text": "#e6edf6",          # near-white
    "muted": "#8aa0b6",
    "up": "#22c55e",            # risk-on / positive
    "down": "#f43f5e",          # risk-off / negative
    "warn": "#f59e0b",
    "orange": "#fb923c",
}

# stress bands (calm → crisis), neon-tuned to read well on the navy background
BAND_COLORS = {
    "calm": "#22c55e",
    "normal": "#a3e635",
    "elevated": "#f59e0b",
    "stressed": "#fb923c",
    "crisis": "#f43f5e",
}

# gauge zone fills — translucent dark tints (the old pastel fills clashed on dark)
GAUGE_STEPS = [
    {"range": [0, 30], "color": "rgba(34,197,94,0.14)"},
    {"range": [30, 55], "color": "rgba(163,230,53,0.12)"},
    {"range": [55, 70], "color": "rgba(245,158,11,0.13)"},
    {"range": [70, 85], "color": "rgba(251,146,60,0.15)"},
    {"range": [85, 100], "color": "rgba(244,63,94,0.16)"},
]

# series color cycle for multi-line charts (equity curves, etc.)
COLORWAY = [PALETTE["cyan"], PALETTE["blue"], PALETTE["violet"], PALETTE["teal"],
            PALETTE["warn"], PALETTE["up"], PALETTE["down"], PALETTE["orange"]]

_FONT_BODY = "Inter, 'Segoe UI', system-ui, sans-serif"
_FONT_HEAD = "'Space Grotesk', Inter, sans-serif"
_FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', ui-monospace, monospace"


# --------------------------------------------------------------------------- CSS
def _css() -> str:
    p = PALETTE
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{
  --ma-cyan: {p['cyan']}; --ma-blue: {p['blue']}; --ma-border: {p['border']};
  --ma-panel: {p['panel']}; --ma-glow: {p['glow']}; --ma-muted: {p['muted']};
}}

/* app background: deep-navy radial wash + faint grid */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(34,211,238,0.06), transparent 60%),
    radial-gradient(1000px 700px at 100% 0%, rgba(59,130,246,0.07), transparent 55%),
    linear-gradient(180deg, {p['bg']} 0%, {p['bg2']} 100%);
  background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .35;
  background-image:
    linear-gradient(to right, rgba(35,48,73,0.25) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(35,48,73,0.25) 1px, transparent 1px);
  background-size: 44px 44px;
}}
[data-testid="stHeader"] {{ background: transparent; }}

html, body, [class*="st-"], .stMarkdown, p, span, label, div {{ font-family: {_FONT_BODY}; }}

/* headings — geometric + cyan accent bar */
h1, h2, h3, [data-testid="stHeading"] {{ font-family: {_FONT_HEAD}; letter-spacing: .2px; color: {p['text']}; }}
h1 {{
  font-weight: 700;
  background: linear-gradient(92deg, {p['text']} 0%, {p['cyan']} 70%, {p['blue']} 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
h2, h3 {{ position: relative; padding-left: .6rem; }}
h2::before, h3::before {{
  content: ""; position: absolute; left: 0; top: .12em; bottom: .12em; width: 3px; border-radius: 3px;
  background: linear-gradient(180deg, {p['cyan']}, {p['blue']}); box-shadow: 0 0 10px {p['glow']};
}}

/* metric cards — glowing panels with mono numerals */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, {p['panel']} 0%, rgba(19,26,46,0.6) 100%);
  border: 1px solid {p['border']}; border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.35);
}}
[data-testid="stMetric"]:hover {{ border-color: rgba(34,211,238,0.45); box-shadow: 0 0 0 1px var(--ma-glow), 0 8px 28px rgba(0,0,0,0.4); }}
[data-testid="stMetricValue"] {{ font-family: {_FONT_MONO}; color: {p['cyan']}; font-weight: 600; }}
[data-testid="stMetricLabel"] {{ color: {p['muted']}; text-transform: uppercase; letter-spacing: .6px; font-size: .72rem; }}

/* tabs — underline + glow on the active tab */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {p['border']}; }}
[data-testid="stTabs"] button[role="tab"] {{ color: {p['muted']}; font-family: {_FONT_HEAD}; font-weight: 500; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color: {p['cyan']}; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {{
  content: ""; position: absolute; left: 8px; right: 8px; bottom: -1px; height: 2px;
  background: linear-gradient(90deg, {p['cyan']}, {p['blue']}); box-shadow: 0 0 8px {p['glow']}; border-radius: 2px;
}}

/* sidebar — translucent panel with a lit right edge */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(19,26,46,0.92), rgba(13,20,40,0.92));
  border-right: 1px solid {p['border']}; backdrop-filter: blur(6px);
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {p['text']}; }}

/* primary buttons — cyan→blue gradient + hover glow */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(92deg, {p['cyan']}, {p['blue']}); border: 0; color: #04121b; font-weight: 600;
  box-shadow: 0 4px 18px rgba(34,211,238,0.25);
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(1.06); box-shadow: 0 6px 24px rgba(34,211,238,0.4);
}}

/* dataframes — tinted header + hairline frame */
[data-testid="stDataFrame"] {{ border: 1px solid {p['border']}; border-radius: 12px; overflow: hidden; }}
[data-testid="stDataFrame"] thead tr th {{ background: {p['panel2']} !important; color: {p['text']} !important; }}

/* misc: code, dividers, scrollbars */
code, kbd {{ font-family: {_FONT_MONO}; color: {p['cyan']}; }}
hr {{ border-color: {p['border']}; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #1c2740; border-radius: 8px; border: 2px solid {p['bg']}; }}
::-webkit-scrollbar-thumb:hover {{ background: #2a3a5c; }}
</style>
"""


def inject_css() -> None:
    """Inject the dashboard CSS once. Call right after ``st.set_page_config``."""
    st.markdown(_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- Plotly
_TEMPLATE_NAME = "macro"


def plotly_template() -> go.layout.Template:
    p = PALETTE
    return go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["text"], family=_FONT_BODY, size=13),
        title=dict(font=dict(color=p["text"], family=_FONT_HEAD, size=16)),
        colorway=COLORWAY,
        xaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["border"], tickcolor=p["border"]),
        yaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["border"], tickcolor=p["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p["muted"])),
        margin=dict(t=50, b=20, l=10, r=10),
        hoverlabel=dict(bgcolor=p["panel2"], font=dict(color=p["text"], family=_FONT_MONO),
                        bordercolor=p["border"]),
    ))


def apply_default() -> None:
    """Register the template and make charts use it by default (call at import)."""
    pio.templates[_TEMPLATE_NAME] = plotly_template()
    pio.templates.default = f"plotly_dark+{_TEMPLATE_NAME}"


def style_fig(fig: go.Figure) -> go.Figure:
    """Touch-up a single figure: transparent bg + theme font (template already handles most)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family=_FONT_BODY),
    )
    return fig
