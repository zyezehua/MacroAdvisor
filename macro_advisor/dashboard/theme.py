"""Dashboard visual theme — "white + sky-blue" (clean, light, tech).

Centralizes every visual asset so ``app.py`` stays about logic, not styling:
  * ``PALETTE``        — the color constants.
  * ``BAND_COLORS``    — stress-band colors (calm→crisis).
  * ``GAUGE_STEPS``    — translucent band fills for the stress gauge.
  * ``inject_css``     — one-shot CSS (fonts, sky-tinted background, rounded cards,
                          pill-button tabs, sidebar/button/input polish).
  * ``apply_default``  — registers a Plotly template and makes it the default so
                          every chart matches (transparent bg, sky colorway, light grid).
  * ``style_fig``      — per-figure touch-up (transparent bg + font).

Pure styling: no data/logic here. Selectors are ``data-testid``/role based and
additive, so a selector miss simply doesn't apply (never breaks the app).
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------- palette
PALETTE = {
    "bg": "#ffffff",            # white
    "bg2": "#e9f3ff",           # gradient end (light sky wash)
    "panel": "#ffffff",         # cards
    "panel2": "#f0f6ff",        # raised / alt panel, dataframe header
    "border": "#cfe0f5",        # hairline borders
    "grid": "#e4eefb",          # chart gridlines
    "glow": "rgba(14,165,233,0.22)",
    "cyan": "#0ea5e9",          # primary accent — sky blue
    "blue": "#2563eb",
    "violet": "#6366f1",
    "teal": "#0891b2",
    "text": "#0f2742",          # dark slate (readable on white)
    "muted": "#5b708f",
    "up": "#16a34a",            # risk-on / positive
    "down": "#e11d48",          # risk-off / negative
    "warn": "#d97706",
    "orange": "#ea580c",
}

# stress bands (calm → crisis)
BAND_COLORS = {
    "calm": "#16a34a",
    "normal": "#65a30d",
    "elevated": "#d97706",
    "stressed": "#ea580c",
    "crisis": "#e11d48",
}

# gauge zone fills — soft translucent tints that read cleanly on white
GAUGE_STEPS = [
    {"range": [0, 30], "color": "rgba(22,163,74,0.12)"},
    {"range": [30, 55], "color": "rgba(101,163,13,0.12)"},
    {"range": [55, 70], "color": "rgba(217,119,6,0.13)"},
    {"range": [70, 85], "color": "rgba(234,88,12,0.14)"},
    {"range": [85, 100], "color": "rgba(225,29,72,0.14)"},
]

# series color cycle for multi-line charts (equity curves, etc.)
COLORWAY = [PALETTE["cyan"], PALETTE["blue"], PALETTE["violet"], PALETTE["teal"],
            PALETTE["warn"], PALETTE["up"], PALETTE["down"], PALETTE["orange"]]

_FONT_BODY = "Inter, 'Segoe UI', system-ui, sans-serif"
_FONT_HEAD = "'Space Grotesk', Inter, sans-serif"
_FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', ui-monospace, monospace"
_SKY = "14,165,233"   # rgb of the sky-blue accent, for rgba() glows


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

/* app background: white with a soft sky wash + very faint grid */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba({_SKY},0.10), transparent 60%),
    radial-gradient(1000px 700px at 100% 0%, rgba(37,99,235,0.07), transparent 55%),
    linear-gradient(180deg, {p['bg']} 0%, {p['bg2']} 100%);
  background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .5;
  background-image:
    linear-gradient(to right, rgba({_SKY},0.05) 1px, transparent 1px),
    linear-gradient(to bottom, rgba({_SKY},0.05) 1px, transparent 1px);
  background-size: 44px 44px;
}}
[data-testid="stHeader"] {{ background: transparent; }}

html, body, [class*="st-"], .stMarkdown, p, span, label, div {{ font-family: {_FONT_BODY}; }}

/* headings — geometric + sky accent bar */
h1, h2, h3, [data-testid="stHeading"] {{ font-family: {_FONT_HEAD}; letter-spacing: .2px; color: {p['text']}; }}
h1 {{
  font-weight: 700;
  background: linear-gradient(92deg, {p['text']} 0%, {p['cyan']} 65%, {p['blue']} 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
h2, h3 {{ position: relative; padding-left: .6rem; }}
h2::before, h3::before {{
  content: ""; position: absolute; left: 0; top: .12em; bottom: .12em; width: 3px; border-radius: 3px;
  background: linear-gradient(180deg, {p['cyan']}, {p['blue']}); box-shadow: 0 0 10px {p['glow']};
}}

/* metric cards — white panels with soft shadow + sky numerals */
[data-testid="stMetric"] {{
  background: {p['panel']};
  border: 1px solid {p['border']}; border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 8px 22px rgba(2,132,199,0.08);
}}
[data-testid="stMetric"]:hover {{ border-color: rgba({_SKY},0.55); box-shadow: 0 0 0 1px var(--ma-glow), 0 10px 26px rgba(2,132,199,0.12); }}
[data-testid="stMetricValue"] {{ font-family: {_FONT_MONO}; color: {p['cyan']}; font-weight: 600; }}
[data-testid="stMetricLabel"] {{ color: {p['muted']}; text-transform: uppercase; letter-spacing: .6px; font-size: .72rem; }}

/* tabs — spaced, rounded "pill" buttons (clearly clickable) */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 10px; border-bottom: none; flex-wrap: wrap; padding: 4px 0 8px;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ background: transparent !important; display: none !important; }}
[data-testid="stTabs"] button[role="tab"] {{
  background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 12px;
  padding: 8px 18px; color: {p['muted']}; font-family: {_FONT_HEAD}; font-weight: 500;
  letter-spacing: .3px; transition: all .15s ease; box-shadow: 0 2px 8px rgba(2,132,199,0.05);
}}
[data-testid="stTabs"] button[role="tab"]:hover {{
  border-color: rgba({_SKY},0.6); color: {p['text']}; box-shadow: 0 0 0 1px var(--ma-glow);
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
  color: #ffffff; border-color: transparent;
  background: linear-gradient(92deg, #38bdf8, {p['cyan']});
  box-shadow: 0 4px 16px {p['glow']};
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{ font-weight: 600; }}

/* sidebar — light sky panel with a lit right edge */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(240,246,255,0.96), rgba(233,243,255,0.96));
  border-right: 1px solid {p['border']}; backdrop-filter: blur(6px);
}}

/* primary buttons — sky gradient */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(92deg, #38bdf8, {p['cyan']}); border: 0; color: #ffffff; font-weight: 600;
  box-shadow: 0 4px 16px rgba({_SKY},0.30);
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(1.04); box-shadow: 0 6px 22px rgba({_SKY},0.42);
}}

/* secondary / download buttons — rounded cards */
button[kind="secondary"], [data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button {{
  background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 11px;
  color: {p['text']}; font-weight: 500; transition: all .15s ease;
}}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stDownloadButton"] button:hover {{
  border-color: rgba({_SKY},0.6); color: {p['cyan']}; box-shadow: 0 0 0 1px var(--ma-glow);
}}

/* inputs (select / multiselect / number / text / file uploader) — rounded cards */
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"],
[data-testid="stNumberInputContainer"], [data-testid="stFileUploaderDropzone"] {{
  background: {p['panel']} !important; border: 1px solid {p['border']} !important;
  border-radius: 11px !important;
}}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within {{
  border-color: {p['cyan']} !important; box-shadow: 0 0 0 1px var(--ma-glow) !important;
}}
[data-baseweb="tag"] {{ border-radius: 8px !important; }}

/* radio options as rounded "pill" cards */
[data-testid="stRadio"] [role="radiogroup"] {{ gap: 8px; flex-wrap: wrap; }}
[data-testid="stRadio"] [role="radiogroup"] > label {{
  background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 11px;
  padding: 7px 14px; margin: 0; transition: all .15s ease;
}}
[data-testid="stRadio"] [role="radiogroup"] > label:hover {{ border-color: rgba({_SKY},0.6); }}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
  border-color: {p['cyan']}; background: rgba({_SKY},0.10); box-shadow: 0 0 0 1px var(--ma-glow);
}}

[data-testid="stSlider"] [role="slider"] {{ box-shadow: 0 0 0 4px rgba({_SKY},0.18); }}

/* dataframes — tinted header + hairline frame */
[data-testid="stDataFrame"] {{ border: 1px solid {p['border']}; border-radius: 12px; overflow: hidden; }}
[data-testid="stDataFrame"] thead tr th {{ background: {p['panel2']} !important; color: {p['text']} !important; }}

/* misc: code, dividers, scrollbars */
code, kbd {{ font-family: {_FONT_MONO}; color: {p['cyan']}; }}
hr {{ border-color: {p['border']}; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #cddcf0; border-radius: 8px; border: 2px solid {p['bg']}; }}
::-webkit-scrollbar-thumb:hover {{ background: #a9c2e0; }}
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
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=p["text"], family=_FONT_MONO),
                        bordercolor=p["border"]),
    ))


def apply_default() -> None:
    """Register the template and make charts use it by default (call at import)."""
    pio.templates[_TEMPLATE_NAME] = plotly_template()
    pio.templates.default = f"plotly_white+{_TEMPLATE_NAME}"


def style_fig(fig: go.Figure) -> go.Figure:
    """Touch-up a single figure: transparent bg + theme font (template already handles most)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family=_FONT_BODY),
    )
    return fig
