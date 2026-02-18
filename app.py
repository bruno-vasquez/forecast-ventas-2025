import os
import io
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl


# Config
st.set_page_config(
    page_title="Forecast Ventas 2025 - Cochabamba",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# ESTILOS DUAL-MODE
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ─────────────────────────────────────────
   LIGHT MODE TOKENS
───────────────────────────────────────── */
:root {
  --bg-base:      #f0f4ff;
  --bg-card:      #ffffff;
  --bg-card2:     #f8faff;
  --bg-sidebar:   #1a1f3a;

  --text-primary:  #0d1333;
  --text-secondary:#3d4575;
  --text-muted:    #7c85b3;
  --text-on-dark:  #e8ecff;

  --border-light:  rgba(99,120,255,0.14);
  --border-card:   rgba(99,120,255,0.20);
  --shadow-card:   0 4px 24px rgba(30,50,180,0.08), 0 1px 4px rgba(30,50,180,0.06);
  --shadow-hover:  0 12px 36px rgba(30,50,180,0.16);

  --accent-blue:   #3d5cff;
  --accent-cyan:   #00c9e0;
  --accent-green:  #00c07a;
  --accent-amber:  #f5a623;
  --accent-red:    #f04b4b;

  --gradient-hero: linear-gradient(135deg, #3d5cff 0%, #00c9e0 100%);
  --gradient-card: linear-gradient(135deg, rgba(61,92,255,0.07) 0%, rgba(0,201,224,0.05) 100%);
  --gradient-btn:  linear-gradient(135deg, #3d5cff 0%, #00c9e0 100%);
  --gradient-dl:   linear-gradient(135deg, #00c07a 0%, #00a868 100%);

  --tab-bg:        rgba(255,255,255,0.85);
  --tab-inactive:  #eef1ff;
  --tab-active-bg: linear-gradient(135deg, rgba(61,92,255,0.12) 0%, rgba(0,201,224,0.10) 100%);
  --tab-active-border: rgba(61,92,255,0.35);

  --chart-bg:      #ffffff;
  --chart-text:    #0d1333;
  --chart-grid:    rgba(61,92,255,0.10);
  --chart-spine:   rgba(61,92,255,0.20);
}

/* ─────────────────────────────────────────
   DARK MODE TOKENS
───────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-base:      #080d1e;
    --bg-card:      rgba(14,20,48,0.90);
    --bg-card2:     rgba(20,28,62,0.85);
    --bg-sidebar:   rgba(8,12,30,0.98);

    --text-primary:  #e8ecff;
    --text-secondary:#a0aad0;
    --text-muted:    #5c6899;
    --text-on-dark:  #e8ecff;

    --border-light:  rgba(99,120,255,0.12);
    --border-card:   rgba(99,120,255,0.18);
    --shadow-card:   0 4px 32px rgba(0,0,0,0.45), 0 1px 6px rgba(0,0,0,0.30);
    --shadow-hover:  0 14px 48px rgba(61,92,255,0.22);

    --gradient-card: linear-gradient(135deg, rgba(61,92,255,0.10) 0%, rgba(0,201,224,0.06) 100%);
    --tab-bg:        rgba(14,20,48,0.80);
    --tab-inactive:  rgba(14,20,48,0.60);
    --tab-active-bg: linear-gradient(135deg, rgba(61,92,255,0.22) 0%, rgba(0,201,224,0.16) 100%);

    --chart-bg:      #0d1330;
    --chart-text:    #c8d0f0;
    --chart-grid:    rgba(99,120,255,0.12);
    --chart-spine:   rgba(99,120,255,0.18);
  }
}

/* ─────────────────────────────────────────
   APP BACKGROUND
───────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 900px 520px at 10% -10%, rgba(61,92,255,0.18), transparent 60%),
    radial-gradient(ellipse 700px 400px at 90% 5%, rgba(0,201,224,0.14), transparent 60%),
    radial-gradient(ellipse 800px 500px at 50% 105%, rgba(0,192,122,0.10), transparent 55%),
    var(--bg-base);
  color: var(--text-primary);
  min-height: 100vh;
}

@media (prefers-color-scheme: dark) {
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 900px 520px at 10% -5%, rgba(61,92,255,0.22), transparent 55%),
      radial-gradient(ellipse 700px 400px at 90% 5%, rgba(0,201,224,0.14), transparent 55%),
      radial-gradient(ellipse 600px 350px at 50% 100%, rgba(0,192,122,0.08), transparent 55%),
      var(--bg-base);
  }
}

/* ─────────────────────────────────────────
   SIDEBAR
───────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border-card);
}

[data-testid="stSidebar"] * {
  color: var(--text-on-dark) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stCaption {
  color: var(--text-muted) !important;
  font-size: 0.80rem !important;
}
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stCheckbox label {
  color: #c8d0ff !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
}
[data-testid="stSidebar"] hr {
  border-color: rgba(99,120,255,0.20) !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input {
  background: rgba(255,255,255,0.06) !important;
  border-color: rgba(99,120,255,0.25) !important;
  color: #e8ecff !important;
  border-radius: 10px !important;
}

/* Sidebar section labels */
[data-testid="stSidebar"] .sidebar-section {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-cyan) !important;
  margin: 1.4rem 0 0.5rem 0;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid rgba(0,201,224,0.20);
}

/* ─────────────────────────────────────────
   MAIN TITLE
───────────────────────────────────────── */
.main-title {
  font-size: 2.8rem;
  font-weight: 800;
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-align: center;
  padding: 1.4rem 0 0.4rem;
  letter-spacing: -0.02em;
  line-height: 1.1;
}

.main-subtitle {
  text-align: center;
  color: var(--text-muted);
  font-size: 1.02rem;
  font-weight: 400;
  margin-bottom: 1.8rem;
  letter-spacing: 0.01em;
}

.subtitle {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--text-primary);
  margin: 1.4rem 0 1rem 0;
  padding-bottom: 0.55rem;
  border-bottom: 2px solid;
  border-image: var(--gradient-hero) 1;
  letter-spacing: -0.01em;
}

/* ─────────────────────────────────────────
   CARDS
───────────────────────────────────────── */
.custom-card {
  background: var(--bg-card);
  background-image: var(--gradient-card);
  border-radius: 18px;
  padding: 1.4rem 1.5rem;
  box-shadow: var(--shadow-card);
  margin: 0.8rem 0;
  border: 1px solid var(--border-card);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
  position: relative;
  overflow: hidden;
}
.custom-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--gradient-hero);
  border-radius: 18px 18px 0 0;
}
.custom-card:hover {
  box-shadow: var(--shadow-hover);
  transform: translateY(-2px);
}
.custom-card h1, .custom-card h2, .custom-card h3, .custom-card h4,
.custom-card p, .custom-card span, .custom-card div {
  color: var(--text-primary) !important;
}
.custom-card h4 {
  font-weight: 700;
  font-size: 1.05rem;
  margin-bottom: 0.6rem;
}

/* ─────────────────────────────────────────
   BADGES
───────────────────────────────────────── */
.badge {
  display: inline-block;
  padding: 0.28rem 0.80rem;
  border-radius: 999px;
  font-size: 0.80rem;
  font-weight: 700;
  margin: 0.20rem 0.20rem 0.20rem 0;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.badge-prophet {
  background: rgba(0,192,122,0.14);
  color: #00c07a;
  border: 1px solid rgba(0,192,122,0.28);
}
.badge-sarimax {
  background: rgba(245,166,35,0.14);
  color: #f5a623;
  border: 1px solid rgba(245,166,35,0.28);
}

/* ─────────────────────────────────────────
   TABS
───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: var(--tab-bg);
  border: 1px solid var(--border-card);
  border-radius: 16px;
  padding: 0.40rem;
  backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
  height: 44px;
  background: var(--tab-inactive);
  border-radius: 12px;
  color: var(--text-secondary);
  font-weight: 700;
  font-size: 0.90rem;
  padding: 0 1.3rem;
  transition: all 0.22s ease;
  border: 1px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover {
  background: rgba(61,92,255,0.09);
  color: var(--text-primary);
}
.stTabs [aria-selected="true"] {
  background: var(--tab-active-bg) !important;
  color: var(--text-primary) !important;
  border-color: var(--tab-active-border) !important;
  box-shadow: 0 4px 16px rgba(61,92,255,0.18);
}

/* ─────────────────────────────────────────
   BUTTONS
───────────────────────────────────────── */
.stButton > button {
  background: var(--gradient-btn);
  color: #ffffff !important;
  border: none;
  border-radius: 12px;
  padding: 0.65rem 1.5rem;
  font-weight: 700;
  font-size: 0.92rem;
  transition: all 0.22s ease;
  letter-spacing: 0.02em;
  box-shadow: 0 4px 16px rgba(61,92,255,0.24);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 28px rgba(61,92,255,0.35);
}
.stButton > button:active {
  transform: translateY(0);
}
.stDownloadButton > button {
  background: var(--gradient-dl);
  color: #ffffff !important;
  border-radius: 12px;
  padding: 0.65rem 1.3rem;
  font-weight: 700;
  border: none;
  box-shadow: 0 4px 14px rgba(0,192,122,0.24);
  transition: all 0.22s ease;
}
.stDownloadButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 26px rgba(0,192,122,0.35);
}

/* ─────────────────────────────────────────
   METRICS
───────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card);
  background-image: var(--gradient-card);
  border: 1px solid var(--border-card);
  border-radius: 16px;
  padding: 1rem 1.2rem !important;
  box-shadow: var(--shadow-card);
  transition: box-shadow 0.22s ease, transform 0.22s ease;
  position: relative;
  overflow: hidden;
}
[data-testid="stMetric"]:hover {
  box-shadow: var(--shadow-hover);
  transform: translateY(-2px);
}
[data-testid="stMetric"]::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--gradient-hero);
  opacity: 0.45;
}
[data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  color: var(--text-primary) !important;
  font-weight: 800 !important;
  font-size: 1.55rem !important;
  letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}

/* ─────────────────────────────────────────
   DATAFRAME
───────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border-card);
  box-shadow: var(--shadow-card);
}
.dataframe thead tr th {
  background: linear-gradient(135deg, #1a2a6c 0%, #3d5cff 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.04em !important;
  padding: 0.75rem 1rem !important;
}
.dataframe tbody tr td {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.88rem !important;
}
.dataframe tbody tr:hover td {
  background: rgba(61,92,255,0.07) !important;
}

/* ─────────────────────────────────────────
   ALERTS / INFO
───────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid var(--border-card) !important;
  font-weight: 500 !important;
}

/* ─────────────────────────────────────────
   HR / SEPARATOR
───────────────────────────────────────── */
hr {
  margin: 1.8rem 0;
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
  opacity: 0.4;
}

/* ─────────────────────────────────────────
   GENERAL TEXT
───────────────────────────────────────── */
.block-container,
.stMarkdown,
.stText,
.stCaption,
.stSubheader,
.stHeader {
  color: var(--text-primary);
}
h1, h2, h3, h4, h5, h6 {
  color: var(--text-primary) !important;
}
p, span, div, label {
  color: var(--text-primary);
}

/* Main block container spacing */
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}

/* Spinner */
[data-testid="stSpinner"] {
  color: var(--accent-blue) !important;
}

/* Number / Select / Date inputs in main area */
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] div {
  color: var(--text-primary) !important;
}

/* Scrollbar polish */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(61,92,255,0.30);
  border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(61,92,255,0.55);
}

/* Footer */
.app-footer {
  text-align: center;
  color: var(--text-muted);
  padding: 1.8rem 0;
  font-size: 0.88rem;
  font-weight: 500;
  border-top: 1px solid var(--border-light);
  margin-top: 1.5rem;
}
.app-footer strong {
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>
""",
    unsafe_allow_html=True,
)

# Paths (repo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "modelos")
PROPHET_PATH = os.path.join(MODELS_DIR, "prophet_model.joblib")
SARIMAX_PATH = os.path.join(MODELS_DIR, "sarimax_model.joblib")

# Feriados 2025
FERIADOS_2025 = pd.to_datetime(
    [
        "2025-01-01", "2025-01-22", "2025-03-03",
        "2025-03-04", "2025-04-18", "2025-05-01",
        "2025-06-21", "2025-08-06", "2025-08-16",
        "2025-09-14", "2025-11-02", "2025-12-25",
    ]
)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def is_prophet_model(obj) -> bool:
    mod = obj.__class__.__module__.lower()
    name = obj.__class__.__name__.lower()
    return ("prophet" in mod) or ("fbprophet" in mod) or (name == "prophet")

def is_statsmodels_model(obj) -> bool:
    mod = obj.__class__.__module__.lower()
    return "statsmodels" in mod

def build_feriado_anticipado(dates: pd.DatetimeIndex, feriados: pd.DatetimeIndex, anticipacion_dias: int = 7) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    s = pd.Series(0, index=idx, dtype=int)
    for f in pd.to_datetime(feriados):
        rango = pd.date_range(f - timedelta(days=anticipacion_dias), f - timedelta(days=1), freq="D")
        s.loc[s.index.isin(rango)] = 1
    return s

def apply_operational_zeros(pred: pd.Series, feriados: pd.DatetimeIndex) -> pd.Series:
    pred = pred.copy()
    pred.index = pd.to_datetime(pred.index)
    mask = (pred.index.dayofweek == 6) | (pred.index.isin(pd.to_datetime(feriados)))
    pred.loc[mask] = 0
    pred.loc[pred < 0] = 0
    return pred

def resample_view(s: pd.Series, vista: str) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if vista == "Semanal":
        return s.resample("W").sum()
    if vista == "Mensual":
        return s.resample("MS").sum()
    return s


def _apply_chart_style(fig, ax_list, title="", xlabel="Fecha", ylabel="Volumen (HL)"):
    """
    Apply a consistent, dual-mode-safe chart style.
    Uses a crisp dark background for charts regardless of OS theme,
    ensuring all text and lines are always visible.
    """
    CHART_BG     = "#0e1428"
    CHART_FG     = "#c8d0f0"
    GRID_C       = (0.38, 0.42, 0.68, 0.18)
    SPINE_C      = (0.38, 0.42, 0.68, 0.25)
    TITLE_C      = "#e8ecff"

    fig.patch.set_facecolor(CHART_BG)
    for ax in (ax_list if isinstance(ax_list, (list, tuple)) else [ax_list]):
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors=CHART_FG, labelsize=10)
        ax.xaxis.label.set_color(CHART_FG)
        ax.yaxis.label.set_color(CHART_FG)
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
            spine.set_color(SPINE_C)
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.6, color=GRID_C)

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, fontweight="600", color=CHART_FG)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight="600", color=CHART_FG)
        if title:
            ax.set_title(title, fontsize=15, fontweight="800", pad=16, color=TITLE_C)

    return CHART_FG, TITLE_C


def plot_with_ci(index, mean, low=None, up=None, title="", color="#3d5cff"):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fg, _ = _apply_chart_style(fig, ax, title=title)

    ax.plot(index, mean, label="Predicción", linewidth=2.4, color=color,
            zorder=3, marker="o", markersize=3.2, alpha=0.95)

    if low is not None and up is not None:
        ax.fill_between(index, low, up, alpha=0.16, color=color, label="IC 95%", zorder=2)

    leg = ax.legend(loc="best", frameon=True, fontsize=10, facecolor="#1a2040", edgecolor="none")
    for text in leg.get_texts():
        text.set_color(fg)

    plt.xticks(rotation=40)
    plt.tight_layout()
    return fig


# ─── HEADER ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">📈 Forecast de Ventas 2025</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Comparación <strong>Prophet</strong> vs <strong>SARIMAX</strong> &nbsp;·&nbsp; Cochabamba, Bolivia</div>',
    unsafe_allow_html=True,
)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">📅 Parámetros del Modelo</div>', unsafe_allow_html=True)
    anticipacion_dias = st.number_input("Días anticipación feriado", min_value=1, max_value=30, value=7, step=1)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">📊 Visualización</div>', unsafe_allow_html=True)
    vista = st.selectbox("Agregación", ["Diario", "Semanal", "Mensual"])
    acumulado = st.checkbox("Mostrar acumulado", value=False)
    mostrar_intervalos = st.checkbox("Intervalos de confianza", value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">📅 Rango de Fechas</div>', unsafe_allow_html=True)
    start_date = st.date_input("Desde", value=pd.to_datetime("2025-01-01").date())
    end_date = st.date_input("Hasta", value=pd.to_datetime("2025-12-31").date())

    st.markdown("---")
    st.caption("ℹ️ Ceros operativos (domingos / feriados) activados por defecto.")

if pd.to_datetime(end_date) < pd.to_datetime(start_date):
    st.error("⚠️ Rango inválido: la fecha de inicio debe ser anterior a la fecha de fin.")
    st.stop()

dates_2025 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
mask_range = (dates_2025 >= pd.to_datetime(start_date)) & (dates_2025 <= pd.to_datetime(end_date))
dates_view = dates_2025[mask_range]

# ─── LOAD & PREDICT ────────────────────────────────────────────────────────
with st.spinner("🔄 Cargando modelos desde el repositorio…"):
    if not os.path.exists(PROPHET_PATH):
        st.error(f"No se encontró: {PROPHET_PATH}")
        st.stop()
    if not os.path.exists(SARIMAX_PATH):
        st.error(f"No se encontró: {SARIMAX_PATH}")
        st.stop()

    m1 = load_model(PROPHET_PATH)
    m2 = load_model(SARIMAX_PATH)

    if is_prophet_model(m1) and is_statsmodels_model(m2):
        prophet_model, sarimax_model = m1, m2
        swapped = False
    elif is_prophet_model(m2) and is_statsmodels_model(m1):
        prophet_model, sarimax_model = m2, m1
        swapped = True
    else:
        st.error("No pude identificar claramente Prophet y SARIMAX en los .joblib del repositorio.")
        st.write("prophet_model.joblib:", type(m1), "module:", m1.__class__.__module__)
        st.write("sarimax_model.joblib:", type(m2), "module:", m2.__class__.__module__)
        st.stop()

    if swapped:
        st.warning("⚠️ Los archivos .joblib estaban cruzados. Se aplicó swap automático (Prophet ↔ SARIMAX).")

    exog_2025 = pd.DataFrame(index=dates_2025)
    exog_2025["ES_FERIADO_ANTICIPADO"] = build_feriado_anticipado(exog_2025.index, FERIADOS_2025, anticipacion_dias).values

    future = pd.DataFrame({"ds": dates_2025})
    future["ES_FERIADO_ANTICIPADO"] = exog_2025["ES_FERIADO_ANTICIPADO"].values
    fc_p = prophet_model.predict(future)

    pred_prophet = pd.Series(fc_p["yhat"].values, index=dates_2025)
    low_p = pd.Series(fc_p["yhat_lower"].values, index=dates_2025)
    up_p  = pd.Series(fc_p["yhat_upper"].values, index=dates_2025)

    fc_s = sarimax_model.get_forecast(steps=len(exog_2025), exog=exog_2025)
    pred_sarimax = pd.Series(np.asarray(fc_s.predicted_mean), index=dates_2025)
    ci_s = fc_s.conf_int()
    low_s = pd.Series(np.asarray(ci_s.iloc[:, 0]), index=dates_2025)
    up_s  = pd.Series(np.asarray(ci_s.iloc[:, 1]), index=dates_2025)

    pred_prophet = apply_operational_zeros(pred_prophet, FERIADOS_2025)
    pred_sarimax = apply_operational_zeros(pred_sarimax, FERIADOS_2025)
    low_p  = apply_operational_zeros(low_p, FERIADOS_2025)
    up_p   = apply_operational_zeros(up_p, FERIADOS_2025)
    low_s  = apply_operational_zeros(low_s, FERIADOS_2025)
    up_s   = apply_operational_zeros(up_s, FERIADOS_2025)

    p_view  = resample_view(pred_prophet, vista)
    s_view  = resample_view(pred_sarimax, vista)
    lp_view = resample_view(low_p, vista)
    up_view = resample_view(up_p, vista)
    ls_view = resample_view(low_s, vista)
    us_view = resample_view(up_s, vista)

    if acumulado:
        p_view, s_view = p_view.cumsum(), s_view.cumsum()
        lp_view, up_view = lp_view.cumsum(), up_view.cumsum()
        ls_view, us_view = ls_view.cumsum(), us_view.cumsum()

# ─── TABS ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊  Resumen", "🔄  Comparación", "🔎  Detalle", "📈  Estadísticas"]
)

# ── TAB 1: RESUMEN ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<h2 class="subtitle">Resumen Ejecutivo</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Prophet",   f"{pred_prophet.sum():,.0f} HL")
    col2.metric("🟠 SARIMAX",   f"{pred_sarimax.sum():,.0f} HL")

    diff = pred_prophet.sum() - pred_sarimax.sum()
    delta_pct = (diff / pred_sarimax.sum() * 100) if pred_sarimax.sum() != 0 else 0.0
    col3.metric("📊 Diferencia", f"{abs(diff):,.0f} HL", delta=f"{delta_pct:+.1f}%")
    col4.metric("📈 Promedio",   f"{((pred_prophet.sum() + pred_sarimax.sum()) / 2):,.0f} HL")

    st.markdown("<hr/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Interpretación")
        if pred_prophet.sum() > pred_sarimax.sum():
            pct = ((pred_prophet.sum() - pred_sarimax.sum()) / max(pred_sarimax.sum(), 1e-9)) * 100
            st.info(f"Prophet proyecta **{pct:.1f}%** más volumen que SARIMAX para el período seleccionado.")
        else:
            pct = ((pred_sarimax.sum() - pred_prophet.sum()) / max(pred_prophet.sum(), 1e-9)) * 100
            st.info(f"SARIMAX proyecta **{pct:.1f}%** más volumen que Prophet para el período seleccionado.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 📅 Información del Análisis")
        st.write(f"**Período:** {start_date} → {end_date}")
        st.write(f"**Vista:** {vista}")
        st.write(f"**Ceros operativos (Prophet):** {(pred_prophet == 0).sum()} días")
        st.write(f"**Ceros operativos (SARIMAX):** {(pred_sarimax == 0).sum()} días")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Comparación Visual 2025")

    idx_common = p_view.index.intersection(s_view.index)
    df_mean = pd.DataFrame(
        {"Prophet": p_view.loc[idx_common], "SARIMAX": s_view.loc[idx_common]},
        index=idx_common,
    )

    FG = "#c8d0f0"
    CHART_BG = "#0e1428"
    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=FG, labelsize=10)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)
        spine.set_color((0.38, 0.42, 0.68, 0.30))
    ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6, color=(0.38, 0.42, 0.68, 0.18))

    ax.plot(df_mean.index, df_mean["Prophet"], label="Prophet",
            linewidth=3, color="#00c07a", marker="o", markersize=4, zorder=3)
    ax.plot(df_mean.index, df_mean["SARIMAX"], label="SARIMAX",
            linewidth=3, color="#f5a623", marker="s", markersize=4, zorder=3)
    ax.fill_between(df_mean.index, df_mean["Prophet"], df_mean["SARIMAX"],
                    alpha=0.10, color="#3d5cff", zorder=2)

    ax.set_title("Prophet vs SARIMAX — Comparación Anual 2025",
                 fontsize=16, fontweight="800", pad=18, color="#e8ecff")
    ax.set_xlabel("Fecha", fontsize=12, fontweight="600", color=FG)
    ax.set_ylabel("Volumen (HL)", fontsize=12, fontweight="600", color=FG)

    leg = ax.legend(loc="best", frameon=True, fontsize=11, facecolor="#1a2040", edgecolor="none")
    for text in leg.get_texts():
        text.set_color(FG)

    plt.xticks(rotation=40)
    plt.tight_layout()
    st.pyplot(fig)

# ── TAB 2: COMPARACIÓN ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<h2 class="subtitle">Comparación Detallada</h2>', unsafe_allow_html=True)

    df_compare = pd.DataFrame({
        "Prophet": pred_prophet.loc[dates_view],
        "SARIMAX": pred_sarimax.loc[dates_view],
    })
    df_compare["Diff"] = df_compare["Prophet"] - df_compare["SARIMAX"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Prophet (rango)", f"{df_compare['Prophet'].sum():,.0f} HL")
    col2.metric("SARIMAX (rango)", f"{df_compare['SARIMAX'].sum():,.0f} HL")
    col3.metric("Diferencia",      f"{df_compare['Diff'].sum():,.0f} HL")

    st.markdown("---")

    FG = "#c8d0f0"
    CHART_BG = "#0e1428"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor(CHART_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors=FG, labelsize=10)
        for spine in ax.spines.values():
            spine.set_alpha(0.25)
            spine.set_color((0.38, 0.42, 0.68, 0.30))
        ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.6, color=(0.38, 0.42, 0.68, 0.18))

    ax1.plot(df_compare.index, df_compare["Prophet"], label="Prophet",
             linewidth=2.5, color="#00c07a", marker="o", markersize=3)
    ax1.plot(df_compare.index, df_compare["SARIMAX"], label="SARIMAX",
             linewidth=2.5, color="#f5a623", marker="s", markersize=3)
    ax1.set_title("Comparación en Rango Seleccionado", fontsize=15, fontweight="800", color="#e8ecff", pad=14)
    ax1.set_ylabel("Volumen (HL)", fontsize=12, fontweight="600", color=FG)
    leg1 = ax1.legend(loc="best", frameon=True, fontsize=10, facecolor="#1a2040", edgecolor="none")
    for t in leg1.get_texts(): t.set_color(FG)

    colors = np.where(df_compare["Diff"] >= 0, "#00c07a", "#f04b4b")
    ax2.bar(df_compare.index, df_compare["Diff"], color=colors, alpha=0.75, width=0.8)
    ax2.axhline(y=0, color="#5c6899", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_title("Diferencias Diarias (Prophet − SARIMAX)", fontsize=15, fontweight="800", color="#e8ecff", pad=14)
    ax2.set_xlabel("Fecha", fontsize=12, fontweight="600", color=FG)
    ax2.set_ylabel("Diferencia (HL)", fontsize=12, fontweight="600", color=FG)
    ax2.grid(True, alpha=0.25, axis="y", color=(0.38, 0.42, 0.68, 0.18))

    plt.xticks(rotation=40)
    fig.subplots_adjust(hspace=0.38)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### 📅 Resumen Mensual")
    mensual = df_compare.resample("MS").sum()
    st.dataframe(mensual.style.format("{:,.0f}"), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        csv = df_compare.reset_index().rename(columns={"index": "fecha"}).to_csv(index=False).encode("utf-8")
        st.download_button("📄 Descargar CSV", csv, "forecast_2025.csv", "text/csv", use_container_width=True)

    with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_compare.reset_index().rename(columns={"index": "fecha"}).to_excel(writer, sheet_name="Diario", index=False)
            mensual.reset_index().rename(columns={"index": "mes"}).to_excel(writer, sheet_name="Mensual", index=False)
        st.download_button("📊 Descargar Excel", output.getvalue(), "forecast_2025.xlsx", use_container_width=True)

# ── TAB 3: DETALLE ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<h2 class="subtitle">Detalle por Modelo</h2>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🟢 Prophet")
        st.markdown('<span class="badge badge-prophet">Machine Learning</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Total",   f"{pred_prophet.sum():,.0f} HL")
        c2.metric("Promedio",f"{pred_prophet.mean():,.1f} HL")
        st.markdown("---")
        if mostrar_intervalos:
            fig = plot_with_ci(p_view.index, p_view.values, lp_view.values, up_view.values,
                               f"Prophet — Vista {vista}", color="#00c07a")
        else:
            fig = plot_with_ci(p_view.index, p_view.values,
                               title=f"Prophet — Vista {vista}", color="#00c07a")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🟠 SARIMAX")
        st.markdown('<span class="badge badge-sarimax">Estadístico</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Total",   f"{pred_sarimax.sum():,.0f} HL")
        c2.metric("Promedio",f"{pred_sarimax.mean():,.1f} HL")
        st.markdown("---")
        if mostrar_intervalos:
            fig = plot_with_ci(s_view.index, s_view.values, ls_view.values, us_view.values,
                               f"SARIMAX — Vista {vista}", color="#f5a623")
        else:
            fig = plot_with_ci(s_view.index, s_view.values,
                               title=f"SARIMAX — Vista {vista}", color="#f5a623")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 4: ESTADÍSTICAS ────────────────────────────────────────────────────
with tab4:
    st.markdown('<h2 class="subtitle">Análisis Estadístico</h2>', unsafe_allow_html=True)

    p = pd.Series(pred_prophet.values, index=pred_prophet.index)
    s = pd.Series(pred_sarimax.values, index=pred_sarimax.index)

    stats_df = pd.DataFrame({
        "Métrica": ["Media", "Mediana", "Desv. Est.", "Mínimo", "Máximo", "Q1", "Q3"],
        "Prophet": [p.mean(), p.median(), p.std(), p.min(), p.max(), p.quantile(0.25), p.quantile(0.75)],
        "SARIMAX": [s.mean(), s.median(), s.std(), s.min(), s.max(), s.quantile(0.25), s.quantile(0.75)],
    })
    st.dataframe(stats_df.style.format({"Prophet": "{:.2f}", "SARIMAX": "{:.2f}"}), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    FG = "#c8d0f0"
    CHART_BG = "#0e1428"

    with col1:
        st.markdown("#### 📊 Distribución de Volúmenes")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors=FG, labelsize=10)
        for spine in ax.spines.values():
            spine.set_alpha(0.25); spine.set_color((0.38, 0.42, 0.68, 0.30))
        ax.grid(True, alpha=0.28, axis="y", linestyle="--", linewidth=0.6, color=(0.38, 0.42, 0.68, 0.18))

        ax.hist(p[p > 0], bins=30, alpha=0.65, color="#00c07a", label="Prophet", edgecolor="#0e1428", linewidth=0.5)
        ax.hist(s[s > 0], bins=30, alpha=0.65, color="#f5a623", label="SARIMAX", edgecolor="#0e1428", linewidth=0.5)
        ax.set_title("Distribución de Volúmenes (días activos)", fontsize=13, fontweight="800", color="#e8ecff", pad=12)
        ax.set_xlabel("Volumen (HL)", fontsize=11, fontweight="600", color=FG)
        ax.set_ylabel("Frecuencia", fontsize=11, fontweight="600", color=FG)
        leg = ax.legend(loc="best", frameon=True, fontsize=10, facecolor="#1a2040", edgecolor="none")
        for t in leg.get_texts(): t.set_color(FG)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### 📈 Correlación entre Modelos")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors=FG, labelsize=10)
        for spine in ax.spines.values():
            spine.set_alpha(0.25); spine.set_color((0.38, 0.42, 0.68, 0.30))
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, color=(0.38, 0.42, 0.68, 0.18))

        sc = ax.scatter(p.values, s.values, alpha=0.45,
                        c=pd.to_datetime(p.index).dayofyear, cmap="plasma", s=46, zorder=3)
        max_val = float(max(p.max(), s.max()))
        ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.8, alpha=0.55, color="#5c6899")

        ax.set_title("Prophet vs SARIMAX — Scatter", fontsize=13, fontweight="800", color="#e8ecff", pad=12)
        ax.set_xlabel("Prophet (HL)", fontsize=11, fontweight="600", color=FG)
        ax.set_ylabel("SARIMAX (HL)", fontsize=11, fontweight="600", color=FG)

        cb = fig.colorbar(sc, ax=ax)
        cb.ax.yaxis.set_tick_params(color=FG, labelsize=9)
        cb.outline.set_edgecolor("none")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
        cb.set_label("Día del año", color=FG, fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

    correlation = p.corr(s)
    st.info(f"📊 **Correlación Pearson entre modelos:** `{correlation:.4f}`")

# ─── FOOTER ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-footer">
  <strong>Forecast de Ventas 2025</strong> &nbsp;·&nbsp; Cochabamba, Bolivia
</div>
""",
    unsafe_allow_html=True,
)