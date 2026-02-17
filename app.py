import os
import io
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# Config
st.set_page_config(
    page_title="Forecast Ventas 2025 - Cochabamba",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Fondo general claro */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 520px at 18% 10%, rgba(59,130,246,0.12), transparent 65%),
    radial-gradient(900px 520px at 82% 12%, rgba(6,182,212,0.10), transparent 65%),
    radial-gradient(1000px 640px at 50% 95%, rgba(16,185,129,0.10), transparent 65%),
    #f7fafc;
  color: #0f172a;
}

/* Sidebar claro */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
  border-right: 1px solid rgba(15,23,42,0.10);
}
[data-testid="stSidebar"] * { color: #0f172a !important; }
[data-testid="stSidebar"] .stMarkdown { color: #0f172a !important; }

/* Título principal */
.main-title {
  font-size: 2.6rem;
  font-weight: 800;
  background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 45%, #06b6d4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  padding: 1.2rem 0 0.6rem 0;
  margin-bottom: 0.4rem;
}
.main-subtitle{
  text-align:center;
  color:#475569;
  font-size: 1.05rem;
  margin-bottom: 1.6rem;
}

/* Secciones */
.subtitle {
  font-size: 1.35rem;
  font-weight: 800;
  color: #0f172a;
  margin: 1.2rem 0 0.8rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 3px solid rgba(59,130,246,0.35);
}

/* Cards */
.custom-card {
  background: rgba(255,255,255,0.90);
  border-radius: 14px;
  padding: 1.2rem 1.2rem;
  box-shadow: 0 10px 24px rgba(15,23,42,0.08);
  margin: 0.8rem 0;
  border: 1px solid rgba(15,23,42,0.10);
}

/* Badges */
.badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 700;
  margin: 0.25rem 0.25rem 0.25rem 0;
  border: 1px solid rgba(15,23,42,0.10);
}
.badge-prophet {
  background: rgba(16,185,129,0.14);
  color: #065f46;
  border-color: rgba(16,185,129,0.25);
}
.badge-sarimax {
  background: rgba(245,158,11,0.16);
  color: #92400e;
  border-color: rgba(245,158,11,0.28);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background-color: rgba(255,255,255,0.60);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 14px;
  padding: 0.35rem;
}
.stTabs [data-baseweb="tab"] {
  height: 46px;
  background-color: rgba(255,255,255,0.92);
  border-radius: 12px;
  color: #334155;
  font-weight: 700;
  padding: 0 1.4rem;
  transition: all 0.25s ease;
  border: 1px solid rgba(15,23,42,0.08);
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(120deg, rgba(59,130,246,0.18) 0%, rgba(6,182,212,0.18) 100%) !important;
  color: #0f172a !important;
  box-shadow: 0 8px 18px rgba(59,130,246,0.18);
  border-color: rgba(59,130,246,0.18) !important;
}

/* Botones */
.stButton>button {
  background: linear-gradient(120deg, #3b82f6 0%, #06b6d4 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 0.6rem 1.4rem;
  font-weight: 700;
  transition: all 0.25s ease;
}
.stButton>button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(59, 130, 246, 0.25);
}

.stDownloadButton>button {
  background: linear-gradient(120deg, #10b981 0%, #059669 100%);
  color: white;
  border-radius: 12px;
  padding: 0.6rem 1.2rem;
  font-weight: 700;
  border: none;
}

/* Dataframe header */
.dataframe thead tr th {
  background: linear-gradient(120deg, rgba(30,58,138,0.95) 0%, rgba(59,130,246,0.95) 100%) !important;
  color: white !important;
  font-weight: 800 !important;
}
.dataframe tbody tr:hover {
  background-color: rgba(59,130,246,0.06) !important;
}

/* Separador */
hr {
  margin: 1.8rem 0;
  border: none;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.60), transparent);
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
    """Siempre activo (por defecto): domingos y feriados -> 0, y no negativos."""
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

def plot_with_ci(index, mean, low=None, up=None, title="", color="#3b82f6"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")

    ax.plot(index, mean, label="Predicción", linewidth=2.6, color=color, zorder=3, marker="o", markersize=3, alpha=0.9)

    if low is not None and up is not None:
        ax.fill_between(index, low, up, alpha=0.18, color=color, label="IC 95%", zorder=2)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=18, color="#0f172a")
    ax.set_xlabel("Fecha", fontsize=12, fontweight="700", color="#334155")
    ax.set_ylabel("Volumen (HL)", fontsize=12, fontweight="700", color="#334155")
    ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6)
    ax.legend(loc="best", frameon=True, shadow=True, fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# Header

st.markdown('<h1 class="main-title">📈 Forecast de Ventas 2025</h1>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Comparación Prophet vs SARIMAX | Cochabamba, Bolivia</div>', unsafe_allow_html=True)

# Sidebar

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    st.markdown("#### 📅 Parámetros del Modelo")
    anticipacion_dias = st.number_input("Días anticipación feriado", min_value=1, max_value=30, value=7, step=1)

    st.markdown("---")
    st.markdown("#### 📊 Visualización")
    vista = st.selectbox("Agregación", ["Diario", "Semanal", "Mensual"])
    acumulado = st.checkbox("Mostrar acumulado", value=False)
    mostrar_intervalos = st.checkbox("Intervalos de confianza", value=True)

    st.markdown("---")
    st.markdown("#### 📅 Rango")
    start_date = st.date_input("Desde", value=pd.to_datetime("2025-01-01").date())
    end_date = st.date_input("Hasta", value=pd.to_datetime("2025-12-31").date())

    st.markdown("---")
    st.caption("ℹ️ *Ceros operativos (domingos/feriados) están activados por defecto.*")

if pd.to_datetime(end_date) < pd.to_datetime(start_date):
    st.error("⚠️ Rango inválido")
    st.stop()

dates_2025 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
mask_range = (dates_2025 >= pd.to_datetime(start_date)) & (dates_2025 <= pd.to_datetime(end_date))
dates_view = dates_2025[mask_range]

# Load & Predict (desde repo)

with st.spinner("🔄 Cargando modelos desde el repo..."):
    # Validación existencia
    if not os.path.exists(PROPHET_PATH):
        st.error(f" No se encontró: {PROPHET_PATH}")
        st.stop()
    if not os.path.exists(SARIMAX_PATH):
        st.error(f" No se encontró: {SARIMAX_PATH}")
        st.stop()

    m1 = load_model(PROPHET_PATH)
    m2 = load_model(SARIMAX_PATH)

    # Swap automático si están cruzados
    if is_prophet_model(m1) and is_statsmodels_model(m2):
        prophet_model, sarimax_model = m1, m2
        swapped = False
    elif is_prophet_model(m2) and is_statsmodels_model(m1):
        prophet_model, sarimax_model = m2, m1
        swapped = True
    else:
        st.error(" No pude identificar claramente Prophet y SARIMAX en los .joblib del repo.")
        st.write("prophet_model.joblib:", type(m1), "module:", m1.__class__.__module__)
        st.write("sarimax_model.joblib:", type(m2), "module:", m2.__class__.__module__)
        st.stop()

    if swapped:
        st.warning(" Tus archivos .joblib están cruzados. Apliqué swap automático (Prophet ↔ SARIMAX).")

    # Exógena 2025
    exog_2025 = pd.DataFrame(index=dates_2025)
    exog_2025["ES_FERIADO_ANTICIPADO"] = build_feriado_anticipado(exog_2025.index, FERIADOS_2025, anticipacion_dias).values

    # Prophet
    future = pd.DataFrame({"ds": dates_2025})
    future["ES_FERIADO_ANTICIPADO"] = exog_2025["ES_FERIADO_ANTICIPADO"].values
    fc_p = prophet_model.predict(future)

    pred_prophet = pd.Series(fc_p["yhat"].values, index=dates_2025)
    low_p = pd.Series(fc_p["yhat_lower"].values, index=dates_2025)
    up_p  = pd.Series(fc_p["yhat_upper"].values, index=dates_2025)

    # SARIMAX (forzar index 2025)
    fc_s = sarimax_model.get_forecast(steps=len(exog_2025), exog=exog_2025)
    pred_sarimax = pd.Series(np.asarray(fc_s.predicted_mean), index=dates_2025)

    ci_s = fc_s.conf_int()
    low_s = pd.Series(np.asarray(ci_s.iloc[:, 0]), index=dates_2025)
    up_s  = pd.Series(np.asarray(ci_s.iloc[:, 1]), index=dates_2025)

    # Ceros operativos SIEMPRE activos (sin checkbox)
    pred_prophet = apply_operational_zeros(pred_prophet, FERIADOS_2025)
    pred_sarimax = apply_operational_zeros(pred_sarimax, FERIADOS_2025)
    low_p = apply_operational_zeros(low_p, FERIADOS_2025)
    up_p  = apply_operational_zeros(up_p, FERIADOS_2025)
    low_s = apply_operational_zeros(low_s, FERIADOS_2025)
    up_s  = apply_operational_zeros(up_s, FERIADOS_2025)

    # Vista agregada
    p_view = resample_view(pred_prophet, vista)
    s_view = resample_view(pred_sarimax, vista)
    lp_view = resample_view(low_p, vista)
    up_view = resample_view(up_p, vista)
    ls_view = resample_view(low_s, vista)
    us_view = resample_view(up_s, vista)

    if acumulado:
        p_view, s_view = p_view.cumsum(), s_view.cumsum()
        lp_view, up_view = lp_view.cumsum(), up_view.cumsum()
        ls_view, us_view = ls_view.cumsum(), us_view.cumsum()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🔄 Comparación", "🔎 Detalle", "📈 Estadísticas"])

with tab1:
    st.markdown('<h2 class="subtitle">Resumen Ejecutivo</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Prophet", f"{pred_prophet.sum():,.0f} HL")
    col2.metric("🟠 SARIMAX", f"{pred_sarimax.sum():,.0f} HL")

    diff = pred_prophet.sum() - pred_sarimax.sum()
    if pred_sarimax.sum() != 0:
        delta_pct = (diff / pred_sarimax.sum()) * 100
    else:
        delta_pct = 0.0
    col3.metric("📊 Diferencia", f"{abs(diff):,.0f} HL", delta=f"{delta_pct:+.1f}%")

    col4.metric("📈 Promedio", f"{((pred_prophet.sum() + pred_sarimax.sum()) / 2):,.0f} HL")

    st.markdown("<hr/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Interpretación")
        if pred_prophet.sum() > pred_sarimax.sum():
            pct = ((pred_prophet.sum() - pred_sarimax.sum()) / max(pred_sarimax.sum(), 1e-9)) * 100
            st.info(f"Prophet proyecta **{pct:.1f}%** más volumen que SARIMAX.")
        else:
            pct = ((pred_sarimax.sum() - pred_prophet.sum()) / max(pred_prophet.sum(), 1e-9)) * 100
            st.info(f"SARIMAX proyecta **{pct:.1f}%** más volumen que Prophet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 📅 Información")
        st.write(f"**Período:** {start_date} → {end_date}")
        st.write(f"**Vista:** {vista}")
        st.write(f"**Ceros operativos (Prophet):** {(pred_prophet == 0).sum()}")
        st.write(f"**Ceros operativos (SARIMAX):** {(pred_sarimax == 0).sum()}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Comparación Visual")

    idx_common = p_view.index.intersection(s_view.index)
    df_mean = pd.DataFrame({"Prophet": p_view.loc[idx_common], "SARIMAX": s_view.loc[idx_common]}, index=idx_common)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")

    ax.plot(df_mean.index, df_mean["Prophet"], label="Prophet", linewidth=3, color="#10b981", marker="o", markersize=4)
    ax.plot(df_mean.index, df_mean["SARIMAX"], label="SARIMAX", linewidth=3, color="#f59e0b", marker="s", markersize=4)
    ax.fill_between(df_mean.index, df_mean["Prophet"], df_mean["SARIMAX"], alpha=0.12, color="#3b82f6")

    ax.set_title("Comparación 2025", fontsize=18, fontweight="bold", pad=18, color="#0f172a")
    ax.set_xlabel("Fecha", fontsize=13, color="#334155")
    ax.set_ylabel("Volumen (HL)", fontsize=13, color="#334155")
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best", frameon=True, shadow=True, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown('<h2 class="subtitle">Comparación Detallada</h2>', unsafe_allow_html=True)

    df_compare = pd.DataFrame(
        {
            "Prophet": pred_prophet.loc[dates_view],
            "SARIMAX": pred_sarimax.loc[dates_view],
        }
    )
    df_compare["Diff"] = df_compare["Prophet"] - df_compare["SARIMAX"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Prophet (rango)", f"{df_compare['Prophet'].sum():,.0f} HL")
    col2.metric("SARIMAX (rango)", f"{df_compare['SARIMAX'].sum():,.0f} HL")
    col3.metric("Diferencia", f"{df_compare['Diff'].sum():,.0f} HL")

    st.markdown("---")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor("white")

    ax1.set_facecolor("#f8fafc")
    ax1.plot(df_compare.index, df_compare["Prophet"], label="Prophet", linewidth=2.5, color="#10b981", marker="o", markersize=3)
    ax1.plot(df_compare.index, df_compare["SARIMAX"], label="SARIMAX", linewidth=2.5, color="#f59e0b", marker="s", markersize=3)
    ax1.set_title("Comparación en Rango", fontsize=16, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Volumen (HL)", fontsize=12, color="#334155")
    ax1.grid(True, alpha=0.30)
    ax1.legend()

    ax2.set_facecolor("#f8fafc")
    colors = np.where(df_compare["Diff"] >= 0, "#10b981", "#ef4444")
    ax2.bar(df_compare.index, df_compare["Diff"], color=colors, alpha=0.70)
    ax2.axhline(y=0, color="#64748b", linestyle="--")
    ax2.set_title("Diferencias (Prophet - SARIMAX)", fontsize=16, fontweight="bold", color="#0f172a")
    ax2.set_xlabel("Fecha", fontsize=12, color="#334155")
    ax2.set_ylabel("Diferencia (HL)", fontsize=12, color="#334155")
    ax2.grid(True, alpha=0.30, axis="y")

    plt.xticks(rotation=45)
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

with tab3:
    st.markdown('<h2 class="subtitle">Detalle por Modelo</h2>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🟢 Prophet")
        st.markdown('<span class="badge badge-prophet">Machine Learning</span>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Total", f"{pred_prophet.sum():,.0f} HL")
        c2.metric("Promedio", f"{pred_prophet.mean():,.1f} HL")

        st.markdown("---")
        if mostrar_intervalos:
            fig = plot_with_ci(p_view.index, p_view.values, lp_view.values, up_view.values, f"Prophet - {vista}", color="#10b981")
        else:
            fig = plot_with_ci(p_view.index, p_view.values, title=f"Prophet - {vista}", color="#10b981")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🟠 SARIMAX")
        st.markdown('<span class="badge badge-sarimax">Estadístico</span>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Total", f"{pred_sarimax.sum():,.0f} HL")
        c2.metric("Promedio", f"{pred_sarimax.mean():,.1f} HL")

        st.markdown("---")
        if mostrar_intervalos:
            fig = plot_with_ci(s_view.index, s_view.values, ls_view.values, us_view.values, f"SARIMAX - {vista}", color="#f59e0b")
        else:
            fig = plot_with_ci(s_view.index, s_view.values, title=f"SARIMAX - {vista}", color="#f59e0b")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<h2 class="subtitle">Análisis Estadístico</h2>', unsafe_allow_html=True)

    # Asegurar Series
    p = pd.Series(pred_prophet.values, index=pred_prophet.index)
    s = pd.Series(pred_sarimax.values, index=pred_sarimax.index)

    stats_df = pd.DataFrame(
        {
            "Métrica": ["Media", "Mediana", "Desv. Est.", "Mínimo", "Máximo", "Q1", "Q3"],
            "Prophet": [p.mean(), p.median(), p.std(), p.min(), p.max(), p.quantile(0.25), p.quantile(0.75)],
            "SARIMAX": [s.mean(), s.median(), s.std(), s.min(), s.max(), s.quantile(0.25), s.quantile(0.75)],
        }
    )

    st.dataframe(stats_df.style.format({"Prophet": "{:.2f}", "SARIMAX": "{:.2f}"}), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Distribución")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("white")
        ax.hist(p[p > 0], bins=30, alpha=0.60, color="#10b981", label="Prophet")
        ax.hist(s[s > 0], bins=30, alpha=0.60, color="#f59e0b", label="SARIMAX")
        ax.set_title("Distribución de Volúmenes", fontsize=14, fontweight="bold", color="#0f172a")
        ax.set_xlabel("Volumen (HL)", color="#334155")
        ax.set_ylabel("Frecuencia", color="#334155")
        ax.legend()
        ax.grid(True, alpha=0.30, axis="y")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### 📈 Correlación")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("white")

        ax.scatter(p.values, s.values, alpha=0.45, c=pd.to_datetime(p.index).dayofyear, cmap="viridis", s=46)
        max_val = float(max(p.max(), s.max()))
        ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=2, alpha=0.7, color="#ef4444")
        ax.set_title("Prophet vs SARIMAX", fontsize=14, fontweight="bold", color="#0f172a")
        ax.set_xlabel("Prophet (HL)", color="#334155")
        ax.set_ylabel("SARIMAX (HL)", color="#334155")
        ax.grid(True, alpha=0.30)
        plt.tight_layout()
        st.pyplot(fig)

    correlation = p.corr(s)
    st.info(f"📊 **Correlación:** {correlation:.4f}")

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    """
<div style='text-align: center; color: #64748b; padding: 1.6rem;'>
  <strong>Forecast de Ventas 2025</strong> | Cochabamba, Bolivia
</div>
""",
    unsafe_allow_html=True,
)
