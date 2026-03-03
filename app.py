import os
import io
import tempfile
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gdown

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# CONFIG DATA SOURCE (CSV EN GOOGLE DRIVE)
# =============================================================================
DRIVE_FILE_ID = "1xxvsoHcjpkG6uvPfFq82EZNbr2MlOWLI"

TMP_DIR = tempfile.gettempdir()
CSV_LOCAL_PATH = os.path.join(TMP_DIR, "CBN_Cochabamba_2024_LIMPIO.csv")

# =============================================================================
# STREAMLIT PAGE
# =============================================================================
st.set_page_config(
    page_title="Forecast Ventas 2025 - Cochabamba",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ESTILOS
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

:root {
  --bg-base:#f0f4ff; --bg-card:#ffffff; --bg-sidebar:#1a1f3a;
  --text-primary:#0d1333; --text-secondary:#3d4575; --text-muted:#7c85b3; --text-on-dark:#e8ecff;
  --border-card:rgba(99,120,255,0.20);
  --shadow-card:0 4px 24px rgba(30,50,180,0.08), 0 1px 4px rgba(30,50,180,0.06);
  --gradient-hero:linear-gradient(135deg,#3d5cff 0%,#00c9e0 100%);
  --gradient-card:linear-gradient(135deg,rgba(61,92,255,0.07) 0%,rgba(0,201,224,0.05) 100%);
  --gradient-btn:linear-gradient(135deg,#3d5cff 0%,#00c9e0 100%);
  --gradient-dl:linear-gradient(135deg,#00c07a 0%,#00a868 100%);
  --tab-bg:rgba(255,255,255,0.85); --tab-inactive:#eef1ff;
  --tab-active-bg:linear-gradient(135deg,rgba(61,92,255,0.12) 0%,rgba(0,201,224,0.10) 100%);
  --tab-active-border:rgba(61,92,255,0.35);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-base:#080d1e; --bg-card:rgba(14,20,48,0.90); --bg-sidebar:rgba(8,12,30,0.98);
    --text-primary:#e8ecff; --text-secondary:#a0aad0; --text-muted:#5c6899; --text-on-dark:#e8ecff;
    --border-card:rgba(99,120,255,0.18);
    --shadow-card:0 4px 32px rgba(0,0,0,0.45), 0 1px 6px rgba(0,0,0,0.30);
    --gradient-card:linear-gradient(135deg,rgba(61,92,255,0.10) 0%,rgba(0,201,224,0.06) 100%);
    --tab-bg:rgba(14,20,48,0.80); --tab-inactive:rgba(14,20,48,0.60);
    --tab-active-bg:linear-gradient(135deg,rgba(61,92,255,0.22) 0%,rgba(0,201,224,0.16) 100%);
  }
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 900px 520px at 10% -10%, rgba(61,92,255,0.18), transparent 60%),
    radial-gradient(ellipse 700px 400px at 90% 5%, rgba(0,201,224,0.14), transparent 60%),
    radial-gradient(ellipse 800px 500px at 50% 105%, rgba(0,192,122,0.10), transparent 55%),
    var(--bg-base);
  color: var(--text-primary);
  min-height: 100vh;
}

[data-testid="stSidebar"] { background: var(--bg-sidebar) !important; border-right: 1px solid var(--border-card); }
[data-testid="stSidebar"] * { color: var(--text-on-dark) !important; }

.main-title {
  font-size:2.8rem; font-weight:800; background: var(--gradient-hero);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  text-align:center; padding:1.4rem 0 0.4rem; letter-spacing:-0.02em; line-height:1.1;
}
.main-subtitle { text-align:center; color: var(--text-muted); font-size:1.02rem; margin-bottom:1.2rem; }

.custom-card {
  background: var(--bg-card); background-image: var(--gradient-card);
  border-radius:18px; padding:1.2rem 1.4rem; box-shadow: var(--shadow-card);
  margin:0.8rem 0; border:1px solid var(--border-card);
  position:relative; overflow:hidden;
}
.custom-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: var(--gradient-hero); }

.stTabs [data-baseweb="tab-list"] { gap:6px; background: var(--tab-bg); border:1px solid var(--border-card); border-radius:16px; padding:0.40rem; }
.stTabs [data-baseweb="tab"] { height:44px; background: var(--tab-inactive); border-radius:12px; color: var(--text-secondary); font-weight:700; font-size:0.90rem; padding:0 1.3rem; border:1px solid transparent; }
.stTabs [aria-selected="true"] { background: var(--tab-active-bg) !important; color: var(--text-primary) !important; border-color: var(--tab-active-border) !important; }

.stButton > button { background: var(--gradient-btn); color:#fff !important; border:none; border-radius:12px; padding:0.65rem 1.5rem; font-weight:700; }
.stDownloadButton > button { background: var(--gradient-dl); color:#fff !important; border:none; border-radius:12px; padding:0.65rem 1.3rem; font-weight:700; }

.app-footer { text-align:center; color: var(--text-muted); padding:1.8rem 0; font-size:0.88rem; border-top:1px solid rgba(99,120,255,0.14); margin-top:1.5rem; }
.app-footer strong { background: var(--gradient-hero); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS
# =============================================================================
def df_show(obj, **kwargs):
    try:
        return st.dataframe(obj, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(obj, use_container_width=True, **kwargs)

def btn_download(label, data, file_name, mime, **kwargs):
    try:
        return st.download_button(label, data, file_name=file_name, mime=mime, width="stretch", **kwargs)
    except TypeError:
        return st.download_button(label, data, file_name=file_name, mime=mime, use_container_width=True, **kwargs)

def show_pyplot(fig):
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)

def _apply_chart_style(fig, ax, title="", xlabel="Fecha", ylabel="Volumen (HL)"):
    CHART_BG = "#0e1428"
    FG = "#c8d0f0"
    GRID = (0.38, 0.42, 0.68, 0.18)

    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=FG, labelsize=10)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)
        spine.set_color((0.38, 0.42, 0.68, 0.30))
    ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6, color=GRID)
    ax.set_title(title, fontsize=15, fontweight="800", pad=16, color="#e8ecff")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="600", color=FG)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="600", color=FG)
    return FG

# =============================================================================
# DRIVE + CSV
# =============================================================================
@st.cache_data(show_spinner=False)
def ensure_csv_from_drive(file_id: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    url = f"https://drive.google.com/uc?id={file_id}"
    ok = gdown.download(url, local_path, quiet=True)

    if (not ok) or (not os.path.exists(local_path)) or os.path.getsize(local_path) == 0:
        raise FileNotFoundError(
            "No se pudo descargar el CSV desde Google Drive. "
            "Asegúrate que esté compartido como: 'cualquiera con el enlace'."
        )
    return local_path

@st.cache_data(show_spinner=False)
def load_base_csv_from_drive(file_id: str) -> pd.DataFrame:
    path = ensure_csv_from_drive(file_id, CSV_LOCAL_PATH)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df

# =============================================================================
# MODELOS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "modelos")
PROPHET_PATH = os.path.join(MODELS_DIR, "prophet_model.joblib")
SARIMAX_PATH = os.path.join(MODELS_DIR, "sarimax_model.joblib")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def is_prophet_model(obj) -> bool:
    mod = obj.__class__.__module__.lower()
    name = obj.__class__.__name__.lower()
    return ("prophet" in mod) or ("fbprophet" in mod) or (name == "prophet")

def is_statsmodels_model(obj) -> bool:
    return "statsmodels" in obj.__class__.__module__.lower()

# =============================================================================
# FERIADOS 2025
# =============================================================================
FERIADOS_2025 = pd.to_datetime(
    [
        "2025-01-01", "2025-01-22", "2025-03-03",
        "2025-03-04", "2025-04-18", "2025-05-01",
        "2025-06-21", "2025-08-06", "2025-08-16",
        "2025-09-14", "2025-11-02", "2025-12-25",
    ]
)

def build_feriado_anticipado(dates: pd.DatetimeIndex, feriados: pd.DatetimeIndex, anticipacion_dias: int = 7) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    s = pd.Series(0, index=idx, dtype=int)
    for f in pd.to_datetime(feriados):
        rango = pd.date_range(f - timedelta(days=anticipacion_dias), f - timedelta(days=1), freq="D")
        s.loc[s.index.isin(rango)] = 1
    return s

def apply_operational_zeros(pred: pd.Series, feriados: pd.DatetimeIndex, activar: bool = True) -> pd.Series:
    pred = pred.copy()
    pred.index = pd.to_datetime(pred.index)
    if not activar:
        pred.loc[pred < 0] = 0
        return pred
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

# =============================================================================
# MIX PRODUCTO
# =============================================================================
DATE_CANDIDATES = ["FECHA_CIERRE", "FECHA_SALIDA", "FECHA"]
PROD_CANDIDATES = ["PRODUCTO", "PRODUCTO "]
VOL_CANDIDATES  = ["VOLUMEN_VENDIDO_NETA", "HL", "VOLUMEN_HL", "VENTA_HL"]

def pick_existing(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    st.error(f"❌ No encontré columna para **{label}**. Probé: {candidates}")
    st.write("📌 Columnas detectadas:", list(df.columns))
    st.stop()

def build_mix_producto(df: pd.DataFrame, col_fecha: str, col_producto: str, col_vol: str, year: int):
    dfx = df.copy()
    dfx[col_fecha] = pd.to_datetime(dfx[col_fecha], errors="coerce")
    dfx = dfx.dropna(subset=[col_fecha, col_producto])
    dfx[col_producto] = dfx[col_producto].astype(str).str.strip()
    dfx = dfx[dfx[col_producto] != ""]
    dfx[col_vol] = pd.to_numeric(dfx[col_vol], errors="coerce").fillna(0)

    dfy = dfx[dfx[col_fecha].dt.year == int(year)].copy()
    total_year = float(dfy[col_vol].sum())

    mix = (
        dfy.groupby(col_producto, as_index=False)[col_vol]
        .sum()
        .rename(columns={col_producto: "PRODUCTO", col_vol: f"VENTA_{year}_HL"})
    )
    mix["PARTICIPACION_%"] = (mix[f"VENTA_{year}_HL"] / total_year) if total_year > 0 else 0.0
    mix = mix.sort_values(f"VENTA_{year}_HL", ascending=False)
    return mix, total_year

# =============================================================================
# SEGMENTACIÓN RFV
# =============================================================================
def compute_rfv_exact(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    dfx = df.copy()
    dfx["FECHA_CIERRE"] = pd.to_datetime(dfx["FECHA_CIERRE"], errors="coerce")
    dfx["FACTURA_TOTAL"] = pd.to_numeric(dfx["FACTURA_TOTAL"], errors="coerce")
    dfx["VOLUMEN_VENDIDO_NETA"] = pd.to_numeric(dfx["VOLUMEN_VENDIDO_NETA"], errors="coerce")

    dfx = dfx[dfx["FECHA_CIERRE"].dt.year == int(year)].copy()
    fecha_corte = dfx["FECHA_CIERRE"].max()

    rfv = (
        dfx.groupby("CODIGO_CLIENTE")
        .agg(
            ULTIMA_FECHA=("FECHA_CIERRE", "max"),
            F=("FECHA_CIERRE", "count"),
            V=("VOLUMEN_VENDIDO_NETA", "sum"),
            M=("FACTURA_TOTAL", "sum"),
        )
        .reset_index()
    )
    rfv["R"] = (fecha_corte - rfv["ULTIMA_FECHA"]).dt.days
    rfv["V"] = rfv["V"].fillna(0)
    rfv["M"] = rfv["M"].fillna(0)
    return rfv[["CODIGO_CLIENTE", "R", "F", "V", "M"]].copy(), fecha_corte

def run_kmeans(data: pd.DataFrame, features: list[str], k: int, random_state: int = 42):
    X = data[features].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=int(k), random_state=int(random_state), n_init=10)
    labels = km.fit_predict(Xs)

    sil = np.nan
    if k >= 2 and len(np.unique(labels)) > 1:
        sil = float(silhouette_score(Xs, labels))
    return labels, sil

def auto_k_by_silhouette(data: pd.DataFrame, features: list[str], k_min: int = 2, k_max: int = 10, random_state: int = 42):
    best_k, best_sil = None, -1.0
    scores = []
    for k in range(int(k_min), int(k_max) + 1):
        labels, sil = run_kmeans(data, features, k=k, random_state=random_state)
        scores.append((k, sil))
        if np.isfinite(sil) and sil > best_sil:
            best_sil, best_k = sil, k
    return best_k, best_sil, pd.DataFrame(scores, columns=["K", "Silhouette"])

def build_cluster_summary(rfv_clustered: pd.DataFrame) -> pd.DataFrame:
    dfc = rfv_clustered.copy()
    total_clients = dfc["CODIGO_CLIENTE"].nunique()
    total_value = dfc["V"].sum()

    summary = (
        dfc.groupby("Cluster")
        .agg(
            Recency_media=("R", "mean"),
            Frequency_media=("F", "mean"),
            Value_HL_media=("V", "mean"),
            N_Clientes=("CODIGO_CLIENTE", "nunique"),
            Volumen_total=("V", "sum"),
        )
        .reset_index()
    )
    summary["%_Clientes"] = (summary["N_Clientes"] / total_clients * 100) if total_clients else 0
    summary["%_Volumen"] = (summary["Volumen_total"] / total_value * 100) if total_value else 0
    summary = summary.round({
        "Recency_media": 2, "Frequency_media": 2, "Value_HL_media": 2,
        "%_Clientes": 2, "%_Volumen": 2, "Volumen_total": 2
    })
    return summary

def plot_kmeans_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    _apply_chart_style(fig, ax, title=title, xlabel=x, ylabel=y)
    sc = ax.scatter(df[x].values, df[y].values, c=df["Cluster"].values, cmap="viridis", alpha=0.65, s=28)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Cluster")
    plt.tight_layout()
    return fig

def plot_box_by_cluster(df: pd.DataFrame, value_col: str, title: str):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _apply_chart_style(fig, ax, title=title, xlabel="Cluster", ylabel=value_col)
    clusters = sorted(df["Cluster"].unique())
    data = [df[df["Cluster"] == c][value_col].values for c in clusters]
    ax.boxplot(data, labels=[str(c) for c in clusters], patch_artist=False, showfliers=False)
    plt.tight_layout()
    return fig

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<h1 class="main-title">📈 Forecast de Ventas 2025</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Comparación <strong>Prophet</strong> vs <strong>SARIMAX</strong> · Productos · Segmentación</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION STATE
# =============================================================================
if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = None
if "seg_cache" not in st.session_state:
    st.session_state.seg_cache = None

# =============================================================================
# LOAD CSV
# =============================================================================
try:
    with st.spinner("📥 Cargando CSV base desde Drive…"):
        df_base = load_base_csv_from_drive(DRIVE_FILE_ID)
except Exception as e:
    st.error(f"No pude descargar/leer el CSV desde Drive: {e}")
    st.stop()

COL_FECHA = pick_existing(df_base, DATE_CANDIDATES, "Fecha")
COL_PRODUCTO = pick_existing(df_base, PROD_CANDIDATES, "Producto")
COL_VOL = pick_existing(df_base, VOL_CANDIDATES, "Volumen")

tmp_years = pd.to_datetime(df_base["FECHA_CIERRE"], errors="coerce").dropna()
available_years = sorted(tmp_years.dt.year.unique().tolist()) if len(tmp_years) else [2024]

# =============================================================================
# SIDEBAR (parámetros globales)
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    anticipacion_dias = st.slider("Días anticipación feriado", 1, 30, 7, 1)
    zeros_operativos = st.checkbox("Ceros operativos (domingos/feriados)", value=True)

    vista = st.selectbox("Agregación", ["Diario", "Semanal", "Mensual"], index=0)
    acumulado = st.checkbox("Acumulado", value=False)
    mostrar_intervalos = st.checkbox("Intervalos de confianza", value=True)

    

clear_cache = st.sidebar.checkbox("🧹 Recalcular (limpiar cachés)", value=False)

if clear_cache:
    st.cache_data.clear()
    st.cache_resource.clear()

    # Solo lo que te interesa recomputar
    for k in ["seg_cache", "pred_cache", "mix_cache"]:
        if k in st.session_state:
            del st.session_state[k]

    st.rerun()

# =============================================================================
# LOAD MODELS
# =============================================================================
with st.spinner("📦 Cargando modelos…"):
    if not os.path.exists(PROPHET_PATH) or not os.path.exists(SARIMAX_PATH):
        st.error("Faltan modelos en /modelos (prophet_model.joblib y sarimax_model.joblib).")
        st.stop()

    m1 = load_model(PROPHET_PATH)
    m2 = load_model(SARIMAX_PATH)

    if is_prophet_model(m1) and is_statsmodels_model(m2):
        prophet_model, sarimax_model = m1, m2
    elif is_prophet_model(m2) and is_statsmodels_model(m1):
        prophet_model, sarimax_model = m2, m1
    else:
        st.error("No pude identificar claramente Prophet y SARIMAX en los .joblib.")
        st.write("m1:", type(m1), m1.__class__.__module__)
        st.write("m2:", type(m2), m2.__class__.__module__)
        st.stop()

# =============================================================================
# PRED COMPUTE
# =============================================================================
@st.cache_data(show_spinner=False)
def compute_predictions_2025(anticipacion_dias: int):
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

    exog = pd.DataFrame(index=dates)
    exog["ES_FERIADO_ANTICIPADO"] = build_feriado_anticipado(exog.index, FERIADOS_2025, anticipacion_dias).values

    # Prophet
    future = pd.DataFrame({"ds": dates})
    future["ES_FERIADO_ANTICIPADO"] = exog["ES_FERIADO_ANTICIPADO"].values
    fc_p = prophet_model.predict(future)

    pred_p = pd.Series(fc_p["yhat"].values, index=dates)
    low_p = pd.Series(fc_p["yhat_lower"].values, index=dates)
    up_p  = pd.Series(fc_p["yhat_upper"].values, index=dates)

    # SARIMAX
    fc_s = sarimax_model.get_forecast(steps=len(exog), exog=exog)
    pred_s = pd.Series(np.asarray(fc_s.predicted_mean), index=dates)
    ci_s = fc_s.conf_int()
    low_s = pd.Series(np.asarray(ci_s.iloc[:, 0]), index=dates)
    up_s  = pd.Series(np.asarray(ci_s.iloc[:, 1]), index=dates)

    return dates, pred_p, low_p, up_p, pred_s, low_s, up_s

def apply_view(pred, low, up, vista, acumulado):
    p = resample_view(pred, vista)
    l = resample_view(low, vista)
    u = resample_view(up, vista)
    if acumulado:
        p, l, u = p.cumsum(), l.cumsum(), u.cumsum()
    return p, l, u

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Resumen", "🔄 Comparación", "🔎 Detalle", "🛒 Productos", "🧩 Segmentación"]
)

# =============================================================================
# CALC PRED (cache)
# =============================================================================
if run_pred or st.session_state.pred_cache is None:
    with st.spinner("🔄 Calculando predicciones…"):
        dates, pred_p, low_p, up_p, pred_s, low_s, up_s = compute_predictions_2025(int(anticipacion_dias))

        pred_p = apply_operational_zeros(pred_p, FERIADOS_2025, activar=zeros_operativos)
        pred_s = apply_operational_zeros(pred_s, FERIADOS_2025, activar=zeros_operativos)
        low_p  = apply_operational_zeros(low_p,  FERIADOS_2025, activar=zeros_operativos)
        up_p   = apply_operational_zeros(up_p,   FERIADOS_2025, activar=zeros_operativos)
        low_s  = apply_operational_zeros(low_s,  FERIADOS_2025, activar=zeros_operativos)
        up_s   = apply_operational_zeros(up_s,   FERIADOS_2025, activar=zeros_operativos)

        st.session_state.pred_cache = {
            "dates": dates,
            "pred_p": pred_p, "low_p": low_p, "up_p": up_p,
            "pred_s": pred_s, "low_s": low_s, "up_s": up_s,
        }

pred_cache = st.session_state.pred_cache
dates = pred_cache["dates"]
pred_p = pred_cache["pred_p"]; low_p = pred_cache["low_p"]; up_p = pred_cache["up_p"]
pred_s = pred_cache["pred_s"]; low_s = pred_cache["low_s"]; up_s = pred_cache["up_s"]

p_view, lp_view, up_view = apply_view(pred_p, low_p, up_p, vista, acumulado)
s_view, ls_view, us_view = apply_view(pred_s, low_s, up_s, vista, acumulado)

df_pred = pd.DataFrame({
    "ds": dates,
    "prophet_yhat": pred_p.values,
    "prophet_low": low_p.values,
    "prophet_up": up_p.values,
    "sarimax_mean": pred_s.values,
    "sarimax_low": low_s.values,
    "sarimax_up": up_s.values,
})
df_pred["diff_prophet_minus_sarimax"] = df_pred["prophet_yhat"] - df_pred["sarimax_mean"]

df_view = pd.DataFrame({
    "fecha": p_view.index.astype(str),
    f"prophet_{vista.lower()}": p_view.values,
    f"sarimax_{vista.lower()}": s_view.values,
})
df_view["diff"] = df_view[f"prophet_{vista.lower()}"] - df_view[f"sarimax_{vista.lower()}"]

# =============================================================================
# TAB 1: RESUMEN (con gráficas de vuelta)
# =============================================================================
with tab1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Total Prophet", f"{pred_p.sum():,.0f} HL")
    c2.metric("🟠 Total SARIMAX", f"{pred_s.sum():,.0f} HL")
    c3.metric("📌 Diferencia", f"{(pred_p.sum() - pred_s.sum()):,.0f} HL")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Descargas rápidas")
    d1, d2, d3 = st.columns(3)
    with d1:
        btn_download("⬇️ Predicciones DIARIAS (CSV)", df_pred.to_csv(index=False).encode("utf-8"),
                     "predicciones_diarias_2025.csv", "text/csv")
    with d2:
        btn_download(f"⬇️ Predicciones {vista.upper()} (CSV)", df_view.to_csv(index=False).encode("utf-8"),
                     f"predicciones_{vista.lower()}_2025.csv", "text/csv")
    with d3:
        df_exec = pd.DataFrame([{
            "total_prophet": float(pred_p.sum()),
            "total_sarimax": float(pred_s.sum()),
            "diff": float(pred_p.sum() - pred_s.sum()),
            "zeros_operativos": bool(zeros_operativos),
            "anticipacion_dias": int(anticipacion_dias),
            "vista": vista,
            "acumulado": bool(acumulado),
        }])
        btn_download("⬇️ Resumen ejecutivo (CSV)", df_exec.to_csv(index=False).encode("utf-8"),
                     "resumen_ejecutivo.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    # Gráfico principal (vuelto)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 6.0))
    _apply_chart_style(fig, ax, title=f"Prophet vs SARIMAX — Vista {vista}", xlabel="Fecha", ylabel="HL")
    ax.plot(p_view.index, p_view.values, label="Prophet", linewidth=2.6, color="#00c07a")
    ax.plot(s_view.index, s_view.values, label="SARIMAX", linewidth=2.6, color="#f5a623")
    ax.legend(loc="best", frameon=True)
    plt.xticks(rotation=35)
    plt.tight_layout()
    show_pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Extra: diferencia
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 4.8))
    _apply_chart_style(fig, ax, title=f"Diferencia (Prophet − SARIMAX) — {vista}", xlabel="Fecha", ylabel="HL")
    ax.axhline(0, color="#c8d0f0", alpha=0.35, linestyle="--")
    ax.plot(df_view["fecha"], df_view["diff"], linewidth=2.2, color="#3d5cff")
    ax.set_xticks([])
    plt.tight_layout()
    show_pyplot(fig)
    st.caption("Valores positivos: Prophet predice más. Negativos: SARIMAX predice más.")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: COMPARACIÓN (descargas aquí también)
# =============================================================================
with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 Comparación + Descargas")
    a1, a2 = st.columns(2)
    with a1:
        btn_download("⬇️ CSV comparativo diario", df_pred.to_csv(index=False).encode("utf-8"),
                     "comparacion_diaria.csv", "text/csv")
    with a2:
        btn_download(f"⬇️ CSV comparativo {vista.lower()}", df_view.to_csv(index=False).encode("utf-8"),
                     f"comparacion_{vista.lower()}.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 6.0))
    _apply_chart_style(fig, ax, title=f"Comparación — {vista}", xlabel="Fecha", ylabel="HL")
    ax.plot(p_view.index, p_view.values, label="Prophet", linewidth=2.6, color="#00c07a")
    ax.plot(s_view.index, s_view.values, label="SARIMAX", linewidth=2.6, color="#f5a623")
    ax.legend(loc="best", frameon=True)
    plt.xticks(rotation=35)
    plt.tight_layout()
    show_pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 📋 Tabla comparativa")
    df_show(df_view)

# =============================================================================
# TAB 3: DETALLE (más creativo)
# =============================================================================
with tab3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🔎 Detalle analítico (para defensa)")

    zeros_p = int((pred_p == 0).sum())
    zeros_s = int((pred_s == 0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Promedio Prophet (HL/día)", f"{pred_p.mean():,.2f}")
    m2.metric("Promedio SARIMAX (HL/día)", f"{pred_s.mean():,.2f}")
    m3.metric("Días en cero (Prophet)", f"{zeros_p}")
    m4.metric("Días en cero (SARIMAX)", f"{zeros_s}")

    st.markdown("---")
    b1, b2 = st.columns(2)

    with b1:
        st.markdown("#### 📌 Top 10 días con mayor volumen (Prophet)")
        top_vol = df_pred.sort_values("prophet_yhat", ascending=False).head(10)[["ds", "prophet_yhat", "sarimax_mean"]]
        df_show(top_vol)

    with b2:
        st.markdown("#### 📌 Top 10 días con mayor diferencia (abs)")
        top_diff = df_pred.assign(abs_diff=np.abs(df_pred["diff_prophet_minus_sarimax"])) \
                          .sort_values("abs_diff", ascending=False).head(10)[["ds", "diff_prophet_minus_sarimax"]]
        df_show(top_diff)

    btn_download("⬇️ Descargar detalle completo (CSV)",
                 df_pred.to_csv(index=False).encode("utf-8"),
                 "detalle_predicciones.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Histograma de diferencias (Prophet − SARIMAX)")
    fig, ax = plt.subplots(figsize=(14, 5.2))
    _apply_chart_style(fig, ax, title="Distribución de diferencias", xlabel="Diferencia (HL)", ylabel="Frecuencia")
    ax.hist(df_pred["diff_prophet_minus_sarimax"].values, bins=30, alpha=0.85, color="#3d5cff")
    plt.tight_layout()
    show_pyplot(fig)
    st.caption("Te ayuda a explicar si un modelo tiende a sobre/infraestimar vs el otro.")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 4: PRODUCTOS (más de 10 + ver todos)
# =============================================================================
with tab4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🛒 Productos (Top dinámico)")

    mix_year = st.selectbox("Año para mix (participación)", available_years, index=len(available_years) - 1)

    # Controles dentro del tab
    cA, cB, cC = st.columns(3)
    with cA:
        show_n = st.number_input("Mostrar N productos", min_value=5, max_value=200, value=30, step=5)
    with cB:
        min_vol = st.number_input("Vol mínimo (HL)", min_value=0.0, value=0.0, step=10.0)
    with cC:
        search_prod = st.text_input("Buscar producto (contiene)", value="")

    mix, total_mix = build_mix_producto(df_base, COL_FECHA, COL_PRODUCTO, COL_VOL, year=int(mix_year))

    if float(min_vol) > 0:
        mix = mix[mix[f"VENTA_{mix_year}_HL"] >= float(min_vol)].copy()
    if search_prod.strip():
        mix = mix[mix["PRODUCTO"].str.contains(search_prod.strip(), case=False, na=False)].copy()

    st.caption(f"Total {mix_year} (HL): {total_mix:,.0f} · Filas luego de filtros: {len(mix):,}")

    mix_top = mix.head(int(show_n)).copy()
    df_show(mix_top.style.format({f"VENTA_{mix_year}_HL": "{:,.2f}", "PARTICIPACION_%": "{:.4%}"}))

    d1, d2 = st.columns(2)
    with d1:
        btn_download("⬇️ Descargar TOP (CSV)",
                     mix_top.to_csv(index=False).encode("utf-8"),
                     f"top_{mix_year}.csv", "text/csv")
    with d2:
        btn_download("⬇️ Descargar TODO filtrado (CSV)",
                     mix.to_csv(index=False).encode("utf-8"),
                     f"productos_filtrados_{mix_year}.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 5: SEGMENTACIÓN (K + más/menos + más gráficas + descargas)
# =============================================================================
with tab5:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🧩 Segmentación (K-Means)")

    required_cols = {"FECHA_CIERRE", "CODIGO_CLIENTE", "VOLUMEN_VENDIDO_NETA", "FACTURA_TOTAL"}
    miss = [c for c in required_cols if c not in df_base.columns]
    if miss:
        st.error(f"Faltan columnas para segmentación: {miss}")
        st.stop()

    # Controles dentro del tab (subir/bajar K)
    s1, s2, s3 = st.columns(3)
    with s1:
        year_seg = st.selectbox("Año base", available_years, index=len(available_years) - 1)
    with s2:
        features_mode = st.selectbox("Variables", ["R,F,V (recomendado)", "F,V (rápido)"], index=0)
    with s3:
        use_auto_k = st.checkbox("AutoK (Silhouette)", value=False)

    if use_auto_k:
        k_min = st.slider("K mínimo", 2, 8, 2, 1)
        k_max = st.slider("K máximo", 3, 12, 8, 1)
        k_used = None
    else:
        k_used = st.slider("Número de clusters (K)", 2, 20, 4, 1)

    btn = st.button("🧩 Recalcular segmentación", use_container_width=True)

    if btn or st.session_state.seg_cache is None:
        with st.spinner("🧩 Calculando segmentación…"):
            rfv, fecha_corte = compute_rfv_exact(df_base, year=int(year_seg))
            feats = ["R", "F", "V"] if features_mode.startswith("R") else ["F", "V"]

            if use_auto_k:
                best_k, best_sil, df_scores = auto_k_by_silhouette(rfv, feats, k_min=int(k_min), k_max=int(k_max), random_state=42)
                k_final = best_k if best_k is not None else int(k_min)
            else:
                df_scores = None
                k_final = int(k_used)

            labels, sil = run_kmeans(rfv, feats, k=int(k_final), random_state=42)
            rfv_k = rfv.copy()
            rfv_k["Cluster"] = labels
            summary = build_cluster_summary(rfv_k)

            st.session_state.seg_cache = {
                "rfv_k": rfv_k,
                "summary": summary,
                "sil": sil,
                "k_final": k_final,
                "feats": feats,
                "df_scores": df_scores,
                "fecha_corte": fecha_corte,
            }

    seg = st.session_state.get("seg_cache", None)

    required = {"rfv_k", "summary", "sil", "k_final"}
    if (not isinstance(seg, dict)) or (not required.issubset(seg.keys())):
        st.session_state["seg_cache"] = None
        st.rerun()

    k_final = seg["k_final"]

    seg = st.session_state.seg_cache
    rfv_k = seg["rfv_k"]
    summary = seg["summary"]
    sil = seg["sil"]
    k_final = seg["k_final"]
    feats = seg["feats"]
    df_scores = seg["df_scores"]
    fecha_corte = seg["fecha_corte"]

    st.write(f"📌 **Año:** {year_seg} · **Fecha corte:** {str(fecha_corte)[:10]} · **Features:** {', '.join(feats)}")
    st.success(f"✅ K: {k_final} · Silhouette: {sil:.3f}" if np.isfinite(sil) else f"✅ K: {k_final} · Silhouette: N/A")

    if df_scores is not None:
        st.markdown("#### 🔍 AutoK — Silhouette por K")
        df_show(df_scores)
        btn_download("⬇️ Descargar AutoK (CSV)", df_scores.to_csv(index=False).encode("utf-8"),
                     f"autok_{year_seg}.csv", "text/csv")

    st.markdown("#### 📋 Resumen por cluster")
    df_show(summary)

    d1, d2 = st.columns(2)
    with d1:
        btn_download("⬇️ Descargar RFV+Cluster (CSV)",
                     rfv_k.sort_values("Cluster").to_csv(index=False).encode("utf-8"),
                     f"segmentacion_{year_seg}_k{k_final}.csv", "text/csv")
    with d2:
        btn_download("⬇️ Descargar resumen clusters (CSV)",
                     summary.to_csv(index=False).encode("utf-8"),
                     f"resumen_clusters_{year_seg}_k{k_final}.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 📌 Scatter (F vs V)")
    fig = plot_kmeans_scatter(rfv_k, "F", "V", "K-Means — Frecuencia vs Volumen")
    show_pyplot(fig)

    st.markdown("#### 📦 Distribución por cluster (Boxplots)")
    fig = plot_box_by_cluster(rfv_k, "V", "Volumen (V) por cluster")
    show_pyplot(fig)
    fig = plot_box_by_cluster(rfv_k, "F", "Frecuencia (F) por cluster")
    show_pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    """
<div class="app-footer">
  <strong>Forecast de Ventas 2025</strong> · Cochabamba, Bolivia
</div>
""",
    unsafe_allow_html=True,
)