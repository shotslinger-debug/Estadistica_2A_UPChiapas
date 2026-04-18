import json
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from modules.utils import calcular_prueba_z
CHART_FACE = "#1A1D27"
# |skewness| < este umbral → "simétrica / sin sesgo marcado" (criterio habitual en aplicaciones)
UMBRAL_SESGO_SIMETRIA = 0.5
plt.rcParams.update(
    {
        "figure.facecolor": CHART_FACE,
        "axes.facecolor": CHART_FACE,
        "axes.edgecolor": "#78909C",
        "axes.labelcolor": "#ECEFF1",
        "axes.titlecolor": "#ECEFF1",
        "text.color": "#ECEFF1",
        "xtick.color": "#CFD8DC",
        "ytick.color": "#CFD8DC",
        "grid.color": "#37474F",
        "grid.alpha": 0.45,
        "legend.facecolor": CHART_FACE,
        "legend.edgecolor": "#2196F3",
        "legend.labelcolor": "#ECEFF1",
        "savefig.facecolor": CHART_FACE,
        "savefig.edgecolor": CHART_FACE,
    }
)
sns.set_theme(
    style="darkgrid",
    rc={
        "figure.facecolor": CHART_FACE,
        "axes.facecolor": CHART_FACE,
    },
)

st.set_page_config(page_title="Probabilidad y estadistica - 2A", layout="wide")

st.markdown(
    """
    <style>
      html {
        font-family: ui-monospace, "Cascadia Code", "Consolas", "Liberation Mono", monospace;
      }
      .stApp {
        font-family: ui-monospace, "Cascadia Code", "Consolas", "Liberation Mono", monospace;
      }
      [data-testid="stMain"] h1,
      [data-testid="stMain"] h2,
      [data-testid="stMain"] h3 {
        border-left: 4px solid #2196F3;
        padding-left: 0.65rem;
        margin-left: 0;
      }
      [data-testid="stBaseButton-primary"] {
        border: 2px solid #2196F3 !important;
        box-shadow: 0 0 0 1px rgba(33, 150, 243, 0.35) inset;
      }
      .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255, 255, 255, 0.12);
        white-space: nowrap;
      }
      .status-badge.ok {
        color: #66BB6A;
        background: rgba(102, 187, 106, 0.12);
        border-color: rgba(102, 187, 106, 0.45);
      }
      .status-badge.idle {
        color: #9E9E9E;
        background: rgba(158, 158, 158, 0.1);
        border-color: rgba(158, 158, 158, 0.35);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if "datos" not in st.session_state:
    st.session_state["datos"] = None

SECCIONES = (
    "1 · Carga de Datos",
    "2 · Visualización Exploratoria",
    "3 · Prueba de Hipótesis (Z)",
    "4 · Asistente IA (Gemini)",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Estadística — 2A")
st.sidebar.divider()

# API Key primero — debe definirse antes de usarse en los indicadores de progreso
st.sidebar.caption("Asistente IA (Sección 4)")
gemini_api_key = st.sidebar.text_input(
    "API Key de Gemini",
    type="password",
    help="Obtén una clave en Google AI Studio. No se almacena en disco.",
)
st.sidebar.divider()

# Indicadores de progreso dinámicos
datos_ok = st.session_state.get("datos") is not None
z_ok = "ultima_prueba_z" in st.session_state
st.sidebar.markdown(f"{'✅' if datos_ok else '○'} Datos cargados")
st.sidebar.markdown(f"{'✅' if datos_ok else '○'} Visualización")
st.sidebar.markdown(f"{'✅' if z_ok else '○'} Prueba Z ejecutada")
st.sidebar.markdown(f"{'✅' if gemini_api_key else '○'} API Key lista")
st.sidebar.divider()

seccion = st.sidebar.radio(
    "Selecciona una sección",
    SECCIONES,
    label_visibility="collapsed",
)

# ── Header ────────────────────────────────────────────────────────────────────
head_l, head_r = st.columns([4, 1.15])
with head_l:
    st.title("Probabilidad y Estadística — 2A")
    st.caption("Visualización · Prueba Z · Asistente IA · Gemini 1.5")
with head_r:
    if st.session_state.datos is not None:
        st.markdown(
            '<span class="status-badge ok">● datos cargados</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge idle">● sin datos</span>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
def _outliers_iqr_count(serie: pd.Series) -> int:
    """Valores atípicos por método IQR (límites Q1 − 1.5·IQR y Q3 + 1.5·IQR)."""
    s = serie.dropna()
    if len(s) == 0:
        return 0
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(((s < low) | (s > high)).sum())


# ═══════════════════════════════════════════════════════════════════════════════
def render_carga_datos() -> None:
    st.subheader("1 · Carga de Datos")
    origen = st.radio(
        "Selecciona el origen de los datos",
        ["Subir CSV", "Generar datos sintéticos"],
        horizontal=True,
    )

    if origen == "Subir CSV":
        archivo_csv = st.file_uploader("Sube un archivo CSV", type=["csv"])
        if archivo_csv is not None:
            datos = pd.read_csv(archivo_csv)
            datos = datos.drop(columns=["ID"], errors="ignore")
            datos.index = datos.index + 1
            st.session_state["datos"] = datos
            st.success("Archivo cargado con exito")
        else:
            st.session_state["datos"] = None
            st.info("Sube un archivo CSV para continuar.")
            return

    else:
        st.info(
            "Los datos sintéticos son una **muestra aleatoria de una distribución "
            "Normal** **N(μ, σ²)** (campana de Gauss), generada con "
            "`numpy.random.Generator.normal`. Elige la **media μ** y la **desviación estándar σ** "
            "de la población, el tamaño muestral **n** y la **semilla**; después pulsa "
            "**Generar muestra Normal**."
        )
        with st.form("form_datos_sinteticos", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                n_muestra = st.number_input(
                    "Tamaño de la muestra (n)",
                    min_value=30,
                    max_value=100_000,
                    value=500,
                    step=10,
                )
                semilla = st.number_input(
                    "Semilla (reproducibilidad)",
                    min_value=0,
                    value=42,
                    step=1,
                )
            with c2:
                media = st.number_input(
                    "Media de la población (μ)",
                    value=5.0,
                    format="%.4f",
                    help="Media de la Normal de la que se muestrea.",
                )
                sigma = st.number_input(
                    "Desviación estándar de la población (σ)",
                    min_value=1e-6,
                    value=1.25,
                    format="%.6f",
                    help="σ > 0. Escalas de la campana de Gauss.",
                )
            generar = st.form_submit_button("Generar muestra Normal")

        if generar:
            rng = np.random.default_rng(int(semilla))
            muestra = rng.normal(
                loc=float(media), scale=float(sigma), size=int(n_muestra)
            )
            datos = pd.DataFrame({"Valor_Generado": muestra})
            datos.index = datos.index + 1
            st.session_state["datos"] = datos
            st.success(
                "Muestra Normal generada: "
                f"N(μ={float(media):.4f}, σ={float(sigma):.4f}), n={int(n_muestra)}."
            )
        elif st.session_state.get("datos") is None:
            st.caption(
                "Configura los parámetros y pulsa **Generar muestra Normal** para "
                "crear el conjunto de datos."
            )

    datos = st.session_state["datos"]
    if datos is not None:
        st.caption(f"Registros: **{len(datos)}** · Índice desde 1")
        st.dataframe(datos, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
def render_visualizacion(datos: pd.DataFrame) -> None:
    st.subheader("2 · Visualización Exploratoria")
    columnas_numericas = datos.select_dtypes(include=["number"]).columns.tolist()
    if not columnas_numericas:
        st.warning("El archivo no contiene columnas numericas.")
        return

    columna_seleccionada = st.selectbox(
        "Selecciona una columna numerica",
        columnas_numericas,
        key="columna_analisis",
    )

    serie = datos[columna_seleccionada].dropna()
    n_obs = int(len(serie))
    media_v = float(serie.mean()) if n_obs else float("nan")
    std_v = float(serie.std(ddof=1)) if n_obs >= 2 else float("nan")
    skewness = float(stats.skew(serie, bias=False)) if n_obs >= 3 else float("nan")

    p_sw_val = None
    if n_obs >= 3:
        muestra_sw = (
            serie if n_obs <= 5000 else serie.sample(5000, random_state=42)
        )
        if n_obs > 5000:
            st.caption(
                "Shapiro-Wilk: n > 5000; se usa submuestra aleatoria de 5000 (semilla 42)."
            )
        _, p_sw = stats.shapiro(muestra_sw)
        p_sw_val = float(p_sw)

    n_outliers = _outliers_iqr_count(serie)

    # ── Métricas clave ────────────────────────────────────────────────────────
    st.markdown("##### Métricas clave")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Tamaño (n)", f"{n_obs}")
    with k2:
        st.metric("Media", f"{media_v:.6f}" if n_obs else "—")
    with k3:
        st.metric("Desv. estándar", f"{std_v:.6f}" if n_obs >= 2 else "—")
    with k4:
        st.metric("Sesgo (skewness)", f"{skewness:.6f}" if n_obs >= 3 else "—")
    with k5:
        st.metric(
            "P-valor (Shapiro)",
            f"{p_sw_val:.4e}" if p_sw_val is not None else "—",
        )

    # ── Diagnóstico automático ────────────────────────────────────────────────
    st.markdown("##### 📝 Diagnóstico Automático")
    col_diag1, col_diag2, col_diag3 = st.columns(3)

    with col_diag1:
        if p_sw_val is None:
            st.info("🔍 **Normalidad**\n\nRequiere n ≥ 3.")
        elif p_sw_val > 0.05:
            st.success("✅ **Distribución Normal**\n\nCompatible con Gauss.")
        else:
            st.error("⚠️ **No Normal**\n\nDatos con distribución irregular.")

    with col_diag2:
        if n_obs >= 3 and not np.isnan(skewness):
            if abs(skewness) < UMBRAL_SESGO_SIMETRIA:
                st.info("⚖️ **Simétrica**\n\nSin sesgo significativo.")
            elif skewness > 0:
                st.info("📐 **Sesgo Positivo**\n\nCola hacia la derecha.")
            else:
                st.info("📐 **Sesgo Negativo**\n\nCola hacia la izquierda.")
        else:
            st.info("📐 **Sesgo**\n\nRequiere n ≥ 3.")

    with col_diag3:
        if n_outliers > 0:
            st.warning(f"🧐 **{n_outliers} Outliers**\n\nValores atípicos detectados.")
        else:
            st.success("✨ **0 Outliers**\n\nDatos limpios de atípicos.")

    # ── Estadísticas descriptivas ─────────────────────────────────────────────
    st.divider()
    st.subheader("Estadisticas Descriptivas")
    st.table(datos[columna_seleccionada].describe().to_frame(name="valor"))

    # ── Gráficas ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.patch.set_facecolor(CHART_FACE)
    for ax in axes:
        ax.set_facecolor(CHART_FACE)
    color_hist = "#2196F3"
    sns.histplot(
        serie,
        kde=True,
        color=color_hist,
        ax=axes[0],
        edgecolor="white",
        line_kws={"color": color_hist, "linewidth": 2},
    )
    axes[0].set_title(f"Histograma + KDE: {columna_seleccionada}")
    axes[0].set_xlabel(columna_seleccionada)
    axes[0].set_ylabel("Frecuencia")

    axes[1].boxplot(
        [serie],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#BBDEFB"),
        medianprops=dict(color="#0D47A1", linewidth=2),
        flierprops=dict(
            marker="o",
            markerfacecolor="#F44336",
            markeredgecolor="#B71C1C",
            markersize=7,
            linestyle="none",
        ),
    )
    axes[1].set_xticklabels([columna_seleccionada])
    axes[1].set_title("Boxplot")
    axes[1].set_ylabel("Valor")
    axes[1].grid(True, axis="y", alpha=0.3)

    stats.probplot(serie, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot (normal teórica)")
    axes[2].grid(True, alpha=0.3)
    for ax in fig.axes:
        ax.set_facecolor(CHART_FACE)

    fig.tight_layout()
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
def render_prueba_z(datos: pd.DataFrame) -> None:
    st.subheader("3 · Prueba de Hipótesis (Z)")
    columnas_numericas = datos.select_dtypes(include=["number"]).columns.tolist()
    if not columnas_numericas:
        st.warning("El archivo no contiene columnas numericas.")
        return
    default_col = st.session_state.get("columna_analisis")
    if default_col not in columnas_numericas:
        default_col = columnas_numericas[0]
    columna_seleccionada = st.selectbox(
        "Columna para la prueba",
        columnas_numericas,
        index=columnas_numericas.index(default_col),
        key="columna_prueba_z",
    )
    serie = datos[columna_seleccionada].dropna()
    mu0 = st.number_input(
        "Media bajo la hipótesis nula (μ₀)",
        value=75.0,
        format="%.6f",
    )
    tipo_prueba = st.selectbox(
        "Tipo de prueba",
        [
            "Bilateral (H₁: μ ≠ μ₀)",
            "Unilateral derecha (H₁: μ > μ₀)",
            "Unilateral izquierda (H₁: μ < μ₀)",
        ],
    )
    alpha_z = st.slider("Nivel de significancia (α)", 0.001, 0.2, 0.05, 0.001)

    n_obs = int(len(serie))
    if n_obs < 2:
        st.warning("Se necesitan al menos 2 observaciones para calcular la prueba Z.")
        return

    xbar = float(serie.mean())
    s_muestral = float(serie.std(ddof=1))
    if s_muestral == 0:
        st.warning(
            "La desviación típica muestral es 0; no se puede calcular el estadístico Z."
        )
        return

    mapeo_tipos = {
        "Bilateral (H₁: μ ≠ μ₀)": "bilateral",
        "Unilateral derecha (H₁: μ > μ₀)": "cola_derecha",
        "Unilateral izquierda (H₁: μ < μ₀)": "cola_izquierda"
    }
    llave_tipo = mapeo_tipos.get(tipo_prueba, "bilateral")

    # Llamamos a la función de utils.py
    z_obs, p_valor = calcular_prueba_z(
        media_muestra=xbar, 
        media_h0=mu0, 
        sigma=s_muestral, 
        n=n_obs, 
        tipo_prueba=llave_tipo
    )

    rechazar = p_valor < alpha_z

    if tipo_prueba.startswith("Bilateral"):
        z_crit = float(stats.norm.ppf(1 - alpha_z / 2))
    elif "derecha" in tipo_prueba:
        z_crit = float(stats.norm.ppf(1 - alpha_z))
    else:
        z_crit = float(stats.norm.ppf(alpha_z))

    st.session_state["ultima_prueba_z"] = {
        "mu0": float(mu0),
        "x_bar": float(xbar),
        "n": int(n_obs),
        "z_calc": float(z_obs),
        "p_value": float(p_valor),
        "z_crit": float(z_crit),
        "decision": "rechazar H₀" if rechazar else "no rechazar H₀",
        "alpha": float(alpha_z),
        "tipo_prueba": tipo_prueba,
        "columna": columna_seleccionada,
    }

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Z crítico", f"{z_crit:.6f}", help="Valor de referencia bajo H₀ (α indicado).")
    with m2:
        st.metric("Media muestral (x̄)", f"{xbar:.6f}")
    with m3:
        st.metric("Estadístico Z", f"{z_obs:.6f}")
    with m4:
        st.metric("Valor p", f"{p_valor:.4e}")

    st.caption(
        "Z = (x̄ − μ₀) / (s / √n); p-valor bajo N(0,1) (σ desconocida, s como aproximación). "
        "En la prueba bilateral, Z crítico mostrado es el valor positivo z_{α/2}; la región "
        "de rechazo incluye también −z_{α/2}."
    )

    x_lo = min(-4.0, float(z_obs) - 1.5)
    x_hi = max(4.0, float(z_obs) + 1.5)
    x = np.linspace(x_lo, x_hi, 500)
    y = stats.norm.pdf(x)

    fig_z, ax_z = plt.subplots(figsize=(14, 4.5))
    fig_z.patch.set_facecolor(CHART_FACE)
    ax_z.set_facecolor(CHART_FACE)
    ax_z.plot(x, y, color="#ECEFF1", linewidth=2)

    if tipo_prueba.startswith("Bilateral"):
        ax_z.fill_between(x, y, where=x <= -z_crit, color="red", alpha=0.35)
        ax_z.fill_between(x, y, where=x >= z_crit, color="red", alpha=0.35)
    elif "derecha" in tipo_prueba:
        ax_z.fill_between(x, y, where=x >= z_crit, color="red", alpha=0.35)
    else:
        ax_z.fill_between(x, y, where=x <= z_crit, color="red", alpha=0.35)

    ax_z.axvline(z_obs, color="orange", linewidth=2.5, zorder=5)
    ax_z.set_title("Distribución bajo H₀ y región de rechazo (curva Z)")
    ax_z.set_xlabel("Z")
    ax_z.set_ylabel("Densidad")
    ax_z.set_ylim(bottom=0)
    ax_z.grid(True, alpha=0.3)

    ax_z.legend(
        handles=[
            Line2D([0], [0], color="#ECEFF1", linewidth=2, label="N(0, 1)"),
            mpatches.Patch(
                facecolor="red",
                edgecolor="none",
                alpha=0.35,
                label="Zona de rechazo",
            ),
            Line2D(
                [0],
                [0],
                color="orange",
                linewidth=2.5,
                label="Z calculado",
            ),
        ],
        loc="upper right",
    )

    fig_z.tight_layout()
    st.pyplot(fig_z)

    if rechazar:
        st.error(
            f"**Decisión:** se **rechaza** H₀ al nivel α = {alpha_z:.4f} "
            f"(p = {p_valor:.4e} < α)."
        )
    else:
        st.success(
            f"**Decisión:** **no se rechaza** H₀ al nivel α = {alpha_z:.4f} "
            f"(p = {p_valor:.4e} ≥ α)."
        )

GEMINI_FLASH_MODEL = "gemini-flash-latest" 

GEMINI_GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_FLASH_MODEL}:generateContent"
)

def _gemini_generate_via_requests(api_key: str, user_text: str) -> str:
    key = str(api_key).strip()
    if not key:
        raise ValueError("API Key vacía.")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
    }
    resp = requests.post(
        GEMINI_GENERATE_URL,
        params={"key": key},
        json=payload,
        timeout=120,
    )
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Respuesta no JSON ({resp.status_code}): {resp.text[:500]}") from exc
    if resp.status_code != 200:
        err = data.get("error", {}) if isinstance(data, dict) else {}
        msg = err.get("message", resp.text[:500])
        raise RuntimeError(f"API Gemini ({resp.status_code}): {msg}")
    if not isinstance(data, dict) or "candidates" not in data:
        raise RuntimeError(f"Respuesta inesperada de Gemini: {data!r}")
    candidates = data["candidates"]
    if not candidates:
        raise RuntimeError("Gemini no devolvió candidatos (¿contenido bloqueado?).")
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    out = "".join(texts).strip()
    return out if out else "(Respuesta vacía.)"

def _resumen_distribucion_para_prompt(serie: pd.Series, col: str) -> str:
    """Resumen para la IA: descriptivos, Shapiro-Wilk y conteo IQR de outliers."""
    s = serie.dropna()
    n = int(len(s))
    lines = [
        f'Variable (columna): "{col}"',
        f"Tamaño muestral (sin NaN): n = {n}",
    ]
    if n == 0:
        lines.append("Media: — (sin valores)")
        lines.append("Desviación típica muestral (s): —")
        lines.append("Asimetría (sesgo, Fisher): —")
        lines.append("Shapiro-Wilk: no aplicable (sin datos).")
        lines.append("Outliers (método IQR, Tukey): — (sin datos)")
        return "\n".join(lines)
    media = float(s.mean())
    lines.append(f"Media (x̄): {media:.6f}")
    if n >= 2:
        s_muestral = float(s.std(ddof=1))
        lines.append(f"Desviación típica muestral (s): {s_muestral:.6f}")
    else:
        lines.append("Desviación típica muestral (s): — (n < 2)")
    if n >= 3:
        skewness = float(stats.skew(s, bias=False))
        lines.append(f"Asimetría (sesgo, Fisher — skewness): {skewness:.6f}")
    else:
        lines.append("Asimetría (sesgo, Fisher — skewness): — (n < 3)")
    if n >= 3:
        muestra_sw = s if n <= 5000 else s.sample(5000, random_state=42)
        w_sw, p_sw = stats.shapiro(muestra_sw)
        nota_sw = "" if n <= 5000 else " (submuestra aleatoria n′=5000, semilla 42)"
        lines.append(
            f"Shapiro-Wilk: estadístico W = {float(w_sw):.6f}, "
            f"p-valor = {float(p_sw):.4e}{nota_sw}"
        )
    else:
        lines.append("Shapiro-Wilk: no aplicable (se requieren al menos 3 observaciones).")
    n_out_iqr = _outliers_iqr_count(s)
    pct_out = (100.0 * n_out_iqr / n) if n else 0.0
    lines.append(
        "Outliers (método IQR de Tukey; fuera del intervalo "
        "[Q1 − 1.5·IQR, Q3 + 1.5·IQR]): "
        f"n = {n_out_iqr} ({pct_out:.2f}% de la muestra)"
    )
    return "\n".join(lines)

def render_asistente_gemini(api_key: str) -> None:
    st.subheader("4 · Asistente IA (Gemini)")
    key_ok = bool(api_key and str(api_key).strip())
    tab_dist, tab_z = st.tabs(
        ["💬 Interpretación de Distribución", "📊 Análisis de Prueba Z"]
    )

    instruccion_distribucion = (
        "Actúa como un experto en estadística. Analiza estos resultados y dime si "
        "los datos siguen una distribución normal, qué significa el sesgo encontrado "
        "y qué acciones sugieres."
    )

    with tab_dist:
        if not key_ok:
            st.info("⚠️ Por favor, ingresa tu API Key en la barra lateral")
        else:
            datos = st.session_state.get("datos")
            if datos is None:
                st.warning("Carga datos en la Sección 1 para generar el resumen y consultar a Gemini.")
            else:
                columnas_numericas = datos.select_dtypes(
                    include=["number"]
                ).columns.tolist()
                if not columnas_numericas:
                    st.warning("El conjunto cargado no tiene columnas numéricas.")
                else:
                    default_c = st.session_state.get("columna_analisis")
                    if default_c not in columnas_numericas:
                        default_c = columnas_numericas[0]
                    col = st.selectbox(
                        "Columna para el análisis",
                        columnas_numericas,
                        index=columnas_numericas.index(default_c),
                        key="gemini_dist_col",
                    )
                    serie = datos[col]
                    resumen = _resumen_distribucion_para_prompt(serie, col)
                    st.markdown("**Resumen actual (se envía a Gemini)**")
                    st.code(resumen, language=None)

                    peticion_completa = (
                        f"Aquí tienes el resumen estadístico calculado en la aplicación:\n\n"
                        f"{resumen}\n\n"
                        f"{instruccion_distribucion}"
                    )

                    if st.button("Consultar a Gemini", key="gemini_dist_btn"):
                        try:
                            with st.spinner("Enviando resumen a Gemini…"):
                                answer = _gemini_generate_via_requests(
                                    str(api_key).strip(),
                                    peticion_completa,
                                )
                            with st.container(border=True):
                                st.markdown(
                                    "### 🤖 Análisis de la IA\n\n" + (answer or "")
                                )
                        except Exception as exc:
                            st.error(f"No se pudo obtener respuesta de Gemini: {exc}")

    with tab_z:
        if not key_ok:
            st.info("⚠️ Por favor, ingresa tu API Key en la barra lateral")
        else:
            tipo_opciones = [
                "Bilateral (H₁: μ ≠ μ₀)",
                "Unilateral derecha (H₁: μ > μ₀)",
                "Unilateral izquierda (H₁: μ < μ₀)",
            ]
            zr = st.session_state.get("ultima_prueba_z")
            if zr:
                z_sig = json.dumps(zr, sort_keys=True, default=str)
                if st.session_state.get("_gemini_z_sync_sig") != z_sig:
                    st.session_state["gemini_z_mu0"] = float(zr["mu0"])
                    st.session_state["gemini_z_xbar"] = float(zr["x_bar"])
                    st.session_state["gemini_z_n"] = int(zr["n"])
                    st.session_state["gemini_z_stat"] = float(zr["z_calc"])
                    st.session_state["gemini_z_p"] = float(zr["p_value"])
                    st.session_state["gemini_z_alpha"] = float(zr["alpha"])
                    st.session_state["gemini_z_zcrit"] = float(zr["z_crit"])
                    tp = zr.get("tipo_prueba")
                    if tp in tipo_opciones:
                        st.session_state["gemini_z_tipo"] = tp
                    st.session_state["gemini_z_decision"] = str(zr.get("decision", ""))
                    st.session_state["_gemini_z_sync_sig"] = z_sig
                st.caption(
                    "Hay resultados de la **Sección 3**: los campos se actualizan cuando "
                    "cambia la última prueba Z. Puedes editarlos antes de consultar a Gemini."
                )
            else:
                st.caption(
                    "Ejecuta la prueba Z en la Sección 3 para rellenar automáticamente, "
                    "o introduce los valores a mano."
                )

            def _z_inicial(campo: str, default):
                if campo in st.session_state:
                    return st.session_state[campo]
                return default

            c1, c2 = st.columns(2)
            with c1:
                mu0_ai = st.number_input(
                    "μ₀ (media bajo H₀)",
                    value=_z_inicial("gemini_z_mu0", 75.0),
                    format="%.6f",
                    key="gemini_z_mu0",
                )
                xbar_ai = st.number_input(
                    "x̄ (media muestral)",
                    value=_z_inicial("gemini_z_xbar", 75.0),
                    format="%.6f",
                    key="gemini_z_xbar",
                )
                n_ai = st.number_input(
                    "n (tamaño muestral)",
                    min_value=2,
                    value=int(_z_inicial("gemini_z_n", 30)),
                    step=1,
                    key="gemini_z_n",
                )
            with c2:
                z_ai = st.number_input(
                    "Estadístico Z observado",
                    value=_z_inicial("gemini_z_stat", 0.0),
                    format="%.6f",
                    key="gemini_z_stat",
                )
                p_ai = st.number_input(
                    "Valor p",
                    value=_z_inicial("gemini_z_p", 0.05),
                    format="%.6e",
                    key="gemini_z_p",
                )
                alpha_ai = st.number_input(
                    "α (significancia)",
                    min_value=0.001,
                    max_value=0.5,
                    value=_z_inicial("gemini_z_alpha", 0.05),
                    format="%.4f",
                    key="gemini_z_alpha",
                )
            zcrit_ai = st.number_input(
                "Z crítico",
                value=_z_inicial("gemini_z_zcrit", 0.0),
                format="%.6f",
                key="gemini_z_zcrit",
            )
            _tipo_prev = _z_inicial("gemini_z_tipo", tipo_opciones[0])
            _idx_tipo = (
                tipo_opciones.index(_tipo_prev)
                if _tipo_prev in tipo_opciones
                else 0
            )
            tipo_ai = st.selectbox(
                "Tipo de prueba",
                tipo_opciones,
                index=_idx_tipo,
                key="gemini_z_tipo",
            )
            decision_ai = st.text_input(
                "Decisión (H₀)",
                value=_z_inicial("gemini_z_decision", ""),
                key="gemini_z_decision",
                placeholder="ej.: rechazar H₀ / no rechazar H₀",
            )
            extra_z = st.text_area(
                "Notas o duda adicional (opcional)",
                key="gemini_z_extra",
                placeholder="Ej.: ¿Cómo explico el p-valor en el informe?",
            )
            if st.button("Generar análisis", key="gemini_z_btn"):
                prompt = (
                    "Eres un asistente de estadística. Responde en español con rigor accesible; "
                    "no inventes números fuera del contexto.\n\n"
                    "Un estudiante ejecutó una prueba Z para la media (σ desconocida, "
                    "usando s en el error estándar, como en muchos textos introductorios). "
                    "Explica qué implican los resultados, cómo interpretar el estadístico Z, "
                    "el valor Z crítico, el p-valor frente a α y la decisión sobre H₀. "
                    "Menciona limitaciones (tamaño muestral, aproximación Z vs t si aplica).\n\n"
                    f"μ₀ = {mu0_ai}\n"
                    f"x̄ = {xbar_ai}\n"
                    f"n = {n_ai}\n"
                    f"Z calculado = {z_ai}\n"
                    f"Z crítico = {zcrit_ai}\n"
                    f"p-valor = {p_ai}\n"
                    f"α = {alpha_ai}\n"
                    f"Tipo: {tipo_ai}\n"
                    f"Decisión indicada: {decision_ai.strip() or '(no indicada)'}\n\n"
                    f"Notas del usuario:\n{extra_z.strip() or '(ninguna)'}"
                )
                try:
                    with st.spinner("Consultando Gemini…"):
                        answer = _gemini_generate_via_requests(
                            str(api_key).strip(),
                            prompt,
                        )
                    with st.container(border=True):
                        st.markdown(
                            "### 🤖 Análisis de la IA\n\n" + (answer or "")
                        )
                except Exception as exc:
                    st.error(f"No se pudo obtener respuesta de Gemini: {exc}")

if seccion == SECCIONES[0]:
    st.divider()
    render_carga_datos()
    st.divider()

elif seccion == SECCIONES[1]:
    st.divider()
    if st.session_state["datos"] is None:
        st.warning("⚠️ Por favor, carga datos en la Sección 1 primero.")
    else:
        render_visualizacion(st.session_state["datos"])
    st.divider()

elif seccion == SECCIONES[2]:
    st.divider()
    if st.session_state["datos"] is None:
        st.warning("⚠️ Por favor, carga datos en la Sección 1 primero.")
    else:
        render_prueba_z(st.session_state["datos"])
    st.divider()

else:
    st.divider()
    render_asistente_gemini(gemini_api_key or "")
    st.divider()