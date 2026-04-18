import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

CHART_FACE = "#1A1D27"
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
      .sec2-metric-cards-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0.75rem 0 1.25rem 0;
      }
      .sec2-metric-card {
        flex: 1 1 220px;
        border: 1px solid #2196F3;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        background: rgba(33, 150, 243, 0.08);
        box-shadow: 0 0 0 1px rgba(33, 150, 243, 0.18);
      }
      .sec2-metric-card .mc-label {
        display: block;
        font-size: 0.82rem;
        color: #90CAF9;
        margin-bottom: 0.35rem;
      }
      .sec2-metric-card .mc-value {
        font-size: 1.45rem;
        font-weight: 700;
        font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
        color: #E3F2FD;
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

st.sidebar.markdown("## 📊 Estadística — 2A")
st.sidebar.divider()
seccion = st.sidebar.radio(
    "Selecciona una sección",
    SECCIONES,
)

head_l, head_r = st.columns([4, 1.15])
with head_l:
    st.title("Probabilidad y estadistica - 2A")
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

st.write("Bienvenido a tu aplicacion basica de Streamlit.")


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
            media = st.slider("Media (μ)", -10.0, 10.0, 5.0, 0.1)
            sigma = st.slider("Desviación típica (σ)", 0.05, 5.0, 1.25, 0.05)

        np.random.seed(int(semilla))
        muestra = np.random.normal(loc=media, scale=sigma, size=int(n_muestra))
        datos = pd.DataFrame({"Valor_Generado": muestra})
        datos.index = datos.index + 1
        st.session_state["datos"] = datos
        st.success("Datos sintéticos generados correctamente.")

    datos = st.session_state["datos"]
    if datos is not None:
        st.caption(f"Registros: **{len(datos)}** · Índice desde 1")
        st.dataframe(datos, use_container_width=True)


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
    st.write(f"Has seleccionado la columna: {columna_seleccionada}")

    st.subheader("Estadisticas Descriptivas")
    st.table(datos[columna_seleccionada].describe().to_frame(name="valor"))

    serie = datos[columna_seleccionada].dropna()

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

    skewness = float(stats.skew(serie, bias=False))
    if len(serie) < 3:
        st.warning("Se requieren al menos 3 observaciones para Shapiro-Wilk.")
        p_html = "—"
    else:
        muestra_sw = (
            serie if len(serie) <= 5000 else serie.sample(5000, random_state=42)
        )
        if len(serie) > 5000:
            st.caption(
                "Shapiro-Wilk: n > 5000; se usa submuestra aleatoria de 5000 (semilla 42)."
            )
        _, p_sw = stats.shapiro(muestra_sw)
        p_html = f"{p_sw:.4e}"

    st.markdown(
        f"""
        <div class="sec2-metric-cards-row">
          <div class="sec2-metric-card">
            <span class="mc-label">Asimetría (skewness)</span>
            <span class="mc-value">{skewness:.6f}</span>
          </div>
          <div class="sec2-metric-card">
            <span class="mc-label">Shapiro-Wilk: valor p</span>
            <span class="mc-value">{p_html}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    se = s_muestral / np.sqrt(n_obs)
    z_obs = (xbar - mu0) / se
    if tipo_prueba.startswith("Bilateral"):
        p_valor = 2 * min(stats.norm.cdf(z_obs), stats.norm.sf(z_obs))
    elif "derecha" in tipo_prueba:
        p_valor = float(stats.norm.sf(z_obs))
    else:
        p_valor = float(stats.norm.cdf(z_obs))

    rechazar = p_valor < alpha_z

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Media muestral (x̄)", f"{xbar:.6f}")
    with m2:
        st.metric("Estadístico Z", f"{z_obs:.6f}")
    with m3:
        st.metric("Valor p", f"{p_valor:.4e}")

    st.caption(
        "Z = (x̄ − μ₀) / (s / √n); p-valor bajo N(0,1) (σ desconocida, s como aproximación)."
    )

    x_lo = min(-4.0, float(z_obs) - 1.5)
    x_hi = max(4.0, float(z_obs) + 1.5)
    x = np.linspace(x_lo, x_hi, 500)
    y = stats.norm.pdf(x)

    fig_z, ax_z = plt.subplots(figsize=(10, 4.5))
    fig_z.patch.set_facecolor(CHART_FACE)
    ax_z.set_facecolor(CHART_FACE)
    ax_z.plot(x, y, color="#ECEFF1", linewidth=2)

    if tipo_prueba.startswith("Bilateral"):
        z_crit = float(stats.norm.ppf(1 - alpha_z / 2))
        ax_z.fill_between(x, y, where=x <= -z_crit, color="red", alpha=0.35)
        ax_z.fill_between(x, y, where=x >= z_crit, color="red", alpha=0.35)
    elif "derecha" in tipo_prueba:
        z_crit = float(stats.norm.ppf(1 - alpha_z))
        ax_z.fill_between(x, y, where=x >= z_crit, color="red", alpha=0.35)
    else:
        z_crit = float(stats.norm.ppf(alpha_z))
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
    st.header("Próximamente")
    st.write(
        "Aquí se integrará un **asistente de IA con Google Gemini** para interpretar "
        "resultados, sugerir análisis y responder dudas sobre tus datos estadísticos."
    )
    st.divider()
