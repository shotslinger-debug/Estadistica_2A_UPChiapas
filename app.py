import streamlit as st
import pandas as pd

st.set_page_config(page_title="Probabilidad y estadistica - 2A", layout="centered")
st.title("Probabilidad y estadistica - 2A")
st.write("Bienvenido a tu aplicacion basica de Streamlit.")

archivo_csv = st.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo_csv is not None:
    datos = pd.read_csv(archivo_csv)
    st.success("Archivo cargado con exito")
    st.dataframe(datos.head(5))

    columnas_numericas = datos.select_dtypes(include=["number"]).columns.tolist()

    if columnas_numericas:
        columna_seleccionada = st.selectbox("Selecciona una columna numerica", columnas_numericas)
        st.write(f"Has seleccionado la columna: {columna_seleccionada}")
        st.subheader("Estadisticas Descriptivas")
        st.table(datos[columna_seleccionada].describe().to_frame(name="valor"))
    else:
        st.warning("El archivo no contiene columnas numericas.")
