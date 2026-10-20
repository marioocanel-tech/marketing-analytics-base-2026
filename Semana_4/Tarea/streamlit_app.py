"""
Recomendador de productos — Laboratorio de Recomendación (Semana 4)

Construida a partir de streamlit_starter.py. Permite a alguien de negocio:
  1) elegir un cliente y un método y ver sus 3 recomendaciones con el "porque";
  2) comparar los recomendadores (hit rate@3, precisión@3 y cobertura) sobre el grupo CONTROL;
  3) entender por qué un hit rate alto no equivale a ingreso extra.

Correr con:  streamlit run streamlit_app.py   (desde la carpeta que contiene data/)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"
DATA = Path(__file__).resolve().parent / "data"     # funciona igual local y en Streamlit Cloud
METODOS = ["Final (ítem + respaldo)", "Filtrado por ítem", "Filtrado por cliente", "SVD", "Popularidad"]


# ----------------------------------------------------------------------------------------------
# Datos (se leen una sola vez)
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando datos…")
def cargar():
    catalogo = pd.read_csv(DATA / "catalogo.csv")
    hist = pd.read_csv(DATA / "ordenes_historial.csv")
    hold = pd.read_csv(DATA / "ordenes_holdout.csv")
    exp = pd.read_csv(DATA / "experimento.csv")
    clientes = pd.read_csv(DATA / "clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return catalogo, productos, M.index.values, X, NUEVO, es_control


catalogo, productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)
tiene_historial = (X.sum(axis=1) > 0)
elegibles = NUEVO.sum(axis=1) > 0
mask_eval = es_control & elegibles                       # control que probó algo nuevo (igual que el notebook)


# ----------------------------------------------------------------------------------------------
# Recomendadores (misma lógica del notebook, calculados para toda la base y guardados en caché)
# ----------------------------------------------------------------------------------------------
@st.cache_data
def similitud_items():
    """Tabla producto × producto (diagonal en 0), calculada UNA sola vez."""
    S = cosine_similarity(X.T)
    np.fill_diagonal(S, 0)
    return S


@st.cache_data(show_spinner="Buscando vecinos…")
def puntajes_cliente(K, bloque=1000):
    """Filtrado por cliente: cada uno de sus K vecinos más parecidos vota con su similitud."""
    out = np.zeros_like(X)
    for ini in range(0, len(X), bloque):
        filas = np.arange(ini, min(ini + bloque, len(X)))
        sims = cosine_similarity(X[filas], X)
        sims[np.arange(len(filas)), filas] = 0                       # un cliente no es su propio vecino
        vec = np.argsort(-sims, axis=1)[:, :K]
        w = np.take_along_axis(sims, vec, axis=1)
        out[filas] = np.einsum("ik,ikj->ij", w, X[vec])
    return out


@st.cache_data
def puntajes_svd(n_factores):
    svd = TruncatedSVD(n_components=n_factores, random_state=42)
    return svd.fit_transform(X) @ svd.components_


def puntajes_todos(metodo, K, n_factores):
    """Matriz de puntajes (clientes × productos) de un método."""
    S_pop = np.tile(popularidad, (len(X), 1))
    if metodo == "Popularidad":
        return S_pop
    S_item = X @ similitud_items()
    if metodo == "Filtrado por ítem":
        return S_item
    if metodo == "Filtrado por cliente":
        return puntajes_cliente(K)
    if metodo == "SVD":
        return puntajes_svd(n_factores)
    return np.where(tiene_historial[:, None], S_item, S_pop)         # Final: ítem + respaldo por popularidad


def top_k_todos(scores, k=3):
    s = np.where(X > 0, -np.inf, scores).astype(np.float64)          # no recomendar lo ya comprado
    s[:, COL_AGUA] = -np.inf                                         # ni el Agua
    return np.argsort(-s, axis=1)[:, :k]


@st.cache_data(show_spinner="Evaluando los recomendadores…")
def comparar(K, n_factores):
    filas = {}
    for m in METODOS:
        top = top_k_todos(puntajes_todos(m, K, n_factores))
        nuevo_en_top = NUEVO[np.arange(len(top))[:, None], top]
        filas[m] = {
            "Hit rate@3": (nuevo_en_top.sum(axis=1) > 0)[mask_eval].mean(),
            "Precisión@3": (nuevo_en_top.sum(axis=1) / 3)[mask_eval].mean(),
            "Cobertura (de 28)": len(np.unique(top)),
        }
    return pd.DataFrame(filas).T


# ----------------------------------------------------------------------------------------------
# Explicación ("porque") de cada recomendación
# ----------------------------------------------------------------------------------------------
def porque(metodo, j, fila, x_vec, K, n_factores):
    S_it = similitud_items()
    if metodo == "Final (ítem + respaldo)":
        metodo = "Filtrado por ítem" if x_vec.sum() > 0 else "Popularidad"
    if metodo == "Popularidad":
        return f"Es de lo más comprado: lo ha comprado el {popularidad[j] / len(X):.0%} de los clientes."
    if metodo == "Filtrado por ítem":
        compradas = np.where(x_vec > 0)[0]
        contrib = S_it[j, compradas]
        orden = np.argsort(-contrib)[:2]
        partes = [f"{productos[compradas[o]]} (similitud {contrib[o]:.2f})" for o in orden if contrib[o] > 0]
        return ("Porque compró " + " y ".join(partes) + ", que suelen comprarse junto con él.") if partes else \
            "Sin señal: este cliente no ha comprado nada parecido."
    if metodo == "Filtrado por cliente":
        sims = cosine_similarity(x_vec[None, :], X)[0]
        sims[fila] = 0
        vec = np.argsort(-sims)[:K]
        n = int(X[vec, j].sum())
        return f"{n} de sus {K} vecinos más parecidos (clientes con compras similares) lo compraron."
    return (f"Encaja con su perfil de gustos ({n_factores} factores latentes). "
            "Es difícil de explicar en palabras: ese es el costo de este método.")


# ----------------------------------------------------------------------------------------------
# Encabezado y controles (barra lateral)
# ----------------------------------------------------------------------------------------------
st.title("☕ Recomendador de productos")
st.caption("Ayuda a Marketing a decidir **qué recomendador poner en el carrusel «Para ti»**: muestra qué le "
           "recomendaría cada método a un cliente (y por qué) y cómo se comparan entre sí.")

with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox("Método de recomendación", METODOS, index=0,
                          help="«Final» usa filtrado por ítem y, para clientes sin historial, popularidad.")
    K = st.slider("Vecinos (K) del filtrado por cliente", 5, 200, 30, step=5,
                  help="Cuántos clientes parecidos «votan» por cada producto.")
    n_factores = st.slider("Factores del SVD", 2, 20, 8,
                           help="Cuántos «gustos escondidos» resume el modelo de factorización.")
    st.divider()
    st.caption("Los parámetros afectan a las pestañas **Recomendar** y **Comparar métodos**. "
               "Las métricas se calculan solo sobre el grupo control (sin contaminación del carrusel).")

tab1, tab2, tab3 = st.tabs(["Recomendar", "Comparar métodos", "Hit rate ≠ ingreso extra"])

# ----------------------------------------------------------------------------------------------
# Pestaña 1: recomendaciones para un cliente
# ----------------------------------------------------------------------------------------------
with tab1:
    primer_sin_historial = int(ids[~tiene_historial][0])
    c1, c2 = st.columns([2, 3])
    with c1:
        ejemplo_frio = st.checkbox(f"Ver un cliente sin historial (ID {primer_sin_historial})")
        cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()),
                              value=primer_sin_historial if ejemplo_frio else 14, step=1,
                              disabled=ejemplo_frio)
        if ejemplo_frio:
            cid = primer_sin_historial
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    compradas = [productos[j] for j in np.where(x_vec > 0)[0]]
    with c2:
        st.markdown(f"**Cliente {cid}** · {len(compradas)} productos distintos comprados antes")
        st.write(", ".join(compradas) if compradas else "_Sin compras previas (cliente nuevo)._")

    scores = puntajes_todos(metodo, K, n_factores)[fila]
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)
    s[COL_AGUA] = -np.inf
    top = np.argsort(-s)[:3]

    sin_senal = (metodo in ("Filtrado por ítem", "Filtrado por cliente", "SVD")) and not compradas
    if sin_senal:
        st.warning("Este cliente no tiene historial: el método colaborativo da puntaje 0 a todos los productos, "
                   "así que estas recomendaciones **no tienen señal**. El método «Final» resuelve esto "
                   "recomendando lo más popular.")
    elif metodo == "Final (ítem + respaldo)" and not compradas:
        st.info("Cliente sin historial: el método «Final» usa **popularidad** como respaldo.")

    st.subheader(f"Las 3 recomendaciones · {metodo}")
    cols = st.columns(3)
    for pos, (col, j) in enumerate(zip(cols, top), start=1):
        fila_cat = catalogo.iloc[j]
        with col, st.container(border=True):
            st.markdown(f"#### {pos}. {productos[j]}")
            st.caption(f"{fila_cat['categoria']} · US$ {fila_cat['precio']:.2f} · puntaje {scores[j]:.2f}")
            if sin_senal:
                st.write("Sin señal: el método no tiene historial de este cliente.")
            else:
                st.write(porque(metodo, j, fila, x_vec, K, n_factores))

# ----------------------------------------------------------------------------------------------
# Pestaña 2: comparación de métodos
# ----------------------------------------------------------------------------------------------
with tab2:
    st.subheader("¿Qué tan bien acierta cada método?")
    st.write(f"Se evalúa con los **{int(mask_eval.sum()):,} clientes del grupo control** que probaron algún "
             "producto nuevo en los 90 días del experimento. Un *acierto* es que al menos una de las 3 "
             "recomendaciones fuera un producto nuevo que el cliente terminó comprando.")
    res = comparar(K, n_factores)
    res.index = [f"{m} (K={K})" if m == "Filtrado por cliente" else
                 f"{m} ({n_factores} factores)" if m == "SVD" else m for m in res.index]
    st.dataframe(
        res.style.format({"Hit rate@3": "{:.1%}", "Precisión@3": "{:.1%}", "Cobertura (de 28)": "{:.0f}"})
           .highlight_max(subset=["Hit rate@3", "Precisión@3", "Cobertura (de 28)"], color="#CFE8DD"),
        width="stretch")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Hit rate@3 en control")
        st.bar_chart(res["Hit rate@3"], color=VERDE, horizontal=True)
    with g2:
        st.caption("Cobertura: productos distintos recomendados (de 28)")
        st.bar_chart(res["Cobertura (de 28)"], color=NARANJA, horizontal=True)
    st.info("Mueve **K** y **factores** en la barra lateral para ver cómo cambian los métodos colaborativos. "
            "Un hit rate alto indica que el método *predice* bien lo que el cliente compraría; "
            "no dice cuánto ingreso **extra** genera la recomendación (ver la pestaña siguiente).")

# ----------------------------------------------------------------------------------------------
# Pestaña 3: hit rate vs ingreso incremental (resultados del notebook, sección «La revelación final»)
# ----------------------------------------------------------------------------------------------
with tab3:
    st.subheader("Un hit rate alto no significa ingreso extra")
    st.write("Cifras del notebook (sección *La revelación final*). Usan columnas ocultas del generador de datos "
             "(`p_organica` y `p_incremental`) que **no existen en una empresa real**.")
    rev = pd.DataFrame({
        "Hit rate@3 (control)": [0.221, 0.401, 0.299, 0.369, 0.410],
        "Ingreso incremental (US$/cliente, 90 días)": [0.998, 1.191, 1.186, 1.213, 1.181],
        "% de compras acertadas que ya iban a ocurrir": [0.307, 0.438, 0.376, 0.411, 0.443],
    }, index=["Popularidad", "Filtrado por ítem", "Filtrado por cliente (K=30)", "SVD (8 factores)",
              "Final (ítem + respaldo)"])
    st.dataframe(rev.style.format({"Hit rate@3 (control)": "{:.3f}",
                                   "Ingreso incremental (US$/cliente, 90 días)": "{:.3f}",
                                   "% de compras acertadas que ya iban a ocurrir": "{:.1%}"}),
                 width="stretch")
    indice = pd.DataFrame({
        "Hit rate (popularidad = 100)": rev.iloc[:, 0] / rev.iloc[0, 0] * 100,
        "Ingreso incremental (popularidad = 100)": rev.iloc[:, 1] / rev.iloc[0, 1] * 100,
    })
    st.bar_chart(indice, color=[VERDE, NARANJA], horizontal=True)
    st.markdown(
        "- El **filtrado por ítem** supera a popularidad en hit rate por ≈ +81 %, pero en ingreso incremental "
        "solo por ≈ +19 %.\n"
        "- Entre **31 % y 44 %** de las compras acertadas **ya iban a ocurrir** sin carrusel: cuanto mejor "
        "«adivina» el método, más compras regaladas contabiliza.\n"
        "- Por eso el valor del carrusel se mide con un **experimento con grupo de control**, no con el hit rate.")
