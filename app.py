import streamlit as st
import pandas as pd

# Configuración general de la aplicación
st.set_page_config(
    page_title="Perceptrón Simple - Sesión 14 UPN",
    page_icon="🧠",
    layout="wide"
)

# Encabezado principal
st.title("🧠 Red Neuronal Artificial: Perceptrón Simple")
st.markdown("**Universidad Privada del Norte (UPN) | Sistemas Inteligentes - Sesión 14**")
st.caption(
    "Basado en las fórmulas de Net, función escalón, error (Yd - y) y regla delta generalizada "
    "de la Sesión 14[cite: 14]."
)

# --- FUNCIONES DE ENTRENAMIENTO ---
def entrenar_perceptron_3_entradas(w1, w2, w3, wb, fa, max_epocas=50):
    # Patrones para 3 unidades: aprueba (Yd=1) si al menos 2 unidades están aprobadas (x=1)[cite: 17]
    patrones = [
        ("P1", 0, 0, 0, 0),
        ("P2", 0, 0, 1, 0),
        ("P3", 0, 1, 0, 0),
        ("P4", 0, 1, 1, 1),
        ("P5", 1, 0, 0, 0),
        ("P6", 1, 0, 1, 1),
        ("P7", 1, 1, 0, 1),
        ("P8", 1, 1, 1, 1),
    ]

    historial = []
    epoca = 0
    convergio = False

    while not convergio and epoca < max_epocas:
        epoca += 1
        errores = 0
        filas = []

        for nombre, x1, x2, x3, yd in patrones:
            net = round((x1 * w1) + (x2 * w2) + (x3 * w3) + (1 * wb), 4)
            y = 1 if net >= 0 else 0
            error = yd - y

            if error != 0:
                errores += 1
                w1 = round(w1 + fa * error * x1, 4)
                w2 = round(w2 + fa * error * x2, 4)
                w3 = round(w3 + fa * error * x3, 4)
                wb = round(wb + fa * error * 1, 4)
                accion = f"Actualizado -> w1={w1}, w2={w2}, w3={w3}, wb={wb}"
            else:
                accion = "Sin cambios (error = 0)"

            filas.append({
                "Patrón": nombre,
                "x1 (U1)": x1,
                "x2 (U2)": x2,
                "x3 (U3)": x3,
                "Yd": yd,
                "Net": net,
                "y": y,
                "Error": error,
                "Actualización de Pesos": accion
            })

        historial.append((epoca, pd.DataFrame(filas)))
        if errores == 0:
            convergio = True

    return (w1, w2, w3, wb), historial, convergio


def entrenar_perceptron_or(w1, w2, wb, fa, max_epocas=20):
    # Tabla de verdad de la compuerta OR (Patrones 01 a 04)[cite: 15, 16]
    patrones = [
        ("Patrón 01", 0, 0, 0),
        ("Patrón 02", 0, 1, 1),
        ("Patrón 03", 1, 0, 1),
        ("Patrón 04", 1, 1, 1),
    ]

    historial = []
    epoca = 0
    convergio = False

    while not convergio and epoca < max_epocas:
        epoca += 1
        errores = 0
        filas = []

        for patron, x1, x2, yd in patrones:
            net = round((x1 * w1) + (x2 * w2) + (1 * wb), 4)
            y = 1 if net >= 0 else 0
            error = yd - y

            if error != 0:
                errores += 1
                w1 = round(w1 + fa * error * x1, 4)
                w2 = round(w2 + fa * error * x2, 4)
                wb = round(wb + fa * error * 1, 4)
                accion = f"Corregido -> w1={w1}, w2={w2}, wbias={wb}"
            else:
                accion = "No se corrigen pesos (error = 0)"

            filas.append({
                "Patrón": patron,
                "x1": x1,
                "x2": x2,
                "Yd (OR)": yd,
                "Net": net,
                "y": y,
                "Error": error,
                "Actualización de Pesos": accion
            })

        historial.append((epoca, pd.DataFrame(filas)))
        if errores == 0:
            convergio = True

    return (w1, w2, wb), historial, convergio


# --- PESTAÑAS PRINCIPALES ---
tab_ej17, tab_or = st.tabs([
    "🎓 Ejercicio(3 Unidades)",
    "⚡ Compuerta Lógica OR "
])

# ==============================================================================
# PESTAÑA 1: EJERCICIO DIAPOSITIVA 17
# ==============================================================================
with tab_ej17:
    st.info(
        "**Reglas del Ejercicio:** Se consideran 3 unidades (notas de 0 a 20). "
        "Aprueba si tiene al menos 2 unidades aprobadas, sin calcular promedio aritmético[cite: 17]."
    )

    with st.expander("⚙️ Configurar parámetros iniciales del ejercicio (3 unidades)", expanded=False):
        c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)
        w1_e_init = c_p1.number_input("w1 inicial", value=0.5, step=0.1, key="e_w1")
        w2_e_init = c_p2.number_input("w2 inicial", value=0.5, step=0.1, key="e_w2")
        w3_e_init = c_p3.number_input("w3 inicial", value=0.5, step=0.1, key="e_w3")
        wb_e_init = c_p4.number_input("wbias inicial", value=0.5, step=0.1, key="e_wb")
        fa_e = c_p5.number_input("Factor fa", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="e_fa")

    nota_min = st.number_input("Nota mínima aprobatoria por unidad (escala 0–20)", min_value=0, max_value=20, value=11)

    pesos_e_fin, hist_e, conv_e = entrenar_perceptron_3_entradas(
        w1_e_init, w2_e_init, w3_e_init, wb_e_init, fa_e
    )
    w1_ef, w2_ef, w3_ef, wb_ef = pesos_e_fin

    st.subheader("🧪 Simulador de Notas del Estudiante")
    n1, n2, n3 = st.columns(3)
    nota1 = n1.number_input("Unidad 1 (0 - 20)", min_value=0.0, max_value=20.0, value=14.0, step=0.5)
    nota2 = n2.number_input("Unidad 2 (0 - 20)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
    nota3 = n3.number_input("Unidad 3 (0 - 20)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)

    x1_e = 1 if nota1 >= nota_min else 0
    x2_e = 1 if nota2 >= nota_min else 0
    x3_e = 1 if nota3 >= nota_min else 0

    net_e = round((x1_e * w1_ef) + (x2_e * w2_ef) + (x3_e * w3_ef) + (1 * wb_ef), 4)
    y_e = 1 if net_e >= 0 else 0

    col_res_e1, col_res_e2 = st.columns([1.2, 1])
    with col_res_e1:
        st.write(f"**Entradas binarizadas:** `x1 = {x1_e}`, `x2 = {x2_e}`, `x3 = {x3_e}` (Aprobadas: `{x1_e + x2_e + x3_e}/3`)")
        st.latex(rf"\text{{Net}} = ({x1_e} \cdot {w1_ef}) + ({x2_e} \cdot {w2_ef}) + ({x3_e} \cdot {w3_ef}) + (1 \cdot {wb_ef}) = {net_e}")
    with col_res_e2:
        if y_e == 1:
            st.success(f"### ✅ APRUEBA (y = 1)\nNet = {net_e} ≥ 0")
        else:
            st.error(f"### ❌ DESAPRUEBA (y = 0)\nNet = {net_e} < 0")

    st.subheader("📈 Pesos Finales y Épocas (Ejercicio 3 Unidades)")
    if conv_e:
        st.success(f"Convergió en **{len(hist_e)} épocas**.")
    m_e1, m_e2, m_e3, m_e4 = st.columns(4)
    m_e1.metric("w1 final", f"{w1_ef}")
    m_e2.metric("w2 final", f"{w2_ef}")
    m_e3.metric("w3 final", f"{w3_ef}")
    m_e4.metric("wbias final", f"{wb_ef}")

    tabs_e = st.tabs([f"Época {ep}" for ep, _ in hist_e])
    for idx, (ep, df_e) in enumerate(hist_e):
        with tabs_e[idx]:
            st.dataframe(df_e, use_container_width=True, hide_index=True)

# ==============================================================================
# PESTAÑA 2: COMPUERTA LÓGICA OR (INTEGRADA EN LA MISMA APLICACIÓN)
# ==============================================================================
with tab_or:
    st.info(
        "**Compuerta Lógica OR:** Recibe dos entradas binarias ($x_1, x_2$) y entrega `1` si al menos "
        "una entrada es `1`, entregando `0` solo cuando ambas son `0`[cite: 15, 16]."
    )

    with st.expander("⚙️ Configurar parámetros iniciales de la compuerta OR", expanded=False):
        o_p1, o_p2, o_p3, o_p4 = st.columns(4)
        w1_o_init = o_p1.number_input("w1 inicial (OR)", value=0.5, step=0.1, key="o_w1")
        w2_o_init = o_p2.number_input("w2 inicial (OR)", value=0.4, step=0.1, key="o_w2")
        wb_o_init = o_p3.number_input("wbias inicial (OR)", value=1.5, step=0.1, key="o_wb")
        fa_o = o_p4.number_input("Factor fa (OR)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="o_fa")

    pesos_o_fin, hist_o, conv_o = entrenar_perceptron_or(
        w1_o_init, w2_o_init, wb_o_init, fa_o
    )
    w1_of, w2_of, wb_of = pesos_o_fin

    st.subheader("🧪 Probador Interactivo Compuerta OR")
    ox1_col, ox2_col, ores_col = st.columns([1, 1, 1.5])
    with ox1_col:
        x1_o = st.selectbox("Entrada x1", options=[0, 1], index=0, key="sel_x1_or")
    with ox2_col:
        x2_o = st.selectbox("Entrada x2", options=[0, 1], index=1, key="sel_x2_or")

    net_o = round((x1_o * w1_of) + (x2_o * w2_of) + (1 * wb_of), 4)
    y_o = 1 if net_o >= 0 else 0
    yd_o = 1 if (x1_o == 1 or x2_o == 1) else 0

    with ores_col:
        st.latex(rf"\text{{Net}} = ({x1_o} \cdot {w1_of}) + ({x2_o} \cdot {w2_of}) + (1 \cdot {wb_of}) = {net_o}")
        if y_o == 1:
            st.success(f"Salida predicha: **y = 1** (Esperado OR: {yd_o}) — Net ≥ 0")
        else:
            st.error(f"Salida predicha: **y = 0** (Esperado OR: {yd_o}) — Net < 0")

    st.subheader("📈 Pesos Finales y Épocas (Compuerta OR)")
    if conv_o:
        st.success(f"Convergió en **{len(hist_o)} épocas**.")
    mo1, mo2, mo3 = st.columns(3)
    mo1.metric("w1 final (OR)", f"{w1_of}")
    mo2.metric("w2 final (OR)", f"{w2_of}")
    mo3.metric("wbias final (OR)", f"{wb_of}")

    tabs_o = st.tabs([f"Época {ep}" for ep, _ in hist_o])
    for idx, (ep, df_o) in enumerate(hist_o):
        with tabs_o[idx]:
            st.dataframe(df_o, use_container_width=True, hide_index=True)