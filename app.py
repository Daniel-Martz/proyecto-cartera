import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize

st.set_page_config(page_title="Analizador de carteras", layout="wide")
st.title("📈 Analizador de carteras")

# --- Barra lateral: entradas del usuario ---
st.sidebar.header("Configuración")
tickers_texto = st.sidebar.text_input(
    "ETFs (separados por comas)", "VOO, VEA, AGG, GLD"
)
fecha_inicio = st.sidebar.date_input("Desde", pd.to_datetime("2019-01-01"))

tickers = [t.strip().upper() for t in tickers_texto.split(",") if t.strip()]

# --- Descarga de datos (con caché para que no repita la descarga) ---
@st.cache_data
def cargar_precios(tickers, inicio):
    return yf.download(tickers, start=inicio, auto_adjust=True)["Close"]

if len(tickers) < 2:
    st.warning("Introduce al menos 2 ETFs.")
    st.stop()

precios = cargar_precios(tickers, fecha_inicio)
if precios.empty or precios.dropna(how="all").shape[1] < 2:
    st.error("No se pudieron descargar datos. Revisa los tickers.")
    st.stop()
precios = precios.dropna(axis=1, how="all")

rent_diaria = precios.pct_change().dropna()

# --- Métricas ---
rent_anual = rent_diaria.mean() * 252
volatilidad = rent_diaria.std() * np.sqrt(252)
sharpe = rent_anual / volatilidad

resumen = pd.DataFrame({
    "Rentabilidad anual": rent_anual,
    "Volatilidad": volatilidad,
    "Sharpe": sharpe,
})

st.subheader("Métricas por activo")
st.dataframe(resumen.style.format("{:.2%}", subset=["Rentabilidad anual", "Volatilidad"])
                          .format("{:.2f}", subset=["Sharpe"]))

st.subheader("Correlaciones")
st.dataframe(rent_diaria.corr().round(2))
# --- Gráfica 1: evolución normalizada (base 100) ---
st.subheader("Evolución de cada ETF (base 100)")
base100 = precios / precios.iloc[0] * 100
fig1 = px.line(base100, labels={"value": "Valor (base 100)", "Date": "Fecha", "Ticker": "ETF"})
st.plotly_chart(fig1, use_container_width=True)

# --- Gráfica 2: mapa de calor de correlaciones ---
st.subheader("Mapa de correlaciones")
fig2 = px.imshow(
    rent_diaria.corr().round(2),
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
)
st.plotly_chart(fig2, use_container_width=True)

# --- Gráfica 3: riesgo vs rentabilidad ---
st.subheader("Riesgo vs rentabilidad")
fig3 = px.scatter(
    resumen.reset_index(),
    x="Volatilidad", y="Rentabilidad anual", text="Ticker", size_max=20,
)
fig3.update_traces(textposition="top center", marker=dict(size=14))
fig3.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
st.plotly_chart(fig3, use_container_width=True)


# ============ OPTIMIZACIÓN DE CARTERA (MARKOWITZ) ============
st.header("Optimización de cartera")

# Controles en la barra lateral
tasa_libre = st.sidebar.number_input(
    "Tasa libre de riesgo (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1
) / 100

mu = rent_diaria.mean() * 252          # rentabilidad anual esperada de cada activo
cov = rent_diaria.cov() * 252          # matriz de covarianzas anualizada
n = len(mu)

peso_max = st.sidebar.slider(
    "Peso máximo por activo (%)", min_value=int(np.ceil(100 / n)), max_value=100, value=100
) / 100

def rent_cartera(w):
    return w @ mu.values

def vol_cartera(w):
    return np.sqrt(w @ cov.values @ w)

def sharpe_negativo(w):
    return -(rent_cartera(w) - tasa_libre) / vol_cartera(w)

restricciones = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)  # los pesos suman 100%
limites = tuple((0, peso_max) for _ in range(n))                   # sin posiciones cortas
w0 = np.repeat(1 / n, n)                                           # punto de partida: igual peso

res_sharpe = minimize(sharpe_negativo, w0, method="SLSQP", bounds=limites, constraints=restricciones)
res_minvol = minimize(vol_cartera, w0, method="SLSQP", bounds=limites, constraints=restricciones)

carteras = {
    "Máximo Sharpe": res_sharpe.x,
    "Mínima volatilidad": res_minvol.x,
    "Pesos iguales": w0,
}

# Tabla comparativa de las carteras
filas = []
for nombre, w in carteras.items():
    r, v = rent_cartera(w), vol_cartera(w)
    filas.append({
        "Cartera": nombre,
        "Rentabilidad anual": r,
        "Volatilidad": v,
        "Sharpe": (r - tasa_libre) / v,
    })
tabla_carteras = pd.DataFrame(filas).set_index("Cartera")

st.subheader("Comparativa de carteras")
st.dataframe(tabla_carteras.style.format("{:.2%}", subset=["Rentabilidad anual", "Volatilidad"])
                                .format("{:.2f}", subset=["Sharpe"]))

# Pesos de cada cartera
st.subheader("Composición de las carteras")
columnas = st.columns(2)
for col, nombre in zip(columnas, ["Máximo Sharpe", "Mínima volatilidad"]):
    pesos = pd.DataFrame({"ETF": mu.index, "Peso": carteras[nombre]})
    pesos = pesos[pesos["Peso"] > 0.001]
    fig = px.pie(pesos, names="ETF", values="Peso", title=nombre, hole=0.4)
    col.plotly_chart(fig, use_container_width=True)

# Nube de carteras aleatorias + las óptimas
st.subheader("Frontera eficiente")
rng = np.random.default_rng(42)
pesos_rand = rng.dirichlet(np.ones(n), size=3000)
pesos_rand = pesos_rand[pesos_rand.max(axis=1) <= peso_max]   # respeta el peso máximo
nube = pd.DataFrame({
    "Volatilidad": [vol_cartera(w) for w in pesos_rand],
    "Rentabilidad": [rent_cartera(w) for w in pesos_rand],
})
nube["Sharpe"] = (nube["Rentabilidad"] - tasa_libre) / nube["Volatilidad"]

fig_f = px.scatter(nube, x="Volatilidad", y="Rentabilidad", color="Sharpe",
                   color_continuous_scale="Viridis", opacity=0.6)
for nombre, color in [("Máximo Sharpe", "red"), ("Mínima volatilidad", "orange")]:
    w = carteras[nombre]
    fig_f.add_scatter(x=[vol_cartera(w)], y=[rent_cartera(w)], mode="markers+text",
                      text=[nombre], textposition="top center",
                      marker=dict(size=16, color=color, symbol="star"), name=nombre)
fig_f = px.scatter(nube, x="Volatilidad", y="Rentabilidad", color="Sharpe",
                   color_continuous_scale="Viridis", opacity=0.6)
for nombre, color in [("Máximo Sharpe", "red"), ("Mínima volatilidad", "orange")]:
    w = carteras[nombre]
    fig_f.add_scatter(x=[vol_cartera(w)], y=[rent_cartera(w)], mode="markers+text",
                      text=[nombre], textposition="top center",
                      marker=dict(size=16, color=color, symbol="star"),
                      showlegend=False)   # quitamos la leyenda duplicada
fig_f.update_layout(
    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
    xaxis_title="Volatilidad (riesgo)", yaxis_title="Rentabilidad anual",
    coloraxis_colorbar=dict(title="Sharpe"),
    margin=dict(t=30),
)
st.plotly_chart(fig_f, use_container_width=True)

# ============ PERFIL DE RIESGO Y RECOMENDACIÓN ============
st.header("Tu perfil de inversor")

p1 = st.radio("¿Cuándo necesitarás este dinero?",
              ["Menos de 3 años", "Entre 3 y 10 años", "Más de 10 años"], index=1)
p2 = st.radio("Tu inversión cae un 20 % en un mes. ¿Qué haces?",
              ["Vendo todo", "No hago nada", "Compro más"], index=1)
p3 = st.radio("¿Qué prefieres?",
              ["Ganar poco pero seguro", "Un equilibrio", "Ganar mucho aunque arriesgue"], index=1)

puntos = (["Menos de 3 años", "Entre 3 y 10 años", "Más de 10 años"].index(p1)
          + ["Vendo todo", "No hago nada", "Compro más"].index(p2)
          + ["Ganar poco pero seguro", "Un equilibrio", "Ganar mucho aunque arriesgue"].index(p3))

w_sharpe, w_minvol = carteras["Máximo Sharpe"], carteras["Mínima volatilidad"]
if puntos <= 2:
    perfil, w_rec = "Conservador", w_minvol
elif puntos <= 4:
    perfil, w_rec = "Moderado", 0.5 * w_sharpe + 0.5 * w_minvol
else:
    perfil, w_rec = "Arriesgado", w_sharpe

r_rec, v_rec = rent_cartera(w_rec), vol_cartera(w_rec)
pesos_rec = pd.Series(w_rec, index=mu.index).sort_values(ascending=False)
pesos_rec = pesos_rec[pesos_rec > 0.01]

st.subheader(f"Perfil: {perfil}")
col1, col2 = st.columns([1, 1])
col1.plotly_chart(px.pie(names=pesos_rec.index, values=pesos_rec.values, hole=0.4,
                         title="Cartera recomendada"), use_container_width=True)

# Explicación generada con reglas (sin API)
principal = pesos_rec.index[0]
peor_caida = vol_cartera(w_rec) * 2  # aprox. un año malo = 2 desviaciones
explicacion = f"""
**¿Qué significa esto?**

- Por tu perfil **{perfil.lower()}**, te sale una cartera con **{pesos_rec.iloc[0]:.0%} en {principal}**
  y el resto repartido entre {", ".join(pesos_rec.index[1:]) or "ningún otro activo"}.
- Con los datos desde {fecha_inicio}, esta mezcla habría dado **{r_rec:.1%} de rentabilidad anual**
  con una volatilidad (riesgo) del **{v_rec:.1%}**.
- En la práctica, un mal año podría suponer una caída cercana al **{peor_caida:.0%}** respecto a la media.
- Repartir el dinero entre activos que no se mueven igual (por ejemplo bolsa, bonos y oro)
  reduce el riesgo sin renunciar a toda la rentabilidad. Eso es la **diversificación**.
"""
col2.markdown(explicacion)

st.warning("Herramienta educativa. No es asesoramiento financiero. "
           "Las rentabilidades pasadas no garantizan rentabilidades futuras.")