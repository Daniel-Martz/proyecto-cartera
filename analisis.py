import yfinance as yf
import pandas as pd
import numpy as np

# 1. ETFs a analizar (puedes cambiarlos luego)
tickers = ["VOO", "VEA", "AGG", "GLD"]

# 2. Descargar precios de cierre ajustados
precios = yf.download(tickers, start="2019-01-01", auto_adjust=True)["Close"]

# 3. Rentabilidad diaria: cuánto sube o baja cada día en %
rent_diaria = precios.pct_change().dropna()

# 4. Métricas anualizadas (hay ~252 días de bolsa al año)
rent_anual = rent_diaria.mean() * 252
volatilidad = rent_diaria.std() * np.sqrt(252)
sharpe = rent_anual / volatilidad  # simplificado, sin tasa libre de riesgo

resumen = pd.DataFrame({
    "Rentabilidad anual": rent_anual,
    "Volatilidad": volatilidad,
    "Sharpe": sharpe,
})

print(resumen.round(3))
print("\nCorrelaciones:")
print(rent_diaria.corr().round(2))