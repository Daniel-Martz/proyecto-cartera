# proyecto-cartera
# Analizador de carteras

App en Streamlit que descarga precios de ETFs, calcula rentabilidad y riesgo,
optimiza la cartera con Markowitz y recomienda una según el perfil del inversor.

## Instalación

    git clone https://github.com/Daniel-Martz/proyecto-cartera.git
    cd proyecto-cartera
    python3 -m venv venv
    source venv/bin/activate   # En Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Ejecución

    streamlit run app.py

## Qué hace

- Descarga datos reales con yfinance
- Calcula rentabilidad, volatilidad, Sharpe y correlaciones
- Optimiza carteras (máximo Sharpe y mínima volatilidad)
- Cuestionario de perfil de riesgo con recomendación

