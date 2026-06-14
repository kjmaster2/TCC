import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Predição de Defeitos",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-weight: bold;
    border-radius: 10px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent

modelo = pickle.load(
    open(BASE_DIR / "previsoes" / "modelo_xgb.pkl", "rb")
)

encoders = pickle.load(
    open(BASE_DIR / "previsoes" / "encoders.pkl", "rb")
)

target_encoder = pickle.load(
    open(BASE_DIR / "previsoes" / "target_encoder.pkl", "rb")
)

features = pickle.load(
    open(BASE_DIR / "previsoes" / "features.pkl", "rb")
)

df = pd.read_csv(
    BASE_DIR / "dados" / "dados_tratados.csv"
)

st.title("Predição de Defeitos com XGBoost")
st.caption("Sistema de apoio à decisão para análise preditiva de defeitos")

dados = {}

st.subheader("Parâmetros de Entrada")

col1, col2 = st.columns(2)

metade = len(features) // 2

with col1:

    with st.container(border=True):

        for coluna in features[:metade]:

            if coluna in encoders:

                valor = st.selectbox(
                    coluna,
                    sorted(df[coluna].astype(str).unique()),
                    key=coluna
                )

                dados[coluna] = encoders[coluna].transform([valor])[0]

            else:

                valor = st.number_input(
                    coluna,
                    value=float(df[coluna].median()),
                    key=coluna
                )

                dados[coluna] = valor

with col2:

    with st.container(border=True):

        for coluna in features[metade:]:

            if coluna in encoders:

                valor = st.selectbox(
                    coluna,
                    sorted(df[coluna].astype(str).unique()),
                    key=coluna
                )

                dados[coluna] = encoders[coluna].transform([valor])[0]

            else:

                valor = st.number_input(
                    coluna,
                    value=float(df[coluna].median()),
                    key=coluna
                )

                dados[coluna] = valor

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "Realizar Previsão",
    type="primary"
):

    entrada = pd.DataFrame([dados])

    pred = modelo.predict(entrada)[0]

    probabilidades = modelo.predict_proba(entrada)[0]

    classe = target_encoder.inverse_transform([pred])[0]

    resultado = pd.DataFrame({
        "Defeito": target_encoder.classes_,
        "Probabilidade (%)": probabilidades * 100
    })

    resultado = resultado.sort_values(
        "Probabilidade (%)",
        ascending=False
    )

    top10 = resultado.head(10)

    st.divider()

    st.subheader("Resultado da Predição")

    col_result1, col_result2 = st.columns([1, 3])

    with col_result1:

        st.metric(
            "Defeito Previsto",
            classe
        )

    with col_result2:

        if classe == "SEM_DEFEITO":
            st.success(
                f"O modelo prevê: {classe}"
            )
        else:
            st.warning(
                f"O modelo prevê: {classe}"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_grafico, col_tabela = st.columns([1, 1])

    with col_grafico:

        st.subheader("Top Probabilidades")

        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True
        )

    with col_tabela:

        st.subheader("Detalhamento")

        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True
        )