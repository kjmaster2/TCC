import streamlit as st
import pandas as pd
from previsoes.apriori_model import executar_apriori
from previsoes.logistic_analysis import executar_regressao_logistica_analitica
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="Análise por Defeito",layout="wide",page_icon="🔎")

st.title("🔎 Análise de Padrões por Defeito")
st.divider()

# =========================================================
# LOAD DATA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():

    arquivo = BASE_DIR / "dados" / "problemas_filtrados.csv"

    df = pd.read_csv(arquivo)
    df.columns = df.columns.str.strip()

    # DATA
    if {"ANO","MES","DIA"}.issubset(df.columns):
        df["DATA"] = pd.to_datetime({
            "year": df["ANO"],
            "month": df["MES"],
            "day": df["DIA"]
        })

    # =====================================================
    # CRIAR AGRUPAMENTOS
    # =====================================================

    df["BITOLA_GRUPO"] = pd.cut(
        df["BITOLA"],
        bins=[0, 8, 14, 100],
        labels=["PEQUENA","MEDIA","GRANDE"]
    )

    df["COMP_GRUPO"] = pd.cut(
        df["COMPRIMENTO"],
        bins=[0, 40, 80, 1000],
        labels=["CURTO","MEDIO","LONGO"]
    )

    return df

df = load_data()

# =========================================================
# DEFEITOS
# =========================================================

col_inicio = "PECA_EMPENADA/TORTA"
col_fim = "ARRUELA_DEFORMADA/BATIDA"

defeitos = df.loc[:, col_inicio:col_fim].columns.tolist()

# =========================================================
# INTERFACE GLOBAL
# =========================================================

st.subheader("⚙️ Configuração Geral")

modo = st.radio("Modo de análise", ["Agrupado", "Detalhado"], horizontal=True)

defeito = st.selectbox("Defeito", defeitos)

# variáveis conforme modo
if modo == "Detalhado":
    variaveis = st.multiselect(
        "Variáveis",
        ["FAMILIA", "BITOLA", "MAQUINA", "MATERIA_PRIMA", "COMPRIMENTO"],
        default=["FAMILIA", "BITOLA", "MAQUINA", "MATERIA_PRIMA", "COMPRIMENTO"]
    )
else:
    variaveis = st.multiselect(
        "Variáveis",
        ["FAMILIA", "BITOLA_GRUPO", "COMP_GRUPO", "MAQUINA", "MATERIA_PRIMA"],
        default=["FAMILIA", "BITOLA_GRUPO", "COMP_GRUPO", "MAQUINA", "MATERIA_PRIMA"]
    )

if not variaveis:
    st.warning("Selecione ao menos uma variável")
    st.stop()

st.divider()

# =========================================================
# ==================== APRIORI =============================
# =========================================================

st.subheader("🧠 Apriori - Regras de Associação")

c1, c2, c3 = st.columns(3)

min_lift = c1.slider("Lift mínimo", 1.00, 5.0, 1.0, 0.1)
min_conf = c2.slider("Confiança mínima", 0.000, 1.000, 0.005, 0.001)
min_support = c3.slider("Suporte mínimo", 0.001, 1.0, 0.001, 0.001)

regras = None

with st.spinner("Executando Apriori..."):
    try:
        regras = executar_apriori(
            df,
            defeito,
            variaveis,
            min_support=min_support,
            min_confidence=min_conf,
            min_lift=min_lift
        )
    except Exception as e:
        st.error(f"Erro Apriori: {e}")

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def formatar_itemset(itemset):
    if itemset is None:
        return ""

    if isinstance(itemset, (set, frozenset, list, tuple)):
        itens = [str(i).strip() for i in itemset]
        return ", ".join(itens)

    return str(itemset).strip()


def interpretar_lift(lift):
    if lift >= 2.0:
        return "forte associação"
    elif lift >= 1.2:
        return "associação moderada"
    else:
        return "associação fraca"


def interpretar_confiança(confidence):
    if confidence >= 0.8:
        return "muito alta"
    elif confidence >= 0.6:
        return "alta"
    elif confidence >= 0.4:
        return "moderada"
    else:
        return "baixa"


def interpretar_suporte(support):
    if support >= 0.10:
        return "frequente na base analisada"
    elif support >= 0.05:
        return "presente com frequência intermediária"
    elif support >= 0.01:
        return "presente com baixa frequência"
    else:
        return "pouco recorrente na base analisada"


def gerar_texto_regra(antecedentes, consequentes, support, confidence, lift):
    ant = formatar_itemset(antecedentes)
    cons = formatar_itemset(consequentes)

    texto = (
        f"Quando ocorre **{ant}**, observa-se associação com **{cons}**.\n\n"

        f"• **Suporte ({support:.2%})**: indica a frequência com que essa combinação aparece nos dados. "
        f"Quanto maior o suporte, mais comum é essa situação no processo.\n\n"

        f"• **Confiança ({confidence:.2%})**: representa a probabilidade de o defeito ocorrer quando essa condição está presente. "
        f"Valores mais altos indicam maior chance de ocorrência.\n\n"

        f"• **Lift ({lift:.2f})**: mede a força da relação entre as variáveis. "
        f"Valores acima de 1 indicam que existe associação; quanto maior o valor, mais forte é essa relação.\n"
    )
    return texto


# =========================================================
# RESULTADO
# =========================================================

if regras is not None and not regras.empty and "lift" in regras.columns:

    regras_filtradas = regras[
        (regras["lift"] >= min_lift) &
        (regras["confidence"] >= min_conf) &
        (regras["support"] >= min_support)
    ].copy()

    if not regras_filtradas.empty:

        regras_filtradas = regras_filtradas.sort_values(
            by=["lift", "confidence", "support"],
            ascending=False
        ).reset_index(drop=True)

        st.markdown("### 📌 Interpretação das regras")

        # =====================================================
        # CONTAINER COM SCROLL NATIVO
        # =====================================================

        container_regras = st.container(height=700)

        with container_regras:

            for i, row in regras_filtradas.iterrows():

                titulo = (
                    f"Regra {i + 1}: "
                    f"{formatar_itemset(row['antecedents'])} "
                    f"→ "
                    f"{formatar_itemset(row['consequents'])}"
                )

                with st.expander(titulo):

                    st.markdown(
                        gerar_texto_regra(
                            row["antecedents"],
                            row["consequents"],
                            row["support"],
                            row["confidence"],
                            row["lift"]
                        )
                    )

                    st.caption(
                        f"Suporte: {row['support']:.2%} | "
                        f"Confiança: {row['confidence']:.2%} | "
                        f"Lift: {row['lift']:.2f}"
                    )

        # =====================================================
        # TABELA COMPLETA
        # =====================================================

        with st.expander("Ver tabela completa das regras"):

            st.dataframe(
                regras_filtradas[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift"
                    ]
                ],
                use_container_width=True,
                height=500
            )

    else:

        st.info(
            "Nenhuma regra encontrada com os filtros definidos."
        )

else:

    st.warning(
        "Não foi possível gerar regras de associação."
    )


st.divider()

# =========================================================
# REGRESSÃO LOGÍSTICA ANALÍTICA
# =========================================================

st.subheader(
    "📈 Regressão Logística Binária - Explicativa"
)

resultado_logistico = None

with st.spinner(
    "Executando regressão logística..."
):

    try:

        resultado_logistico = (

            executar_regressao_logistica_analitica(

                df=df,

                defeito=defeito,

                variaveis=variaveis,

                regras_apriori=regras_filtradas
            )
        )

    except Exception as e:

        st.error(
            f"Erro regressão logística: {e}"
        )

# =====================================================
# RESULTADOS
# =====================================================

# =====================================================
# RESULTADOS
# =====================================================

if resultado_logistico is not None:

    tabela = resultado_logistico["tabela"].copy()

    # =====================================================
    # RENOMEIA COLUNAS
    # =====================================================

    tabela = tabela.rename(columns={

        "Variavel": "Variável",

        "Odds_Ratio": "Chance Relativa",

        "Direcao": "Influência"
    })

    # =====================================================
    # REMOVE COLUNAS DESNECESSÁRIAS
    # =====================================================

    tabela = tabela[
        [
            "Variável",
            "Chance Relativa",
            "Influência"
        ]
    ]

    # =====================================================
    # FILTRA ODDS RATIO EXTREMOS
    # =====================================================

    tabela = tabela[

        (tabela["Chance Relativa"] < 20)

        &

        (tabela["Chance Relativa"] > 0.05)
    ]

    # =====================================================
    # INTERPRETAÇÃO AUTOMÁTICA
    # =====================================================

    def interpretar_odds(valor):

        if valor >= 3:
            return "Alta associação"

        elif valor >= 1.5:
            return "Associação moderada"

        elif valor > 1:
            return "Baixa associação"

        elif valor <= 0.5:
            return "Reduz propensão"

        return "Neutro"

    tabela["Interpretação"] = tabela[
        "Chance Relativa"
    ].apply(interpretar_odds)

    # =====================================================
    # ARREDONDAMENTO
    # =====================================================

    tabela["Chance Relativa"] = tabela[
        "Chance Relativa"
    ].round(2)

    # =====================================================
    # TEXTO EXPLICATIVO
    # =====================================================

    st.markdown(
        """
        A tabela apresenta as variáveis com maior
        associação estatística à ocorrência do defeito.

        - **Chance Relativa (Odds Ratio)**:
            valores maiores que 1 aumentam a chance
            do defeito ocorrer.

        - **Influência**:
            indica se a variável aumenta ou reduz
            a propensão ao defeito.
        """
    )

    # =====================================================
    # CORES
    # =====================================================

    def cor_direcao(valor):

        if "Aumenta" in str(valor):
            return "color: #ff4b4b; font-weight: bold"

        return "color: #00c853; font-weight: bold"

    # =====================================================
    # EXIBE
    # =====================================================

    st.dataframe(

        tabela.style.map(

            cor_direcao,

            subset=["Influência"]
        ),

        use_container_width=True
    )