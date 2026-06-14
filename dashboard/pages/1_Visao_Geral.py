import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from suporte.maps import (familias_map, bitola_map, resistencia_map, usina_map, materia_map, trat_map, maq_map
)

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Dashboard Qualidade Fixadores",
    layout="wide",
    page_icon="🔩"
)

st.title("🔩 Dashboard de Qualidade - Fixadores")
st.divider()

# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():

    arquivo = BASE_DIR / "dados" / "dados_tratados.csv"

    df = pd.read_csv(arquivo)

    df.columns = df.columns.str.strip()

    df["DATA"] = pd.to_datetime({
        "year": df["ANO"],
        "month": df["MES"],
        "day": df["DIA"]
    })

    return df

df = load_data()

# =========================================================
# COLUNAS DE DEFEITOS
# =========================================================
defeitos = df.loc[:, "PECA_EMPENADA/TORTA":"ARRUELA_DEFORMADA/BATIDA"].columns

# =========================================================
# FILTROS LATERAIS
# =========================================================
st.sidebar.header("FILTROS")

# =========================================================
# FUNÇÃO GENÉRICA PARA FILTROS
# =========================================================
# =========================================================
# BASE DE DADOS (IMUTÁVEL)
# =========================================================
df_base = df.copy()

# =========================================================
# CONTROLE DE ATUALIZAÇÃO (CRÍTICO)
# =========================================================
if "filtro_version" not in st.session_state:
    st.session_state.filtro_version = 0

# =========================================================
# LISTA DE COLUNAS DE FILTRO
# =========================================================
COLUNAS_FILTRO = [
    "FAMILIA_ROT","BITOLA","COMPRIMENTO","RESISTENCIA_ROT",
    "USINA_ROT","MATERIA_PRIMA_ROT","TRAT_SUPERFICIAL_ROT","MAQUINA_ROT"
]

# =========================================================
# INICIALIZAR SESSION STATE
# =========================================================
for col in COLUNAS_FILTRO:
    if col not in st.session_state:
        st.session_state[col] = list(df_base[col].dropna().unique())

# =========================================================
# FUNÇÃO DE FILTRO DINÂMICO (FINAL CORRETA)
# =========================================================
def criar_filtro(nome, coluna, mapa=None):

    with st.sidebar.expander(nome, expanded=False):

        col_btn1, col_btn2 = st.columns(2)

        valores = sorted(df_base[coluna].dropna().unique())

        # -------------------------------
        # BOTÕES
        # -------------------------------
        if col_btn1.button("✓", key=f"{coluna}_all"):
            st.session_state[coluna] = valores
            st.session_state.filtro_version += 1
            st.rerun()

        if col_btn2.button("✕", key=f"{coluna}_clear"):
            st.session_state[coluna] = []
            st.session_state.filtro_version += 1
            st.rerun()

        # -------------------------------
        # CONTEXTO (OUTROS FILTROS)
        # -------------------------------
        df_temp = df_base.copy()

        for col in COLUNAS_FILTRO:
            if col != coluna and st.session_state[col]:
                df_temp = df_temp[df_temp[col].isin(st.session_state[col])]

        contagem = df_temp[coluna].value_counts()

        # -------------------------------
        # CHECKBOXES DINÂMICOS
        # -------------------------------
        novos_valores = []

        for v in valores:

            label_nome = mapa.get(v, v) if mapa else v
            count = contagem.get(v, 0)

            label = f"{label_nome} ({count})"

            checked = v in st.session_state[coluna]

            estado = st.checkbox(
                label,
                value=checked,
                key=f"{coluna}_{v}_{st.session_state.filtro_version}"
            )

            if estado:
                novos_valores.append(v)

        # -------------------------------
        # DETECTAR MUDANÇA
        # -------------------------------
        if set(novos_valores) != set(st.session_state[coluna]):
            st.session_state[coluna] = novos_valores
            st.session_state.filtro_version += 1
            st.rerun()

        return novos_valores

# =========================================================
# CRIAÇÃO DOS FILTROS
# =========================================================
bitola = criar_filtro("Bitola","BITOLA",bitola_map)
comprimento = criar_filtro("Comprimento","COMPRIMENTO")
familia = criar_filtro("Família","FAMILIA_ROT",familias_map)
resistencia = criar_filtro("Resistência","RESISTENCIA_ROT",resistencia_map)
usina = criar_filtro("Usina","USINA_ROT",usina_map)
materia = criar_filtro("Matéria Prima","MATERIA_PRIMA_ROT",materia_map)
tratamento = criar_filtro("Tratamento","TRAT_SUPERFICIAL_ROT",trat_map)
maquina = criar_filtro("Máquina","MAQUINA_ROT",maq_map)

# =========================================================
# APLICAR FILTROS
# =========================================================
df_filtrado = df_base.copy()

for col in COLUNAS_FILTRO:
    if st.session_state[col]:
        df_filtrado = df_filtrado[df_filtrado[col].isin(st.session_state[col])]

df = df_filtrado

# =========================================================
#INDICADORES
# =========================================================

st.subheader("📊 Contexto do Dataset")

total_registros = df.shape[0]
periodo_inicio = df["DATA"].min()
periodo_fim = df["DATA"].max()

c1,c2 = st.columns(2)

c2.metric("Período", f"{periodo_inicio.date()} → {periodo_fim.date()}")

st.divider()

st.subheader("🔎 Diversidade dos Itens")

c1,c2,c3,c4,c5,c6, c7, c8 = st.columns(8)

c1.metric("Famílias", df["FAMILIA_ROT"].nunique())
c2.metric("Bitolas", df["BITOLA"].nunique())
c3.metric("Comprimentos", df["COMPRIMENTO"].nunique())
c4.metric("Resistências", df["RESISTENCIA_ROT"].nunique())
c5.metric("Matérias Primas", df["MATERIA_PRIMA_ROT"].nunique())
c6.metric("Usinas", df["USINA_ROT"].nunique())
c7.metric("Tratamentos", df["TRAT_SUPERFICIAL_ROT"].nunique())
c8.metric("Máquinas", df["MAQUINA_ROT"].nunique())

st.divider()

st.subheader("⚠ Indicadores de Qualidade")

total_pecas = df["QTD_SELECIONADA"].sum()
total_defeitos = df["TOTAL_DEFEITOS"].sum()
lotes_com_problema = df["LOTE_COM_PROBLEMA"].sum()
total_registros = df.shape[0]

ppm = (total_defeitos / total_pecas) * 1000000 if total_pecas > 0 else 0
probabilidade_defeito = (lotes_com_problema / total_registros) * 100 if lotes_com_problema > 0 else 0

c1,c2,c3 = st.columns(3)

c1.metric("Total Peças", f"{total_pecas:,.0f}")
c2.metric("Peças com Defeito", f"{total_defeitos:,.0f}")
c3.metric("PPM", f"{ppm:,.0f}")

c4,c5,c6 = st.columns(3)

c4.metric("Total de Lotes", f"{total_registros:,}")
c5.metric("Lotes com Problema", f"{lotes_com_problema:,.0f}")
c6.metric("Probabilidade de Defeito", f"{probabilidade_defeito:.1f}%")

st.divider()

st.subheader("📉 Análise de Problemas")

col1,col2 = st.columns(2)

ranking_defeitos = df[defeitos].sum().sort_values(ascending=False).head(10)

fig1 = px.bar(
    ranking_defeitos,
    orientation="h",
    title="Top 10 Problemas - Quantidade Total"
)

col1.plotly_chart(fig1,use_container_width=True)

contagem_problemas = (df[defeitos]>0).sum().sort_values(ascending=False).head(10)

fig2 = px.bar(
    contagem_problemas,
    orientation="h",
    title="Top 10 Problemas - Ocorrências"
)

col2.plotly_chart(fig2,use_container_width=True)

st.subheader("📊 Pareto de Defeitos")

ranking = (df[defeitos]>0).sum().sort_values(ascending=False)

pareto = ranking.head(10).reset_index()
pareto.columns=["Defeito","Ocorrencias"]

pareto["Acumulado"] = pareto["Ocorrencias"].cumsum()
pareto["% Acumulado"] = pareto["Acumulado"]/pareto["Ocorrencias"].sum()*100

fig_pareto = px.bar(pareto,x="Defeito",y="Ocorrencias")

fig_pareto.add_scatter(
    x=pareto["Defeito"],
    y=pareto["% Acumulado"],
    mode="lines+markers",
    name="% Acumulado",
    yaxis="y2"
)

fig_pareto.update_layout(
    yaxis=dict(title="Ocorrências"),
    yaxis2=dict(
        title="% Acumulado",
        overlaying="y",
        side="right",
        range=[0,100]
    )
)

st.plotly_chart(fig_pareto,use_container_width=True)

st.subheader("📈 Evolução Mensal da Produção e Qualidade")

serie = (
    df.set_index("DATA")
    .resample("ME")
    .agg({
        "TOTAL_DEFEITOS": "sum",
        "QTD_SELECIONADA": "sum",
        "LOTE_COM_PROBLEMA": "sum"
    })
    .reset_index()
)   

# PPM
serie["PPM"] = (serie["TOTAL_DEFEITOS"] / serie["QTD_SELECIONADA"]) * 1_000_000

fig = go.Figure()

# Lotes com problema
fig.add_trace(
    go.Scatter(
        x=serie["DATA"],
        y=serie["LOTE_COM_PROBLEMA"],
        name="Lotes com Problema",
        mode="lines+markers",
        yaxis="y1"
    )
)

# Produção
fig.add_trace(
    go.Scatter(
        x=serie["DATA"],
        y=serie["QTD_SELECIONADA"],
        name="Peças Produzidas",
        mode="lines+markers",
        yaxis="y2"
    )
)

# PPM
fig.add_trace(
    go.Scatter(
        x=serie["DATA"],
        y=serie["PPM"],
        name="PPM",
        mode="lines+markers",
        yaxis="y3"
    )
)

fig.update_layout(
    xaxis=dict(title="Mês"),
    yaxis=dict(title="Lotes com Problema", side="left"),
    yaxis2=dict(
        title="Produção",
        overlaying="y",
        side="right"
    ),
    yaxis3=dict(
        title="PPM",
        overlaying="y",
        side="right",
        anchor="free",
        position=0.98
    ),
    legend=dict(x=0, y=1.1)
)

st.plotly_chart(fig, use_container_width=True)