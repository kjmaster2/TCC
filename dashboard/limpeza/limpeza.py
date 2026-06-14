import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata

df = pd.read_csv("../dados/dados_tcc.csv", sep=";")

#Subsitui o titulo das colunas por com espaço por _.
df.columns = (
    df.columns
        .str.strip()
        .str.replace(" ", "_")
)

#Remover til ç e demais caracteres especiais
df.columns = [
    unicodedata.normalize("NFKD", col)
    .encode("ASCII", "ignore")
    .decode("ASCII")
    .strip()
    .replace(" ", "_")
    for col in df.columns
]

#print(df.columns)

#Verifica quais colunas possuem valores nulos e a quantidade de valores nulos
colunas_nan_qtd = df.isnull().sum()
colunas_nan_qtd = colunas_nan_qtd[colunas_nan_qtd > 0]

#print(colunas_nan_qtd)

df[df["TRAT.SUPERFICIAL"].isnull()]
df.loc[df["TRAT.SUPERFICIAL"].isnull(), "TRAT.SUPERFICIAL"] = "ZA CR3"

df[df["MATERIAL"].isnull()]
df = df.dropna(subset=["MATERIAL", "USINA"])

#Criar uma coluna bitola
bitola_x = df["ITEM_HASSMANN"].str.extract(
    r'((?:M\d+(?:,\d+)?)|(?:\d+/\d+)|(?:\d+(?:,\d+)?))\s*x'
)

# bitola isolada (caso das porcas)
bitola_sem_x = df["ITEM_HASSMANN"].str.extract(
    r'\b(M\d+(?:,\d+)?|\d+/\d+)\b'
)

#combinar resultados
df["BITOLA"] = bitola_x[0].fillna(bitola_sem_x[0])

# =========================================================
# EXTRAÇÃO
# =========================================================
def extrair_comprimento(item):

    item = str(item).strip().upper()

    # PORCA = comprimento zero
    if item.startswith("PORCA"):
        return "0"

    # remover parte após ROSCA (evita pegar passo)
    item = item.split("ROSCA")[0]

    # pegar valores após X
    valores = re.findall(r'[Xx]\s*([\d.,/]+)', item)

    if not valores:
        return np.nan

    return valores[-1]


df["COMPRIMENTO"] = df["ITEM_HASSMANN"].apply(extrair_comprimento)

# =========================================================
# TRABALHAR COMO STRING
# =========================================================
comp_str = df["COMPRIMENTO"].astype(str)

# =========================================================
# FRAÇÃO SIMPLES (1/2)
# =========================================================
mask_frac = comp_str.str.contains(r"^\d+/\d+$", na=False)

frac = comp_str[mask_frac].str.extract(r"(\d+)/(\d+)")

comp_str.loc[mask_frac] = (
    frac[0].astype(float) / frac[1].astype(float) * 25.4
).round(2).astype(str)

# =========================================================
# FRAÇÃO MISTA (1,1/2 ou 1.1/2)
# =========================================================
mask_mista = comp_str.str.contains(r"^\d+[.,]\d+/\d+$", na=False)

partes = comp_str[mask_mista].str.extract(r"(\d+)[.,](\d+)/(\d+)")

comp_str.loc[mask_mista] = (
    (
        partes[0].astype(float) +
        (partes[1].astype(float) / partes[2].astype(float))
    ) * 25.4
).round(2).astype(str)

# =========================================================
# TROCAR VÍRGULA POR PONTO
# =========================================================
comp_str = comp_str.str.replace(",", ".", regex=False)

# =========================================================
# CONVERTER FINAL PARA NUMÉRICO
# =========================================================
df["COMPRIMENTO"] = pd.to_numeric(comp_str, errors="coerce")

# =========================================================
# AJUSTE FINAL (valores pequenos = polegada → mm)
# =========================================================
mask = df["COMPRIMENTO"] < 4

df.loc[mask, "COMPRIMENTO"] = (
    df.loc[mask, "COMPRIMENTO"] * 25.4
).round(2)

colunas_nan_qtd = df.isnull().sum()
colunas_nan_qtd = colunas_nan_qtd[colunas_nan_qtd > 0]

#print(colunas_nan_qtd)

#Rotular dados da coluna familia. transformar em números
# garantir string e remover espaços
df["FAMILIA"] = df["FAMILIA"].astype(str).str.strip()

# obter valores únicos ordenados
familias_unicas = sorted(df["FAMILIA"].unique())

# criar mapeamento iniciando em 1
mapa_familia = {nome: i+1 for i, nome in enumerate(familias_unicas)}

# criar nova coluna (_ROT)
df["FAMILIA_ROT"] = df["FAMILIA"].map(mapa_familia)

# imprimir relação
# print("MAPEAMENTO FAMILIA -> ID\n")
# for nome, numero in mapa_familia.items():
#     print(f"{numero} -> {nome}")

#Rotular dados da coluna resistencia. transformar em números
# garantir string padronizada
df["RESISTENCIA"] = df["RESISTENCIA"].replace("C1", "C0")

df["RESISTENCIA"] = df["RESISTENCIA"].astype(str).str.strip()

# obter valores únicos ordenados
resistencias_unicas = sorted(df["RESISTENCIA"].unique())

# criar mapeamento iniciando em 1
mapa_resistencia = {nome: i+1 for i, nome in enumerate(resistencias_unicas)}

# aplicar no dataframe
df["RESISTENCIA_ROT"] = df["RESISTENCIA"].map(mapa_resistencia)

# imprimir relação
#print("MAPEAMENTO RESISTENCIA -> ID\n")
#for nome, numero in mapa_resistencia.items():
#    print(f"{numero} -> {nome}")

#Rotular dados da coluna tratamento superficial. transformar em números
# padronizar texto
df["TRAT.SUPERFICIAL"] = (
    df["TRAT.SUPERFICIAL"]
    .astype(str)
    .str.strip()
)

# obter valores únicos ordenados
trat_unicos = sorted(df["TRAT.SUPERFICIAL"].unique())

# criar mapeamento iniciando em 1
mapa_trat = {nome: i+1 for i, nome in enumerate(trat_unicos)}

# aplicar no dataframe
df["TRAT_SUPERFICIAL_ROT"] = df["TRAT.SUPERFICIAL"].map(mapa_trat)

# imprimir relação
# print("MAPEAMENTO TRAT.SUPERFICIAL -> ID\n")
# for nome, numero in mapa_trat.items():
#     print(f"{numero} -> {nome}")

#Rotular dados da coluna usina. transformar em números
# padronizar texto
df["USINA"] = (
    df["USINA"]
    .astype(str)
    .str.strip()
)

# obter valores únicos ordenados
usinas_unicas = sorted(df["USINA"].unique())

# criar mapeamento iniciando em 1
mapa_usina = {nome: i+1 for i, nome in enumerate(usinas_unicas)}

# aplicar no dataframe
df["USINA_ROT"] = df["USINA"].map(mapa_usina)

# imprimir relação
# print("MAPEAMENTO USINA -> ID\n")
# for nome, numero in mapa_usina.items():
#     print(f"{numero} -> {nome}")

#Extrair a materia-prima utilizada
# garantir string limpa
# garantir string limpa
df["MATERIAL"] = df["MATERIAL"].astype(str).str.strip().str.upper()

# últimos 5 caracteres
ultimos5 = df["MATERIAL"].str[-5:]

# remover PA, PC, PL para testar se sobra alguma letra diferente
teste = ultimos5.str.replace("PA", "", regex=False) \
                  .str.replace("PC", "", regex=False) \
                  .str.replace("PL", "", regex=False)

# verificar se ainda existe alguma letra A-Z
tem_letra_valida = teste.str.contains(r"[A-Z]", na=False)

# =========================================================
# CRIAR COLUNA REAL (STRING) → USO NO APRIORI
# =========================================================

df["MATERIA_PRIMA"] = np.where(
     tem_letra_valida,
     df["MATERIAL"].str[-5:],  # pega 5 se tiver letra diferente de PA/PC/PL
     df["MATERIAL"].str[-4:]   # senão pega 4
)

# garantir padronização
df["MATERIA_PRIMA"] = df["MATERIA_PRIMA"].astype(str).str.strip()

# =========================================================
# CRIAR COLUNA ROTULADA (_ROT) → USO NO MODELO
# =========================================================

# obter valores únicos ordenados
mp_unicas = sorted(df["MATERIA_PRIMA"].unique())

# criar mapeamento iniciando em 1
mapa_mp = {nome: i+1 for i, nome in enumerate(mp_unicas)}

# aplicar no dataframe (sem perder original)
df["MATERIA_PRIMA_ROT"] = df["MATERIA_PRIMA"].map(mapa_mp)

# =========================================================
# (OPCIONAL) PRINT PARA DEBUG
# =========================================================

# print("MAPEAMENTO MATERIA_PRIMA -> ID\n")
# for nome, numero in mapa_mp.items():
#     print(f"{numero} -> {nome}")

#Rotular dados da maquina. transformar em números
# garantir string padronizada
df["MAQUINA"] = df["MAQUINA"].astype(str).str.strip()

# obter valores únicos ordenados
mp_unicas = sorted(df["MAQUINA"].unique())

# criar mapeamento iniciando em 1
mp = {nome: i+1 for i, nome in enumerate(mp_unicas)}

# aplicar no dataframe
df["MAQUINA_ROT"] = df["MAQUINA"].map(mp)

# imprimir relação
# print("MAPEAMENTO MAQUINAS -> ID\n")
# for nome, numero in mp.items():
#     print(f"{numero} -> {nome}")

#Rotular dados da materia prima. transformar em números
# garantir string padronizada
df["MATERIA_PRIMA"] = df["MATERIA_PRIMA"].astype(str).str.strip()

# obter valores únicos ordenados
mp_unicas = sorted(df["MATERIA_PRIMA"].unique())

# criar mapeamento iniciando em 1
mp = {nome: i+1 for i, nome in enumerate(mp_unicas)}

# aplicar no dataframe
df["MATERIA_PRIMA_ROT"] = df["MATERIA_PRIMA"].map(mp)

# imprimir relação
# print("MAPEAMENTO MP -> ID\n")
# for nome, numero in mp.items():
#     print(f"{numero} -> {nome}")

bit_str = df["BITOLA"].astype(str).str.strip()

# =========================================================
# REMOVER "M"
# =========================================================
bit_str = bit_str.str.replace(r"[Mm]", "", regex=True)

# =========================================================
# FRAÇÃO (ex: 1/2, 5/16)
# =========================================================
mask_frac = bit_str.str.contains(r"^\d+/\d+$", na=False)

frac = bit_str[mask_frac].str.extract(r"(\d+)/(\d+)")

bit_str.loc[mask_frac] = (
    frac[0].astype(float) /
    frac[1].astype(float) * 25.4
).round(2).astype(str)

# =========================================================
# TROCAR VÍRGULA POR PONTO
# =========================================================
bit_str = bit_str.str.replace(",", ".", regex=False)

# =========================================================
# CONVERTER FINAL PARA NUMÉRICO
# =========================================================
df["BITOLA"] = pd.to_numeric(bit_str, errors="coerce")

colunas_nan_qtd = df.isnull().sum()
colunas_nan_qtd = colunas_nan_qtd[colunas_nan_qtd > 0]

# print(colunas_nan_qtd)

for col in ["BITOLA", "COMPRIMENTO"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

#Remover colunas que nao importam
colunas_remover = [
    "ITEM_HASSMANN",
    "HORA",
    "TEMPO_SETUP",
    "TEMPO_MAQ._PARADA",
    "MOTIVO_PARADA",
    "OBSERVACAO_PARADA",
    "PRENSA",
    "MATERIAL",
    "LAMINADORA",
    "CHANFRADEIRA",
    "LINHA_TT",
    "GALVANICA",
    "NSP",
    "NRPY",
    "STATUS",
    "OBSERVACAO",
    "DESTINO_AMOSTRA",
    "ORDEM_PRODUCAO",
    "TEMPO_SELECAO",
]

df = df.drop(columns=colunas_remover, errors="ignore")

df = df[df["QTD_SELECIONADA"] >= 500]
df["QTD_SELECIONADA"].min()

# Linha possui defeito?
df["LOTE_COM_PROBLEMA"] = (df["TOTAL_DEFEITOS"] > 0).astype(int)

#Criar uma coluna descrevendo o tipo de defeito
defeitos = df.loc[:, "PECA_EMPENADA/TORTA":"ARRUELA_DEFORMADA/BATIDA"].columns

def obter_tipo_defeito(row):
    for defeito in defeitos:
        if row[defeito] > 0:
            return defeito
    return "SEM_DEFEITO"

df["TIPO_DEFEITO"] = df.apply(obter_tipo_defeito, axis=1)

df.to_csv("../dados/dados_tratados.csv", index=False)