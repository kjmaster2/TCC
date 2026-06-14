# =========================================================
# REGRESSÃO LOGÍSTICA ANALÍTICA
# =========================================================

import numpy as np
import pandas as pd

import statsmodels.api as sm


def executar_regressao_logistica_analitica(

    df,

    defeito,

    variaveis,

    regras_apriori=None
):

    # =====================================================
    # CÓPIA DO DATAFRAME
    # =====================================================

    df = df.copy()

    # =====================================================
    # CRIA ALVO DO DEFEITO SELECIONADO
    # =====================================================

    df["ALVO"] = (

        pd.to_numeric(

            df[defeito],

            errors="coerce"
        )

        .fillna(0)

        .gt(0)

        .astype(int)
    )

    # =====================================================
    # FEATURES
    # =====================================================

    X = df[variaveis].copy()

    y = df["ALVO"]

    # =====================================================
    # REMOVE NULOS
    # =====================================================

    dados = pd.concat(
        [X, y],
        axis=1
    ).dropna()

    X = dados[variaveis]

    y = dados["ALVO"]

    # =====================================================
    # REMOVE CLASSES INVÁLIDAS
    # =====================================================

    if y.nunique() < 2:

        raise ValueError(
            "O defeito selecionado não possui registros suficientes."
        )

    # =====================================================
    # FILTRA CATEGORIAS RARAS
    # =====================================================

    for col in X.columns:

        freq = X[col].value_counts(
            normalize=True
        )

        categorias_validas = freq[
            freq >= 0.01
        ].index

        X[col] = X[col].where(

            X[col].isin(categorias_validas),

            "OUTROS"
        )

    # =====================================================
    # ONE HOT ENCODING
    # =====================================================

    X = pd.get_dummies(

        X,

        drop_first=True
    )

    # =====================================================
    # BOOLEAN → INT
    # =====================================================

    for col in X.columns:

        if X[col].dtype == bool:

            X[col] = X[col].astype(int)

    # =====================================================
    # REMOVE COLUNAS MUITO RARAS
    # =====================================================

    freq_colunas = X.mean()

    X = X.loc[
        :,
        freq_colunas > 0.005
    ]

    # =====================================================
    # REMOVE COLUNAS CONSTANTES
    # =====================================================

    X = X.loc[
        :,
        X.nunique() > 1
    ]

    # =====================================================
    # ADICIONA INTERCEPTO
    # =====================================================

    X = sm.add_constant(

        X,

        has_constant="add"
    )

    # =====================================================
    # MODELO
    # =====================================================

    modelo = sm.Logit(
        y,
        X
    )

    resultado = modelo.fit_regularized(

        alpha=1.0,

        disp=False
    )

    pseudo_r2 = resultado.prsquared
    aic = resultado.aic

    # =====================================================
    # COEFICIENTES
    # =====================================================

    coef = resultado.params

    odds_ratio = np.exp(coef)

    # =====================================================
    # TABELA
    # =====================================================

    tabela = pd.DataFrame({

        "Variavel": coef.index,

        "Coeficiente": coef.values,

        "Odds_Ratio": odds_ratio.values
    })

    # =====================================================
    # REMOVE INTERCEPTO
    # =====================================================

    tabela = tabela[
        tabela["Variavel"] != "const"
    ]

    # =====================================================
    # DIREÇÃO
    # =====================================================

    tabela["Direcao"] = np.where(

        tabela["Coeficiente"] > 0,

        "Aumenta propensão",

        "Reduz propensão"
    )

    # =====================================================
    # IMPACTO
    # =====================================================

    tabela["Impacto"] = (
        tabela["Coeficiente"]
        .abs()
    )

    # =====================================================
    # FILTRO APRIORI
    # =====================================================

    if (
        regras_apriori is not None
        and
        not regras_apriori.empty
    ):

        variaveis_apriori = []

        # =================================================
        # EXTRAI VARIÁVEIS
        # =================================================

        for _, row in regras_apriori.iterrows():

            antecedente = row["antecedents"]

            if not isinstance(

                antecedente,

                (list, set, frozenset, tuple)
            ):

                antecedente = [str(antecedente)]

            for item in antecedente:

                item = str(item)

                item = (

                    item

                    .replace("frozenset({", "")

                    .replace("})", "")

                    .replace("'", "")

                    .strip()
                )

                if "=" in item:

                    nome_var = (

                        item
                        .split("=")[0]
                        .strip()
                        .upper()
                    )

                    variaveis_apriori.append(
                        nome_var
                    )

        # =================================================
        # REMOVE DUPLICADOS
        # =================================================

        variaveis_apriori = list(
            set(variaveis_apriori)
        )

        # =================================================
        # FILTRA RESULTADOS
        # =================================================

        tabela = tabela[
            tabela["Variavel"].apply(

                lambda x: any(

                    v in str(x).upper()

                    for v in variaveis_apriori
                )
            )
        ]

    # =====================================================
    # ORDENA
    # =====================================================

    tabela = tabela.sort_values(

        by="Impacto",

        ascending=False
    ).reset_index(drop=True)

    # =====================================================
    # RETORNO
    # =====================================================

    return {

    "modelo": resultado,

    "tabela": tabela,

    "metricas": {

        "Pseudo_R2": pseudo_r2,

        "AIC": aic,

        "LogLikelihood": resultado.llf
    }
}
