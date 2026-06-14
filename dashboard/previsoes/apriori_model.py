from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd


def executar_apriori(
    df,
    defeito_alvo,
    variaveis,
    min_support,
    min_confidence,
    min_lift,
    max_len=3
):

    df = df.copy()

    # =====================================================
    # TRANSAÇÕES
    # =====================================================

    def gerar_transacao(row):

        itens = []

        # variáveis processo
        for var in variaveis:

            if var not in row.index:
                continue

            valor = row[var]

            if pd.isna(valor):
                continue

            valor = str(valor).strip()

            itens.append(f"{var}={valor}")

        # defeito alvo
        if defeito_alvo in row.index:

            try:

                if float(row[defeito_alvo]) > 0:

                    itens.append(
                        f"DEFEITO={defeito_alvo}"
                    )

            except:
                pass

        return itens

    transacoes = df.apply(
        gerar_transacao,
        axis=1
    )

    # remove vazias
    transacoes = transacoes[
        transacoes.map(len) > 0
    ]

    if transacoes.empty:
        return pd.DataFrame()

    # =====================================================
    # ONE HOT ENCODING
    # =====================================================

    basket = (
        pd.DataFrame(transacoes.tolist())
        .stack()
        .str.get_dummies()
        .groupby(level=0)
        .sum()
        .astype(bool)
    )

    # =====================================================
    # FILTRAR ITENS MUITO RAROS
    # =====================================================

    min_item_freq = min_support / 3

    cols_validas = []

    for col in basket.columns:

        freq = basket[col].mean()

        # mantém defeito sempre
        if col.startswith("DEFEITO="):

            cols_validas.append(col)

        elif freq >= min_item_freq:

            cols_validas.append(col)

    basket = basket[cols_validas]

    if basket.empty:
        return pd.DataFrame()

    # =====================================================
    # APRIORI
    # =====================================================

    itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    if itemsets.empty:
        return pd.DataFrame()

    # =====================================================
    # REGRAS
    # =====================================================

    try:

        regras = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

    except:
        return pd.DataFrame()

    if regras.empty:
        return pd.DataFrame()

    # =====================================================
    # GARANTE COLUNAS
    # =====================================================

    if "lift" not in regras.columns:

        if (
            "confidence" in regras.columns and
            "consequent support" in regras.columns
        ):

            regras["lift"] = (
                regras["confidence"] /
                regras["consequent support"]
            )

        else:

            regras["lift"] = 1.0

    # =====================================================
    # FILTRAR REGRAS
    # =====================================================

    regras = regras[

        # consequente precisa ser defeito
        regras["consequents"].apply(
            lambda x: any(
                str(item).startswith("DEFEITO=")
                for item in x
            )
        )

        &

        # antecedente NÃO pode ser defeito
        ~regras["antecedents"].apply(
            lambda x: any(
                str(item).startswith("DEFEITO=")
                for item in x
            )
        )

        &

        # lift mínimo
        (regras["lift"] >= min_lift)

        &

        # confiança mínima
        (regras["confidence"] >= min_confidence)

        &

        # suporte mínimo
        (regras["support"] >= min_support)

    ].copy()

    if regras.empty:
        return pd.DataFrame()

    # =====================================================
    # LIMITAR COMPLEXIDADE
    # =====================================================

    regras = regras[
        regras["antecedents"].apply(len) <= 3
    ]

    if regras.empty:
        return pd.DataFrame()

    # =====================================================
    # FORMATAR
    # =====================================================

    regras["antecedents"] = regras[
        "antecedents"
    ].apply(
        lambda x: ", ".join(
            sorted(map(str, x))
        )
    )

    regras["consequents"] = regras[
        "consequents"
    ].apply(
        lambda x: ", ".join(
            sorted(map(str, x))
        )
    )

    # =====================================================
    # ORDENAR
    # =====================================================

    regras = regras.sort_values(
        by=[
            "lift",
            "confidence",
            "support"
        ],
        ascending=False
    )

    return regras[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]