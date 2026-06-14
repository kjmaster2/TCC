from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# =========================================================
# LIMPAR REGRAS (TORNAR INTERPRETÁVEL)
# =========================================================
def extrair_regras_legiveis(tree, feature_names):

    regras = []
    tree_ = tree.tree_

    def recurse(node, condicoes):

        if tree_.feature[node] != -2:
            nome_feature = feature_names[tree_.feature[node]]

            threshold = tree_.threshold[node]

            # condição falsa (<=)
            recurse(
                tree_.children_left[node],
                condicoes + [f"{nome_feature} = NÃO"]
            )

            # condição verdadeira (>)
            recurse(
                tree_.children_right[node],
                condicoes + [f"{nome_feature} = SIM"]
            )

        else:
            # folha
            valor = tree_.value[node][0]
            classe = valor.argmax()
            total = valor.sum()
            prob = valor[classe] / total if total > 0 else 0

            regras.append({
                "condicoes": condicoes,
                "classe": classe,
                "probabilidade": prob,
                "amostras": int(total)
            })

    recurse(0, [])

    return pd.DataFrame(regras)


# =========================================================
# TRADUZIR DUMMIES PARA NOME ORIGINAL
# =========================================================
def traduzir_condicoes(condicoes):

    traduzidas = []

    for c in condicoes:

        if "cat__" in c:
            c = c.replace("cat__", "")

        if "=" in c:
            var_val, status = c.split("=")
            var_val = var_val.strip()
            status = status.strip()

            if "_" in var_val:
                var, val = var_val.split("_", 1)

                if status == "SIM":
                    traduzidas.append(f"{var} = {val}")
                else:
                    continue

    return traduzidas


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================
def executar_arvore(df, defeito_coluna, variaveis, max_depth=5):

    df = df.copy()

    # =========================
    # TARGET
    # =========================
    y = (df[defeito_coluna] > 0).astype(int)
    X = df[variaveis]

    # =========================
    # PREPROCESSAMENTO
    # =========================
    preprocessador = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), variaveis)
    ])

    # =========================
    # MODELO
    # =========================
    modelo = Pipeline([
        ("prep", preprocessador),
        ("tree", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # treino
    modelo.fit(X, y)

    # =========================
    # FEATURE NAMES
    # =========================
    feature_names = modelo.named_steps["prep"].get_feature_names_out()

    arvore = modelo.named_steps["tree"]

    # =========================
    # REGRAS ESTRUTURADAS
    # =========================
    regras_df = extrair_regras_legiveis(arvore, feature_names)

    # =========================
    # FILTRAR REGRAS DE DEFEITO
    # =========================
    regras_defeito = regras_df[regras_df["classe"] == 1].copy()

    # traduzir
    regras_defeito["regras_texto"] = regras_defeito["condicoes"].apply(
        lambda x: " AND ".join(traduzir_condicoes(x))
    )

    # remover vazias
    regras_defeito = regras_defeito[regras_defeito["regras_texto"] != ""]

    # ordenar por probabilidade
    regras_defeito = regras_defeito.sort_values(
        ["probabilidade", "amostras"],
        ascending=[False, False]
    )

    # =========================
    # SAÍDA FINAL
    # =========================
    return {
        "modelo": modelo,
        "regras": regras_defeito[[
            "regras_texto",
            "probabilidade",
            "amostras"
        ]]
    }