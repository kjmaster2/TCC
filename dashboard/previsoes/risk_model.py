import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE


def _criar_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Cria o pré-processador para variáveis categóricas e numéricas.
    """
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformers = []

    if cat_cols:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if num_cols:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("num", num_pipe, num_cols))

    if not transformers:
        raise ValueError("Nenhuma variável válida foi encontrada para modelagem.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def executar_regressao_logistica(
    df,
    defeito,
    variaveis,
    test_size=0.2,
    random_state=42,
    smote_strategy=0.30,
    threshold=0.30,
    max_iter=4000
):
    df = df.copy()

    if defeito not in df.columns:
        raise ValueError(f"Defeito '{defeito}' não encontrado no DataFrame.")

    if not variaveis:
        raise ValueError("Nenhuma variável foi selecionada.")

    faltantes = [c for c in variaveis if c not in df.columns]
    if faltantes:
        raise ValueError(f"Variáveis inexistentes no DataFrame: {faltantes}")

    # =====================================================
    # ALVO BINÁRIO
    # =====================================================

    df["ALVO"] = (
        pd.to_numeric(df[defeito], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    X = df[variaveis].copy()
    y = df["ALVO"].copy()

    # =====================================================
    # TRAIN / TEST
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # =====================================================
    # PREPROCESSAMENTO
    # =====================================================

    preprocessor = _criar_preprocessor(X_train)

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    # garante matriz densa
    if hasattr(X_train_pre, "toarray"):
        X_train_pre = X_train_pre.toarray()

    if hasattr(X_test_pre, "toarray"):
        X_test_pre = X_test_pre.toarray()

    # =====================================================
    # SMOTE
    # =====================================================

    y_train_series = pd.Series(y_train).reset_index(drop=True)

    contagem_classes = y_train_series.value_counts()
    if contagem_classes.min() < 2:
        raise ValueError(
            "A classe minoritária tem menos de 2 exemplos no treino; "
            "SMOTE não pode ser aplicado com segurança neste defeito."
        )

    k_neighbors = min(5, int(contagem_classes.min()) - 1)
    k_neighbors = max(k_neighbors, 1)

    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=smote_strategy,
        k_neighbors=k_neighbors
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_pre,
        y_train_series
    )

    y_train_smote = pd.Series(y_train_smote).reset_index(drop=True)

    # =====================================================
    # MODELO
    # =====================================================

    modelo = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
        class_weight="balanced"
    )

    modelo.fit(X_train_smote, y_train_smote)

    # =====================================================
    # PREDIÇÃO
    # =====================================================

    y_prob = modelo.predict_proba(X_test_pre)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # =====================================================
    # MÉTRICAS
    # =====================================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    matriz = confusion_matrix(y_test, y_pred)

    relatorio = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # =====================================================
    # IMPORTÂNCIA DAS VARIÁVEIS
    # =====================================================

    feature_names = preprocessor.get_feature_names_out()

    importancia = pd.DataFrame({
        "Variavel": feature_names,
        "Coeficiente": modelo.coef_[0]
    })

    importancia["Impacto"] = importancia["Coeficiente"].abs()
    importancia["Odds_Ratio"] = np.exp(importancia["Coeficiente"])
    importancia["Direcao"] = np.where(
        importancia["Coeficiente"] > 0,
        "Aumenta probabilidade",
        "Reduz probabilidade"
    )

    importancia = importancia.sort_values(
        by="Impacto",
        ascending=False
    ).reset_index(drop=True)

    # =====================================================
    # DISTRIBUIÇÕES
    # =====================================================

    dist_original = y_train_series.value_counts().sort_index()
    dist_smote = y_train_smote.value_counts().sort_index()

    # =====================================================
    # ROC CURVE
    # =====================================================

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig_roc, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC ROC = {auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Taxa de falso positivo")
    ax.set_ylabel("Taxa de verdadeiro positivo")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # =====================================================
    # PRECISION-RECALL CURVE
    # =====================================================

    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)

    fig_pr, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_recall, pr_precision, label=f"AUC PR = {auc_pr:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # =====================================================
    # MATRIZ DE CONFUSÃO
    # =====================================================

    fig_cm, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matriz)

    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sem Defeito", "Com Defeito"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sem Defeito", "Com Defeito"])

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, int(matriz[i, j]), ha="center", va="center")

    fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # =====================================================
    # IMPORTÂNCIA (TOP 15)
    # =====================================================

    top_importancia = importancia.head(15).sort_values(
        by="Impacto",
        ascending=True
    )

    fig_coef, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_importancia["Variavel"], top_importancia["Impacto"])
    ax.set_title("Top 15 variáveis por impacto absoluto")
    ax.set_xlabel("Impacto absoluto do coeficiente")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.3)

    # =====================================================
    # BALANCEAMENTO SMOTE
    # =====================================================

    classes = ["Sem Defeito", "Com Defeito"]
    original_vals = [
        int(dist_original.get(0, 0)),
        int(dist_original.get(1, 0))
    ]
    smote_vals = [
        int(dist_smote.get(0, 0)),
        int(dist_smote.get(1, 0))
    ]

    x = np.arange(len(classes))
    width = 0.35

    fig_balance, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, original_vals, width, label="Original")
    ax.bar(x + width/2, smote_vals, width, label="Após SMOTE")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Quantidade")
    ax.set_title("Distribuição das classes antes e após SMOTE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # =====================================================
    # RETORNO
    # =====================================================

    return {
        "modelo": modelo,
        "preprocessor": preprocessor,
        "feature_names": feature_names,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,

        "relatorio": relatorio,
        "matriz": matriz,

        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,

        "importancia": importancia,

        "dist_original": dist_original,
        "dist_smote": dist_smote,

        "threshold": threshold,
        "smote_strategy": smote_strategy,
        "k_neighbors": k_neighbors,

        "fig_roc": fig_roc,
        "fig_pr": fig_pr,
        "fig_cm": fig_cm,
        "fig_coef": fig_coef,
        "fig_balance": fig_balance
    }