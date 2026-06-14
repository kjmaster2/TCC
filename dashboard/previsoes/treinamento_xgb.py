import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df = pd.read_csv('../dados/dados_tratados.csv')

features = [
    'FAMILIA',
    'RESISTENCIA',
    'TRAT.SUPERFICIAL',
    'USINA',
    'MAQUINA',
    'BITOLA',
    'COMPRIMENTO',
    'MATERIA_PRIMA'
]

target = 'TIPO_DEFEITO'

X = df[features].copy()
y = df[target].copy()

# Remover classes muito raras
contagem = y.value_counts()

classes_validas = contagem[contagem >= 5].index

df = df[df[target].isin(classes_validas)]

# Recriar X e y após o filtro
X = df[features].copy()
y = df[target].copy()

print("Classes utilizadas:")
print(y.value_counts())

encoders = {}

for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(
    random_state=42,
    k_neighbors=3
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

modelo = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

modelo.fit(
    X_train_smote,
    y_train_smote
)

pred = modelo.predict(X_test)

print(
    classification_report(
        y_test,
        pred,
        target_names=target_encoder.classes_
    )
)

pickle.dump(modelo, open('../previsoes/modelo_xgb.pkl', 'wb'))
pickle.dump(encoders, open('../previsoes/encoders.pkl', 'wb'))
pickle.dump(target_encoder, open('../previsoes/target_encoder.pkl', 'wb'))
pickle.dump(features, open('../previsoes/features.pkl', 'wb'))

print('Modelo treinado com sucesso.')