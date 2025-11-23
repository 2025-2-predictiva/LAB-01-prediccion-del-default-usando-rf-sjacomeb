# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import pickle
import os
import json
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV

# -----------------------------
# Paso 1: limpieza de datos
# -----------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    # 1. Renombrar y eliminar columnas
    if "default payment next month" in df_copy.columns:
        df_copy.rename(
            columns={"default payment next month": "default"},
            inplace=True,
        )
    if "ID" in df_copy.columns:
        df_copy.drop("ID", axis=1, inplace=True)

    # 2. Eliminar NaNs
    df_copy.dropna(inplace=True)

    # 3. Eliminar registros con información no disponible (valor 0)
    #    en variables categóricas codificadas: 0 = N/A
    cols_with_na_code = ["SEX", "EDUCATION", "MARRIAGE"]
    for col in cols_with_na_code:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            df_copy = df_copy[df_copy[col] != 0]

    # 4. Recategorizar EDUCATION: valores > 4 -> 4 ("others")
    if "EDUCATION" in df_copy.columns:
        df_copy["EDUCATION"] = df_copy["EDUCATION"].where(
            df_copy["EDUCATION"] <= 4,
            4,
        )

    return df_copy


# -----------------------------
# Paso 2: división X / y
# -----------------------------

def split_xy(df: pd.DataFrame):
    y = df["default"]
    X = df.drop("default", axis=1)
    return X, y


# -----------------------------
# Métricas y matrices de confusión
# -----------------------------

def precision_metrics(estimator, X, y, dataset_name: str) -> dict:
    y_pred = estimator.predict(X)

    precision = precision_score(y_true=y, y_pred=y_pred)
    balanced = balanced_accuracy_score(y_true=y, y_pred=y_pred)
    recall = recall_score(y_true=y, y_pred=y_pred)
    f1 = f1_score(y_true=y, y_pred=y_pred)

    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": round(precision, 4),
        "balanced_accuracy": round(balanced, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def cm_metrics(estimator, X, y, dataset_name: str) -> dict:
    """Calcula y formatea la matriz de confusión completa."""
    y_pred = estimator.predict(X)
    # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


# -----------------------------
# Main
# -----------------------------

def main():
    # Cargar datos
    data_train_raw = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    data_test_raw = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    # Paso 1: limpieza
    data_train = clean_data(data_train_raw)
    data_test = clean_data(data_test_raw)
    print("Datos limpiados.")
    # Paso 2: división en X / y
    X_train, y_train = split_xy(data_train)
    X_test, y_test = split_xy(data_test)
    print("Datos divididos en X / y.")
    # Paso 3: pipeline (one-hot en EDUCATION + RandomForest)
    categorical_features = ["EDUCATION"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("RandomForest", RandomForestClassifier(random_state=42)),
        ]
    )
    print("Pipeline creado.")
    # Paso 4: grid search con validación cruzada (balanced_accuracy)
    param_grid = {
        "RandomForest__n_estimators": [200],
        "RandomForest__max_depth": [None],
        "RandomForest__min_samples_leaf": [2],
        "RandomForest__min_samples_split": [5],
        "RandomForest__max_features": ["sqrt"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    print("Grid search completado.")
    # Paso 5: guardar el modelo comprimido
    os.makedirs("/files/models/", exist_ok=True)
    model_path = "/files/models/model.pkl.gz"
    with gzip.open(model_path, "wb") as f:
        pickle.dump(grid_search, f)
    print("Modelo guardado en model.pkl.gz.")
    # Paso 6: métricas de precisión, balanced_accuracy, recall, f1
    metrics_train = precision_metrics(grid_search, X_train, y_train, "train")
    metrics_test = precision_metrics(grid_search, X_test, y_test, "test")
    print("Métricas calculadas.")
    # Paso 7: matrices de confusión
    cm_train = cm_metrics(grid_search, X_train, y_train, "train")
    cm_test = cm_metrics(grid_search, X_test, y_test, "test")

    # Guardar todo en metrics.json, una línea por diccionario
    output_dir = "/files/output/"
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.json")

    all_metrics = [metrics_train, metrics_test, cm_train, cm_test]

    with open(metrics_file, "w", encoding="utf-8") as f:
        for metric_dict in all_metrics:
            f.write(json.dumps(metric_dict) + "\n")
    print("Métricas guardadas en metrics.json.")

if __name__ == "__main__":
    main()