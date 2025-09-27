import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import logistic_regression.softmax_regression as softmax_regression
import logistic_regression.logistic_regression as logistic_regression


# Cargar los datasets
train_df = pd.read_csv("datos_entrenamiento_riesgo.csv")
test_df = pd.read_csv("datos_prueba_riesgo.csv")


def impute_variables(train_df, test_df):
    categorical_encoded = [
        "sector_laboral",
        "tipo_vivienda",
        "nivel_educativo",
        "estado_civil",
    ]
    for col in categorical_encoded:
        mode_train = train_df[col].mode()[0]
        train_df[col].fillna(mode_train, inplace=True)
        test_df[col].fillna(mode_train, inplace=True)

    numerical = [
        "porcentaje_utilizacion_credito",
        "proporcion_pagos_a_tiempo",
        "residencia_antiguedad_meses",
        "lineas_credito_abiertas",
    ]
    for col in numerical:
        median_train = train_df[col].median()
        train_df[col].fillna(median_train, inplace=True)
        test_df[col].fillna(median_train, inplace=True)
    return train_df, test_df


def delete_nan_instances(train_df, test_df):
    return train_df.dropna(), test_df.dropna()


def normalize(x_train, x_test):
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)
    return x_train_norm, x_test_norm


def add_bias(x):
    return np.column_stack([np.ones(x.shape[0]), x])


def encode_labels(y):
    label_map = {"Bajo": 0, "Medio": 1, "Alto": 2}
    return y.map(label_map).values


def analyze_results(y_true, y_pred, title="Matriz de Confusión"):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Bajo", "Medio", "Alto"]

    cm_decimal = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm_decimal,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )

    plt.title(title)
    plt.tight_layout()
    filename = f'{title.replace(" ", "_").replace("-", "_")}_decimal.png'
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado como: {filename}")


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def information_gain_selection(X_train, y_train, X_test, k):
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    top_k_indices = np.argsort(mi_scores)[-k:]

    X_train_selected = X_train[:, top_k_indices]
    X_test_selected = X_test[:, top_k_indices]

    return X_train_selected, X_test_selected, top_k_indices


def random_forest_feature_importance(X_train, y_train, X_test, k):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    top_k_indices = np.argsort(importances)[-k:]

    X_train_selected = X_train[:, top_k_indices]
    X_test_selected = X_test[:, top_k_indices]

    return X_train_selected, X_test_selected, top_k_indices


def pca_dimensionality_reduction(X_train, X_test, n_components):

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    return X_train_pca, X_test_pca, cumulative_variance


def main():
    # Hiperparametros
    alpha = 0.9
    epochs = 10000

    # Preparar datasets con imputación
    train_imputed, test_imputed = impute_variables(train_df.copy(), test_df.copy())

    # Separar features y targets
    X_train_imputed = train_imputed.drop("nivel_riesgo", axis=1)
    X_test_imputed = test_imputed.drop("nivel_riesgo", axis=1)
    y_train_imputed = train_imputed["nivel_riesgo"]
    y_test_imputed = test_imputed["nivel_riesgo"]

    # Normalize
    X_train_norm_A, X_test_norm_A = normalize(X_train_imputed, X_test_imputed)

    # Convertir etiquetas
    y_train_encoded_A = encode_labels(y_train_imputed)
    y_test_encoded_A = encode_labels(y_test_imputed)


if __name__ == "__main__":
    main()
