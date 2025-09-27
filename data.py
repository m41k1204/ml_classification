import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "logistic_regression"))

# Ahora importar directamente
import softmax_regression

# Cargar los datasets
train_df = pd.read_csv("datos_entrenamiento_riesgo.csv")
test_df = pd.read_csv("datos_prueba_riesgo.csv")

# # Verificar la carga de datos
# print("Datos de entrenamiento:")
# print(f"Shape: {train_df.shape}")
# print(f"Columnas: {train_df.columns.tolist()}")
# print("\nPrimeras 5 filas:")
# print(train_df.head())

# print("\n" + "="*50)

# print("Datos de prueba:")
# print(f"Shape: {test_df.shape}")
# print(f"Columnas: {test_df.columns.tolist()}")
# print("\nPrimeras 5 filas:")
# print(test_df.head())


# # Verificar distribución de clases (solo en train)
# if 'nivel_riesgo' in train_df.columns:
#     print(f"\nDistribución de clases:")
#     print(train_df['nivel_riesgo'].value_counts().sort_index())


# # Analizar valores faltantes por columna
# print("Valores faltantes por columna (train):")
# missing_train = train_df.isnull().sum()
# print(missing_train[missing_train > 0].sort_values(ascending=False))

# print("\nValores faltantes por columna (test):")
# missing_test = test_df.isnull().sum()
# print(missing_test[missing_test > 0].sort_values(ascending=False))

# # Separar variables numéricas de categóricas
# # Primero veamos los tipos de datos actuales
# print("Tipos de datos:")
# print(train_df.dtypes.value_counts())

# print("\nEjemplos de variables categóricas:")
# categorical_cols = ['nivel_educativo', 'estado_civil', 'propiedad_vivienda',
#                    'tipo_vivienda', 'sector_laboral']
# for col in categorical_cols:
#     if col in train_df.columns:
#         print(f"{col}: {train_df[col].unique()[:5]}")

# # Verificar cuál es la variable object
# print("Variable object:")
# print(train_df.select_dtypes(include=['object']).columns.tolist())

# # Confirmar que nivel_riesgo es nuestra variable objetivo
# print("\nClases únicas en nivel_riesgo:")
# print(train_df['nivel_riesgo'].unique())

# # Verificar si hay patrones en los valores faltantes
# print("Porcentaje de valores faltantes por columna:")
# missing_cols = ['porcentaje_utilizacion_credito', 'sector_laboral', 'proporcion_pagos_a_tiempo',
#                 'tipo_vivienda', 'residencia_antiguedad_meses', 'nivel_educativo',
#                 'estado_civil', 'lineas_credito_abiertas']

# for col in missing_cols:
#     missing_pct = (train_df[col].isnull().sum() / len(train_df)) * 100
#     print(f"{col}: {missing_pct:.1f}%")

# # Ver si los valores faltantes están correlacionados
# print(f"\nFilas con al menos un valor faltante: {train_df.isnull().any(axis=1).sum()}")
# print(f"Total de filas: {len(train_df)}")

# # Verificar valores únicos de las categóricas (ejecuta esto primero)
# categorical_encoded = ['sector_laboral', 'tipo_vivienda', 'nivel_educativo', 'estado_civil']
# numerical = ['porcentaje_utilizacion_credito', 'proporcion_pagos_a_tiempo',
#             'residencia_antiguedad_meses', 'lineas_credito_abiertas']

# print("Verificar valores únicos de las categóricas:")
# for col in categorical_encoded:
#     unique_vals = train_df[col].dropna().unique()
#     print(f"{col}: {sorted(unique_vals)} (n={len(unique_vals)})")


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


def univariate_analysis(df, title="Análisis Univariado"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = (
        numeric_cols.drop("nivel_riesgo")
        if "nivel_riesgo" in numeric_cols
        else numeric_cols
    )

    print(f"\n=== {title} ===")
    print(f"Variables numéricas a analizar: {len(numeric_cols)}")

    print("\nEstadísticas descriptivas:")
    print(df[numeric_cols].describe())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{title} - Distribuciones")

    for i, col in enumerate(numeric_cols[:6]):
        row, col_idx = i // 3, i % 3
        axes[row, col_idx].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        axes[row, col_idx].set_title(col)
        axes[row, col_idx].set_xlabel("Valor")
        axes[row, col_idx].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(
        f'{title.replace(" ", "_")}_distribuciones.png', dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Gráfico guardado: {title.replace(' ', '_')}_distribuciones.png")


def bivariate_analysis(df, title="Análisis Bivariado"):
    """
    Análisis de correlaciones entre variables
    """
    # Separar variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "nivel_riesgo" in numeric_cols:
        numeric_cols = numeric_cols.drop("nivel_riesgo")

    print(f"\n=== {title} ===")

    # Matriz de correlación
    correlation_matrix = df[numeric_cols].corr()

    # Crear heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.1,
        cbar=False,  # Agregar esta línea
    )
    plt.title(f"{title} - Matriz de Correlación", fontsize=22)
    plt.tight_layout()
    plt.show()
    # Identificar correlaciones altas (>0.8)
    high_corr = np.where(np.abs(correlation_matrix) > 0.8)
    high_corr_pairs = [
        (
            correlation_matrix.index[x],
            correlation_matrix.columns[y],
            correlation_matrix.iloc[x, y],
        )
        for x, y in zip(*high_corr)
        if x != y and x < y
    ]

    print(f"Correlaciones altas (>0.8): {len(high_corr_pairs)}")
    for var1, var2, corr in high_corr_pairs[:10]:  # Mostrar primeras 10
        print(f"{var1} - {var2}: {corr:.3f}")

    return correlation_matrix


def detect_outliers(df, title="Detección de Outliers"):
    """
    Detecta outliers usando IQR method
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "nivel_riesgo" in numeric_cols:
        numeric_cols = numeric_cols.drop("nivel_riesgo")

    print(f"\n=== {title} ===")

    outliers_summary = []

    for col in numeric_cols[:35]:  # Analizar primeras 10 variables
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count = len(outliers)
        outliers_percentage = (outliers_count / len(df)) * 100

        outliers_summary.append(
            {
                "variable": col,
                "outliers_count": outliers_count,
                "outliers_percentage": outliers_percentage,
            }
        )

    # Mostrar resumen
    outliers_df = pd.DataFrame(outliers_summary)
    print(outliers_df)

    return outliers_df


def create_boxplots_analysis(df, title="Análisis de Boxplots"):
    """
    Crea 4 imágenes con boxplots: 3 imágenes con 9 gráficos c/u, 1 imagen con 7 gráficos
    """
    # Separar variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "nivel_riesgo" in numeric_cols:
        numeric_cols = numeric_cols.drop("nivel_riesgo")

    numeric_cols = numeric_cols.tolist()

    print(f"\n=== {title} ===")
    print(f"Total variables numéricas: {len(numeric_cols)}")

    # Dividir en grupos: 9, 9, 9, 7
    groups = [
        numeric_cols[0:9],  # Primeras 9
        numeric_cols[9:18],  # Siguientes 9
        numeric_cols[18:27],  # Siguientes 9
        numeric_cols[27:34],  # Últimas 7
    ]

    for group_idx, cols_group in enumerate(groups, 1):
        n_cols = len(cols_group)

        if n_cols == 9:
            rows, cols = 3, 3
            figsize = (15, 12)
        else:  # n_cols == 7
            rows, cols = 3, 3  # 3x3 grid, pero solo usar 7 posiciones
            figsize = (15, 12)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f"{title} - Grupo {group_idx} ({n_cols} variables)", fontsize=16)

        # Aplanar axes para fácil indexación
        axes_flat = axes.flatten()

        for i, col in enumerate(cols_group):
            ax = axes_flat[i]

            # Crear boxplot
            df[col].plot.box(ax=ax)
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("Valor")

            # Rotar etiquetas si son muy largas
            ax.tick_params(axis="x", rotation=45)

        # Ocultar subplots vacíos (solo para el grupo de 7)
        if n_cols < 9:
            for i in range(n_cols, 9):
                axes_flat[i].set_visible(False)

        plt.tight_layout()

        # Guardar imagen
        filename = f'{title.replace(" ", "_")}_boxplots_grupo_{group_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Grupo {group_idx}: {n_cols} variables - Guardado como {filename}")

        # Mostrar variables incluidas en este grupo
        print(f"Variables: {', '.join(cols_group)}")
        print()


def main():

    univariate_analysis(train_df)
    bivariate_analysis(train_df)
    # Hiperparametros
    alpha = 0.9
    epochs = 10000

    print("=" * 60)
    print("ENFOQUE A: IMPUTACIÓN DE DATOS")
    print("=" * 60)

    # ENFOQUE A: Preparar datasets con imputación
    train_imputed, test_imputed = impute_variables(train_df.copy(), test_df.copy())

    # Separar features y targets
    X_train_imputed = train_imputed.drop("nivel_riesgo", axis=1)
    X_test_imputed = test_imputed.drop("nivel_riesgo", axis=1)
    y_train_imputed = train_imputed["nivel_riesgo"]
    y_test_imputed = test_imputed["nivel_riesgo"]

    # Normalizar
    X_train_norm_A, X_test_norm_A = normalize(X_train_imputed, X_test_imputed)

    # Agregar bias
    X_train_A = add_bias(X_train_norm_A)
    X_test_A = add_bias(X_test_norm_A)

    # Convertir etiquetas
    y_train_A = encode_labels(y_train_imputed)
    y_test_A = encode_labels(y_test_imputed)

    # Entrenar Enfoque A
    W_A, LossTrain_A, LossTest_A = softmax_regression.train(
        X_train_A, y_train_A, epochs, alpha, X_test_A, y_test_A
    )
    y_pred_A, accuracy_A = softmax_regression.test(X_test_A, y_test_A, W_A)

    print("\n" + "=" * 60)
    print("ENFOQUE B: ELIMINACIÓN DE INSTANCIAS")
    print("=" * 60)

    # ENFOQUE B: Preparar datasets eliminando filas con NaN
    train_no_nan, test_no_nan = delete_nan_instances(train_df.copy(), test_df.copy())

    print(f"Tamaño original entrenamiento: {len(train_df)}")
    print(f"Tamaño después de eliminar NaN entrenamiento: {len(train_no_nan)}")
    print(f"Tamaño original prueba: {len(test_df)}")
    print(f"Tamaño después de eliminar NaN prueba: {len(test_no_nan)}")

    # Separar features y targets
    X_train_no_nan = train_no_nan.drop("nivel_riesgo", axis=1)
    X_test_no_nan = test_no_nan.drop("nivel_riesgo", axis=1)
    y_train_no_nan = train_no_nan["nivel_riesgo"]
    y_test_no_nan = test_no_nan["nivel_riesgo"]

    # Normalizar
    X_train_norm_B, X_test_norm_B = normalize(X_train_no_nan, X_test_no_nan)

    # Agregar bias
    X_train_B = add_bias(X_train_norm_B)
    X_test_B = add_bias(X_test_norm_B)

    # Convertir etiquetas
    y_train_B = encode_labels(y_train_no_nan)
    y_test_B = encode_labels(y_test_no_nan)

    # Entrenar Enfoque B
    W_B, LossTrain_B, LossTest_B = softmax_regression.train(
        X_train_B, y_train_B, epochs, alpha, X_test_B, y_test_B
    )
    y_pred_B, accuracy_B = softmax_regression.test(X_test_B, y_test_B, W_B)

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 60)

    # Importar f1_score
    from sklearn.metrics import f1_score

    # Calcular F1-scores
    f1_A = f1_score(y_test_A, y_pred_A, average="macro")
    f1_B = f1_score(y_test_B, y_pred_B, average="macro")

    print(f"F1-Score Enfoque A (Imputación): {f1_A:.4f}")
    print(f"F1-Score Enfoque B (Eliminación): {f1_B:.4f}")
    print(f"Accuracy Enfoque A: {accuracy_A:.2f}%")
    print(f"Accuracy Enfoque B: {accuracy_B:.2f}%")

    # Crear gráfico de barras comparativo
    create_comparison_chart(f1_A, f1_B, accuracy_A, accuracy_B)

    # Generar matrices de confusión
    analyze_results(y_test_A, y_pred_A, "Enfoque A - Imputación")
    analyze_results(y_test_B, y_pred_B, "Enfoque B - Eliminación")


def create_comparison_chart(f1_A, f1_B, accuracy_A, accuracy_B):
    """
    Crear gráfico de barras comparativo entre los dos enfoques
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Datos para el gráfico
    enfoques = ["Enfoque A\n(Imputación)", "Enfoque B\n(Eliminación)"]
    f1_scores = [f1_A, f1_B]
    accuracies = [
        accuracy_A / 100,
        accuracy_B / 100,
    ]  # Convertir a decimales para coherencia

    # Configurar el gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Colores
    colors = ["#2E86C1", "#E74C3C"]

    # Gráfico F1-Score
    bars1 = ax1.bar(enfoques, f1_scores, color=colors, alpha=0.8)
    ax1.set_title("Comparación F1-Score", fontsize=14, fontweight="bold")
    ax1.set_ylabel("F1-Score", fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # Agregar valores CENTRADOS en las barras
    for i, (bar, score) in enumerate(zip(bars1, f1_scores)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,  # Cambio aquí: dividir por 2
            f"{score:.3f}",
            ha="center",
            va="center",  # Cambio aquí: center en lugar de bottom
            fontweight="bold",
            color="white",  # Agregar color blanco para contraste
            fontsize=11,
        )

    # Gráfico Accuracy
    bars2 = ax2.bar(enfoques, accuracies, color=colors, alpha=0.8)
    ax2.set_title("Comparación Accuracy", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)

    # Agregar valores CENTRADOS en las barras
    for i, (bar, acc) in enumerate(zip(bars2, accuracies)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,  # Cambio aquí: dividir por 2
            f"{acc:.3f}",
            ha="center",
            va="center",  # Cambio aquí: center en lugar de bottom
            fontweight="bold",
            color="white",  # Agregar color blanco para contraste
            fontsize=11,
        )

    plt.tight_layout()

    # Guardar el gráfico
    filename = "comparacion_enfoques.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Gráfico de comparación guardado como: {filename}")

    # También crear un gráfico combinado
    create_combined_chart(f1_A, f1_B, accuracy_A / 100, accuracy_B / 100)


def create_combined_chart(f1_A, f1_B, acc_A, acc_B):
    """
    Crear gráfico combinado con ambas métricas
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Configurar datos
    x = np.arange(2)
    width = 0.35

    enfoques = ["Enfoque A\n(Imputación)", "Enfoque B\n(Eliminación)"]
    f1_scores = [f1_A, f1_B]
    accuracies = [acc_A, acc_B]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear barras
    bars1 = ax.bar(
        x - width / 2, f1_scores, width, label="F1-Score", color="#3498DB", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, accuracies, width, label="Accuracy", color="#E67E22", alpha=0.8
    )

    # Personalizar gráfico
    ax.set_title(
        "Comparación de Enfoques: F1-Score y Accuracy", fontsize=16, fontweight="bold"
    )
    ax.set_ylabel("Puntuación", fontsize=12)
    ax.set_xlabel("Enfoques", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(enfoques)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Agregar valores CENTRADOS en las barras
    for bar, score in zip(bars1, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,  # Cambio aquí: dividir por 2
            f"{score:.3f}",
            ha="center",
            va="center",  # Cambio aquí: center en lugar de bottom
            fontweight="bold",
            color="white",  # Agregar color blanco para contraste
            fontsize=10,
        )

    for bar, acc in zip(bars2, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,  # Cambio aquí: dividir por 2
            f"{acc:.3f}",
            ha="center",
            va="center",  # Cambio aquí: center en lugar de bottom
            fontweight="bold",
            color="white",  # Agregar color blanco para contraste
            fontsize=10,
        )

    plt.tight_layout()

    # Guardar el gráfico
    filename = "comparacion_combinada_enfoques.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Gráfico combinado guardado como: {filename}")


if __name__ == "__main__":
    main()
