import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import logistic_regression.softmax_regression as softmax_regression

# Cargar los datasets
train_df = pd.read_csv('datos_entrenamiento_riesgo.csv')
test_df = pd.read_csv('datos_prueba_riesgo.csv')

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

# # Verificar valores faltantes
# print(f"\nValores faltantes en train: {train_df.isnull().sum().sum()}")
# print(f"Valores faltantes en test: {test_df.isnull().sum().sum()}")

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
    categorical_encoded = ['sector_laboral', 'tipo_vivienda', 'nivel_educativo', 'estado_civil']
    for col in categorical_encoded:
        mode_train = train_df[col].mode()[0]
        train_df[col].fillna(mode_train, inplace=True)
        test_df[col].fillna(mode_train, inplace=True)

    numerical = ['porcentaje_utilizacion_credito', 'proporcion_pagos_a_tiempo', 
                'residencia_antiguedad_meses', 'lineas_credito_abiertas']
    for col in numerical:
        median_train = train_df[col].median()
        train_df[col].fillna(median_train, inplace=True)
        test_df[col].fillna(median_train, inplace=True)  
    return train_df, test_df

def delete_nan_columns(train_df, test_df):    
    return train_df.dropna(), test_df.dropna()

def normalize(x_train, x_test):
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)  
    x_test_norm = scaler.transform(x_test)        
    return x_train_norm, x_test_norm

def add_bias(x):
    return np.column_stack([np.ones(x.shape[0]), x])

def encode_labels(y):
    label_map = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
    return y.map(label_map).values

def analyze_results(y_true, y_pred, title="Matriz de Confusión"):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Bajo', 'Medio', 'Alto']
    
    cm_decimal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_decimal, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    
    plt.title(title)
    plt.tight_layout()
    filename = f'{title.replace(" ", "_").replace("-", "_")}_decimal.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  
    print(f"Gráfico guardado como: {filename}")
    
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def main():
    # definimos alpha y epochs
    alpha = 0.9
    epochs = 10000

    # ENFOQUE A - Imputación
    print("="*50)
    print("PROBANDO ENFOQUE A - IMPUTACIÓN")
    print("="*50)
    
    # Preparar datasets con imputación
    train_imputed, test_imputed = impute_variables(train_df.copy(), test_df.copy())
    
    # Separar features y targets
    X_train_imputed = train_imputed.drop('nivel_riesgo', axis=1)
    X_test_imputed = test_imputed.drop('nivel_riesgo', axis=1)
    y_train_imputed = train_imputed['nivel_riesgo']
    y_test_imputed = test_imputed['nivel_riesgo']
    
    # normalize
    X_train_norm_A, X_test_norm_A = normalize(X_train_imputed, X_test_imputed)
    
    # Añadir bias
    X_train_bias_A = add_bias(X_train_norm_A)
    X_test_bias_A = add_bias(X_test_norm_A)
    
    # Convertir etiquetas
    y_train_encoded_A = encode_labels(y_train_imputed)
    y_test_encoded_A = encode_labels(y_test_imputed)
    
    # ENFOQUE B - Eliminación de filas
    print("\n" + "="*50)
    print("PROBANDO ENFOQUE B - ELIMINACIÓN")
    print("="*50)
    
    train_clean, test_clean = delete_nan_columns(train_df.copy(), test_df.copy())
    
    # Preparar datasets B
    X_train_clean = train_clean.drop('nivel_riesgo', axis=1)
    X_test_clean = test_clean.drop('nivel_riesgo', axis=1)
    y_train_clean = train_clean['nivel_riesgo']
    y_test_clean = test_clean['nivel_riesgo']
    
    X_train_norm_B, X_test_norm_B = normalize(X_train_clean, X_test_clean)
    X_train_bias_B = add_bias(X_train_norm_B)
    X_test_bias_B = add_bias(X_test_norm_B)
    
    y_train_encoded_B = encode_labels(y_train_clean)
    y_test_encoded_B = encode_labels(y_test_clean)


    print("---------------------------- Comenzando entrenamiento ----------------------------")
    W_A, LossTrain_A, LossTest_A = softmax_regression.train(
        X_train_bias_A, y_train_encoded_A, epochs, alpha, X_test_bias_A, y_test_encoded_A
    )
    
    y_pred_A, accuracy_A = softmax_regression.test(X_test_bias_A, y_test_encoded_A, W_A)
    
    print(f"\nRESULTADOS ENFOQUE A:")
    analyze_results(y_test_encoded_A, y_pred_A, "Softmax Regression without PCA A")

    W_B, LossTrain_B, LossTest_B = softmax_regression.train(
        X_train_bias_B, y_train_encoded_B, epochs, alpha, X_test_bias_B, y_test_encoded_B
    )
    
    y_pred_B, accuracy_B = softmax_regression.test(X_test_bias_B, y_test_encoded_B, W_B)
    
    print(f"\nRESULTADOS ENFOQUE B:")
    analyze_results(y_test_encoded_B, y_pred_B, "Softmax Regression without PCA B")


if __name__ == '__main__':
    main()
