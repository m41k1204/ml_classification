import pandas as pd

# Cargar los datasets
train_df = pd.read_csv('datos_entrenamiento_riesgo.csv')
test_df = pd.read_csv('datos_prueba_riesgo.csv')

# Verificar la carga de datos
print("Datos de entrenamiento:")
print(f"Shape: {train_df.shape}")
print(f"Columnas: {train_df.columns.tolist()}")
print("\nPrimeras 5 filas:")
print(train_df.head())

print("\n" + "="*50)

print("Datos de prueba:")
print(f"Shape: {test_df.shape}")
print(f"Columnas: {test_df.columns.tolist()}")
print("\nPrimeras 5 filas:")
print(test_df.head())

# Verificar valores faltantes
print(f"\nValores faltantes en train: {train_df.isnull().sum().sum()}")
print(f"Valores faltantes en test: {test_df.isnull().sum().sum()}")

# Verificar distribución de clases (solo en train)
if 'nivel_riesgo' in train_df.columns:
    print(f"\nDistribución de clases:")
    print(train_df['nivel_riesgo'].value_counts().sort_index())