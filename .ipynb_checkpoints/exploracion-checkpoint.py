import pandas as pd

# Cargar dataset principal
df = pd.read_parquet("data/at_urg_respiratorio_semanal.parquet")

print("DIMENSIONES")
print(df.shape)

print("\nCOLUMNAS")
print(df.columns.tolist())

print("\nPRIMERAS FILAS")
print(df.head())

print("\nTIPOS DE DATOS")
print(df.dtypes)

print("\nINFO GENERAL")
df.info()
