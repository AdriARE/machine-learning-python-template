import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/workspaces/machine-learning-python-template/data/raw/AB_NYC_2019.csv')
df.head()

# INFORMACIÓN DEL DATASET
print(f"Número de propiedades: {df.shape[0]}")
print(f"Número de características: {df.shape[1]}")
print(f"Esto significa: {df.shape[0]} filas y {df.shape[1]} columnas")

# TIPOS DE DATOS
df.info()

# ESTADÍSTICAS BÁSICAS
df.describe()

# VALORES FALTANTES
valores_faltantes = df.isnull().sum()
print("Cantidad de valores faltantes por columna:")
print(valores_faltantes)

# HISTOGRAMA DE PRECIOS
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='price', bins=500)
plt.title('Distribución de Precios de Airbnb en NYC')
plt.xlim(0, 1000)
plt.show()

#TIPOS DE HABITACIÓN
plt.figure(figsize=(6,6))
sns.countplot(data=df, x='room_type')
plt.title('Distribución de Tipos de Habitación')
plt.xticks(rotation=15)
plt.show()

#PRECIO POR TIPO DE HABITACIÓN
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='room_type', y='price')
plt.title('Precio por Tipo de Habitación')
plt.xticks(rotation=15)
plt.ylim(0, 500)

# DISTRIBUCIÓN POR BARRIOS
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='neighbourhood_group')
plt.title('Número de Propiedades por Barrio')
plt.xticks(rotation=45)
plt.show()

# DISTRIBUCION DE PRECIOS POR BARRIO
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[df['price'] <= 500], x='neighbourhood_group', y='price')
plt.title('Distribución de Precios por Barrio')
plt.xlabel('Barrio')
plt.ylabel('Precio ($)')
plt.ylim(0, 400)
plt.show()

# MAPA DE CALOR DE CORRELACIONES
plt.figure(figsize=(10, 8))
numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'availability_365']

sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlaciones entre Variables Numéricas')
plt.tight_layout()
plt.show()

# CÁLCULO DE PORCENTAJES PARA ANALISIS
print("CÁLCULO DE PORCENTAJES")

# 1. Porcentajes de tipos de habitación
print("1. PORCENTAJES DE TIPOS DE HABITACIÓN:")
porcentajes_habitacion = df['room_type'].value_counts(normalize=True) * 100
print(porcentajes_habitacion.round(1))

# 2. Porcentajes por barrios
print("2. PORCENTAJES POR BARRIOS:")
porcentajes_barrios = df['neighbourhood_group'].value_counts(normalize=True) * 100
print(porcentajes_barrios.round(1))

# 3. Medianas de precios por tipo
print("3. PRECIOS MEDIANOS POR TIPO:")
medianas_precios = df.groupby('room_type')['price'].median()
print(medianas_precios.round(0))


print("ANÁLISIS DE OUTLIERS EN PRECIOS (MÉTODO IQR):")
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Q1 (25%): {Q1}")
print(f"Q3 (75%): {Q3}")
print(f"IQR: {IQR}")
print(f"Límite inferior para outliers: {limite_inferior}")
print(f"Límite superior para outliers: {limite_superior}")

outliers_price = df[(df['price'] < limite_inferior) | (df['price'] > limite_superior)]
print(f"Propiedades consideradas outliers en precio: {len(outliers_price)} ({len(outliers_price)/len(df)*100:.1f}%)")

# RESUMEN FINAL CONCLUSIONES
print("CONCLUSIONES FINALES DEL EDA")
print("1. Manhattan concentra 44.3% de propiedades y precios más altos")
print("2. 52% son casas enteras, 45.7% habitaciones privadas")
print("3. 20.6% de propiedades nunca han recibido reseñas")
print("4. Los precios tienen alta variabilidad (outliers extremos)")

# LIMPIEZA ANTES DEL ENTRENAMIENTO Y TEST
columnas_eliminar = ['id', 'name', 'host_name', 'last_review', 'host_id']
df_clean = df.drop(columns=columnas_eliminar)

print(f"Eliminadas columnas: {columnas_eliminar}")
print(f"Dataset original: {df.shape}")
print(f"Dataset limpio: {df_clean.shape}")

# DIVIDIR EN TEST & TRAIN
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# GUARDAR
train_df.to_csv('/workspaces/machine-learning-python-template/data/processed/train.csv', index=False)
test_df.to_csv('/workspaces/machine-learning-python-template/data/processed/test.csv', index=False)

