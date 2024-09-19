import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



data = pd.read_csv('game_data_all.csv')

# Exploración inicial del dataset
print("Primeras filas del dataset:")
print(data.head())
print("\nInformación del dataset:")
print(data.info())
print("\nEstadísticas descriptivas del dataset:")
print(data.describe())

# Estadística básica del dataset
numericas = data.select_dtypes(include=['float64', 'int64'])
no_numericas = data.select_dtypes(exclude=['float64', 'int64'])



# Media, mediana y desviación estándar para variables numéricas
print("\nAnálisis descriptivo de variables numéricas:")
for col in numericas.columns:
    print(f"\nColumna: {col}")
    print(f"Media: {numericas[col].mean()}")
    print(f"Mediana: {numericas[col].median()}")
    print(f"Desviación Estándar: {numericas[col].std()}")

# Moda para variables no numéricas
print("\nAnálisis descriptivo de variables no numéricas:")
for col in no_numericas.columns:
    print(f"\nColumna: {col}")
    print(f"Moda: {no_numericas[col].mode()[0]}")



# Agrupar los datos de 'positive_reviews' en rangos
bins = [0, 100, 500, 1000, 5000, 10000, 50000, 100000, data['positive_reviews'].max()]
labels = ['0-100', '101-500', '501-1000', '1001-5000', '5001-10000', '10001-50000', '50001-100000', '100001+']
data['positive_reviews_group'] = pd.cut(data['positive_reviews'], bins=bins, labels=labels, include_lowest=True)


# Tabla de frecuencia para la columna agrupada 'positive_reviews_group'
tabla_frecuencia_positive_review_grouped = data['positive_reviews_group'].value_counts().sort_index()
print("\nTabla de Frecuencia de Positive Reviews Agrupada en Rangos:\n")
print(tabla_frecuencia_positive_review_grouped)


# Gráfico de barras para los grupos de 'positive_reviews'
plt.figure(figsize=(10, 6))
plt.bar(tabla_frecuencia_positive_review_grouped.index.astype(str),
         tabla_frecuencia_positive_review_grouped.values, color='lightgreen', edgecolor='black')
plt.title('Frecuencia de Positive Reviews (Agrupado por Rangos)')
plt.xlabel('Rango de Positive Reviews')
plt.ylabel('Frecuencia')
plt.xticks(rotation=20)
plt.grid(True)
plt.show()


#  Detección de Anomalías
# Seleccionar las columnas para la detección de anomalías
data_clean = data[['positive_reviews', 'negative_reviews', 'peak_players']].dropna()

# Escalar los datos para normalizar
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_clean)

# Definir y entrenar el modelo 
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
data_clean['anomaly'] = isolation_forest.fit_predict(scaled_data)

# Anomalías detectadas
anomalies = data_clean[data_clean['anomaly'] == -1]
print(f"Anomalías detectadas: {len(anomalies)}")

# Visualización de anomalías en Positive Reviews y Negative Reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='positive_reviews', y='negative_reviews', hue='anomaly', data=data_clean, palette='coolwarm', alpha=0.6)
plt.title('Detección de Anomalías: Positive Reviews vs Negative Reviews')
plt.xlabel('Positive Reviews')
plt.ylabel('Negative Reviews')
plt.grid(True)
plt.show()

# Visualización de anomalías en Peak Players y Positive Reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='peak_players', y='positive_reviews', hue='anomaly', data=data_clean, palette='coolwarm', alpha=0.6)
plt.title('Detección de Anomalías: Peak Players vs Positive Reviews')
plt.xlabel('Peak Players')
plt.ylabel('Positive Reviews')
plt.grid(True)
plt.show()


# Mapa de Calor
# Calcular la matriz de correlación entre variables numéricas
correlation_matrix = data_clean[['positive_reviews', 'negative_reviews', 'peak_players']].corr()

# Visualizar el mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Clustering

kmeans = KMeans(n_clusters=3, random_state=42)

# Ajustar el modelo a los datos escalados
data_clean['cluster'] = kmeans.fit_predict(scaled_data)

# Visualización del clustering para Positive Reviews vs Negative Reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='positive_reviews', y='negative_reviews', hue='cluster', data=data_clean, palette='viridis', alpha=0.6)
plt.title('Clustering K-Means: Positive Reviews vs Negative Reviews')
plt.xlabel('Positive Reviews')
plt.ylabel('Negative Reviews')
plt.grid(True)
plt.show()

# Visualización del clustering para Peak Players vs Positive Reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='peak_players', y='positive_reviews', hue='cluster', data=data_clean, palette='viridis', alpha=0.6)
plt.title('Clustering K-Means: Peak Players vs Positive Reviews')
plt.xlabel('Peak Players')
plt.ylabel('Positive Reviews')
plt.grid(True)
plt.show()
