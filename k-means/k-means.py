import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Download do conjunto de dados
path = kagglehub.dataset_download("lesumitkumarroy/obesity-data-set")
print("Caminho para os arquivos do conjunto de dados:", path)

# Carregar o conjunto de dados
file_path = path + '/ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)

# Extrair características relevantes: Altura e Peso
X = df[['Height', 'Weight']].copy()

# Escalar as características para o K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do Cotovelo para encontrar o número ótimo de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotar o Método do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.title('Método do Cotovelo para Número Ótimo de Clusters')
plt.show()

# A partir do Método do Cotovelo, vamos assumir que o número ótimo de clusters é 3
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Adicionar rótulos de clusters ao conjunto de dados original
df['Cluster'] = labels

# Plotar os clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroides')
plt.xlabel('Altura (escalada)')
plt.ylabel('Peso (escalado)')
plt.title('Clusters do Conjunto de Dados de Obesidade')
plt.legend()
plt.show()

# Avaliação dos clusters
silhouette_avg = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)

print(f'Silhouette Score: {silhouette_avg:.2f}')
print(f'Davies-Bouldin Score: {davies_bouldin:.2f}')

# Usar um classificador KNN para avaliar a acurácia do agrupamento
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Métrica de Acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia: {acuracia:.2f}')
