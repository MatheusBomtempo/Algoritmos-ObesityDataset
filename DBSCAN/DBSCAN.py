import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns

# Baixar a última versão do dataset de obesidade
path = kagglehub.dataset_download("lesumitkumarroy/obesity-data-set")
print("Path to dataset files:", path)

# Carregar o dataset
df = pd.read_csv(f"{path}/ObesityDataSet_raw_and_data_sinthetic.csv")

# Visualizar as primeiras linhas do dataset
print(df.head())

# Selecionar features relevantes para o agrupamento
# Utilizando features numéricas para que o DBSCAN possa processar corretamente
features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF']
X = df[features]

# Normalizar os dados para melhorar o desempenho do DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar e ajustar o modelo DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)

# Adicionar os rótulos de cluster ao DataFrame
df['Cluster'] = dbscan.labels_

# Contar quantos pontos estão em cada cluster
cluster_counts = df['Cluster'].value_counts()
print("Contagem de pontos por cluster:")
print(cluster_counts)

# Medir a confiabilidade dos agrupamentos
# Calcular a pontuação de Silhouette (ignorar ruídos)
labels = df['Cluster']
if len(set(labels)) > 1 and -1 not in labels:
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
else:
    print("Silhouette Score: Não aplicável (apenas um cluster ou todos são ruídos)")

# Calcular a pontuação de Davies-Bouldin (ignorar ruídos)
if len(set(labels)) > 1 and -1 not in labels:
    davies_bouldin_avg = davies_bouldin_score(X_scaled, labels)
    print(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")
else:
    print("Davies-Bouldin Score: Não aplicável (apenas um cluster ou todos são ruídos)")

# Visualizar os clusters com a correlação entre Altura e Peso
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Height', y='Weight', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Clusters Identificados pelo DBSCAN - Altura vs Peso')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend(title='Cluster')
plt.show()

# Gráfico de cotovelo para determinar o valor ideal de eps para DBSCAN
eps_values = np.linspace(0.1, 1.0, 10)
scores = []
for eps in eps_values:
    dbscan_temp = DBSCAN(eps=eps, min_samples=5)
    dbscan_temp.fit(X_scaled)
    labels_temp = dbscan_temp.labels_
    if len(set(labels_temp)) > 1 and -1 not in labels_temp:
        score = silhouette_score(X_scaled, labels_temp)
        scores.append(score)
    else:
        scores.append(-1)

# Plotar o gráfico de cotovelo
plt.figure(figsize=(10, 6))
plt.plot(eps_values, scores, marker='o', linestyle='-', color='b')
plt.xlabel('Valor de eps')
plt.ylabel('Pontuação de Silhouette')
plt.title('Pontuação de Silhouette para Diferentes Valores de eps (DBSCAN)')
plt.show()

# Analisar a distribuição dos clusters
unique_clusters = df['Cluster'].unique()
print(f"Número de clusters identificados (incluindo ruído): {len(unique_clusters)}")

# Verificar quantos pontos foram identificados como ruído
n_noise = sum(df['Cluster'] == -1)
print(f"Número de pontos identificados como ruído: {n_noise}")
