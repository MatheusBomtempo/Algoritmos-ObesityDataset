import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, f1_score, recall_score, precision_score
import seaborn as sns

# Baixar a última versão do dataset de obesidade
path = kagglehub.dataset_download("lesumitkumarroy/obesity-data-set")
print("Path to dataset files:", path)

# Carregar o dataset
df = pd.read_csv(f"{path}/ObesityDataSet_raw_and_data_sinthetic.csv")

# Visualizar as primeiras linhas do dataset
print(df.head())

# Selecionar features relevantes para o agrupamento
features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF']
X = df[features]

# Normalizar os dados para melhorar o desempenho do algoritmo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar o dendrograma
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrograma - Algoritmo de Agrupamento Hierárquico')
plt.xlabel('Amostras')
plt.ylabel('Distância Euclidiana')
plt.show()

# Criar e ajustar o modelo de Agrupamento Hierárquico
hierarchical = AgglomerativeClustering(n_clusters=4,  linkage='ward')
labels = hierarchical.fit_predict(X_scaled)

# Adicionar os rótulos de cluster ao DataFrame
df['Cluster'] = labels

# Contar quantos pontos estão em cada cluster
cluster_counts = df['Cluster'].value_counts()
print("Contagem de pontos por cluster:")
print(cluster_counts)

# Medir a confiabilidade dos agrupamentos
# Calcular a pontuação de Silhouette
if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
else:
    print("Silhouette Score: Não aplicável (apenas um cluster)")

# Visualizar os clusters com a correlação entre Altura e Peso
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Height', y='Weight', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Clusters Identificados pelo Agrupamento Hierárquico - Altura vs Peso')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend(title='Cluster')
plt.show()

# Avaliar a qualidade dos agrupamentos usando F1, Recall e Precisão
# Aqui assumimos que temos rótulos verdadeiros (y) e as previsões são os rótulos de cluster (labels)
# Ajustar os rótulos verdadeiros para corresponder aos clusters para cálculos de métricas de avaliação
f1 = f1_score(df['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 2, 'Obesity_Type_I': 3, 'Obesity_Type_II': 3, 'Obesity_Type_III': 3}), labels, average='weighted')
recall = recall_score(df['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 2, 'Obesity_Type_I': 3, 'Obesity_Type_II': 3, 'Obesity_Type_III': 3}), labels, average='weighted')
precision = precision_score(df['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 2, 'Obesity_Type_I': 3, 'Obesity_Type_II': 3, 'Obesity_Type_III': 3}), labels, average='weighted')

print("F1 Score:", f1)
print("Recall:", recall)
print("Precisão:", precision)
