import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Baixar a última versão do dataset de obesidade
path = kagglehub.dataset_download("lesumitkumarroy/obesity-data-set")

# Carregar o dataset
df = pd.read_csv(f"{path}/ObesityDataSet_raw_and_data_sinthetic.csv")

# Pré-processamento: Selecionar features e alvo
X = df.drop(columns=['NObeyesdad'])  # Todas as colunas, exceto o alvo
y = df['NObeyesdad']  # Alvo (abaixo do peso, normal, sobrepeso, obesidade)

# Codificar a coluna de rótulo (transformar categorias em números)
y = y.map({
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 2,
    'Obesity_Type_I': 3,
    'Obesity_Type_II': 3,
    'Obesity_Type_III': 3
})

# Codificar variáveis categóricas usando One-Hot Encoding
X = pd.get_dummies(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um modelo de classificação k-NN
knn = KNeighborsClassifier(n_neighbors=12)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Abaixo do Peso', 'Normal', 'Sobrepeso', 'Obesidade'], yticklabels=['Abaixo do Peso', 'Normal', 'Sobrepeso', 'Obesidade'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.show()

# Verificando a distribuição das classes no dataset original
df['Obesidade'] = y  # Adiciona a coluna 'Obesidade' ao DataFrame original
sns.countplot(x='Obesidade', data=df)
plt.title('Distribuição das Classes de Obesidade no Dataset Original')
plt.xlabel('Classe (0: Abaixo do Peso, 1: Normal, 2: Sobrepeso, 3: Obesidade)')
plt.show()

# Gráfico de cotovelo para determinar o melhor valor de k
plt.figure(figsize=(10, 6))
k_range = range(1, 21)
accuracies = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_temp_pred = knn_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_temp_pred))

# Plotar a acurácia para diferentes valores de k
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia')
plt.title('Acurácia do KNN para Diferentes Valores de k')
plt.show()

# Gráfico de dispersão usando Altura vs Peso
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test['Obesidade_Predita'] = y_pred
df_test['Obesidade_Predita'] = df_test['Obesidade_Predita'].map({
    0: 'Abaixo do Peso', 1: 'Normal', 2: 'Sobrepeso', 3: 'Obesidade'
})

for i, categoria in enumerate(df_test['Obesidade_Predita'].unique()):
    plt.scatter(df_test[df_test['Obesidade_Predita'] == categoria]['Height'],
                df_test[df_test['Obesidade_Predita'] == categoria]['Weight'],
                color=colors[i], label=categoria)

plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Classificação de Obesidade - Altura vs Peso')
plt.legend()
plt.show()