import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import kagglehub
import seaborn as sns

# Baixar a última versão do dataset de obesidade
path = kagglehub.dataset_download("lesumitkumarroy/obesity-data-set")

# Carregar o dataset
df = pd.read_csv(f"{path}/ObesityDataSet_raw_and_data_sinthetic.csv")

# Visualizar as primeiras linhas dos dados
print(df.head())

# Pré-processamento: Selecionar apenas as features 'Age', 'Height' e 'Weight', e o alvo
df = df[['Age', 'Height', 'Weight', 'NObeyesdad']]
X = df[['Age', 'Height', 'Weight']]
y = df['NObeyesdad']

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

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo SVM {'C': 1, 'gamma': 0.1, 'kernel': 'linear'}
clf = svm.SVC(C=1, gamma=0.1, kernel="linear")
clf.fit(X_train, y_train)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
print("Acurácia:", accuracy)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precisão:", precision)
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Abaixo do Peso', 'Normal', 'Sobrepeso', 'Obesidade'], yticklabels=['Abaixo do Peso', 'Normal', 'Sobrepeso', 'Obesidade'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.show()

# Gráfico de dispersão usando Altura vs Peso
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'orange', 'red']
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test['Obesidade_Predita'] = y_pred
df_test['Obesidade_Predita'] = df_test['Obesidade_Predita'].map({
    0: 'Abaixo do Peso', 1: 'Normal', 2: 'Sobrepeso', 3: 'Obesidade'
})

for i, categoria in enumerate(df_test['Obesidade_Predita'].unique()):
    plt.scatter(df_test[df_test['Obesidade_Predita'] == categoria]['Height'],
                df_test[df_test['Obesidade_Predita'] == categoria]['Weight'],
                color=colors[i], label=categoria, alpha=0.6)

plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Classificação de Obesidade - Altura vs Peso')
plt.legend()
plt.show()

# Gráfico de cotovelo para determinar o valor ideal de C para SVM
c_range = np.logspace(-3, 3, 10)
accuracies = []
for c in c_range:
    clf_temp = svm.SVC(C=c, gamma=0.1, kernel="linear")
    clf_temp.fit(X_train, y_train)
    y_temp_pred = clf_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_temp_pred))

# Plotar a acurácia para diferentes valores de C
plt.figure(figsize=(10, 6))
plt.plot(c_range, accuracies, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('Valor de C (Regularização)')
plt.ylabel('Acurácia')
plt.title('Acurácia do SVM para Diferentes Valores de C')
plt.show()
