import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# Carregar a base de dados
data = pd.read_excel('C:\\Users\\Operador\\Documents\\MeusProjetos\\Final_Roney\\Base.xlsx')

# Criar coluna de Classe
def classificar_time(row):
    if row['Número de triunfos'] > row['Número de derrotas']:
        return 'Superior'
    elif row['Número de triunfos'] < row['Número de derrotas']:
        return 'Inferior'
    else:
        return 'Igual'

data['Classe'] = data.apply(classificar_time, axis=1)

# Selecionar as features e o target
X = data[['Número de jogos', 'Número de triunfos', '% de triunfos', 'Número de empates', 
          '% de empates', 'Número de derrotas', '% de derrotas', 'Gols marcados', 'Gols sofridos']]
y = data['Classe']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalonar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Exibir os resultados
print(f"Acurácia: {accuracy}")
print("Relatório de Classificação:\n", classification_rep)
print("Matriz de Confusão:\n", conf_matrix)

# Reduzir para 2D usando PCA
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
X_test_2D = pca.transform(X_test)

# Treinar novamente o SVM com os dados 2D
svm_model_2D = SVC(kernel='linear', random_state=42)
svm_model_2D.fit(X_train_2D, y_train)

# Plotando as margens de decisão e vetores de suporte
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_2D[:, 0], y=X_train_2D[:, 1], hue=y_train, style=y_train, palette='Set2')

# Adicionar as margens de decisão
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Criação da grade para as margens de decisão
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
Z = svm_model_2D.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Contornos das margens
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Vetores de suporte
ax.scatter(svm_model_2D.support_vectors_[:, 0], svm_model_2D.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("Margens de Decisão e Vetores de Suporte (SVM com PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()
