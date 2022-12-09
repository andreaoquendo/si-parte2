from sklearn.neural_network import MLPClassifier
from sklearn import tree
import pandas as pd
from joblib import dump, load

from  sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, accuracy_score, f1_score
from matplotlib import pyplot as plt
import numpy as np

sinais_vitais = pd.read_csv(r'sinais_vitais.txt')
sinais_vitais.set_index('id', inplace=True)

# TREINO DA REGRESSÃO

sinais = sinais_vitais[['qPA','pulso', 'resp']]
gravidade = sinais_vitais['grav']

sinais = sinais.to_numpy()
gravidade = np.ravel(gravidade)

sinais_treino, sinais_teste, gravidade_treino, gravidade_teste = train_test_split(sinais, gravidade,random_state=1)

# parametrização da RN
clf = tree.DecisionTreeRegressor(max_depth=15)

clf = clf.fit(sinais_treino, gravidade_treino)

predicao = clf.predict(sinais_teste)

# erros
print("MSE:", mean_squared_error(gravidade_teste, predicao, squared=False))
print("MAE:", mean_absolute_error(gravidade_teste, predicao))

dump(clf, 'cartRegre.joblib') 
clf = load('cartRegre.joblib') 

# TREINO DA CLASSIFICAÇÃO

gravidade = sinais_vitais[['grav']]
risco = sinais_vitais['risco']
risco = np.ravel(risco)

gravidade_treino, gravidade_teste, risco_treino, risco_teste = train_test_split(gravidade, risco, stratify=risco, random_state=1)

clf2 = tree.DecisionTreeClassifier(criterion="gini", splitter="best")

clf2 = clf2.fit(gravidade_treino, risco_treino)

predicao = clf2.predict(gravidade_teste)

print("Precisão:",precision_score(risco_teste, predicao, average=None))
print("Recall:",recall_score(risco_teste, predicao, average=None))
print("Acuracidade:",accuracy_score(risco_teste, predicao))
print("F1:",f1_score(risco_teste, predicao, average='weighted'))

dump(clf2, 'cartClass.joblib') 
clf2 = load('cartClass.joblib') 

predicao = clf2.predict(gravidade_teste)