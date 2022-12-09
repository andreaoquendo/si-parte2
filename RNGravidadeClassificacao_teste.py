import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, accuracy_score, f1_score

# Conseguir sinais vitais
sinais_vitais = pd.read_csv(r'sinais_vitais_teste.txt')
sinais = sinais_vitais[['qPA','pulso', 'resp']]
sinais = sinais.to_numpy()

gravidade = sinais_vitais['grav']

regressao = load('neuralNetRegre.joblib') 
predicao_gravidade = regressao.predict(sinais)

print("MSE:", mean_squared_error(gravidade, predicao_gravidade, squared=False))
print("MAE:", mean_absolute_error(gravidade, predicao_gravidade))

gravidade = pd.DataFrame(predicao_gravidade, columns = ['grav'])
classificacao = load('neuralNetClass.joblib') 
predicao_risco = classificacao.predict(gravidade)

risco = sinais_vitais['risco']

print("Precisão:",precision_score(risco, predicao_risco, average=None))
print("Recall:",recall_score(risco, predicao_risco, average=None))
print("Acuracidade:",accuracy_score(risco, predicao_risco))
print("F1:",f1_score(risco, predicao_risco, average='weighted'))
