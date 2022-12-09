import pandas as pd
import numpy as np
from joblib import load

# Conseguir sinais vitais
sinais_vitais = pd.read_csv(r'sinais_vitais_teste.txt')
#sinais_vitais.set_index('id', inplace=True)
sinais = sinais_vitais[['qPA','pulso', 'resp']]
sinais = sinais.to_numpy()

regressao = load('neuralNetRegre.joblib') 
predicao_gravidade = regressao.predict(sinais)

gravidade = pd.DataFrame(predicao_gravidade, columns = ['grav'])
classificacao = load('neuralNetClass.joblib') 
predicao_risco = classificacao.predict(gravidade)

analise = pd.DataFrame({ 'id': sinais_vitais['id'],
                        'gravidade': predicao_gravidade, 
                         'risco' : predicao_risco})

analise.to_csv('dados_neurais.csv', index=False)
