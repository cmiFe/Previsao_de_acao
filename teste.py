from joblib import dump, load
import pandas as pd
import numpy as np
from joblib import dump, load

def classifica(previsao):
    parado =[]
    escrevendo = []
    doidao = []
    for i in previsao:
        if i == 0:
            parado.append(i)
        elif i == 1:
            escrevendo.append(i)
        elif i == 2:
            doidao.append(i)
    print(len(parado),len(escrevendo),len(doidao))
    if parado > escrevendo and parado > doidao:
        print('parado')
    elif escrevendo > parado and escrevendo > doidao:
        print('escrevendo')
    elif doidao > escrevendo and doidao > parado:
        print('doidao')


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
modelo = load('modeloRegressaoL.joblib') 


base = pd.read_csv('parado_teste.csv',sep=';')

data = pd.DataFrame(base)
data.columns = ['time','magx','magy','magz','filtro(magx)','filtro(magy)','filtro(magz)','acelx','acely','acelz','filtro(acelx)','filtro(acely)','filtro(acelz)']

data = data.drop('time',1)
data = data.drop('magx',1)
data = data.drop('magy',1)
data = data.drop('magz',1)
data = data.drop('filtro(magx)',1)
data = data.drop('filtro(magy)',1)
data = data.drop('filtro(magz)',1)
data = data.drop('acelx',1)
data = data.drop('acely',1)
data = data.drop('acelz',1)
previsao = modelo.predict(data)

classifica(previsao)

