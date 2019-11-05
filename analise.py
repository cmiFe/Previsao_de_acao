import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy as sp
from scipy import signal ,fftpack
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

def positivo(x):
    lista = []
    for i in x:
        i = i**2
        lista.append(np.sqrt(i))
    return lista

def transformTempoA(x):
    tempo = []
    for i in x['time']:
        i = i - 18812
        tempo.append(i)
    x['time'] = tempo
    return x

def transformTempoB(x):
    tempo = []
    for i in x['time']:
        i = i - 2857
        tempo.append(i)
    x['time'] = tempo
    return x

def transformTempoC(x):
    tempo = []
    for i in x['time']:
        i = i - 24038
        tempo.append(i)
    x['time'] = tempo
    return x

def juntaDF(df,df2,df3):
    df = df.append(df2)
    df = df.append(df3)
    return df


base = pd.read_csv('escrevendo.csv',sep=';')
base2 = pd.read_csv('parado.csv',sep=';')
base3 = pd.read_csv('doidao.csv',sep=';')

dataEscrevendo = pd.DataFrame(base)
dataEscrevendo.columns = ['time','magx','magy','magz','filtro(magx)','filtro(magy)','filtro(magz)','acelx','acely','acelz','filtro(acelx)','filtro(acely)','filtro(acelz)']
dataParado = pd.DataFrame(base2)
dataParado.columns = ['time','magx','magy','magz','filtro(magx)','filtro(magy)','filtro(magz)','acelx','acely','acelz','filtro(acelx)','filtro(acely)','filtro(acelz)']
dataDoidao = pd.DataFrame(base3)
dataDoidao.columns = ['time','magx','magy','magz','filtro(magx)','filtro(magy)','filtro(magz)','acelx','acely','acelz','filtro(acelx)','filtro(acely)','filtro(acelz)']

dataEscrevendo = transformTempoA(dataEscrevendo)
dataParado = transformTempoB(dataParado)
dataDoidao = transformTempoC(dataDoidao)

acelFx1 = dataEscrevendo.iloc[:,10]
acelFx2 = dataParado.iloc[:,10]
acelFx3 = dataDoidao.iloc[:,10]
acelFy1 = dataEscrevendo.iloc[:,11]
acelFy2 = dataParado.iloc[:,11]
acelFy3 = dataDoidao.iloc[:,11]
acelFz1 = dataEscrevendo.iloc[:,12]
acelFz2 = dataParado.iloc[:,12]
acelFz3 = dataDoidao.iloc[:,12]


acelF1 = pd.DataFrame({'filtro(acelx)': acelFx1, 'filtro(acely)': acelFy1,'filtro(acelz)':acelFz1})
acelF2 = pd.DataFrame({'filtro(acelx)': acelFx2, 'filtro(acely)': acelFy2,'filtro(acelz)':acelFz2})
acelF3 = pd.DataFrame({'filtro(acelx)': acelFx3, 'filtro(acely)': acelFy3,'filtro(acelz)':acelFz3})

x0 = np.zeros((len(acelF1),1))
x1 = np.ones((len(acelF2),1))
x2 = np.ones((len(acelF3),1))


for i in x2:
    i+=1

acelF1.insert(3, "saida",x0.astype(int))
acelF2.insert(3, "saida",x1.astype(int))
acelF3.insert(3, "saida",x2.astype(int))

print(acelF1.shape,acelF2.shape,acelF3.shape)
teste = []
ac = acelF3.drop('saida', 1)
teste = ac[365:371]

acelF2 = acelF2[:365]
acelF3 = acelF3[:365]

acelF = juntaDF(acelF1,acelF2,acelF3)
colunas = ['filtro(acelx)','filtro(acely)','filtro(acelz)']

x = acelF[colunas] 
y = acelF.saida

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

RandomForest = RandomForestClassifier()
RandomForest.fit(x_treino,y_treino)

AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
AdaBoost.fit(x_treino,y_treino)  
logisticRegression = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
logisticRegression.fit(x_treino,y_treino) 

y_pred = RandomForest.predict(x_teste)
y_pred2 = AdaBoost.predict(x_teste)
y_pred3 = logisticRegression.predict(x_teste)
print("precisao:",metrics.accuracy_score(y_teste, y_pred))
print("precisao:",metrics.accuracy_score(y_teste, y_pred2))
print("precisao:",metrics.accuracy_score(y_teste, y_pred3))



y_pred3 = logisticRegression.predict(teste)
print(y_pred3)
dump(logisticRegression, 'modeloRegressaoL.joblib') 






#PARTE DOS GRAFICOS COM O SINAL FILTRADO POR UM FILTRO DE MEDIA
'''
acelFx1 = sp.signal.medfilt(acelFx1.values,21)
acelFx2 = sp.signal.medfilt(acelFx2.values,21)
acelFx3 = sp.signal.medfilt(acelFx3.values,21)
acelFy1 = sp.signal.medfilt(acelFy1.values,21)
acelFy2 = sp.signal.medfilt(acelFy2.values,21)
acelFy3 = sp.signal.medfilt(acelFy3.values,21)
acelFz1 = sp.signal.medfilt(acelFz1.values,21)
acelFz2 = sp.signal.medfilt(acelFz2.values,21)
acelFz3 = sp.signal.medfilt(acelFz3.values,21)

plt.plot(dataEscrevendo.iloc[:,0], acelFx1, "b")
plt.plot(dataParado.iloc[:,0],acelFx2, "r")
plt.plot(dataDoidao.iloc[:,0], acelFx3, "g")
plt.xlabel("Time")
plt.ylabel("Acelx")
plt.show()

plt.plot(dataEscrevendo.iloc[:,0], acelFy1, "b")
plt.plot(dataParado.iloc[:,0], acelFy2, "r")
plt.plot(dataDoidao.iloc[:,0], acelFy3, "g")
plt.xlabel("Time")
plt.ylabel("Acely")
plt.show()

plt.plot(dataEscrevendo.iloc[:,0], acelFz1, "b")
plt.plot(dataParado.iloc[:,0], acelFz2, "r")
plt.plot(dataDoidao.iloc[:,0], acelFz3, "g")
plt.xlabel("Time")
plt.ylabel("Acelz")
plt.show()
'''


