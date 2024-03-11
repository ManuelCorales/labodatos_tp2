# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Grupo       : Grupo 2
Detalle     : En este codigo se busca responder al punto 3 del tp2
        Dada una imagen, corresponde a qué seña de las vocales representa
    
Autores     : Corales, Biasoni y Soler

"""

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from inline_sql import sql
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import sklearn.metrics
from src.utils_imagenes import obtenerDfGeneral



def main():
    evaluacionMulticlase()

#%%
def evaluacionMulticlase():
    print('######## EVALUACIÓN MULTICLASE ########')
    vocales: pd.DataFrame = datos()
    X_train, X_test, y_train, y_test = separarDatos(vocales)
    arbol = obtenerMejorArbol(X_train, X_test, y_train, y_test)
    matrizConfusion = evaluarModelo(arbol, X_test, y_test)
    graficarMatrizConfusion(matrizConfusion)


#%%
def datos():
    # Importamos datos
    datos = obtenerDfGeneral()

    # creamos dataframe con subconjunto
    # van a ser requeridos los datos con label=0 (A), label =4 (E) 
    # label = 8 (I), label =13 (O) y label = 19 (U)

    # CREO UN DATASET SOLO DE VOCALES#
    # Hay 1126 datos para la letra a
    # Hay 957 datos para la letra e
    # Hay 1162 datos para la letra i
    # Hay 1196 datos para la letra o
    # Hay 1161 datos para la letra u

    vocales= sql^"""
             Select * 
             FROM datos
             WHERE label = 0 OR label =4 OR label = 8
              OR label = 13 OR label = 19
             """
    return vocales




#%%
## Generacion archivos TEST / TRAIN
# Dividimos los datos en Desarrollo y Evaluacion
# Reservamos vocales_Evaluacion para testear el modelo al final, luego de elegir uno
def separarDatos(vocales):
    X= vocales.drop(columns = ['label'])
    Y= vocales['label']

    X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=0.15 , shuffle=True, stratify=  Y)

    return X_train, X_test, y_train, y_test

#%%
# Busco el mejor arbol
def analiza_modelos(X_train, y_train):
      hyper_params = {'criterion' : ["gini", "entropy"],
                   'max_depth' : [2,3,4,5,6] }
      arbol = DecisionTreeClassifier()
      clf = RandomizedSearchCV(arbol, hyper_params, random_state= 0, n_iter= 5)
      clf.fit(X_train, y_train)
      print(clf.best_params_)
      print(clf.best_score_)
      return clf


def obtenerMejorArbol(X_train, X_test, y_train, y_test):
    arbol = analiza_modelos(X_train, y_train)
    # El mejor arbol tienen los hyperparametros max depth = 6 y criterio = entropy
    score = arbol.score(X_test, y_test) # Evaluamos el modelo
    print("Score del árbol: ", score)

    return arbol


#%%
# Evaluo el modelo utilizando diferentes metricas
def evaluarModelo(arbol, X_test, y_test):
    y_real= y_test
    y_predicho= arbol.predict(X_test)
    exactitud = sklearn.metrics.accuracy_score(y_real, y_predicho)
    precision = sklearn.metrics.precision_score(y_real, y_predicho, average=None)
    recall = sklearn.metrics.recall_score(y_real, y_predicho, average=None)
    print('exactitud del modelo:', exactitud)
    print('Precisión por clase:', precision)
    print('Exhaustividad por clase:', recall)
    #matriz de confusion
    mc= sklearn.metrics.confusion_matrix(y_real, y_predicho)
    return mc



#%%
# Graficamos matiz de confusion
def graficarMatrizConfusion(matrizConfusion):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  
    sns.heatmap(matrizConfusion,
                annot=True,
                fmt='d', 
                cmap='PuRd', 
                xticklabels=["A","E","I","O","U"], 
                yticklabels=["A","E","I","O","U"])  
    plt.title('Matriz de Confusión')
    plt.xlabel('Clases Predictivas')
    plt.ylabel('Clases Verdaderas')
    plt.show()


if(__name__ == "__main__"):
    main()