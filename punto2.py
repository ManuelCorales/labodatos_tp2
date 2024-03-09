# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Grupo       : Grupo 2
Detalle     : En este codigo se busca responder al punto 2 del tp2
        Dada una imagen, corresponde a una seña de la L o a una seña de la A
    
Autores     : Corales, Biasoni y Soler

"""

#%% importamos librerias
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from inline_sql import sql, sql_val
from pandas import DataFrame


#%%  Construir nuevo dataframe a partir del original que imagenes de letras L y A solamente

def datos():
    ### importamos datos
    carpeta = "./labodatos_tp2/"
    datos = pd.read_csv(carpeta + "sign_mnist_train.csv")

    ### creamos dataframe con subconjunto
    ### van a ser requeridos los datos con label=0 (A), y label=11(L)
    consulta = sql^"""
                    SELECT * 
                    FROM datos
                    WHERE label= 0 OR label= 11 
                   """
    LoA: DataFrame = consulta
    
#%% Sobre el subconjunto, analizar cuantas muestras se tienen

def analizar_muestras():
    ### contemos si las muestras estan balanceadas
    consulta2 = sql^"""
                    SELECT count(*) AS 'muestrasxletra'
                    FROM LoA
                    GROUP BY label
                   """
    print(consulta2)
    
    ### obtenemos que de la A hay 1.126, y 1.241 de la L...No tan desbalanceado
    ###pero tampoco cantidades iguales
   
    
#%%
#################################################
## Generacion archivos TEST / TRAIN
#################################################
# Dividimos los datos en Desarrollo y Evaluacion
# Reservamos LoA_Evaluacion para testear el modelo al final, luego de elegir uno

LoA_Desarrollo, LoA_Evaluacion = train_test_split(LoA, test_size = 0.15, shuffle= True, stratify = LoA['label'], random_state= 314)
    

#%%

# vamos a dividir el dataset en distintas cantidades de atributos
cant_atributos = [4, 300, 302, 304]

def experimento():

        # Rango de valores por los que se va a mover k
        valores_k = range(1, 9)
        
        resultados = {}
        
        # Realizamos la combinacion de todos los modelos (atributos x k)
        for atributo in cant_atributos:
            resultados[atributo] = {}
            posicion_atributo = cant_atributos.index(atributo)
            if posicion_atributo == 0:
                X = LoA_Desarrollo.iloc[:,1:atributo] # variables predictoras. agrego uno porque no es 3 inclusivo por ejemplo
                Y = LoA_Desarrollo.iloc[:,0]
                  
            else:    
                X = LoA_Desarrollo.iloc[:,cant_atributos[posicion_atributo-1]:atributo] # variables predictoras. agrego uno porque no es 3 inclusivo por ejemplo
                Y = LoA_Desarrollo.iloc[:,0]
                    
############# Dividimos en test(30%) y train(70%)
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, shuffle= True, random_state= 314) 
            
############# Generamos el modelo y lo evaluamos
            for k in valores_k:
############# Matrices y dic donde vamos a ir guardando los resultados

                # Declaramos el tipo de modelo
                neigh = KNeighborsClassifier(n_neighbors = k)
                
                # Entrenamos el modelo (con datos de train)
                neigh.fit(X_train, Y_train) 
                
                # Evaluamos el modelo con datos de train y luego de test
                resultados_train = neigh.score(X_train, Y_train)
                resultados_test = neigh.score(X_test, Y_test)
                
                #########################
                resultados[atributo][k] = (resultados_train, resultados_test)
     
                                       
                ##################################################################
                ## Graficamos R2 en función de k (para train y test)
                ##################################################################
                plt.plot(valores_k, resultados_train[i], label=f'Train - {atributo} atributos')
                plt.plot(valores_k, resultados_test[i], label=f'Test - {atributo} atributos')

                plt.legend()
                plt.title('Performance del modelo de knn')
                plt.xlabel('Cantidad de vecinos')
                plt.ylabel('R^2')
                plt.xticks(valores_k)
                plt.ylim(0.90,1.00)
#%% Separar los datos en conjunto de train y test

#%% Ajustar un modelo de knn, 
### probar con distintos conj. de 3 atributos y comparar resultados
### analizar otra cantidad de atributos
### utilizar metricas para problemas de clasificacion como exactitud
