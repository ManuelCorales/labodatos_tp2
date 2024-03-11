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
from src.utils_imagenes import obtenerDfGeneral 

#%%
def main():
    clasificacionBinaria()

def clasificacionBinaria():
    print('######## CLASIFICACION BINARIA ########')
    datos()
    analisis_muestras()
    resultados = entrenar_y_evaluar_modelos()
    analisis_experimento(resultados)
    visualizacion(resultados)


#%%  Construir nuevo dataframe a partir del original que imagenes de letras L y A solamente
LoA = None

###  cambiar pixeles rango 
# Rango de valores por los que se va a mover k (hiperparámetro de vecinos de modelo knn)
valores_k = range(1, 9)
        
def datos():
    global LoA
    ### importamos datos
    datos = obtenerDfGeneral()

    ### creamos dataframe con subconjunto
    ### van a ser requeridos los datos con label=0 (A), y label=10(L)
    consulta = sql^"""
                    SELECT * 
                    FROM datos
                    WHERE label= 0 OR label= 10 
                   """
    LoA = DataFrame(consulta)
    
#%% Sobre el subconjunto, analizar cuantas muestras se tienen

def analisis_muestras():
    ### contemos si las muestras estan balanceadas
    consulta2 = sql^"""
                    SELECT count(*) AS 'muestrasxletra'
                    FROM LoA
                    GROUP BY label
                   """
    print(consulta2)
    
    ### obtenemos que de la A hay 1.126, y 1.241 de la L...No tan desbalanceado
    ### pero tampoco cantidades iguales
   
    

#%%
#################################################
## Generacion archivos TEST / TRAIN
#################################################
def entrenar_y_evaluar_modelos():
        resultados = {}
        # vamos a dividir el dataset en distintas cantidades de atributos

        cant_atributos = [2, 4, 100, 102, 265, 267, 303, 305, 465, 467, 550, 552,
                625, 627, 698, 700]

        # Realizamos los modelos con distintas cantidades de atributos
        for i in range(1, len(cant_atributos), 2):
            atributo =cant_atributos[i]
            resultados[atributo] = {} 
            
            X = LoA.iloc[:,cant_atributos[i-1]:atributo] # variables predictoras. agrego uno porque no es 3 inclusivo por ejemplo
            Y = LoA.iloc[:,0]
                    
            ## Dividimos en test(30%) y train(70%)
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, shuffle= True, random_state= 314) 

            ## stratify         

            ## Generamos el modelo y lo evaluamos
            for k in valores_k:
                # Declaramos el tipo de modelo
                neigh = KNeighborsClassifier(n_neighbors = k)
                
                # Entrenamos el modelo (con datos de train)
                neigh.fit(X_train, Y_train) 
                
                # Evaluamos el modelo con datos de train y luego de test
                resultados_train = neigh.score(X_train, Y_train)
                resultados_test = neigh.score(X_test, Y_test)
                
                
                resultados[atributo][k] = (resultados_train, resultados_test)

        return resultados
     
              
def analisis_experimento(resultadosEvaluacion):
    medias_entrenamiento = []
    medias_prueba = []
    
    # Iteramos sobre los elementos del diccionario resultados
    for diccionario in resultadosEvaluacion.values():
        puntuaciones_entrenamiento = []
        puntuaciones_prueba = []
        # Iteramos sobre las tuplas dentro de cada subdiccionario
        for tupla in diccionario.values():
            # Agregamos la puntuación de entrenamiento a la lista
            puntuaciones_entrenamiento.append(tupla[0])
            puntuaciones_prueba.append(tupla[1])
        
        # Calculamos la media de las puntuaciones de entrenamiento
        media_entrenamiento = sum(puntuaciones_entrenamiento) / len(puntuaciones_entrenamiento)
        medias_entrenamiento.append(media_entrenamiento)
        
        media_prueba = sum(puntuaciones_prueba) / len(puntuaciones_prueba)
        medias_prueba.append(media_prueba)
        
    # Iteramos sobre los elementos del diccionario resultados
    maximos_entrenamiento = []
    maximos_prueba = []
    for diccionario in resultadosEvaluacion.values():
        max_entrenamiento = []
        max_prueba = []
        # Iteramos sobre las tuplas dentro de cada subdiccionario
        for tupla in diccionario.values():
             # Agregamos la puntuación de entrenamiento a la lista
             max_entrenamiento.append(tupla[0])
             max_prueba.append(tupla[1])
        
        maximos_entrenamiento.append(max(max_entrenamiento))
        maximos_prueba.append(max(max_prueba))
    
    
    print('--------------')  
    print("medias_prueba")
    print(medias_prueba)
### vemos que la mayor media en prueba la tiene el conjunto 4    
    
    print('--------------')  
    print("maximos_prueba")
    print(maximos_prueba)
### vemos que el maximo se encuentra en el conjunto 4    
    
## ademas, con k = 6 se ve en el grafico que alcanza su max score


def visualizacion(resultadosEvaluacion):
#%% graficos de score vs k
    contador = 1
    # Iteramos sobre los elementos del diccionario resultados
    for diccionario in resultadosEvaluacion.values():
            
        puntuaciones_entrenamiento = []
        puntuaciones_prueba = []
            
        # Iteramos sobre las tuplas dentro de cada subdiccionario
        for tupla in diccionario.values():
            # Agregamos la puntuación de entrenamiento a la lista
            puntuaciones_entrenamiento.append(tupla[0])
            puntuaciones_prueba.append(tupla[1])
            
        # Crear el gráfico
        plt.plot(valores_k, puntuaciones_entrenamiento, marker='o', linestyle='-', color = 'firebrick', label = 'entrenamiento')
        plt.plot(valores_k, puntuaciones_prueba, marker='o', linestyle='-', color = 'orangered', label = 'test')
        
        # Etiquetas y título del gráfico
        plt.xlabel('Valores de vecinos (k)')
        plt.ylabel('Score')
        plt.title(f'Valores de k vs Scores - {contador}')
        plt.legend()
        plt.ylim(0.50, 1.00)
        # Mostrar el gráfico
        plt.show()
   
        contador += 1

#%% grafico donde marcamos los pixeles usados
    pixeles = [2, 3, 100, 101, 265, 266, 303, 304, 465, 466, 550, 551,
                      625, 626, 698, 699]
    
    # Cargar la imagen de fondo desde el archivo CSV
    imagen_fondo = pd.read_csv("./data/rango_pixeles.csv").values
    
    # Crear una máscara con los píxeles a resaltar
    mascara = np.isin(np.arange(imagen_fondo.size), pixeles).reshape(imagen_fondo.shape)
    
    # Mostrar la imagen de fondo con los píxeles resaltados
    plt.imshow(imagen_fondo, cmap='gray')  # Visualizar la imagen de fondo en escala de grises
    plt.imshow(mascara, cmap='PuRd',alpha=0.4)  # Resaltar los píxeles en color PuRd con una transparencia del 40%
    plt.title('Imagen de fondo con píxeles resaltados')
    plt.show()
    



if(__name__ == "__main__"):
    main()