# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Grupo       : Grupo 2
Detalle     : 
    
Autores     : Corales, Biasoni y Soler

"""

#%% importamos librerias
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from src.utils_imagenes import obtenerListaLetras, convertirArrayACuadrado


#%% importamos datos

def main():
    generarGraficosDeAnalisis()

#%%
def generarGraficosDeAnalisis():
    print('######## GRÁFICOS DE ANÁLISIS DE DATOS ########')
    generarGraficoDeDiferencias()
    calcularRangoDePixelesImagenesReferencia()

#%%
def generarGraficoDeDiferencias():
    indiceLetraComparacion = 4 # Letra E
    
    medianas = []
    for i in range(24):
        medianas.append(restaDeImagenesReferencia(indiceLetraComparacion, i))
    
    fig, axes = plt.subplots()
    x = list(obtenerListaLetras())
    axes.bar(x, medianas)
    
    plt.ylabel("Métrica", fontsize=13)
    plt.xlabel("Letras", fontsize=13)
    plt.show()


#%%
def restaDeImagenesReferencia(indiceImagen1, indiceImagen2):
    carpeta = './data/DfImagenesReferencia/'
    referencia1 = np.loadtxt(carpeta+str(indiceImagen1)+'.csv')
    referencia2 = np.loadtxt(carpeta+str(indiceImagen2)+'.csv')

    pixelesReferencia1 = referencia1.transpose()[1:]
    pixelesReferencia2 = referencia2.transpose()[1:]

    resultado = np.subtract(pixelesReferencia1, pixelesReferencia2)
    resultado = np.absolute(resultado)

    return np.median(resultado)

#%%
def calcularRangoDePixelesImagenesReferencia():
    pixelesMax = [0] * 784
    pixelesMin = [255] * 784
    carpeta = './data/DfImagenesReferencia/'
    for i in range(24):
        referencia = np.loadtxt(carpeta+str(i)+'.csv')
        referencia = referencia.transpose()[1:]
        pixelesMax = np.fmax(referencia, pixelesMax)
        pixelesMin = np.fmin(referencia, pixelesMin)
    
    resultado = convertirArrayACuadrado(pixelesMax - pixelesMin)
    plt.matshow(resultado, cmap = "gray")
    plt.show()
    
    # Guardar el resultado en un archivo CSV
    carpeta = "./data/"
    df_resultado = pd.DataFrame(resultado)
    df_resultado.to_csv(carpeta + 'rango_pixeles.csv', index=False)

#%%
if(__name__ == "__main__"):
    main()