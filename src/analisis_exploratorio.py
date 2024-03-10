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
from utils_imagenes import obtenerDfGeneral, obtenerDfLetra, obtenerLetraSegunIndice, obtenerListaLetras


#%% importamos datos

def main():
    generarGraficoDeDiferencias()
    # obtenerValorPixelDeMayorDiferencia()



# def obtenerValorPixelDeMayorDiferencia():
#     carpeta = '../data/DfImagenesReferencia/'
#     mediasPorLetra = []
#     # for i in range(25):
#     diferenciasMaximasPorLetra = []
#     maximo = 0
#     maximosPorClase = []
#     minimosPorClase = []
#     dfReferencia = np.loadtxt(carpeta+str(1)+'.csv')
#     pixelesReferencia = dfReferencia.transpose()[1:]
#     dfLetra = obtenerDfLetra(3)

#     print(len(pixelesReferencia))
#     for j in range(1, 785):
#         max = dfLetra['pixel'+str(j)].max()
#         min = dfLetra['pixel'+str(j)].min()
#         difConMinimo = abs(min - pixelesReferencia[j - 1])
#         difConMaximo = abs(max - pixelesReferencia[j - 1])
#         maxDif = np.max([difConMaximo, difConMinimo])
#         print(maxDif)
#         diferenciasMaximasPorLetra.append(maxDif)

#     mediaDeDiferencias = np.median(diferenciasMaximasPorLetra)
#     mediasPorLetra.append(mediaDeDiferencias)

#     print(48, mediasPorLetra)

def generarGraficoDeDiferencias():
    indiceLetraComparacion = 4 # Letra E
    
    medianas = []
    for i in range(24):
        medianas.append(restaDeImagenesReferencia(indiceLetraComparacion, i))
    
    fig, axes = plt.subplots()
    x = list(obtenerListaLetras())
    axes.bar(x, medianas)
    plt.show()



def restaDeImagenesReferencia(indiceImagen1, indiceImagen2):
    carpeta = '../data/DfImagenesReferencia/'
    dfReferencia1 = np.loadtxt(carpeta+str(indiceImagen1)+'.csv')
    dfReferencia2 = np.loadtxt(carpeta+str(indiceImagen2)+'.csv')

    pixelesReferencia1 = dfReferencia1.transpose()[1:]
    pixelesReferencia2 = dfReferencia2.transpose()[1:]

    resultado = np.subtract(pixelesReferencia1, pixelesReferencia2)
    resultado = np.absolute(resultado)

    return np.median(resultado)


if(__name__ == "__main__"):
    main()