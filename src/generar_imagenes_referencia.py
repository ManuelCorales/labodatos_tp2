# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Grupo       : Grupo 2
Detalle     : 
    
Autores     : Corales, Biasoni y Soler

"""

#%%
import matplotlib.pyplot as plt
from src.utils_imagenes import obtenerDfLetra, obtenerDfImagenReferencia, convertirArrayACuadrado


def main():
    generarImagenesReferencia()

#%% Generamos imágenes de referencia
def generarImagenesReferencia():
    print('######## GENERACIÓN DE IMÁGENES DE REFERENCIA ########')
    for i in range(24):
        datosLetra = obtenerDfLetra(i)
        resultado = obtenerDfImagenReferencia(datosLetra, True, i)
        resultado = convertirArrayACuadrado(resultado)

        plt.matshow(resultado, cmap = "gray")
        plt.savefig(fname='./data/ImagenesReferencia/'+ str(i))
        plt.close()



if(__name__ == "__main__"):
    main()