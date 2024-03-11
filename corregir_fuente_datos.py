# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Grupo       : Grupo 2
Detalle     : 
    
Autores     : Corales, Biasoni y Soler

"""

#%% importamos librerias
import pandas as pd


#%% importamos datos

def main():
    corregirFuenteDatos()

# Corregimos los datos restando en 1 el índice de los labels mayores a 9
def corregirFuenteDatos():
    carpeta = './data/'
    df = pd.read_csv(carpeta + "sign_mnist_train.csv")
    df[df['label'] > 9] = df[df['label'] > 9] - 1
    df.to_csv(carpeta+'info_limpia.csv', index=False)


if(__name__ == "__main__"):
    main()