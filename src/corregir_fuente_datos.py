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


#%% importamos datos

def main():
    corregirFuenteDatos()


def corregirFuenteDatos():
    carpeta = '../'
    df = pd.read_csv(carpeta + "sign_mnist_train.csv")
    df[df['label'] > 9] = df[df['label'] > 9] - 1
    df.to_csv(carpeta+'info_limpia.csv', index=False)


if(__name__ == "__main__"):
    main()