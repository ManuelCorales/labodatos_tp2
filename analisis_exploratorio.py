# -*- coding: utf-8 -*-
"""
Materia     : Laboratorio de datos - FCEyN - UBA
Grupo       : Grupo 2
Detalle     : 
    
Autores     : Corales, Biasoni y Soler

"""

#%% importamos librerias
import pandas as pd
from inline_sql import sql, sql_val
from pandas import DataFrame

#%% importamos datos

carpeta = "./labodatos_tp2/"

datos = pd.read_csv(carpeta + "sign_mnist_train.csv")


#%% 