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

#%% cant de clases (cada clase es una letra)

## no c si se hace con train o con datos, creeria que con train

consulta = sql^"""
           SELECT DISTINCT COUNT(*) AS 'cant_clases'
           FROM train
          """
print(consulta)

