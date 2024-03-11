
from pandas import DataFrame
import pandas as pd
import numpy as np

#%%
carpeta = "./data/"
df = pd.read_csv(carpeta + "info_limpia.csv")

mappingLetras = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}



def obtenerDfGeneral():
    return df

# Dado un df obtenemos la matriz de imagen de referencia definida en el informe del trabajo
def obtenerDfImagenReferencia(dfImagen: DataFrame, guardar=False, indiceLetra=-1):
    resultado = []
    for j in range(1, 785):
        pixelMediana = dfImagen['pixel'+str(j)].median()
        resultado.append(pixelMediana)

    resultado = DataFrame(resultado)

    if(guardar and indiceLetra != -1):
        resultado.to_csv('./data/DfImagenesReferencia/'+str(indiceLetra)+'.csv', index=False)

    return resultado


# Dado el df general devuelve todos los registros de un indice de letra dada
def obtenerDfLetra(indiceLetra: int):
    return df[df['label'] == indiceLetra]


def convertirArrayACuadrado(array):
    return np.reshape(array, (28, 28))


def obtenerLetraSegunIndice(indiceLetra):
    return mappingLetras[indiceLetra]

def obtenerListaLetras():
    return mappingLetras.values()
