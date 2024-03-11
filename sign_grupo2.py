from src.generar_imagenes_referencia import generarImagenesReferencia
from src.evaluacion_multiclase import evaluacionMulticlase
from src.analisis_exploratorio import generarGraficosDeAnalisis
from src.clasificacion_binaria import clasificacionBinaria

def main():
    #%%
    # Generamos imágenes de referencia
    generarImagenesReferencia()

    #%%
    # Mostramos análisis exploratorios
    generarGraficosDeAnalisis()

    #%%
    # Hacemos la evaluacion binaria
    clasificacionBinaria()

    #%%
    # Hacemos la evaluación multiclase 
    evaluacionMulticlase()




if(__name__ == "__main__"):
    main()