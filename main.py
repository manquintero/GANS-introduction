""" Red Neuronal """

import numpy as np
from numpy import random

from lib.gan import Discriminator, Generator
from lib.graficar import ver_muestras, ver_errores

# Hyperparameters
ALPHA = 0.01
TMAX = 1000

if __name__ == "__main__":

    # Ejemplos de Horizontes
    horizontes = [np.array([1.0, 0.0, 0.0, 1.0]),
                  np.array([0.9, 0.1, 0.2, 0.8]),
                  np.array([0.9, 0.2, 0.1, 0.8]),
                  np.array([0.8, 0.1, 0.2, 0.9]),
                  np.array([0.8, 0.2, 0.1, 0.9])]
    _ = ver_muestras(horizontes, filas=1, columnas=4, titulo="Muestras")

    # Ejemplos de ruido
    ruido = [np.random.randn(2, 2) for i in range(20)]
    _ = ver_muestras(ruido, filas=4, columnas=5, titulo="Ruido")

    # Semilla de reproducibilidad
    np.random.seed(42)

    # La Red
    discriminator = Discriminator(ALPHA)
    generador = Generator(ALPHA)

    # Almacenar los errores
    errores= {
        'discriminador': [],
        'generador': []
    }

    for epoch in range(TMAX):
        for horizonte in horizontes:
            # Actualizar los pesos del discriminador con salidas etiquetadas
            discriminator.actualizar_desde_etiqueta(horizonte)

            # Número Aleatorio para generar ruido (falso horizonte)
            z = random.rand()

            # Calcular el error del discriminador
            error = sum(discriminator.error_de_etiqueta(horizonte) + discriminator.error_de_ruido(z))
            errores['discriminador'].append(error)

            # Calcular el error del generador
            error = generador.error(z, discriminator)
            errores['generador'].append(error)

            # Construir un horizonte falso
            ruido = generador.forward(z)

            # Actualizar los pesos del discriminador a partir del horizonte falso
            discriminator.actualizar_desde_ruido(ruido)

            # Actualizar los pesos del generador desde el horizonte falso
            generador.actualizar(z, discriminator)

    # Diagramar errores
    _ = ver_errores(errores)

    # Generating Images
    horizontes_generados = []
    for i in range(4):
        z = random.random()
        horizonte_generado = generador.forward(z)
        horizontes_generados.append(horizonte_generado)
    _ = ver_muestras(horizontes_generados, filas=1, columnas=4, titulo="Horizontes Generado")

    for horizonte in horizontes_generados:
        print(horizonte)

    # Resultados
    print("Generator pesos", generador.pesos)
    print("Generator sesgos", generador.sesgos)
    print("Discriminator pesos", discriminator.pesos)
    print("Discriminator sesgos", discriminator.sesgos)

    print('END')
