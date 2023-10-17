""" Clases para Redes Nueronales Adversarias """

import numpy as np
from numpy import ndarray

from lib.utils import sigmoid


class Red:
    """ Red Neuronal """
    def __init__(self, alpha):
        self._pesos = None
        self._sesgos = None
        self._alpha = alpha

    @property
    def alpha(self):
        """ Getter """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """ Setter """
        self._alpha = value

    @property
    def pesos(self):
        """ Getter """
        return self._pesos

    @pesos.setter
    def pesos(self, value):
        """ Setter """
        print(f'[{self.__class__.__name__.ljust(13)}] Pesos de {self._pesos} a {value}')
        self._pesos = value

    @property
    def sesgos(self):
        """ Getter """
        return self._sesgos

    @sesgos.setter
    def sesgos(self, value):
        """ Setter """
        print(f'[{self.__class__.__name__.ljust(13)}] Sesgo de {self.sesgos} a {value}')
        self._sesgos = value

    def forward(self, instancia: ndarray):
        """Cálculo hacia adelante"""
        return sigmoid(np.dot(instancia, self.pesos) + self.sesgos)


class Discriminator(Red):
    """ Discriminador """

    def __init__(self, alpha):
        # Inicializar al padre
        Red.__init__(self, alpha)

        # Inicializados aleatoriamente
        self.pesos = np.array([np.random.normal() for _ in range(4)])
        self.sesgos = np.random.normal()

    #
    # Funciones con base en las imágenes
    #
    def error_de_etiqueta(self, image):
        """

        :param image:
        :return:
        """
        prediction = self.forward(image)
        # We want the prediction to be 1, so the error is -log(prediction)
        return -np.log(prediction)

    def derivada_para_etiqueta(self, instancia: ndarray):
        """ Buscamos acertar a la clasificación """
        prediccion = self.forward(instancia)
        derivadas_pesos = -instancia * (1 - prediccion)
        derivadas_sesgo = -(1 - prediccion)
        return derivadas_pesos, derivadas_sesgo

    def actualizar_desde_etiqueta(self, instancia: ndarray):
        """ Backpropagation de la salida a las entradas """

        # Calcular la derivada
        derivada_pesos, derivada_sesgo = self.derivada_para_etiqueta(instancia)
        # Algoritmo Backpropagation para 2 capas
        self.pesos -= self.alpha * derivada_pesos
        self.sesgos -= self.alpha * derivada_sesgo

    #
    # Funciones con base en el ruido
    #
    def error_de_ruido(self, noise):
        """

        :param noise:
        :return:
        """
        prediction = self.forward(noise)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        return -np.log(1 - prediction)

    def derivadas_para_ruido(self, ruido):
        """

        :param ruido:
        :return:
        """
        prediccion = self.forward(ruido)
        derivadas_pesos = ruido * prediccion
        derivadas_sesgos = prediccion
        return derivadas_pesos, derivadas_sesgos

    def actualizar_desde_ruido(self, ruido):
        """

        :param ruido:
        :return:
        """
        derivadas_pesos, derivadas_sesgos = self.derivadas_para_ruido(ruido)
        self.pesos -= self._alpha * derivadas_pesos
        self.sesgos -= self._alpha * derivadas_sesgos


class Generator(Red):
    """ Generador """

    def __init__(self, alpha):
        """

        :param alpha:
        """
        # Inicializar al padre
        Red.__init__(self, alpha)

        self.pesos = np.array([np.random.normal() for _ in range(4)])
        self.sesgos = np.array([np.random.normal() for _ in range(4)])

    def error(self, instancia, discriminador):
        """

        :param instancia:
        :param discriminador:
        :return:
        """
        x = self.forward(instancia)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        y = discriminador.forward(x)
        return -np.log(y)

    def derivadas(self, z, discriminador):
        """

        :param z:
        :param discriminador:
        :return:
        """
        discriminador_pesos = discriminador.pesos
        x = self.forward(z)
        y = discriminador.forward(x)
        #
        derivadas_sesgo = -(1 - y) * discriminador_pesos * x * (1 - x)
        derivadas_pesos = derivadas_sesgo * z
        return derivadas_pesos, derivadas_sesgo

    def actualizar(self, z, discriminator):
        """

        :param z:
        :param discriminator:
        :return:
        """
        error_antes = self.error(z, discriminator)
        derivadas_pesos, derivadas_sesgo = self.derivadas(z, discriminator)
        self.pesos -= self.alpha * derivadas_pesos
        self.sesgos -= self.alpha * derivadas_sesgo
        error_despues = self.error(z, discriminator)
        print(f'[{self.__class__.__name__.ljust(13)}] Error de {error_antes} a {error_despues}')
