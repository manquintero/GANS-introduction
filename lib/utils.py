""" Utils """
import numpy as np


def sigmoid(x: float):
    """ Función Sigmoidea

    :param x:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-x))
