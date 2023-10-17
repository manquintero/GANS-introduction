""" Plot """
from matplotlib import pyplot as plt


COLOR_MAP = 'BuGn_r'  # Green-Blue


def ver_muestras(muestras, filas, columnas, titulo):
    """

    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    :param muestras:
    :param filas:
    :param columnas:
    :return:
    """
    fig, axes = plt.subplots(figsize=(10, 10), nrows=filas, ncols=columnas,
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), muestras):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        _ = ax.imshow(1 - img.reshape((2, 2)), cmap=COLOR_MAP)

    if titulo:
        fig.suptitle(titulo)

    return fig, axes


def ver_errores(errores: dict):
    """

    :param errores:
    :return:
    """

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Errores')
    # Generador
    ax1.set_title('Generador')
    ax1.plot(errores['generador'])
    # Discriminador
    ax2.set_title('Discriminador')
    ax2.plot(errores['discriminador'])

    return fig, ax1, ax2
