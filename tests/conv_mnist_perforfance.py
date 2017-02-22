
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32'

import cv2
import keras
import numpy as np


def test_loop(corte_array: np.ndarray, classificador: keras.models.Model) -> None:
    """ Loop para teste de performance, esta funcao sera executada muitas vezes.

    :param classificador: Um model do Keras.
    :param corte_array: Matriz (imagem) no tamanho correto.
    """
    repeticoes = 1000
    for i in range(repeticoes):
        previsao = classificador.predict_classes([corte_array.reshape(-1, 1, 28, 28), ], verbose=0)
        previsao = previsao + i


if __name__ == '__main__':
    fake_array = np.zeros((28, 28), dtype=np.uint8) + 10
    model = keras.models.load_model("../modelos/model_mnist_convulocional")
    test_loop(fake_array, model)
