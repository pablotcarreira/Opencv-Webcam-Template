# Pablo Carreira - 16/01/17

import os

# os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32'

from output_scenes import CameraOutputScene
import sys
import cv2
import keras
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from devices import CameraDevice
from gui import Ui_MainWindow





class ClassifiedOutputScene(CameraOutputScene):
    """Uma cena classificada."""
    def __init__(self, cameraDevice, nome="Classificada"):
        super(ClassifiedOutputScene, self).__init__(cameraDevice, nome)
        self.classificador = keras.models.load_model("modelos/model_mnist_convulocional")
        self.previsoes = []

        self.numero = self.addText("n", QFont("Arial", 50))
        self.numero.setDefaultTextColor(QColor(255, 255, 0))
        self.numero.setPos(420, 420)

    @QtCore.pyqtSlot(np.ndarray)
    def _on_new_frame(self, array):
        # Caso nao tenha o retangulo passa.
        if not self.rectangle:
            return
        x, y, w, h = self.rectangle
        mini_w, mini_h = 28, 28
        corte_array = array[y:y + h, x:x + w]
        gray_array = cv2.cvtColor(corte_array, cv2.COLOR_RGB2GRAY)
        mini_array = cv2.resize(gray_array, (mini_w, mini_h), cv2.INTER_LINEAR)
        mini_array = np.invert(mini_array)

        threshold = 130
        mini_array = cv2.threshold(mini_array, threshold, 255, cv2.THRESH_BINARY)[1]

        big_out = cv2.resize(mini_array, (480, 480), cv2.INTER_LINEAR)

        # cortada
        height, width, depth = corte_array.shape
        pixmap = QPixmap(QImage(big_out, 480, 480, QImage.Format_Grayscale8))

        self.frame_graphics_item.setPixmap(pixmap)
        previsao = self.classificador.predict_classes([mini_array.reshape(-1, 1, 28, 28), ], verbose=0)
        # proba = self.classificador.predict_proba([mini_array.reshape(-1, 784), ])

        # Coloca nas previsoes e caso atinja a contagem, mostra o resultado.
        self.previsoes.append(previsao[0])
        if len(self.previsoes) == 10:
            print(self.previsoes)
            numero = np.bincount(self.previsoes).argmax()
            print(numero)
            self.numero.setPlainText(str(numero))
            self.previsoes = []


if __name__ == '__main__':
    app = QApplication(sys.argv)
    minha_janela = Ui_MainWindow()
    janela_main = QMainWindow()
    minha_janela.setupUi(janela_main)

    # Escolher um ou outro:
    my_device = CameraDevice()
    # my_device = VideoDevice("data/road.mp4")

    cena1 = CameraOutputScene(my_device, nome="Imagem Bruta")
    cena2 = ClassifiedOutputScene(my_device, nome="Imagem Processada")
    minha_janela.imagem1.setScene(cena1)
    minha_janela.imagem2.setScene(cena2)

    janela_main.show()
    sys.exit(app.exec_())



