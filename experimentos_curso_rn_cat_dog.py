# Pablo Carreira - 16/01/17

import os

# os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32'


import sys
import cv2
import keras
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGraphicsPixmapItem
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QMainWindow
from devices import CameraDevice
from gui import Ui_MainWindow


class CameraOutputScene(QGraphicsScene):
    rectangle = None

    def __init__(self, cameraDevice, nome="Sem nome"):
        super(CameraOutputScene, self).__init__()
        self.frame_graphics_item = QGraphicsPixmapItem()
        self.addItem(self.frame_graphics_item)
        self._cameraDevice = cameraDevice
        self._cameraDevice.new_frame.connect(self._on_new_frame)
        self.addText(nome)

    def _draw_retangulo(self, frame):
        # Pode ser definido pen e brush.
        altura, largura, bandas = frame.shape
        print("Largura {}  Altura {}  Bandas {}".format(largura, altura, bandas))
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor(0, 255, 0))
        tamanho_retangulo = 300

        # posicao do retangulo.

        # Centralizado
        x = int((largura / 2) - (tamanho_retangulo/2))
        y = int((altura / 2) - (tamanho_retangulo / 2))

        # Canto inferior esquerdo.
        #x = 0
        #y = altura - tamanho_retangulo

        # Canto inferior direito.
        # x = largura - tamanho_retangulo
        # y = altura - tamanho_retangulo

        self.addRect(x, y, tamanho_retangulo, tamanho_retangulo, pen)
        CameraOutputScene.rectangle = [x, y, tamanho_retangulo, tamanho_retangulo]

    @QtCore.pyqtSlot(np.ndarray)
    def _on_new_frame(self, array):
        # Desenha o retangulo no primeiro frame recebido.
        if not self.rectangle:
            self._draw_retangulo(array)
        # normal
        height, width, depth = array.shape
        pixmap = QPixmap(QImage(array, width, height, QImage.Format_RGB888))

        self.frame_graphics_item.setPixmap(pixmap)
        self.process_data(array)

    def process_data(self, array: np.ndarray):
        pass




class ClassifiedOutputScene(CameraOutputScene):
    """Uma cena classificada."""
    def __init__(self, cameraDevice, nome="Classificada"):
        super(ClassifiedOutputScene, self).__init__(cameraDevice, nome)
        self.classificador = keras.models.load_model("modelos/model_cat_dog_1")
        # self.classificador = keras.models.load_model("modelos/model_cat_dog_vgg")
        self.previsoes = []

        self.numero = self.addText("n", QFont("Arial", 50))
        self.numero.setDefaultTextColor(QColor(255, 0, 0))
        self.numero.setPos(420, 420)

    @QtCore.pyqtSlot(np.ndarray)
    def _on_new_frame(self, array):
        # Caso nao tenha o retangulo passa.
        if not self.rectangle:
            return
        x, y, w, h = self.rectangle
        mini_w, mini_h = 150, 150

        corte_array = array[y:y + h, x:x + w]

        # gray_array = cv2.cvtColor(corte_array, cv2.COLOR_RGB2GRAY)
        gray_array = corte_array

        mini_array = cv2.resize(gray_array, (mini_w, mini_h), cv2.INTER_LINEAR)

        big_out = cv2.resize(mini_array, (480, 480), cv2.INTER_LINEAR)

        # cortada
        pixmap = QPixmap(QImage(big_out, 480, 480, QImage.Format_RGB888))
        self.frame_graphics_item.setPixmap(pixmap)

        array_preparada = mini_array.reshape(-1, 3, 150, 150)
        previsao = self.classificador.predict_classes([array_preparada, ], verbose=0)

        resultado = ["cachorro", "gato"][previsao[0][0]]
        print(resultado)
        self.numero.setPlainText(resultado)

        # Coloca nas previsoes e caso atinja a contagem, mostra o resultado.
        # self.previsoes.append(previsao[0])
        # if len(self.previsoes) == 10:
        #     print(self.previsoes)
        #     numero = np.bincount(self.previsoes).argmax()
        #     print(numero)
        #     self.numero.setPlainText(str(numero))
        #     self.previsoes = []


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



