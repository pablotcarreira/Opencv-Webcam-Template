# Pablo Carreira - 16/01/17

import os
#os.environ['KERAS_BACKEND'] = 'theano'
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

from detect import detectRegionsOfInterest
from gui import Ui_MainWindow


# https://rafaelbarreto.wordpress.com/2011/08/27/a-pyqt-widget-for-opencv-camera-preview/
# http://pyqt.sourceforge.net/Docs/PyQt5/signals_slots.html
# https://github.com/andrewssobral/vehicle_detection_haarcascades
# https://pythonspot.com/car-tracking-with-cascades/
# https://deeplearning4j.org/compare-dl4j-torch7-pylearn.html

# Caffe
# https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.g129385c8da_651_21

# CARROS!!
# http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

# http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

# Fonte de dados:
# ImageNet



class CameraDevice(QtCore.QObject):
    # original newFrame = QtCore.pyqtSignal(cv2.iplimage)
    newFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraId=0, mirrored=False, parent=None, fps=25):
        super(CameraDevice, self).__init__(parent)
        self.mirrored = mirrored
        self._camera_device = cv2.VideoCapture(cameraId)
        self._camera_device.open(cameraId)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000 / fps)
        self.paused = False

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        """Captura um frame da camera ou do video."""
        ok, frame = self._camera_device.read()
        image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.newFrame.emit(image_array)

    @property
    def paused(self):
        return not self._timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self._timer.stop()
        else:
            self._timer.start()


class VideoDevice(CameraDevice):
    def __init__(self, video_src, mirrored=False, parent=None, fps=25):
        super(CameraDevice, self).__init__(parent)
        self.mirrored = mirrored
        self._cameraDevice = cv2.VideoCapture(video_src)
        if self._cameraDevice.isOpened() is False:
            raise RuntimeError("Error opening the video file.")
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000 / fps)
        self.paused = False


class CameraOutputScene(QGraphicsScene):
    rectangle = None

    def __init__(self, cameraDevice, nome="Sem nome"):
        super(CameraOutputScene, self).__init__()
        self.frame_graphics_item = QGraphicsPixmapItem()
        self.addItem(self.frame_graphics_item)
        self._cameraDevice = cameraDevice
        self._cameraDevice.newFrame.connect(self._on_new_frame)
        self.addText(nome)

    def _draw_retangulo(self, frame):
        # Pode ser definido pen e brush.
        altura, largura, bandas = frame.shape
        print("Largura {}  Altura {}  Bandas {}".format(largura, altura, bandas))
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor(0, 255, 0))
        tamanho_retangulo = 100

        # posicao do retangulo.

        # Centralizado
        #x = int((largura / 2) - (tamanho_retangulo/2))
        #y = int((altura / 2) - (tamanho_retangulo / 2))

        # Canto inferior esquerdo.
        #x = 0
        #y = altura - tamanho_retangulo

        # Canto inferior direito.
        x = largura - tamanho_retangulo
        y = altura - tamanho_retangulo

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
        self.classificador = keras.models.load_model("model_mnist_convulacional")
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

        #corte_array = array[y:y + h, x:x + w ]

        gray_array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        #mini_array = cv2.resize(gray_array, (mini_w, mini_h), cv2.INTER_LINEAR)

        negativo_array = np.invert(gray_array)
        binary_array = cv2.threshold(negativo_array, 150, 255, cv2.THRESH_BINARY)[1]
        #big_out = cv2.resize(mini_array, (480, 480), cv2.INTER_LINEAR)

        #cortada
        #height, width, depth = corte_array.shape

        pixmap = QPixmap(QImage(binary_array, array.shape[1], array.shape[0], QImage.Format_Grayscale8))
        self.frame_graphics_item.setPixmap(pixmap)

        previsao = self.classificador.predict_classes([array, ], verbose=0)
        #proba = self.classificador.predict_proba([mini_array.reshape(-1, 784), ])

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


