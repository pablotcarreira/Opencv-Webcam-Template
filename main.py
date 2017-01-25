# Pablo Carreira - 16/01/17
import sys

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QColor
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
    newFrame = QtCore.pyqtSignal(QPixmap, np.ndarray)
    frame_width = 640
    frame_height = 480

    def __init__(self, cameraId=0, mirrored=False, parent=None, fps=25):
        super(CameraDevice, self).__init__(parent)
        self.mirrored = mirrored
        self._cameraDevice = cv2.VideoCapture(cameraId)
        self._cameraDevice.open(cameraId)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000 / fps)
        self.paused = False

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        ok, frame = self._cameraDevice.read()
        print(frame)
        image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, depth = image_array.shape
        pixmap = QPixmap(QImage(image_array, width, height, QImage.Format_RGB888))
        self.newFrame.emit(pixmap, image_array)

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
    def __init__(self, cameraDevice, nome="Sem nome"):
        super(CameraOutputScene, self).__init__()
        self.frame_graphics_item = QGraphicsPixmapItem()
        self.addItem(self.frame_graphics_item)
        self._cameraDevice = cameraDevice
        self._cameraDevice.newFrame.connect(self._onNewFrame)
        self.addText(nome)

    @QtCore.pyqtSlot(QPixmap, np.ndarray)
    def _onNewFrame(self, frame, array):
        self.frame_graphics_item.setPixmap(frame)
        self.process_data(array)

    def process_data(self, array: np.ndarray):
        pass


class ClassifiedOutputScene(CameraOutputScene):
    def __init__(self, cameraDevice, nome="Classificada"):
        super(ClassifiedOutputScene, self).__init__(cameraDevice, nome)
        self.retangulos = []
        self.classificador = cv2.CascadeClassifier('data/cars.xml')

        if not self.classificador:
            raise(IOError("Classificador não carregado."))


    def _draw_rectangulo(self, matches):
        # Pode ser definido pen e brush.
        # This convenience; function is equivalent to calling
        # addRect(QRectF(x, y, w, h), pen, brush)

        for old in self.retangulos:
            self.removeItem(old)

        for item in matches:
            pen = QPen()
            pen.setWidth(4)
            pen.setColor(QColor(0, 255, 0))
            self.retangulos.append(self.addRect(item[0], item[1], item[2], item[3], pen))

    def process_data(self, frame_array: np.ndarray):
        newRegions = detectRegionsOfInterest(frame_array.copy(), self.classificador)
        if newRegions is False:
            return
        else:
            self._draw_rectangulo(newRegions)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    minha_janela = Ui_MainWindow()
    janela_main = QMainWindow()
    minha_janela.setupUi(janela_main)

    # cameraDevice = CameraDevice(mirrored=True)
    cameraDevice = VideoDevice("data/road.mp4")

    cena1 = CameraOutputScene(cameraDevice, nome="Imagem Bruta")
    cena2 = ClassifiedOutputScene(cameraDevice, nome="Imagem Processada")
    minha_janela.imagem1.setScene(cena1)
    minha_janela.imagem2.setScene(cena2)

    janela_main.show()
    sys.exit(app.exec_())



