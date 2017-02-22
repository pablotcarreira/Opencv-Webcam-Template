# Pablo T. Carreira 2016.


from PyQt5.QtWidgets import QGraphicsPixmapItem
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import numpy as np


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
