from PyQt5 import QtCore
import cv2
import numpy as np


class CameraDevice(QtCore.QObject):
    """Um device que captura da Webcam a 25 fps.
    Notar que o OpenCV 3.1 para Python 3.x precisa ser compilado da fonte para que a camera funcione.
    """
    new_frame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_id: int=0, mirrored: bool=False, parent: QtCore.QObject=None, fps: int=25):
        super(CameraDevice, self).__init__(parent)
        self.mirrored = mirrored
        self._camera_device = cv2.VideoCapture(camera_id)
        self._camera_device.open(camera_id)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000 / fps)
        self.paused = False

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        """Captura um frame da camera ou do video."""
        ok, frame = self._camera_device.read()
        image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.new_frame.emit(image_array)

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

