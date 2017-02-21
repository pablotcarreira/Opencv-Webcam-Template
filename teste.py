

import cv2
camera_id = 0
camera_device = cv2.VideoCapture(camera_id)
camera_device.open(camera_id)
ok, frame = camera_device.read()

print(ok, frame)