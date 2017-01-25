#! /usr/bin/python
 

import cv2

def detectRegionsOfInterest(frame, cascade):

    # haar detection.
    cars = cascade.detectMultiScale(frame, 1.1, 1)
    if len(cars) == 0:
        return False
    return cars

