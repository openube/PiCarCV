#import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)

#allow camera to warm up
time.sleep(0.1)

#grab image
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

#display image and wait for keypress
cv2.imshow("Image", image)
cv2.waitKey(0)
