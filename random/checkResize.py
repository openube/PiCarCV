from collections import deque
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])

print(image.size)
print(image[1:3][1:3])
cv2.imshow("original",image)
cv2.waitKey(0)
resized = imutils.resize(image,width = 200)
print(resized.size)
print(resized[1:3][1:3])
cv2.imshow("resized", resized)
cv2.waitKey(0)
