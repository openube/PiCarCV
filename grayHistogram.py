

#import packages
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

#load image
img = cv2.imread(args["image"])
color = ('b','g','r')


#calculate hist


#plot hist
histData = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histData)
plt.xlim([0,256])
plt.ylim([0,50000])

plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.title("Grayscale Histogram")
plt.show()

#plt.hist(img.ravel(),bins,[0,256])
