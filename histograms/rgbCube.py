import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d,Axes3D
import argparse
import imutils

#Parse arguments for image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

#load image
img = cv2.imread(args["image"])

#Hardcoded dimensions because constant on pi
#To get new dimensions use the code below:

#width, height = img.shape[:2];
cols = 2464;
rows = 3280;

#imgReshape = np.reshape(img,(1,2425760))
#print(len(imgReshape))

pixelNums = 20
b = img[:pixelNums,:pixelNums,0]
g = img[:pixelNums,:pixelNums,1]
r = img[:pixelNums,:pixelNums,2]

#print(b.shape)
#b = np.reshape(b,(b.size,1))
#g = np.reshape(g,(g.size,1))
#r = np.reshape(r,(r.size,1))

#print(b.size)
#print(b.shape)
print(img.shape)
imgRes = imutils.resize(img,width=600)
print(imgRes.shape)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(r,g,b)
#ax.set_xlabel('Red')
#ax.set_ylabel('Green')
#ax.set_zlabel('Blue')
#ax.set_xlim([0,255])
#ax.set_ylim([0,255])
#ax.set_zlim([0,255])
#ax.set_title('Landscape RGB Cube')
#plt.savefig('landscapeRGBcube.png')
#plt.show()

#print(img.size)
#print(b.size)
#print(b.max())
#rgbList
#for r in range(0,rows-1):
#    for c in range(0,cols-1):
        

