# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])
# define the list of boundaries
boundaries = [
	([17, 15, 90], [50, 56, 255]),
	([80, 25, 4], [255, 100, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]



# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	print(lower)
	print(upper)
	output = cv2.bitwise_and(image, image, mask = mask)
 
	# show the images
	cv2.namedWindow("Display Frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Display Frame", np.hstack([image, output]))
	cv2.waitKey(0)
