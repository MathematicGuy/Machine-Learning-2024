import cv2 as cv
import os
print('current dir:', os.getcwd())

img = cv.imread('Resources/Photos/cat_large.jpg')
cv.imshow('Cat', img)
cv.waitKey(0)  # wait for specific deplay for a key to be pressed
